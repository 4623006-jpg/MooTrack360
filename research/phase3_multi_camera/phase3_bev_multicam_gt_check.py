"""
phase3_bev_multicam_gt_check.py

【vml36で実行】

単一カメラでのIDF1評価(Phase1/2)に続き、マルチカメラでの評価。方針は
「検出を先にBEV(鳥瞰図)融合してから追跡」:

  1. 各カメラのGT bboxを、GT姿勢(lying/standing)に応じた高さ仮定で
     Double Sphereモデルにより床面(BEV)座標に逆投影する
  2. カメラ間の距離閾値(dist_thr_fusion)でHungarian割当+Union-Findにより、
     同一個体とみなせる検出同士をクラスタ化して1点に融合する(nameは融合判定には
     一切使わず、評価のためだけに後から突き合わせる — Mootrack_core.pyの
     compute_camera_assignments_by_distance / union_find_clusters をそのまま流用)
  3. 融合後の点列を、フレーム間の距離閾値(dist_thr_track)による単純な
     最近傍追跡(BEV版の簡易トラッカー、自前実装)でトラックIDを付与する
  4. motmetrics(点ベース、norm2squared_matrix)でIDF1を評価する

これはPhase1(GT bboxをそのままConfTrackTrackerに入力し、検出ノイズを除いた
トラッカー単体の上限性能を見た)と同じ考え方で、YOLOの検出ノイズを除いた
「融合+追跡ロジック単体」の上限性能を見るための upper-bound テスト。
これに続けて実際のYOLO検出を使うPhase3B相当のテストを行う予定。

--- 使い方 ---
    python phase3_bev_multicam_gt_check.py
"""

import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

MOOTRACK_CODE_DIR = "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/research/calibration"
if MOOTRACK_CODE_DIR not in sys.path:
    sys.path.insert(0, MOOTRACK_CODE_DIR)

from Mootrack_core import (
    NUM_CAMERAS, CAM_IDS, ROT_FMT,
    Pose, get_intrinsics,
    load_floorplan_cameras, apply_camera_overrides, resolve_poses,
    unproject_to_floor,
    compute_camera_assignments_by_distance,
    union_find_clusters,
)

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
XML_DIR = "/mnt/vmlqnap02/dataset/MooTrack360/1-hour-farm1-area1-sample-sequence/xml-1fps"
FLOORPLAN_JSON_PATH = "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/calibration/Farm2_Area_1_Artboard_config_.json"
OUTPUT_DIR = Path(
    "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/research/idf1_check"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FRAME_START = 0
FRAME_END = 290
FRAME_STEP = 10

IMAGE_WIDTH_PX = 8832

# 高さ仮定(再キャリブレーション後のセンター方式ベスト値。SAM/centerグリッドサーチ由来)
LYING_HEIGHT_M = 0.6
STANDING_HEIGHT_M = 1.0
DEFAULT_HEIGHT_M = 0.8  # GTにpose情報が無い場合のフォールバック

# カメラ間融合の距離閾値(同一個体とみなす距離、メートル)
DIST_THR_FUSION_M = 1.4

# フレーム間追跡の距離閾値(1秒間でこれ以上動いたら別個体とみなす、メートル)
DIST_THR_TRACK_M = 1.0
MAX_MISSED_FRAMES = 2  # この回数だけ未マッチでもトラックを残す


def frame_name(idx: int) -> str:
    return f"{idx:06d}"


def parse_voc_annotation(xml_path) -> List[dict]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        pose_el = obj.find("pose")
        pose = pose_el.text.strip().lower() if pose_el is not None and pose_el.text else None
        bnd = obj.find("bndbox")
        objects.append({
            "name": name,
            "pose": pose,
            "xmin": float(bnd.find("xmin").text),
            "ymin": float(bnd.find("ymin").text),
            "xmax": float(bnd.find("xmax").text),
            "ymax": float(bnd.find("ymax").text),
        })
    return objects


def assign_camera(center_x_full: float, image_width: float) -> str:
    tile_w = image_width / NUM_CAMERAS
    cam_idx = int(center_x_full // tile_w)
    cam_idx = max(0, min(cam_idx, NUM_CAMERAS - 1))
    return CAM_IDS[cam_idx]


def height_for(pose: Optional[str]) -> float:
    if pose == "lying":
        return LYING_HEIGHT_M
    if pose == "standing":
        return STANDING_HEIGHT_M
    return DEFAULT_HEIGHT_M


def name_to_gt_id(name: str) -> int:
    m = re.search(r"\d+", name)
    if not m:
        raise ValueError(f"nameから数値IDを抽出できません: {name}")
    return int(m.group())


def build_world_points_per_camera(
    objects: List[dict], poses: Dict[str, "Pose"], image_width: float,
) -> Dict[str, List[Tuple[str, np.ndarray]]]:
    """(name, world_xy) のリストをカメラごとに構築する。nameは融合判定には使わず、
    後段の評価のためだけに保持する。"""
    tile_w = image_width / NUM_CAMERAS
    per_camera: Dict[str, List[Tuple[str, np.ndarray]]] = {cid: [] for cid in CAM_IDS}

    for obj in objects:
        center_x_full = (obj["xmin"] + obj["xmax"]) / 2.0
        cam_idx = int(center_x_full // tile_w)
        cam_idx = max(0, min(cam_idx, NUM_CAMERAS - 1))
        cam_id = CAM_IDS[cam_idx]

        fx, fy, cx, cy, xi, alpha = get_intrinsics(cam_id)
        local_x = center_x_full - cam_idx * tile_w
        local_y = (obj["ymin"] + obj["ymax"]) / 2.0
        pts_px = np.array([[local_x, local_y]], dtype=np.float64)

        height_m = height_for(obj["pose"])
        cam_pose = poses[cam_id]
        world_xy, valid = unproject_to_floor(
            pts_px, cam_pose.R, cam_pose.t, fx, fy, cx, cy, xi, alpha,
            floor_z=-height_m,
        )
        if not valid[0]:
            continue
        per_camera[cam_id].append((obj["name"], world_xy[0]))

    return per_camera


def fuse_frame(per_camera: Dict[str, List[Tuple[str, np.ndarray]]], dist_thr: float):
    """カメラ間距離閾値でクラスタ化し、フレーム内の融合済み点リストを返す。
    各要素: {"world_xy": np.ndarray, "gt_name": str(クラスタ内の多数決名、評価用)}"""
    assignments, _gt_same_pairs = compute_camera_assignments_by_distance(per_camera)
    accepted_pairs = [
        ((a.cam_i, a.idx_i), (a.cam_j, a.idx_j))
        for a in assignments if a.dist <= dist_thr
    ]
    all_keys = [(cam_id, idx) for cam_id, items in per_camera.items() for idx in range(len(items))]
    if not all_keys:
        return []
    roots = union_find_clusters(all_keys, accepted_pairs)

    clusters: Dict[Tuple[str, int], List[Tuple[str, np.ndarray]]] = defaultdict(list)
    for key in all_keys:
        cam_id, idx = key
        name, world_xy = per_camera[cam_id][idx]
        clusters[roots[key]].append((name, world_xy))

    fused = []
    for members in clusters.values():
        xy_mean = np.mean(np.array([m[1] for m in members]), axis=0)
        names = [m[0] for m in members]
        majority_name = max(set(names), key=names.count)
        fused.append({"world_xy": xy_mean, "gt_name": majority_name, "n_cams": len(members)})
    return fused


class SimpleBevTracker:
    """フレーム間の最近傍(距離閾値付きHungarian)によるBEV版の簡易トラッカー。"""

    def __init__(self, dist_thr: float, max_missed: int):
        self.dist_thr = dist_thr
        self.max_missed = max_missed
        self.tracks: Dict[int, dict] = {}  # track_id -> {"xy": ndarray, "missed": int}
        self.next_id = 1

    def update(self, fused_points: List[dict]) -> List[Tuple[int, np.ndarray]]:
        from scipy.optimize import linear_sum_assignment

        track_ids = list(self.tracks.keys())
        n_trk, n_det = len(track_ids), len(fused_points)
        matched_trk, matched_det = set(), set()

        if n_trk > 0 and n_det > 0:
            cost = np.zeros((n_trk, n_det))
            for i, tid in enumerate(track_ids):
                for j, fp in enumerate(fused_points):
                    cost[i, j] = float(np.linalg.norm(self.tracks[tid]["xy"] - fp["world_xy"]))
            row_idx, col_idx = linear_sum_assignment(cost)
            for i, j in zip(row_idx, col_idx):
                if cost[i, j] <= self.dist_thr:
                    tid = track_ids[i]
                    self.tracks[tid]["xy"] = fused_points[j]["world_xy"]
                    self.tracks[tid]["missed"] = 0
                    matched_trk.add(tid)
                    matched_det.add(j)

        for tid in track_ids:
            if tid not in matched_trk:
                self.tracks[tid]["missed"] += 1

        self.tracks = {
            tid: t for tid, t in self.tracks.items() if t["missed"] <= self.max_missed
        }

        for j, fp in enumerate(fused_points):
            if j not in matched_det:
                self.tracks[self.next_id] = {"xy": fp["world_xy"], "missed": 0}
                self.next_id += 1

        return [(tid, t["xy"]) for tid, t in self.tracks.items() if t["missed"] == 0]


TRACKEVAL_DIR = "/home/ara/myproject/project2/TrackEval"
if TRACKEVAL_DIR not in sys.path:
    sys.path.insert(0, TRACKEVAL_DIR)


def compute_hota_trackeval(frames_data: List[dict], dist_cap_m: float) -> dict:
    """本家TrackEvalのtrackeval.metrics.HOTAを直接呼び出す(ファイルI/O不要)。
    phase3_bev_multicam_yolo_check.pyのcompute_hota_trackeval()と同一ロジック。
    HOTA.eval_sequence()はIoUを前提にしていないので、代わりに距離ベースの類似度
    sim = max(0, 1 - dist/dist_cap_m) を similarity_scores として渡す。

    frames_data: 各要素が {"gt_names": [...], "gt_points": [...], "hyp_ids": [...], "hyp_points": [...]}
    """
    import trackeval

    gt_name_to_idx: Dict[str, int] = {}
    trk_id_to_idx: Dict[int, int] = {}
    for fr in frames_data:
        for name in fr["gt_names"]:
            if name not in gt_name_to_idx:
                gt_name_to_idx[name] = len(gt_name_to_idx)
        for tid in fr["hyp_ids"]:
            if tid not in trk_id_to_idx:
                trk_id_to_idx[tid] = len(trk_id_to_idx)

    gt_ids_per_frame, tracker_ids_per_frame, similarity_per_frame = [], [], []
    num_gt_dets, num_tracker_dets = 0, 0

    for fr in frames_data:
        gt_idx = np.array([gt_name_to_idx[n] for n in fr["gt_names"]], dtype=int)
        trk_idx = np.array([trk_id_to_idx[t] for t in fr["hyp_ids"]], dtype=int)
        gt_ids_per_frame.append(gt_idx)
        tracker_ids_per_frame.append(trk_idx)
        num_gt_dets += len(gt_idx)
        num_tracker_dets += len(trk_idx)

        if len(gt_idx) == 0 or len(trk_idx) == 0:
            similarity_per_frame.append(np.zeros((len(gt_idx), len(trk_idx))))
            continue

        gt_pts = np.array(fr["gt_points"])
        hyp_pts = np.array(fr["hyp_points"])
        dist = np.linalg.norm(gt_pts[:, None, :] - hyp_pts[None, :, :], axis=-1)
        sim = np.clip(1.0 - dist / dist_cap_m, 0.0, 1.0)
        similarity_per_frame.append(sim)

    data = {
        "num_gt_ids": len(gt_name_to_idx),
        "num_tracker_ids": len(trk_id_to_idx),
        "num_gt_dets": num_gt_dets,
        "num_tracker_dets": num_tracker_dets,
        "gt_ids": gt_ids_per_frame,
        "tracker_ids": tracker_ids_per_frame,
        "similarity_scores": similarity_per_frame,
    }

    hota_metric = trackeval.metrics.HOTA()
    res = hota_metric.eval_sequence(data)
    return res


def main():
    import motmetrics as mm

    print("floorplan.json 読み込み中...")
    cameras_json = load_floorplan_cameras(FLOORPLAN_JSON_PATH)
    cameras_json = apply_camera_overrides(cameras_json)
    poses = resolve_poses(cameras_json, root="Area1", fmt=ROT_FMT)

    xml_dir = Path(XML_DIR)
    frame_indices = list(range(FRAME_START, FRAME_END + 1, FRAME_STEP))

    tracker = SimpleBevTracker(dist_thr=DIST_THR_TRACK_M, max_missed=MAX_MISSED_FRAMES)
    acc = mm.MOTAccumulator(auto_id=True)
    frames_data: List[dict] = []  # HOTA計算用に生データを溜めておく

    pose_tag_seen = False
    n_processed = 0

    for frame_idx in frame_indices:
        xml_path = xml_dir / f"{frame_name(frame_idx)}.xml"
        if not xml_path.exists():
            continue

        objects = parse_voc_annotation(xml_path)
        if any(o["pose"] is not None for o in objects):
            pose_tag_seen = True

        per_camera = build_world_points_per_camera(objects, poses, IMAGE_WIDTH_PX)
        fused_points = fuse_frame(per_camera, DIST_THR_FUSION_M)

        # --- GT側: name -> このフレームでの真の世界座標(複数カメラに映っていれば平均) ---
        gt_world: Dict[str, List[np.ndarray]] = defaultdict(list)
        for cam_id, items in per_camera.items():
            for name, xy in items:
                gt_world[name].append(xy)
        gt_ids = [name_to_gt_id(name) for name in gt_world]
        gt_points = [np.mean(np.array(xys), axis=0) for xys in gt_world.values()]

        # --- 追跡更新 ---
        track_outputs = tracker.update(fused_points)
        hyp_ids = [tid for tid, _xy in track_outputs]
        hyp_points = [xy for _tid, xy in track_outputs]

        if gt_points and hyp_points:
            dist = mm.distances.norm2squared_matrix(
                np.array(gt_points), np.array(hyp_points), max_d2=(2.0 * DIST_THR_TRACK_M) ** 2
            )
        else:
            dist = np.empty((len(gt_points), len(hyp_points)))

        acc.update(gt_ids, hyp_ids, dist)
        frames_data.append({
            "gt_names": list(gt_world.keys()), "gt_points": gt_points,
            "hyp_ids": hyp_ids, "hyp_points": hyp_points,
        })
        n_processed += 1
        print(f"  {frame_name(frame_idx)}: GT個体={len(gt_points)} 融合点={len(fused_points)} "
              f"track出力={len(track_outputs)}")

    print(f"\n処理フレーム数: {n_processed}/{len(frame_indices)}")
    print(f"GT XMLにpose(lying/standing)タグ: {'あり' if pose_tag_seen else 'なし(全個体デフォルト高さ' + str(DEFAULT_HEIGHT_M) + 'm使用)'}")

    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=["idf1", "idp", "idr", "mota", "motp", "num_switches",
                 "num_objects", "num_matches", "num_false_positives", "num_misses"],
        name="multicam_bev_gt",
    )
    strsummary = mm.io.render_summary(
        summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names
    )
    print(f"\n{strsummary}")
    print(f"\n(参考: 単一カメラPhase1上限 IDF1=96.6%, Phase2実運用 IDF1=86.4%)")

    dist_cap_m = 2.0 * DIST_THR_TRACK_M
    hota_res = compute_hota_trackeval(frames_data, dist_cap_m=dist_cap_m)
    print(f"\n=== HOTA(本家TrackEval, alpha=0.05〜0.95を距離閾値<= (1-alpha)*{dist_cap_m}mに対応させて計算) ===")
    print(f"HOTA(平均)  = {np.mean(hota_res['HOTA']):.3f}")
    print(f"DetA(平均)  = {np.mean(hota_res['DetA']):.3f}")
    print(f"AssA(平均)  = {np.mean(hota_res['AssA']):.3f}")
    print(f"LocA(平均)  = {np.mean(hota_res['LocA']):.3f}")
    print(f"HOTA(0)     = {hota_res['HOTA(0)']:.3f}  (alpha->0での極限値、参考)")

    csv_path = OUTPUT_DIR / "phase3_bev_multicam_gt_summary.csv"
    summary.to_csv(csv_path)
    hota_csv_path = csv_path.with_name("phase3_bev_multicam_gt_hota.csv")
    with open(hota_csv_path, "w") as f:
        f.write("alpha,HOTA,DetA,AssA,LocA\n")
        for a, alpha in enumerate(np.arange(0.05, 0.99, 0.05)):
            f.write(f"{alpha:.2f},{hota_res['HOTA'][a]},{hota_res['DetA'][a]},"
                    f"{hota_res['AssA'][a]},{hota_res['LocA'][a]}\n")
    print(f"\nCSV保存: {csv_path}")
    print(f"HOTA内訳CSV保存: {hota_csv_path}")


if __name__ == "__main__":
    main()