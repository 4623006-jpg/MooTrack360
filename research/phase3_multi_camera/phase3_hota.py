"""
phase3_bev_multicam_yolo_check.py

【vml36で実行】

phase3_bev_multicam_gt_check.py(GT bboxを使った融合+追跡ロジック単体の上限性能
テスト、IDF1=73.3%)に対応する、実際のYOLO検出を使った実運用版。Phase1に対する
Phase2と同じ関係。

GT版との違いは検出の入力元だけ:
  - 各カメラのタイル画像でYOLO推論(phase2_yolo_conftrack_check.pyと同じく、
    ステッチ画像全体ではなくカメラ別クロップに対して推論する)
  - YOLO検出にはGTのようなpose(lying/standing)情報が無いため、固定の高さ仮定
    (DEFAULT_HEIGHT_M)で床面に逆投影する
  - 融合・追跡ロジック(距離閾値によるHungarian+Union-Find融合、フレーム間
    最近傍追跡)はGT版と全く同じコードを流用

評価用のGT(個体名・真の位置)は別途XMLから読み込み、YOLO由来のtrack出力と
distance-gateで突き合わせてIDF1を算出する。

--- 使い方 ---
    python phase3_bev_multicam_yolo_check.py
"""

import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

MOOTRACK_CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calibration")
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
# 設定(phase3_bev_multicam_gt_check.pyと共通の値はそのまま揃える)
# ---------------------------------------------------------------------------
XML_DIR = "/mnt/vmlqnap02/dataset/MooTrack360/1-hour-farm1-area1-sample-sequence/xml-1fps"
IMAGE_DIR = "/mnt/vmlqnap02/dataset/MooTrack360/1-hour-farm1-area1-sample-sequence/1-hour-farm1-area1-frames"
FLOORPLAN_JSON_PATH = "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/calibration/Farm2_Area_1_Artboard_config_.json"
OUTPUT_DIR = Path(
    "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/research/idf1_check"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

YOLO_WEIGHTS_PATH = "/mnt/vmlqnap02/dataset/MooTrack360/Trained models/yolo11x/train_ultralytics/weights/best.pt"
YOLO_CONF_THRESH = 0.1

FRAME_START = 0
FRAME_END = 290
FRAME_STEP = 10

IMAGE_WIDTH_PX = 8832

LYING_HEIGHT_M = 0.6
STANDING_HEIGHT_M = 1.0
DEFAULT_HEIGHT_M = 0.8  # YOLO検出にはpose情報が無いので固定でこれを使う

DIST_THR_FUSION_M = 1.4
DIST_THR_TRACK_M = 1.0
MAX_MISSED_FRAMES = 2

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


def gt_world_points_this_frame(objects: List[dict], poses: Dict[str, "Pose"], image_width: float) -> Dict[str, List[np.ndarray]]:
    """評価用: GT個体名 -> このフレームでの世界座標リスト(複数カメラに映っていれば複数)"""
    tile_w = image_width / NUM_CAMERAS
    result: Dict[str, List[np.ndarray]] = defaultdict(list)
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
            pts_px, cam_pose.R, cam_pose.t, fx, fy, cx, cy, xi, alpha, floor_z=-height_m,
        )
        if valid[0]:
            result[obj["name"]].append(world_xy[0])
    return result


def yolo_world_points_per_camera(
    yolo_model, image_path: Path, poses: Dict[str, "Pose"],
) -> Dict[str, List[Tuple[None, np.ndarray]]]:
    """YOLO検出(nameは無いのでNone)を各カメラのタイルクロップで推論し、
    bbox中心を固定高さで床面に逆投影する。"""
    full_img = cv2.imread(str(image_path))
    per_camera: Dict[str, List[Tuple[None, np.ndarray]]] = {cid: [] for cid in CAM_IDS}
    if full_img is None:
        return per_camera

    tile_w = IMAGE_WIDTH_PX / NUM_CAMERAS

    for cam_idx, cam_id in enumerate(CAM_IDS):
        x0 = int(cam_idx * tile_w)
        x1_edge = int((cam_idx + 1) * tile_w)
        frame_img = full_img[:, x0:x1_edge]

        results = yolo_model.predict(
            frame_img, conf=YOLO_CONF_THRESH, device=DEVICE, verbose=False
        )[0]
        if len(results.boxes) == 0:
            continue

        fx, fy, cx, cy, xi, alpha = get_intrinsics(cam_id)
        pts_px = []
        for box in results.boxes:
            bx1, by1, bx2, by2 = box.xyxy[0].tolist()
            pts_px.append([(bx1 + bx2) / 2.0, (by1 + by2) / 2.0])
        pts_px = np.array(pts_px, dtype=np.float64)

        cam_pose = poses[cam_id]
        world_xy, valid = unproject_to_floor(
            pts_px, cam_pose.R, cam_pose.t, fx, fy, cx, cy, xi, alpha,
            floor_z=-DEFAULT_HEIGHT_M,
        )
        for j in range(len(pts_px)):
            if valid[j]:
                per_camera[cam_id].append((None, world_xy[j]))

    return per_camera


def fuse_frame(per_camera: Dict[str, List[Tuple[Optional[str], np.ndarray]]], dist_thr: float) -> List[dict]:
    assignments, _gt_same_pairs = compute_camera_assignments_by_distance(per_camera)
    accepted_pairs = [
        ((a.cam_i, a.idx_i), (a.cam_j, a.idx_j))
        for a in assignments if a.dist <= dist_thr
    ]
    all_keys = [(cam_id, idx) for cam_id, items in per_camera.items() for idx in range(len(items))]
    if not all_keys:
        return []
    roots = union_find_clusters(all_keys, accepted_pairs)

    clusters: Dict[Tuple[str, int], List[np.ndarray]] = defaultdict(list)
    for key in all_keys:
        cam_id, idx = key
        _name, world_xy = per_camera[cam_id][idx]
        clusters[roots[key]].append(world_xy)

    fused = []
    for members in clusters.values():
        xy_mean = np.mean(np.array(members), axis=0)
        fused.append({"world_xy": xy_mean, "n_cams": len(members)})
    return fused


class SimpleBevTracker:
    def __init__(self, dist_thr: float, max_missed: int):
        self.dist_thr = dist_thr
        self.max_missed = max_missed
        self.tracks: Dict[int, dict] = {}
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

        self.tracks = {tid: t for tid, t in self.tracks.items() if t["missed"] <= self.max_missed}

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
    HOTA.eval_sequence()はIoUを前提にしていないので、代わりに距離ベースの類似度
    sim = max(0, 1 - dist/dist_cap_m) を similarity_scores として渡す。
    これによりalpha(0.05刻みで0.05〜0.95)は「dist <= (1-alpha)*dist_cap_m」という
    距離条件に対応する。

    frames_data: 各要素が {"gt_names": [...], "gt_points": [...], "hyp_ids": [...], "hyp_points": [...]}
    """
    import trackeval

    # 全フレームを通してのユニークなGT名・トラックIDを集め、0始まりの密なインデックスに変換する
    # (HOTA内部でnum_gt_ids x num_tracker_idsの行列を作るため必須)
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
    from ultralytics import YOLO

    print("floorplan.json 読み込み中...")
    cameras_json = load_floorplan_cameras(FLOORPLAN_JSON_PATH)
    cameras_json = apply_camera_overrides(cameras_json)
    poses = resolve_poses(cameras_json, root="Area1", fmt=ROT_FMT)

    print(f"YOLOモデル読み込み中: {YOLO_WEIGHTS_PATH}")
    yolo_model = YOLO(YOLO_WEIGHTS_PATH)

    xml_dir = Path(XML_DIR)
    image_dir = Path(IMAGE_DIR)
    frame_indices = list(range(FRAME_START, FRAME_END + 1, FRAME_STEP))

    tracker = SimpleBevTracker(dist_thr=DIST_THR_TRACK_M, max_missed=MAX_MISSED_FRAMES)
    acc = mm.MOTAccumulator(auto_id=True)
    n_processed = 0
    frames_data: List[dict] = []  # HOTA計算用に生データを溜めておく

    for frame_idx in frame_indices:
        xml_path = xml_dir / f"{frame_name(frame_idx)}.xml"
        image_path = image_dir / f"{frame_name(frame_idx)}.jpg"
        if not xml_path.exists() or not image_path.exists():
            continue

        gt_objects = parse_voc_annotation(xml_path)
        gt_world = gt_world_points_this_frame(gt_objects, poses, IMAGE_WIDTH_PX)
        gt_names = list(gt_world.keys())
        gt_ids = [name_to_gt_id(name) for name in gt_names]
        gt_points = [np.mean(np.array(gt_world[name]), axis=0) for name in gt_names]

        per_camera = yolo_world_points_per_camera(yolo_model, image_path, poses)
        fused_points = fuse_frame(per_camera, DIST_THR_FUSION_M)

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
            "gt_names": gt_names, "gt_points": gt_points,
            "hyp_ids": hyp_ids, "hyp_points": hyp_points,
        })
        n_processed += 1
        n_yolo_total = sum(len(v) for v in per_camera.values())
        print(f"  {frame_name(frame_idx)}: GT個体={len(gt_points)} YOLO検出={n_yolo_total} "
              f"融合点={len(fused_points)} track出力={len(track_outputs)}")

    print(f"\n処理フレーム数: {n_processed}/{len(frame_indices)}")

    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=["idf1", "idp", "idr", "mota", "motp", "num_switches",
                 "num_objects", "num_matches", "num_false_positives", "num_misses"],
        name="multicam_bev_yolo",
    )
    strsummary = mm.io.render_summary(
        summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names
    )
    print(f"\n{strsummary}")

    dist_cap_m = 2.0 * DIST_THR_TRACK_M
    hota_res = compute_hota_trackeval(frames_data, dist_cap_m=dist_cap_m)
    print(f"\n=== HOTA(本家TrackEval, alpha=0.05〜0.95を距離閾値<= (1-alpha)*{dist_cap_m}mに対応させて計算) ===")
    print(f"HOTA(平均)  = {np.mean(hota_res['HOTA']):.3f}")
    print(f"DetA(平均)  = {np.mean(hota_res['DetA']):.3f}")
    print(f"AssA(平均)  = {np.mean(hota_res['AssA']):.3f}")
    print(f"LocA(平均)  = {np.mean(hota_res['LocA']):.3f}")
    print(f"HOTA(0)     = {hota_res['HOTA(0)']:.3f}  (alpha->0での極限値、参考)")

    print(f"\n(参考: GTベース上限 IDF1=73.3%, 単一カメラPhase2実運用 IDF1=86.4%)")

    csv_path = OUTPUT_DIR / "phase3_bev_multicam_yolo_summary.csv"
    summary.to_csv(csv_path)
    hota_csv_path = csv_path.with_name("phase3_bev_multicam_yolo_hota.csv")
    with open(hota_csv_path, "w") as f:
        f.write("alpha,HOTA,DetA,AssA,LocA\n")
        for a, alpha in enumerate(np.arange(0.05, 0.99, 0.05)):
            f.write(f"{alpha:.2f},{hota_res['HOTA'][a]},{hota_res['DetA'][a]},"
                    f"{hota_res['AssA'][a]},{hota_res['LocA'][a]}\n")
    print(f"\nCSV保存: {csv_path}")
    print(f"HOTA内訳CSV保存: {hota_csv_path}")


if __name__ == "__main__":
    main()