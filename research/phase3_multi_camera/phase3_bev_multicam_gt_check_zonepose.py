"""
phase3_bev_multicam_gt_check_zonepose.py

phase3_bev_multicam_gt_check.py と同じ「複数フレームを読み込み→BEV融合→
SimpleBevTrackerで追跡→motmetrics/HOTAでIDF1・MOTA等を算出」という
フレーム間追跡精度の評価枠組みはそのまま流用し、フレーム内のカメラ間融合ロジック
だけを、make_bev_zone_pose_fusion.py で作った以下3制約版に差し替えたもの
(エピポーラ拘束は使わない):

  1. 有効範囲外の除外   : 各カメラの専属範囲+隣接カメラとの重なり範囲を超えて
                          投影された検出(魚眼の周辺視野で誤差が大きい)を除外
  2. 隣接カメラ制約     : Camera_1-Camera_2, Camera_2-Camera_3 の隣接ペアのみを
                          統合対象にする(物理的に重ならないCamera_1-Camera_3は除外)
  3. 姿勢クラス制約     : lying/standingが一致する検出同士だけを統合候補にする

またHungarian割当のコスト行列は、距離閾値を超える組を生の距離のまま渡さず
1e6(実質除外)に固定してから linear_sum_assignment に渡す
(make_bev_zone_pose_fusion.py で見つけた「閾値超えペアがグローバル最適化のために
正しい近距離ペアの割当枠を奪ってしまう」不具合の修正込み)。

phase3_bev_multicam_gt_check.py(Mootrack_core.compute_camera_assignments_by_distance
を使う版、Hungarianコストが同じ問題を抱えている)と同じフレーム範囲・同じ高さ仮定・
同じ追跡パラメータで実行し、IDF1/MOTA/HOTAを比較することで、
「フレーム内融合ロジックの違いがフレーム間追跡精度にどう影響するか」を見るためのもの。

--- 使い方 ---
    python phase3_bev_multicam_gt_check_zonepose.py
"""

import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

# 自己配置型の相対パス(フォルダ再編でも壊れないように)
MOOTRACK_CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calibration")
if MOOTRACK_CODE_DIR not in sys.path:
    sys.path.insert(0, MOOTRACK_CODE_DIR)

from Mootrack_core import (
    NUM_CAMERAS, CAM_IDS, ROT_FMT,
    get_intrinsics,
    load_floorplan_cameras, apply_camera_overrides, resolve_poses,
    unproject_to_floor,
    union_find_clusters,
)

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
# phase3_bev_multicam_gt_check.py と同じ複数フレームの入力元(直接比較するため)
XML_DIR = "/mnt/vmlqnap02/dataset/MooTrack360/1-hour-farm1-area1-sample-sequence/xml-1fps"
# 注意: このFLOORPLAN_JSON_PATHは以前から要検証(値が古い可能性あり)と指摘済みのファイル
FLOORPLAN_JSON_PATH = "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/calibration/Farm2_Area_1_Artboard_config_.json"
OUTPUT_DIR = Path(
    "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/research/idf1_check"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FRAME_START = 0
FRAME_END = 290
FRAME_STEP = 10

IMAGE_WIDTH_PX = 8832

# 高さ仮定はphase3_bev_multicam_gt_check.pyと同じ値に揃える(融合ロジックの違いだけを
# 比較するため)。make_bev_zone_pose_fusion.pyでは現在0.5/0.75を試しているので、
# 高さ仮定自体の効果も見たい場合はここを書き換えて再実行する。
LYING_HEIGHT_M = 0.6
STANDING_HEIGHT_M = 1.0
DEFAULT_HEIGHT_M = 0.8

DIST_THR_FUSION_M = 1.4      # カメラ間融合の距離閾値(phase3と同じ)
OVERLAP_HALF_WIDTH_M = 1.5    # 隣接カメラ境界からの片側の幅(合計3m幅)

DIST_THR_TRACK_M = 1.0        # フレーム間追跡の距離閾値(phase3と同じ)
MAX_MISSED_FRAMES = 2


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


def name_to_gt_id(name: str) -> int:
    m = re.search(r"\d+", name)
    if not m:
        raise ValueError(f"nameから数値IDを抽出できません: {name}")
    return int(m.group())


def height_for(pose: Optional[str]) -> float:
    if pose == "lying":
        return LYING_HEIGHT_M
    if pose == "standing":
        return STANDING_HEIGHT_M
    return DEFAULT_HEIGHT_M


# ---------------------------------------------------------------------------
# 1. 各カメラの検出をワールド座標へ(姿勢別高さで逆投影)
#    make_bev_zone_pose_fusion.py の compute_world_points_with_pose と同じロジック
# ---------------------------------------------------------------------------

def compute_world_points_with_pose(objects: List[dict], poses, img_w: float) -> Dict[str, List[dict]]:
    """cam_id -> [{"name":..., "pose":..., "world_xy":...}, ...]"""
    tile_w = img_w / NUM_CAMERAS
    by_cam_objs: Dict[str, List[dict]] = defaultdict(list)
    for obj in objects:
        center_x_full = (obj["xmin"] + obj["xmax"]) / 2.0
        cam_idx = int(center_x_full // tile_w)
        cam_idx = max(0, min(cam_idx, NUM_CAMERAS - 1))
        by_cam_objs[CAM_IDS[cam_idx]].append(obj)

    per_camera: Dict[str, List[dict]] = {cid: [] for cid in CAM_IDS}
    for cam_id, objs in by_cam_objs.items():
        cam_idx = CAM_IDS.index(cam_id)
        fx, fy, cx, cy, xi, alpha = get_intrinsics(cam_id)
        cam_pose = poses[cam_id]

        pts_px, heights = [], []
        for obj in objs:
            local_x = (obj["xmin"] + obj["xmax"]) / 2.0 - cam_idx * tile_w
            local_y = (obj["ymin"] + obj["ymax"]) / 2.0
            pts_px.append([local_x, local_y])
            heights.append(height_for(obj["pose"]))
        pts_px = np.array(pts_px, dtype=np.float64)
        floor_z = -np.array(heights, dtype=np.float64)

        world_xy, valid = unproject_to_floor(
            pts_px, cam_pose.R, cam_pose.t, fx, fy, cx, cy, xi, alpha, floor_z=floor_z
        )
        for obj, xy, ok in zip(objs, world_xy, valid):
            if not ok:
                continue
            per_camera[cam_id].append({"name": obj["name"], "pose": obj["pose"], "world_xy": xy})

    return per_camera


# ---------------------------------------------------------------------------
# 2. 隣接カメラの重なり範囲・各カメラの有効範囲(フレームに依存しないので1回だけ計算)
# ---------------------------------------------------------------------------

def get_adjacent_zones(poses, cam_ids: List[str], half_width: float) -> Dict[Tuple[str, str], Tuple[float, float]]:
    zones = {}
    for cam_i, cam_j in zip(cam_ids[:-1], cam_ids[1:]):
        x_i = poses[cam_i].t[0]
        x_j = poses[cam_j].t[0]
        boundary = (x_i + x_j) / 2.0
        zones[(cam_i, cam_j)] = (boundary - half_width, boundary + half_width)
        print(f"  {cam_i}-{cam_j} 境界: x={boundary:.2f}m -> 重なり範囲 [{boundary - half_width:.2f}, {boundary + half_width:.2f}]")
    return zones


def get_camera_valid_ranges(cam_ids: List[str], zones: Dict[Tuple[str, str], Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
    ordered_pairs = list(zip(cam_ids[:-1], cam_ids[1:]))
    ranges: Dict[str, Tuple[float, float]] = {}
    ranges[cam_ids[0]] = (-np.inf, zones[ordered_pairs[0]][1])
    ranges[cam_ids[-1]] = (zones[ordered_pairs[-1]][0], np.inf)
    for i in range(1, len(cam_ids) - 1):
        lo = zones[ordered_pairs[i - 1]][0]
        hi = zones[ordered_pairs[i]][1]
        ranges[cam_ids[i]] = (lo, hi)
    return ranges


def filter_out_of_range_detections(
    per_camera: Dict[str, List[dict]], valid_ranges: Dict[str, Tuple[float, float]],
) -> Dict[str, List[dict]]:
    filtered: Dict[str, List[dict]] = {cid: [] for cid in per_camera}
    for cam_id, items in per_camera.items():
        lo, hi = valid_ranges[cam_id]
        for it in items:
            if lo <= it["world_xy"][0] <= hi:
                filtered[cam_id].append(it)
    return filtered


# ---------------------------------------------------------------------------
# 3. 隣接カメラ制約 + 姿勢クラス制約でマッチング(Hungarianのコストは
#    閾値以内の組にしか実距離を入れない = 修正済み版)
# ---------------------------------------------------------------------------

def match_zone_pose_constrained(
    per_camera: Dict[str, List[dict]],
    cam_ids: List[str],
    zones: Dict[Tuple[str, str], Tuple[float, float]],
    dist_thr: float,
) -> List[Tuple[Tuple[str, int], Tuple[str, int]]]:
    accepted_pairs: List[Tuple[Tuple[str, int], Tuple[str, int]]] = []

    for cam_i, cam_j in zones.keys():
        items_i = per_camera[cam_i]
        items_j = per_camera[cam_j]
        if not items_i or not items_j:
            continue

        n_i, n_j = len(items_i), len(items_j)
        cost = np.full((n_i, n_j), 1e6)
        for a, it_i in enumerate(items_i):
            for b, it_j in enumerate(items_j):
                if it_i["pose"] is not None and it_j["pose"] is not None and it_i["pose"] != it_j["pose"]:
                    continue
                d = float(np.linalg.norm(it_i["world_xy"] - it_j["world_xy"]))
                if d <= dist_thr:
                    cost[a, b] = d
                # 閾値超えは生の距離を入れず1e6のままにする(Hungarianの誤選択を防ぐ)

        row_idx, col_idx = linear_sum_assignment(cost)
        for a, b in zip(row_idx, col_idx):
            if cost[a, b] <= dist_thr:
                accepted_pairs.append(((cam_i, a), (cam_j, b)))

    return accepted_pairs


def fuse_frame_zone_pose(
    per_camera_filtered: Dict[str, List[dict]],
    cam_ids: List[str],
    zones: Dict[Tuple[str, str], Tuple[float, float]],
    dist_thr: float,
) -> List[dict]:
    """有効範囲フィルタ済みのper_cameraから、zone+pose制約でマッチングし、
    Union-Findでクラスタ化して融合点リストを返す。
    各要素: {"world_xy": ndarray, "gt_name": str(クラスタ内の多数決名、評価用), "n_cams": int}"""
    accepted_pairs = match_zone_pose_constrained(per_camera_filtered, cam_ids, zones, dist_thr)
    all_keys = [(cam_id, idx) for cam_id, items in per_camera_filtered.items() for idx in range(len(items))]
    if not all_keys:
        return []
    roots = union_find_clusters(all_keys, accepted_pairs)

    clusters: Dict[Tuple[str, int], List[dict]] = defaultdict(list)
    for key in all_keys:
        cam_id, idx = key
        clusters[roots[key]].append(per_camera_filtered[cam_id][idx])

    fused = []
    for members in clusters.values():
        xy_mean = np.mean(np.array([m["world_xy"] for m in members]), axis=0)
        names = [m["name"] for m in members]
        majority_name = max(set(names), key=names.count)
        fused.append({"world_xy": xy_mean, "gt_name": majority_name, "n_cams": len(members)})
    return fused


# ---------------------------------------------------------------------------
# フレーム間追跡(phase3_bev_multicam_gt_check.pyと同一)
# ---------------------------------------------------------------------------

class SimpleBevTracker:
    """フレーム間の最近傍(距離閾値付きHungarian)によるBEV版の簡易トラッカー。"""

    def __init__(self, dist_thr: float, max_missed: int):
        self.dist_thr = dist_thr
        self.max_missed = max_missed
        self.tracks: Dict[int, dict] = {}
        self.next_id = 1

    def update(self, fused_points: List[dict]) -> List[Tuple[int, np.ndarray]]:
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
    """本家TrackEvalのtrackeval.metrics.HOTAを直接呼び出す。
    phase3_bev_multicam_gt_check.pyのcompute_hota_trackeval()と同一ロジック。"""
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
    for cid in CAM_IDS:
        print(f"  {cid}: t={poses[cid].t.round(2)}")

    print("\n隣接カメラの重なり範囲(±{:.1f}m)を計算中...".format(OVERLAP_HALF_WIDTH_M))
    zones = get_adjacent_zones(poses, CAM_IDS, OVERLAP_HALF_WIDTH_M)
    valid_ranges = get_camera_valid_ranges(CAM_IDS, zones)
    for cid in CAM_IDS:
        lo, hi = valid_ranges[cid]
        print(f"  {cid} 有効範囲: [{lo:.2f}, {hi:.2f}]")

    xml_dir = Path(XML_DIR)
    frame_indices = list(range(FRAME_START, FRAME_END + 1, FRAME_STEP))

    tracker = SimpleBevTracker(dist_thr=DIST_THR_TRACK_M, max_missed=MAX_MISSED_FRAMES)
    acc = mm.MOTAccumulator(auto_id=True)
    frames_data: List[dict] = []

    pose_tag_seen = False
    n_processed = 0

    for frame_idx in frame_indices:
        xml_path = xml_dir / f"{frame_name(frame_idx)}.xml"
        if not xml_path.exists():
            continue

        objects = parse_voc_annotation(xml_path)
        if any(o["pose"] is not None for o in objects):
            pose_tag_seen = True

        # GT側の真の位置は、有効範囲フィルタの影響を受けない生の検出から作る
        # (融合パイプラインの候補絞り込みとは独立に「その名前の牛の真の位置」を表すため)
        per_camera_raw = compute_world_points_with_pose(objects, poses, IMAGE_WIDTH_PX)
        gt_world: Dict[str, List[np.ndarray]] = defaultdict(list)
        for cam_id, items in per_camera_raw.items():
            for it in items:
                gt_world[it["name"]].append(it["world_xy"])
        gt_ids = [name_to_gt_id(name) for name in gt_world]
        gt_points = [np.mean(np.array(xys), axis=0) for xys in gt_world.values()]

        # 融合パイプラインは有効範囲フィルタ済みの検出だけを使う
        per_camera_filtered = filter_out_of_range_detections(per_camera_raw, valid_ranges)
        fused_points = fuse_frame_zone_pose(per_camera_filtered, CAM_IDS, zones, DIST_THR_FUSION_M)

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
        name="multicam_bev_gt_zonepose",
    )
    strsummary = mm.io.render_summary(
        summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names
    )
    print(f"\n{strsummary}")
    print(f"\n(参考: phase3_bev_multicam_gt_check.py [compute_camera_assignments_by_distance版] "
          f"のIDF1と比較すること。単一カメラPhase1上限 IDF1=96.6%, Phase2実運用 IDF1=86.4%)")

    dist_cap_m = 2.0 * DIST_THR_TRACK_M
    hota_res = compute_hota_trackeval(frames_data, dist_cap_m=dist_cap_m)
    print(f"\n=== HOTA(本家TrackEval, alpha=0.05〜0.95を距離閾値<= (1-alpha)*{dist_cap_m}mに対応させて計算) ===")
    print(f"HOTA(平均)  = {np.mean(hota_res['HOTA']):.3f}")
    print(f"DetA(平均)  = {np.mean(hota_res['DetA']):.3f}")
    print(f"AssA(平均)  = {np.mean(hota_res['AssA']):.3f}")
    print(f"LocA(平均)  = {np.mean(hota_res['LocA']):.3f}")
    print(f"HOTA(0)     = {hota_res['HOTA(0)']:.3f}  (alpha->0での極限値、参考)")

    csv_path = OUTPUT_DIR / "phase3_bev_multicam_gt_zonepose_summary.csv"
    summary.to_csv(csv_path)
    hota_csv_path = csv_path.with_name("phase3_bev_multicam_gt_zonepose_hota.csv")
    with open(hota_csv_path, "w") as f:
        f.write("alpha,HOTA,DetA,AssA,LocA\n")
        for a, alpha in enumerate(np.arange(0.05, 0.99, 0.05)):
            f.write(f"{alpha:.2f},{hota_res['HOTA'][a]},{hota_res['DetA'][a]},"
                    f"{hota_res['AssA'][a]},{hota_res['LocA'][a]}\n")
    print(f"\nCSV保存: {csv_path}")
    print(f"HOTA内訳CSV保存: {hota_csv_path}")


if __name__ == "__main__":
    main()