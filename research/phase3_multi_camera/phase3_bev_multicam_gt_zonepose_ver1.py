"""
phase3_bev_multicam_gt_zonepose_ver1.py

phase3_bev_multicam_yolo_check_zonepose_ver6.py のYOLO検出を、GT(XML)のbboxに
置き換えた版。YOLOの検出ミス(見逃し・誤検出・重複)を取り除いた状態で、
「個体統合(曲線境界+隣接カメラ+姿勢制約+距離閾値)とトラッキング」だけの精度を測る。

差分:
  - 検出 = GTのbbox(中心を高さ仮定で床に投影)。YOLO・画像の読み込みは不要
  - 同一カメラ内の重複除去は行わない(GTには重複が無いため)
  - 評価のGTは、同じbboxを牛名ごとに平均した位置(従来版と同じ作り方)
  - それ以外(曲線境界の除外、牛舎範囲の除外、隣接ペアのHungarian統合、トラッカー、
    MOTA/IDF1/HOTAの計算、EXCLUDE_GT_NAMESの扱い)は従来版と同じ

読み方の注意: 検出もGTも同じ「bbox中心+高さ仮定の投影」で作っているため、
正しく統合された牛は位置がGTとほぼ一致する。この版で下がる指標は、統合の取り違えや
統合漏れ、範囲の除外、トラッキング(IDの継続)の問題が原因。YOLOとの差が、
YOLOの検出品質による悪化分の目安になる。

--- 使い方 ---
    python phase3_bev_multicam_gt_zonepose_ver1.py
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

MOOTRACK_CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calibration")
if MOOTRACK_CODE_DIR not in sys.path:
    sys.path.insert(0, MOOTRACK_CODE_DIR)

ANALYZE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analyze")
if ANALYZE_DIR not in sys.path:
    sys.path.insert(0, ANALYZE_DIR)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from Mootrack_core import (
    NUM_CAMERAS, CAM_IDS, ROT_FMT,
    Pose, get_intrinsics,
    load_floorplan_cameras, load_floorplan_dimensions, apply_camera_overrides, resolve_poses,
    unproject_to_floor,
    union_find_clusters,
    filter_outside_floorplan_bounds,
    ReIDBevTracker,
)

from curved_zone_utils_ver1 import (
    ZONE_HEIGHT_M, N_ANGLES,
    build_camera_polygons, filter_by_curved_zone,
    get_adjacent_pairs,
)

# ---------------------------------------------------------------------------
# 設定(従来版 phase3_bev_multicam_yolo_check_zonepose_ver6.py と揃える)
# ---------------------------------------------------------------------------
XML_DIR = "/mnt/vmlqnap02/dataset/MooTrack360/1-hour-farm1-area1-sample-sequence/xml-1fps"
# 注意: このFLOORPLAN_JSON_PATHは以前から要検証(値が古い可能性あり)と指摘済みのファイル
FLOORPLAN_JSON_PATH = "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/calibration/Farm2_Area_1_Artboard_config_.json"
OUTPUT_DIR = Path(
    "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/research/idf1_check"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FRAME_START = 0
FRAME_END = 1400
FRAME_STEP = 10

IMAGE_WIDTH_PX = 8832

LYING_HEIGHT_M = 0.5
STANDING_HEIGHT_M = 0.75
DEFAULT_HEIGHT_M = 0.8

DIST_THR_FUSION_M = 1.5
DIST_THR_TRACK_M = 1.5
MAX_MISSED_FRAMES = 2

USE_REID_TRACKER = True
EDGE_MARGIN_M = 0.7
REID_DIST_THR_M = 3.0
LOST_MAX_AGE = 10

USE_POSE_CONSTRAINT = True

# True: 従来版と同じく曲線境界/牛舎範囲の外の検出を除外する。False: 除外しない
# (GT入力でも範囲の除外がFNの原因になるかを切り分けるためのスイッチ)。
USE_ZONE_FILTER = True
USE_FLOORPLAN_FILTER = True

EXCLUDE_GT_NAMES = {"cow109", "cow115", "cow098"}


def frame_name(idx: int) -> str:
    return f"{idx:06d}"


def parse_voc_annotation(xml_path) -> List[dict]:
    root = ET.parse(xml_path).getroot()
    objects = []
    for obj in root.findall("object"):
        pose_el = obj.find("pose")
        bnd = obj.find("bndbox")
        objects.append({
            "name": obj.find("name").text,
            "pose": pose_el.text.strip().lower() if pose_el is not None and pose_el.text else None,
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
        if obj["name"] in EXCLUDE_GT_NAMES:
            continue
        center_x_full = (obj["xmin"] + obj["xmax"]) / 2.0
        cam_idx = max(0, min(int(center_x_full // tile_w), NUM_CAMERAS - 1))
        cam_id = CAM_IDS[cam_idx]

        fx, fy, cx, cy, xi, alpha = get_intrinsics(cam_id)
        local_x = center_x_full - cam_idx * tile_w
        local_y = (obj["ymin"] + obj["ymax"]) / 2.0
        pts_px = np.array([[local_x, local_y]], dtype=np.float64)
        cam_pose = poses[cam_id]
        world_xy, valid = unproject_to_floor(
            pts_px, cam_pose.R, cam_pose.t, fx, fy, cx, cy, xi, alpha, floor_z=-height_for(obj["pose"]),
        )
        if valid[0]:
            result[obj["name"]].append(world_xy[0])
    return result


def gt_detections_per_camera(objects: List[dict], poses: Dict[str, "Pose"]) -> Dict[str, List[dict]]:
    """YOLO検出の代わりに、GTのbboxをカメラごとの検出リスト
    [{"name", "pose", "world_xy"}, ...] にして返す(YOLO版と同じ形式)。"""
    tile_w = IMAGE_WIDTH_PX / NUM_CAMERAS
    per_camera: Dict[str, List[dict]] = {cid: [] for cid in CAM_IDS}
    for obj in objects:
        if obj["name"] in EXCLUDE_GT_NAMES:
            continue
        center_x_full = (obj["xmin"] + obj["xmax"]) / 2.0
        cam_idx = max(0, min(int(center_x_full // tile_w), NUM_CAMERAS - 1))
        cam_id = CAM_IDS[cam_idx]
        fx, fy, cx, cy, xi, alpha = get_intrinsics(cam_id)
        pts_px = np.array([[center_x_full - cam_idx * tile_w, (obj["ymin"] + obj["ymax"]) / 2.0]],
                          dtype=np.float64)
        cam_pose = poses[cam_id]
        world_xy, valid = unproject_to_floor(
            pts_px, cam_pose.R, cam_pose.t, fx, fy, cx, cy, xi, alpha, floor_z=-height_for(obj["pose"]),
        )
        if valid[0]:
            per_camera[cam_id].append({"name": obj["name"], "pose": obj["pose"], "world_xy": world_xy[0]})
    return per_camera


# ---------------------------------------------------------------------------
# 隣接カメラ制約 + 姿勢クラス制約でマッチング(従来版と同一)
# ---------------------------------------------------------------------------

def match_zone_pose_constrained(per_camera, adjacent_pairs, dist_thr, use_pose_constraint=True):
    accepted_pairs = []
    n_pose_rejected = 0
    for cam_i, cam_j in adjacent_pairs:
        items_i, items_j = per_camera[cam_i], per_camera[cam_j]
        if not items_i or not items_j:
            continue
        cost = np.full((len(items_i), len(items_j)), 1e6)
        for a, it_i in enumerate(items_i):
            for b, it_j in enumerate(items_j):
                d = float(np.linalg.norm(it_i["world_xy"] - it_j["world_xy"]))
                pose_mismatch = (it_i["pose"] is not None and it_j["pose"] is not None
                                 and it_i["pose"] != it_j["pose"])
                if use_pose_constraint and pose_mismatch:
                    if d <= dist_thr:
                        n_pose_rejected += 1
                    continue
                if d <= dist_thr:
                    cost[a, b] = d
        for a, b in zip(*linear_sum_assignment(cost)):
            if cost[a, b] <= dist_thr:
                accepted_pairs.append(((cam_i, a), (cam_j, b)))
    return accepted_pairs, n_pose_rejected


def fuse_frame_zone_pose(per_camera_filtered, adjacent_pairs, dist_thr, use_pose_constraint=True):
    """戻り値は (fused, n_pose_rejected, pair_stats)。pair_stats は牛名による統合の正誤
    (tp: 正しい統合, fn: 統合漏れ, fp: 誤統合) のペア数。"""
    accepted_pairs, n_pose_rejected = match_zone_pose_constrained(
        per_camera_filtered, adjacent_pairs, dist_thr, use_pose_constraint)
    all_keys = [(c, i) for c, items in per_camera_filtered.items() for i in range(len(items))]
    if not all_keys:
        return [], n_pose_rejected, (0, 0, 0)
    roots = union_find_clusters(all_keys, accepted_pairs)

    clusters: Dict = defaultdict(list)
    for key in all_keys:
        clusters[roots[key]].append(key)

    fused = []
    for members in clusters.values():
        xy = np.mean([per_camera_filtered[c][i]["world_xy"] for c, i in members], axis=0)
        fused.append({"world_xy": xy, "n_cams": len(members)})

    name_of = {k: per_camera_filtered[k[0]][k[1]]["name"] for k in all_keys}
    tp = fn = fp = 0
    for a in range(len(all_keys)):
        for b in range(a + 1, len(all_keys)):
            ka, kb = all_keys[a], all_keys[b]
            if ka[0] == kb[0]:
                continue
            same_name = name_of[ka] == name_of[kb]
            same_cl = roots[ka] == roots[kb]
            tp += same_name and same_cl
            fn += same_name and not same_cl
            fp += (not same_name) and same_cl
    return fused, n_pose_rejected, (tp, fn, fp)


class SimpleBevTracker:
    def __init__(self, dist_thr: float, max_missed: int):
        self.dist_thr = dist_thr
        self.max_missed = max_missed
        self.tracks: Dict[int, dict] = {}
        self.next_id = 1

    def update(self, fused_points: List[dict]) -> List[Tuple[int, np.ndarray]]:
        track_ids = list(self.tracks.keys())
        matched_trk, matched_det = set(), set()
        if track_ids and fused_points:
            cost = np.zeros((len(track_ids), len(fused_points)))
            for i, tid in enumerate(track_ids):
                for j, fp in enumerate(fused_points):
                    cost[i, j] = float(np.linalg.norm(self.tracks[tid]["xy"] - fp["world_xy"]))
            for i, j in zip(*linear_sum_assignment(cost)):
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
    import trackeval

    gt_name_to_idx: Dict[str, int] = {}
    trk_id_to_idx: Dict[int, int] = {}
    for fr in frames_data:
        for name in fr["gt_names"]:
            gt_name_to_idx.setdefault(name, len(gt_name_to_idx))
        for tid in fr["hyp_ids"]:
            trk_id_to_idx.setdefault(tid, len(trk_id_to_idx))

    gt_ids_per_frame, tracker_ids_per_frame, similarity_per_frame = [], [], []
    num_gt_dets = num_tracker_dets = 0
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
        dist = np.linalg.norm(np.array(fr["gt_points"])[:, None, :] - np.array(fr["hyp_points"])[None, :, :], axis=-1)
        similarity_per_frame.append(np.clip(1.0 - dist / dist_cap_m, 0.0, 1.0))

    data = {
        "num_gt_ids": len(gt_name_to_idx), "num_tracker_ids": len(trk_id_to_idx),
        "num_gt_dets": num_gt_dets, "num_tracker_dets": num_tracker_dets,
        "gt_ids": gt_ids_per_frame, "tracker_ids": tracker_ids_per_frame,
        "similarity_scores": similarity_per_frame,
    }
    return trackeval.metrics.HOTA().eval_sequence(data)


def main():
    import motmetrics as mm

    print("floorplan.json 読み込み中...")
    cameras_json = load_floorplan_cameras(FLOORPLAN_JSON_PATH)
    real_w, real_h = load_floorplan_dimensions(FLOORPLAN_JSON_PATH)
    cameras_json = apply_camera_overrides(cameras_json)
    poses = resolve_poses(cameras_json, root="Area1", fmt=ROT_FMT)

    cam_polygons = build_camera_polygons(poses, CAM_IDS, get_intrinsics, height_m=ZONE_HEIGHT_M, n_angles=N_ANGLES)
    adjacent_pairs = get_adjacent_pairs(CAM_IDS)

    print(f"\n入力: GTのbbox(YOLOなし) / 姿勢制約: {'有効' if USE_POSE_CONSTRAINT else '無効'} / "
          f"曲線境界除外: {USE_ZONE_FILTER} / 牛舎範囲除外: {USE_FLOORPLAN_FILTER}")
    print(f"トラッカー: {'ReIDBevTracker' if USE_REID_TRACKER else 'SimpleBevTracker(ReIDなし旧版)'}")

    if USE_REID_TRACKER:
        tracker = ReIDBevTracker(
            dist_thr=DIST_THR_TRACK_M, max_missed=MAX_MISSED_FRAMES,
            real_w=real_w, real_h=real_h, edge_margin_m=EDGE_MARGIN_M,
            reid_dist_thr=REID_DIST_THR_M, lost_max_age=LOST_MAX_AGE,
        )
    else:
        tracker = SimpleBevTracker(dist_thr=DIST_THR_TRACK_M, max_missed=MAX_MISSED_FRAMES)

    xml_dir = Path(XML_DIR)
    acc = mm.MOTAccumulator(auto_id=True)
    frames_data: List[dict] = []
    n_processed = 0
    tot_tp = tot_fn = tot_fp = 0

    for frame_idx in range(FRAME_START, FRAME_END + 1, FRAME_STEP):
        xml_path = xml_dir / f"{frame_name(frame_idx)}.xml"
        if not xml_path.exists():
            continue

        gt_objects = parse_voc_annotation(xml_path)
        gt_world = gt_world_points_this_frame(gt_objects, poses, IMAGE_WIDTH_PX)
        gt_names = list(gt_world.keys())
        gt_ids = [name_to_gt_id(n) for n in gt_names]
        gt_points = [np.mean(np.array(gt_world[n]), axis=0) for n in gt_names]

        per_camera = gt_detections_per_camera(gt_objects, poses)
        n_raw = sum(len(v) for v in per_camera.values())
        if USE_ZONE_FILTER:
            per_camera = filter_by_curved_zone(per_camera, cam_polygons)
        if USE_FLOORPLAN_FILTER:
            per_camera = filter_outside_floorplan_bounds(per_camera, real_w, real_h, margin_m=0.0, verbose=False)
        n_kept = sum(len(v) for v in per_camera.values())

        fused_points, n_pose_rejected, (tp, fn, fp) = fuse_frame_zone_pose(
            per_camera, adjacent_pairs, DIST_THR_FUSION_M, USE_POSE_CONSTRAINT)
        tot_tp += tp; tot_fn += fn; tot_fp += fp

        track_outputs = tracker.update(fused_points)
        hyp_ids = [tid for tid, _xy in track_outputs]
        hyp_points = [xy for _tid, xy in track_outputs]

        if gt_points and hyp_points:
            dist = mm.distances.norm2squared_matrix(
                np.array(gt_points), np.array(hyp_points), max_d2=(2.0 * DIST_THR_TRACK_M) ** 2)
        else:
            dist = np.empty((len(gt_points), len(hyp_points)))
        acc.update(gt_ids, hyp_ids, dist)
        frames_data.append({"gt_names": gt_names, "gt_points": gt_points,
                            "hyp_ids": hyp_ids, "hyp_points": hyp_points})
        n_processed += 1
        print(f"  {frame_name(frame_idx)}: GT個体={len(gt_points)} GTbbox={n_raw}(範囲内{n_kept}) "
              f"融合点={len(fused_points)} track出力={len(track_outputs)} "
              f"統合ペア TP={tp} FN(統合漏れ)={fn} FP(誤統合)={fp}")

    print(f"\n処理フレーム数: {n_processed}")
    p = tot_tp / (tot_tp + tot_fp) if tot_tp + tot_fp else float("nan")
    r = tot_tp / (tot_tp + tot_fn) if tot_tp + tot_fn else float("nan")
    print(f"統合ペア全体(牛名で判定): TP={tot_tp} FN={tot_fn} FP={tot_fp}  precision={p:.3f} recall={r:.3f}")

    tracker_suffix = "reid" if USE_REID_TRACKER else "noreid"
    pose_suffix = "posecon" if USE_POSE_CONSTRAINT else "noposecon"
    suffix = f"gtinput_{tracker_suffix}_{pose_suffix}"

    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=["idf1", "idp", "idr", "mota", "motp", "num_switches",
                 "num_objects", "num_matches", "num_false_positives", "num_misses"],
        name=f"multicam_bev_gt_zonepose_curved_{suffix}",
    )
    print("\n" + mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))

    dist_cap_m = 2.0 * DIST_THR_TRACK_M
    hota_res = compute_hota_trackeval(frames_data, dist_cap_m=dist_cap_m)
    print(f"\n=== HOTA(本家TrackEval, 距離上限 {dist_cap_m}m) ===")
    print(f"HOTA(平均)  = {np.mean(hota_res['HOTA']):.3f}")
    print(f"DetA(平均)  = {np.mean(hota_res['DetA']):.3f}")
    print(f"AssA(平均)  = {np.mean(hota_res['AssA']):.3f}")
    print(f"LocA(平均)  = {np.mean(hota_res['LocA']):.3f}")

    csv_path = OUTPUT_DIR / f"phase3_bev_multicam_gt_zonepose_curved_{suffix}_summary.csv"
    summary.to_csv(csv_path)
    print(f"\nCSV保存: {csv_path}")


if __name__ == "__main__":
    main()