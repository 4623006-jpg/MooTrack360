"""
phase3_id_switch_diagnosis.py

【vml36で実行】phase3_multi_camera/ フォルダに置いて実行する

phase3_bev_multicam_yolo_check.py でIDs=167(30フレームでの追跡ID切り替わり)と
非常に多かった原因を、個体別に追跡して切り分けるための診断スクリプト。
phase2でやったphase2_confidence_timeseries.py / phase2_fp_breakdown.pyと同じ考え方。

各GT個体(name)について、真の位置(GTから逆投影した世界座標)の近く(半径
NEARBY_RADIUS_M以内)にあるtrack出力を毎フレーム探し、
  - 何個のtrack出力(融合点)が近くにあったか(2個以上なら重複ID発生中)
  - 割り当てられていたtrack_id
  - 前フレームの位置からの移動距離(追跡の距離閾値1.0mと比較)
を記録する。track_idが変化した回数(ID切り替わり)が多い個体を上位表示し、
それぞれの切り替わりが「重複融合(2点同時存在)」によるものか、「距離閾値超過による
再割当」によるものかを切り分ける。

--- 使い方 ---
    python phase3_id_switch_diagnosis.py
"""

import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
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
# 設定(phase3_bev_multicam_yolo_check.pyと共通)
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
DEFAULT_HEIGHT_M = 0.8

DIST_THR_FUSION_M = 1.4
DIST_THR_TRACK_M = 1.0
MAX_MISSED_FRAMES = 2

NEARBY_RADIUS_M = 1.5  # GTの真の位置からこの範囲内のtrack出力を「この個体のもの」とみなす

YOLO_CLASS_TO_POSE = {0: "lying", 1: "standing"}

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
            "name": name, "pose": pose,
            "xmin": float(bnd.find("xmin").text), "ymin": float(bnd.find("ymin").text),
            "xmax": float(bnd.find("xmax").text), "ymax": float(bnd.find("ymax").text),
        })
    return objects


def height_for(pose: Optional[str]) -> float:
    if pose == "lying":
        return LYING_HEIGHT_M
    if pose == "standing":
        return STANDING_HEIGHT_M
    return DEFAULT_HEIGHT_M


def gt_world_points_this_frame(objects: List[dict], poses: Dict[str, "Pose"], image_width: float) -> Dict[str, List[np.ndarray]]:
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


def yolo_world_points_per_camera(yolo_model, image_path: Path, poses: Dict[str, "Pose"]):
    full_img = cv2.imread(str(image_path))
    per_camera: Dict[str, List[Tuple[None, np.ndarray]]] = {cid: [] for cid in CAM_IDS}
    if full_img is None:
        return per_camera
    tile_w = IMAGE_WIDTH_PX / NUM_CAMERAS
    for cam_idx, cam_id in enumerate(CAM_IDS):
        x0 = int(cam_idx * tile_w)
        x1_edge = int((cam_idx + 1) * tile_w)
        frame_img = full_img[:, x0:x1_edge]
        results = yolo_model.predict(frame_img, conf=YOLO_CONF_THRESH, device=DEVICE, verbose=False)[0]
        if len(results.boxes) == 0:
            continue
        fx, fy, cx, cy, xi, alpha = get_intrinsics(cam_id)
        pts_px, heights = [], []
        for box in results.boxes:
            bx1, by1, bx2, by2 = box.xyxy[0].tolist()
            pts_px.append([(bx1 + bx2) / 2.0, (by1 + by2) / 2.0])
            heights.append(height_for(YOLO_CLASS_TO_POSE.get(int(box.cls[0]))))
        pts_px = np.array(pts_px, dtype=np.float64)
        floor_z = -np.array(heights, dtype=np.float64)
        cam_pose = poses[cam_id]
        world_xy, valid = unproject_to_floor(pts_px, cam_pose.R, cam_pose.t, fx, fy, cx, cy, xi, alpha, floor_z=floor_z)
        for j in range(len(pts_px)):
            if valid[j]:
                per_camera[cam_id].append((None, world_xy[j]))
    return per_camera


def fuse_frame(per_camera, dist_thr: float) -> List[dict]:
    assignments, _ = compute_camera_assignments_by_distance(per_camera)
    accepted_pairs = [((a.cam_i, a.idx_i), (a.cam_j, a.idx_j)) for a in assignments if a.dist <= dist_thr]
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


def main():
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

    # name -> [(frame_idx, n_nearby, track_id_or_None, dist_from_prev), ...]
    history: Dict[str, List[dict]] = defaultdict(list)
    prev_assigned_xy: Dict[str, np.ndarray] = {}

    for frame_idx in frame_indices:
        xml_path = xml_dir / f"{frame_name(frame_idx)}.xml"
        image_path = image_dir / f"{frame_name(frame_idx)}.jpg"
        if not xml_path.exists() or not image_path.exists():
            continue

        gt_objects = parse_voc_annotation(xml_path)
        gt_world = gt_world_points_this_frame(gt_objects, poses, IMAGE_WIDTH_PX)

        per_camera = yolo_world_points_per_camera(yolo_model, image_path, poses)
        fused_points = fuse_frame(per_camera, DIST_THR_FUSION_M)
        track_outputs = tracker.update(fused_points)

        for name, xys in gt_world.items():
            gt_xy = np.mean(np.array(xys), axis=0)

            nearby = [
                (tid, xy) for tid, xy in track_outputs
                if np.linalg.norm(xy - gt_xy) <= NEARBY_RADIUS_M
            ]
            n_nearby = len(nearby)

            if n_nearby == 0:
                assigned_tid, assigned_xy = None, None
            else:
                # 一番近いものをこの個体の代表とみなす
                nearby.sort(key=lambda t: np.linalg.norm(t[1] - gt_xy))
                assigned_tid, assigned_xy = nearby[0]

            dist_from_prev = None
            if assigned_xy is not None and name in prev_assigned_xy:
                dist_from_prev = float(np.linalg.norm(assigned_xy - prev_assigned_xy[name]))
            if assigned_xy is not None:
                prev_assigned_xy[name] = assigned_xy

            history[name].append({
                "frame": frame_idx, "n_nearby": n_nearby,
                "track_id": assigned_tid, "dist_from_prev": dist_from_prev,
            })

        n_yolo_total = sum(len(v) for v in per_camera.values())
        print(f"  {frame_name(frame_idx)}: GT個体={len(gt_world)} YOLO検出={n_yolo_total} "
              f"融合点={len(fused_points)} track出力={len(track_outputs)}")

    # --- 個体別のID切り替わり回数を集計 ---
    switch_counts = {}
    duplicate_frame_counts = {}
    for name, records in history.items():
        prev_tid = None
        n_switch = 0
        n_dup = 0
        for r in records:
            if r["n_nearby"] >= 2:
                n_dup += 1
            if r["track_id"] is not None:
                if prev_tid is not None and r["track_id"] != prev_tid:
                    n_switch += 1
                prev_tid = r["track_id"]
        switch_counts[name] = n_switch
        duplicate_frame_counts[name] = n_dup

    ranked = sorted(switch_counts.items(), key=lambda kv: -kv[1])

    print(f"\n=== 個体別 ID切り替わり回数(上位20) ===")
    print(f"{'name':<10}{'ID切替回数':>12}{'重複融合フレーム数':>20}")
    for name, n_switch in ranked[:20]:
        print(f"{name:<10}{n_switch:>12}{duplicate_frame_counts[name]:>20}")

    total_switch = sum(switch_counts.values())
    total_dup_frames = sum(duplicate_frame_counts.values())
    print(f"\n全体のID切替回数合計(この集計方式): {total_switch}")
    print(f"重複融合(同じ個体の近くに2点以上)が起きたフレーム件数合計: {total_dup_frames}")

    # --- 上位個体の詳細タイムラインを表示 ---
    print(f"\n=== 上位5個体の詳細タイムライン ===")
    for name, _ in ranked[:5]:
        print(f"\n--- {name} ---")
        print(f"{'frame':>8}{'n_nearby':>10}{'track_id':>10}{'前フレームからの距離[m]':>24}")
        for r in history[name]:
            dist_str = f"{r['dist_from_prev']:.2f}" if r["dist_from_prev"] is not None else "-"
            tid_str = str(r["track_id"]) if r["track_id"] is not None else "無し"
            flag = "  <-- 重複融合" if r["n_nearby"] >= 2 else ""
            print(f"{frame_name(r['frame']):>8}{r['n_nearby']:>10}{tid_str:>10}{dist_str:>24}{flag}")


if __name__ == "__main__":
    main()