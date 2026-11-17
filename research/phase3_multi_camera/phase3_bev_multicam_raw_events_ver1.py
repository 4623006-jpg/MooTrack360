"""
phase3_bev_multicam_raw_events_ver1.py

phase3_bev_multicam_yolo_check_zonepose.py(直線境界版・ReIDなしSimpleBevTracker)
と全く同じ検出・融合・追跡パイプラインを実行し、motmetrics.MOTAccumulatorが
内部で記録している「生の」フレームごとの対応付けイベント(mot_events: Type/OId/
HId/D)を、指定したフレーム範囲(DUMP_FRAME_START〜DUMP_FRAME_END)だけ
そのまま標準出力にダンプする。

目的: 「数フレーム検知できず、新IDが同じ場所に現れているのにIDs(SWITCH)が
検知されない」という疑いについて、集計済みの件数(num_switches等)だけでは
原因を特定できないため、実際にそのフレーム帯でGTがどう対応付けられている
(MATCH/MISS/FP/SWITCH、そのときの距離D)かを直接確認する。

出力:
  - frame範囲内の各フレームについて、GT個体名・位置、track出力(ID・位置)を
    そのまま表示(距離を自分で目算できるように)
  - 同じ範囲のmot_eventsの生テーブル(Type/OId/HId/D、実フレーム番号付き)

このファイルはcurved_zone_utils_ver1.pyには依存していない(直線境界版のため)。
Mootrack_core.py/ultralytics/motmetrics/torch/scipy/cv2が必要。

バージョン管理方針: この先さらに変更する場合は、このファイルを直接書き換えず
_ver2.py を新しく作ること。

--- 使い方 ---
    python phase3_bev_multicam_raw_events_ver1.py
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
from scipy.optimize import linear_sum_assignment

MOOTRACK_CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calibration")
if MOOTRACK_CODE_DIR not in sys.path:
    sys.path.insert(0, MOOTRACK_CODE_DIR)

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
)

# ---------------------------------------------------------------------------
# 設定(ユーザーが最後に貼った直線境界版・ReIDなし版と同じ値を使う)
# ---------------------------------------------------------------------------
XML_DIR = "/mnt/vmlqnap02/dataset/MooTrack360/1-hour-farm1-area1-sample-sequence/xml-1fps"
IMAGE_DIR = "/mnt/vmlqnap02/dataset/MooTrack360/1-hour-farm1-area1-sample-sequence/1-hour-farm1-area1-frames"
FLOORPLAN_JSON_PATH = "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/calibration/Farm2_Area_1_Artboard_config_.json"
OUTPUT_DIR = Path(
    "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/research/idf1_check"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

YOLO_WEIGHTS_PATH = "/mnt/vmlqnap02/dataset/MooTrack360/Trained models/yolo11x/train_ultralytics/weights/best.pt"
YOLO_CONF_THRESH = 0.33

FRAME_START = 0
FRAME_END = 1470
FRAME_STEP = 10

# この範囲(実フレーム番号)のGT-track対応付けを生でダンプする
DUMP_FRAME_START = 700
DUMP_FRAME_END = 900

IMAGE_WIDTH_PX = 8832

LYING_HEIGHT_M = 0.75
STANDING_HEIGHT_M = 0.75
DEFAULT_HEIGHT_M = 0.8

DIST_THR_FUSION_M = 1.5
OVERLAP_HALF_WIDTH_M = 1.5

DIST_THR_TRACK_M = 1.5
MAX_MISSED_FRAMES = 2

USE_POSE_CONSTRAINT = True
DEDUP_IOU_THRESHOLD = 0.5

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
YOLO_CLASS_TO_POSE = {0: "lying", 1: "standing"}


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


def yolo_world_points_per_camera_with_pose(
    yolo_model, image_path: Path, poses: Dict[str, "Pose"],
) -> Dict[str, List[dict]]:
    full_img = cv2.imread(str(image_path))
    per_camera: Dict[str, List[dict]] = {cid: [] for cid in CAM_IDS}
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
        pts_px, poses_list, bboxes = [], [], []
        for box in results.boxes:
            bx1, by1, bx2, by2 = box.xyxy[0].tolist()
            pts_px.append([(bx1 + bx2) / 2.0, (by1 + by2) / 2.0])
            poses_list.append(YOLO_CLASS_TO_POSE.get(int(box.cls[0])))
            bboxes.append((bx1, by1, bx2, by2))
        pts_px = np.array(pts_px, dtype=np.float64)
        heights = np.array([height_for(p) for p in poses_list], dtype=np.float64)
        floor_z = -heights

        cam_pose = poses[cam_id]
        world_xy, valid = unproject_to_floor(
            pts_px, cam_pose.R, cam_pose.t, fx, fy, cx, cy, xi, alpha,
            floor_z=floor_z,
        )
        for j in range(len(pts_px)):
            if valid[j]:
                per_camera[cam_id].append({
                    "name": None, "pose": poses_list[j],
                    "bbox_local": bboxes[j], "world_xy": world_xy[j],
                })

    return per_camera


def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _bbox_area(bbox) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def deduplicate_by_iou(items: List[dict], iou_thr: float) -> List[dict]:
    order = sorted(range(len(items)), key=lambda i: _bbox_area(items[i]["bbox_local"]), reverse=True)
    kept: List[dict] = []
    for i in order:
        it = items[i]
        is_dup = any(iou_xyxy(it["bbox_local"], k["bbox_local"]) >= iou_thr for k in kept)
        if not is_dup:
            kept.append(it)
    return kept


def deduplicate_per_camera(per_camera: Dict[str, List[dict]], iou_thr: float) -> Dict[str, List[dict]]:
    return {cam_id: deduplicate_by_iou(items, iou_thr) for cam_id, items in per_camera.items()}


def get_adjacent_zones(poses, cam_ids: List[str], half_width: float) -> Dict[Tuple[str, str], Tuple[float, float]]:
    zones = {}
    for cam_i, cam_j in zip(cam_ids[:-1], cam_ids[1:]):
        x_i = poses[cam_i].t[0]
        x_j = poses[cam_j].t[0]
        boundary = (x_i + x_j) / 2.0
        zones[(cam_i, cam_j)] = (boundary - half_width, boundary + half_width)
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


def filter_out_of_range_detections(per_camera, valid_ranges) -> Dict[str, List[dict]]:
    filtered: Dict[str, List[dict]] = {cid: [] for cid in per_camera}
    for cam_id, items in per_camera.items():
        lo, hi = valid_ranges[cam_id]
        for it in items:
            if lo <= it["world_xy"][0] <= hi:
                filtered[cam_id].append(it)
    return filtered


def match_zone_pose_constrained(
    per_camera, cam_ids, zones, dist_thr, use_pose_constraint=True,
) -> Tuple[List[Tuple[Tuple[str, int], Tuple[str, int]]], int]:
    accepted_pairs: List[Tuple[Tuple[str, int], Tuple[str, int]]] = []
    n_pose_rejected = 0
    for cam_i, cam_j in zones.keys():
        items_i = per_camera[cam_i]
        items_j = per_camera[cam_j]
        if not items_i or not items_j:
            continue
        n_i, n_j = len(items_i), len(items_j)
        cost = np.full((n_i, n_j), 1e6)
        for a, it_i in enumerate(items_i):
            for b, it_j in enumerate(items_j):
                d = float(np.linalg.norm(it_i["world_xy"] - it_j["world_xy"]))
                pose_mismatch = (
                    it_i["pose"] is not None and it_j["pose"] is not None
                    and it_i["pose"] != it_j["pose"]
                )
                if use_pose_constraint and pose_mismatch:
                    if d <= dist_thr:
                        n_pose_rejected += 1
                    continue
                if d <= dist_thr:
                    cost[a, b] = d
        row_idx, col_idx = linear_sum_assignment(cost)
        for a, b in zip(row_idx, col_idx):
            if cost[a, b] <= dist_thr:
                accepted_pairs.append(((cam_i, a), (cam_j, b)))
    return accepted_pairs, n_pose_rejected


def fuse_frame_zone_pose(
    per_camera_filtered, cam_ids, zones, dist_thr, use_pose_constraint=True,
) -> Tuple[List[dict], int]:
    accepted_pairs, n_pose_rejected = match_zone_pose_constrained(
        per_camera_filtered, cam_ids, zones, dist_thr, use_pose_constraint
    )
    all_keys = [(cam_id, idx) for cam_id, items in per_camera_filtered.items() for idx in range(len(items))]
    if not all_keys:
        return [], n_pose_rejected
    roots = union_find_clusters(all_keys, accepted_pairs)
    clusters: Dict[Tuple[str, int], List[np.ndarray]] = defaultdict(list)
    for key in all_keys:
        cam_id, idx = key
        clusters[roots[key]].append(per_camera_filtered[cam_id][idx]["world_xy"])
    fused = []
    for members in clusters.values():
        xy_mean = np.mean(np.array(members), axis=0)
        fused.append({"world_xy": xy_mean, "n_cams": len(members)})
    return fused, n_pose_rejected


class SimpleBevTracker:
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

        self.tracks = {tid: t for tid, t in self.tracks.items() if t["missed"] <= self.max_missed}

        for j, fp in enumerate(fused_points):
            if j not in matched_det:
                self.tracks[self.next_id] = {"xy": fp["world_xy"], "missed": 0}
                self.next_id += 1

        return [(tid, t["xy"]) for tid, t in self.tracks.items() if t["missed"] == 0]


def main():
    import motmetrics as mm
    from ultralytics import YOLO

    print("floorplan.json 読み込み中...")
    cameras_json = load_floorplan_cameras(FLOORPLAN_JSON_PATH)
    real_w, real_h = load_floorplan_dimensions(FLOORPLAN_JSON_PATH)
    cameras_json = apply_camera_overrides(cameras_json)
    poses = resolve_poses(cameras_json, root="Area1", fmt=ROT_FMT)

    zones = get_adjacent_zones(poses, CAM_IDS, OVERLAP_HALF_WIDTH_M)
    valid_ranges = get_camera_valid_ranges(CAM_IDS, zones)

    print(f"YOLOモデル読み込み中: {YOLO_WEIGHTS_PATH}")
    yolo_model = YOLO(YOLO_WEIGHTS_PATH)

    xml_dir = Path(XML_DIR)
    image_dir = Path(IMAGE_DIR)
    frame_indices = list(range(FRAME_START, FRAME_END + 1, FRAME_STEP))

    tracker = SimpleBevTracker(dist_thr=DIST_THR_TRACK_M, max_missed=MAX_MISSED_FRAMES)
    acc = mm.MOTAccumulator(auto_id=True)

    # acc.update()を呼んだ順(sequential FrameId)と、実フレーム番号・
    # そのフレームのGT/track出力の生データを両方残しておく
    frame_order: List[dict] = []

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

        per_camera_raw = yolo_world_points_per_camera_with_pose(yolo_model, image_path, poses)
        per_camera_deduped = deduplicate_per_camera(per_camera_raw, DEDUP_IOU_THRESHOLD)
        per_camera_filtered = filter_out_of_range_detections(per_camera_deduped, valid_ranges)
        per_camera_filtered = filter_outside_floorplan_bounds(
            per_camera_filtered, real_w, real_h, margin_m=0.0, verbose=False
        )

        fused_points, _n_pose_rejected = fuse_frame_zone_pose(
            per_camera_filtered, CAM_IDS, zones, DIST_THR_FUSION_M, USE_POSE_CONSTRAINT
        )

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
        frame_order.append({
            "frame_idx": frame_idx,
            "gt_names": gt_names, "gt_ids": gt_ids, "gt_points": gt_points,
            "hyp_ids": hyp_ids, "hyp_points": hyp_points,
        })

        if len(frame_order) % 50 == 0:
            print(f"  {len(frame_order)}/{len(frame_indices)} フレーム処理済み(直近: {frame_name(frame_idx)})")

    print(f"\n処理フレーム数: {len(frame_order)}/{len(frame_indices)}")

    # --- 指定範囲の生データ(GT/track出力そのもの)をダンプ ---
    print(f"\n=== frame {DUMP_FRAME_START}〜{DUMP_FRAME_END} の生データ(GT個体・位置、track出力) ===")
    dump_seq_ids = []
    for seq_id, info in enumerate(frame_order):
        if DUMP_FRAME_START <= info["frame_idx"] <= DUMP_FRAME_END:
            dump_seq_ids.append(seq_id)
            gt_str = ", ".join(
                f"{name}(oid={oid}) xy={pt.round(2)}"
                for name, oid, pt in zip(info["gt_names"], info["gt_ids"], info["gt_points"])
            )
            hyp_str = ", ".join(
                f"ID{tid} xy={pt.round(2)}"
                for tid, pt in zip(info["hyp_ids"], info["hyp_points"])
            )
            print(f"  {frame_name(info['frame_idx'])}: GT=[{gt_str}]")
            print(f"            track=[{hyp_str}]")

    if not dump_seq_ids:
        print(f"  (frame {DUMP_FRAME_START}〜{DUMP_FRAME_END}の範囲で、GT・画像が両方存在して"
              f"処理されたフレームが1つも無かった。範囲やFRAME_STEPとの噛み合わせを確認すること)")
        return

    # --- 同じ範囲のmot_eventsの生テーブル(Type/OId/HId/D)をダンプ ---
    print(f"\n=== frame {DUMP_FRAME_START}〜{DUMP_FRAME_END} のmot_events(生の対応付け結果) ===")
    events = acc.mot_events
    dump_seq_set = set(dump_seq_ids)
    seq_to_frame_idx = {seq_id: frame_order[seq_id]["frame_idx"] for seq_id in dump_seq_ids}

    # FrameIdがdump_seq_setに含まれる行だけ抜き出し、実フレーム番号列を追加して表示
    sub = events[events.index.get_level_values("FrameId").isin(dump_seq_set)].copy()
    sub["frame_name"] = [
        frame_name(seq_to_frame_idx[fid]) for fid in sub.index.get_level_values("FrameId")
    ]
    print(sub[["frame_name", "Type", "OId", "HId", "D"]].to_string())

    print(f"\n見方:")
    print(f"  Type=MATCH  : 正しく対応付けられた(OId=GTのoid, HId=対応したtrack ID)")
    print(f"  Type=MISS   : そのGT(OId)がどのtrackにも対応付けられなかった(距離gate={2.0*DIST_THR_TRACK_M:.1f}m以内に"
          f"候補が無かった)")
    print(f"  Type=FP     : そのtrack(HId)がどのGTにも対応付けられなかった")
    print(f"  Type=SWITCH : そのGT(OId)が、前回対応付けられていたhidとは別のHIdに対応付けられた"
          f"(これがいわゆるIDs)")
    print(f"  Type=ASCEND : 新しく現れたHIdが、既存のOIdに初めて対応付けられた(SWITCHと同時に出ることが多い)")
    print(f"  D列はGT-track間の距離の2乗(norm2squared)なので、実際の距離[m]は sqrt(D) で見ること"
          f"(gate={2.0*DIST_THR_TRACK_M:.1f}m = D<= {(2.0*DIST_THR_TRACK_M)**2:.1f})")


if __name__ == "__main__":
    main()