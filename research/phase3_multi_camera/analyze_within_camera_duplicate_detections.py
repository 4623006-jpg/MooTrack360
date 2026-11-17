"""
analyze_within_camera_duplicate_detections.py

phase3_bev_multicam_yolo_check_zonepose.py で観測されたFP=158件(GT個体数59〜61
に対し融合点が62〜68と一貫して超過)の原因調査。

姿勢クラス制約の有効/無効アブレーションでは結果が完全に一致し、「隣接カメラ間で
姿勢不一致が原因で統合が壊れているケース」は今回のデータでは無かったことが分かった。
残る本命の仮説は「同一カメラ内の重複検出」:

  YOLOモデルはcow/lying(クラス0)とcow/standing(クラス1)を別クラスとして
  扱っており、NMSは通常クラスごとに独立して行われる。conf閾値0.1という低めの
  設定で、1頭の牛がlying・standing両方として弱く検出されると、bboxがほぼ
  重なっていてもクラスが違うためNMSでは消えず、同一カメラ内に2つの検出が
  残ってしまう可能性がある。

  match_zone_pose_constrained()は隣接カメラ「間」のペアしか見ておらず、
  同一カメラ「内」の重複はそもそも統合の対象にすらならないため、この重複は
  そのままFPとして最終出力に残る。

このスクリプトは、フィルタ後(有効範囲外除外後)の各カメラの検出について、
同一カメラ内で異なるクラス(lying vs standing)かつbbox IoUが高いペアを数え上げ、
その件数が「GT個体数に対する融合点の超過数」とどれだけ対応するかを確認する。
(sanity check用に、同一クラス同士の高IoUペアも数える。NMSが正しく効いていれば
 これはほぼ0件のはず)

--- 使い方 ---
    python analyze_within_camera_duplicate_detections.py
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

from Mootrack_core import (
    NUM_CAMERAS, CAM_IDS, ROT_FMT,
    Pose, get_intrinsics,
    load_floorplan_cameras, apply_camera_overrides, resolve_poses,
    unproject_to_floor,
    union_find_clusters,
)

# ---------------------------------------------------------------------------
# 設定(phase3_bev_multicam_yolo_check_zonepose.pyと共通の値はそのまま揃える)
# ---------------------------------------------------------------------------
XML_DIR = "/mnt/vmlqnap02/dataset/MooTrack360/1-hour-farm1-area1-sample-sequence/xml-1fps"
IMAGE_DIR = "/mnt/vmlqnap02/dataset/MooTrack360/1-hour-farm1-area1-sample-sequence/1-hour-farm1-area1-frames"
# 注意: このFLOORPLAN_JSON_PATHは以前から要検証(値が古い可能性あり)と指摘済みのファイル
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

LYING_HEIGHT_M = 0.5
STANDING_HEIGHT_M = 0.75
DEFAULT_HEIGHT_M = 0.8

DIST_THR_FUSION_M = 1.4
OVERLAP_HALF_WIDTH_M = 1.5

IOU_THRESHOLDS = [0.1, 0.3, 0.5]  # 重複疑いと判定するbbox IoUの閾値(複数試す)

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


def iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
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


# ---------------------------------------------------------------------------
# YOLO検出(bbox・姿勢クラス・confidenceを保持したままワールド座標へ)
# ---------------------------------------------------------------------------

def yolo_detections_per_camera(
    yolo_model, image_path: Path, poses: Dict[str, "Pose"],
) -> Dict[str, List[dict]]:
    """cam_id -> [{"name": None, "pose": ..., "conf": ..., "bbox_local": (x1,y1,x2,y2),
                   "world_xy": ndarray}, ...]"""
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
        pts_px, poses_list, confs, bboxes = [], [], [], []
        for box in results.boxes:
            bx1, by1, bx2, by2 = box.xyxy[0].tolist()
            pts_px.append([(bx1 + bx2) / 2.0, (by1 + by2) / 2.0])
            poses_list.append(YOLO_CLASS_TO_POSE.get(int(box.cls[0])))
            confs.append(float(box.conf[0]))
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
                    "name": None, "pose": poses_list[j], "conf": confs[j],
                    "bbox_local": bboxes[j], "world_xy": world_xy[j],
                })

    return per_camera


# ---------------------------------------------------------------------------
# 有効範囲・隣接カメラ制約付き融合(phase3_bev_multicam_yolo_check_zonepose.pyと同じ)
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
    accepted_pairs = match_zone_pose_constrained(per_camera_filtered, cam_ids, zones, dist_thr)
    all_keys = [(cam_id, idx) for cam_id, items in per_camera_filtered.items() for idx in range(len(items))]
    if not all_keys:
        return []
    roots = union_find_clusters(all_keys, accepted_pairs)
    clusters: Dict[Tuple[str, int], List[np.ndarray]] = defaultdict(list)
    for key in all_keys:
        cam_id, idx = key
        clusters[roots[key]].append(per_camera_filtered[cam_id][idx]["world_xy"])
    fused = []
    for members in clusters.values():
        xy_mean = np.mean(np.array(members), axis=0)
        fused.append({"world_xy": xy_mean, "n_cams": len(members)})
    return fused


# ---------------------------------------------------------------------------
# 同一カメラ内の重複検出チェック(本題)
# ---------------------------------------------------------------------------

def find_within_camera_duplicates(
    per_camera_filtered: Dict[str, List[dict]], iou_thresholds: List[float],
) -> dict:
    """同一カメラ内の全ペアについてbbox IoUを計算し、
    (a) 異クラス(lying/standing不一致)かつ高IoU: 重複検出の疑いが強い
    (b) 同クラスかつ高IoU: NMSが正しく効いていればほぼ0件のはずのsanity check
    をそれぞれ閾値ごとに数える。"""
    cross_counts = {t: 0 for t in iou_thresholds}
    same_counts = {t: 0 for t in iou_thresholds}
    detail_pairs = []  # (cam_id, iou, pose_a, pose_b, conf_a, conf_b)

    for cam_id, items in per_camera_filtered.items():
        n = len(items)
        for i in range(n):
            for j in range(i + 1, n):
                iou = iou_xyxy(items[i]["bbox_local"], items[j]["bbox_local"])
                if iou <= 0:
                    continue
                same_class = items[i]["pose"] == items[j]["pose"]
                for t in iou_thresholds:
                    if iou >= t:
                        if same_class:
                            same_counts[t] += 1
                        else:
                            cross_counts[t] += 1
                if iou >= min(iou_thresholds) and not same_class:
                    detail_pairs.append({
                        "cam_id": cam_id, "iou": iou,
                        "pose_a": items[i]["pose"], "pose_b": items[j]["pose"],
                        "conf_a": items[i]["conf"], "conf_b": items[j]["conf"],
                    })

    return {"cross_class": cross_counts, "same_class": same_counts, "detail_pairs": detail_pairs}


def main():
    from ultralytics import YOLO

    print("floorplan.json 読み込み中...")
    cameras_json = load_floorplan_cameras(FLOORPLAN_JSON_PATH)
    cameras_json = apply_camera_overrides(cameras_json)
    poses = resolve_poses(cameras_json, root="Area1", fmt=ROT_FMT)

    print("\n隣接カメラの重なり範囲(±{:.1f}m)を計算中...".format(OVERLAP_HALF_WIDTH_M))
    zones = get_adjacent_zones(poses, CAM_IDS, OVERLAP_HALF_WIDTH_M)
    valid_ranges = get_camera_valid_ranges(CAM_IDS, zones)

    print(f"\nYOLOモデル読み込み中: {YOLO_WEIGHTS_PATH}")
    yolo_model = YOLO(YOLO_WEIGHTS_PATH)

    xml_dir = Path(XML_DIR)
    image_dir = Path(IMAGE_DIR)
    frame_indices = list(range(FRAME_START, FRAME_END + 1, FRAME_STEP))

    all_detail_rows = []  # CSV出力用: 全フレーム分の疑わしいペアを集約
    total_excess = 0
    total_cross_dup = {t: 0 for t in IOU_THRESHOLDS}
    n_processed = 0

    print(f"\n{'frame':>8s} {'GT':>4s} {'融合点':>6s} {'超過':>4s} "
          + " ".join(f"異クラスIoU>={t}" for t in IOU_THRESHOLDS)
          + f" {'同クラス高IoU(sanity)':>10s}")

    for frame_idx in frame_indices:
        xml_path = xml_dir / f"{frame_name(frame_idx)}.xml"
        image_path = image_dir / f"{frame_name(frame_idx)}.jpg"
        if not xml_path.exists() or not image_path.exists():
            continue

        gt_objects = parse_voc_annotation(xml_path)
        gt_world = gt_world_points_this_frame(gt_objects, poses, IMAGE_WIDTH_PX)
        n_gt = len(gt_world)

        per_camera_raw = yolo_detections_per_camera(yolo_model, image_path, poses)
        per_camera_filtered = filter_out_of_range_detections(per_camera_raw, valid_ranges)
        fused_points = fuse_frame_zone_pose(per_camera_filtered, CAM_IDS, zones, DIST_THR_FUSION_M)
        n_fused = len(fused_points)
        excess = n_fused - n_gt

        dup_info = find_within_camera_duplicates(per_camera_filtered, IOU_THRESHOLDS)
        for row in dup_info["detail_pairs"]:
            row["frame"] = frame_name(frame_idx)
            all_detail_rows.append(row)

        total_excess += excess
        for t in IOU_THRESHOLDS:
            total_cross_dup[t] += dup_info["cross_class"][t]

        cross_str = " ".join(f"{dup_info['cross_class'][t]:>13d}" for t in IOU_THRESHOLDS)
        same_high = dup_info["same_class"][max(IOU_THRESHOLDS)]
        print(f"{frame_name(frame_idx):>8s} {n_gt:>4d} {n_fused:>6d} {excess:>4d} {cross_str} {same_high:>10d}")
        n_processed += 1

    print(f"\n処理フレーム数: {n_processed}/{len(frame_indices)}")
    print(f"\n=== 集計(全{n_processed}フレーム合計) ===")
    print(f"  GT個体数に対する融合点の超過合計: {total_excess}")
    for t in IOU_THRESHOLDS:
        print(f"  異クラス(lying/standing不一致)かつIoU>={t}の重複疑いペア合計: {total_cross_dup[t]}")
    print(f"  (超過合計と重複疑いペア合計が近ければ、同一カメラ内の重複検出が主原因という仮説を支持)")

    if all_detail_rows:
        csv_path = OUTPUT_DIR / "within_camera_duplicate_detections.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            f.write("frame,cam_id,iou,pose_a,pose_b,conf_a,conf_b\n")
            for row in all_detail_rows:
                f.write(f"{row['frame']},{row['cam_id']},{row['iou']:.3f},"
                        f"{row['pose_a']},{row['pose_b']},{row['conf_a']:.3f},{row['conf_b']:.3f}\n")
        print(f"\n疑わしいペアの詳細CSV保存: {csv_path}")
        print(f"  (件数: {len(all_detail_rows)}件、IoU>={min(IOU_THRESHOLDS)}で異クラスのペアを全て記録)")
    else:
        print("\n異クラスの重複疑いペアは1件も見つかりませんでした"
              "(この仮説では超過を説明できないので、別の原因を探す必要があります)")


if __name__ == "__main__":
    main()