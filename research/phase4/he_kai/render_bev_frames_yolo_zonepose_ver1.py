"""
render_bev_frames_yolo_zonepose_ver1.py

render_bev_frames_yolo_zonepose.py(直線境界版)との差分は1点のみ:
隣接カメラの有効範囲の区切りを、「ワールドx座標の中点 ± OVERLAP_HALF_WIDTH_M」
という直線(x軸に垂直な帯)から、curved_zone_utils_ver1.pyが計算する「許容できる
魚眼カメラの範囲(画像上の半径R_max)」に基づく曲線(カメラごとの多角形)に
置き換えた。

具体的には:
  - get_adjacent_zones / get_camera_valid_ranges / classify_exclusive_zone
    (直線境界版) を削除し、curved_zone_utils_ver1.pyの
    build_camera_polygons / filter_by_curved_zone / classify_curved_zone /
    get_adjacent_pairs に置き換えた。
  - match_zone_pose_constrained / fuse_frame_zone_pose は、「隣接カメラペアの
    リスト」さえあれば動くロジックだったため、zones辞書(境界値lo,hi付き)の
    代わりにget_adjacent_pairs()が返す単純なペアのリストを渡すよう変更した。
  - draw_frame()のオレンジの帯描画を、カメラごとの曲線境界(色分けした輪郭線)の
    描画に置き換えた。

他のロジック(YOLO推論、GT読み込み、姿勢別高さ、重複除去、floorplan範囲外
フィルタ、excess診断、動画化)はrender_bev_frames_yolo_zonepose.pyと同一。

バージョン管理方針: この先さらに変更する場合は、このファイルを直接書き換えず
_ver2.py を新しく作ること。

出力先: OUTPUT_DIR/bev_frames_yolo_zonepose_curved/{frame}.png

--- 使い方 ---
    python render_bev_frames_yolo_zonepose_ver1.py
"""

import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESEARCH_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
PHASE4_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
MOOTRACK_CODE_DIR = os.path.abspath(os.path.join(RESEARCH_DIR, "calibration"))
for d in (RESEARCH_DIR, PHASE4_DIR, MOOTRACK_CODE_DIR,
          os.path.join(RESEARCH_DIR, "analyze"), os.path.join(RESEARCH_DIR, "annotate"),
          os.path.join(RESEARCH_DIR, "phase3_multi_camera"), THIS_DIR):
    if os.path.isdir(d) and d not in sys.path:
        sys.path.insert(0, d)

from Mootrack_core import (
    NUM_CAMERAS, CAM_IDS, ROT_FMT,
    Pose, get_intrinsics,
    load_floorplan, apply_camera_overrides, resolve_poses,
    unproject_to_floor,
    union_find_clusters,
    filter_outside_floorplan_bounds,
)

from curved_zone_utils_ver1 import (
    ZONE_HEIGHT_M, N_ANGLES,
    build_camera_polygons, filter_by_curved_zone, classify_curved_zone,
    get_adjacent_pairs,
)

# ---------------------------------------------------------------------------
# 設定(phase3_bev_multicam_yolo_check_zonepose_ver1.pyと同じ値)
# ---------------------------------------------------------------------------
import xml.etree.ElementTree as ET

XML_DIR = "/mnt/vmlqnap02/dataset/MooTrack360/1-hour-farm1-area1-sample-sequence/xml-1fps"
IMAGE_DIR = "/mnt/vmlqnap02/dataset/MooTrack360/1-hour-farm1-area1-sample-sequence/1-hour-farm1-area1-frames"
# 注意: このFLOORPLAN_JSON_PATHは以前から要検証(値が古い可能性あり)と指摘済みのファイル
FLOORPLAN_JSON_PATH = "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/calibration/Farm2_Area_1_Artboard_config_.json"
OUTPUT_DIR = Path(
    "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/research/phase4/diagnose/check_yolo"
)
FRAMES_OUT_DIR = OUTPUT_DIR / "bev_frames_yolo_zonepose_curved"
FRAMES_OUT_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATED_OUT_DIR = OUTPUT_DIR / "annotated_frames_yolo_zonepose_curved"
ANNOTATED_OUT_DIR.mkdir(parents=True, exist_ok=True)

YOLO_WEIGHTS_PATH = "/mnt/vmlqnap02/dataset/MooTrack360/Trained models/yolo11x/train_ultralytics/weights/best.pt"
YOLO_CONF_THRESH = 0.33

FRAME_START = 600
FRAME_END = 600
FRAME_STEP = 10

IMAGE_WIDTH_PX = 8832

LYING_HEIGHT_M = 0.5   # phase3_bev_multicam_yolo_check_zonepose_ver1.pyと揃える(暫定値)
STANDING_HEIGHT_M = 0.75
DEFAULT_HEIGHT_M = 0.8

DIST_THR_FUSION_M = 1.5

USE_POSE_CONSTRAINT = False

# 同一カメラ内の重複検出除去: IoUがこの閾値以上のペアを同一個体とみなし、
# bboxが大きい方だけを残す。
DEDUP_IOU_THRESHOLD = 0.4

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

YOLO_CLASS_TO_POSE = {0: "lying", 1: "standing"}

MAKE_VIDEO = True     # Trueなら最後にPNG群を1本のmp4にまとめる
VIDEO_FPS = 2

TYPE_COLORS_BGR = {
    "wall": (0, 200, 0), "cow_beds": (0, 140, 255), "feeding_bay": (255, 140, 0),
    "water": (255, 255, 0), "gate": (0, 0, 255), "milking_station": (255, 0, 255),
    "hay_area": (0, 255, 255), "cow_brushes": (255, 0, 180),
}
CAM_ZONE_COLORS_BGR = {
    "Camera_1": (255, 120, 60),
    "Camera_2": (60, 200, 60),
    "Camera_3": (60, 140, 255),
}
OVERLAP_ZONE_COLOR_BGR = (220, 220, 220)
GT_MARKER_COLOR_BGR = (0, 255, 255)  # 黄色

POSE_BOX_COLORS_BGR = {"lying": (60, 200, 60), "standing": (60, 140, 255), None: (180, 180, 180)}
GT_BOX_COLOR_BGR = (255, 255, 0)  # シアン(BGR順)


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


def gt_world_points_per_camera(objects: List[dict], poses: Dict[str, "Pose"], image_width: float) -> Dict[str, List[dict]]:
    tile_w = image_width / NUM_CAMERAS
    per_camera: Dict[str, List[dict]] = {cid: [] for cid in CAM_IDS}
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
            per_camera[cam_id].append({"name": obj["name"], "pose": obj["pose"], "world_xy": world_xy[0]})
    return per_camera


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


def _bbox_area(bbox: Tuple[float, float, float, float]) -> float:
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


def voc_objects_local_by_camera(objects: List[dict], image_width: float) -> Dict[str, List[dict]]:
    tile_w = image_width / NUM_CAMERAS
    result: Dict[str, List[dict]] = {cid: [] for cid in CAM_IDS}
    for obj in objects:
        center_x_full = (obj["xmin"] + obj["xmax"]) / 2.0
        cam_idx = int(center_x_full // tile_w)
        cam_idx = max(0, min(cam_idx, NUM_CAMERAS - 1))
        cam_id = CAM_IDS[cam_idx]
        x0 = cam_idx * tile_w
        result[cam_id].append({
            "name": obj["name"], "pose": obj["pose"],
            "bbox_local": (obj["xmin"] - x0, obj["ymin"], obj["xmax"] - x0, obj["ymax"]),
        })
    return result


def save_annotated_camera_images(
    frame_idx: int, image_path: Path,
    per_camera_raw: Dict[str, List[dict]],
    per_camera_deduped: Dict[str, List[dict]],
    per_camera_filtered: Dict[str, List[dict]],
) -> List[Path]:
    full_img = cv2.imread(str(image_path))
    if full_img is None:
        return []
    tile_w = IMAGE_WIDTH_PX / NUM_CAMERAS
    saved: List[Path] = []

    for cam_idx, cam_id in enumerate(CAM_IDS):
        x0 = int(cam_idx * tile_w)
        x1 = int((cam_idx + 1) * tile_w)
        crop = full_img[:, x0:x1].copy()

        deduped_ids = {id(it) for it in per_camera_deduped.get(cam_id, [])}
        filtered_ids = {id(it) for it in per_camera_filtered.get(cam_id, [])}

        for it in per_camera_raw.get(cam_id, []):
            bx1, by1, bx2, by2 = [int(round(v)) for v in it["bbox_local"]]
            color = POSE_BOX_COLORS_BGR.get(it["pose"], (180, 180, 180))

            if id(it) not in deduped_ids:
                thickness, tag = 1, "(重複除去)"
            elif id(it) not in filtered_ids:
                thickness, tag = 1, "(範囲外除外)"
            else:
                thickness, tag = 2, ""

            cv2.rectangle(crop, (bx1, by1), (bx2, by2), color, thickness, lineType=cv2.LINE_AA)
            label = f"{it['pose']}{tag}"
            cv2.putText(crop, label, (bx1, max(0, by1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2, cv2.LINE_AA)

        out_path = ANNOTATED_OUT_DIR / f"{frame_name(frame_idx)}_{cam_id}.jpg"
        cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        saved.append(out_path)

    return saved


# ---------------------------------------------------------------------------
# 有効範囲(曲線)・隣接カメラ制約付き融合
# ---------------------------------------------------------------------------

def match_zone_pose_constrained(
    per_camera: Dict[str, List[dict]],
    adjacent_pairs: List[Tuple[str, str]],
    dist_thr: float,
    use_pose_constraint: bool = True,
) -> List[Tuple[Tuple[str, int], Tuple[str, int]]]:
    accepted_pairs: List[Tuple[Tuple[str, int], Tuple[str, int]]] = []
    for cam_i, cam_j in adjacent_pairs:
        items_i = per_camera[cam_i]
        items_j = per_camera[cam_j]
        if not items_i or not items_j:
            continue
        n_i, n_j = len(items_i), len(items_j)
        cost = np.full((n_i, n_j), 1e6)
        for a, it_i in enumerate(items_i):
            for b, it_j in enumerate(items_j):
                if (use_pose_constraint and it_i["pose"] is not None
                        and it_j["pose"] is not None and it_i["pose"] != it_j["pose"]):
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
    adjacent_pairs: List[Tuple[str, str]],
    dist_thr: float,
    use_pose_constraint: bool = True,
) -> List[dict]:
    accepted_pairs = match_zone_pose_constrained(per_camera_filtered, adjacent_pairs, dist_thr, use_pose_constraint)
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
# BEV描画
# ---------------------------------------------------------------------------

def draw_regions(canvas, regions, scale: float):
    for region in regions:
        pts = (region.polygon_m * scale).astype(np.int32).reshape(-1, 1, 2)
        color = TYPE_COLORS_BGR.get(region.type, (200, 200, 200))
        closed = region.type not in ("wall", "gate")
        cv2.polylines(canvas, [pts], isClosed=closed, color=color, thickness=2, lineType=cv2.LINE_AA)


def match_fused_to_gt_and_flag_extras(
    gt_raw: List[dict], fused_points: List[dict], dist_thr: float = 1.0,
) -> List[int]:
    gt_by_name: Dict[str, List[np.ndarray]] = defaultdict(list)
    for item in gt_raw:
        gt_by_name[item["name"]].append(item["world_xy"])
    gt_names = list(gt_by_name.keys())
    gt_points = [np.mean(np.array(gt_by_name[n]), axis=0) for n in gt_names]

    if not gt_points or not fused_points:
        return list(range(len(fused_points)))

    gt_arr = np.array(gt_points)
    fused_arr = np.array([fp["world_xy"] for fp in fused_points])
    cost = np.linalg.norm(fused_arr[:, None, :] - gt_arr[None, :, :], axis=-1)
    cost_capped = np.where(cost <= dist_thr, cost, 1e6)
    row_idx, col_idx = linear_sum_assignment(cost_capped)
    matched_fused = {r for r, c in zip(row_idx, col_idx) if cost_capped[r, c] <= dist_thr}

    extra_idx = [i for i in range(len(fused_points)) if i not in matched_fused]
    for i in extra_idx:
        d_to_nearest = cost[i].min() if len(gt_arr) else float("inf")
        nearest_name = gt_names[int(np.argmin(cost[i]))] if len(gt_arr) else "?"
        print(f"    [extra-fused] xy={fused_arr[i].round(2)} n_cams={fused_points[i]['n_cams']} "
              f"最近傍GT={nearest_name} 距離={d_to_nearest:.2f}m "
              f"{'(分裂の疑い:すぐ近くに本物がいる)' if d_to_nearest <= dist_thr * 2 else '(幽霊検出の疑い:近くに本物がいない)'}")
    return extra_idx


def draw_frame(
    frame_idx: int, regions, poses, cam_polygons: Dict[str, np.ndarray], real_w: float, real_h: float,
    gt_raw: List[dict], fused_points: List[dict], has_gt: bool = True,
) -> np.ndarray:
    scale = 40
    margin = 110
    w_px = int(real_w * scale) + margin
    h_px = int(real_h * scale) + 40
    if w_px % 2 != 0:
        w_px += 1
    if h_px % 2 != 0:
        h_px += 1
    canvas = np.full((h_px, w_px, 3), (30, 30, 30), dtype=np.uint8)
    draw_regions(canvas, regions, scale)

    # 曲線境界(カメラごとの許容半径R_maxを床面に逆投影した多角形)を
    # 色分けした輪郭線で描く(直線境界版のオレンジの帯の代わり)
    for cid in CAM_IDS:
        poly = cam_polygons.get(cid)
        if poly is None or len(poly) < 3:
            continue
        pts = (poly * scale).astype(np.int32).reshape(-1, 1, 2)
        color = CAM_ZONE_COLORS_BGR.get(cid, (255, 255, 255))
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)

    for cid in CAM_IDS:
        p = poses[cid]
        px, py = int(p.t[0] * scale), int(p.t[1] * scale)
        if 0 <= px < w_px and 0 <= py < h_px:
            cv2.circle(canvas, (px, py), 9, (0, 0, 255), -1)
            cv2.circle(canvas, (px, py), 11, (255, 255, 255), 2)
            cv2.putText(canvas, cid, (px + 14, py - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)

    for item in gt_raw:
        xy = item["world_xy"]
        px, py = int(xy[0] * scale), int(xy[1] * scale)
        if 0 <= px < w_px and 0 <= py < h_px:
            cv2.drawMarker(canvas, (px, py), GT_MARKER_COLOR_BGR, markerType=cv2.MARKER_CROSS,
                           markerSize=10, thickness=1)
            cam_suffix = item["cam_id"].replace("Camera_", "C")
            label = f"{item['name'].replace('cow', '')}({cam_suffix})"
            cv2.putText(canvas, label, (px + 6, py - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.32, GT_MARKER_COLOR_BGR, 1, cv2.LINE_AA)

    extra_idx = set(match_fused_to_gt_and_flag_extras(gt_raw, fused_points)) if has_gt else set()

    for i, fp in enumerate(fused_points):
        xy = fp["world_xy"]
        px, py = int(xy[0] * scale), int(xy[1] * scale)
        if 0 <= px < w_px and 0 <= py < h_px:
            zone = classify_curved_zone(xy, CAM_IDS, cam_polygons)
            color = CAM_ZONE_COLORS_BGR.get(zone, OVERLAP_ZONE_COLOR_BGR)
            border = (255, 255, 255) if fp["n_cams"] > 1 else (0, 0, 0)
            thickness = 2 if fp["n_cams"] > 1 else 1
            cv2.circle(canvas, (px, py), 6, color, -1)
            cv2.circle(canvas, (px, py), 8, border, thickness)
            if i in extra_idx:
                cv2.circle(canvas, (px, py), 13, (255, 0, 255), 2)
                cv2.putText(canvas, "EXCESS", (px + 8, py + 18), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (255, 0, 255), 1, cv2.LINE_AA)

    n_gt = len({item["name"] for item in gt_raw})
    n_fused = len(fused_points)
    if has_gt:
        header = f"{frame_name(frame_idx)}  GT={n_gt}(点{len(gt_raw)})  fused={n_fused}  excess={n_fused - n_gt}  [曲線境界]"
    else:
        header = f"{frame_name(frame_idx)}  GT=無し(アノテーション無し)  fused={n_fused}  [曲線境界]"
    cv2.putText(canvas, header, (8, h_px - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)

    legend_items = [(cid, CAM_ZONE_COLORS_BGR[cid]) for cid in CAM_IDS] + [
        ("overlap(curved)", OVERLAP_ZONE_COLOR_BGR), ("GT(+)", GT_MARKER_COLOR_BGR),
    ]
    for i, (label, color) in enumerate(legend_items):
        y = 20 + i * 22
        cv2.rectangle(canvas, (w_px - margin + 5, y), (w_px - margin + 21, y + 16), color, -1)
        cv2.putText(canvas, label, (w_px - margin + 26, y + 13), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "white縁=2cam統合", (w_px - margin + 5, 20 + len(legend_items) * 22 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "black縁=1camのみ", (w_px - margin + 5, 20 + len(legend_items) * 22 + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas


def make_video_from_frames(frames_dir: Path, video_path: Path, fps: float, saved_paths: List[Path]) -> None:
    if not saved_paths:
        return

    if shutil.which("ffmpeg") is not None:
        duration = 1.0 / fps
        list_path = frames_dir / "_ffmpeg_concat_list.txt"
        with open(list_path, "w", encoding="utf-8") as f:
            for p in saved_paths:
                f.write(f"file '{Path(p).resolve()}'\n")
                f.write(f"duration {duration}\n")
            f.write(f"file '{Path(saved_paths[-1]).resolve()}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", str(list_path),
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-r", str(fps),
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        list_path.unlink(missing_ok=True)
        if result.returncode == 0:
            print(f"動画も保存しました(ffmpeg): {video_path} ({fps}fps, {len(saved_paths)}フレーム分)")
            return
        print(f"ffmpegでの動画作成に失敗しました(終了コード{result.returncode}):\n{result.stderr[-2000:]}")
        print("cv2.VideoWriterにフォールバックします...")

    first = cv2.imread(str(saved_paths[0]))
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
    for p in saved_paths:
        writer.write(cv2.imread(str(p)))
    writer.release()
    print(f"動画も保存しました(cv2.VideoWriterフォールバック、"
          f"環境によっては色が壊れる場合があります): {video_path} ({fps}fps)")


def main():
    from ultralytics import YOLO

    print("floorplan.json 読み込み中...")
    regions, cameras_json, real_w, real_h = load_floorplan(FLOORPLAN_JSON_PATH)
    cameras_json = apply_camera_overrides(cameras_json)
    poses = resolve_poses(cameras_json, root="Area1", fmt=ROT_FMT)

    print(f"各カメラの曲線境界(許容半径R_max)を計算中(高さ仮定={ZONE_HEIGHT_M}m)...")
    cam_polygons = build_camera_polygons(poses, CAM_IDS, get_intrinsics, height_m=ZONE_HEIGHT_M, n_angles=N_ANGLES)
    adjacent_pairs = get_adjacent_pairs(CAM_IDS)

    print(f"YOLOモデル読み込み中: {YOLO_WEIGHTS_PATH}")
    yolo_model = YOLO(YOLO_WEIGHTS_PATH)

    xml_dir = Path(XML_DIR)
    image_dir = Path(IMAGE_DIR)
    frame_indices = list(range(FRAME_START, FRAME_END + 1, FRAME_STEP))

    saved_paths = []
    for frame_idx in frame_indices:
        xml_path = xml_dir / f"{frame_name(frame_idx)}.xml"
        image_path = image_dir / f"{frame_name(frame_idx)}.jpg"
        if not image_path.exists():
            continue

        has_gt = xml_path.exists()
        if has_gt:
            gt_objects = parse_voc_annotation(xml_path)
            gt_per_camera_raw = gt_world_points_per_camera(gt_objects, poses, IMAGE_WIDTH_PX)
            gt_per_camera_filtered = filter_by_curved_zone(gt_per_camera_raw, cam_polygons)
            gt_raw = [
                {"name": it["name"], "cam_id": cam_id, "world_xy": it["world_xy"]}
                for cam_id, items in gt_per_camera_filtered.items() for it in items
            ]
        else:
            print(f"  {frame_name(frame_idx)}: GTアノテーション(xml)が無いため、"
                  f"GT無しでBEV画像のみ生成します(精度評価・excess判定は不可)")
            gt_raw = []

        per_camera_raw = yolo_world_points_per_camera_with_pose(yolo_model, image_path, poses)
        per_camera_deduped = deduplicate_per_camera(per_camera_raw, DEDUP_IOU_THRESHOLD)
        per_camera_filtered = filter_by_curved_zone(per_camera_deduped, cam_polygons)
        # 牛舎(floorplan)の矩形範囲の外側に投影された検出(幽霊検出の主因)を除外
        per_camera_filtered = filter_outside_floorplan_bounds(
            per_camera_filtered, real_w, real_h, margin_m=0.0, verbose=False
        )
        fused_points = fuse_frame_zone_pose(
            per_camera_filtered, adjacent_pairs, DIST_THR_FUSION_M, USE_POSE_CONSTRAINT
        )

        n_gt_preview = len({item["name"] for item in gt_raw})
        if has_gt and len(fused_points) > n_gt_preview:
            print(f"  {frame_name(frame_idx)}: fused({len(fused_points)}) > GT({n_gt_preview}) の原因を特定中...")
        canvas = draw_frame(frame_idx, regions, poses, cam_polygons, real_w, real_h, gt_raw, fused_points, has_gt=has_gt)
        out_path = FRAMES_OUT_DIR / f"{frame_name(frame_idx)}.png"
        cv2.imwrite(str(out_path), canvas)
        saved_paths.append(out_path)

        annotated_paths = save_annotated_camera_images(
            frame_idx, image_path, per_camera_raw, per_camera_deduped, per_camera_filtered
        )

        n_gt_names = len({item["name"] for item in gt_raw})
        gt_label = f"GT={n_gt_names}(点{len(gt_raw)})" if has_gt else "GT=無し"
        print(f"  保存: {out_path} + 元画像{len(annotated_paths)}枚 "
              f"({gt_label} fused={len(fused_points)})")

    print(f"\n{len(saved_paths)}枚のBEV画像を保存しました: {FRAMES_OUT_DIR}")

    if MAKE_VIDEO and saved_paths:
        video_path = OUTPUT_DIR / "bev_frames_yolo_zonepose_curved.mp4"
        make_video_from_frames(FRAMES_OUT_DIR, video_path, VIDEO_FPS, saved_paths)


if __name__ == "__main__":
    main()