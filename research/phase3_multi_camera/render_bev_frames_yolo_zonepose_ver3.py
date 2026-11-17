"""
render_bev_frames_yolo_zonepose_ver3.py

phase3_bev_multicam_yolo_check_zonepose.py と同じ融合パイプライン(有効範囲外の
除外+隣接カメラ制約+姿勢クラス制約、Hungarianバグ修正込み)を使い、対象フレーム
1枚1枚についてBEV(鳥瞰図)画像を出力する。「境目での統合漏れ」「過剰統合」
「YOLOの誤検出」がどこでどれくらい起きているかを目視で判断するためのもの。

各画像には:
  - フロアプラン(壁・牛床などの領域)
  - 隣接カメラの重なり範囲(オレンジの帯)
  - カメラ位置(赤丸)
  - GTの真の位置(黄色い十字、個体名付き)
  - YOLO+融合パイプラインの出力点(カメラ専属範囲の色 or 重なり範囲はグレー、
    縁の太さ・色でn_cams(統合に使われたカメラ数、1台のみ=黒縁細線、
    2台統合=白縁太線)を表現)
を重ねて描画する。GTの黄十字と融合点の丸がズレていたり、1つのGTに対して融合点が
2つ並んでいたりする箇所を見れば、統合漏れ・過剰統合の場所が視覚的に分かる。

出力先: OUTPUT_DIR/bev_frames_yolo_zonepose/{frame}.png (1フレーム1枚)
オプションでMAKE_VIDEO=Trueにすると、それらをつなげたmp4も書き出す。

--- 使い方 ---
    python render_bev_frames_yolo_zonepose.py
"""

import os
import sys
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
    load_floorplan, apply_camera_overrides, resolve_poses,
    unproject_to_floor,
    union_find_clusters,
    filter_outside_floorplan_bounds,
)

# ---------------------------------------------------------------------------
# 設定(phase3_bev_multicam_yolo_check_zonepose.pyと同じ値)
# ---------------------------------------------------------------------------
import xml.etree.ElementTree as ET

XML_DIR = "/mnt/vmlqnap02/dataset/MooTrack360/1-hour-farm1-area1-sample-sequence/xml-1fps"
IMAGE_DIR = "/mnt/vmlqnap02/dataset/MooTrack360/1-hour-farm1-area1-sample-sequence/1-hour-farm1-area1-frames"
# 注意: このFLOORPLAN_JSON_PATHは以前から要検証(値が古い可能性あり)と指摘済みのファイル
FLOORPLAN_JSON_PATH = "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/calibration/Farm2_Area_1_Artboard_config_.json"
OUTPUT_DIR = Path(
    "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/research/idf1_check"
)
FRAMES_OUT_DIR = OUTPUT_DIR / "bev_frames_yolo_zonepose"
FRAMES_OUT_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATED_OUT_DIR = OUTPUT_DIR / "annotated_frames_yolo_zonepose"
ANNOTATED_OUT_DIR.mkdir(parents=True, exist_ok=True)

YOLO_WEIGHTS_PATH = "/mnt/vmlqnap02/dataset/MooTrack360/Trained models/yolo11x/train_ultralytics/weights/best.pt"
YOLO_CONF_THRESH = 0.4

FRAME_START = 0
FRAME_END = 35990
FRAME_STEP = 10

IMAGE_WIDTH_PX = 8832

LYING_HEIGHT_M = 0.75   # phase3_bev_multicam_yolo_check_zonepose.pyと揃える(暫定値)
STANDING_HEIGHT_M = 0.75
DEFAULT_HEIGHT_M = 0.8

DIST_THR_FUSION_M = 1.5
OVERLAP_HALF_WIDTH_M = 1.5

USE_POSE_CONSTRAINT = True

# 同一カメラ内の重複検出除去: IoUがこの閾値以上のペアを同一個体とみなし、
# bboxが大きい方だけを残す。
DEDUP_IOU_THRESHOLD = 0.5

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
    """phase3_bev_multicam_gt_check_zonepose.py の compute_world_points_with_pose と
    同じロジック(カメラのタイル位置で振り分け、pose別高さで逆投影)。
    戻り値: cam_id -> [{"name":..., "pose":..., "world_xy":...}, ...] (フィルタ前)"""
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
    """同一カメラ内の検出について、bbox IoUがiou_thr以上のペアを同一個体とみなし、
    bboxが大きい方だけを残す(クラスを跨いで比較する)。"""
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
    """GTのbboxをカメラごとのローカル座標(そのカメラのタイル内でのxy)に変換する。
    可視化用(元画像へのYOLO検出重ね描き)にのみ使う。"""
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
    """カメラごとのクロップ画像に、YOLO検出を重ね描きして保存する。
    色=姿勢クラス、太線=最終的に生き残った検出、細線=重複除去(IoU>=閾値で
    より小さいbboxとして捨てられた)、点線風の細線=有効範囲外として除外、
    をそれぞれ区別して表示する。"""
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
# 有効範囲・隣接カメラ制約付き融合(既存スクリプトと同じ)
# ---------------------------------------------------------------------------

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


def classify_exclusive_zone(
    x: float, cam_ids: List[str], zones: Dict[Tuple[str, str], Tuple[float, float]],
) -> str:
    ordered_pairs = list(zip(cam_ids[:-1], cam_ids[1:]))
    band_ranges = [zones[pair] for pair in ordered_pairs]

    if x < band_ranges[0][0]:
        return cam_ids[0]
    for i, (lo, hi) in enumerate(band_ranges):
        if lo <= x <= hi:
            return f"overlap_{ordered_pairs[i][0]}_{ordered_pairs[i][1]}"
        if i < len(band_ranges) - 1 and hi < x < band_ranges[i + 1][0]:
            return cam_ids[i + 1]
    if x > band_ranges[-1][1]:
        return cam_ids[-1]
    return "unknown"


def match_zone_pose_constrained(
    per_camera: Dict[str, List[dict]],
    cam_ids: List[str],
    zones: Dict[Tuple[str, str], Tuple[float, float]],
    dist_thr: float,
    use_pose_constraint: bool = True,
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
    cam_ids: List[str],
    zones: Dict[Tuple[str, str], Tuple[float, float]],
    dist_thr: float,
    use_pose_constraint: bool = True,
) -> List[dict]:
    accepted_pairs = match_zone_pose_constrained(per_camera_filtered, cam_ids, zones, dist_thr, use_pose_constraint)
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
    """「鳥瞰図上はGTと個体数が同じに見えるのにfusedだけ多い」原因を特定するため、
    fused_pointsをGT(個体名で平均した1個体1点)に1対1対応させ、どのGTにも
    対応しなかった(=余った)fused_pointのインデックスを返す。
    この「余ったfused_point」こそが、excess(fused-GT)を生んでいる具体的な点。
    余った点が本物の個体のすぐそば(distで言うと近い)にあれば分裂(統合漏れ)、
    遠く離れていれば幽霊検出(誤検出・誤投影)だと判断できる。"""
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
    frame_idx: int, regions, poses, zones, real_w: float, real_h: float,
    gt_raw: List[dict], fused_points: List[dict],
) -> np.ndarray:
    scale = 40
    margin = 110
    w_px = int(real_w * scale) + margin
    h_px = int(real_h * scale) + 40
    canvas = np.full((h_px, w_px, 3), (30, 30, 30), dtype=np.uint8)
    draw_regions(canvas, regions, scale)

    for (cam_i, cam_j), (lo, hi) in zones.items():
        x0, x1 = int(lo * scale), int(hi * scale)
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x0, 0), (x1, h_px), (0, 165, 255), -1)
        cv2.addWeighted(overlay, 0.15, canvas, 0.85, 0, canvas)
        cv2.putText(canvas, f"{cam_i}-{cam_j}", (x0 + 4, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 165, 255), 1, cv2.LINE_AA)

    for cid in CAM_IDS:
        p = poses[cid]
        px, py = int(p.t[0] * scale), int(p.t[1] * scale)
        if 0 <= px < w_px and 0 <= py < h_px:
            cv2.circle(canvas, (px, py), 9, (0, 0, 255), -1)
            cv2.circle(canvas, (px, py), 11, (255, 255, 255), 2)
            cv2.putText(canvas, cid, (px + 14, py - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # GTの真の位置(平均せず、カメラごとの実際の投影点をそのまま黄十字で描画)
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

    # GTと1対1対応しなかった「余り」のfused_pointを特定する
    # (鳥瞰図上はGTと個体数が同じに見えてもfusedだけ多い原因を特定するため)
    extra_idx = set(match_fused_to_gt_and_flag_extras(gt_raw, fused_points))

    # 融合パイプラインの出力点
    for i, fp in enumerate(fused_points):
        xy = fp["world_xy"]
        px, py = int(xy[0] * scale), int(xy[1] * scale)
        if 0 <= px < w_px and 0 <= py < h_px:
            zone = classify_exclusive_zone(float(xy[0]), CAM_IDS, zones)
            color = CAM_ZONE_COLORS_BGR.get(zone, OVERLAP_ZONE_COLOR_BGR)
            border = (255, 255, 255) if fp["n_cams"] > 1 else (0, 0, 0)
            thickness = 2 if fp["n_cams"] > 1 else 1
            cv2.circle(canvas, (px, py), 6, color, -1)
            cv2.circle(canvas, (px, py), 8, border, thickness)
            if i in extra_idx:
                # GTのどれにも対応しなかった「余り」の点を目立たせる
                # (マゼンタの太いリング+EXCESSラベル)
                cv2.circle(canvas, (px, py), 13, (255, 0, 255), 2)
                cv2.putText(canvas, "EXCESS", (px + 8, py + 18), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (255, 0, 255), 1, cv2.LINE_AA)

    n_gt = len({item["name"] for item in gt_raw})
    n_fused = len(fused_points)
    header = f"{frame_name(frame_idx)}  GT={n_gt}(点{len(gt_raw)})  fused={n_fused}  excess={n_fused - n_gt}"
    cv2.putText(canvas, header, (8, h_px - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # 凡例
    legend_items = [(cid, CAM_ZONE_COLORS_BGR[cid]) for cid in CAM_IDS] + [
        ("overlap", OVERLAP_ZONE_COLOR_BGR), ("GT(+)", GT_MARKER_COLOR_BGR),
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


def main():
    from ultralytics import YOLO

    print("floorplan.json 読み込み中...")
    regions, cameras_json, real_w, real_h = load_floorplan(FLOORPLAN_JSON_PATH)
    cameras_json = apply_camera_overrides(cameras_json)
    poses = resolve_poses(cameras_json, root="Area1", fmt=ROT_FMT)

    zones = get_adjacent_zones(poses, CAM_IDS, OVERLAP_HALF_WIDTH_M)
    valid_ranges = get_camera_valid_ranges(CAM_IDS, zones)

    print(f"YOLOモデル読み込み中: {YOLO_WEIGHTS_PATH}")
    yolo_model = YOLO(YOLO_WEIGHTS_PATH)

    xml_dir = Path(XML_DIR)
    image_dir = Path(IMAGE_DIR)
    frame_indices = list(range(FRAME_START, FRAME_END + 1, FRAME_STEP))

    saved_paths = []
    for frame_idx in frame_indices:
        xml_path = xml_dir / f"{frame_name(frame_idx)}.xml"
        image_path = image_dir / f"{frame_name(frame_idx)}.jpg"
        if not xml_path.exists() or not image_path.exists():
            continue

        gt_objects = parse_voc_annotation(xml_path)
        # phase3_bev_multicam_gt_check_zonepose.pyと同じく、カメラのタイル位置で
        # 振り分けたあと、そのカメラの有効範囲外の投影は除外する(見た目のノイズになる
        # 周辺視野の誤投影を消すため。評価用のGT位置とは別物、あくまで可視化用)
        gt_per_camera_raw = gt_world_points_per_camera(gt_objects, poses, IMAGE_WIDTH_PX)
        gt_per_camera_filtered = filter_out_of_range_detections(gt_per_camera_raw, valid_ranges)
        gt_raw = [
            {"name": it["name"], "cam_id": cam_id, "world_xy": it["world_xy"]}
            for cam_id, items in gt_per_camera_filtered.items() for it in items
        ]

        per_camera_raw = yolo_world_points_per_camera_with_pose(yolo_model, image_path, poses)
        per_camera_deduped = deduplicate_per_camera(per_camera_raw, DEDUP_IOU_THRESHOLD)
        per_camera_filtered = filter_out_of_range_detections(per_camera_deduped, valid_ranges)
        # 牛舎(floorplan)の矩形範囲の外側に投影された検出(幽霊検出の主因)を除外
        per_camera_filtered = filter_outside_floorplan_bounds(
            per_camera_filtered, real_w, real_h, margin_m=0.0, verbose=False
        )
        fused_points = fuse_frame_zone_pose(
            per_camera_filtered, CAM_IDS, zones, DIST_THR_FUSION_M, USE_POSE_CONSTRAINT
        )

        n_gt_preview = len({item["name"] for item in gt_raw})
        if len(fused_points) > n_gt_preview:
            print(f"  {frame_name(frame_idx)}: fused({len(fused_points)}) > GT({n_gt_preview}) の原因を特定中...")
        canvas = draw_frame(frame_idx, regions, poses, zones, real_w, real_h, gt_raw, fused_points)
        out_path = FRAMES_OUT_DIR / f"{frame_name(frame_idx)}.png"
        cv2.imwrite(str(out_path), canvas)
        saved_paths.append(out_path)

        annotated_paths = save_annotated_camera_images(
            frame_idx, image_path, per_camera_raw, per_camera_deduped, per_camera_filtered
        )

        n_gt_names = len({item["name"] for item in gt_raw})
        print(f"  保存: {out_path} + 元画像{len(annotated_paths)}枚 "
              f"(GT={n_gt_names}(点{len(gt_raw)}) fused={len(fused_points)})")

    print(f"\n{len(saved_paths)}枚のBEV画像を保存しました: {FRAMES_OUT_DIR}")

    if MAKE_VIDEO and saved_paths:
        first = cv2.imread(str(saved_paths[0]))
        h, w = first.shape[:2]
        video_path = OUTPUT_DIR / "bev_frames_yolo_zonepose.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, VIDEO_FPS, (w, h))
        for p in saved_paths:
            writer.write(cv2.imread(str(p)))
        writer.release()
        print(f"動画も保存しました: {video_path} ({VIDEO_FPS}fps)")


if __name__ == "__main__":
    main()