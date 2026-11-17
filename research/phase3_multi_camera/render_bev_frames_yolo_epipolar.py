"""
render_bev_frames_yolo_epipolar.py

phase3_bev_multicam_yolo_check_epipolar.pyと同じエピポーラ統合パイプライン
(レイの最近接距離ベース、隣接カメラ制約+姿勢クラス制約+同一カメラ内重複除去)を
使い、各フレームのBEV画像を1枚ずつ出力する。

エピポーラ版で観測されたIDs(ID切り替わり)=53件という異常な多さの原因を
目視で確認するためのもの。各点は:
  - 色/縁: n_cams=2(レイの三角測量に成功、実座標)は緑の太い白縁、
           n_cams=1(単独カメラの高さ仮定フォールバック)は橙の細い黒縁
  - ラベル: SimpleBevTrackerが割り当てた追跡ID
を表示する。同じ牛がフレームごとに三角測量の成功(緑)/失敗(橙)を行き来して、
その度にラベルの追跡IDが変わっている様子が見えれば、「三角測量とフォールバックの
切り替わりでフレーム間の位置がジャンプし、追跡が途切れている」という仮説の
裏付けになる。

出力先: OUTPUT_DIR/bev_frames_yolo_epipolar/{frame}.png (1フレーム1枚)
MAKE_VIDEO=Trueなら、それらをつなげたmp4も書き出す。

--- 使い方 ---
    python render_bev_frames_yolo_epipolar.py
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
    load_floorplan, apply_camera_overrides, resolve_poses,
    unproject_ds, unproject_to_floor,
    union_find_clusters,
)

# ---------------------------------------------------------------------------
# 設定(phase3_bev_multicam_yolo_check_epipolar.pyと同じ値)
# ---------------------------------------------------------------------------
XML_DIR = "/mnt/vmlqnap02/dataset/MooTrack360/1-hour-farm1-area1-sample-sequence/xml-1fps"
IMAGE_DIR = "/mnt/vmlqnap02/dataset/MooTrack360/1-hour-farm1-area1-sample-sequence/1-hour-farm1-area1-frames"
FLOORPLAN_JSON_PATH = "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/calibration/Farm2_Area_1_Artboard_config_.json"
OUTPUT_DIR = Path(
    "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/research/idf1_check"
)
FRAMES_OUT_DIR = OUTPUT_DIR / "bev_frames_yolo_epipolar"
FRAMES_OUT_DIR.mkdir(parents=True, exist_ok=True)

YOLO_WEIGHTS_PATH = "/mnt/vmlqnap02/dataset/MooTrack360/Trained models/yolo11x/train_ultralytics/weights/best.pt"
YOLO_CONF_THRESH = 0.33

FRAME_START = 0
FRAME_END = 290
FRAME_STEP = 10

IMAGE_WIDTH_PX = 8832

LYING_HEIGHT_M = 0.5
STANDING_HEIGHT_M = 0.75
DEFAULT_HEIGHT_M = 0.8

RAY_DIST_THR_M = 1
HEIGHT_RANGE_Z = (-1.6, 0.1)

USE_POSE_CONSTRAINT = True
DEDUP_IOU_THRESHOLD = 0.5

DIST_THR_TRACK_M = 1.0
MAX_MISSED_FRAMES = 2

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
YOLO_CLASS_TO_POSE = {0: "lying", 1: "standing"}
ADJACENT_PAIRS: List[Tuple[str, str]] = list(zip(CAM_IDS[:-1], CAM_IDS[1:]))

MAKE_VIDEO = True
VIDEO_FPS = 2

TYPE_COLORS_BGR = {
    "wall": (0, 200, 0), "cow_beds": (0, 140, 255), "feeding_bay": (255, 140, 0),
    "water": (255, 255, 0), "gate": (0, 0, 255), "milking_station": (255, 0, 255),
    "hay_area": (0, 255, 255), "cow_brushes": (255, 0, 180),
}
GT_MARKER_COLOR_BGR = (0, 255, 255)      # 黄色
TRIANGULATED_COLOR_BGR = (60, 200, 60)   # 緑(n_cams=2, 三角測量成功)
FALLBACK_COLOR_BGR = (60, 140, 255)      # 橙(n_cams=1, 高さ仮定フォールバック)


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


def unproject_to_ray(pts_px, R, fx, fy, cx, cy, xi, alpha):
    dirs_cam, valid = unproject_ds(pts_px, fx, fy, cx, cy, xi, alpha)
    dirs_world = (R @ dirs_cam.T).T
    return dirs_world, valid


def yolo_detections_per_camera(
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
        cam_pose = poses[cam_id]

        pts_px, poses_list, bboxes = [], [], []
        for box in results.boxes:
            bx1, by1, bx2, by2 = box.xyxy[0].tolist()
            pts_px.append([(bx1 + bx2) / 2.0, (by1 + by2) / 2.0])
            poses_list.append(YOLO_CLASS_TO_POSE.get(int(box.cls[0])))
            bboxes.append((bx1, by1, bx2, by2))
        pts_px = np.array(pts_px, dtype=np.float64)

        dirs_world, valid_ray = unproject_to_ray(pts_px, cam_pose.R, fx, fy, cx, cy, xi, alpha)

        heights = np.array([height_for(p) for p in poses_list], dtype=np.float64)
        world_xy_fb, valid_fb = unproject_to_floor(
            pts_px, cam_pose.R, cam_pose.t, fx, fy, cx, cy, xi, alpha, floor_z=-heights,
        )

        for j in range(len(pts_px)):
            if valid_ray[j] and valid_fb[j]:
                per_camera[cam_id].append({
                    "pose": poses_list[j],
                    "bbox_local": bboxes[j],
                    "dir_world": dirs_world[j],
                    "world_xy_fallback": world_xy_fb[j],
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


def closest_points_between_rays(
    p1: np.ndarray, d1: np.ndarray, p2: np.ndarray, d2: np.ndarray,
) -> Tuple[float, float, np.ndarray, np.ndarray, float]:
    w0 = p1 - p2
    a = float(np.dot(d1, d1))
    b = float(np.dot(d1, d2))
    c = float(np.dot(d2, d2))
    d = float(np.dot(d1, w0))
    e = float(np.dot(d2, w0))
    denom = a * c - b * b
    if abs(denom) < 1e-9:
        return -1.0, -1.0, p1, p2, float("inf")
    t1 = (b * e - c * d) / denom
    t2 = (a * e - b * d) / denom
    point1 = p1 + t1 * d1
    point2 = p2 + t2 * d2
    dist = float(np.linalg.norm(point1 - point2))
    return t1, t2, point1, point2, dist


def match_epipolar(
    per_camera: Dict[str, List[dict]],
    poses: Dict[str, "Pose"],
    adjacent_pairs: List[Tuple[str, str]],
    ray_dist_thr: float,
    height_range: Tuple[float, float],
    use_pose_constraint: bool = True,
) -> Tuple[List[Tuple[Tuple[str, int], Tuple[str, int]]], Dict[Tuple[Tuple[str, int], Tuple[str, int]], np.ndarray]]:
    accepted_pairs: List[Tuple[Tuple[str, int], Tuple[str, int]]] = []
    triangulated: Dict[Tuple[Tuple[str, int], Tuple[str, int]], np.ndarray] = {}

    for cam_i, cam_j in adjacent_pairs:
        items_i = per_camera[cam_i]
        items_j = per_camera[cam_j]
        if not items_i or not items_j:
            continue
        p_i, p_j = poses[cam_i].t, poses[cam_j].t
        n_i, n_j = len(items_i), len(items_j)
        cost = np.full((n_i, n_j), 1e6)
        cand_points: Dict[Tuple[int, int], np.ndarray] = {}

        for a, it_i in enumerate(items_i):
            for b, it_j in enumerate(items_j):
                if (use_pose_constraint and it_i["pose"] is not None
                        and it_j["pose"] is not None and it_i["pose"] != it_j["pose"]):
                    continue
                t1, t2, pt1, pt2, dist = closest_points_between_rays(
                    p_i, it_i["dir_world"], p_j, it_j["dir_world"]
                )
                if t1 <= 0 or t2 <= 0:
                    continue
                mid = (pt1 + pt2) / 2.0
                if not (height_range[0] <= mid[2] <= height_range[1]):
                    continue
                if dist <= ray_dist_thr:
                    cost[a, b] = dist
                    cand_points[(a, b)] = mid

        row_idx, col_idx = linear_sum_assignment(cost)
        for a, b in zip(row_idx, col_idx):
            if cost[a, b] <= ray_dist_thr:
                key = ((cam_i, a), (cam_j, b))
                accepted_pairs.append(key)
                triangulated[key] = cand_points[(a, b)]

    return accepted_pairs, triangulated


def fuse_frame_epipolar(
    per_camera: Dict[str, List[dict]],
    poses: Dict[str, "Pose"],
    adjacent_pairs: List[Tuple[str, str]],
    ray_dist_thr: float,
    height_range: Tuple[float, float],
    use_pose_constraint: bool = True,
) -> List[dict]:
    accepted_pairs, triangulated = match_epipolar(
        per_camera, poses, adjacent_pairs, ray_dist_thr, height_range, use_pose_constraint
    )
    all_keys = [(cam_id, idx) for cam_id, items in per_camera.items() for idx in range(len(items))]
    if not all_keys:
        return []
    roots = union_find_clusters(all_keys, accepted_pairs)

    cluster_tri_points: Dict[Tuple[str, int], List[np.ndarray]] = defaultdict(list)
    for (k1, _k2), pt in triangulated.items():
        cluster_tri_points[roots[k1]].append(pt)

    clusters: Dict[Tuple[str, int], List[Tuple[str, int]]] = defaultdict(list)
    for k in all_keys:
        clusters[roots[k]].append(k)

    fused = []
    for root, members in clusters.items():
        if root in cluster_tri_points:
            pts = np.array(cluster_tri_points[root])
            xy = pts[:, :2].mean(axis=0)
        else:
            cam_id, idx = members[0]
            xy = per_camera[cam_id][idx]["world_xy_fallback"]
        fused.append({"world_xy": xy, "n_cams": len(members)})
    return fused


class SimpleBevTracker:
    def __init__(self, dist_thr: float, max_missed: int):
        self.dist_thr = dist_thr
        self.max_missed = max_missed
        self.tracks: Dict[int, dict] = {}
        self.next_id = 1

    def update(self, fused_points: List[dict]) -> List[Tuple[int, dict]]:
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
                    self.tracks[tid]["n_cams"] = fused_points[j]["n_cams"]
                    self.tracks[tid]["missed"] = 0
                    matched_trk.add(tid)
                    matched_det.add(j)

        for tid in track_ids:
            if tid not in matched_trk:
                self.tracks[tid]["missed"] += 1

        self.tracks = {tid: t for tid, t in self.tracks.items() if t["missed"] <= self.max_missed}

        for j, fp in enumerate(fused_points):
            if j not in matched_det:
                self.tracks[self.next_id] = {"xy": fp["world_xy"], "n_cams": fp["n_cams"], "missed": 0}
                self.next_id += 1

        return [(tid, t) for tid, t in self.tracks.items() if t["missed"] == 0]


def draw_regions(canvas, regions, scale: float):
    for region in regions:
        pts = (region.polygon_m * scale).astype(np.int32).reshape(-1, 1, 2)
        color = TYPE_COLORS_BGR.get(region.type, (200, 200, 200))
        closed = region.type not in ("wall", "gate")
        cv2.polylines(canvas, [pts], isClosed=closed, color=color, thickness=2, lineType=cv2.LINE_AA)


def draw_frame(
    frame_idx: int, regions, poses, real_w: float, real_h: float,
    gt_world: Dict[str, List[np.ndarray]], track_outputs: List[Tuple[int, dict]],
) -> np.ndarray:
    scale = 40
    margin = 130
    w_px = int(real_w * scale) + margin
    h_px = int(real_h * scale) + 40
    canvas = np.full((h_px, w_px, 3), (30, 30, 30), dtype=np.uint8)
    draw_regions(canvas, regions, scale)

    for cid in CAM_IDS:
        p = poses[cid]
        px, py = int(p.t[0] * scale), int(p.t[1] * scale)
        if 0 <= px < w_px and 0 <= py < h_px:
            cv2.circle(canvas, (px, py), 9, (0, 0, 255), -1)
            cv2.circle(canvas, (px, py), 11, (255, 255, 255), 2)
            cv2.putText(canvas, cid, (px + 14, py - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)

    for name, xys in gt_world.items():
        xy = np.mean(np.array(xys), axis=0)
        px, py = int(xy[0] * scale), int(xy[1] * scale)
        if 0 <= px < w_px and 0 <= py < h_px:
            cv2.drawMarker(canvas, (px, py), GT_MARKER_COLOR_BGR, markerType=cv2.MARKER_CROSS,
                           markerSize=10, thickness=1)

    for tid, t in track_outputs:
        xy = t["xy"]
        px, py = int(xy[0] * scale), int(xy[1] * scale)
        if 0 <= px < w_px and 0 <= py < h_px:
            if t["n_cams"] > 1:
                color, border, thickness, radius = TRIANGULATED_COLOR_BGR, (255, 255, 255), 2, 6
            else:
                color, border, thickness, radius = FALLBACK_COLOR_BGR, (0, 0, 0), 1, 5
            cv2.circle(canvas, (px, py), radius, color, -1)
            cv2.circle(canvas, (px, py), radius + 2, border, thickness)
            cv2.putText(canvas, f"id{tid}", (px + 6, py + 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (255, 255, 255), 1, cv2.LINE_AA)

    n_gt = len(gt_world)
    n_tracks = len(track_outputs)
    n_tri = sum(1 for _tid, t in track_outputs if t["n_cams"] > 1)
    header = f"{frame_name(frame_idx)}  GT={n_gt}  track={n_tracks}(三角測量={n_tri})"
    cv2.putText(canvas, header, (8, h_px - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)

    legend_items = [
        ("GT(+)", GT_MARKER_COLOR_BGR),
        ("triangulated(n_cams=2)", TRIANGULATED_COLOR_BGR),
        ("fallback(n_cams=1)", FALLBACK_COLOR_BGR),
    ]
    for i, (label, color) in enumerate(legend_items):
        y = 20 + i * 22
        cv2.rectangle(canvas, (w_px - margin + 5, y), (w_px - margin + 21, y + 16), color, -1)
        cv2.putText(canvas, label, (w_px - margin + 26, y + 13), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas


def main():
    from ultralytics import YOLO

    print("floorplan.json 読み込み中...")
    regions, cameras_json, real_w, real_h = load_floorplan(FLOORPLAN_JSON_PATH)
    cameras_json = apply_camera_overrides(cameras_json)
    poses = resolve_poses(cameras_json, root="Area1", fmt=ROT_FMT)

    print(f"YOLOモデル読み込み中: {YOLO_WEIGHTS_PATH}")
    yolo_model = YOLO(YOLO_WEIGHTS_PATH)

    xml_dir = Path(XML_DIR)
    image_dir = Path(IMAGE_DIR)
    frame_indices = list(range(FRAME_START, FRAME_END + 1, FRAME_STEP))

    tracker = SimpleBevTracker(dist_thr=DIST_THR_TRACK_M, max_missed=MAX_MISSED_FRAMES)
    saved_paths = []

    for frame_idx in frame_indices:
        xml_path = xml_dir / f"{frame_name(frame_idx)}.xml"
        image_path = image_dir / f"{frame_name(frame_idx)}.jpg"
        if not xml_path.exists() or not image_path.exists():
            continue

        gt_objects = parse_voc_annotation(xml_path)
        gt_world = gt_world_points_this_frame(gt_objects, poses, IMAGE_WIDTH_PX)

        per_camera_raw = yolo_detections_per_camera(yolo_model, image_path, poses)
        per_camera_deduped = deduplicate_per_camera(per_camera_raw, DEDUP_IOU_THRESHOLD)
        fused_points = fuse_frame_epipolar(
            per_camera_deduped, poses, ADJACENT_PAIRS, RAY_DIST_THR_M, HEIGHT_RANGE_Z, USE_POSE_CONSTRAINT
        )
        track_outputs = tracker.update(fused_points)

        canvas = draw_frame(frame_idx, regions, poses, real_w, real_h, gt_world, track_outputs)
        out_path = FRAMES_OUT_DIR / f"{frame_name(frame_idx)}.png"
        cv2.imwrite(str(out_path), canvas)
        saved_paths.append(out_path)

        n_tri = sum(1 for _tid, t in track_outputs if t["n_cams"] > 1)
        print(f"  保存: {out_path} (GT={len(gt_world)} track={len(track_outputs)} 三角測量={n_tri})")

    print(f"\n{len(saved_paths)}枚のBEV画像を保存しました: {FRAMES_OUT_DIR}")

    if MAKE_VIDEO and saved_paths:
        first = cv2.imread(str(saved_paths[0]))
        h, w = first.shape[:2]
        video_path = OUTPUT_DIR / "bev_frames_yolo_epipolar.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, VIDEO_FPS, (w, h))
        for p in saved_paths:
            writer.write(cv2.imread(str(p)))
        writer.release()
        print(f"動画も保存しました: {video_path} ({VIDEO_FPS}fps)")


if __name__ == "__main__":
    main()