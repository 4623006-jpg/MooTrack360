"""
evaluate_yolo_distance_matching.py で使っているのと同じ処理
(YOLO検出 -> カメラ別confidenceしきい値で足切り -> 姿勢(class_id)別の高さで
World座標へ逆投影)を行い、その結果をフロアプラン上の鳥瞰図(BEV)として画像に
描画・保存するスクリプト。

個体統合(Hungarian+Union-Find)は行わず、カメラ別の生の検出点をそのまま
プロットする(どのカメラでどんな点が検出されているかを目視確認する用途)。
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import cv2
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# 設定 (evaluate_yolo_distance_matching.py / make_bev.py と同じ値を使う)
# ---------------------------------------------------------------------------

FLOORPLAN_JSON_PATH = "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/calibration/Farm2_Area_1_Artboard_config_.json"
SOURCE_IMAGE_PATH = "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/annotation_process/3600_frames_result/000000.jpg"
MODEL_PATH = "/mnt/vmlqnap02/dataset/MooTrack360/Trained models/yolo11x/train_ultralytics/weights/best.pt"
OUTPUT_DIR = Path("/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/calibration")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CAMERAS = 3
CAM_IDS = ["Camera_1", "Camera_2", "Camera_3"]

DS_FX, DS_FY = 708.5, 735.2
DS_CX, DS_CY = 1472.0, 1470.1
DS_XI, DS_ALPHA = 0.006105, 0.48977

CAM_INTRINSICS = {
    "Camera_2": {
        "fx": 708.5, "fy": 735.2, "cx": 1472.0, "cy": 1470.1,
        "xi": 0.056105, "alpha": 0.428780,
    },
    "Camera_3": {
        "fx": 708.5, "fy": 735.2, "cx": 1472.0, "cy": 1470.1,
        "xi": -0.143895, "alpha": 0.533116,
    },
}

CAMERA_PARAM_OVERRIDES = {
    "Camera_1->Camera_2": [13.113, 0.1269, -0.0743, -7.2382, 1.3079, -0.8149],
    "Camera_2->Camera_3": [13.641, 0.1279, -1.6374, 11.0855, -0.3336, 0.7534],
}

ROT_FMT = "euler_zyx_deg"

YOLO_PREDICT_CONF = 0.01
YOLO_PREDICT_IOU = 0.5
YOLO_PREDICT_IMGSZ = 2560
YOLO_PREDICT_CLASSES = [0, 1]

CAMERA_CONF_THR = {
    "Camera_1": 0.5,
    "Camera_2": 0.5,
    "Camera_3": 0.15,
}

# class_id 0 = lying(座位) -> 0.4m, class_id 1 = standing(立位) -> 0.75m
CLASS_HEIGHT_M = {
    0: 0.4,
    1: 0.75,
}
DEFAULT_HEIGHT_M = 0.575

TYPE_COLORS_BGR = {
    "wall": (0, 200, 0), "cow_beds": (0, 140, 255), "feeding_bay": (255, 140, 0),
    "water": (255, 255, 0), "gate": (0, 0, 255), "milking_station": (255, 0, 255),
    "hay_area": (0, 255, 255), "cow_brushes": (255, 0, 180),
}

CAM_COLORS_BGR = {
    "Camera_1": (255, 80, 80),
    "Camera_2": (80, 220, 80),
    "Camera_3": (80, 180, 255),
}


@dataclass
class Region:
    name: str
    type: str
    polygon_m: np.ndarray


@dataclass
class Pose:
    R: np.ndarray
    t: np.ndarray


def get_intrinsics(cam_id: str) -> Tuple[float, float, float, float, float, float]:
    if cam_id in CAM_INTRINSICS:
        p = CAM_INTRINSICS[cam_id]
        return p["fx"], p["fy"], p["cx"], p["cy"], p["xi"], p["alpha"]
    return DS_FX, DS_FY, DS_CX, DS_CY, DS_XI, DS_ALPHA


def load_floorplan(path: str):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    regions = []
    for r in data.get("semantic_regions", []):
        poly = np.array(r["polygon_world_meters"], dtype=np.float64)
        regions.append(Region(r["name"], r.get("type", ""), poly))
    real_w = data["metadata"]["real_world_dimensions"]["width"]
    real_h = data["metadata"]["real_world_dimensions"]["height"]
    cameras_json = data.get("cameras", [])
    return regions, cameras_json, real_w, real_h


def apply_camera_overrides(cameras_json: list) -> list:
    result = []
    for entry in cameras_json:
        key = f"{entry['parent']}->{entry['child']}"
        new_entry = dict(entry)
        if key in CAMERA_PARAM_OVERRIDES:
            new_entry["params"] = list(CAMERA_PARAM_OVERRIDES[key])
        result.append(new_entry)
    return result


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def build_R(a, b, c, fmt: str) -> np.ndarray:
    if "_deg" in fmt:
        a, b, c = np.radians([a, b, c])
        base_fmt = fmt.replace("_deg", "")
    else:
        base_fmt = fmt
    if base_fmt == "euler_zyx":
        return _rz(a) @ _ry(b) @ _rx(c)
    raise ValueError(f"このスクリプトは euler_zyx(_deg) のみサポート: {fmt}")


def resolve_poses(cameras_json: list, root: str = "Area1", fmt: str = ROT_FMT) -> Dict[str, Pose]:
    poses: Dict[str, Pose] = {root: Pose(R=np.eye(3), t=np.zeros(3))}
    remaining = list(cameras_json)
    while remaining:
        progressed = False
        for entry in list(remaining):
            parent, child = entry["parent"], entry["child"]
            if parent not in poses:
                continue
            px, py, pz, a, b, c = entry["params"]
            R_abs = build_R(a, b, c, fmt)
            t_local = np.array([px, py, pz], dtype=np.float64)
            t_world = poses[parent].t + t_local
            poses[child] = Pose(R=R_abs, t=t_world)
            remaining.remove(entry)
            progressed = True
        if not progressed:
            raise RuntimeError(f"解決できないカメラがあります: {[e['child'] for e in remaining]}")
    return poses


def unproject_ds(pts_px, fx, fy, cx, cy, xi, alpha):
    mx = (pts_px[:, 0] - cx) / fx
    my = (pts_px[:, 1] - cy) / fy
    r2 = mx ** 2 + my ** 2

    if alpha <= 0.5:
        valid = np.ones(len(pts_px), dtype=bool)
    else:
        valid = r2 < 1.0 / (2.0 * alpha - 1.0) - 1e-9

    inner = np.maximum(1.0 - (2.0 * alpha - 1.0) * r2, 0.0)
    mz = (1.0 - alpha ** 2 * r2) / (alpha * np.sqrt(inner) + (1.0 - alpha))

    scale = np.sqrt(mx ** 2 + my ** 2 + mz ** 2)
    safe = valid & (scale > 1e-9)
    dx = np.where(safe, mx, 0.0)
    dy = np.where(safe, my, 0.0)
    dz = np.where(safe, mz - xi * scale, 0.0)

    norm = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    norm = np.where(norm > 1e-9, norm, 1.0)
    dirs = np.stack([dx / norm, dy / norm, dz / norm], axis=-1)
    return dirs, safe


def unproject_to_floor(pts_px, R, t, fx, fy, cx, cy, xi, alpha, floor_z=0.0):
    dirs_cam, valid = unproject_ds(pts_px, fx, fy, cx, cy, xi, alpha)
    dirs_world = (R @ dirs_cam.T).T

    dz = dirs_world[:, 2]
    t_safe = np.abs(dz) > 1e-9
    tt = np.where(t_safe, (floor_z - t[2]) / dz, 0.0)
    valid = valid & t_safe & (tt > 0)

    world_x = np.where(valid, t[0] + tt * dirs_world[:, 0], 0.0)
    world_y = np.where(valid, t[1] + tt * dirs_world[:, 1], 0.0)
    return np.stack([world_x, world_y], axis=-1), valid


def yolo_results_to_raw_boxes(results, conf_threshold: float = YOLO_PREDICT_CONF) -> List[dict]:
    boxes = results.boxes
    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    raw_boxes = []
    for (x1, y1, x2, y2), c, cf in zip(xyxy, cls, conf):
        if cf < conf_threshold:
            continue
        raw_boxes.append({
            "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
            "class_id": int(c), "conf": float(cf),
        })
    return raw_boxes


def bucket_detections_by_camera(
    raw_boxes: List[dict],
    tile_w: float,
    cam_ids: List[str] = CAM_IDS,
    cam_conf_thr: Dict[str, float] = CAMERA_CONF_THR,
) -> Dict[str, List[dict]]:
    num_cameras = len(cam_ids)
    bucketed: Dict[str, List[dict]] = {cam_id: [] for cam_id in cam_ids}
    for box in raw_boxes:
        center_x = (box["x1"] + box["x2"]) / 2.0
        cam_idx = int(center_x // tile_w)
        cam_idx = max(0, min(cam_idx, num_cameras - 1))
        cam_id = cam_ids[cam_idx]
        if box["conf"] < cam_conf_thr.get(cam_id, 0.0):
            continue
        bucketed[cam_id].append(box)
    return bucketed


def build_world_points_from_yolo(
    bucketed: Dict[str, List[dict]],
    poses: Dict[str, Pose],
    tile_w: float,
) -> Dict[str, List[dict]]:
    """各検出に world_xy, class_id, conf を持たせて返す。"""
    result: Dict[str, List[dict]] = {cid: [] for cid in CAM_IDS}
    for cam_idx, cam_id in enumerate(CAM_IDS):
        dets = bucketed.get(cam_id, [])
        if not dets:
            continue
        fx, fy, cx, cy, xi, alpha = get_intrinsics(cam_id)
        pose = poses[cam_id]

        for det in dets:
            center_x_full = (det["x1"] + det["x2"]) / 2.0
            center_y = (det["y1"] + det["y2"]) / 2.0
            local_x = center_x_full - cam_idx * tile_w
            height_m = CLASS_HEIGHT_M.get(det["class_id"], DEFAULT_HEIGHT_M)

            pts_px = np.array([[local_x, center_y]], dtype=np.float64)
            world_xy, valid = unproject_to_floor(
                pts_px, pose.R, pose.t, fx, fy, cx, cy, xi, alpha,
                floor_z=-height_m,
            )
            if not valid[0]:
                continue

            result[cam_id].append({
                "world_xy": world_xy[0],
                "class_id": det["class_id"],
                "conf": det["conf"],
            })
    return result


def draw_regions(canvas, regions: List[Region], scale: float):
    for region in regions:
        pts = (region.polygon_m * scale).astype(np.int32).reshape(-1, 1, 2)
        color = TYPE_COLORS_BGR.get(region.type, (200, 200, 200))
        closed = region.type not in ("wall", "gate")
        cv2.polylines(canvas, [pts], isClosed=closed, color=color,
                      thickness=2, lineType=cv2.LINE_AA)


def main():
    print("floorplan.json 読み込み中...")
    regions, cameras_json, real_w, real_h = load_floorplan(FLOORPLAN_JSON_PATH)

    print("カメラパラメータの上書き設定を適用中...")
    cameras_json = apply_camera_overrides(cameras_json)

    print("カメラ姿勢を解決中...")
    poses = resolve_poses(cameras_json, root="Area1", fmt=ROT_FMT)

    print(f"元画像を読み込み中: {SOURCE_IMAGE_PATH}")
    source_image = cv2.imread(SOURCE_IMAGE_PATH)
    if source_image is None:
        raise FileNotFoundError(f"画像を読み込めませんでした: {SOURCE_IMAGE_PATH}")
    img_w = source_image.shape[1]
    tile_w = img_w / NUM_CAMERAS

    print("YOLOモデル読み込み中...")
    model = YOLO(MODEL_PATH)

    print("推論を実行中...")
    results = model.predict(
        source=SOURCE_IMAGE_PATH,
        classes=YOLO_PREDICT_CLASSES,
        conf=YOLO_PREDICT_CONF,
        iou=YOLO_PREDICT_IOU,
        imgsz=YOLO_PREDICT_IMGSZ,
    )
    raw_boxes = yolo_results_to_raw_boxes(results[0])
    bucketed = bucket_detections_by_camera(raw_boxes, tile_w)

    for cam_id in CAM_IDS:
        print(f"  {cam_id}: {len(bucketed[cam_id])}件検出")

    per_camera = build_world_points_from_yolo(bucketed, poses, tile_w)

    # ------------------------------------------------------------------
    # BEV描画
    # ------------------------------------------------------------------
    scale = 40  # px per meter
    margin = 220
    w_px = int(real_w * scale) + margin
    h_px = int(real_h * scale) + 40
    canvas = np.zeros((h_px, w_px, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30)

    draw_regions(canvas, regions, scale)

    # カメラ位置
    for cid in CAM_IDS:
        p = poses[cid]
        px, py = int(p.t[0] * scale), int(p.t[1] * scale)
        if 0 <= px < w_px and 0 <= py < h_px:
            cv2.circle(canvas, (px, py), 9, (0, 0, 255), -1)
            cv2.circle(canvas, (px, py), 11, (255, 255, 255), 2)
            cv2.putText(canvas, cid, (px + 14, py - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # YOLO検出点(カメラ別に色分け。standing=塗りつぶし円、lying=四角マーカー)
    n_total = 0
    for cam_id, dets in per_camera.items():
        color = CAM_COLORS_BGR.get(cam_id, (200, 200, 200))
        for det in dets:
            wx, wy = det["world_xy"]
            px, py = int(wx * scale), int(wy * scale)
            if not (0 <= px < w_px and 0 <= py < h_px):
                continue
            n_total += 1
            if det["class_id"] == 1:  # standing
                cv2.circle(canvas, (px, py), 6, color, -1)
                cv2.circle(canvas, (px, py), 7, (0, 0, 0), 1)
            else:  # lying
                cv2.drawMarker(canvas, (px, py), color,
                               markerType=cv2.MARKER_SQUARE, markerSize=10, thickness=2)
            cv2.putText(canvas, f"{det['conf']:.2f}", (px + 6, py + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    # 凡例
    legend_items = [(cid, CAM_COLORS_BGR[cid]) for cid in CAM_IDS]
    legend_items += [
        ("standing (circle)", (255, 255, 255)),
        ("lying (square)", (255, 255, 255)),
    ]
    for i, (label, type_color) in enumerate(list(TYPE_COLORS_BGR.items())):
        legend_items.append((label, type_color))

    x0, y0 = int(real_w * scale) + 15, 20
    for i, (label, color) in enumerate(legend_items):
        y = y0 + i * 22
        cv2.rectangle(canvas, (x0, y), (x0 + 16, y + 16), color, -1)
        cv2.putText(canvas, label, (x0 + 24, y + 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (255, 255, 255), 1, cv2.LINE_AA)

    out_path = OUTPUT_DIR / "bev_yolo_detections.png"
    cv2.imwrite(str(out_path), canvas)
    print(f"\n描画した検出点数(枠外を除く): {n_total}")
    print(f"保存しました: {out_path}")


if __name__ == "__main__":
    main()