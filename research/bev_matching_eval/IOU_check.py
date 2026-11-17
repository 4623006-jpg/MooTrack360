import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from scipy.optimize import linear_sum_assignment
from itertools import combinations

from ultralytics import YOLO

import numpy as np
import cv2
from shapely.geometry import Polygon


# ---------------------------------------------------------------------------
# 設定 (make_bev.py と同じ値を使う。変更した場合は両方に反映すること)
# ---------------------------------------------------------------------------

FLOORPLAN_JSON_PATH = "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/calibration/Farm2_Area_1_Artboard_config_.json"

# 3タイル合成済みのフレーム画像(annotation.xmlと同じフレーム)へのパス。
# ★ 実際のフレーム画像パスに書き換えること ★
SOURCE_IMAGE_PATH = "/mnt/vmlqnap02/dataset/MooTrack360/1-hour-farm1-area1-sample-sequence/1-hour-farm1-area1-frames/000000.jpg"
# 学習済みモデル (アノテーション時に使用したものと同じ重みに合わせる)
model = YOLO("/mnt/vmlqnap02/dataset/MooTrack360/Trained models/yolo11x/train_ultralytics/weights/best.pt")

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

# ---------------------------------------------------------------------------
# ★ パノラマ生成のパラメータ ★
# ---------------------------------------------------------------------------
BEV_PLANE_HEIGHT_M = 0.0     # 貼り合わせる平面のWorld Z(通常は床面=0.0)
BEV_SCALE_PX_PER_M = 60      # 出力パノラマの解像度(1mあたりのピクセル数)
DRAW_REGION_OVERLAY = False  # Trueにすると、フロアプランの領域を上から重ね描きする

TYPE_COLORS_BGR = {
    "wall": (0, 200, 0), "cow_beds": (0, 140, 255), "feeding_bay": (255, 140, 0),
    "water": (255, 255, 0), "gate": (0, 0, 255), "milking_station": (255, 0, 255),
    "hay_area": (0, 255, 255), "cow_brushes": (255, 0, 180),
}

# ---------------------------------------------------------------------------
# ★ YOLO検出の可視化用パラメータ (デバッグ用に追加) ★
# ---------------------------------------------------------------------------
CAM_COLORS_BGR = {
    "Camera_1": (0, 255, 0),    # 緑
    "Camera_2": (255, 0, 0),    # 青
    "Camera_3": (0, 0, 255),    # 赤
}
DETECTION_VIS_FILENAME = "yolo_detections_visualization.png"

# ---------------------------------------------------------------------------
# ★ YOLO推論パラメータ / カメラ(タイル)別confidenceしきい値 ★
# アノテーション作成時に使っていたロジック(右側=Camera_3は歪みが強く検出confidenceが
# 低く出るため、閾値を緩めていた)を踏襲する。
# YOLO自体の足切り(conf)は一番緩いしきい値より低くしておき、実際の採否は
# bucket_detections_by_camera側でカメラごとのしきい値を見て判定する。
# ---------------------------------------------------------------------------
YOLO_PREDICT_CONF = 0.01
YOLO_PREDICT_IOU = 0.5
YOLO_PREDICT_IMGSZ = 2560
YOLO_PREDICT_CLASSES = [0, 1]

CAMERA_CONF_THR = {
    "Camera_1": 0.5,
    "Camera_2": 0.5,
    "Camera_3": 0.2,
}


@dataclass
class Pose:
    R: np.ndarray
    t: np.ndarray


@dataclass
class Region:
    name: str
    type: str
    polygon_m: np.ndarray


@dataclass
class Bbox:
    cam_id: str
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    frame_id: int = -1
    det_id: int = -1
    conf: float = 1.0
    posture: Optional[str] = None


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


# ---------------------------------------------------------------------------
# Double Sphere 順投影/逆投影
# ---------------------------------------------------------------------------

def project_ds(points_cam: np.ndarray, fx, fy, cx, cy, xi, alpha) -> Tuple[np.ndarray, np.ndarray]:
    x = points_cam[..., 0]
    y = points_cam[..., 1]
    z = points_cam[..., 2]

    d1 = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    z_shift = xi * d1 + z
    d2 = np.sqrt(x ** 2 + y ** 2 + z_shift ** 2)
    denom = alpha * d2 + (1.0 - alpha) * z_shift

    valid = denom > 1e-6
    safe_denom = np.where(valid, denom, 1.0)

    u = fx * x / safe_denom + cx
    v = fy * y / safe_denom + cy
    return np.stack([u, v], axis=-1), valid


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


def world_to_camera(points_world: np.ndarray, pose: Pose) -> np.ndarray:
    """P_world = R @ P_cam + t  なので、 P_cam = R^T @ (P_world - t)。"""
    rel = points_world - pose.t
    return rel @ pose.R


# ---------------------------------------------------------------------------
# 2. YOLOの生検出データをカメラIDごとの辞書データへ変換する処理
# ---------------------------------------------------------------------------

def bucket_detections_by_camera(
    yolo_raw_boxes: List[dict],
    tile_w: float,
    cam_ids: List[str] = CAM_IDS,
    straddle_warn_ratio: float = 0.1,
    cam_conf_thr: Dict[str, float] = CAMERA_CONF_THR,
) -> Dict[str, List[dict]]:
    """
    3タイル合成画像に対するYOLO生検出(フル画像座標)を、x座標でカメラごとに
    バケツ分けする。tile_w は1カメラタイル分の幅(= フル画像幅 / カメラ数)。
    カメラごとに検出confidenceのしきい値が異なる(アノテーション時の運用を踏襲)ため、
    ここでカメラ振り分け後にカメラ別しきい値で採否を判定する。
    """
    num_cameras = len(cam_ids)
    bucketed: Dict[str, List[dict]] = {cam_id: [] for cam_id in cam_ids}

    for box in yolo_raw_boxes:
        center_x = (box["x1"] + box["x2"]) / 2.0
        cam_idx = int(center_x // tile_w)
        cam_idx = max(0, min(cam_idx, num_cameras - 1))  # 万一の範囲外はクランプ
        cam_id = cam_ids[cam_idx]

        # カメラ別confidenceしきい値による足切り
        if box["conf"] < cam_conf_thr.get(cam_id, 0.0):
            continue

        tile_left = cam_idx * tile_w
        tile_right = tile_left + tile_w
        outside = max(0.0, tile_left - box["x1"]) + max(0.0, box["x2"] - tile_right)
        box_width = box["x2"] - box["x1"]
        if box_width > 0 and outside / box_width > straddle_warn_ratio:
            print(f"[warn] bbox may straddle tile boundary: {box}")

        bucketed[cam_id].append(box)#ここで{"Camera1":Bbox "Camera2":Bbox}の形になる

    return bucketed


def process_frame_detections(
    detection_data: Dict[str, List[dict]],
    frame_id: int = -1,
) -> Dict[str, List[Bbox]]:
    """
    {cam_id: [box辞書, ...]} を {cam_id: [Bbox, ...]} に変換する。
    """
    frame_dict: Dict[str, List[Bbox]] = {}

    for cam_id, detections_data in detection_data.items():
        detection_list: List[Bbox] = []
        for det_id, det in enumerate(detections_data):
            box = Bbox(
                cam_id=cam_id,
                class_id=det["class_id"],
                x1=det["x1"],
                y1=det["y1"],
                x2=det["x2"],
                y2=det["y2"],
                frame_id=frame_id,
                det_id=det_id,
                conf=det.get("conf", 1.0),
            )
            detection_list.append(box)
        frame_dict[cam_id] = detection_list

    return frame_dict


# ---------------------------------------------------------------------------
# 3. bbox(4隅) -> World座標系のPolygon
# ---------------------------------------------------------------------------

def bbox_to_world_polygon(
    box: Bbox,
    pose: Pose,
    tile_w: float,
    cam_ids: List[str] = CAM_IDS,
) -> Optional[Polygon]:
    fx, fy, cx, cy, xi, alpha = get_intrinsics(box.cam_id)
    cam_idx = cam_ids.index(box.cam_id)

    corners_full = np.array([
        [box.x1, box.y1],
        [box.x2, box.y1],
        [box.x2, box.y2],
        [box.x1, box.y2],
    ])
    corners_local = corners_full.copy()
    corners_local[:, 0] -= cam_idx * tile_w

    world_xy, valid = unproject_to_floor(
        corners_local, pose.R, pose.t, fx, fy, cx, cy, xi, alpha,
        floor_z=BEV_PLANE_HEIGHT_M,
    )
    if not valid.all():
        return None

    poly = Polygon(world_xy.tolist())
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


# ---------------------------------------------------------------------------
# 4. フレーム内の全bboxに変換を適用する
# ---------------------------------------------------------------------------

def build_world_polygons_for_frame(
    frame_dict: Dict[str, List[Bbox]],
    poses: Dict[str, Pose],
    tile_w: float,
) -> Dict[str, List[Tuple[Bbox, Polygon]]]:
    result: Dict[str, List[Tuple[Bbox, Polygon]]] = {}
    for cam_id, boxes in frame_dict.items():
        pose = poses[cam_id]
        valid_pairs: List[Tuple[Bbox, Polygon]] = []
        dropped = 0
        for box in boxes:
            poly = bbox_to_world_polygon(box, pose, tile_w)
            if poly is None:
                dropped += 1
                continue
            valid_pairs.append((box, poly))
        if dropped:
            print(f"[warn] {cam_id}: {dropped}件のbboxをWorld変換できず除外")
        result[cam_id] = valid_pairs
    return result


# ---------------------------------------------------------------------------
# 5. カメラ間のIOU(bboxの閾値が一定値以上かどうか)
# ---------------------------------------------------------------------------

def polygon_iou(poly_a: Polygon, poly_b: Polygon) -> float:
    """2つのWorld Polygon間のIOUを返す。"""
    if poly_a is None or poly_b is None:
        return 0.0
    if not poly_a.is_valid or not poly_b.is_valid:
        return 0.0
    inter_area = poly_a.intersection(poly_b).area
    union_area = poly_a.union(poly_b).area
    return inter_area / union_area if union_area > 0 else 0.0


def is_same_individual(poly_a: Polygon, poly_b: Polygon, iou_threshold: float) -> bool:
    """IOUが閾値以上なら同一個体とみなす。"""
    return polygon_iou(poly_a, poly_b) >= iou_threshold

def match_camera_pair(pairs_i:List[Tuple[Bbox,Polygon]],pairs_j:List[Tuple[Bbox,Polygon]],iou_threshold)-> List[Tuple[Bbox,Bbox,float]]:
    n_i,n_j = len(pairs_i),len(pairs_j)
    if n_i == 0 or n_j ==0:
        return []
    iou_matrix = np.zeros((n_i,n_j))
    for a,(box_i,poly_i) in enumerate(pairs_i):
        for b,(box_j,poly_j) in enumerate(pairs_j):
            if box_i.class_id != box_j.class_id:
                continue
            iou_matrix[a,b] = polygon_iou(poly_i,poly_j)
    cost_matrix = 1.0 - iou_matrix
    row_idx, col_idx = linear_sum_assignment(cost_matrix)

    matches: List[Tuple[Bbox,Bbox,float]] = []
    for a, b in zip(row_idx, col_idx):
        iou = iou_matrix[a,b]
        if iou >= iou_threshold:
            matches.append((pairs_i[a][0], pairs_j[b][0], iou))
    return matches

def match_all_camera_pairs(world_polys: Dict[str,List[Tuple[Bbox,Polygon]]],iou_threshold: float) ->  Dict[Tuple[str,str],List[Tuple[Bbox,Bbox,float]]]:
    pairwise_matches = {}
    cam_ids = list(world_polys.keys())
    for cam_i, cam_j in combinations(cam_ids, 2):
        matches = match_camera_pair(world_polys[cam_i],world_polys[cam_j],iou_threshold)
        pairwise_matches[(cam_i,cam_j)] = matches
    return pairwise_matches

# ---------------------------------------------------------------------------
# ペアごとのマッチ結果を推移的に統合し，個体統合
# ---------------------------------------------------------------------------
def union_find_merge(world_polys:Dict[str, List[Tuple[Bbox, Polygon]]],pairwise_matches: Dict[Tuple[str,str],List[Tuple[Bbox, Bbox, float]]])->Dict[Tuple[str,int],int]:
    parent: Dict[Tuple[str,int],Tuple[str,int]] = {}
    for cam_id,pairs in world_polys.items():
        for box,_poly in pairs:
            key = (box.cam_id, box.det_id)
            parent[key] = key
    def find(x: Tuple[str,int]) -> Tuple[str,int]:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x: Tuple[str,int],y: Tuple[str,int]) -> None:
        rx, ry = find(x),find(y)
        if rx != ry:
            parent[rx] = ry
    for (_cam_i, _cam_j), matches in pairwise_matches.items():
        for box_i, box_j, _iou in matches:
            union((box_i.cam_id, box_i.det_id),(box_j.cam_id,box_j.det_id))

    root_to_individual_id: Dict[Tuple[str,int],int] = {}
    key_to_individual_id: Dict[Tuple[str,int],int] = {}
    next_id = 0
    for key in parent:
        root = find(key)
        if root not in root_to_individual_id:
            root_to_individual_id[root] = next_id
            next_id += 1
        key_to_individual_id[key] = root_to_individual_id[root]
    return key_to_individual_id

def group_by_individual(key_to_individual_id: Dict[Tuple[str,int],int]) -> Dict[int, List[Tuple[str,int]]]:
    groups: Dict[int, List[Tuple[str,int]]] = {}
    for key, individual_id in key_to_individual_id.items():
        groups.setdefault(individual_id,[]).append(key)
    return groups

def yolo_results_to_raw_boxes(results, conf_threshold: float = YOLO_PREDICT_CONF) -> List[dict]:
    """
    ここでの conf_threshold は最終的な採否のしきい値ではなく、YOLO自体の足切り(predict時のconf)
    と同じ最低ラインを想定している。カメラごとの実際のしきい値判定は
    bucket_detections_by_camera 側(CAMERA_CONF_THR)で行う。
    """
    boxes = results.boxes
    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    raw_boxes = []
    for (x1, y1, x2, y2) , c , cf in zip(xyxy,cls,conf):
        if cf < conf_threshold:
            continue
        raw_boxes.append({
            "x1": float(x1),"y1":float(y1),"x2":float(x2),"y2":float(y2),
            "class_id":int(c),"conf":float(cf)
        })
    return raw_boxes


# ---------------------------------------------------------------------------
# ★ YOLO検出結果の可視化・保存 (デバッグ用に追加) ★
# カメラごとに色分けしたbboxとタイル境界線、カメラ別検出数を描画して保存する。
# カメラ3の検出が少ない原因(検出漏れ/タイル境界ズレ/露出差など)を目視確認するため。
# ---------------------------------------------------------------------------

def save_detection_visualization(
    source_image: np.ndarray,
    bucketed: Dict[str, List[dict]],
    tile_w: float,
    output_dir: Path,
    filename: str = DETECTION_VIS_FILENAME,
) -> Path:
    vis = source_image.copy()
    h = vis.shape[0]

    # タイル境界線を描画
    for i in range(1, NUM_CAMERAS):
        x = int(i * tile_w)
        cv2.line(vis, (x, 0), (x, h), (255, 255, 255), 2)

    counts = {}
    for cam_id, boxes in bucketed.items():
        counts[cam_id] = len(boxes)
        color = CAM_COLORS_BGR.get(cam_id, (200, 200, 200))
        for box in boxes:
            p1 = (int(box["x1"]), int(box["y1"]))
            p2 = (int(box["x2"]), int(box["y2"]))
            cv2.rectangle(vis, p1, p2, color, 2)
            label = f"{cam_id} cls{box['class_id']} {box['conf']:.2f}"
            cv2.putText(vis, label, (p1[0], max(0, p1[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    summary = " / ".join(f"{cam_id}: {counts.get(cam_id, 0)}" for cam_id in CAM_IDS)
    cv2.putText(vis, summary, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    out_path = output_dir / filename
    cv2.imwrite(str(out_path), vis)
    print("カメラ別検出数:", counts)
    print(f"検出結果の可視化画像を保存しました: {out_path}")
    return out_path


def main():
    print("floorplan.json 読み込み中...")
    regions, cameras_json, real_w, real_h = load_floorplan(FLOORPLAN_JSON_PATH)

    print("カメラパラメータの上書き設定を適用中...")
    cameras_json = apply_camera_overrides(cameras_json)

    print("カメラ姿勢を解決中...")
    poses = resolve_poses(cameras_json, root="Area1", fmt=ROT_FMT)
    for cid in CAM_IDS:
        p = poses[cid]
        print(f"  {cid}: t={p.t.round(2)}")

    print(f"元画像を読み込み中: {SOURCE_IMAGE_PATH}")
    source_image = cv2.imread(SOURCE_IMAGE_PATH)
    if source_image is None:
        raise FileNotFoundError(f"画像を読み込めませんでした: {SOURCE_IMAGE_PATH}")

    results = model.predict(
        source=SOURCE_IMAGE_PATH,
        classes=YOLO_PREDICT_CLASSES,
        conf=YOLO_PREDICT_CONF,
        iou=YOLO_PREDICT_IOU,
        imgsz=YOLO_PREDICT_IMGSZ,
    )
    tile_w = source_image.shape[1]/NUM_CAMERAS
    raw_boxes = yolo_results_to_raw_boxes(results[0])
    bucketed = bucket_detections_by_camera(raw_boxes,tile_w)

    # ★ 追加: YOLO検出結果を画像に描画してcalibrationフォルダに保存
    save_detection_visualization(source_image, bucketed, tile_w, OUTPUT_DIR)

    frame_dict = process_frame_detections(bucketed,frame_id = 0)
    world_polys = build_world_polygons_for_frame(frame_dict,poses,tile_w)
    pairwise_matches = match_all_camera_pairs(world_polys, iou_threshold= 0.3)
    individual_ids = union_find_merge(world_polys, pairwise_matches)
    for iid, keys in sorted(group_by_individual(individual_ids).items()):
        print(f"個体ID={iid}: {keys}")


if __name__ == "__main__":
    # 実データ(floorplan.json / 元画像)が無い環境で動作確認したい場合は
    # main() の代わりに _smoke_test() を呼ぶ。
    main()
    # main()