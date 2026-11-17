"""
make_bev_adaptive.py

MooTrack360 (Farm1-Area1) の 3台の魚眼カメラで撮影された合成画像
(8832 x 2944 px = 2944px幅のタイル x3) に含まれる牛のバウンディングボックスを、
フロアプラン(floorplan.json)の座標系(メートル単位, 44m x 15m)に投影し、
鳥瞰図(BEV)上に表示する。

このファイルは元の make_bev.py (center方式のみ) をベースに、
「画像周辺部でオクルージョンにより牛の背中しか写っていない検出」に対応するための
アダプティブ方式(center方式 + 周辺部=遠い辺方式)を追加したバージョン。

--- アダプティブ方式(center方式 / 周辺部=遠い辺方式)について ---
画像中心(cx,cy)からの距離 r が小さい(=歪みが弱く、足元まできちんと写っている
可能性が高い)検出は、従来通り bbox中心を CENTER_HEIGHT_M(=牛の体高の半分)の
高さ平面に逆投影する center方式を使う。

一方、r が大きい(=歪みが強い周辺部で、手前の牛に遮られて背中側しか
写っていない可能性が高い)検出は、bbox中心ではなく「cx,cyから最も遠い辺の中点」
を牛の背中とみなし、COW_HEIGHT_M(=牛の体高そのもの)の高さ平面に逆投影する。

判定の境界半径は、カメラ1台あたりのタイル幅(tile_w)の1/4
(PERIPHERAL_RADIUS_RATIO=0.25)としている。この値はあくまで初期値であり、
実データで center_sweep_results.csv 的な検証(周辺部検出だけを対象にした
dist_from_mean_mの比較)を行って調整することを想定している。

なお、周辺部で座っている牛がいたとしても、座っている牛は視線が低くなる分、
手前の立っている牛に遮られてそもそも写らなくなる可能性が高いため、
「周辺部で背中だけ写っている検出 = ほぼ確実に立っている牛」という前提は
比較的破綻しにくいと考えられる。

--- 元のcenter方式 / 過去に検討したgamma・corner(relief)方式について ---
以前は接地点の推定方式として角度ベースの経験的ヒューリスティック(gamma方式)や
bboxの近い隅・遠い隅を使う方式(corner/relief方式)も比較検討していたが、
精度検証(dist_from_mean_mのスイープ比較)の結果、center方式が全検出に対して
一律適用した場合には最も誤差が小さかった。今回のアダプティブ方式は、
corner/relief方式的な発想を「周辺部の検出だけ」に限定して復活させたもの、
という位置づけになる。
"""

import json
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

import numpy as np
import cv2

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------

FLOORPLAN_JSON_PATH = "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/calibration/Farm2_Area_1_Artboard_config_.json"
ANNOTATION_XML_PATH = "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/annotation_process/3600_frames_result/annotations/000000.xml"
OUTPUT_DIR = Path("/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/calibration")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CAMERAS = 3
CAM_IDS = ["Camera_1", "Camera_2", "Camera_3"]

# Double Sphere (DS)モデル 内部パラメータ (共通デフォルト値)
DS_FX, DS_FY = 708.5, 735.2
DS_CX, DS_CY = 1472.0, 1470.1
DS_XI, DS_ALPHA = 0.006105, 0.48977

# ---------------------------------------------------------------------------
# ★ カメラ別の内部パラメータ上書き設定 ★
# ここに指定したカメラは、上の DS_* 共通値の代わりにこちらを使う。
# 指定がないカメラは共通デフォルト値 (DS_FX, DS_FY, DS_CX, DS_CY, DS_XI, DS_ALPHA) を使う。
#
# Camera_2: calibrate_camera2.py による再キャリブレーション結果 (5点フィット, feedbay2除外版)
#   mean誤差 1.570m -> 0.126m, max誤差 2.653m -> 0.245m まで改善したセット
# ---------------------------------------------------------------------------
CAM_INTRINSICS = {
    "Camera_2": {
        "fx": 708.5,
        "fy": 735.2,
        "cx": 1472.0,
        "cy": 1470.1,
        "xi": 0.056105,
        "alpha": 0.428780,
    },
    'Camera_3': {'fx': 708.5, 'fy': 735.2, 'cx': 1472.0, 'cy': 1470.1, 'xi': -0.143895, 'alpha': 0.533116}
}


def get_intrinsics(cam_id: str) -> Tuple[float, float, float, float, float, float]:
    """カメラIDに対応する内部パラメータ (fx, fy, cx, cy, xi, alpha) を返す。
    CAM_INTRINSICS に個別設定があればそちらを、無ければ共通デフォルト値を返す。"""
    if cam_id in CAM_INTRINSICS:
        p = CAM_INTRINSICS[cam_id]
        return p["fx"], p["fy"], p["cx"], p["cy"], p["xi"], p["alpha"]
    return DS_FX, DS_FY, DS_CX, DS_CY, DS_XI, DS_ALPHA


ROT_FMT = "euler_zyx_deg"
FLOOR_Z = 0.0

# ---------------------------------------------------------------------------
# ★ カメラ別の外部パラメータ(位置・姿勢)上書き設定 ★
# floorplan.json 内の cameras[].params を、ここに指定した値で上書きする。
# キーは "parent->child" の形式。指定がなければ floorplan.json の値をそのまま使う。
#
# Camera_1->Camera_2: calibrate_camera2.py による再キャリブレーション結果
#   (5点フィット, feedbay2除外版。内部パラメータ xi/alpha も上のCAM_INTRINSICSで上書き済み)
# ---------------------------------------------------------------------------
CAMERA_PARAM_OVERRIDES = {
    "Camera_1->Camera_2": [13.113, 0.1269, -0.0743, -7.2382, 1.3079, -0.8149],
    'Camera_2->Camera_3': [13.641, 0.1279, -1.6374, 11.0855, -0.3336, 0.7534],
}

# ---------------------------------------------------------------------------
# ★ 接地点計算(center方式)のパラメータ ★
# bboxの中心ピクセルを、床面(z=0)ではなく高さ CENTER_HEIGHT_M の水平面
# (z=-CENTER_HEIGHT_M)に逆投影する。牛の体高の半分を初期値とする。
# ---------------------------------------------------------------------------
COW_HEIGHT_M = 1.45  # 想定する乳牛の体高(肩高)。実測値があればそちらを優先すること
CENTER_HEIGHT_M = COW_HEIGHT_M / 2.0  # bbox中心が対応すると仮定する高さ(m)

# 高さ仮定を振って比較表示するための値リスト(比較表示のみに使用)
CENTER_HEIGHT_VALUES = [0.50, 0.60, 0.70, 0.725, 0.80, 0.90, 1.00]

# ---------------------------------------------------------------------------
# ★ アダプティブ方式(周辺部=遠い辺方式)のパラメータ ★
# 画像(タイル)幅 tile_w に対して、bbox中心と(cx,cy)の距離が
#   tile_w * PERIPHERAL_RADIUS_RATIO 以上
# の検出を「周辺部」とみなし、center方式ではなく遠い辺方式を使う。
#
# 遠い辺方式: bboxの4辺(上下左右)の中点のうち、(cx,cy)から最も遠い点を
# 牛の背中とみなし、z = -PERIPHERAL_HEIGHT_M(=COW_HEIGHT_M) の高さ平面に逆投影する。
# ---------------------------------------------------------------------------
PERIPHERAL_RADIUS_RATIO = 0.4            # tile_w に対する境界半径の比率(初期値。要検証)
PERIPHERAL_HEIGHT_M = COW_HEIGHT_M        # 周辺部で使う高さ仮定(牛の背中 = 体高そのもの)

TYPE_COLORS_BGR = {
    "wall":            (0, 200, 0),
    "cow_beds":        (0, 140, 255),
    "feeding_bay":     (255, 140, 0),
    "water":           (255, 255, 0),
    "gate":            (0, 0, 255),
    "milking_station": (255, 0, 255),
    "hay_area":        (0, 255, 255),
    "cow_brushes":     (255, 0, 180),
}

# カメラごとの点の色 (BGR) — per-camera BEVで各視点を区別するため
CAM_COLORS_BGR = {
    "Camera_1": (255, 80, 80),    # 青系
    "Camera_2": (80, 220, 80),    # 緑系
    "Camera_3": (80, 180, 255),   # 橙系
}


# ---------------------------------------------------------------------------
# データ構造
# ---------------------------------------------------------------------------

@dataclass
class Region:
    name: str
    type: str
    polygon_m: np.ndarray


@dataclass
class Pose:
    R: np.ndarray
    t: np.ndarray


@dataclass
class CowDetection:
    name: str
    cam_index: int
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    difficult: bool
    world_xy: Optional[np.ndarray] = None
    valid: bool = False
    peripheral: bool = False  # True: 周辺部(遠い辺)方式を使った検出


# ---------------------------------------------------------------------------
# 1. floorplan.json 読み込み
# ---------------------------------------------------------------------------

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
    """CAMERA_PARAM_OVERRIDES に指定されたカメラの params を上書きしたコピーを返す。
    元のcameras_json(floorplan.jsonから読み込んだもの)は変更しない。"""
    result = []
    for entry in cameras_json:
        key = f"{entry['parent']}->{entry['child']}"
        new_entry = dict(entry)
        if key in CAMERA_PARAM_OVERRIDES:
            new_entry = dict(entry)
            new_entry["params"] = list(CAMERA_PARAM_OVERRIDES[key])
            print(f"  [override] {key}: params -> {new_entry['params']}")
        result.append(new_entry)
    return result


# ---------------------------------------------------------------------------
# 2. カメラ姿勢の解決 (camera_pose_chain.py と同じロジック)
# ---------------------------------------------------------------------------

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

    if base_fmt == "euler_xyz":
        return _rx(a) @ _ry(b) @ _rz(c)
    elif base_fmt == "euler_xzy":
        return _rx(a) @ _rz(c) @ _ry(b)
    elif base_fmt == "euler_yxz":
        return _ry(a) @ _rx(b) @ _rz(c)
    elif base_fmt == "euler_yzx":
        return _ry(a) @ _rz(c) @ _rx(b)
    elif base_fmt == "euler_zxy":
        return _rz(a) @ _rx(b) @ _ry(c)
    elif base_fmt == "euler_zyx":
        return _rz(a) @ _ry(b) @ _rx(c)
    elif base_fmt == "rotvec":
        R, _ = cv2.Rodrigues(np.array([a, b, c], dtype=np.float64))
        return R
    raise ValueError(f"未知のフォーマット: {fmt}")


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
            t_world = poses[parent].t + t_local   # 位置は単純加算
            R_world = R_abs                       # 姿勢は単独絶対値
            poses[child] = Pose(R=R_world, t=t_world)
            remaining.remove(entry)
            progressed = True
        if not progressed:
            raise RuntimeError(f"解決できないカメラがあります: {[e['child'] for e in remaining]}")
    return poses


# ---------------------------------------------------------------------------
# 3. Double Sphere unprojection -> 高さ平面との交点
# ---------------------------------------------------------------------------

def unproject_ds(pts_px: np.ndarray, fx, fy, cx, cy, xi, alpha) -> Tuple[np.ndarray, np.ndarray]:
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


def unproject_to_floor(pts_px: np.ndarray, pose: Pose, fx, fy, cx, cy, xi, alpha,
                        floor_z: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    floor_z はスカラーでも、pts_px と同じ長さの配列(検出ごとに異なる高さ)でもよい。
    アダプティブ方式(center方式 / 周辺部方式)で検出ごとに高さを変えるのに使う。
    """
    dirs_cam, valid = unproject_ds(pts_px, fx, fy, cx, cy, xi, alpha)
    dirs_world = (pose.R @ dirs_cam.T).T

    floor_z_arr = np.asarray(floor_z, dtype=np.float64)
    if floor_z_arr.ndim == 0:
        floor_z_arr = np.full(len(pts_px), float(floor_z_arr))

    cam_pos = pose.t
    dz = dirs_world[:, 2]
    t_safe = np.abs(dz) > 1e-9
    t = np.where(t_safe, (floor_z_arr - cam_pos[2]) / dz, 0.0)

    valid = valid & t_safe & (t > 0)

    world_x = np.where(valid, cam_pos[0] + t * dirs_world[:, 0], 0.0)
    world_y = np.where(valid, cam_pos[1] + t * dirs_world[:, 1], 0.0)
    return np.stack([world_x, world_y], axis=-1), valid


# ---------------------------------------------------------------------------
# 4. XMLアノテーション読み込み + タイル分割 + 接地点算出
# ---------------------------------------------------------------------------

def load_annotation(path: str):
    tree = ET.parse(path)
    root = tree.getroot()
    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    detections: List[CowDetection] = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        difficult = obj.find("difficult").text == "1"
        bb = obj.find("bndbox")
        xmin = float(bb.find("xmin").text)
        ymin = float(bb.find("ymin").text)
        xmax = float(bb.find("xmax").text)
        ymax = float(bb.find("ymax").text)
        detections.append(CowDetection(name, -1, xmin, ymin, xmax, ymax, difficult))

    return detections, img_w, img_h


def compute_footpoints_center_batch(detections_idx: List[int], detections: List["CowDetection"],
                                     cam_idx: int, tile_w: float, pose: Pose,
                                     fx, fy, cx, cy, xi, alpha,
                                     center_height_m: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    bboxの中心ピクセルを、床面(z=0)ではなく高さ center_height_m の水平面
    (z = -center_height_m)に逆投影する方式(center方式)。
    CENTER_HEIGHT_VALUES のスイープ検証(print_center_sweep)専用に残してある。
    """
    pts_px = []
    for i in detections_idx:
        d = detections[i]
        local_xmin = d.xmin - cam_idx * tile_w
        local_xmax = d.xmax - cam_idx * tile_w
        box_center_x = (local_xmin + local_xmax) / 2.0
        box_center_y = (d.ymin + d.ymax) / 2.0
        pts_px.append([box_center_x, box_center_y])

    pts_px = np.array(pts_px, dtype=np.float64)
    world_xy, valid = unproject_to_floor(pts_px, pose, fx, fy, cx, cy, xi, alpha,
                                          floor_z=-center_height_m)
    return world_xy, valid


def compute_footpoints_adaptive_batch(detections_idx: List[int], detections: List["CowDetection"],
                                       cam_idx: int, tile_w: float, pose: Pose,
                                       fx, fy, cx, cy, xi, alpha) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    アダプティブ方式: 検出ごとに (cx,cy) からの距離 r で center方式 / 周辺部方式を切り替える。

    - r < tile_w * PERIPHERAL_RADIUS_RATIO (中心付近):
        従来通り bbox中心を CENTER_HEIGHT_M(体高の半分)の高さ平面に逆投影する
        (center方式)。歪みが弱く、足元まで含めた全身が写っている前提。

    - r >= tile_w * PERIPHERAL_RADIUS_RATIO (周辺部):
        (cx,cy)から最も遠い辺(上下左右のうちの1つ)の中点を「牛の背中」とみなし、
        PERIPHERAL_HEIGHT_M(=COW_HEIGHT_M、体高そのもの)の高さ平面に逆投影する。
        画像周辺部は歪みが強く、手前の牛に遮られて背中側しか写っていないことが
        多いという想定に基づく。bboxの角(corner)ではなく辺の中点を使うことで、
        体の横幅方向の広がり(高さ方向の情報と関係ないノイズ)の影響を打ち消す。

    戻り値: world_xy, valid, peripheral_mask (それぞれ detections_idx と同じ順)
    """
    radius_threshold = tile_w * PERIPHERAL_RADIUS_RATIO

    pts_px = []
    floor_z_list = []
    peripheral_mask = []

    for i in detections_idx:
        d = detections[i]
        local_xmin = d.xmin - cam_idx * tile_w
        local_xmax = d.xmax - cam_idx * tile_w
        local_ymin = d.ymin
        local_ymax = d.ymax
        box_center_x = (local_xmin + local_xmax) / 2.0
        box_center_y = (local_ymin + local_ymax) / 2.0

        r = float(np.hypot(box_center_x - cx, box_center_y - cy))

        if r < radius_threshold:
            # --- 中心付近: center方式 ---
            pts_px.append([box_center_x, box_center_y])
            floor_z_list.append(-CENTER_HEIGHT_M)
            peripheral_mask.append(False)
        else:
            # --- 周辺部: (cx,cy)から最も遠い辺の中点を背中とみなす ---
            edge_midpoints = [
                (box_center_x, local_ymin),   # 上辺の中点
                (box_center_x, local_ymax),   # 下辺の中点
                (local_xmin, box_center_y),   # 左辺の中点
                (local_xmax, box_center_y),   # 右辺の中点
            ]
            dists = [np.hypot(ex - cx, ey - cy) for ex, ey in edge_midpoints]
            far_edge = edge_midpoints[int(np.argmax(dists))]
            pts_px.append([far_edge[0], far_edge[1]])
            floor_z_list.append(-PERIPHERAL_HEIGHT_M)
            peripheral_mask.append(True)

    pts_px = np.array(pts_px, dtype=np.float64)
    floor_z_arr = np.array(floor_z_list, dtype=np.float64)

    world_xy, valid = unproject_to_floor(pts_px, pose, fx, fy, cx, cy, xi, alpha,
                                          floor_z=floor_z_arr)
    return world_xy, valid, np.array(peripheral_mask, dtype=bool)


def assign_cameras_and_footpoints(detections: List[CowDetection], img_w: int, num_cams: int):
    tile_w = img_w / num_cams
    for d in detections:
        cx_full = (d.xmin + d.xmax) / 2.0
        cam_idx = int(cx_full // tile_w)
        cam_idx = min(max(cam_idx, 0), num_cams - 1)
        d.cam_index = cam_idx
    return tile_w


# ---------------------------------------------------------------------------
# 4b. centerスイープ(bbox中心を仮定する高さを振って比較。比較表示専用)
# ---------------------------------------------------------------------------

def run_one_center_stats(center_height_m: float, poses: Dict[str, Pose],
                          detections_src: List[CowDetection], tile_w: float) -> dict:
    """
    指定したcenter_height_mでworld座標を計算し、複数カメラに写っている牛について
    平均点からのズレ(dist_from_mean_m)の統計を返す。draw用のdetectionsは変更しない。
    """
    detections = [
        CowDetection(d.name, d.cam_index, d.xmin, d.ymin, d.xmax, d.ymax, d.difficult)
        for d in detections_src
    ]

    by_cam = defaultdict(list)
    for i, d in enumerate(detections):
        by_cam[d.cam_index].append(i)

    for cam_idx, idxs in by_cam.items():
        cam_id = CAM_IDS[cam_idx]
        pose = poses[cam_id]
        fx, fy, cx, cy, xi, alpha = get_intrinsics(cam_id)

        world_xy, valid = compute_footpoints_center_batch(
            idxs, detections, cam_idx, tile_w, pose, fx, fy, cx, cy, xi, alpha, center_height_m
        )
        for j, i in enumerate(idxs):
            detections[i].world_xy = world_xy[j]
            detections[i].valid = bool(valid[j])

    grouped: Dict[str, List[CowDetection]] = defaultdict(list)
    for d in detections:
        if d.valid:
            grouped[d.name].append(d)

    dists = []
    for name, dets in grouped.items():
        if len(dets) < 2:
            continue
        xy = np.array([d.world_xy for d in dets])
        mean_xy = xy.mean(axis=0)
        for d in dets:
            dists.append(float(np.linalg.norm(d.world_xy - mean_xy)))

    dists = np.array(dists)
    return {
        "center_height_m": center_height_m,
        "n_pairs": len(dists),
        "sum": float(dists.sum()) if len(dists) else 0.0,
        "mean": float(dists.mean()) if len(dists) else 0.0,
        "median": float(np.median(dists)) if len(dists) else 0.0,
        "max": float(dists.max()) if len(dists) else 0.0,
    }


def print_center_sweep(poses: Dict[str, Pose], detections: List[CowDetection], tile_w: float):
    print(f"\n{'h(m)':>6} | {'n_pairs':>8} | {'sum(m)':>9} | {'mean(m)':>8} | {'median(m)':>10} | {'max(m)':>8}")
    print("-" * 62)
    results = []
    for h in CENTER_HEIGHT_VALUES:
        res = run_one_center_stats(h, poses, detections, tile_w)
        results.append(res)
        print(f"{res['center_height_m']:>6.3f} | {res['n_pairs']:>8d} | {res['sum']:>9.3f} | "
              f"{res['mean']:>8.4f} | {res['median']:>10.4f} | {res['max']:>8.3f}")

    best = min(results, key=lambda r: r["mean"])
    print(f"\n最も平均ズレが小さい高さ仮定: {best['center_height_m']}m (mean={best['mean']:.4f}m)")
    print(f"※ 現在描画に使用中のCENTER_HEIGHT_M = {CENTER_HEIGHT_M} (中心付近のみ。周辺部はPERIPHERAL_HEIGHT_M={PERIPHERAL_HEIGHT_M})")

    import csv
    out_csv = OUTPUT_DIR / "center_sweep_results.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["center_height_m", "n_pairs", "sum", "mean", "median", "max"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"保存しました: {out_csv}")
    return results


# ---------------------------------------------------------------------------
# 4c. アダプティブ方式(center方式+周辺部=遠い辺方式)の統計比較
#
# 4b.の center_sweep は「center方式のみを、高さ仮定を振って」比較するものだったが、
# こちらは実際に描画に使っているアダプティブ方式(compute_footpoints_adaptive_batch)
# の結果を、
#   ・全体(center+周辺部を混ぜた現在の本番ロジック)
#   ・center方式が適用された検出のみ
#   ・周辺部(遠い辺)方式が適用された検出のみ
# の3グループに分けて dist_from_mean_m の統計を出す。
#
# 各検出のdist_from_mean_mは、常に「その牛の全視点(center+周辺部混在)の平均点」
# を基準に計算する(本番のBEV描画・per_camera CSVと同じ基準)。そのうえで、
# 集計時にどちらの方式が適用された検出かでグループ分けするので、
# 「周辺部方式が実際にズレを悪化させているか」をそのまま数値で確認できる。
# ---------------------------------------------------------------------------

def run_adaptive_stats(poses: Dict[str, Pose], detections_src: List[CowDetection], tile_w: float) -> List[dict]:
    detections = [
        CowDetection(d.name, d.cam_index, d.xmin, d.ymin, d.xmax, d.ymax, d.difficult)
        for d in detections_src
    ]

    by_cam = defaultdict(list)
    for i, d in enumerate(detections):
        by_cam[d.cam_index].append(i)

    for cam_idx, idxs in by_cam.items():
        cam_id = CAM_IDS[cam_idx]
        pose = poses[cam_id]
        fx, fy, cx, cy, xi, alpha = get_intrinsics(cam_id)

        world_xy, valid, peripheral_mask = compute_footpoints_adaptive_batch(
            idxs, detections, cam_idx, tile_w, pose, fx, fy, cx, cy, xi, alpha
        )
        for j, i in enumerate(idxs):
            detections[i].world_xy = world_xy[j]
            detections[i].valid = bool(valid[j])
            detections[i].peripheral = bool(peripheral_mask[j])

    grouped: Dict[str, List[CowDetection]] = defaultdict(list)
    for d in detections:
        if d.valid:
            grouped[d.name].append(d)

    def collect_dists(filter_fn) -> np.ndarray:
        dists = []
        for name, dets in grouped.items():
            if len(dets) < 2:
                continue
            # 平均は常に全視点(center+周辺部混在)から計算する(本番の merged_points と同じ)
            xy_all = np.array([d.world_xy for d in dets])
            mean_xy = xy_all.mean(axis=0)
            for d in dets:
                if filter_fn(d):
                    dists.append(float(np.linalg.norm(d.world_xy - mean_xy)))
        return np.array(dists)

    def stats_dict(label: str, dists: np.ndarray) -> dict:
        return {
            "group": label,
            "n_pairs": len(dists),
            "sum": float(dists.sum()) if len(dists) else 0.0,
            "mean": float(dists.mean()) if len(dists) else 0.0,
            "median": float(np.median(dists)) if len(dists) else 0.0,
            "max": float(dists.max()) if len(dists) else 0.0,
        }

    dists_all = collect_dists(lambda d: True)
    dists_center = collect_dists(lambda d: not d.peripheral)
    dists_peripheral = collect_dists(lambda d: d.peripheral)

    return [
        stats_dict("全体(center+周辺部)", dists_all),
        stats_dict("center方式のみ", dists_center),
        stats_dict("周辺部(遠い辺)方式のみ", dists_peripheral),
    ]


def _display_width(s: str) -> int:
    """全角(F/W/A)文字を2、半角文字を1として、ターミナル上の表示幅を計算する。
    Python の len()/f-string の :< 幅指定は「文字数」しか見ないため、
    日本語(全角)とASCII(半角)が混在するラベルを同じ文字数指定で揃えても
    見た目の表示幅がバラバラになり、欄がズレて見える。これを防ぐためのヘルパー。
    """
    width = 0
    for ch in s:
        if unicodedata.east_asian_width(ch) in ("F", "W", "A"):
            width += 2
        else:
            width += 1
    return width


def _ljust_display(s: str, target_width: int) -> str:
    """表示幅ベースで左詰め(右側にスペースを埋める)する。全角文字を考慮した ljust。"""
    pad = max(target_width - _display_width(s), 0)
    return s + " " * pad


def print_adaptive_stats(poses: Dict[str, Pose], detections: List[CowDetection], tile_w: float):
    # group列は表示幅(全角=2, 半角=1)基準で揃える。最長ラベル
    # "周辺部(遠い辺)方式のみ" の表示幅が22なので、少し余裕を持たせて24にしている。
    group_col_width = 24

    print(f"\n{_ljust_display('group', group_col_width)} | {'n_pairs':>8} | {'sum(m)':>9} | "
          f"{'mean(m)':>8} | {'median(m)':>10} | {'max(m)':>8}")
    print("-" * 80)
    results = run_adaptive_stats(poses, detections, tile_w)
    for r in results:
        group_label = _ljust_display(r["group"], group_col_width)
        print(f"{group_label} | {r['n_pairs']:>8d} | {r['sum']:>9.3f} | "
              f"{r['mean']:>8.4f} | {r['median']:>10.4f} | {r['max']:>8.3f}")

    n_all = results[0]["n_pairs"]
    n_peripheral = results[2]["n_pairs"]
    if n_all > 0:
        print(f"\n※ 周辺部(遠い辺)方式が占める割合(距離集計対象のペア数ベース): "
              f"{n_peripheral}/{n_all} ({100.0 * n_peripheral / n_all:.1f}%)")
    print(f"※ 比較用: center方式のみ・高さ0.725m一律(元のcenter_sweepのCENTER_HEIGHT_M行)"
          f" と同じ基準で見たい場合は print_center_sweep の結果も参照。")

    import csv
    out_csv = OUTPUT_DIR / "adaptive_method_stats.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "n_pairs", "sum", "mean", "median", "max"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"保存しました: {out_csv}")
    return results


# ---------------------------------------------------------------------------
# 5. BEV描画
# ---------------------------------------------------------------------------

def draw_regions(canvas, regions: List[Region], scale: float):
    for region in regions:
        pts = (region.polygon_m * scale).astype(np.int32).reshape(-1, 1, 2)
        color = TYPE_COLORS_BGR.get(region.type, (200, 200, 200))
        closed = region.type not in ("wall", "gate")
        thickness = 2 if region.type in ("wall", "gate") else 2
        cv2.polylines(canvas, [pts], isClosed=closed, color=color,
                      thickness=thickness, lineType=cv2.LINE_AA)


def draw_legend(canvas, x0, y0):
    items = list(TYPE_COLORS_BGR.items()) + [("camera", (0, 0, 255)), ("cow (detected)", (255, 255, 255))]
    for i, (label, color) in enumerate(items):
        y = y0 + i * 22
        cv2.rectangle(canvas, (x0, y), (x0 + 16, y + 16), color, -1)
        cv2.putText(canvas, label, (x0 + 24, y + 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    print("floorplan.json 読み込み中...")
    regions, cameras_json, real_w, real_h = load_floorplan(FLOORPLAN_JSON_PATH)

    print("カメラパラメータの上書き設定を適用中...")
    cameras_json = apply_camera_overrides(cameras_json)

    print("カメラ姿勢を解決中...")
    poses = resolve_poses(cameras_json, root="Area1", fmt=ROT_FMT)
    print("DEBUG Camera_1 t =", poses["Camera_1"].t)
    for cid in CAM_IDS:
        p = poses[cid]
        fx, fy, cx, cy, xi, alpha = get_intrinsics(cid)
        override_tag = " [内部パラメータ上書きあり]" if cid in CAM_INTRINSICS else ""
        print(f"  {cid}: t={p.t.round(2)}  xi={xi} alpha={alpha}{override_tag}")

    print("annotation.xml 読み込み中...")
    detections, img_w, img_h = load_annotation(ANNOTATION_XML_PATH)
    tile_w = assign_cameras_and_footpoints(detections, img_w, NUM_CAMERAS)
    radius_threshold = tile_w * PERIPHERAL_RADIUS_RATIO
    print(f"  画像サイズ: {img_w}x{img_h} -> タイル幅 {tile_w:.1f}px x {NUM_CAMERAS}台")
    print(f"  検出数: {len(detections)}")
    print(f"  接地点計算方式: アダプティブ方式")
    print(f"    中心付近 (r < {radius_threshold:.1f}px = tile_w*{PERIPHERAL_RADIUS_RATIO}) : "
          f"center方式 (高さ{CENTER_HEIGHT_M}m)")
    print(f"    周辺部   (r >= {radius_threshold:.1f}px)                     : "
          f"遠い辺方式 (高さ{PERIPHERAL_HEIGHT_M}m)")

    # --- 現在のアダプティブ方式(center方式+周辺部=遠い辺方式)の統計を表示 ---
    # 全体 / center方式のみ / 周辺部(遠い辺)方式のみ、の3グループでdist_from_mean_mを集計する。
    # 周辺部方式が実際にズレを悪化させていないか、ここで確認できる。
    print("\n[比較] 現在のアダプティブ方式(center方式+周辺部=遠い辺方式)の内訳:")
    print_adaptive_stats(poses, detections, tile_w)

    # 参考: center方式のみを一律に使った場合の高さ仮定スイープ(旧来の比較表)。
    # 上のアダプティブ方式の「全体」行と見比べたい場合はコメントを外す。
    # print("\n[参考] center方式のみ(高さ仮定CENTER_HEIGHT_Mを振る、全検出対象):")
    # print_center_sweep(poses, detections, tile_w)

    # カメラごとにグルーピングして一括unproject
    by_cam = defaultdict(list)
    for i, d in enumerate(detections):
        by_cam[d.cam_index].append(i)

    n_peripheral_total = 0
    for cam_idx, idxs in by_cam.items():
        cam_id = CAM_IDS[cam_idx]
        pose = poses[cam_id]
        fx, fy, cx, cy, xi, alpha = get_intrinsics(cam_id)

        # アダプティブ方式: 中心付近はcenter方式、周辺部は遠い辺方式
        world_xy, valid, peripheral_mask = compute_footpoints_adaptive_batch(
            idxs, detections, cam_idx, tile_w, pose, fx, fy, cx, cy, xi, alpha
        )

        for j, i in enumerate(idxs):
            detections[i].world_xy = world_xy[j]
            detections[i].valid = bool(valid[j])
            detections[i].peripheral = bool(peripheral_mask[j])

        n_cam_peripheral = int(peripheral_mask.sum())
        n_peripheral_total += n_cam_peripheral
        print(f"  {cam_id}: 周辺部方式を適用した検出数 {n_cam_peripheral} / {len(idxs)}")

    n_valid = sum(d.valid for d in detections)
    print(f"  床面に投影できた検出数: {n_valid} / {len(detections)}")
    print(f"  周辺部(遠い辺)方式を適用した検出数(合計): {n_peripheral_total} / {len(detections)}")

    # 同一牛(name)の複数カメラ検出を平均化して1点にまとめる
    grouped: Dict[str, List[CowDetection]] = defaultdict(list)
    for d in detections:
        if d.valid:
            grouped[d.name].append(d)

    merged_points: Dict[str, np.ndarray] = {}
    for name, dets in grouped.items():
        xy = np.array([d.world_xy for d in dets])
        merged_points[name] = xy.mean(axis=0)

    print(f"  ユニークな牛の数: {len(merged_points)}")

    # ------------------------------------------------------------------
    # BEVキャンバス描画(平均化後)
    # ------------------------------------------------------------------
    scale = 40  # px per meter
    margin = 220  # 凡例用の余白
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

    # 牛の位置(接地点をフロアプランに投影したもの)
    for name, xy in merged_points.items():
        px, py = int(xy[0] * scale), int(xy[1] * scale)
        if 0 <= px < w_px and 0 <= py < h_px:
            cv2.circle(canvas, (px, py), 6, (255, 255, 255), -1)
            cv2.circle(canvas, (px, py), 7, (0, 0, 0), 1)
            cv2.putText(canvas, name.replace("cow", ""), (px + 6, py + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1, cv2.LINE_AA)

    draw_legend(canvas, int(real_w * scale) + 15, 20)

    out_path = OUTPUT_DIR / "bev_cow_positions.png"
    cv2.imwrite(str(out_path), canvas)
    print(f"保存しました: {out_path}")

    # 座標一覧もCSVで出力
    import csv
    csv_path = OUTPUT_DIR / "bev_cow_positions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["cow_name", "world_x_m", "world_y_m", "num_camera_views", "num_peripheral_views"])
        for name, xy in sorted(merged_points.items()):
            n_views = len(grouped[name])
            n_peripheral_views = sum(1 for d in grouped[name] if d.peripheral)
            writer.writerow([name, round(float(xy[0]), 3), round(float(xy[1]), 3), n_views, n_peripheral_views])
    print(f"保存しました: {csv_path}")

    # ------------------------------------------------------------------
    # カメラ別BEV描画 + CSV出力
    # (平均化する前の、各カメラ単体の投影結果を可視化してズレの原因を調査する)
    # center方式は丸、周辺部(遠い辺)方式は四角マーカーで区別する。
    # ------------------------------------------------------------------
    print("\nカメラ別BEVを作成中...")

    canvas_per_cam = np.zeros((h_px, w_px, 3), dtype=np.uint8)
    canvas_per_cam[:] = (30, 30, 30)
    draw_regions(canvas_per_cam, regions, scale)

    # カメラ位置を描画
    for cid in CAM_IDS:
        p = poses[cid]
        px, py = int(p.t[0] * scale), int(p.t[1] * scale)
        if 0 <= px < w_px and 0 <= py < h_px:
            cv2.circle(canvas_per_cam, (px, py), 9, (0, 0, 255), -1)
            cv2.circle(canvas_per_cam, (px, py), 11, (255, 255, 255), 2)
            cv2.putText(canvas_per_cam, cid, (px + 14, py - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # 各カメラの生の投影結果(平均前)をカメラごとの色で描画
    # + 平均点との間に線を引いて、どの視点がどれだけズレているか可視化
    for name, dets in grouped.items():
        mean_xy = merged_points[name]
        mpx, mpy = int(mean_xy[0] * scale), int(mean_xy[1] * scale)

        for d in dets:
            cam_id = CAM_IDS[d.cam_index]
            color = CAM_COLORS_BGR.get(cam_id, (200, 200, 200))
            px, py = int(d.world_xy[0] * scale), int(d.world_xy[1] * scale)

            if 0 <= px < w_px and 0 <= py < h_px:
                # 平均点からその視点の点へ線を引く(ズレの可視化)
                if 0 <= mpx < w_px and 0 <= mpy < h_px:
                    cv2.line(canvas_per_cam, (mpx, mpy), (px, py), color, 1, cv2.LINE_AA)
                if d.peripheral:
                    # 周辺部(遠い辺)方式: 四角マーカー
                    cv2.rectangle(canvas_per_cam, (px - 5, py - 5), (px + 5, py + 5), color, -1)
                    cv2.rectangle(canvas_per_cam, (px - 6, py - 6), (px + 6, py + 6), (0, 0, 0), 1)
                else:
                    # center方式: 丸マーカー
                    cv2.circle(canvas_per_cam, (px, py), 5, color, -1)
                    cv2.circle(canvas_per_cam, (px, py), 6, (0, 0, 0), 1)

        # 平均点(白)も表示
        if 0 <= mpx < w_px and 0 <= mpy < h_px:
            cv2.drawMarker(canvas_per_cam, (mpx, mpy), (255, 255, 255),
                           markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
            cv2.putText(canvas_per_cam, name.replace("cow", ""), (mpx + 8, mpy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1, cv2.LINE_AA)

    # 凡例(カメラ別の色 + 平均点マーカー + 方式の形状凡例)
    legend_items = [(cid, CAM_COLORS_BGR[cid]) for cid in CAM_IDS] + [("mean (x mark)", (255, 255, 255))]
    x0, y0 = int(real_w * scale) + 15, 20
    for i, (label, color) in enumerate(legend_items):
        y = y0 + i * 22
        cv2.rectangle(canvas_per_cam, (x0, y), (x0 + 16, y + 16), color, -1)
        cv2.putText(canvas_per_cam, label, (x0 + 24, y + 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
    shape_legend_y = y0 + len(legend_items) * 22 + 10
    cv2.circle(canvas_per_cam, (x0 + 8, shape_legend_y + 8), 5, (200, 200, 200), -1)
    cv2.putText(canvas_per_cam, "center method", (x0 + 24, shape_legend_y + 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(canvas_per_cam, (x0 + 3, shape_legend_y + 25), (x0 + 13, shape_legend_y + 35), (200, 200, 200), -1)
    cv2.putText(canvas_per_cam, "peripheral (far-edge)", (x0 + 24, shape_legend_y + 34), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (255, 255, 255), 1, cv2.LINE_AA)

    out_path_per_cam = OUTPUT_DIR / "bev_cow_positions_per_camera.png"
    cv2.imwrite(str(out_path_per_cam), canvas_per_cam)
    print(f"保存しました: {out_path_per_cam}")

    # カメラ別の生座標とズレ量(平均点からの距離)をCSVに出力
    csv_path_per_cam = OUTPUT_DIR / "bev_cow_positions_per_camera.csv"
    with open(csv_path_per_cam, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cow_name", "camera_id", "world_x_m", "world_y_m",
            "mean_x_m", "mean_y_m", "dist_from_mean_m",
            "method", "assumed_height_m",
            "bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"
        ])
        for name, dets in sorted(grouped.items()):
            mean_xy = merged_points[name]
            for d in sorted(dets, key=lambda x: x.cam_index):
                cam_id = CAM_IDS[d.cam_index]
                dist = float(np.linalg.norm(d.world_xy - mean_xy))
                method = "peripheral(far-edge)" if d.peripheral else "center"
                assumed_height = PERIPHERAL_HEIGHT_M if d.peripheral else CENTER_HEIGHT_M
                writer.writerow([
                    name, cam_id,
                    round(float(d.world_xy[0]), 3), round(float(d.world_xy[1]), 3),
                    round(float(mean_xy[0]), 3), round(float(mean_xy[1]), 3),
                    round(dist, 3),
                    method, assumed_height,
                    round(d.xmin, 1), round(d.ymin, 1), round(d.xmax, 1), round(d.ymax, 1),
                ])
    print(f"保存しました: {csv_path_per_cam}")

    # ズレが大きい順に上位を表示(どのカメラ・どの方式が原因か調査する手がかり)
    print("\n--- ズレ(平均からの距離)が大きい検出 TOP10 ---")
    rows = []
    for name, dets in grouped.items():
        mean_xy = merged_points[name]
        for d in dets:
            cam_id = CAM_IDS[d.cam_index]
            dist = float(np.linalg.norm(d.world_xy - mean_xy))
            rows.append((dist, name, cam_id, d.peripheral, d.world_xy, mean_xy))
    rows.sort(key=lambda r: -r[0])
    for dist, name, cam_id, peripheral, wxy, mxy in rows[:10]:
        method_tag = "peripheral" if peripheral else "center"
        print(f"  {name:10s} {cam_id:10s} [{method_tag:10s}] world=({wxy[0]:.2f},{wxy[1]:.2f}) "
              f"mean=({mxy[0]:.2f},{mxy[1]:.2f}) dist={dist:.3f}m")


if __name__ == "__main__":
    main()