"""
make_bev.py

MooTrack360 (Farm1-Area1) の 3台の魚眼カメラで撮影された合成画像
(8832 x 2944 px = 2944px幅のタイル x3) に含まれる牛のバウンディングボックスを、
フロアプラン(floorplan.json)の座標系(メートル単位, 44m x 15m)に投影し、
鳥瞰図(BEV)上に表示する。

手順:
  1. floorplan.json から semantic_regions と cameras(パラメータ) を読み込む
  2. camera_pose_chain.py と同じロジックでカメラの位置・姿勢(World座標系)を解決する
     - 位置: 単純加算 (t_world = t_parent + t_local)
     - 姿勢: 親と合成せず、そのカメラ単独の絶対姿勢として扱う
  3. Double Sphere (DS) 魚眼カメラモデルで各カメラ画像タイルの
     ピクセル座標 -> 3D方向ベクトル(unproject) -> 指定した高さ平面との交点(World XY) を計算
     ※ DSの内部パラメータ(fx,fy,cx,cy,xi,alpha)はカメラごとに個別に持てる
        (CAM_INTRINSICS で上書きされていないカメラは共通デフォルト値を使う)
  4. annotation.xml のバウンディングボックスから「接地点」を求める(center方式)。
     bboxの中心ピクセルを、床面(z=0)ではなく牛の体高相当の高さ平面に逆投影する。
     この高さは、annotation.xml の <pose> タグ(lying/standing)に応じて
     POSE_HEIGHT_M から取得する(pose未指定・未知の場合は CENTER_HEIGHT_M にフォールバック)。
     DSモデルの曲率(xi, alpha)は unproject_ds でレイ方向そのものに反映されているため、
     追加の経験的シフト処理は不要で、高さ平面を変えるだけで結果的にカメラ直下方向
     (画像中心)寄りの点になる。
     その牛がどのカメラタイルに写っているかをx座標から判定し、タイル内ローカル座標に
     変換してから 3. の処理でWorld座標を得る
  5. 同一牛(name)が複数カメラに写っている場合は平均をとって1点にする
  6. フロアプランの領域(壁・牛床・給餌エリア等)とカメラ位置、牛の位置を
     BEV画像として描画・保存する
  7. カメラ別の投影結果(平均前)も可視化し、視点ごとのズレを確認できるようにする

--- 2024 Camera_2 再キャリブレーションについて ---
calibrate_camera2.py にて、Camera_2 タイル内の6基準点(gate1・feeding_bay等の
既知の床面座標)を使い、Camera_2 の外部パラメータ(位置・姿勢)と内部パラメータ
(xi, alpha)を再フィッティングした。結果、xi が共通デフォルト値(0.006105)から
大きく外れていた(0.056105)ことが「中心付近と周辺で投影ズレの傾向が逆転する」
症状の主因と判明した。この結果を CAMERA_PARAM_OVERRIDES / CAM_INTRINSICS に反映する。

--- 備考: gamma方式 / corner(relief)方式について ---
以前は接地点の推定方式として角度ベースの経験的ヒューリスティック(gamma方式)や
bboxの近い隅・遠い隅を使う方式(corner/relief方式)も比較検討していたが、
精度検証(dist_from_mean_mのスイープ比較)の結果、center方式が最も誤差が小さかった
ため、本スクリプトではcenter方式のみを実装している。

--- 備考: 接地点の高さ仮定(pose別) ---
以前は CENTER_HEIGHT_M(= COW_HEIGHT_M / 2)を全検出に一律適用していたが、
実際には立位(standing)と伏臥位(lying)とで肩高相当の高さが大きく異なるため、
annotation.xml の <pose> タグを読み取り、POSE_HEIGHT_M の値を検出ごとに使い分ける
方式に変更した。CENTER_HEIGHT_M / CENTER_HEIGHT_VALUES による高さスイープは
比較検証用として引き続き残してある(固定高さで揃えた場合との比較に使う)。

--- 備考: ハイブリッド方式(center方式 + 奥辺方式)について(試験導入) ---
カメラ直下付近(ほぼ真上から見ている)ではbbox中心が体の中心付近に対応しやすいが、
カメラから離れた周辺部ではほぼ真横から見た状態になり、bboxの手前側の辺と奥側の辺は
体のまったく違う部位(手前=脚・腹側、奥=背中の稜線)に対応してしまうため、
bbox中心という「中間の適当な高さ」を仮定するcenter方式は物理的に無理が出てくる。

そこで、まずcenter方式で仮のworld座標を計算し、カメラ位置からの水平距離が
FAR_EDGE_DISTANCE_THRESHOLD_M を超える検出については、画像主点(cx, cy)から見て
bboxの上下左右4辺のうち最も遠い辺(=背中側の辺とみなす)の中点を新たに接地点候補とし、
高さも FAR_EDGE_HEIGHT_M(体高に近い値)に切り替えて再計算する「奥辺方式」に
差し替える。閾値以下の検出はcenter方式の結果をそのまま使う。

現時点では FAR_EDGE_DISTANCE_THRESHOLD_M・FAR_EDGE_HEIGHT_M はいずれも初期値の
試験導入であり、精度検証によるチューニングは今後の課題。
"""

import json
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
    'Camera_3' : {'fx': 708.5, 'fy': 735.2, 'cx': 1472.0, 'cy': 1470.1, 'xi': -0.143895, 'alpha': 0.533116}
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
# bboxの中心ピクセルを、床面(z=0)ではなく高さのある水平面(z=-height)に逆投影する。
#
# CENTER_HEIGHT_M: pose(lying/standing)が不明・未指定の検出に使うフォールバック値。
#                  牛の体高の半分を初期値とする。
# POSE_HEIGHT_M:   annotation.xml の <pose> タグ値ごとに使う高さ(m)。
#                  立位(standing)と伏臥位(lying)とでbbox中心が対応する実際の高さは
#                  大きく異なるため、一律のCENTER_HEIGHT_Mより実態に近い値を個別に持つ。
#                  <pose>の値はload_annotation()内で小文字化して保持しているので、
#                  キーは小文字("lying" / "standing")で指定する。
# ---------------------------------------------------------------------------
COW_HEIGHT_M = 1.45  # 想定する乳牛の体高(肩高)。実測値があればそちらを優先すること
CENTER_HEIGHT_M = COW_HEIGHT_M / 2.0  # pose不明時のフォールバック高さ(m)

POSE_HEIGHT_M = {
    "lying": 0.4,
    "standing": 0.75,
}

# 高さ仮定を振って比較表示するための値リスト(比較表示のみに使用。固定高さ版との比較用)
CENTER_HEIGHT_VALUES = [0.50, 0.60, 0.70, 0.725, 0.80, 0.90, 1.00]

# ---------------------------------------------------------------------------
# ★ ハイブリッド方式(奥辺方式への切り替え)のパラメータ ★(試験導入・初期値)
#
# FAR_EDGE_DISTANCE_THRESHOLD_M: center方式で計算した仮のworld座標が、カメラ位置
#   (XY平面上)からこの距離(m)を超えて離れている場合に「奥辺方式」へ切り替える。
# FAR_EDGE_HEIGHT_M: 奥辺方式で使う高さ(m)。画像主点から最も遠いbbox辺の中点は
#   背中の稜線に近いと仮定するため、体高そのものに近い値を使う。
#   POSE_HEIGHT_M(center方式用)に無いposeが来た場合は COW_HEIGHT_M にフォールバック。
# ---------------------------------------------------------------------------
FAR_EDGE_DISTANCE_THRESHOLD_M = 25.0

FAR_EDGE_HEIGHT_M = {
    "lying": 0.9,
    "standing": 1.8,
}

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
    pose: str = "unknown"
    world_xy: Optional[np.ndarray] = None
    valid: bool = False
    footpoint_method: str = "center"  # "center" または "far_edge"(ハイブリッド方式でどちらを使ったか)


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
    元の cameras_json(floorplan.jsonから読み込んだもの)は変更しない。"""
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
    pose(lying/standing)ごとに異なる接地高さを使うため、通常は配列で渡される。
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
# 4. XMLアノテーション読み込み + タイル分割 + 接地点算出(center方式)
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

        pose_elem = obj.find("pose")
        if pose_elem is not None and pose_elem.text:
            pose = pose_elem.text.strip().lower()
        else:
            pose = "unknown"

        bb = obj.find("bndbox")
        xmin = float(bb.find("xmin").text)
        ymin = float(bb.find("ymin").text)
        xmax = float(bb.find("xmax").text)
        ymax = float(bb.find("ymax").text)
        detections.append(CowDetection(name, -1, xmin, ymin, xmax, ymax, difficult, pose))

    return detections, img_w, img_h


def compute_footpoints_center_batch(detections_idx: List[int], detections: List["CowDetection"],
                                     cam_idx: int, tile_w: float, pose: Pose,
                                     fx, fy, cx, cy, xi, alpha,
                                     center_height_m: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    bboxの中心ピクセルを、床面(z=0)ではなく高さのある水平面(z = -height)に逆投影する方式。

    center_height_m が指定された場合は、その固定値を全検出に一律適用する
    (CENTER_HEIGHT_VALUESを振って比較するスイープ用)。
    指定されない(None)場合は、検出ごとの pose(lying/standing)に応じて
    POSE_HEIGHT_M から高さを取得する(POSE_HEIGHT_Mに無いposeはCENTER_HEIGHT_Mに
    フォールバック)。

    DSモデルの魚眼としての曲率(xi, alphaによるレイの曲がり方)は unproject_ds の
    時点でレイの方向そのものに反映済みなので、「画像中心寄りにシフトする」ための
    追加の経験的補正は不要。高さ0ではなく高さのある平面とレイを交差させるだけで、
    その分だけ(曲率も含めて自動的に)カメラ直下方向寄りの座標が得られる。
    """
    pts_px = []
    heights = []
    for i in detections_idx:
        d = detections[i]
        local_xmin = d.xmin - cam_idx * tile_w
        local_xmax = d.xmax - cam_idx * tile_w
        box_center_x = (local_xmin + local_xmax) / 2.0
        box_center_y = (d.ymin + d.ymax) / 2.0
        pts_px.append([box_center_x, box_center_y])

        if center_height_m is not None:
            heights.append(center_height_m)
        else:
            heights.append(POSE_HEIGHT_M.get(d.pose, CENTER_HEIGHT_M))

    pts_px = np.array(pts_px, dtype=np.float64)
    floor_z = -np.array(heights, dtype=np.float64)
    world_xy, valid = unproject_to_floor(pts_px, pose, fx, fy, cx, cy, xi, alpha,
                                          floor_z=floor_z)
    return world_xy, valid


def get_far_edge_point(local_xmin: float, local_ymin: float, local_xmax: float, local_ymax: float,
                        cx: float, cy: float) -> Tuple[float, float]:
    """
    画像主点(cx, cy)から見て、bboxの上下左右4辺のうち最も遠い辺(=背中側の辺とみなす)の
    中点を返す(奥辺方式)。

    主点からbbox中心へ向かう方向ベクトルを基準に、4辺の中点をその方向へ射影した値が
    最大になる辺を選ぶ(=主点から見て一番奥にある辺)。
    """
    box_center_x = (local_xmin + local_xmax) / 2.0
    box_center_y = (local_ymin + local_ymax) / 2.0
    dir_x = box_center_x - cx
    dir_y = box_center_y - cy
    norm = float(np.hypot(dir_x, dir_y))

    if norm < 1e-9:
        # bbox中心が主点の真下(方向不定)の場合はbbox中心をそのまま使う
        return box_center_x, box_center_y

    ux, uy = dir_x / norm, dir_y / norm

    edge_midpoints = [
        (box_center_x, local_ymin),  # 上辺
        (box_center_x, local_ymax),  # 下辺
        (local_xmin, box_center_y),  # 左辺
        (local_xmax, box_center_y),  # 右辺
    ]

    best_pt = edge_midpoints[0]
    best_proj = -np.inf
    for (ex, ey) in edge_midpoints:
        proj = (ex - cx) * ux + (ey - cy) * uy
        if proj > best_proj:
            best_proj = proj
            best_pt = (ex, ey)
    return best_pt


def compute_footpoints_hybrid_batch(detections_idx: List[int], detections: List["CowDetection"],
                                     cam_idx: int, tile_w: float, pose: Pose,
                                     fx, fy, cx, cy, xi, alpha
                                     ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    ハイブリッド方式(center方式 + 奥辺方式、試験導入)。

      1. まずcenter方式(pose別 POSE_HEIGHT_M)で仮のworld座標を計算する。
      2. カメラ位置(XY)からの水平距離が FAR_EDGE_DISTANCE_THRESHOLD_M を超える検出は、
         画像主点から最も遠い辺の中点を、pose別 FAR_EDGE_HEIGHT_M の高さ平面に
         逆投影した結果に差し替える(奥辺方式)。
      3. 閾値以下の検出はcenter方式の結果をそのまま使う。

    戻り値: world_xy, valid, footpoint_method (各検出が "center" / "far_edge" の
    どちらで計算されたかのラベル。CSV出力・デバッグ用)
    """
    # --- pass 1: center方式で仮の座標を計算 ---
    world_xy_center, valid_center = compute_footpoints_center_batch(
        detections_idx, detections, cam_idx, tile_w, pose, fx, fy, cx, cy, xi, alpha
    )

    cam_xy = pose.t[:2]
    dist_from_cam = np.linalg.norm(world_xy_center - cam_xy[None, :], axis=1)
    use_far_edge = valid_center & (dist_from_cam > FAR_EDGE_DISTANCE_THRESHOLD_M)

    world_xy = world_xy_center.copy()
    valid = valid_center.copy()
    footpoint_method = ["center"] * len(detections_idx)

    far_edge_local_idxs = np.nonzero(use_far_edge)[0]
    if len(far_edge_local_idxs) > 0:
        pts_px = []
        heights = []
        for local_i in far_edge_local_idxs:
            i = detections_idx[local_i]
            d = detections[i]
            local_xmin = d.xmin - cam_idx * tile_w
            local_xmax = d.xmax - cam_idx * tile_w
            ex, ey = get_far_edge_point(local_xmin, d.ymin, local_xmax, d.ymax, cx, cy)
            pts_px.append([ex, ey])
            heights.append(FAR_EDGE_HEIGHT_M.get(d.pose, COW_HEIGHT_M))

        pts_px = np.array(pts_px, dtype=np.float64)
        floor_z = -np.array(heights, dtype=np.float64)
        world_xy_far, valid_far = unproject_to_floor(pts_px, pose, fx, fy, cx, cy, xi, alpha,
                                                       floor_z=floor_z)

        for k, local_i in enumerate(far_edge_local_idxs):
            world_xy[local_i] = world_xy_far[k]
            valid[local_i] = valid_far[k]
            footpoint_method[local_i] = "far_edge"

    return world_xy, valid, footpoint_method


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
    指定したcenter_height_mを全検出に一律適用してworld座標を計算し、複数カメラに
    写っている牛について平均点からのズレ(dist_from_mean_m)の統計を返す。
    POSE_HEIGHT_Mによるpose別の高さ切り替えとは独立した、固定高さ版との比較用。
    draw用のdetectionsは変更しない。
    """
    detections = [
        CowDetection(d.name, d.cam_index, d.xmin, d.ymin, d.xmax, d.ymax, d.difficult, d.pose)
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
    print(f"\n最も平均ズレが小さい高さ仮定(固定値版): {best['center_height_m']}m (mean={best['mean']:.4f}m)")
    print(f"※ 実際の描画には POSE_HEIGHT_M = {POSE_HEIGHT_M} (pose別) を使用中"
          f" / フォールバック CENTER_HEIGHT_M = {CENTER_HEIGHT_M}")

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
    print(f"  画像サイズ: {img_w}x{img_h} -> タイル幅 {tile_w:.1f}px x {NUM_CAMERAS}台")
    print(f"  検出数: {len(detections)}")
    n_unknown_pose = sum(1 for d in detections if d.pose not in POSE_HEIGHT_M)
    print(f"  接地点計算方式: center方式 (pose別 POSE_HEIGHT_M={POSE_HEIGHT_M}, "
          f"フォールバック CENTER_HEIGHT_M={CENTER_HEIGHT_M}, pose不明/未対応の検出数={n_unknown_pose})")

    # --- 高さ仮定スイープの比較表を表示(描画には影響しない。参考情報) ---
    print("\n[比較] center方式(固定高さを振った場合。実際の描画はpose別POSE_HEIGHT_Mを使用):")
    print_center_sweep(poses, detections, tile_w)

    # カメラごとにグルーピングして一括unproject
    by_cam = defaultdict(list)
    for i, d in enumerate(detections):
        by_cam[d.cam_index].append(i)

    for cam_idx, idxs in by_cam.items():
        cam_id = CAM_IDS[cam_idx]
        pose = poses[cam_id]
        fx, fy, cx, cy, xi, alpha = get_intrinsics(cam_id)

        # ハイブリッド方式: center方式(pose別POSE_HEIGHT_M)を基本としつつ、
        # カメラから水平距離 FAR_EDGE_DISTANCE_THRESHOLD_M を超える検出は
        # 奥辺方式(pose別FAR_EDGE_HEIGHT_M)に切り替える(試験導入)
        world_xy, valid, footpoint_method = compute_footpoints_hybrid_batch(
            idxs, detections, cam_idx, tile_w, pose, fx, fy, cx, cy, xi, alpha
        )

        for j, i in enumerate(idxs):
            detections[i].world_xy = world_xy[j]
            detections[i].valid = bool(valid[j])
            detections[i].footpoint_method = footpoint_method[j]

    n_valid = sum(d.valid for d in detections)
    n_far_edge = sum(1 for d in detections if d.valid and d.footpoint_method == "far_edge")
    print(f"  床面に投影できた検出数: {n_valid} / {len(detections)}")
    print(f"  うち奥辺方式に切り替わった検出数: {n_far_edge} / {n_valid} "
          f"(閾値: カメラから{FAR_EDGE_DISTANCE_THRESHOLD_M}m超)")

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
        writer.writerow(["cow_name", "world_x_m", "world_y_m", "num_camera_views"])
        for name, xy in sorted(merged_points.items()):
            n_views = len(grouped[name])
            writer.writerow([name, round(float(xy[0]), 3), round(float(xy[1]), 3), n_views])
    print(f"保存しました: {csv_path}")

    # ------------------------------------------------------------------
    # カメラ別BEV描画 + CSV出力
    # (平均化する前の、各カメラ単体の投影結果を可視化してズレの原因を調査する)
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
                cv2.circle(canvas_per_cam, (px, py), 5, color, -1)
                cv2.circle(canvas_per_cam, (px, py), 6, (0, 0, 0), 1)

        # 平均点(白)も表示
        if 0 <= mpx < w_px and 0 <= mpy < h_px:
            cv2.drawMarker(canvas_per_cam, (mpx, mpy), (255, 255, 255),
                           markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
            cv2.putText(canvas_per_cam, name.replace("cow", ""), (mpx + 8, mpy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1, cv2.LINE_AA)

    # 凡例(カメラ別の色 + 平均点マーカー)
    legend_items = [(cid, CAM_COLORS_BGR[cid]) for cid in CAM_IDS] + [("mean (x mark)", (255, 255, 255))]
    x0, y0 = int(real_w * scale) + 15, 20
    for i, (label, color) in enumerate(legend_items):
        y = y0 + i * 22
        cv2.rectangle(canvas_per_cam, (x0, y), (x0 + 16, y + 16), color, -1)
        cv2.putText(canvas_per_cam, label, (x0 + 24, y + 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)

    out_path_per_cam = OUTPUT_DIR / "bev_cow_positions_per_camera.png"
    cv2.imwrite(str(out_path_per_cam), canvas_per_cam)
    print(f"保存しました: {out_path_per_cam}")

    # カメラ別の生座標とズレ量(平均点からの距離)をCSVに出力
    csv_path_per_cam = OUTPUT_DIR / "bev_cow_positions_per_camera.csv"
    with open(csv_path_per_cam, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cow_name", "camera_id", "pose", "footpoint_method", "world_x_m", "world_y_m",
            "mean_x_m", "mean_y_m", "dist_from_mean_m",
            "bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"
        ])
        for name, dets in sorted(grouped.items()):
            mean_xy = merged_points[name]
            for d in sorted(dets, key=lambda x: x.cam_index):
                cam_id = CAM_IDS[d.cam_index]
                dist = float(np.linalg.norm(d.world_xy - mean_xy))
                writer.writerow([
                    name, cam_id, d.pose, d.footpoint_method,
                    round(float(d.world_xy[0]), 3), round(float(d.world_xy[1]), 3),
                    round(float(mean_xy[0]), 3), round(float(mean_xy[1]), 3),
                    round(dist, 3),
                    round(d.xmin, 1), round(d.ymin, 1), round(d.xmax, 1), round(d.ymax, 1),
                ])
    print(f"保存しました: {csv_path_per_cam}")

    # ズレが大きい順に上位を表示(どのカメラが原因か調査する手がかり)
    print("\n--- ズレ(平均からの距離)が大きい検出 TOP10 ---")
    rows = []
    for name, dets in grouped.items():
        mean_xy = merged_points[name]
        for d in dets:
            cam_id = CAM_IDS[d.cam_index]
            dist = float(np.linalg.norm(d.world_xy - mean_xy))
            rows.append((dist, name, cam_id, d.footpoint_method, d.world_xy, mean_xy))
    rows.sort(key=lambda r: -r[0])
    for dist, name, cam_id, method, wxy, mxy in rows[:10]:
        print(f"  {name:10s} {cam_id:10s} [{method:9s}] world=({wxy[0]:.2f},{wxy[1]:.2f}) "
              f"mean=({mxy[0]:.2f},{mxy[1]:.2f}) dist={dist:.3f}m")

if __name__ == "__main__":
    main()