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
     ピクセル座標 -> 3D方向ベクトル(unproject) -> 床面(z=0)との交点(World XY) を計算
     ※ DSの内部パラメータ(fx,fy,cx,cy,xi,alpha)はカメラごとに個別に持てる
        (CAM_INTRINSICS で上書きされていないカメラは共通デフォルト値を使う)
  4. annotation.xml のバウンディングボックスから「接地点」
     (角度ベース + CURVE_GAMMA による非線形補正) を求め、その牛がどのカメラタイルに
     写っているかをx座標から判定し、タイル内ローカル座標に変換してから 3. の処理で
     World座標を得る
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
    "Camera_2": {"fx": 708.5, "fy": 735.2, "cx": 1472.0, "cy": 1470.1, "xi": 0.056105, "alpha": 0.421186},
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
    "Area1->Camera_1":   [7.2616, 6.4629, -4.2726, 0.3028, -1.247, -0.1977],
    "Camera_1->Camera_2": [13.5137, 0.0119, -0.47, -7.3981, 1.3065, -0.1517],
    "Camera_2->Camera_3": [13.3907, 0.167, 0.1008, 11.21, 1.5444, 1.0268],
} 

# 角度ベース(compute_foot_point_angular_gamma)のパラメータ
# theta_max_deg: blend が BLEND_MAX に達する入射角(度)
# BLEND_MAX: blendの最大値
# CURVE_GAMMA: tan(theta)比に対するべき乗補正(曲率)。gamma_sweepの結果、2.0が最良だった
THETA_MAX_DEG = 74.92
BLEND_MAX = 0.6033
CURVE_GAMMA = 2.0  # 描画に使う最終的なgamma値

# gammaを振って比較表示するための値リスト(比較表示のみに使用。実際の描画にはCURVE_GAMMAを使う)
GAMMA_VALUES = [0.5, 1.0, 1.5, 2.0]

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
# 3. Double Sphere unprojection -> 床面交点
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
    dirs_cam, valid = unproject_ds(pts_px, fx, fy, cx, cy, xi, alpha)
    dirs_world = (pose.R @ dirs_cam.T).T

    cam_pos = pose.t
    dz = dirs_world[:, 2]
    t_safe = np.abs(dz) > 1e-9
    t = np.where(t_safe, (floor_z - cam_pos[2]) / dz, 0.0)

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


def compute_incidence_angle(px: float, py: float, cx: float, cy: float,
                             fx: float, fy: float, xi: float, alpha: float) -> float:
    """
    Double Sphere (DS) モデルにおける入射角θ(光軸からの角度, ラジアン)を計算する。
    """
    mx = (px - cx) / fx
    my = (py - cy) / fy
    r2 = mx ** 2 + my ** 2

    inner = max(1.0 - (2.0 * alpha - 1.0) * r2, 0.0)
    denom = alpha * np.sqrt(inner) + (1.0 - alpha)
    if abs(denom) < 1e-9:
        denom = 1e-9
    mz = (1.0 - alpha ** 2 * r2) / denom

    scale = np.sqrt(mx ** 2 + my ** 2 + mz ** 2)
    dz = mz - xi * scale
    dr = np.sqrt(mx ** 2 + my ** 2)

    theta = np.arctan2(dr, dz)
    return float(theta)


def compute_foot_point_angular_gamma(local_xmin: float, local_xmax: float, ymin: float, ymax: float,
                                      cx: float, cy: float, fx: float, fy: float, xi: float, alpha: float,
                                      theta_max_deg: float = 75.0, blend_max: float = 0.6,
                                      curve_gamma: float = 1.0) -> Tuple[float, float]:
    """
    魚眼レンズの入射角θに基づいて接地点を計算する(tan(θ)ベース + CURVE_GAMMAによる曲率補正)。

    curve_gamma:
      1.0 : tan(theta)比に対して線形
      >1.0: 中心付近はblendを抑え、周辺(theta大)で急激に立ち上がる
      <1.0: 中心付近から早めにblendが効き始め、周辺では頭打ちに近づく
    """
    box_center_x = (local_xmin + local_xmax) / 2.0
    box_center_y = (ymin + ymax) / 2.0

    theta = compute_incidence_angle(box_center_x, box_center_y, cx, cy, fx, fy, xi, alpha)
    theta_max = np.radians(theta_max_deg)

    tan_theta = np.tan(min(theta, theta_max - 1e-6))
    tan_theta_max = np.tan(theta_max)
    ratio = min(tan_theta / tan_theta_max, 1.0)
    ratio = max(ratio, 0.0)

    dynamic_blend = blend_max * (ratio ** curve_gamma)

    # 画像中心に近い側の辺
    near_edge_x = local_xmax if box_center_x < cx else local_xmin
    near_edge_y = ymax if box_center_y < cy else ymin

    foot_x = box_center_x + dynamic_blend * (near_edge_x - box_center_x)
    foot_y = box_center_y + dynamic_blend * (near_edge_y - box_center_y)

    return foot_x, foot_y


def assign_cameras_and_footpoints(detections: List[CowDetection], img_w: int, num_cams: int):
    tile_w = img_w / num_cams
    for d in detections:
        cx_full = (d.xmin + d.xmax) / 2.0
        cam_idx = int(cx_full // tile_w)
        cam_idx = min(max(cam_idx, 0), num_cams - 1)
        d.cam_index = cam_idx
    return tile_w


# ---------------------------------------------------------------------------
# 4b. gammaスイープ(比較表示専用。描画には影響しない)
# ---------------------------------------------------------------------------

def run_one_gamma_stats(curve_gamma: float, poses: Dict[str, Pose],
                         detections_src: List[CowDetection], tile_w: float) -> dict:
    """
    指定したcurve_gammaでworld座標を計算し、複数カメラに写っている牛について
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

        pts_px = []
        for i in idxs:
            d = detections[i]
            local_xmin = d.xmin - cam_idx * tile_w
            local_xmax = d.xmax - cam_idx * tile_w
            foot_x, foot_y = compute_foot_point_angular_gamma(
                local_xmin, local_xmax, d.ymin, d.ymax,
                cx, cy, fx, fy, xi, alpha,
                theta_max_deg=THETA_MAX_DEG, blend_max=BLEND_MAX,
                curve_gamma=curve_gamma,
            )
            pts_px.append([foot_x, foot_y])
        pts_px = np.array(pts_px, dtype=np.float64)
        world_xy, valid = unproject_to_floor(pts_px, pose, fx, fy, cx, cy, xi, alpha, FLOOR_Z)

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
        "gamma": curve_gamma,
        "n_pairs": len(dists),
        "sum": float(dists.sum()) if len(dists) else 0.0,
        "mean": float(dists.mean()) if len(dists) else 0.0,
        "median": float(np.median(dists)) if len(dists) else 0.0,
        "max": float(dists.max()) if len(dists) else 0.0,
    }


def print_gamma_sweep(poses: Dict[str, Pose], detections: List[CowDetection], tile_w: float):
    print(f"\n{'gamma':>6} | {'n_pairs':>8} | {'sum(m)':>9} | {'mean(m)':>8} | {'median(m)':>10} | {'max(m)':>8}")
    print("-" * 64)
    results = []
    for gamma in GAMMA_VALUES:
        res = run_one_gamma_stats(gamma, poses, detections, tile_w)
        results.append(res)
        print(f"{res['gamma']:>6.2f} | {res['n_pairs']:>8d} | {res['sum']:>9.3f} | "
              f"{res['mean']:>8.4f} | {res['median']:>10.4f} | {res['max']:>8.3f}")

    best = min(results, key=lambda r: r["mean"])
    print(f"\n最も平均ズレが小さいgamma: {best['gamma']} (mean={best['mean']:.4f}m)")
    print(f"※ 現在描画に使用中のCURVE_GAMMA = {CURVE_GAMMA}")

    import csv
    out_csv = OUTPUT_DIR / "gamma_sweep_results.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["gamma", "n_pairs", "sum", "mean", "median", "max"])
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
    print(f"  接地点補正: THETA_MAX_DEG={THETA_MAX_DEG}, BLEND_MAX={BLEND_MAX}, CURVE_GAMMA={CURVE_GAMMA}")

    # --- gammaごとの比較表を表示(描画には影響しない。参考情報) ---
    print_gamma_sweep(poses, detections, tile_w)

    # カメラごとにグルーピングして一括unproject
    by_cam = defaultdict(list)
    for i, d in enumerate(detections):
        by_cam[d.cam_index].append(i)

    for cam_idx, idxs in by_cam.items():
        cam_id = CAM_IDS[cam_idx]
        pose = poses[cam_id]
        fx, fy, cx, cy, xi, alpha = get_intrinsics(cam_id)

        # 入射角(θ)ベース + CURVE_GAMMA補正の動的blendで接地点を計算
        pts_px = []
        for i in idxs:
            d = detections[i]
            local_xmin = d.xmin - cam_idx * tile_w
            local_xmax = d.xmax - cam_idx * tile_w
            foot_x, foot_y = compute_foot_point_angular_gamma(
                local_xmin, local_xmax, d.ymin, d.ymax,
                cx, cy, fx, fy, xi, alpha,
                theta_max_deg=THETA_MAX_DEG, blend_max=BLEND_MAX,
                curve_gamma=CURVE_GAMMA,
            )
            pts_px.append([foot_x, foot_y])
        pts_px = np.array(pts_px, dtype=np.float64)
        world_xy, valid = unproject_to_floor(pts_px, pose, fx, fy, cx, cy, xi, alpha, FLOOR_Z)

        for j, i in enumerate(idxs):
            detections[i].world_xy = world_xy[j]
            detections[i].valid = bool(valid[j])

    n_valid = sum(d.valid for d in detections)
    print(f"  床面に投影できた検出数: {n_valid} / {len(detections)}")

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
            "cow_name", "camera_id", "world_x_m", "world_y_m",
            "mean_x_m", "mean_y_m", "dist_from_mean_m",
            "bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"
        ])
        for name, dets in sorted(grouped.items()):
            mean_xy = merged_points[name]
            for d in sorted(dets, key=lambda x: x.cam_index):
                cam_id = CAM_IDS[d.cam_index]
                dist = float(np.linalg.norm(d.world_xy - mean_xy))
                writer.writerow([
                    name, cam_id,
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
            rows.append((dist, name, cam_id, d.world_xy, mean_xy))
    rows.sort(key=lambda r: -r[0])
    for dist, name, cam_id, wxy, mxy in rows[:10]:
        print(f"  {name:10s} {cam_id:10s} world=({wxy[0]:.2f},{wxy[1]:.2f}) "
              f"mean=({mxy[0]:.2f},{mxy[1]:.2f}) dist={dist:.3f}m")


if __name__ == "__main__":
    main()