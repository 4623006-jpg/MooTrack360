"""
fit_cow_cylinder_model_ver1.py

目的: 「bboxを傾ける」のではなく、牛を3D形状としてモデル化し、そのモデルを
魚眼カメラで順投影(forward projection)した時のシルエット(bbox)が、観測
された(軸平行の)bboxに一致するように、牛の地面座標(X,Y)をフィッティング
する。bboxの傾き自体を直接操作するのではなく、3Dモデル→投影→bbox比較、
という間接的な方法で魚眼歪みの影響(動径方向の伸び等)を自然に考慮する。

モデル化:
  standing(立位): 鉛直な楕円柱。中心高さ STANDING_CENTER_HEIGHT_M、
                   半径 (STANDING_RADIUS_X_M, STANDING_RADIUS_Y_M)、
                   全高 STANDING_FULL_HEIGHT_M (床=0 〜 全高)
  lying(座位)   : 鉛直な円柱。中心高さ LYING_CENTER_HEIGHT_M、
                   半径 LYING_RADIUS_M、全高 LYING_FULL_HEIGHT_M

--- 重要: 座標系の符号規則について ---
Mootrack_core.unproject_to_floor()は「ピクセル→世界座標(床面)」の逆投影
関数だが、本ファイルで必要な「世界座標→ピクセル」の順投影関数はおそらく
Mootrack_core側に無いため、Double Sphereモデルの順投影式(Usenko et al.
2018)をこのファイル内に独自実装している。

R, tの意味(カメラ姿勢をワールド座標系でどう表すか)や、floor_z=-height_m
が「地面からheight_m上」をどちら符号で表しているか、といった符号規則は、
これまでの会話から類推して実装しているが、実際のMootrack_core.pyの実装と
食い違っている可能性がある。そのため main() の最初に必ず

    「既知の世界座標(X,Y,既知の高さ)を本ファイルの順投影で画像座標に変換し、
     その画像座標をMootrack_core.unproject_to_floor()で逆投影して
     元の(X,Y)に戻るか」

というラウンドトリップ確認を行う。ズレが大きい場合は CONVENTION セクションの
Z_SIGN / TRANSPOSE_R を反転させて再実行すること。

やること:
  1. 牛を鉛直シリンダーとしてモデル化する(上記)。
  2. シリンダー表面上の多数の点(複数の高さ×複数の角度)を世界座標で生成し、
     Double Sphereモデルで画像平面に順投影する。
  3. 投影された点群の軸平行bboxを計算し、観測bboxとの差を最小化するように
     (X, Y)をscipy.optimize.least_squaresでフィッティングする。
     初期値はunproject_to_floor(bbox中心)の結果を使う。
  4. フィッティング結果と、従来のbbox中心逆投影結果(ベースライン)を比較する。
  5. フィッティングしたモデルの投影シルエット(各高さの輪をワイヤーフレーム
     状に描画)を元の実画像上に重ねて保存する
     (「3Dモデルをbboxに重ね合わせる」の可視化)。

--- 使い方 ---
    1. 設定セクションのパスを実際の環境に合わせて書き換える
    2. TARGET_DETECTIONSに確認したい検出(フレーム・カメラ・bbox・pose)を指定
    3. python fit_cow_cylinder_model_ver1.py
    4. 最初に出力されるラウンドトリップ確認の誤差が大きければ、
       CONVENTIONのZ_SIGN/TRANSPOSE_Rを反転して再実行する
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares

MOOTRACK_CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calibration")
if MOOTRACK_CODE_DIR not in sys.path:
    sys.path.insert(0, MOOTRACK_CODE_DIR)

from Mootrack_core import (
    ROT_FMT,
    get_intrinsics, load_floorplan, apply_camera_overrides, resolve_poses,
    unproject_to_floor,
)

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------

FLOORPLAN_JSON_PATH = "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/calibration/Farm2_Area_1_Artboard_config_.json"
IMAGE_DIR = "/mnt/vmlqnap02/dataset/MooTrack360/1-hour-farm1-area1-sample-sequence/1-hour-farm1-area1-frames"
OUTPUT_DIR = Path(
    "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/research/idf1_check"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OVERLAY_OUT_DIR = OUTPUT_DIR / "cylinder_model_overlay"
OVERLAY_OUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_WIDTH_PX = 8832
NUM_CAMERAS = 3
CAM_IDS = ["Camera_1", "Camera_2", "Camera_3"]

# --- 牛の3Dモデルパラメータ(実測値が分かっていれば書き換える) ---
STANDING_CENTER_HEIGHT_M = 0.75
STANDING_FULL_HEIGHT_M = 1.5          # 床(0m)〜背中上(1.5m)
STANDING_RADIUS_X_M = 0.35            # 体幅方向の半径
STANDING_RADIUS_Y_M = 0.35            # (向き不明のため等方円柱からスタート)

LYING_CENTER_HEIGHT_M = 0.45
LYING_FULL_HEIGHT_M = 0.9
LYING_RADIUS_M = 0.55                 # 座ると横に広がるため半径を大きめに

N_ANGLES = 24
N_HEIGHT_RINGS = 4

# --- 座標系の符号規則(要確認。ラウンドトリップ確認の結果に応じて調整) ---
Z_SIGN = -1.0        # 「地面より上」をワールドZの負方向として表す場合は-1
TRANSPOSE_R = True   # p_cam = R^T @ (p_world - t) を使うか(Falseなら R @ (p_world - t))

# --- 確認・フィッティング対象の検出(実際のデータに合わせて書き換える) ---
# frame_idx, cam_id, local_bbox=(x1,y1,x2,y2), pose("standing"/"lying")
TARGET_DETECTIONS: List[dict] = [
    {"frame_idx": 100, "cam_id": "Camera_2", "local_bbox": (1200.0, 800.0, 1500.0, 1100.0), "pose": "standing"},
]

ROUNDTRIP_TEST_POINT_XY = (2.0, 3.0)   # ラウンドトリップ確認用の適当な世界座標(床上)
ROUNDTRIP_TEST_HEIGHT_M = 0.75


def frame_name(idx: int) -> str:
    return f"{idx:06d}"


# ---------------------------------------------------------------------------
# Double Sphereモデルの順投影(Usenko et al. 2018)
# ---------------------------------------------------------------------------

def project_point_fisheye(p_cam: np.ndarray, fx: float, fy: float, cx: float, cy: float,
                           xi: float, alpha: float) -> Tuple[float, float, bool]:
    """カメラ座標系の3D点(x,y,z)を、Double Sphereモデルで画像座標(u,v)に
    順投影する。zはカメラの光軸方向(前方)を想定。"""
    x, y, z = float(p_cam[0]), float(p_cam[1]), float(p_cam[2])
    d1 = np.sqrt(x * x + y * y + z * z)
    if d1 < 1e-9:
        return 0.0, 0.0, False
    zbar = xi * d1 + z
    d2 = np.sqrt(x * x + y * y + zbar * zbar)
    denom = alpha * d2 + (1.0 - alpha) * zbar
    if denom < 1e-9:
        return 0.0, 0.0, False
    u = fx * x / denom + cx
    v = fy * y / denom + cy
    # 有効性チェック(Usenko et al.のw1,w2による可視領域条件の簡易版)
    if alpha > 0.5:
        w1 = (1.0 - alpha) / alpha
    else:
        w1 = alpha / (1.0 - alpha)
    w2 = (w1 + xi) / np.sqrt(2.0 * w1 * xi + xi * xi + 1.0) if (2.0 * w1 * xi + xi * xi + 1.0) > 0 else 1.0
    valid = z > -w2 * d1
    return u, v, valid


def world_to_cam(p_world: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """ワールド座標点をカメラ座標系に変換する。p_world = R @ p_cam + t という
    規則を想定(TRANSPOSE_R=Trueならp_cam = R^T @ (p_world - t))。"""
    diff = np.asarray(p_world, dtype=np.float64) - np.asarray(t, dtype=np.float64)
    if TRANSPOSE_R:
        return R.T @ diff
    return R @ diff


def project_world_points(pts_world: np.ndarray, R: np.ndarray, t: np.ndarray,
                          fx: float, fy: float, cx: float, cy: float,
                          xi: float, alpha: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    us, vs, valids = [], [], []
    for p in pts_world:
        p_cam = world_to_cam(p, R, t)
        u, v, valid = project_point_fisheye(p_cam, fx, fy, cx, cy, xi, alpha)
        us.append(u)
        vs.append(v)
        valids.append(valid)
    return np.array(us), np.array(vs), np.array(valids, dtype=bool)


# ---------------------------------------------------------------------------
# 牛の3D円柱/楕円柱モデル
# ---------------------------------------------------------------------------

def cylinder_model_params(pose: str) -> Tuple[float, Tuple[float, float], float]:
    if pose == "lying":
        return LYING_CENTER_HEIGHT_M, (LYING_RADIUS_M, LYING_RADIUS_M), LYING_FULL_HEIGHT_M
    return STANDING_CENTER_HEIGHT_M, (STANDING_RADIUS_X_M, STANDING_RADIUS_Y_M), STANDING_FULL_HEIGHT_M


def cylinder_sample_points(X: float, Y: float, pose: str,
                            n_angles: int = N_ANGLES, n_height_rings: int = N_HEIGHT_RINGS) -> np.ndarray:
    """牛を表す鉛直シリンダー表面上の点群を世界座標で生成する。"""
    _, (rx, ry), full_h = cylinder_model_params(pose)
    heights = np.linspace(0.0, full_h, n_height_rings)
    angles = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    pts = []
    for h in heights:
        z_world = Z_SIGN * h
        for a in angles:
            px = X + rx * np.cos(a)
            py = Y + ry * np.sin(a)
            pts.append((px, py, z_world))
    return np.array(pts, dtype=np.float64)


def model_bbox(X: float, Y: float, pose: str, R: np.ndarray, t: np.ndarray,
               fx: float, fy: float, cx: float, cy: float, xi: float, alpha: float
               ) -> Optional[Tuple[float, float, float, float]]:
    pts = cylinder_sample_points(X, Y, pose)
    us, vs, valid = project_world_points(pts, R, t, fx, fy, cx, cy, xi, alpha)
    if valid.sum() < 3:
        return None
    us, vs = us[valid], vs[valid]
    return float(us.min()), float(vs.min()), float(us.max()), float(vs.max())


def fit_xy_to_bbox(observed_bbox: Tuple[float, float, float, float], pose: str,
                    R: np.ndarray, t: np.ndarray, fx: float, fy: float, cx: float, cy: float,
                    xi: float, alpha: float, init_xy: Tuple[float, float]
                    ) -> Tuple[float, float, Optional[Tuple[float, float, float, float]]]:
    """モデルのbboxが観測bboxに最も近くなるように(X,Y)を最適化する。"""

    def residuals(params):
        X, Y = params
        mb = model_bbox(X, Y, pose, R, t, fx, fy, cx, cy, xi, alpha)
        if mb is None:
            return np.array([1e3, 1e3, 1e3, 1e3])
        return np.array(mb) - np.array(observed_bbox)

    result = least_squares(residuals, x0=list(init_xy), method="lm", max_nfev=2000)
    X_fit, Y_fit = result.x
    final_bbox = model_bbox(X_fit, Y_fit, pose, R, t, fx, fy, cx, cy, xi, alpha)
    return float(X_fit), float(Y_fit), final_bbox


# ---------------------------------------------------------------------------
# ラウンドトリップ確認(符号規則が合っているかのチェック)
# ---------------------------------------------------------------------------

def roundtrip_check(poses, cam_id: str) -> None:
    fx, fy, cx, cy, xi, alpha = get_intrinsics(cam_id)
    cam_pose = poses[cam_id]
    X0, Y0 = ROUNDTRIP_TEST_POINT_XY
    p_world = np.array([X0, Y0, Z_SIGN * ROUNDTRIP_TEST_HEIGHT_M])

    p_cam = world_to_cam(p_world, cam_pose.R, cam_pose.t)
    u, v, valid = project_point_fisheye(p_cam, fx, fy, cx, cy, xi, alpha)
    print(f"[ラウンドトリップ確認] 世界座標({X0},{Y0}, h={ROUNDTRIP_TEST_HEIGHT_M}) "
          f"-> 順投影(本ファイル実装) -> pixel=({u:.1f},{v:.1f}) valid={valid}")

    if not valid:
        print("  !! 順投影がvalid=Falseを返した。Z_SIGN/TRANSPOSE_Rの設定を見直してください。")
        return

    pts_px = np.array([[u, v]], dtype=np.float64)
    world_xy, ok = unproject_to_floor(
        pts_px, cam_pose.R, cam_pose.t, fx, fy, cx, cy, xi, alpha,
        floor_z=-ROUNDTRIP_TEST_HEIGHT_M,
    )
    if not ok[0]:
        print("  !! unproject_to_floorがvalid=Falseを返した。")
        return

    err = np.linalg.norm(world_xy[0] - np.array([X0, Y0]))
    print(f"  -> 逆投影(Mootrack_core.unproject_to_floor) -> 世界座標={world_xy[0]} "
          f"(元の値との誤差={err:.4f}m)")
    if err > 0.05:
        print("  !! 誤差が大きい。CONVENTIONセクションのZ_SIGN/TRANSPOSE_Rを反転して"
              "再実行してください(例: Z_SIGN *= -1 または TRANSPOSE_R = not TRANSPOSE_R)。")
    else:
        print("  -> OK: 符号規則は整合しているようです。")


# ---------------------------------------------------------------------------
# 可視化: フィッティングした3Dモデルのシルエットを実画像に重ねて保存
# ---------------------------------------------------------------------------

def save_overlay_image(frame_idx: int, cam_id: str, local_bbox, pose: str,
                        X_fit: float, Y_fit: float, R: np.ndarray, t: np.ndarray,
                        fx: float, fy: float, cx: float, cy: float, xi: float, alpha: float) -> Optional[str]:
    image_path = Path(IMAGE_DIR) / f"{frame_name(frame_idx)}.jpg"
    if not image_path.exists():
        print(f"  (画像が見つからないためオーバーレイをスキップ: {image_path})")
        return None
    full_img = cv2.imread(str(image_path))
    if full_img is None:
        return None

    tile_w = IMAGE_WIDTH_PX / NUM_CAMERAS
    cam_idx = CAM_IDS.index(cam_id)
    x0 = int(cam_idx * tile_w)
    x1 = int((cam_idx + 1) * tile_w)
    tile_img = full_img[:, x0:x1].copy()

    # 観測bbox(白)
    bx1, by1, bx2, by2 = [int(round(v)) for v in local_bbox]
    cv2.rectangle(tile_img, (bx1, by1), (bx2, by2), (255, 255, 255), 3, cv2.LINE_AA)

    # フィッティングした3Dモデルの各高さの輪を描画(緑)
    _, _, full_h = cylinder_model_params(pose)
    heights = np.linspace(0.0, full_h, N_HEIGHT_RINGS)
    angles = np.linspace(0.0, 2.0 * np.pi, N_ANGLES, endpoint=False)
    _, (rx, ry), _ = cylinder_model_params(pose)

    for h in heights:
        ring_pts = []
        for a in angles:
            px = X_fit + rx * np.cos(a)
            py = Y_fit + ry * np.sin(a)
            p_world = np.array([px, py, Z_SIGN * h])
            p_cam = world_to_cam(p_world, R, t)
            u, v, valid = project_point_fisheye(p_cam, fx, fy, cx, cy, xi, alpha)
            if valid:
                ring_pts.append((int(round(u)), int(round(v))))
        if len(ring_pts) >= 3:
            pts_arr = np.array(ring_pts, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(tile_img, [pts_arr], isClosed=True, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

    cv2.putText(tile_img, f"{frame_name(frame_idx)} {cam_id} pose={pose} "
                           f"(white=observed bbox, green=fitted 3D model)",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

    out_path = OVERLAY_OUT_DIR / f"{frame_name(frame_idx)}_{cam_id}_cylinder_overlay.jpg"
    # 確認しやすいよう、bbox周辺を拡大して保存
    margin = 400
    cx0 = max(0, bx1 - margin)
    cy0 = max(0, by1 - margin)
    cx1 = min(tile_img.shape[1], bx2 + margin)
    cy1 = min(tile_img.shape[0], by2 + margin)
    crop = tile_img[cy0:cy1, cx0:cx1]
    cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return str(out_path)


def main():
    print("floorplan.json 読み込み中...")
    regions, cameras_json, real_w, real_h = load_floorplan(FLOORPLAN_JSON_PATH)
    cameras_json = apply_camera_overrides(cameras_json)
    poses = resolve_poses(cameras_json, root="Area1", fmt=ROT_FMT)

    print("\n=== Step1: 座標系の符号規則のラウンドトリップ確認 ===")
    roundtrip_check(poses, TARGET_DETECTIONS[0]["cam_id"] if TARGET_DETECTIONS else CAM_IDS[0])

    print("\n=== Step2: 各検出についてシリンダーモデルをフィッティング ===")
    for det in TARGET_DETECTIONS:
        frame_idx = det["frame_idx"]
        cam_id = det["cam_id"]
        local_bbox = det["local_bbox"]
        pose = det["pose"]

        fx, fy, cx, cy, xi, alpha = get_intrinsics(cam_id)
        cam_pose = poses[cam_id]

        # ベースライン: bbox中心をそのまま逆投影(従来法)
        local_x = (local_bbox[0] + local_bbox[2]) / 2.0
        local_y = (local_bbox[1] + local_bbox[3]) / 2.0
        center_height, _, _ = cylinder_model_params(pose)
        pts_px = np.array([[local_x, local_y]], dtype=np.float64)
        baseline_xy, ok = unproject_to_floor(
            pts_px, cam_pose.R, cam_pose.t, fx, fy, cx, cy, xi, alpha,
            floor_z=-center_height,
        )
        if not ok[0]:
            print(f"  frame={frame_name(frame_idx)} {cam_id}: ベースライン逆投影が失敗。スキップ。")
            continue
        X0, Y0 = baseline_xy[0]

        # 3Dモデルフィッティング
        X_fit, Y_fit, final_bbox = fit_xy_to_bbox(
            local_bbox, pose, cam_pose.R, cam_pose.t, fx, fy, cx, cy, xi, alpha, (X0, Y0),
        )

        diff_m = float(np.linalg.norm([X_fit - X0, Y_fit - Y0]))
        print(f"\n  frame={frame_name(frame_idx)} {cam_id} pose={pose} bbox={local_bbox}")
        print(f"    ベースライン(bbox中心逆投影): (X,Y)=({X0:.3f}, {Y0:.3f})")
        print(f"    3Dモデルフィッティング      : (X,Y)=({X_fit:.3f}, {Y_fit:.3f})")
        print(f"    差分: {diff_m:.3f} m")
        if final_bbox is not None:
            print(f"    フィット後のモデルbbox: {tuple(round(v, 1) for v in final_bbox)}")
            print(f"    観測bbox              : {local_bbox}")

        overlay_path = save_overlay_image(
            frame_idx, cam_id, local_bbox, pose, X_fit, Y_fit,
            cam_pose.R, cam_pose.t, fx, fy, cx, cy, xi, alpha,
        )
        if overlay_path:
            print(f"    オーバーレイ画像保存: {overlay_path}")


if __name__ == "__main__":
    main()