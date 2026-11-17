"""
calibrate_camera.py (汎用版)

任意のカメラ(Camera_1 / Camera_2 / Camera_3)について、既知の基準点
(画像上のピクセル座標 <-> 実世界座標(メートル)のペア)を使って
外部パラメータ(位置・姿勢)と内部パラメータ(xi, alpha)をフィッティングする。

Camera_2 で行ったのと同じロジックを、CAM_ID / PARENT_ID を書き換えるだけで
どのカメラにも使えるようにした汎用版。

--- 使い方 ---
  1. CAM_ID, PARENT_ID を対象カメラに設定する
     - Camera_1 を較正するとき: CAM_ID="Camera_1", PARENT_ID="Area1"
     - Camera_2 を較正するとき: CAM_ID="Camera_2", PARENT_ID="Camera_1"
     - Camera_3 を較正するとき: CAM_ID="Camera_3", PARENT_ID="Camera_2"
  2. KNOWN_OVERRIDES に、「すでに較正済みの他カメラの params」を入れておく
     (例: Camera_3を較正するときは、先に較正したCamera_2のparamsをここに入れる。
      入れないと、Camera_1由来のズレたCamera_2の位置を土台にしてCamera_3を
      最適化することになり、誤った結果になる)
  3. REFERENCE_POINTS に対象カメラの基準点(ローカルピクセル座標 0〜2944, 
     実世界座標メートル)を4点以上入力する
  4. python calibrate_camera.py を実行する
  5. 3つのモード(回転のみ/回転+位置/回転+位置+内部パラメータ)の結果が出るので、
     誤差が最も小さく、かつ物理的に妥当なものを選んで
     make_bev.py の CAM_INTRINSICS / CAMERA_PARAM_OVERRIDES に反映する
"""

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy.optimize import least_squares

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------

FLOORPLAN_JSON_PATH = "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/calibration/Farm2_Area_1_Artboard_config_.json"

# 共通デフォルトのDS内部パラメータ(まだ較正していないカメラはこれを使う)
DS_FX, DS_FY = 708.5, 735.2
DS_CX, DS_CY = 1472.0, 1470.1
DS_XI, DS_ALPHA = 0.006105, 0.48977

ROT_FMT = "euler_zyx_deg"
FLOOR_Z = 0.0

# ---------------------------------------------------------------------------
# ★★★ ここを対象カメラに合わせて書き換える ★★★
# ---------------------------------------------------------------------------

CAM_ID = "Camera_3"       # 較正したいカメラ
PARENT_ID = "Camera_2"    # そのカメラの親 (Camera_1なら"Area1", Camera_2なら"Camera_1", Camera_3なら"Camera_2")

# すでに較正済みの他カメラがあれば、そのparamsをここに入れる (key: "parent->child")
# 入れないと floorplan.json の元の値がそのまま使われる。
# 例: Camera_3を較正する場合、先に較正したCamera_2の結果をここに入れておくこと。
KNOWN_OVERRIDES = {
    "Camera_1->Camera_2": [13.113, 0.1269, -0.0743, -7.2382, 1.3079, -0.8149],
}
KNOWN_INTRINSICS_OVERRIDES = {
    "Camera_2": {"xi": 0.056105, "alpha": 0.421186},
}


# 対象カメラ(CAM_ID)の基準点。 local_x, local_y はそのカメラのタイル内ローカル
# ピクセル座標(0〜2944)。world_x, world_y は floorplan.json と同じ実世界座標(メートル)。
# REFERENCE_POINTS = [
#     {"label": "feedbay2_topleft",  "local_x": 773.0,  "local_y": 832.0,  "world_x": 0.0,     "world_y": -0.0089},
#     {"label": "gate1_left",        "local_x": 800.0,  "local_y": 2046.0, "world_x": 0.0,     "world_y": 12.0046},
#     {"label": "wall12_top",        "local_x": 2670.0, "local_y": 1753.0, "world_x": 31.08,   "world_y": 12.0046},
#     {"label": "feedbay2_topright", "local_x": 2656.0, "local_y": 1136.0, "world_x": 30.8952, "world_y": -0.0089},
#     {"label": "wall9_top",         "local_x": 1100.0, "local_y": 1086.0, "world_x": 4.88,    "world_y": 3.7046},
# ]カメラ1の参照ポイント
REFERENCE_POINTS = [
    {"label": "feedbay2_topleft",     "local_x": 183.0,  "local_y": 1476.0, "world_x": 0.0,     "world_y": -0.0089},
    {"label": "gate1_left",           "local_x": 260.0,  "local_y": 1935.0, "world_x": 0.0,     "world_y": 12.0046},
    {"label": "wall12_top",           "local_x": 1216.0, "local_y": 2133.0, "world_x": 31.08,   "world_y": 12.0046},
    {"label": "feedbay3_topright",    "local_x": 2110.0, "local_y": 776.0,  "world_x": 44.0,    "world_y": -0.0089},
    {"label": "hayarea_bottomright",  "local_x": 2300.0, "local_y": 1954.0, "world_x": 42.7929, "world_y": 13.6114},
    {"label": "wall10_top",           "local_x": 682.0,  "local_y": 1345.0, "world_x": 26.968899999999998, "world_y":3.7045999999999997}
]#カメラ3の参照ポイント


# ---------------------------------------------------------------------------
# モデル関数 (make_bev.py と同じロジック)
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
    if base_fmt == "euler_zyx":
        return _rz(a) @ _ry(b) @ _rx(c)
    raise ValueError(f"このスクリプトは euler_zyx(_deg) のみサポートしています: {fmt}")


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


def resolve_poses(cameras_json, root="Area1", fmt=ROT_FMT): #jsonファイルにあるカメラ情報の整理
    poses = {root: {"R": np.eye(3), "t": np.zeros(3)}}
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
            t_world = poses[parent]["t"] + t_local
            poses[child] = {"R": R_abs, "t": t_world}
            remaining.remove(entry)
            progressed = True
        if not progressed:
            raise RuntimeError(f"解決できないカメラがあります: {[e['child'] for e in remaining]}")
    return poses


def apply_known_overrides(cameras_json): #override処理
    """既に較正済みの他カメラのparamsを反映したコピーを返す(元データは変更しない)"""
    result = []
    for entry in cameras_json:
        key = f"{entry['parent']}->{entry['child']}"
        new_entry = dict(entry)
        if key in KNOWN_OVERRIDES:
            new_entry["params"] = list(KNOWN_OVERRIDES[key])
        result.append(new_entry)
    return result


def get_known_intrinsics(cam_id: str) -> Tuple[float, float]:
    if cam_id in KNOWN_INTRINSICS_OVERRIDES:
        p = KNOWN_INTRINSICS_OVERRIDES[cam_id]
        return p["xi"], p["alpha"]
    return DS_XI, DS_ALPHA


# ---------------------------------------------------------------------------
# 最適化
# ---------------------------------------------------------------------------

def predict_world_xy(params6, parent_t, pts_local_px, xi, alpha):
    px, py, pz, a, b, c = params6
    R = build_R(a, b, c, ROT_FMT)#radから度数表記へと変換する
    t = parent_t + np.array([px, py, pz])#親カメラの位置からの相対位置を計算
    return unproject_to_floor(pts_local_px, R, t, DS_FX, DS_FY, DS_CX, DS_CY, xi, alpha, FLOOR_Z)#Rで回転，rで並進移動して魚眼カメラの世界を見ている角度と位置を計算する


def residuals(free_params, fixed_params, free_mask, parent_t, pts_local_px, world_gt,
              fixed_xi, fixed_alpha):#外部パラメータのみ
    full = np.array(fixed_params, dtype=np.float64)
    full[free_mask] = free_params
    world_xy, valid = predict_world_xy(full, parent_t, pts_local_px, fixed_xi, fixed_alpha)
    res = (world_xy - world_gt).ravel()
    res = np.where(~np.repeat(valid, 2), 10.0, res)
    return res


def residuals_with_intrinsics(free_params, parent_t, pts_local_px, world_gt):#外部・内部パラメータ
    px, py, pz, a, b, c, xi, alpha = free_params
    world_xy, valid = predict_world_xy([px, py, pz, a, b, c], parent_t, pts_local_px, xi, alpha)
    res = (world_xy - world_gt).ravel()
    res = np.where(~np.repeat(valid, 2), 10.0, res)
    return res


def print_errors(label, init_params6, fitted_params6, parent_t, pts_local_px, world_gt,
                  init_xi, init_alpha, fitted_xi=None, fitted_alpha=None):
    fxi = init_xi if fitted_xi is None else fitted_xi
    falpha = init_alpha if fitted_alpha is None else fitted_alpha
    world_before, _ = predict_world_xy(init_params6, parent_t, pts_local_px, init_xi, init_alpha)
    world_after, _ = predict_world_xy(fitted_params6, parent_t, pts_local_px, fxi, falpha)
    err_before = np.linalg.norm(world_before - world_gt, axis=1)
    err_after = np.linalg.norm(world_after - world_gt, axis=1)

    print(f"\n=== {label} ===")
    print(f"  最適化前 params: {np.round(init_params6, 4).tolist()}  xi={init_xi}, alpha={init_alpha}")
    print(f"  最適化後 params: {np.round(fitted_params6, 4).tolist()}  xi={fxi:.6f}, alpha={falpha:.6f}")
    print(f"  {'label':<20}{'誤差(前)m':>12}{'誤差(後)m':>12}")
    for i, p in enumerate(REFERENCE_POINTS):
        print(f"  {p.get('label', f'pt{i}'):<20}{err_before[i]:>12.4f}{err_after[i]:>12.4f}")
    print(f"  --- mean: {err_before.mean():.4f} -> {err_after.mean():.4f}   "
          f"max: {err_before.max():.4f} -> {err_after.max():.4f} ---")
    return err_after


def main():
    if len(REFERENCE_POINTS) < 3:
        print("REFERENCE_POINTS を最低3点(できれば4〜6点)入力してから実行してください。")
        return

    data = json.loads(Path(FLOORPLAN_JSON_PATH).read_text(encoding="utf-8"))
    cameras_json = apply_known_overrides(data["cameras"])

    # 対象カメラのエントリを取得
    target_entry = None
    for entry in cameras_json:
        if entry["child"] == CAM_ID and entry["parent"] == PARENT_ID:
            target_entry = entry
            break
    if target_entry is None:
        raise RuntimeError(f"{PARENT_ID} -> {CAM_ID} のエントリが見つかりません")

    init_params = np.array(target_entry["params"], dtype=np.float64)
    init_xi, init_alpha = get_known_intrinsics(CAM_ID)

    # 親の world 位置を確定(親より上のチェーンは、既知の較正結果を反映済みの状態で解決される)
    poses = resolve_poses(cameras_json, root="Area1", fmt=ROT_FMT)
    parent_t = poses[PARENT_ID]["t"]

    pts_local_px = np.array([[p["local_x"], p["local_y"]] for p in REFERENCE_POINTS], dtype=np.float64)
    world_gt = np.array([[p["world_x"], p["world_y"]] for p in REFERENCE_POINTS], dtype=np.float64)

    print(f"対象カメラ: {PARENT_ID} -> {CAM_ID}")
    print(f"親({PARENT_ID})の world 位置: {parent_t.round(3)}")
    print(f"初期 xi={init_xi}, alpha={init_alpha}")

    # --- モード1: 回転のみ ---
    free_mask_rot = np.array([False, False, False, True, True, True])
    bounds_rot = np.array([0.0, 0.0, 0.0, 8.0, 8.0, 8.0])
    lo = init_params - bounds_rot
    hi = init_params + bounds_rot
    result_rot = least_squares(
        residuals, init_params[free_mask_rot],
        args=(init_params, free_mask_rot, parent_t, pts_local_px, world_gt, init_xi, init_alpha),
        bounds=(lo[free_mask_rot], hi[free_mask_rot]),
    )
    fitted_rot = init_params.copy()
    fitted_rot[free_mask_rot] = result_rot.x
    err_rot = print_errors("回転のみ最適化 (位置固定)", init_params, fitted_rot,
                           parent_t, pts_local_px, world_gt, init_xi, init_alpha)

    # --- モード2: 回転+位置 ---
    free_mask_all = np.array([True] * 6)
    bounds_all = np.array([1.0, 1.0, 1.0, 8.0, 8.0, 8.0])
    lo = init_params - bounds_all
    hi = init_params + bounds_all
    result_all = least_squares(
        residuals, init_params[free_mask_all],
        args=(init_params, free_mask_all, parent_t, pts_local_px, world_gt, init_xi, init_alpha),
        bounds=(lo, hi),
    )
    fitted_all = result_all.x
    err_all = print_errors("回転+位置 最適化", init_params, fitted_all,
                           parent_t, pts_local_px, world_gt, init_xi, init_alpha)

    # --- モード3: 回転+位置+内部パラメータ(xi, alpha) ---
    xi_bound, alpha_bound = 0.15, 0.15
    x0 = np.concatenate([init_params, [init_xi, init_alpha]])
    lo3 = np.concatenate([init_params - bounds_all, [init_xi - xi_bound, init_alpha - alpha_bound]])
    hi3 = np.concatenate([init_params + bounds_all, [init_xi + xi_bound, init_alpha + alpha_bound]])
    result_intr = least_squares(
        residuals_with_intrinsics, x0,
        args=(parent_t, pts_local_px, world_gt),
        bounds=(lo3, hi3),
    )
    fitted_intr6 = result_intr.x[:6]
    fitted_xi, fitted_alpha = result_intr.x[6], result_intr.x[7]
    err_intr = print_errors("回転+位置+内部パラメータ(xi,alpha)", init_params, fitted_intr6,
                            parent_t, pts_local_px, world_gt, init_xi, init_alpha,
                            fitted_xi=fitted_xi, fitted_alpha=fitted_alpha)

    print("\n=== 比較まとめ ===")
    print(f"  回転のみ            : mean={err_rot.mean():.4f}m  max={err_rot.max():.4f}m")
    print(f"  回転+位置           : mean={err_all.mean():.4f}m  max={err_all.max():.4f}m")
    print(f"  回転+位置+内部パラメータ: mean={err_intr.mean():.4f}m  max={err_intr.max():.4f}m")

    print(f"\nmake_bev.py に反映する場合の値 ({PARENT_ID}->{CAM_ID}):")
    print(f"  CAMERA_PARAM_OVERRIDES['{PARENT_ID}->{CAM_ID}'] = {fitted_intr6.round(4).tolist()}")
    print(f"  CAM_INTRINSICS['{CAM_ID}'] = {{'fx': {DS_FX}, 'fy': {DS_FY}, "
          f"'cx': {DS_CX}, 'cy': {DS_CY}, 'xi': {fitted_xi:.6f}, 'alpha': {fitted_alpha:.6f}}}")


if __name__ == "__main__":
    main()