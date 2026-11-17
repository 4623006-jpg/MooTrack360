"""
calibrate_camera_from_landmarks.py

【vml36で実行】

verify_camera_calibration.py で集めたLANDMARKS(camera/region/vertex_idx/pixel/label)
をそのまま流用し、Camera_1 -> Camera_2 -> Camera_3 の順にカメラチェーン
(Area1->Camera_1->Camera_2->Camera_3)に沿って外部パラメータ(位置・姿勢)と
内部パラメータ(xi, alpha)を最小二乗フィッティングする。

カメラ姿勢はチェーンになっているため、必ず親から順番に較正する。このスクリプトは
1回の実行内でCamera_1較正->その結果を土台にCamera_2較正->その結果を土台に
Camera_3較正、と自動で連鎖させる(手動でKNOWN_OVERRIDESを差し替える必要はない)。

各カメラについて3モードを試して比較表示する:
  1. 回転のみ最適化(位置固定)
  2. 回転+位置 最適化
  3. 回転+位置+内部パラメータ(xi, alpha) 最適化
CHAIN_MODE で指定したモードの結果が、次のカメラを較正する際の「親の位置」
として使われる(デフォルト"full"=モード3)。

誤差の比較として、以下3つを並べて表示する:
  - 現状本番値(Mootrack_core.pyのCAMERA_PARAM_OVERRIDES / CAM_INTRINSICS)での誤差
  - 最適化前(floorplan.json生値)での誤差
  - 最適化後(各モード)での誤差

★注意★ LANDMARKSの一部(壁・門の「上側」ラベルが付いたもの)は、物理的に
床面(z=0)より高い位置を指している可能性がある。その場合、floor_z=0を仮定した
このフィッティング自体に系統誤差が乗る。まずは「とりあえず今のデータで」
較正した結果を見て、精度が頭打ちになるようなら、床に接している点(壁・門の
下端など)だけに絞って再度較正することを推奨する。

--- 使い方 ---
    python calibrate_camera_from_landmarks.py
(パス・LANDMARKSは全て下の設定セクションに直書きしてあるので、引数なしで実行する)
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import least_squares

from Mootrack_core import (
    ROT_FMT, DS_FX, DS_FY, DS_CX, DS_CY, DS_XI, DS_ALPHA,
    get_intrinsics, build_R, unproject_to_floor,
    load_floorplan, load_floorplan_cameras, apply_camera_overrides, resolve_poses,
)

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
FLOORPLAN_JSON_PATH = "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/calibration/Farm2_Area_1_Artboard_config_.json"

# 較正するカメラチェーン(親から順に並べること。途中のカメラだけ較正することも可)
CAMERA_CHAIN = [
    ("Area1", "Camera_1"),
    ("Camera_1", "Camera_2"),
    ("Camera_2", "Camera_3"),
]

# 次のカメラの土台(parent_t)に使う結果のモード: "rot" / "rot_pos" / "full"
CHAIN_MODE = "full"

# 探索範囲(初期値からの±)
POS_BOUND_M = 1.0
ROT_BOUND_DEG = 8.0
XI_BOUND = 0.15
ALPHA_BOUND = 0.15

# ---------------------------------------------------------------------------
# ランドマーク(verify_camera_calibration.pyと同じ形式・同じデータ)
# ---------------------------------------------------------------------------
LANDMARKS = [
    {"camera": "Camera_1", "region": "Wall_9", "vertex_idx": 1,
     "pixel": (1100, 1090), "label": "wall_9上側"},
    {"camera": "Camera_1", "region": "Wall_11", "vertex_idx": 1,
     "pixel": (887, 2065), "label": "wall_11上側"},
    {"camera": "Camera_1", "region": "Gate_1", "vertex_idx": 1,
     "pixel": (801, 2044), "label": "Gate_1左側"},
    {"camera": "Camera_1", "region": "Feeding_Bay_2", "vertex_idx": 0,
     "pixel": (777, 839), "label": "Feeding_Bay_2の左上側"},
    {"camera": "Camera_1", "region": "Wall_9", "vertex_idx": 0,
     "pixel": (1093, 1859), "label": "Wall_9下側"},
    {"camera": "Camera_1", "region": "Feeding_Bay_3", "vertex_idx": 1,
     "pixel": (2760, 1231), "label": "Feeding_Bay_3の右上"},
    {"camera": "Camera_1", "region": "Wall_12", "vertex_idx": 1,
     "pixel": (2665, 1755), "label": "Wall_12の上"},
    {"camera": "Camera_1", "region": "Wall_7", "vertex_idx": 0,
     "pixel": (2726, 1768), "label": "Wall_7の右"},

    {"camera": "Camera_2", "region": "Feeding_Bay_2", "vertex_idx": 0,
     "pixel": (370, 948), "label": "Feeding_Bay_2の左上"},
    {"camera": "Camera_2", "region": "Wall_9", "vertex_idx": 1,
     "pixel": (382, 1130), "label": "Wall_9の上"},
    {"camera": "Camera_2", "region": "Gate_1", "vertex_idx": 1,
     "pixel": (272, 1637), "label": "Gate_1の左"},
    {"camera": "Camera_2", "region": "Gate_1", "vertex_idx": 0,
     "pixel": (289, 1646), "label": "Gate_1の右"},
    {"camera": "Camera_2", "region": "Wall_11", "vertex_idx": 0,
     "pixel": (297, 1779), "label": "Wall_11の下"},
    {"camera": "Camera_2", "region": "Wall_10", "vertex_idx": 1,
     "pixel": (2148, 1232), "label": "Wall_10の上"},
    {"camera": "Camera_2", "region": "Wall_10", "vertex_idx": 0,
     "pixel": (2042, 1842), "label": "Wall_10の下"},
    {"camera": "Camera_2", "region": "Wall_12", "vertex_idx": 1,
     "pixel": (2234, 2041), "label": "Wall_12の上"},
    {"camera": "Camera_2", "region": "Wall_12", "vertex_idx": 0,
     "pixel": (2163, 2213), "label": "Wall_12の下"},
    {"camera": "Camera_2", "region": "Feeding_Bay_3", "vertex_idx": 1,
     "pixel": (2643, 1282), "label": "Feeding_Bay_3の右上"},
    {"camera": "Camera_2", "region": "Wall_3", "vertex_idx": 1,
     "pixel": (2467, 1383), "label": "Wall_3左"},
    {"camera": "Camera_2", "region": "Wall_4", "vertex_idx": 1,
     "pixel": (2307, 2082), "label": "Wall_4左"},

    {"camera": "Camera_3", "region": "Feeding_Bay_2", "vertex_idx": 0,
     "pixel": (185, 1476), "label": "Feeding_Bay_2の左上"},
    {"camera": "Camera_3", "region": "Gate_1", "vertex_idx": 1,
     "pixel": (269, 1931), "label": "Gate_1の左"},
    {"camera": "Camera_3", "region": "Wall_10", "vertex_idx": 1,
     "pixel": (682, 1350), "label": "Wall_10の上"},
    {"camera": "Camera_3", "region": "Wall_10", "vertex_idx": 0,
     "pixel": (791, 1896), "label": "Wall_10の下"},
    {"camera": "Camera_3", "region": "Wall_12", "vertex_idx": 1,
     "pixel": (1222, 2128), "label": "Wall_12の上"},
    {"camera": "Camera_3", "region": "Wall_12", "vertex_idx": 0,
     "pixel": (1302, 2303), "label": "Wall_12の下"},
    {"camera": "Camera_3", "region": "Wall_3", "vertex_idx": 1,
     "pixel": (1265, 1133), "label": "Wall_3左"},
    {"camera": "Camera_3", "region": "Wall_4", "vertex_idx": 1,
     "pixel": (1515, 2217), "label": "Wall_4左"},
    {"camera": "Camera_3", "region": "Feeding_Bay_3", "vertex_idx": 1,
     "pixel": (2109, 779), "label": "Feeding_Bay_3の右上"},
    {"camera": "Camera_3", "region": "Wall_14", "vertex_idx": 0,
     "pixel": (1692, 731), "label": "Wall_14の上"},
    {"camera": "Camera_3", "region": "Milking_Station_2", "vertex_idx": 0,
     "pixel": (2141, 959), "label": "Milking_Station_2の左上"},
    {"camera": "Camera_3", "region": "Milking_Station_2", "vertex_idx": 3,
     "pixel": (2293, 1838), "label": "Milking_Station_2の左下"},

    {"region": "Feeding_Bay_2", "vertex_idx": 1, "camera": "Camera_1", "pixel": (2654,1141), "label": "Feeding_Bay_2右上(x=0.0)"},
    {"region": "Wall_10", "vertex_idx": 1, "camera": "Camera_1", "pixel": (2635,1309), "label": "Wall_10上"},
    {"region": "Wall_10", "vertex_idx": 0, "camera": "Camera_1", "pixel": (2621,1650), "label": "Wall_10下"},
    {"region": "Wall_2", "vertex_idx": 0, "camera": "Camera_2", "pixel": (2552,1432), "label": "Wall_2上"},
    {"region": "Wall_2", "vertex_idx": 1, "camera": "Camera_2", "pixel": (2417,2052), "label": "Wall_2下"},
    {"region": "Wall_6", "vertex_idx": 0, "camera": "Camera_2", "pixel": (2239,1809), "label": "Wall_6上"},
    {"region": "Wall_6", "vertex_idx": 1, "camera": "Camera_2", "pixel": (2303,1496) ,"label": "Wall_6下"},
    {"region": "Feeding_Bay_2", "vertex_idx": 1, "camera": "Camera_2", "pixel": (2346,1015), "label": "Feeding_Bay_2右上(x=30.90)"},
    {"region": "Wall_6", "vertex_idx": 1, "camera": "Camera_3", "pixel": (869,1494), "label": "Wall_6上"},
    {"region": "Wall_1", "vertex_idx": 0, "camera": "Camera_3", "pixel": (1117,1843) ,"label": "Wall_1下"},
    {"region": "Feeding_Bay_2", "vertex_idx": 1, "camera": "Camera_3", "pixel": (986,863), "label": "Feeding_Bay_1右上"},
]


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------

def find_region_vertex(regions, region_name: str, vertex_idx: int) -> Tuple[float, float]:
    for r in regions:
        if r.name == region_name:
            if vertex_idx < 0 or vertex_idx >= len(r.polygon_m):
                raise IndexError(
                    f"region=\"{region_name}\" にはvertex_idx={vertex_idx}が存在しません"
                    f"(頂点数={len(r.polygon_m)})"
                )
            x, y = r.polygon_m[vertex_idx]
            return float(x), float(y)
    raise KeyError(f"region=\"{region_name}\" がfloorplan.jsonに見つかりません")


def group_landmarks_by_camera(landmarks: List[dict]) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for lm in landmarks:
        grouped[lm["camera"]].append(lm)
    return grouped


def predict_world_xy(params6, parent_t, pts_local_px, xi, alpha):
    px, py, pz, a, b, c = params6
    R = build_R(a, b, c, ROT_FMT)
    t = parent_t + np.array([px, py, pz])
    return unproject_to_floor(pts_local_px, R, t, DS_FX, DS_FY, DS_CX, DS_CY, xi, alpha, floor_z=0.0)


def residuals_free6(free_params, fixed_params, free_mask, parent_t, pts_local_px, world_gt, xi, alpha):
    full = np.array(fixed_params, dtype=np.float64)
    full[free_mask] = free_params
    world_xy, valid = predict_world_xy(full, parent_t, pts_local_px, xi, alpha)
    res = (world_xy - world_gt).ravel()
    res = np.where(~np.repeat(valid, 2), 10.0, res)
    return res


def residuals_with_intrinsics(free_params, parent_t, pts_local_px, world_gt):
    px, py, pz, a, b, c, xi, alpha = free_params
    world_xy, valid = predict_world_xy([px, py, pz, a, b, c], parent_t, pts_local_px, xi, alpha)
    res = (world_xy - world_gt).ravel()
    res = np.where(~np.repeat(valid, 2), 10.0, res)
    return res


def errors_for(params6, parent_t, pts_local_px, world_gt, xi, alpha):
    world_xy, valid = predict_world_xy(params6, parent_t, pts_local_px, xi, alpha)
    err = np.linalg.norm(world_xy - world_gt, axis=1)
    return np.where(valid, err, np.nan)


def errors_for_full_pose(R, t, pts_local_px, world_gt, xi, alpha):
    world_xy, valid = unproject_to_floor(pts_local_px, R, t, DS_FX, DS_FY, DS_CX, DS_CY, xi, alpha, floor_z=0.0)
    err = np.linalg.norm(world_xy - world_gt, axis=1)
    return np.where(valid, err, np.nan)


def fmt_err(arr) -> str:
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return "N/A(全点invalid)"
    return f"mean={np.nanmean(arr):.4f}m  max={np.nanmax(arr):.4f}m  RMSE={np.sqrt(np.nanmean(arr ** 2)):.4f}m"


# ---------------------------------------------------------------------------
# 1カメラぶんの較正
# ---------------------------------------------------------------------------

def calibrate_one_camera(cam_id, parent_id, parent_t, init_params, init_xi, init_alpha,
                          pts_local_px, world_gt, labels, current_R, current_t,
                          current_xi, current_alpha):
    n = len(labels)
    print(f"\n{'=' * 70}")
    print(f"■ {parent_id} -> {cam_id} (ランドマーク{n}点)")
    print(f"{'=' * 70}")

    err_current = errors_for_full_pose(current_R, current_t, pts_local_px, world_gt, current_xi, current_alpha)
    err_raw = errors_for(init_params, parent_t, pts_local_px, world_gt, init_xi, init_alpha)
    print(f"  現状本番値(Mootrack_core.py): {fmt_err(err_current)}")
    print(f"  floorplan.json生値(較正前)  : {fmt_err(err_raw)}")

    results = {}

    # --- モード1: 回転のみ ---
    free_mask_rot = np.array([False, False, False, True, True, True])
    bounds = np.array([POS_BOUND_M] * 3 + [ROT_BOUND_DEG] * 3)
    lo, hi = init_params - bounds, init_params + bounds
    r1 = least_squares(
        residuals_free6, init_params[free_mask_rot],
        args=(init_params, free_mask_rot, parent_t, pts_local_px, world_gt, init_xi, init_alpha),
        bounds=(lo[free_mask_rot], hi[free_mask_rot]),
    )
    fitted_rot = init_params.copy()
    fitted_rot[free_mask_rot] = r1.x
    err_rot = errors_for(fitted_rot, parent_t, pts_local_px, world_gt, init_xi, init_alpha)
    print(f"\n  [モード1] 回転のみ最適化(位置固定): {fmt_err(err_rot)}")
    print(f"    params: {np.round(fitted_rot, 4).tolist()}  xi={init_xi}, alpha={init_alpha}")
    results["rot"] = (fitted_rot, init_xi, init_alpha, err_rot)

    # --- モード2: 回転+位置 ---
    free_mask_all = np.array([True] * 6)
    r2 = least_squares(
        residuals_free6, init_params[free_mask_all],
        args=(init_params, free_mask_all, parent_t, pts_local_px, world_gt, init_xi, init_alpha),
        bounds=(lo, hi),
    )
    fitted_all = r2.x
    err_all = errors_for(fitted_all, parent_t, pts_local_px, world_gt, init_xi, init_alpha)
    print(f"\n  [モード2] 回転+位置 最適化: {fmt_err(err_all)}")
    print(f"    params: {np.round(fitted_all, 4).tolist()}  xi={init_xi}, alpha={init_alpha}")
    results["rot_pos"] = (fitted_all, init_xi, init_alpha, err_all)

    # --- モード3: 回転+位置+内部パラメータ(xi, alpha) ---
    x0 = np.concatenate([init_params, [init_xi, init_alpha]])
    bounds8 = np.concatenate([bounds, [XI_BOUND, ALPHA_BOUND]])
    lo3, hi3 = x0 - bounds8, x0 + bounds8
    r3 = least_squares(
        residuals_with_intrinsics, x0,
        args=(parent_t, pts_local_px, world_gt),
        bounds=(lo3, hi3),
    )
    fitted_intr6 = r3.x[:6]
    fitted_xi, fitted_alpha = r3.x[6], r3.x[7]
    err_intr = errors_for(fitted_intr6, parent_t, pts_local_px, world_gt, fitted_xi, fitted_alpha)
    print(f"\n  [モード3] 回転+位置+内部パラメータ(xi,alpha): {fmt_err(err_intr)}")
    print(f"    params: {np.round(fitted_intr6, 4).tolist()}  xi={fitted_xi:.6f}, alpha={fitted_alpha:.6f}")
    results["full"] = (fitted_intr6, fitted_xi, fitted_alpha, err_intr)

    print(f"\n  --- 点ごとの誤差(較正前 -> モード3較正後) ---")
    print(f"  {'label':<24}{'較正前[m]':>12}{'較正後[m]':>12}")
    for i, lbl in enumerate(labels):
        print(f"  {lbl:<24}{err_raw[i]:>12.4f}{err_intr[i]:>12.4f}")

    return results


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print("floorplan.json 読み込み中...")
    regions, cameras_json_raw, real_w, real_h = load_floorplan(FLOORPLAN_JSON_PATH)
    print(f"床面サイズ: {real_w}m x {real_h}m / 区画数: {len(regions)}")

    # 「現状本番値」比較用: Mootrack_core.pyのCAMERA_PARAM_OVERRIDES適用済みの姿勢
    cameras_json_current = apply_camera_overrides(load_floorplan_cameras(FLOORPLAN_JSON_PATH))
    poses_current = resolve_poses(cameras_json_current, root="Area1", fmt=ROT_FMT)

    raw_params_by_key = {f"{e['parent']}->{e['child']}": np.array(e["params"], dtype=np.float64)
                          for e in cameras_json_raw}

    landmarks_by_cam = group_landmarks_by_camera(LANDMARKS)

    # 較正結果を積み上げていくposesチェーン(Area1は原点固定)
    poses_fitted: Dict[str, Tuple[np.ndarray, np.ndarray]] = {"Area1": (np.eye(3), np.zeros(3))}
    final_results = {}  # cam_id -> (parent_id, mode, params6, xi, alpha)

    for parent_id, cam_id in CAMERA_CHAIN:
        parent_R, parent_t = poses_fitted[parent_id]

        if cam_id not in landmarks_by_cam:
            print(f"\n■ {parent_id} -> {cam_id}: ランドマークが無いためスキップ"
                  f"(floorplan.json生値をそのままチェーンに使用)")
            key = f"{parent_id}->{cam_id}"
            params6 = raw_params_by_key[key]
            R = build_R(*params6[3:], ROT_FMT)
            t = parent_t + params6[:3]
            poses_fitted[cam_id] = (R, t)
            continue

        landmarks_this = landmarks_by_cam[cam_id]
        pts_local_px = np.array([lm["pixel"] for lm in landmarks_this], dtype=np.float64)
        world_gt = np.array(
            [find_region_vertex(regions, lm["region"], lm["vertex_idx"]) for lm in landmarks_this],
            dtype=np.float64,
        )
        labels = [lm["label"] for lm in landmarks_this]

        key = f"{parent_id}->{cam_id}"
        init_params = raw_params_by_key[key]
        _, _, _, _, init_xi, init_alpha = get_intrinsics(cam_id)

        current_pose = poses_current[cam_id]
        _, _, _, _, current_xi, current_alpha = get_intrinsics(cam_id)

        results = calibrate_one_camera(
            cam_id, parent_id, parent_t, init_params, init_xi, init_alpha,
            pts_local_px, world_gt, labels,
            current_pose.R, current_pose.t, current_xi, current_alpha,
        )

        chosen_params6, chosen_xi, chosen_alpha, _ = results[CHAIN_MODE]
        R = build_R(*chosen_params6[3:], ROT_FMT)
        t = parent_t + chosen_params6[:3]
        poses_fitted[cam_id] = (R, t)
        final_results[cam_id] = (parent_id, CHAIN_MODE, chosen_params6, chosen_xi, chosen_alpha)

    print(f"\n{'=' * 70}")
    print(f"■ Mootrack_core.py へ反映する場合の値(CHAIN_MODE=\"{CHAIN_MODE}\"を採用)")
    print(f"{'=' * 70}")
    print("CAMERA_PARAM_OVERRIDES = {")
    for cam_id, (parent_id, mode, params6, xi, alpha) in final_results.items():
        print(f"    \"{parent_id}->{cam_id}\": {np.round(params6, 4).tolist()},")
    print("}")
    print("\nCAM_INTRINSICS = {")
    for cam_id, (parent_id, mode, params6, xi, alpha) in final_results.items():
        print(f"    \"{cam_id}\": {{'fx': {DS_FX}, 'fy': {DS_FY}, 'cx': {DS_CX}, 'cy': {DS_CY}, "
              f"'xi': {xi:.6f}, 'alpha': {alpha:.6f}}},")
    print("}")
    print("\n(他のモード[rot / rot_pos]の数値を使いたい場合は、上のカメラごとの出力を参照してください)")


if __name__ == "__main__":
    main()