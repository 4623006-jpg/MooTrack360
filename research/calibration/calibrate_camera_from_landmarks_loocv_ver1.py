"""
calibrate_camera_from_landmarks_loocv_ver1.py

calibrate_camera_from_landmarks.py(カメラチェーンに沿ったランドマークからの
最小二乗較正)に、leave-one-out cross-validationを追加したもの。

目的: 「どのランドマークが較正誤差を増やしている(=信頼できない)か」を
特定し、較正に使うランドマークを絞り込むための判断材料を作る。

やり方:
  1. まずcalibrate_camera_from_landmarks.pyと同じ手順で、CAMERA_CHAINに沿って
     全ランドマークを使った「フル較正」をカメラごとに順番に行い、各カメラの
     親の位置(parent_t)を確定させる(これはLOOCVの基準点として固定する。
     LOOCVは「そのカメラ自身のランドマーク」をleave-one-outするだけで、
     チェーンの親子関係はフル較正の結果をそのまま使う)。
  2. 各カメラについて、そのカメラのランドマークを1つずつ除いて
     (モード3: 回転+位置+内部パラメータ)再較正し、除いたランドマークの
     世界座標を予測→正解との誤差(held-out誤差)を記録する。
  3. held-out誤差が大きいランドマーク(=そのランドマークを較正に使うと、
     他のランドマークとの整合性が悪くなる、または他の点からは外れ値に見える)
     を大きい順にランキング表示する。
  4. 参考として、全点を使ったフル較正でのその点自身の残差(in-sample誤差)も
     並べて表示する(held-out誤差 >> in-sample誤差なら、その点が較正結果を
     過剰に引っ張っている「レバレッジポイント」である可能性が高い)。
  5. held-out誤差が大きい上位N点を除外して再較正した場合の全体RMSEの変化も
     表示し、実際に除外する価値があるかを確認できるようにする。

★注意(元スクリプトと同じ)★ LANDMARKSの「壁・門の上側」ラベルの点は、
floor_z=0の仮定からズレている可能性があり、その場合は系統的に誤差が乗る。
held-out誤差が高いランドマークの多くが「上側」ラベルに集中していれば、
それはランドマーク選びのミスではなく、floor_z=0仮定そのものの限界を
示している可能性が高い。

--- 使い方 ---
    python calibrate_camera_from_landmarks_loocv_ver1.py
(パス・LANDMARKSは全て下の設定セクションに直書き。calibrate_camera_from_
landmarks.pyと全く同じLANDMARKSを使っているので、そちらを更新したら
ここも合わせて更新すること)
"""

import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import least_squares

MOOTRACK_CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calibration")
if MOOTRACK_CODE_DIR not in sys.path:
    sys.path.insert(0, MOOTRACK_CODE_DIR)

from Mootrack_core import (
    ROT_FMT, DS_FX, DS_FY, DS_CX, DS_CY,
    get_intrinsics, build_R, unproject_to_floor,
    load_floorplan, load_floorplan_cameras, apply_camera_overrides, resolve_poses,
)

# ---------------------------------------------------------------------------
# 設定(calibrate_camera_from_landmarks.pyと同じ値)
# ---------------------------------------------------------------------------
FLOORPLAN_JSON_PATH = "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/calibration/Farm2_Area_1_Artboard_config_.json"

CAMERA_CHAIN = [
    ("Area1", "Camera_1"),
    ("Camera_1", "Camera_2"),
    ("Camera_2", "Camera_3"),
]

POS_BOUND_M = 1.0
ROT_BOUND_DEG = 8.0
XI_BOUND = 0.15
ALPHA_BOUND = 0.15

# held-out誤差が大きい順に、上位何点を「除外候補」として再較正・比較するか
N_EXCLUDE_CANDIDATES = 3

# ---------------------------------------------------------------------------
# ランドマーク(calibrate_camera_from_landmarks.pyと全く同じもの)
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

    {"region": "Feeding_Bay_2", "vertex_idx": 1, "camera": "Camera_1", "pixel": (2654, 1141), "label": "Feeding_Bay_2右上(x=0.0)"},
    {"region": "Wall_10", "vertex_idx": 1, "camera": "Camera_1", "pixel": (2635, 1309), "label": "Wall_10上"},
    {"region": "Wall_10", "vertex_idx": 0, "camera": "Camera_1", "pixel": (2621, 1650), "label": "Wall_10下"},
    {"region": "Wall_2", "vertex_idx": 0, "camera": "Camera_2", "pixel": (2552, 1432), "label": "Wall_2上"},
    {"region": "Wall_2", "vertex_idx": 1, "camera": "Camera_2", "pixel": (2417, 2052), "label": "Wall_2下"},
    {"region": "Wall_6", "vertex_idx": 0, "camera": "Camera_2", "pixel": (2239, 1809), "label": "Wall_6上"},
    {"region": "Wall_6", "vertex_idx": 1, "camera": "Camera_2", "pixel": (2303, 1496), "label": "Wall_6下"},
    {"region": "Feeding_Bay_2", "vertex_idx": 1, "camera": "Camera_2", "pixel": (2346, 1015), "label": "Feeding_Bay_2右上(x=30.90)"},
    {"region": "Wall_6", "vertex_idx": 1, "camera": "Camera_3", "pixel": (869, 1494), "label": "Wall_6上"},
    {"region": "Wall_1", "vertex_idx": 0, "camera": "Camera_3", "pixel": (1117, 1843), "label": "Wall_1下"},
    {"region": "Feeding_Bay_2", "vertex_idx": 1, "camera": "Camera_3", "pixel": (986, 863), "label": "Feeding_Bay_1右上"},
]


# ---------------------------------------------------------------------------
# ヘルパー(calibrate_camera_from_landmarks.pyと同じ)
# ---------------------------------------------------------------------------

def find_region_vertex(regions, region_name: str, vertex_idx: int) -> Tuple[float, float]:
    for r in regions:
        if r.name == region_name:
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


def residuals_with_intrinsics(free_params, parent_t, pts_local_px, world_gt):
    px, py, pz, a, b, c, xi, alpha = free_params
    world_xy, valid = predict_world_xy([px, py, pz, a, b, c], parent_t, pts_local_px, xi, alpha)
    res = (world_xy - world_gt).ravel()
    res = np.where(~np.repeat(valid, 2), 10.0, res)
    return res


def fit_full(parent_t, pts_local_px, world_gt, init_params6, init_xi, init_alpha):
    """モード3(回転+位置+内部パラメータ)の最小二乗フィッティング。
    戻り値: (fitted_params6, fitted_xi, fitted_alpha)"""
    x0 = np.concatenate([init_params6, [init_xi, init_alpha]])
    bounds8 = np.concatenate([[POS_BOUND_M] * 3 + [ROT_BOUND_DEG] * 3, [XI_BOUND, ALPHA_BOUND]])
    lo, hi = x0 - bounds8, x0 + bounds8
    r = least_squares(
        residuals_with_intrinsics, x0,
        args=(parent_t, pts_local_px, world_gt),
        bounds=(lo, hi),
    )
    return r.x[:6], r.x[6], r.x[7]


def point_error(params6, parent_t, px_xy, world_xy_gt, xi, alpha) -> float:
    pred, valid = predict_world_xy(params6, parent_t, np.array([px_xy]), xi, alpha)
    if not valid[0]:
        return float("nan")
    return float(np.linalg.norm(pred[0] - world_xy_gt))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print("floorplan.json 読み込み中...")
    regions, cameras_json_raw, real_w, real_h = load_floorplan(FLOORPLAN_JSON_PATH)
    print(f"床面サイズ: {real_w}m x {real_h}m / 区画数: {len(regions)}")

    raw_params_by_key = {
        f"{e['parent']}->{e['child']}": np.array(e["params"], dtype=np.float64)
        for e in cameras_json_raw
    }
    landmarks_by_cam = group_landmarks_by_camera(LANDMARKS)

    # --- 1. まずチェーンに沿ってフル較正し、各カメラの親位置(parent_t)を確定させる ---
    poses_fitted: Dict[str, Tuple[np.ndarray, np.ndarray]] = {"Area1": (np.eye(3), np.zeros(3))}
    full_fit_by_cam: Dict[str, dict] = {}

    for parent_id, cam_id in CAMERA_CHAIN:
        parent_R, parent_t = poses_fitted[parent_id]
        key = f"{parent_id}->{cam_id}"
        init_params6 = raw_params_by_key[key]
        _, _, _, _, init_xi, init_alpha = get_intrinsics(cam_id)

        if cam_id not in landmarks_by_cam:
            R = build_R(*init_params6[3:], ROT_FMT)
            t = parent_t + init_params6[:3]
            poses_fitted[cam_id] = (R, t)
            continue

        landmarks_this = landmarks_by_cam[cam_id]
        pts_local_px = np.array([lm["pixel"] for lm in landmarks_this], dtype=np.float64)
        world_gt = np.array(
            [find_region_vertex(regions, lm["region"], lm["vertex_idx"]) for lm in landmarks_this],
            dtype=np.float64,
        )

        fitted_params6, fitted_xi, fitted_alpha = fit_full(
            parent_t, pts_local_px, world_gt, init_params6, init_xi, init_alpha
        )
        R = build_R(*fitted_params6[3:], ROT_FMT)
        t = parent_t + fitted_params6[:3]
        poses_fitted[cam_id] = (R, t)
        full_fit_by_cam[cam_id] = {
            "parent_t": parent_t, "params6": fitted_params6, "xi": fitted_xi, "alpha": fitted_alpha,
            "landmarks": landmarks_this, "pts_local_px": pts_local_px, "world_gt": world_gt,
        }

    # --- 2. カメラごとにleave-one-out cross-validation ---
    for cam_id, fit in full_fit_by_cam.items():
        landmarks_this = fit["landmarks"]
        pts_local_px = fit["pts_local_px"]
        world_gt = fit["world_gt"]
        parent_t = fit["parent_t"]
        n = len(landmarks_this)

        print(f"\n{'=' * 70}")
        print(f"■ {cam_id} のLeave-One-Out Cross-Validation(ランドマーク{n}点)")
        print(f"{'=' * 70}")

        # 参考: 全点使ったフル較正での各点自身の残差(in-sample誤差)
        in_sample_err = np.array([
            point_error(fit["params6"], parent_t, pts_local_px[i], world_gt[i], fit["xi"], fit["alpha"])
            for i in range(n)
        ])

        loo_err = np.full(n, np.nan)
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            if mask.sum() < 4:
                print(f"  {landmarks_this[i]['label']}: 残り点数が{mask.sum()}点で8パラメータの"
                      f"フィッティングには不足気味のためスキップ")
                continue

            init_params6 = raw_params_by_key[
                [k for k in raw_params_by_key.keys() if k.endswith(f"->{cam_id}")][0]
            ]
            train_px = pts_local_px[mask]
            train_gt = world_gt[mask]
            fitted_params6, fitted_xi, fitted_alpha = fit_full(
                parent_t, train_px, train_gt, init_params6, fit["xi"], fit["alpha"]
            )
            loo_err[i] = point_error(fitted_params6, parent_t, pts_local_px[i], world_gt[i], fitted_xi, fitted_alpha)

        order = np.argsort(-np.nan_to_num(loo_err, nan=-1.0))
        print(f"\n  {'label':<28}{'held-out誤差[m]':>16}{'in-sample誤差[m]':>18}")
        for i in order:
            print(f"  {landmarks_this[i]['label']:<28}{loo_err[i]:>16.4f}{in_sample_err[i]:>18.4f}")

        valid_loo = loo_err[~np.isnan(loo_err)]
        if len(valid_loo) > 0:
            print(f"\n  LOO誤差: mean={np.nanmean(loo_err):.4f}m  median={np.nanmedian(loo_err):.4f}m  "
                  f"RMSE={np.sqrt(np.nanmean(loo_err ** 2)):.4f}m")

        # --- 3. held-out誤差が大きい上位N点を除外して再較正した場合の比較 ---
        worst_idx = order[:min(N_EXCLUDE_CANDIDATES, n)]
        print(f"\n  held-out誤差が大きい上位{len(worst_idx)}点を除外候補とした場合:")
        for i in worst_idx:
            print(f"    - {landmarks_this[i]['label']} (held-out誤差={loo_err[i]:.4f}m)")

        keep_mask = np.ones(n, dtype=bool)
        keep_mask[worst_idx] = False
        if keep_mask.sum() >= 4:
            init_params6 = raw_params_by_key[
                [k for k in raw_params_by_key.keys() if k.endswith(f"->{cam_id}")][0]
            ]
            refit_params6, refit_xi, refit_alpha = fit_full(
                parent_t, pts_local_px[keep_mask], world_gt[keep_mask], init_params6, fit["xi"], fit["alpha"]
            )
            # 除外した点も含めて全点での誤差を再計算(除外点はテスト用として見る)
            all_err_after_exclude = np.array([
                point_error(refit_params6, parent_t, pts_local_px[i], world_gt[i], refit_xi, refit_alpha)
                for i in range(n)
            ])
            print(f"\n  [除外後に再較正]全{n}点での誤差: "
                  f"mean={np.nanmean(all_err_after_exclude):.4f}m  "
                  f"RMSE={np.sqrt(np.nanmean(all_err_after_exclude ** 2)):.4f}m")
            print(f"  [除外前(フル較正)]全{n}点での誤差: "
                  f"mean={np.nanmean(in_sample_err):.4f}m  "
                  f"RMSE={np.sqrt(np.nanmean(in_sample_err ** 2)):.4f}m")
            print(f"  (除外後の方がRMSEが小さければ、その上位{len(worst_idx)}点を較正から"
                  f"外す価値がある。逆に除外後に悪化するなら、その点は外れ値ではなく"
                  f"単に難しい/境界的な点だった可能性がある)")


if __name__ == "__main__":
    main()