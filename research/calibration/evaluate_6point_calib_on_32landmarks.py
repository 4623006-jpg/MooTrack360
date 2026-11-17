"""
evaluate_6point_calib_on_32landmarks.py

calibrate_camera.py(汎用版、REFERENCE_POINTS=6点)でCamera_2->Camera_3を
較正して得たパラメータを、calibrate_camera_from_landmarks.py /
verify_camera_calibration.py で使っている「32点のランドマーク全体」に
当てはめたときの誤差を計算する。

目的: 「わずか6点だけで較正したパラメータが、より広い32点の検証セットに対しても
汎化しているか(=6点への過学習ではないか)」をスライド用に確認すること。

--- 使うパラメータ(このスクリプト実行の直前にcalibrate_camera.pyで得られた値) ---
  Camera_1        : 較正前の生値のまま(floorplan.json由来、内部パラメータもDS_XI/DS_ALPHAの共通デフォルト)
                     ※ calibrate_camera.py(CAM_ID="Camera_3"実行時)のKNOWN_OVERRIDESに
                        Area1->Camera_1 が含まれていなかったため、この6点較正が前提として
                        使った値は「Camera_1は未較正のまま」という状態。ここでも同じ前提を再現する。
  Camera_1->Camera_2: KNOWN_OVERRIDES に入っていた値(既に較正済みとして扱われていたもの)
                        [13.113, 0.1269, -0.0743, -7.2382, 1.3079, -0.8149]
                        xi=0.056105, alpha=0.421186
  Camera_2->Camera_3: 今回6点で新たに較正した値
                        [13.641, 0.1279, -1.6374, 11.0855, -0.3336, 0.7534]
                        xi=-0.143895, alpha=0.533116

--- 使い方 ---
    python evaluate_6point_calib_on_32landmarks.py
"""

import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Mootrack_core.py の読み込み(自己位置基準パス。実際の配置と違う場合は書き換えること)
# ---------------------------------------------------------------------------
MOOTRACK_CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calibration")
if MOOTRACK_CODE_DIR not in sys.path:
    sys.path.insert(0, MOOTRACK_CODE_DIR)

from Mootrack_core import (
    ROT_FMT, DS_FX, DS_FY, DS_CX, DS_CY, DS_XI, DS_ALPHA,
    build_R, unproject_to_floor,
    load_floorplan,
)

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
FLOORPLAN_JSON_PATH = "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/calibration/Farm2_Area_1_Artboard_config_.json"

# --- 今回評価したい「6点較正パラメータ」一式(このスクリプト実行前のcalibrate_camera.py出力そのまま) ---
PARAM_OVERRIDES = {
    # Area1->Camera_1 は指定しない = floorplan.json の生値をそのまま使う(未較正のまま)
    "Camera_1->Camera_2": [13.113, 0.1269, -0.0743, -7.2382, 1.3079, -0.8149],
    "Camera_2->Camera_3": [13.641, 0.1279, -1.6374, 11.0855, -0.3336, 0.7534],
}
INTRINSICS_OVERRIDES = {
    # Camera_1 は指定しない = 共通デフォルト(DS_XI, DS_ALPHA)のまま
    "Camera_2": {"xi": 0.056105, "alpha": 0.421186},
    "Camera_3": {"xi": -0.143895, "alpha": 0.533116},
}


def get_intrinsics(cam_id: str) -> Tuple[float, float, float, float, float, float]:
    if cam_id in INTRINSICS_OVERRIDES:
        p = INTRINSICS_OVERRIDES[cam_id]
        return DS_FX, DS_FY, DS_CX, DS_CY, p["xi"], p["alpha"]
    return DS_FX, DS_FY, DS_CX, DS_CY, DS_XI, DS_ALPHA


def apply_overrides(cameras_json: list) -> list:
    result = []
    for entry in cameras_json:
        key = f"{entry['parent']}->{entry['child']}"
        new_entry = dict(entry)
        if key in PARAM_OVERRIDES:
            new_entry["params"] = list(PARAM_OVERRIDES[key])
        result.append(new_entry)
    return result


def resolve_poses(cameras_json: list, root: str = "Area1", fmt: str = ROT_FMT):
    poses: Dict[str, Tuple[np.ndarray, np.ndarray]] = {root: (np.eye(3), np.zeros(3))}
    remaining = list(cameras_json)
    while remaining:
        progressed = False
        for entry in list(remaining):
            parent, child = entry["parent"], entry["child"]
            if parent not in poses:
                continue
            px, py, pz, a, b, c = entry["params"]
            R = build_R(a, b, c, fmt)
            t_local = np.array([px, py, pz], dtype=np.float64)
            t_world = poses[parent][1] + t_local
            poses[child] = (R, t_world)
            remaining.remove(entry)
            progressed = True
        if not progressed:
            raise RuntimeError(f"解決できないカメラがあります: {[e['child'] for e in remaining]}")
    return poses


def find_region_vertex(regions, region_name: str, vertex_idx: int) -> Tuple[float, float]:
    for r in regions:
        if r.name == region_name:
            x, y = r.polygon_m[vertex_idx]
            return float(x), float(y)
    raise KeyError(f"region=\"{region_name}\" がfloorplan.jsonに見つかりません")


# ---------------------------------------------------------------------------
# ランドマーク(calibrate_camera_from_landmarks.py / verify_camera_calibration.py と同一、32点)
# ---------------------------------------------------------------------------
LANDMARKS = [
    {"camera": "Camera_1", "region": "Wall_9", "vertex_idx": 1, "pixel": (1100, 1090), "label": "wall_9上側"},
    {"camera": "Camera_1", "region": "Wall_11", "vertex_idx": 1, "pixel": (887, 2065), "label": "wall_11上側"},
    {"camera": "Camera_1", "region": "Gate_1", "vertex_idx": 1, "pixel": (801, 2044), "label": "Gate_1左側"},
    {"camera": "Camera_1", "region": "Feeding_Bay_2", "vertex_idx": 0, "pixel": (777, 839), "label": "Feeding_Bay_2の左上側"},
    {"camera": "Camera_1", "region": "Wall_9", "vertex_idx": 0, "pixel": (1093, 1859), "label": "Wall_9下側"},
    {"camera": "Camera_1", "region": "Feeding_Bay_3", "vertex_idx": 1, "pixel": (2760, 1231), "label": "Feeding_Bay_3の右上"},
    {"camera": "Camera_1", "region": "Wall_12", "vertex_idx": 1, "pixel": (2665, 1755), "label": "Wall_12の上"},
    {"camera": "Camera_1", "region": "Wall_7", "vertex_idx": 0, "pixel": (2726, 1768), "label": "Wall_7の右"},

    {"camera": "Camera_2", "region": "Feeding_Bay_2", "vertex_idx": 0, "pixel": (370, 948), "label": "Feeding_Bay_2の左上"},
    {"camera": "Camera_2", "region": "Wall_9", "vertex_idx": 1, "pixel": (382, 1130), "label": "Wall_9の上"},
    {"camera": "Camera_2", "region": "Gate_1", "vertex_idx": 1, "pixel": (272, 1637), "label": "Gate_1の左"},
    {"camera": "Camera_2", "region": "Gate_1", "vertex_idx": 0, "pixel": (289, 1646), "label": "Gate_1の右"},
    {"camera": "Camera_2", "region": "Wall_11", "vertex_idx": 0, "pixel": (297, 1779), "label": "Wall_11の下"},
    {"camera": "Camera_2", "region": "Wall_10", "vertex_idx": 1, "pixel": (2148, 1232), "label": "Wall_10の上"},
    {"camera": "Camera_2", "region": "Wall_10", "vertex_idx": 0, "pixel": (2042, 1842), "label": "Wall_10の下"},
    {"camera": "Camera_2", "region": "Wall_12", "vertex_idx": 1, "pixel": (2234, 2041), "label": "Wall_12の上"},
    {"camera": "Camera_2", "region": "Wall_12", "vertex_idx": 0, "pixel": (2163, 2213), "label": "Wall_12の下"},
    {"camera": "Camera_2", "region": "Feeding_Bay_3", "vertex_idx": 1, "pixel": (2643, 1282), "label": "Feeding_Bay_3の右上"},
    {"camera": "Camera_2", "region": "Wall_3", "vertex_idx": 1, "pixel": (2467, 1383), "label": "Wall_3左"},
    {"camera": "Camera_2", "region": "Wall_4", "vertex_idx": 1, "pixel": (2307, 2082), "label": "Wall_4左"},

    {"camera": "Camera_3", "region": "Feeding_Bay_2", "vertex_idx": 0, "pixel": (185, 1476), "label": "Feeding_Bay_2の左上"},
    {"camera": "Camera_3", "region": "Gate_1", "vertex_idx": 1, "pixel": (269, 1931), "label": "Gate_1の左"},
    {"camera": "Camera_3", "region": "Wall_10", "vertex_idx": 1, "pixel": (682, 1350), "label": "Wall_10の上"},
    {"camera": "Camera_3", "region": "Wall_10", "vertex_idx": 0, "pixel": (791, 1896), "label": "Wall_10の下"},
    {"camera": "Camera_3", "region": "Wall_12", "vertex_idx": 1, "pixel": (1222, 2128), "label": "Wall_12の上"},
    {"camera": "Camera_3", "region": "Wall_12", "vertex_idx": 0, "pixel": (1302, 2303), "label": "Wall_12の下"},
    {"camera": "Camera_3", "region": "Wall_3", "vertex_idx": 1, "pixel": (1265, 1133), "label": "Wall_3左"},
    {"camera": "Camera_3", "region": "Wall_4", "vertex_idx": 1, "pixel": (1515, 2217), "label": "Wall_4左"},
    {"camera": "Camera_3", "region": "Feeding_Bay_3", "vertex_idx": 1, "pixel": (2109, 779), "label": "Feeding_Bay_3の右上"},
    {"camera": "Camera_3", "region": "Wall_14", "vertex_idx": 0, "pixel": (1692, 731), "label": "Wall_14の上"},
    {"camera": "Camera_3", "region": "Milking_Station_2", "vertex_idx": 0, "pixel": (2141, 959), "label": "Milking_Station_2の左上"},
    {"camera": "Camera_3", "region": "Milking_Station_2", "vertex_idx": 3, "pixel": (2293, 1838), "label": "Milking_Station_2の左下"},

]


def main():
    print("floorplan.json 読み込み中...")
    regions, cameras_json_raw, real_w, real_h = load_floorplan(FLOORPLAN_JSON_PATH)
    print(f"床面サイズ: {real_w}m x {real_h}m / 区画数: {len(regions)}")

    cameras_json = apply_overrides(cameras_json_raw)
    poses = resolve_poses(cameras_json, root="Area1", fmt=ROT_FMT)

    print("\n使用するパラメータ:")
    for cam_id in ["Camera_1", "Camera_2", "Camera_3"]:
        R, t = poses[cam_id]
        _, _, _, _, xi, alpha = get_intrinsics(cam_id)
        tag = " [6点較正 or 既知較正値]" if cam_id != "Camera_1" else " [未較正・floorplan生値のまま]"
        print(f"  {cam_id}: t={t.round(3)}  xi={xi}  alpha={alpha}{tag}")

    print(f"\n=== ランドマークごとの誤差(32点) ===")
    print(f"  {'camera':>8} | {'region/vertex':>24} | {'誤差[m]':>9} | 備考")

    rows = []
    for lm in LANDMARKS:
        cam_id = lm["camera"]
        R, t = poses[cam_id]
        fx, fy, cx, cy, xi, alpha = get_intrinsics(cam_id)
        pts_px = np.array([lm["pixel"]], dtype=np.float64)
        world_gt = np.array([find_region_vertex(regions, lm["region"], lm["vertex_idx"])], dtype=np.float64)

        world_xy, valid = unproject_to_floor(pts_px, R, t, fx, fy, cx, cy, xi, alpha, floor_z=0.0)
        err = float(np.linalg.norm(world_xy[0] - world_gt[0])) if valid[0] else float("nan")

        rows.append({"camera": cam_id, "region": lm["region"], "vertex_idx": lm["vertex_idx"],
                      "label": lm["label"], "err": err})
        region_vertex = f"{lm['region']}#{lm['vertex_idx']}"
        print(f"  {cam_id:>8} | {region_vertex:>24} | {err:>9.3f} | {lm['label']}")

    errs = np.array([r["err"] for r in rows])
    print(f"\n=== 全体サマリー ===")
    print(f"件数: {len(errs)} / 平均誤差: {np.nanmean(errs):.3f}m / 中央値: {np.nanmedian(errs):.3f}m / "
          f"最大: {np.nanmax(errs):.3f}m / RMSE: {np.sqrt(np.nanmean(errs ** 2)):.3f}m")

    print(f"\n=== カメラ別サマリー ===")
    by_cam: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        by_cam[r["camera"]].append(r["err"])
    for cam_id in ["Camera_1", "Camera_2", "Camera_3"]:
        if cam_id not in by_cam:
            continue
        e = np.array(by_cam[cam_id])
        print(f"{cam_id}: 件数={len(e)} 平均={np.nanmean(e):.3f}m 最大={np.nanmax(e):.3f}m "
              f"RMSE={np.sqrt(np.nanmean(e ** 2)):.3f}m")

    print(f"\n(参考: 32点全部を使ってチェーン較正した場合[calibrate_camera_from_landmarks.py, "
          f"Mode3]は 平均0.296m / RMSE0.329m)")


if __name__ == "__main__":
    main()