"""
evaluate_refined_calib_on_32landmarks_ver1.py

refine_calibration_ba_ver1.py(牛の背中のクリックを使ったバンドル調整)で得たパラメータを、
32点の床のランドマーク(実測の床面座標が分かっている点)に当てはめたときの誤差を出す。
evaluate_6point_calib_on_32landmarks.py の続き。

ランドマークは床の点(Z=0)で、バンドル調整には使っていない独立のデータなので、
調整後のパラメータが「床の上の位置を正しく出せるか」(絶対的な精度)の検証になる。

比較する3つの設定:
  A. 6点較正スクリプトの値   : evaluate_6point_calib_on_32landmarks.py と同じ(Camera_2のalpha=0.421186など)
  B. Mootrack_core.py の現在値: バンドル調整の出発点(CAMERA_PARAM_OVERRIDES と CAM_INTRINSICS)
  C. バンドル調整後           : calibration_refined_ver1.json の値

見方:
  - C が B より小さい  : 調整で、カメラ間の整合だけでなく、床の絶対位置も良くなった
  - C が B より大きい  : 画像の対応(カメラ間の整合)は良くなったが、全体の位置・大きさ・向きが
                         床の座標からずれた。バンドル調整は、画像の対応だけでは全体の位置・大きさが
                         決まらず、初期値の近くに留めているだけなので起こりうる。
                         その場合は、このランドマーク(床の既知の点)をバンドル調整の拘束に加えると解決しやすい。
  - 符号つきの平均誤差(dx, dy)が大きい: 全体がその方向にずれている(系統誤差)。

使い方(phase4 などに置く。refine_calibration_ba_ver1.py の出力が必要):
    python evaluate_refined_calib_on_32landmarks_ver1.py
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for d in (os.path.join(THIS_DIR, "..", "calibration"), os.path.join(THIS_DIR, "..", "analyze"),
          os.path.join(THIS_DIR, "..", "annotate"), os.path.join(THIS_DIR, "..", "phase3_multi_camera"), THIS_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

from Mootrack_core import (
    ROT_FMT, DS_FX, DS_FY, DS_CX, DS_CY, DS_XI, DS_ALPHA,
    apply_camera_overrides, get_intrinsics, load_floorplan, resolve_poses, unproject_to_floor,
)

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
FLOORPLAN_JSON_PATH = "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/calibration/Farm2_Area_1_Artboard_config_.json"
REFINED_JSON_PATH = ("/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/"
                     "calibration_process/research/idf1_check/bev_calibration_refined/calibration_refined_cv_ver6.json")
CAM_IDS = ["Camera_1", "Camera_2", "Camera_3"]

# A: 6点較正スクリプトの値(evaluate_6point_calib_on_32landmarks.py と同じ)
A_PARAM_OVERRIDES = {
    "Camera_1->Camera_2": [13.113, 0.1269, -0.0743, -7.2382, 1.3079, -0.8149],
    "Camera_2->Camera_3": [13.641, 0.1279, -1.6374, 11.0855, -0.3336, 0.7534],
}
A_INTRINSICS_OVERRIDES = {
    "Camera_2": {"xi": 0.056105, "alpha": 0.421186},
    "Camera_3": {"xi": -0.143895, "alpha": 0.533116},
}


def intrinsics_A(cam_id):
    if cam_id in A_INTRINSICS_OVERRIDES:
        p = A_INTRINSICS_OVERRIDES[cam_id]
        return np.array([DS_FX, DS_FY, DS_CX, DS_CY, p["xi"], p["alpha"]], dtype=np.float64)
    return np.array([DS_FX, DS_FY, DS_CX, DS_CY, DS_XI, DS_ALPHA], dtype=np.float64)


def find_region_vertex(regions, region_name: str, vertex_idx: int) -> Tuple[float, float]:
    for r in regions:
        if r.name == region_name:
            x, y = r.polygon_m[vertex_idx]
            return float(x), float(y)
    raise KeyError(f"region=\"{region_name}\" がfloorplan.jsonに見つかりません")


# 32点のランドマーク(calibrate_camera_from_landmarks.py / verify_camera_calibration.py / 6点評価スクリプトと同一)
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


def project_landmarks(setting, regions):
    """setting: {cam: (R, t, intr[6])}。各ランドマークの (推定xy, 正解xy) を返す(無効はNone)"""
    out = []
    for lm in LANDMARKS:
        R, t, intr = setting[lm["camera"]]
        gt = np.array(find_region_vertex(regions, lm["region"], lm["vertex_idx"]), dtype=np.float64)
        xy, valid = unproject_to_floor(np.array([lm["pixel"]], dtype=np.float64), R, t, *intr, floor_z=0.0)
        out.append((xy[0] if valid[0] else None, gt))
    return out


def errors_of(proj):
    return np.array([float(np.linalg.norm(e - g)) if e is not None else np.nan for e, g in proj])


def signed_of(proj):
    return np.array([(e - g) if e is not None else (np.nan, np.nan) for e, g in proj])


def summary(label, e):
    return (f"{label}: 件数={int(np.sum(~np.isnan(e)))} 平均={np.nanmean(e):.3f}m 中央値={np.nanmedian(e):.3f}m "
            f"最大={np.nanmax(e):.3f}m RMSE={np.sqrt(np.nanmean(e ** 2)):.3f}m")


def main():
    regions, cameras_json_raw, real_w, real_h = load_floorplan(FLOORPLAN_JSON_PATH)
    print(f"床面サイズ: {real_w}m x {real_h}m / 区画数: {len(regions)} / ランドマーク {len(LANDMARKS)}点")

    # A: 6点較正スクリプトの値
    cj_A = []
    for entry in cameras_json_raw:
        e = dict(entry)
        key = f"{entry['parent']}->{entry['child']}"
        if key in A_PARAM_OVERRIDES:
            e["params"] = list(A_PARAM_OVERRIDES[key])
        cj_A.append(e)
    poses_A = resolve_poses(cj_A, root="Area1", fmt=ROT_FMT)
    setting_A = {c: (poses_A[c].R, poses_A[c].t, intrinsics_A(c)) for c in CAM_IDS}

    # B: Mootrack_core.py の現在値
    poses_B = resolve_poses(apply_camera_overrides(cameras_json_raw), root="Area1", fmt=ROT_FMT)
    setting_B = {c: (poses_B[c].R, poses_B[c].t, np.array(get_intrinsics(c), dtype=np.float64)) for c in CAM_IDS}

    # C: バンドル調整後
    ref = json.loads(Path(REFINED_JSON_PATH).read_text(encoding="utf-8"))
    setting_C = {}
    for c in CAM_IDS:
        cam = ref["cameras"][c]
        i = cam["intrinsics"]
        setting_C[c] = (np.array(cam["R"], dtype=np.float64), np.array(cam["t"], dtype=np.float64),
                        np.array([i["fx"], i["fy"], i["cx"], i["cy"], i["xi"], i["alpha"]], dtype=np.float64))
    print(f"調整後のパラメータ: {REFINED_JSON_PATH}\n  (使った牛 train={ref.get('n_train')} 頭, 検証 test={ref.get('n_test')} 頭)")

    proj = {k: project_landmarks(s, regions) for k, s in (("A", setting_A), ("B", setting_B), ("C", setting_C))}
    err = {k: errors_of(v) for k, v in proj.items()}

    print("\n=== ランドマークごとの誤差[m] (A:6点較正スクリプト / B:Mootrack_core現在値 / C:バンドル調整後) ===")
    print(f"  {'camera':>8} | {'region/vertex':>24} | {'A':>7} | {'B':>7} | {'C':>7} | {'C-B':>7} | 備考")
    for k, lm in enumerate(LANDMARKS):
        rv = f"{lm['region']}#{lm['vertex_idx']}"
        d = err["C"][k] - err["B"][k]
        print(f"  {lm['camera']:>8} | {rv:>24} | {err['A'][k]:7.3f} | {err['B'][k]:7.3f} | {err['C'][k]:7.3f} | {d:+7.3f} | {lm['label']}")

    print("\n=== 全体サマリー ===")
    for k, name in (("A", "A 6点較正スクリプト   "), ("B", "B Mootrack_core現在値 "), ("C", "C バンドル調整後     ")):
        print("  " + summary(name, err[k]))
    n_better = int(np.sum(err["C"] < err["B"]))
    n_worse = int(np.sum(err["C"] > err["B"]))
    print(f"  C が B より小さい点: {n_better}点 / 大きい点: {n_worse}点")

    print("\n=== カメラ別サマリー ===")
    for c in CAM_IDS:
        idx = [k for k, lm in enumerate(LANDMARKS) if lm["camera"] == c]
        print(f"[{c}] {len(idx)}点")
        for k, name in (("A", "A"), ("B", "B"), ("C", "C")):
            e = err[k][idx]
            s = signed_of(proj[k])[idx]
            print(f"    {name}: 平均={np.nanmean(e):.3f}m RMSE={np.sqrt(np.nanmean(e ** 2)):.3f}m  "
                  f"符号つき平均 dx={np.nanmean(s[:, 0]):+.3f} dy={np.nanmean(s[:, 1]):+.3f} m")

    print("\n(参考: 32点全部を使ってチェーン較正した場合[calibrate_camera_from_landmarks.py, Mode3]は 平均0.296m / RMSE0.329m)")
    rb, rc = np.sqrt(np.nanmean(err["B"] ** 2)), np.sqrt(np.nanmean(err["C"] ** 2))
    if rc < rb:
        print(f"\n判定: RMSE {rb:.3f}m -> {rc:.3f}m と改善。調整後のパラメータは、床の絶対位置も良くしている。")
    else:
        print(f"\n判定: RMSE {rb:.3f}m -> {rc:.3f}m と悪化。画像の対応は良くなっても、床の座標からずれている可能性。"
              "反映する前に、ランドマークを拘束に入れたバンドル調整を検討すること。")


if __name__ == "__main__":
    main()