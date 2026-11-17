"""
evaluate_refined_calib_on_all_landmarks_ver3.py

ver2 の32点に、calibrate_camera_from_landmarks.py の追加ランドマーク11点(計43点)を加えて評価する。
  - 43点は Mode3(D)の校正に使った全点。B(現在値)・A は校正にランドマークを使っていない/別の点で作成。
  - C(BA)が学習に使ったのは元の32点(除外1点)のみ。追加の11点は C にとって未使用。
  最後に「元の32点」と「追加11点」に分けた RMSE も出す。

ver1 に D(calibrate_camera_from_landmarks.py の Mode3 チェーン較正の結果)を追加。
A: 6点較正 / B: Mootrack_core現在値 / C: バンドル調整後(REFINED_JSON_PATH) / D: Mode3(full)
32点の床のランドマークに当てはめたときの誤差を出す。

使い方: python evaluate_refined_calib_on_32landmarks_ver2.py
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# 配置: research/phase4/calibration/(このファイル), research/phase4/web_annotate/, research/ 直下に他のフォルダ
RESEARCH_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
PHASE4_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
MOOTRACK_CODE_DIR = (os.path.abspath(os.path.join(RESEARCH_DIR, "calibration")))
for d in (RESEARCH_DIR, PHASE4_DIR, os.path.join(PHASE4_DIR, "web_annotate"),
          MOOTRACK_CODE_DIR,
          os.path.join(RESEARCH_DIR, "analyze"), os.path.join(RESEARCH_DIR, "annotate"),
          os.path.join(RESEARCH_DIR, "phase3_multi_camera"), THIS_DIR):
    if os.path.isdir(d) and d not in sys.path:
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
                     "calibration_process/research/idf1_check/bev_calibration_refined/calibration_refined_cv_ver7.json")
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


# D: calibrate_camera_from_landmarks.py の Mode3 (CHAIN_MODE="full") の出力
D_PARAM_OVERRIDES = {
    "Area1->Camera_1": [7.5458, 6.2912, -4.9393, -0.0302, -1.113, -0.9927],
    "Camera_1->Camera_2": [13.5447, 0.4388, -0.397, -7.1617, 1.2404, -0.1662],
    "Camera_2->Camera_3": [13.5215, -0.1578, -0.3039, 11.1117, 0.2251, -0.2347],
}
D_INTRINSICS = {
    "Camera_1": {"xi": -0.065738, "alpha": 0.519379},
    "Camera_2": {"xi": -0.093895, "alpha": 0.518379},
    "Camera_3": {"xi": -0.060158, "alpha": 0.500745},
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


# calibrate_camera_from_landmarks.py の追加ランドマーク11点(143〜153行)
LANDMARKS_EXTRA = [
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
N_BASE_LANDMARKS = len(LANDMARKS)
LANDMARKS = LANDMARKS + LANDMARKS_EXTRA


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

    # D: Mode3 (full)
    cj_D = []
    for entry in cameras_json_raw:
        e = dict(entry)
        key = f"{entry['parent']}->{entry['child']}"
        if key in D_PARAM_OVERRIDES:
            e["params"] = list(D_PARAM_OVERRIDES[key])
        cj_D.append(e)
    poses_D = resolve_poses(cj_D, root="Area1", fmt=ROT_FMT)
    setting_D = {c: (poses_D[c].R, poses_D[c].t,
                     np.array([DS_FX, DS_FY, DS_CX, DS_CY, D_INTRINSICS[c]["xi"], D_INTRINSICS[c]["alpha"]], dtype=np.float64))
                 for c in CAM_IDS}

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
    print(f"C のパラメータ: {REFINED_JSON_PATH}")

    proj = {k: project_landmarks(s, regions) for k, s in (("A", setting_A), ("B", setting_B), ("C", setting_C), ("D", setting_D))}
    err = {k: errors_of(v) for k, v in proj.items()}

    print("\n=== ランドマークごとの誤差[m] (A:6点 / B:現在値 / C:BA / D:Mode3) ===")
    print(f"  {'camera':>8} | {'region/vertex':>24} | {'A':>7} | {'B':>7} | {'C':>7} | {'D':>7} | {'D-B':>7} | 備考")
    for k, lm in enumerate(LANDMARKS):
        rv = f"{lm['region']}#{lm['vertex_idx']}"
        d = err["D"][k] - err["B"][k]
        print(f"  {lm['camera']:>8} | {rv:>24} | {err['A'][k]:7.3f} | {err['B'][k]:7.3f} | {err['C'][k]:7.3f} | {err['D'][k]:7.3f} | {d:+7.3f} | {lm['label']}")

    print("\n=== 全体サマリー ===")
    for k, name in (("A", "A 6点較正スクリプト   "), ("B", "B Mootrack_core現在値 "), ("C", "C バンドル調整後     "), ("D", "D Mode3(full)        ")):
        print("  " + summary(name, err[k]))
    for k in ("C", "D"):
        print(f"  {k} が B より小さい点: {int(np.sum(err[k] < err['B']))}点 / 大きい点: {int(np.sum(err[k] > err['B']))}点")

    print("\n=== カメラ別サマリー ===")
    for c in CAM_IDS:
        idx = [k for k, lm in enumerate(LANDMARKS) if lm["camera"] == c]
        print(f"[{c}] {len(idx)}点")
        for k, name in (("A", "A"), ("B", "B"), ("C", "C"), ("D", "D")):
            e = err[k][idx]
            s = signed_of(proj[k])[idx]
            print(f"    {name}: 平均={np.nanmean(e):.3f}m RMSE={np.sqrt(np.nanmean(e ** 2)):.3f}m  "
                  f"符号つき平均 dx={np.nanmean(s[:, 0]):+.3f} dy={np.nanmean(s[:, 1]):+.3f} m")

    r = {k: np.sqrt(np.nanmean(err[k] ** 2)) for k in err}
    print(f"\nRMSE: A {r['A']:.3f} / B {r['B']:.3f} / C {r['C']:.3f} / D {r['D']:.3f} m")
    nb = N_BASE_LANDMARKS
    print(f"\n=== 内訳 (元の{nb}点 / 追加{len(LANDMARKS) - nb}点) RMSE[m] ===")
    for k, name in (("A", "A 6点"), ("B", "B 現在値"), ("C", "C BA"), ("D", "D Mode3")):
        e1, e2 = err[k][:nb], err[k][nb:]
        print(f"  {name:<10} 元の{nb}点 {np.sqrt(np.nanmean(e1 ** 2)):.3f} / 追加{len(e2)}点 {np.sqrt(np.nanmean(e2 ** 2)):.3f}")
    print("※ Mode3(D)は43点すべてを学習に使っています。C は追加11点を使っていません(未使用)。B は校正にこれらを使っていません。"
          "学習に使った点での誤差なので、未使用データでの精度ではありません。")


if __name__ == "__main__":
    main()