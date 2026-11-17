"""
report_lm_and_cow_error_ver1.py

校正 B(現在値) / C(ver7の最終版) / D(Mode3) について、2つの指標を出力する。
  [LM誤差]       43点のランドマークを床(Z=0)に投影したときの位置誤差 [m]
  [牛の投影誤差] 牛のクリックを三角測量したときの再投影誤差 [擬似px] (カメラ間の整合)
学習に使ったデータは表示に明記する(学習済みの誤差は検証ではない):
  B : どちらも学習に使っていない
  C : 43点のLM と フレーム100の牛 で学習
  D : 43点のLM で学習 (牛は未使用 -> 牛の誤差は未使用データでの値)

使い方: python report_lm_and_cow_error_ver1.py [フレーム ...]     (牛のフレーム。省略時は保存済みの全フレーム)
  フレーム100以外の牛を足すと、C の牛の誤差も未使用データでの値になる(そのフレームの分)。
出力: bev_calibration_refined/report_lm_and_cow_error_ver1.csv (ランドマークごと)
配置: research/phase4/calibration/
"""

import csv
import json
import os
import sys

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

import refine_calibration_ba_ver1 as b1
import refine_calibration_ba_ver2 as b2
import refine_calibration_ba_ver3 as b3

C_JSON_NAME = "calibration_refined_cv_ver7.json"
TRAIN_NOTE = {
    "B 現在値": "LM:未使用 / 牛:未使用",
    "C ver7最終版": "LM:43点で学習 / 牛:フレーム100で学習",
    "D Mode3": "LM:43点で学習 / 牛:未使用",
}
D_PARAM_OVERRIDES = {
    "Area1->Camera_1": [7.5458, 6.2912, -4.9393, -0.0302, -1.113, -0.9927],
    "Camera_1->Camera_2": [13.5447, 0.4388, -0.397, -7.1617, 1.2404, -0.1662],
    "Camera_2->Camera_3": [13.5215, -0.1578, -0.3039, 11.1117, 0.2251, -0.2347],
}
D_XI_ALPHA = {"Camera_1": (-0.065738, 0.519379), "Camera_2": (-0.093895, 0.518379), "Camera_3": (-0.060158, 0.500745)}


def stats(a):
    a = np.asarray(a, dtype=float)
    a = a[~np.isnan(a)]
    return dict(n=len(a), mean=a.mean(), med=np.median(a), p90=np.percentile(a, 90), max=a.max(),
                rmse=float(np.sqrt(np.mean(a ** 2))))


def main():
    import phase3_bev_multicam_gt_zonepose_ver1 as g
    from Mootrack_core import (DS_CX, DS_CY, DS_FX, DS_FY, ROT_FMT, apply_camera_overrides, get_intrinsics,
                               load_floorplan, resolve_poses)
    from evaluate_refined_calib_on_all_landmarks_ver3 import LANDMARKS, find_region_vertex

    out_dir = g.OUTPUT_DIR / "bev_calibration_refined"
    regions, cameras_json, _w, _h = load_floorplan(g.FLOORPLAN_JSON_PATH)
    cam_ids = list(g.CAM_IDS)
    poses_B = resolve_poses(apply_camera_overrides(cameras_json), root="Area1", fmt=ROT_FMT)
    B = {c: (poses_B[c].R.copy(), poses_B[c].t.copy(), np.array(get_intrinsics(c), dtype=np.float64)) for c in cam_ids}
    ref = json.loads((out_dir / C_JSON_NAME).read_text(encoding="utf-8"))
    C = {c: (np.array(ref["cameras"][c]["R"]), np.array(ref["cameras"][c]["t"]),
             np.array([ref["cameras"][c]["intrinsics"][nm] for nm in b1.INTR_NAMES], dtype=np.float64)) for c in cam_ids}
    poses_D = resolve_poses(apply_camera_overrides(cameras_json, D_PARAM_OVERRIDES), root="Area1", fmt=ROT_FMT)
    D = {c: (poses_D[c].R.copy(), poses_D[c].t.copy(),
             np.array([DS_FX, DS_FY, DS_CX, DS_CY, *D_XI_ALPHA[c]], dtype=np.float64)) for c in cam_ids}
    settings = {"B 現在値": B, "C ver7最終版": C, "D Mode3": D}

    landmarks = b2.build_landmarks(regions, LANDMARKS, find_region_vertex)
    for lm, d in zip(landmarks, LANDMARKS):
        lm["name"] = f'{d["region"]}#{d["vertex_idx"]}'
    _g, _cj, _ci, _base, _lms, cows, frames, _od = b3.load_data()

    err_lm = {k: b2.landmark_errors(s, landmarks) for k, s in settings.items()}
    err_cow = {k: b1.evaluate_cows(s, cows)["reproj_px"] for k, s in settings.items()}
    cam_of = np.array([lm["cam"] for lm in landmarks])

    print(f"ランドマーク {len(landmarks)}点 / 牛 {len(cows)}頭 (フレーム {frames}, 観測 {len(err_cow['B 現在値'])}件)")
    print("\n=== LM誤差 [m] (床の位置誤差) ===")
    print(f"{'校正':<14}{'RMSE':>7}{'平均':>7}{'中央値':>8}{'90%点':>8}{'最大':>7}   学習に使ったデータ")
    for k in settings:
        s = stats(err_lm[k])
        print(f"{k:<14}{s['rmse']:>7.3f}{s['mean']:>7.3f}{s['med']:>8.3f}{s['p90']:>8.3f}{s['max']:>7.3f}   {TRAIN_NOTE[k]}")
    print("  カメラ別 RMSE [m]:")
    for k in settings:
        print(f"    {k:<14}" + "  ".join(f"{c}: {stats(err_lm[k][cam_of == c])['rmse']:.3f}" for c in cam_ids))

    print("\n=== 牛の投影誤差 [擬似px] (カメラ間の整合) ===")
    print(f"{'校正':<14}{'中央値':>8}{'平均':>8}{'90%点':>8}{'最大':>8}")
    for k in settings:
        s = stats(err_cow[k])
        print(f"{k:<14}{s['med']:>8.2f}{s['mean']:>8.2f}{s['p90']:>8.2f}{s['max']:>8.2f}")
    print("\n※ 学習に使ったデータでの誤差は、検証ではなく、あてはまりの良さです。")
    print("  牛の誤差: D は未使用データでの値。C は、フレーム100の牛を学習に使っているので、100だけで測ると学習済みの値になります。")
    print("  C を牛で公平に測るには、別フレームの牛を足してください(引数にそのフレーム番号)。")

    with open(out_dir / "report_lm_and_cow_error_ver1.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["camera", "landmark", "B_lm_err_m", "C_lm_err_m", "D_lm_err_m"])
        for i, lm in enumerate(landmarks):
            w.writerow([lm["cam"], lm["name"]] + [f"{err_lm[k][i]:.4f}" for k in settings])
    print(f"\nCSV: {out_dir / 'report_lm_and_cow_error_ver1.csv'}")


if __name__ == "__main__":
    main()