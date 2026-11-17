"""
report_edge_lm_cow_calib_ver1.py

ver6 の方法: 「各カメラの画面の端のランドマーク6点(固定) + 牛」で学習したパラメータで、
LM誤差と牛の誤差を出力する。ランドマークの座標の取り方が間違っている可能性も見るため、
全ランドマークの誤差を「学習に使った点 / 使っていない点」を区別して一覧にし、
怪しいランドマーク(複数の校正でも誤差が大きいまま)に印を付ける。

  学習 : 各カメラ画像中心から遠い順に N_TRAIN_PER_CAM 点(固定) + 牛
  校正の手順: ver6 の3案から選ぶ (SETTINGS_TO_REPORT)。初期値は B (Mootrack_core.py の現在値)
  出力 :
    [1] 牛の誤差(検証): 牛をK分割して、外した牛の再投影誤差 (校正に使っていない牛の値)
    [2] LM誤差: 最終パラメータ(端のLM + 全牛で作成)の、ランドマーク別の誤差
        - 学習に使った端の点 / 使っていない内側の点 を区別して集計
        - 内側の点の誤差が、未使用データでの値
    [3] 怪しいランドマーク: B と最終パラメータの両方で誤差が大きい点
    [4] 参考: 牛の誤差(全牛を学習に使った最終パラメータ。検証ではない)
  CSV : bev_calibration_refined/report_edge_lm_cow_calib_ver1.csv

配置: research/phase4/calibration/
使い方: python report_edge_lm_cow_calib_ver1.py [フレーム ...]
"""

import csv
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
import refine_calibration_ba_ver4 as b4
import refine_calibration_ba_ver5 as b5

N_TRAIN_PER_CAM = 6
K_COW_FOLDS = 5
REPEATS = 3
SEED = 0
SUSPICIOUS_M = 0.8          # B と最終パラメータの両方でこれを超えたら「怪しい」
# ver6 の3案 (名前, 動かす内部パラメータ, prior回転[deg], prior位置[m])
VARIANTS = {
    "外部のみ・強prior": ("外部のみ・強prior", (), 0.5, 0.05),
    "外部のみ・標準prior": ("外部のみ・標準prior", (), 1.0, 0.15),
    "内部込み・強prior": ("内部込み・強prior", ("fx", "fy", "cx", "cy"), 0.5, 0.05),
}
WEIGHTS = {"LMのみ": (0.0, 1.0), "牛+LM x1": (1.0, 1.0), "牛+LM x2": (1.0, 2.0)}
# 出力する手順 (ver6 の採用候補 = 外部のみ・強prior / 牛+LM x2。比較用に LMのみ も出す)
SETTINGS_TO_REPORT = [("外部のみ・強prior", "牛+LM x2"), ("外部のみ・強prior", "LMのみ")]


def analyze(base, cows, landmarks, settings, rng, n_train=N_TRAIN_PER_CAM, repeats=REPEATS, k=K_COW_FOLDS):
    train_idx = b5.edge_pick_fixed(landmarks, base, n_train)
    tr_l = [landmarks[i] for i in train_idx]
    out = {"train_idx": train_idx}
    for vname, wname in settings:
        variant, (w_cow, w_lm) = VARIANTS[vname], WEIGHTS[wname]
        held = []                                              # 外した牛の再投影誤差(検証)
        for _ in range(repeats):
            cow_fold = b2.make_folds(len(cows), k, rng)
            for f in range(k):
                tr_c = [c for c, fo in zip(cows, cow_fold) if fo != f]
                te_c = [c for c, fo in zip(cows, cow_fold) if fo == f]
                cams = b4.adjust(base, tr_c, tr_l, variant, w_cow, w_lm)
                held.append(b1.evaluate_cows(cams, te_c)["reproj_px"])
        cams_final = b4.adjust(base, cows, tr_l, variant, w_cow, w_lm)
        out[(vname, wname)] = dict(
            held_cow=np.concatenate(held),
            final_cow=b1.evaluate_cows(cams_final, cows)["reproj_px"],
            lm_err=b2.landmark_errors(cams_final, landmarks), cams=cams_final)
    return out


def main():
    import phase3_bev_multicam_gt_zonepose_ver1 as g
    from Mootrack_core import ROT_FMT, apply_camera_overrides, get_intrinsics, load_floorplan, resolve_poses
    import refine_calibration_ba_ver3 as b3
    from evaluate_refined_calib_on_all_landmarks_ver3 import LANDMARKS, find_region_vertex

    out_dir = g.OUTPUT_DIR / "bev_calibration_refined"
    regions, cameras_json, _w, _h = load_floorplan(g.FLOORPLAN_JSON_PATH)
    poses = resolve_poses(apply_camera_overrides(cameras_json), root="Area1", fmt=ROT_FMT)
    cam_ids = list(g.CAM_IDS)
    base = {c: (poses[c].R.copy(), poses[c].t.copy(), np.array(get_intrinsics(c), dtype=np.float64)) for c in cam_ids}
    landmarks = b2.build_landmarks(regions, LANDMARKS, find_region_vertex)
    for lm, d in zip(landmarks, LANDMARKS):
        lm["name"] = f'{d["region"]}#{d["vertex_idx"]}'
        lm["label"] = d.get("label", "")
    _g, _cj, _ci, _base, _lms, cows, frames, _od = b3.load_data()
    rng = np.random.default_rng(SEED)

    res = analyze(base, cows, landmarks, SETTINGS_TO_REPORT, rng)
    train_idx = res["train_idx"]
    is_train = np.zeros(len(landmarks), dtype=bool)
    is_train[train_idx] = True
    cam_of = np.array([lm["cam"] for lm in landmarks])
    e_B = b2.landmark_errors(base, landmarks)
    rp_B = b1.evaluate_cows(base, cows)["reproj_px"]
    print(f"フレーム {frames}: 牛 {len(cows)}頭 / ランドマーク {len(landmarks)}点 (端で学習 {len(train_idx)}点 / 内側 {int((~is_train).sum())}点)")
    print("学習に使った端の点: " + ", ".join(
        f'{landmarks[i]["cam"][-1]}:{landmarks[i]["name"]}({b5.edge_dist(landmarks[i], base):.0f}px)' for i in train_idx))

    def st(a):
        a = np.asarray(a)
        return f"RMSE {np.sqrt(np.mean(a ** 2)):.3f} / 平均 {np.mean(a):.3f} / 中央値 {np.median(a):.3f} / 最大 {np.max(a):.3f}"

    def st_cow(a):
        return f"中央値 {np.median(a):.2f} / 平均 {np.mean(a):.2f} / 90%点 {np.percentile(a, 90):.2f} / 最大 {np.max(a):.2f}"

    print("\n=== [1] 牛の誤差[px] (検証: 校正に使っていない牛。5分割×3回) ===")
    print(f"  B 現在値                   : {st_cow(rp_B)}")
    for key in SETTINGS_TO_REPORT:
        print(f"  {key[0]}/{key[1]:<10} : {st_cow(res[key]['held_cow'])}")

    for key in SETTINGS_TO_REPORT:
        r = res[key]
        e = r["lm_err"]
        print(f"\n=== [2] LM誤差[m]: {key[0]}/{key[1]} (端のLM{len(train_idx)}点" + (" + 全牛" if key[1] != "LMのみ" else "") + "で学習) ===")
        print(f"  学習に使った端の点 ({int(is_train.sum())}点): {st(e[is_train])}   <- あてはまり")
        print(f"  使っていない内側の点 ({int((~is_train).sum())}点): {st(e[~is_train])}   <- 未使用データでの値")
        print(f"  [B 現在値]  端 {st(e_B[is_train])}\n              内側 {st(e_B[~is_train])}")
        print("  カメラ別 (内側の点) RMSE: " + "  ".join(
            f"{c}: B {np.sqrt(np.mean(e_B[(~is_train) & (cam_of == c)] ** 2)):.3f} -> {np.sqrt(np.mean(e[(~is_train) & (cam_of == c)] ** 2)):.3f}"
            for c in cam_ids if np.any((~is_train) & (cam_of == c))))

    key0 = SETTINGS_TO_REPORT[0]
    e0 = res[key0]["lm_err"]
    print(f"\n=== [3] ランドマーク別の誤差[m] ({key0[0]}/{key0[1]}) ===  ★=B と最終パラメータの両方で {SUSPICIOUS_M} m 超 (座標/クリックの疑い)")
    print(f"  {'camera':>8} {'region/vertex':>22} {'使用':>4} {'B':>7} {'最終':>7} {'差':>7}  label")
    sus = []
    for i, lm in enumerate(landmarks):
        flag = "★" if (e_B[i] > SUSPICIOUS_M and e0[i] > SUSPICIOUS_M) else " "
        if flag == "★":
            sus.append(i)
        print(f"  {lm['cam']:>8} {lm['name']:>22} {'学習' if is_train[i] else '検証':>4} {e_B[i]:7.3f} {e0[i]:7.3f} {e0[i] - e_B[i]:+7.3f} {flag} {lm['label']}")
    print(f"\n怪しいランドマーク(★): " + (", ".join(f'{landmarks[i]["cam"][-1]}:{landmarks[i]["name"]}' for i in sus) if sus else "なし"))

    print("\n=== [4] 参考: 牛の誤差[px] (全牛を学習に使った最終パラメータ。検証ではない) ===")
    for key in SETTINGS_TO_REPORT:
        print(f"  {key[0]}/{key[1]:<10} : {st_cow(res[key]['final_cow'])}")

    csv_path = out_dir / "report_edge_lm_cow_calib_ver1.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["camera", "landmark", "used_in_training", "B_err_m"] +
                   [f"{k[0]}/{k[1]}_err_m" for k in SETTINGS_TO_REPORT])
        for i, lm in enumerate(landmarks):
            w.writerow([lm["cam"], lm["name"], int(is_train[i]), f"{e_B[i]:.4f}"] +
                       [f"{res[k]['lm_err'][i]:.4f}" for k in SETTINGS_TO_REPORT])
    print(f"\nCSV: {csv_path}")


if __name__ == "__main__":
    main()