"""
refine_calibration_ba_ver8.py

「各カメラの画面の端にあるランドマーク + 牛」だけで校正を作り、
使っていない点(内側のランドマーク / 外した牛)での精度を測る。

  調整用ランドマーク : 各カメラで画像中心(cx,cy)から遠い順に N_TRAIN_PER_CAM 点 (端) -- 固定
  検証用ランドマーク : 残り(内側)。全カメラ合計 約25点。校正に一切使わない
  牛                 : K分割。検証用に外した牛は校正に使わない (全牛が1回ずつ検証用になる)
  初期値             : B(Mootrack_core.py の現在値)
  指標               : LM誤差[m] = 検証用ランドマークの床位置誤差 / 牛の投影誤差[擬似px] = 外した牛の再投影誤差
  比べる相手         : B / 「LMのみ」(端のLMだけで校正) / D(Mode3, 43点で学習済みなので参考)
最後に、最良の手順を 端のLM + 全牛 で作り直した校正 (E) を出力し、
E の検証用ランドマークでの誤差(これは未使用の点での値)も出す。

配置: research/phase4/calibration/ (ver1〜ver7 と同じ場所)
使い方: python refine_calibration_ba_ver8.py [フレーム ...]
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
import refine_calibration_ba_ver5 as b5
import refine_calibration_ba_ver7 as b7

N_TRAIN_PER_CAM = 6        # 各カメラの調整用(端)ランドマーク数 (最低6)
K_COW_FOLDS = 5
REPEATS = 3
SEED = 0


def split_by_edge(landmarks, base, n):
    train = b5.edge_pick_fixed(landmarks, base, n)
    test = [i for i in range(len(landmarks)) if i not in set(train)]
    return train, test


def evaluate(base, cows, landmarks, train_idx, test_idx, rng):
    """戻り値: 検証用LMごとの誤差 (設定ごと, 繰り返し平均) と 外した牛の再投影誤差"""
    tr_l = [landmarks[i] for i in train_idx]
    te_l = [landmarks[i] for i in test_idx]
    keys = [(v[0], w[0]) for v in b7.VARIANTS for w in b7.WEIGHTS]
    e_lm = {k: np.zeros(len(te_l)) for k in keys}
    rp = {k: [] for k in keys}
    lm_only_cache = {}
    for _ in range(REPEATS):
        cow_fold = b2.make_folds(len(cows), K_COW_FOLDS, rng)
        for f in range(K_COW_FOLDS):
            tr_c = [c for c, fo in zip(cows, cow_fold) if fo != f]
            te_c = [c for c, fo in zip(cows, cow_fold) if fo == f]
            for v in b7.VARIANTS:
                for wn, w_cow, w_lm in b7.WEIGHTS:
                    k = (v[0], wn)
                    if w_cow == 0.0:
                        if k not in lm_only_cache:                       # LMのみは牛に依存しないので1回だけ計算
                            lm_only_cache[k] = b7.adjust(base, [], tr_l, v, 0.0, w_lm)
                        cams = lm_only_cache[k]
                    else:
                        cams = b7.adjust(base, tr_c, tr_l, v, w_cow, w_lm)
                    e_lm[k] += b2.landmark_errors(cams, te_l) / (REPEATS * K_COW_FOLDS)
                    rp[k].append(b1.evaluate_cows(cams, te_c)["reproj_px"])
    return e_lm, {k: np.concatenate(v) for k, v in rp.items()}


def main():
    import phase3_bev_multicam_gt_zonepose_ver1 as g
    from Mootrack_core import (DS_CX, DS_CY, DS_FX, DS_FY, ROT_FMT, apply_camera_overrides, get_intrinsics,
                               load_floorplan, resolve_poses)
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
    _g, _cj, _ci, _base, _lms, cows, frames, _od = b3.load_data()

    train_idx, test_idx = split_by_edge(landmarks, base, N_TRAIN_PER_CAM)
    te_l = [landmarks[i] for i in test_idx]
    print(f"フレーム {frames}: 牛 {len(cows)}頭 / ランドマーク {len(landmarks)}点 -> 調整用(端) {len(train_idx)}点 / 検証用(内側) {len(te_l)}点")
    print("調整用(端): " + ", ".join(f'{landmarks[i]["cam"][-1]}:{landmarks[i]["name"]}({b5.edge_dist(landmarks[i], base):.0f}px)' for i in train_idx))
    print("検証用(内側): " + ", ".join(f'{lm["cam"][-1]}:{lm["name"]}' for lm in te_l))
    rng = np.random.default_rng(SEED)

    # B と D (参考)
    e_B = b2.landmark_errors(base, te_l)
    poses_D = resolve_poses(apply_camera_overrides(cameras_json, b7.D_PARAM_OVERRIDES), root="Area1", fmt=ROT_FMT)
    cams_D = {c: (poses_D[c].R.copy(), poses_D[c].t.copy(),
                  np.array([DS_FX, DS_FY, DS_CX, DS_CY, *b7.D_XI_ALPHA[c]], dtype=np.float64)) for c in cam_ids}
    e_D = b2.landmark_errors(cams_D, te_l)
    rp_B = b1.evaluate_cows(base, cows)["reproj_px"]
    rp_D = b1.evaluate_cows(cams_D, cows)["reproj_px"]

    e_lm, rp = evaluate(base, cows, landmarks, train_idx, test_idx, rng)
    cam_of = np.array([lm["cam"] for lm in te_l])

    def line(name, e, r, pb=None, pl=None, note=""):
        s = np.asarray(e)
        pbs = "-" if pb is None else f"{pb:.0%}"
        pls = "-" if pl is None else f"{pl:.0%}"
        print(f"{name:<26}{b7.rmse(s):>8.3f}{np.median(s):>8.3f}{np.max(s):>7.3f}{pbs:>10}{pls:>14}"
              f"{np.median(r):>9.2f}{np.mean(r):>7.2f}{np.percentile(r, 90):>8.2f}  {note}")

    print(f"\n=== 検証用ランドマーク(内側{len(te_l)}点)の LM誤差[m] と 外した牛の投影誤差[px] ===")
    print(f"{'校正':<26}{'LM RMSE':>8}{'中央値':>8}{'最大':>7}{'P(Bに勝つ)':>10}{'P(LMのみに勝つ)':>14}{'牛 中央値':>9}{'平均':>7}{'90%点':>8}")
    line("B 現在値", e_B, rp_B, note="LM:未使用 / 牛:未使用")
    line("D Mode3 (参考)", e_D, rp_D, note="LM:内側も学習済み / 牛:未使用")
    rows = []
    for v in b7.VARIANTS:
        for wn, _wc, _wl in b7.WEIGHTS:
            k = (v[0], wn)
            pb = b7.boot_p(e_lm[k], e_B, rng)
            pl = None if wn == "LMのみ" else b7.boot_p(e_lm[k], e_lm[(v[0], "LMのみ")], rng)
            rows.append((k, b7.rmse(e_lm[k]), pb, pl))
            line(f"{v[0]}/{wn}", e_lm[k], rp[k], pb, pl, note="LM:端のみ学習 / 牛:外した牛")

    print("\n--- 検証用LMのカメラ別 RMSE[m] ---")
    print(f"{'':<26}" + "".join(f"{c:>11}" for c in cam_ids))
    for label, e in [("B 現在値", e_B), ("D Mode3(参考)", e_D)] + [(f"{k[0]}/{k[1]}", e_lm[k]) for k, *_ in rows]:
        print(f"{label:<26}" + "".join(
            f"{b7.rmse(np.asarray(e)[cam_of == c]) if np.any(cam_of == c) else float('nan'):>11.3f}" for c in cam_ids))

    with open(out_dir / "calibration_cv_ver8.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant", "weights", "heldout_lm_rmse_m", "p_beat_B", "p_beat_lm_only", "cow_reproj_median_px"])
        w.writerow(["B現在値", "", f"{b7.rmse(e_B):.4f}", "", "", f"{np.median(rp_B):.3f}"])
        for (k, r, pb, pl) in rows:
            w.writerow([k[0], k[1], f"{r:.4f}", f"{pb:.3f}", "" if pl is None else f"{pl:.3f}", f"{np.median(rp[k]):.3f}"])

    # 最終: 端のLM + 全牛 で作り直し (E)
    best = min((r for r in rows if r[0][1] != "LMのみ"), key=lambda r: r[1])
    k = best[0]
    variant = next(v for v in b7.VARIANTS if v[0] == k[0])
    _n, w_cow, w_lm = next(x for x in b7.WEIGHTS if x[0] == k[1])
    cams = b7.adjust(base, cows, [landmarks[i] for i in train_idx], variant, w_cow, w_lm)
    e_E = b2.landmark_errors(cams, te_l)
    print(f"\n--- E: 最良の牛+LM手順 ({k[0]}/{k[1]}) を 端のLM + 全牛{len(cows)}頭 で作り直した校正 ---")
    print(f"検証用LM({len(te_l)}点, 校正に未使用)の誤差: RMSE {b7.rmse(e_E):.3f} m / 中央値 {np.median(e_E):.3f} / 最大 {np.max(e_E):.3f}  (B: {b7.rmse(e_B):.3f} m)")
    print("牛の誤差は全牛を学習に使っているので、この校正では検証になりません(上の表の「外した牛」の値を使うこと)。")
    ov = b1.overrides_from_cams(cameras_json, cams)
    intr_txt = {c: {nm: round(float(cams[c][2][i]), 6) for i, nm in enumerate(b1.INTR_NAMES)} for c in cam_ids}
    snippet = ["# Mootrack_core.py に貼る内容(ver8: 端のLM + 全牛)", f"# 手順: {k[0]} / {k[1]}", "CAMERA_PARAM_OVERRIDES = {"] + \
              [f'    "{a}": {b},' for a, b in ov.items()] + ["}", "", "CAM_INTRINSICS = {"] + \
              [f'    "{cc}": {x},' for cc, x in intr_txt.items()] + ["}"]
    (out_dir / "calibration_refined_edge_ver8.txt").write_text("\n".join(snippet), encoding="utf-8")
    (out_dir / "calibration_refined_edge_ver8.json").write_text(json.dumps({
        "frames": frames, "variant": k[0], "weights": k[1],
        "train_landmarks": [f'{landmarks[i]["cam"]}:{landmarks[i]["name"]}' for i in train_idx],
        "cameras": {c: {"R": cams[c][0].tolist(), "t": cams[c][1].tolist(),
                        "intrinsics": dict(zip(b1.INTR_NAMES, map(float, cams[c][2])))} for c in cam_ids},
        "overrides": ov}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"出力: {out_dir}")


if __name__ == "__main__":
    main()