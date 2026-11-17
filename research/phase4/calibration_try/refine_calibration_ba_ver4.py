"""
refine_calibration_ba_ver4.py

ver3 からの変更:
  1. 疑わしいランドマークを除外 (EXCLUDE_LANDMARKS)。調整にも検証にも使わない
  2. 外部パラメータだけ動かす案を追加。事前分布(prior)を強くした案も比較する (VARIANTS)
  3. 採用の判定を「検証用ランドマークの床誤差が実際に下がったか」で行う
     (ver3 は牛の再投影誤差を混ぜたスコアで選んだため、床誤差が悪化しても良く見えた)
     採用条件: 床誤差RMSE平均が調整なしの ADOPT_RMSE_RATIO 倍以下 かつ
               調整なしより良かった割合が ADOPT_WIN_FRAC 以上 かつ 牛の再投影誤差(中央値)が悪化していない
     条件を満たさなければ json/txt は出力しない(現在の校正のまま)
配置: ver3 と同じフォルダ (research/phase4/calibration/)。
使い方: python refine_calibration_ba_ver4.py [フレーム ...]
"""

import csv
import json
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot

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

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
EXCLUDE_LANDMARKS = {"Feeding_Bay_3#1", "Feeding_Bay_2#0"}   # 調整・検証の両方から除く
N_TRAIN_PER_CAM = {"Camera_1": 3, "Camera_2": 4, "Camera_3": 4}
N_REPEATS = 15
COW_TEST_FRAC = 0.25
SEED = 0
# (名前, 動かす内部パラメータ, prior回転[deg], prior位置[m])
VARIANTS = [
    ("外部のみ・強prior", (), 0.5, 0.05),
    ("外部のみ・標準prior", (), 1.0, 0.15),
    ("内部込み・強prior", ("fx", "fy", "cx", "cy"), 0.5, 0.05),
]
# (名前, w_cow, w_lm)
WEIGHTS = [("牛のみ", 1.0, 0.0), ("牛+LM x0.5", 1.0, 0.5), ("牛+LM x1", 1.0, 1.0), ("牛+LM x2", 1.0, 2.0)]
ADOPT_RMSE_RATIO = 0.9
ADOPT_WIN_FRAC = 0.7
b3.N_TRAIN_PER_CAM = N_TRAIN_PER_CAM


def adjust(base, cows, lms, variant, w_cow, w_lm):
    _n, free, rot, tr = variant
    old = dict(b1.PRIOR)
    b1.PRIOR["rot_deg"], b1.PRIOR["trans_m"] = rot, tr
    try:
        return b2.bundle_adjust2(base, cows, lms, w_cow, w_lm, free_intr=free)[0]
    finally:
        b1.PRIOR.update(old)


def run_split(base, cows, landmarks, train_lm, train_cow):
    trs = set(train_lm)
    tr_l = [landmarks[i] for i in train_lm]
    te_idx = [i for i in range(len(landmarks)) if i not in trs]
    te_l = [landmarks[i] for i in te_idx]
    tcs = set(train_cow)
    tr_c = [cows[i] for i in train_cow]
    te_c = [cows[i] for i in range(len(cows)) if i not in tcs]
    ev0 = b1.evaluate_cows(base, te_c)
    out = {("調整なし", ""): dict(te_idx=te_idx, e_l=b2.landmark_errors(base, te_l), tr_l=b2.landmark_errors(base, tr_l),
                               rp=ev0["reproj_px"])}
    for v in VARIANTS:
        for wn, w_cow, w_lm in WEIGHTS:
            cams = adjust(base, tr_c, tr_l, v, w_cow, w_lm)
            ev = b1.evaluate_cows(cams, te_c)
            out[(v[0], wn)] = dict(te_idx=te_idx, e_l=b2.landmark_errors(cams, te_l),
                                   tr_l=b2.landmark_errors(cams, tr_l), rp=ev["reproj_px"])
    return out


def main():
    g, cameras_json, cam_ids, base, landmarks, cows, frames, out_dir = b3.load_data()
    n0 = len(landmarks)
    landmarks = [lm for lm in landmarks if lm["name"] not in EXCLUDE_LANDMARKS]
    print(f"フレーム {frames}: 牛 {len(cows)}頭 / ランドマーク {n0}点 -> 除外後 {len(landmarks)}点 (除外: {sorted(EXCLUDE_LANDMARKS)})")
    if len(cows) < 8:
        raise SystemExit("牛の数が少なすぎます")
    rng = np.random.default_rng(SEED)
    n_te_cow = max(2, int(round(len(cows) * COW_TEST_FRAC)))

    def cow_train_idx():
        te = set(rng.choice(len(cows), size=n_te_cow, replace=False).tolist())
        return [i for i in range(len(cows)) if i not in te]

    # ---- 固定分割 ----
    train_lm = b3.fixed_pick(landmarks)
    res = run_split(base, cows, landmarks, train_lm, cow_train_idx())
    r0 = res[("調整なし", "")]
    print(f"\n=== [固定分割] 調整用 {len(train_lm)}点 / 検証用 {len(landmarks) - len(train_lm)}点 ===")
    print(f"{'案':<20}{'重み':<14}{'検証LM RMSE[m]':>15}{'最大[m]':>9}{'牛再投影中央値[px]':>18}")
    for k, r in res.items():
        print(f"{k[0]:<20}{k[1]:<14}{b2.rmse(r['e_l']):>15.3f}{np.nanmax(r['e_l']):>9.3f}{np.median(r['rp']):>18.2f}")

    # ---- ランダム分割 ----
    keys = [k for k in res if k[0] != "調整なし"]
    agg = {k: {"rmse": [], "rp": []} for k in keys}
    base_rmse, base_rp = [], []
    for _ in range(N_REPEATS):
        tl = b3.random_pick(landmarks, rng)
        rs = run_split(base, cows, landmarks, tl, cow_train_idx())
        base_rmse.append(b2.rmse(rs[("調整なし", "")]["e_l"]))
        base_rp.append(float(np.median(rs[("調整なし", "")]["rp"])))
        for k in keys:
            agg[k]["rmse"].append(b2.rmse(rs[k]["e_l"]))
            agg[k]["rp"].append(float(np.median(rs[k]["rp"])))
    br, brp = float(np.mean(base_rmse)), float(np.mean(base_rp))
    print(f"\n=== [ランダム分割 {N_REPEATS}回] 調整なし: 検証LM RMSE 平均 {br:.3f} m / 牛再投影 {brp:.2f} px ===")
    print(f"{'案':<20}{'重み':<14}{'RMSE平均':>9}{'標準偏差':>9}{'比(調整なし=1)':>14}{'良かった割合':>12}{'牛再投影[px]':>13}")
    table = []
    for k in keys:
        m, s = float(np.mean(agg[k]["rmse"])), float(np.std(agg[k]["rmse"]))
        win = float(np.mean(np.array(agg[k]["rmse"]) < np.array(base_rmse)))
        rp = float(np.mean(agg[k]["rp"]))
        table.append((k, m, s, m / br, win, rp))
        print(f"{k[0]:<20}{k[1]:<14}{m:>9.3f}{s:>9.3f}{m / br:>14.2f}{win:>12.0%}{rp:>13.2f}")

    best = min(table, key=lambda r: r[1])
    k, m, s, ratio, win, rp = best
    ok = ratio <= ADOPT_RMSE_RATIO and win >= ADOPT_WIN_FRAC and rp <= brp * 1.05
    print(f"\n床誤差が最小の案: {k[0]} / {k[1]} (比 {ratio:.2f}, 良かった割合 {win:.0%}, 牛再投影 {rp:.2f}px)")

    with open(out_dir / "calibration_cv_ver4.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant", "weights", "lm_rmse_mean", "lm_rmse_std", "ratio_to_no_adjust", "win_frac", "cow_reproj_px"])
        w.writerow(["調整なし", "", f"{br:.4f}", "", "1.00", "", f"{brp:.3f}"])
        for (kk, mm, ss, rr, ww, pp) in table:
            w.writerow([kk[0], kk[1], f"{mm:.4f}", f"{ss:.4f}", f"{rr:.3f}", f"{ww:.3f}", f"{pp:.3f}"])

    if not ok:
        print(f"\n不採用: 採用条件(比<={ADOPT_RMSE_RATIO}, 良かった割合>={ADOPT_WIN_FRAC:.0%}, 牛再投影が悪化しない)を満たしません。")
        print("現在の校正(Mootrack_core.py)をそのまま使ってください。json/txtは出力していません。")
        return

    variant = next(v for v in VARIANTS if v[0] == k[0])
    _wn, w_cow, w_lm = next(x for x in WEIGHTS if x[0] == k[1])
    cams = adjust(base, cows, landmarks, variant, w_cow, w_lm)
    print("\n採用条件を満たしました。全データで最終調整")
    for c in cam_ids:
        R0, t0, i0 = base[c]
        R1, t1, i1 = cams[c]
        rot_deg = float(np.degrees(np.linalg.norm(Rot.from_matrix(R0.T @ R1).as_rotvec())))
        print(f"  {c}: 向き {rot_deg:.3f}°  位置 {np.round(t1 - t0, 4).tolist()} m")
    ov = b1.overrides_from_cams(cameras_json, cams)
    intr_txt = {c: {nm: round(float(cams[c][2][i]), 6) for i, nm in enumerate(b1.INTR_NAMES)} for c in cam_ids}
    snippet = ["# Mootrack_core.py に貼る内容(ver4)", f"# 案: {k[0]} / {k[1]}", "CAMERA_PARAM_OVERRIDES = {"] + \
              [f'    "{a}": {b},' for a, b in ov.items()] + ["}", "", "CAM_INTRINSICS = {"] + \
              [f'    "{c}": {v},' for c, v in intr_txt.items()] + ["}"]
    (out_dir / "calibration_refined_cv_ver4.txt").write_text("\n".join(snippet), encoding="utf-8")
    (out_dir / "calibration_refined_cv_ver4.json").write_text(json.dumps({
        "frames": frames, "variant": k[0], "weights": k[1], "excluded": sorted(EXCLUDE_LANDMARKS),
        "cameras": {c: {"R": cams[c][0].tolist(), "t": cams[c][1].tolist(),
                        "intrinsics": dict(zip(b1.INTR_NAMES, map(float, cams[c][2])))} for c in cam_ids},
        "overrides": ov, "rmse_ratio": ratio, "win_frac": win}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"出力: {out_dir}")


if __name__ == "__main__":
    main()