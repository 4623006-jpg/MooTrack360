"""
refine_calibration_ba_ver5.py

ver4 の比較は不公平だった: 現在の校正は32点すべてを使って作られているため、検証用ランドマークでの
誤差は現在の校正にとって「学習済み」で、BAより有利になる。ver5 で直す点:
  1. 比較の基準を「同じ調整用ランドマークだけでBAした結果(ランドマークのみ)」にする
     → 「牛を足すと良くなるか」を、同じ条件で公平に比べられる
  2. 調整用ランドマークは画面の端を優先して、各カメラ N_TRAIN_PER_CAM 点(最低6点)選ぶ
     [固定分割] 画面中心(cx,cy)から遠い順に上位N点
     [ランダム分割] 中心からの距離^2 に比例する確率でN点(15回)
  3. 現在の校正は「参考」として表示する(検証用ランドマークは学習済みなので、床誤差の比較には使えない)。
     牛の再投影誤差は現在の校正にとっても未使用のデータなので、そちらは公平に比べられる
  4. 採用判定: 「牛+LM」が「LMのみ」より床誤差で悪化せず(比<=1.05)、牛の再投影誤差が現在の校正より改善
     し、かつ「LMのみ」より牛の再投影誤差が改善している場合に出力する

使い方: python refine_calibration_ba_ver5.py [フレーム ...]
配置: research/phase4/calibration/ (ver2, ver3, ver4 と同じ場所)
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
import refine_calibration_ba_ver4 as b4

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
EXCLUDE_LANDMARKS = {"Feeding_Bay_3#1"}      # 3カメラで矛盾した点のみ除外。Feeding_Bay_2#0 は使う(Camera_1の点数確保)
N_TRAIN_PER_CAM = 6                          # 各カメラの調整用ランドマーク数(最低6)
N_REPEATS = 15
COW_TEST_FRAC = 0.25
SEED = 0
VARIANTS = [
    ("外部のみ・標準prior", (), 1.0, 0.15),
    ("外部のみ・強prior", (), 0.5, 0.05),
    ("内部込み・強prior", ("fx", "fy", "cx", "cy"), 0.5, 0.05),
]
WEIGHTS = [("LMのみ", 0.0, 1.0), ("牛+LM x0.5", 1.0, 0.5), ("牛+LM x1", 1.0, 1.0), ("牛+LM x2", 1.0, 2.0)]
ADOPT_RMSE_RATIO_VS_LM_ONLY = 1.05
b4.VARIANTS = VARIANTS


def edge_dist(lm, base):
    intr = base[lm["cam"]][2]
    return float(np.hypot(lm["uv"][0] - intr[2], lm["uv"][1] - intr[3]))


def edge_pick_fixed(landmarks, base, n):
    out = []
    for cam in base:
        idx = [i for i, lm in enumerate(landmarks) if lm["cam"] == cam]
        idx.sort(key=lambda i: -edge_dist(landmarks[i], base))
        out += idx[:n]
    return sorted(out)


def edge_pick_random(landmarks, base, n, rng):
    out = []
    for cam in base:
        idx = [i for i, lm in enumerate(landmarks) if lm["cam"] == cam]
        p = np.array([edge_dist(landmarks[i], base) ** 2 for i in idx])
        out += rng.choice(idx, size=min(n, len(idx)), replace=False, p=p / p.sum()).tolist()
    return sorted(int(i) for i in out)


def run_split(base, cows, landmarks, train_lm, train_cow):
    trs, tcs = set(train_lm), set(train_cow)
    tr_l = [landmarks[i] for i in train_lm]
    te_idx = [i for i in range(len(landmarks)) if i not in trs]
    te_l = [landmarks[i] for i in te_idx]
    tr_c = [cows[i] for i in train_cow]
    te_c = [cows[i] for i in range(len(cows)) if i not in tcs]
    ev0 = b1.evaluate_cows(base, te_c)
    out = {("現在の校正(参考)", ""): dict(te_idx=te_idx, e_l=b2.landmark_errors(base, te_l),
                                      tr_l=b2.landmark_errors(base, tr_l), rp=ev0["reproj_px"])}
    for v in VARIANTS:
        for wn, w_cow, w_lm in WEIGHTS:
            cams = b4.adjust(base, tr_c, tr_l, v, w_cow, w_lm)
            ev = b1.evaluate_cows(cams, te_c)
            out[(v[0], wn)] = dict(te_idx=te_idx, e_l=b2.landmark_errors(cams, te_l),
                                   tr_l=b2.landmark_errors(cams, tr_l), rp=ev["reproj_px"])
    return out


def main():
    g, cameras_json, cam_ids, base, landmarks, cows, frames, out_dir = b3.load_data()
    n0 = len(landmarks)
    landmarks = [lm for lm in landmarks if lm["name"] not in EXCLUDE_LANDMARKS]
    per = {c: sum(lm["cam"] == c for lm in landmarks) for c in cam_ids}
    print(f"フレーム {frames}: 牛 {len(cows)}頭 / ランドマーク {n0}点 -> 除外後 {len(landmarks)}点 {per}")
    if len(cows) < 8:
        raise SystemExit("牛の数が少なすぎます")
    rng = np.random.default_rng(SEED)
    n_te_cow = max(2, int(round(len(cows) * COW_TEST_FRAC)))

    def cow_train_idx():
        te = set(rng.choice(len(cows), size=n_te_cow, replace=False).tolist())
        return [i for i in range(len(cows)) if i not in te]

    # ---- 固定分割(画面の端の上位N点) ----
    train_lm = edge_pick_fixed(landmarks, base, N_TRAIN_PER_CAM)
    res = run_split(base, cows, landmarks, train_lm, cow_train_idx())
    te_n = len(landmarks) - len(train_lm)
    print(f"\n=== [固定分割] 調整用 {len(train_lm)}点(各カメラの画面端から{N_TRAIN_PER_CAM}点) / 検証用 {te_n}点 ===")
    print("調整用: " + ", ".join(f'{landmarks[i]["cam"][-1]}:{landmarks[i]["name"]}' for i in train_lm))
    print("検証用: " + ", ".join(f'{landmarks[i]["cam"][-1]}:{landmarks[i]["name"]}' for i in res[("現在の校正(参考)", "")]["te_idx"]))
    print(f"\n{'案':<20}{'重み':<12}{'検証LM RMSE[m]':>15}{'最大[m]':>9}{'牛再投影中央値[px]':>18}")
    for k, r in res.items():
        print(f"{k[0]:<20}{k[1]:<12}{b2.rmse(r['e_l']):>15.3f}{np.nanmax(r['e_l']):>9.3f}{np.median(r['rp']):>18.2f}")

    # ---- ランダム分割(端ほど選ばれやすい) ----
    keys = [k for k in res if k[0] != "現在の校正(参考)"]
    agg = {k: {"rmse": [], "rp": []} for k in keys}
    cur_rmse, cur_rp = [], []
    for _ in range(N_REPEATS):
        tl = edge_pick_random(landmarks, base, N_TRAIN_PER_CAM, rng)
        rs = run_split(base, cows, landmarks, tl, cow_train_idx())
        cur = rs[("現在の校正(参考)", "")]
        cur_rmse.append(b2.rmse(cur["e_l"]))
        cur_rp.append(float(np.median(cur["rp"])))
        for k in keys:
            agg[k]["rmse"].append(b2.rmse(rs[k]["e_l"]))
            agg[k]["rp"].append(float(np.median(rs[k]["rp"])))
    cr, crp = float(np.mean(cur_rmse)), float(np.mean(cur_rp))
    print(f"\n=== [ランダム分割 {N_REPEATS}回] 現在の校正(参考。LMは学習済み): LM RMSE {cr:.3f} m / 牛再投影 {crp:.2f} px ===")
    print(f"{'案':<20}{'重み':<12}{'LM RMSE平均':>12}{'標準偏差':>9}{'LMのみ比':>9}{'牛再投影[px]':>13}{'現在比(牛)':>11}")
    table = {}
    lm_only = {v[0]: float(np.mean(agg[(v[0], "LMのみ")]["rmse"])) for v in VARIANTS}
    lm_only_rp = {v[0]: float(np.mean(agg[(v[0], "LMのみ")]["rp"])) for v in VARIANTS}
    for k in keys:
        m, s = float(np.mean(agg[k]["rmse"])), float(np.std(agg[k]["rmse"]))
        rp = float(np.mean(agg[k]["rp"]))
        table[k] = (m, s, m / lm_only[k[0]], rp, rp / crp)
        print(f"{k[0]:<20}{k[1]:<12}{m:>12.3f}{s:>9.3f}{m / lm_only[k[0]]:>9.2f}{rp:>13.2f}{rp / crp:>11.2f}")

    with open(out_dir / "calibration_cv_ver5.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant", "weights", "lm_rmse_mean", "lm_rmse_std", "ratio_to_lm_only", "cow_reproj_px", "cow_ratio_to_current"])
        w.writerow(["現在の校正(参考)", "", f"{cr:.4f}", "", "", f"{crp:.3f}", "1.00"])
        for k, (m, s, rr, rp, rc) in table.items():
            w.writerow([k[0], k[1], f"{m:.4f}", f"{s:.4f}", f"{rr:.3f}", f"{rp:.3f}", f"{rc:.3f}"])

    # ---- 採用判定: 牛を足した案のうち、LMのみと比べて床誤差が悪化せず、牛の整合が良くなる案 ----
    cands = [(k, v) for k, v in table.items() if k[1] != "LMのみ" and v[2] <= ADOPT_RMSE_RATIO_VS_LM_ONLY
             and v[3] < lm_only_rp[k[0]] and v[4] < 1.0]
    print("\n--- 判定 ---")
    best_lm_only = min(VARIANTS, key=lambda v: lm_only[v[0]])
    print(f"同条件の「LMのみ」の最良: {best_lm_only[0]} (検証LM RMSE {lm_only[best_lm_only[0]]:.3f} m)")
    if not cands:
        print("「牛を足すと、床誤差を悪化させずに牛の整合が良くなる」案はありませんでした。json/txtは出力しません。")
        print("→ 牛のクリットはキャリブレーションの追加情報として使えていません。現在の校正を使ってください。")
        return
    k, v = min(cands, key=lambda kv: kv[1][0])
    print(f"採用候補: {k[0]} / {k[1]} (LM RMSE {v[0]:.3f} m, LMのみ比 {v[2]:.2f}, 牛再投影 {v[3]:.2f}px, 現在比 {v[4]:.2f})")

    variant = next(x for x in VARIANTS if x[0] == k[0])
    _n, w_cow, w_lm = next(x for x in WEIGHTS if x[0] == k[1])
    cams = b4.adjust(base, cows, landmarks, variant, w_cow, w_lm)
    e_in = b2.landmark_errors(cams, landmarks)
    e_cur = b2.landmark_errors({c: base[c] for c in cam_ids}, landmarks)
    print(f"全データで最終調整。全ランドマーク(学習に使用)の床誤差 RMSE: 現在の校正 {b2.rmse(e_cur):.3f} m -> {b2.rmse(e_in):.3f} m")
    for c in cam_ids:
        R0, t0, _i0 = base[c]
        R1, t1, _i1 = cams[c]
        rot_deg = float(np.degrees(np.linalg.norm(Rot.from_matrix(R0.T @ R1).as_rotvec())))
        print(f"  {c}: 向き {rot_deg:.3f}°  位置 {np.round(t1 - t0, 4).tolist()} m")
    ov = b1.overrides_from_cams(cameras_json, cams)
    intr_txt = {c: {nm: round(float(cams[c][2][i]), 6) for i, nm in enumerate(b1.INTR_NAMES)} for c in cam_ids}
    snippet = ["# Mootrack_core.py に貼る内容(ver5)", f"# 案: {k[0]} / {k[1]}", "CAMERA_PARAM_OVERRIDES = {"] + \
              [f'    "{a}": {b},' for a, b in ov.items()] + ["}", "", "CAM_INTRINSICS = {"] + \
              [f'    "{c}": {x},' for c, x in intr_txt.items()] + ["}"]
    (out_dir / "calibration_refined_cv_ver5.txt").write_text("\n".join(snippet), encoding="utf-8")
    (out_dir / "calibration_refined_cv_ver5.json").write_text(json.dumps({
        "frames": frames, "variant": k[0], "weights": k[1], "excluded": sorted(EXCLUDE_LANDMARKS),
        "cameras": {c: {"R": cams[c][0].tolist(), "t": cams[c][1].tolist(),
                        "intrinsics": dict(zip(b1.INTR_NAMES, map(float, cams[c][2])))} for c in cam_ids},
        "overrides": ov}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"出力: {out_dir}")


if __name__ == "__main__":
    main()