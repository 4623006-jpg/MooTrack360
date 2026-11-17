"""
refine_calibration_ba_ver6.py

考え方: 現在の校正は32点すべてで作られているので、どの検証用ランドマークも「学習済み」で有利。
        その現在の校正と、端の点だけ(各カメラN点)+牛で作ったBAが、検証用の内側の点で同等以上なら価値がある。
  - 調整用ランドマーク: 画面の端から各カメラ N_TRAIN_PER_CAM 点(固定)。検証用: 残り(内側の点)
  - 牛: 検証用に25%を取る分割を N_REPEATS 回ランダムに変える(牛のばらつきを見る)
  - 検証用ランドマークは数が少ないので、点のブートストラップ(復元抽出)で
      P(BA < 現在の校正), P(BA < LMのみ)  を出す(点の選び方によるぶれを見る)
  - 牛の再投影誤差: 毎回の分割で、BAが現在の校正に勝った割合を出す(牛は現在の校正にも未使用で公平)
  - 採用条件(すべて満たす案があるときだけ json/txt を出力):
      P(BA < 現在の校正) >= ADOPT_P_LM,  P(BA < LMのみ) >= 0.4 (LMのみに明確に負けない),
      牛の勝率 >= ADOPT_COW_WIN
  - 出力する校正は、全牛+全ランドマーク(除外点以外)で最終調整したもの

配置: research/phase4/calibration/ (ver2〜ver5 と同じ場所)
使い方: python refine_calibration_ba_ver6.py [フレーム ...]
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
import refine_calibration_ba_ver5 as b5

EXCLUDE_LANDMARKS = {"Feeding_Bay_3#1"}
N_TRAIN_PER_CAM = 6
N_REPEATS = 20
N_BOOT = 2000
COW_TEST_FRAC = 0.25
SEED = 0
ADOPT_P_LM = 0.6          # 検証LMで現在の校正に勝つ確率(ブートストラップ)
ADOPT_P_LMONLY = 0.4
ADOPT_COW_WIN = 0.7       # 牛の再投影誤差で現在の校正に勝った分割の割合
b4.VARIANTS = b5.VARIANTS = b5.VARIANTS


def boot_p_better(e_a, e_b, rng, n=N_BOOT):
    """検証点を復元抽出して、RMSE(a) < RMSE(b) となる割合。e_a, e_b は同じ点の誤差ベクトル"""
    idx = rng.integers(0, len(e_a), size=(n, len(e_a)))
    ra = np.sqrt(np.nanmean(e_a[idx] ** 2, axis=1))
    rb = np.sqrt(np.nanmean(e_b[idx] ** 2, axis=1))
    return float(np.mean(ra < rb))


def evaluate(base, cows, landmarks, rng, n_repeats=N_REPEATS):
    train_lm = b5.edge_pick_fixed(landmarks, base, N_TRAIN_PER_CAM)
    n_te_cow = max(2, int(round(len(cows) * COW_TEST_FRAC)))
    acc = {}
    te_idx = None
    for _ in range(n_repeats):
        te = set(rng.choice(len(cows), size=n_te_cow, replace=False).tolist())
        tr_c = [i for i in range(len(cows)) if i not in te]
        rs = b5.run_split(base, cows, landmarks, train_lm, tr_c)
        te_idx = rs[("現在の校正(参考)", "")]["te_idx"]
        for k, r in rs.items():
            a = acc.setdefault(k, {"e_l": [], "rp": []})
            a["e_l"].append(r["e_l"])
            a["rp"].append(float(np.median(r["rp"])))
    mean_e = {k: np.mean(np.array(a["e_l"]), axis=0) for k, a in acc.items()}      # 検証点ごとの平均誤差
    cur = ("現在の校正(参考)", "")
    rows = []
    for k in acc:
        if k == cur:
            continue
        lo = (k[0], "LMのみ")
        rows.append(dict(
            key=k, rmse=b2.rmse(mean_e[k]), p_cur=boot_p_better(mean_e[k], mean_e[cur], rng),
            p_lmonly=(None if k[1] == "LMのみ" else boot_p_better(mean_e[k], mean_e[lo], rng)),
            cow=float(np.mean(acc[k]["rp"])),
            cow_win=float(np.mean(np.array(acc[k]["rp"]) < np.array(acc[cur]["rp"])))))
    return train_lm, te_idx, acc, mean_e, rows


def main():
    g, cameras_json, cam_ids, base, landmarks, cows, frames, out_dir = b3.load_data()
    n0 = len(landmarks)
    landmarks = [lm for lm in landmarks if lm["name"] not in EXCLUDE_LANDMARKS]
    print(f"フレーム {frames}: 牛 {len(cows)}頭 / ランドマーク {n0}点 -> 除外後 {len(landmarks)}点")
    if len(cows) < 8:
        raise SystemExit("牛の数が少なすぎます")
    rng = np.random.default_rng(SEED)
    train_lm, te_idx, acc, mean_e, rows = evaluate(base, cows, landmarks, rng)
    cur = ("現在の校正(参考)", "")
    print(f"\n調整用 {len(train_lm)}点(各カメラの端から{N_TRAIN_PER_CAM}点)/ 検証用 {len(te_idx)}点(内側)")
    print("検証用: " + ", ".join(f'{landmarks[i]["cam"][-1]}:{landmarks[i]["name"]}' for i in te_idx))
    print(f"\n=== 牛の分割 {N_REPEATS}回 + 検証点ブートストラップ{N_BOOT}回 ===")
    print(f"現在の校正(学習済みのLMで有利): 検証LM RMSE {b2.rmse(mean_e[cur]):.3f} m / 牛再投影 {np.mean(acc[cur]['rp']):.2f} px")
    print(f"{'案':<20}{'重み':<12}{'LM RMSE':>8}{'P(現在に勝つ)':>14}{'P(LMのみに勝つ)':>16}{'牛再投影':>9}{'牛の勝率':>9}")
    for r in rows:
        pl = "-" if r["p_lmonly"] is None else f'{r["p_lmonly"]:.0%}'
        print(f"{r['key'][0]:<20}{r['key'][1]:<12}{r['rmse']:>8.3f}{r['p_cur']:>14.0%}{pl:>16}{r['cow']:>9.2f}{r['cow_win']:>9.0%}")

    with open(out_dir / "calibration_cv_ver6.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant", "weights", "lm_rmse", "p_beat_current", "p_beat_lm_only", "cow_reproj_px", "cow_win_vs_current"])
        for r in rows:
            w.writerow([r["key"][0], r["key"][1], f'{r["rmse"]:.4f}', f'{r["p_cur"]:.3f}',
                        "" if r["p_lmonly"] is None else f'{r["p_lmonly"]:.3f}', f'{r["cow"]:.3f}', f'{r["cow_win"]:.3f}'])

    cands = [r for r in rows if r["key"][1] != "LMのみ" and r["p_cur"] >= ADOPT_P_LM
             and r["p_lmonly"] >= ADOPT_P_LMONLY and r["cow_win"] >= ADOPT_COW_WIN]
    print("\n--- 判定 ---")
    if not cands:
        print(f"採用条件(P(現在に勝つ)>={ADOPT_P_LM:.0%}, P(LMのみに勝つ)>={ADOPT_P_LMONLY:.0%}, 牛の勝率>={ADOPT_COW_WIN:.0%})を満たす案がありません。")
        print("現在の校正を使ってください。json/txtは出力しません。")
        return
    best = min(cands, key=lambda r: r["rmse"])
    k = best["key"]
    print(f"採用候補: {k[0]} / {k[1]} (LM RMSE {best['rmse']:.3f} m, P(現在に勝つ) {best['p_cur']:.0%}, 牛の勝率 {best['cow_win']:.0%})")
    print(f"\n--- 検証用ランドマークごとの平均誤差[m] (現在の校正 vs 採用候補。中心からの距離付き) ---")
    for j, i in enumerate(te_idx):
        lm = landmarks[i]
        print(f'  {lm["cam"]} {lm["name"]:<22} 中心から{b5.edge_dist(lm, base):6.0f}px  {mean_e[cur][j]:6.3f} -> {mean_e[k][j]:6.3f}')

    variant = next(x for x in b5.VARIANTS if x[0] == k[0])
    _n, w_cow, w_lm = next(x for x in b5.WEIGHTS if x[0] == k[1])
    cams = b4.adjust(base, cows, landmarks, variant, w_cow, w_lm)
    print("\n全データ(全牛+全ランドマーク)で最終調整")
    for c in cam_ids:
        R0, t0, _i0 = base[c]
        R1, t1, _i1 = cams[c]
        rot_deg = float(np.degrees(np.linalg.norm(Rot.from_matrix(R0.T @ R1).as_rotvec())))
        print(f"  {c}: 向き {rot_deg:.3f}°  位置 {np.round(t1 - t0, 4).tolist()} m")
    ov = b1.overrides_from_cams(cameras_json, cams)
    intr_txt = {c: {nm: round(float(cams[c][2][i]), 6) for i, nm in enumerate(b1.INTR_NAMES)} for c in cam_ids}
    snippet = ["# Mootrack_core.py に貼る内容(ver6)", f"# 案: {k[0]} / {k[1]}", "CAMERA_PARAM_OVERRIDES = {"] + \
              [f'    "{a}": {b},' for a, b in ov.items()] + ["}", "", "CAM_INTRINSICS = {"] + \
              [f'    "{c}": {x},' for c, x in intr_txt.items()] + ["}"]
    (out_dir / "calibration_refined_cv_ver6.txt").write_text("\n".join(snippet), encoding="utf-8")
    (out_dir / "calibration_refined_cv_ver6.json").write_text(json.dumps({
        "frames": frames, "variant": k[0], "weights": k[1], "excluded": sorted(EXCLUDE_LANDMARKS),
        "cameras": {c: {"R": cams[c][0].tolist(), "t": cams[c][1].tolist(),
                        "intrinsics": dict(zip(b1.INTR_NAMES, map(float, cams[c][2])))} for c in cam_ids},
        "overrides": ov}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"出力: {out_dir}")
    print("反映前に、evaluate_refined_calib_on_32landmarks_ver1.py の REFINED_JSON_PATH を新しいjsonに変えて、"
          "32点全体の床誤差が現在の校正より悪化していないことを確認すること。")

if __name__ == "__main__":
    main()