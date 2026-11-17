"""
refine_calibration_ba_ver7.py

全ランドマーク(43点)+牛 で、K分割クロスバリデーションをやり直す。
Mode3 は評価用のランドマークで校正しているので、Mode3 の値(D)は評価に使えない。
代わりに「Mode3 と同じ種類の校正を、調整用の点だけで作り直して、外した点で測る」ことで公平に比べる。

  校正の手順(どれも、調整用の点だけで作る。B(現在値)を初期値にする):
    LMのみ     : ランドマークだけ(Mode3 に近い手順)
    牛+LM xN   : ランドマーク + 牛のクリック(LMの重みはN倍)
  動かすもの(VARIANTS):
    外部のみ         : 位置・向き
    外部+xi,alpha    : 位置・向き + 内部の xi, alpha (Mode3 と同じ)
  評価: 外したランドマークの床誤差[m](43点が1回ずつ検証用になる。分割をREPEATS回変えて平均)
        外した牛の再投影誤差[pseudo px]
  比べる相手: B(現在値。ランドマークで校正していない) と、同じ動かし方の「LMのみ」。
      P(Bに勝つ)       : 校正の手順そのものが、現在値より良いか
      P(LMのみに勝つ)  : 牛を足すことに意味があるか
  参考として D(Mode3の実際の値。43点で学習済み)の誤差も並べる(検証用ではない)。

注意: ここでは LM の残差を「既知の床点への光線の角度差」にしている(ver2〜6と同じ)。Mode3 は床での位置誤差[m]を
      最小化しているので、重みの付き方は少し違う。

配置: research/phase4/calibration/ (ver1〜ver3 の各スクリプトと同じ場所)
使い方: python refine_calibration_ba_ver7.py [フレーム ...]
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

K_FOLDS = 5
REPEATS = 3
SEED = 0
N_BOOT = 3000
# (名前, 動かす内部パラメータ, prior回転[deg], prior位置[m], prior xi, prior alpha)
VARIANTS = [
    ("外部のみ", (), 3.0, 0.6, 0.15, 0.10),
    ("外部+xi,alpha", ("xi", "alpha"), 3.0, 0.6, 0.15, 0.10),
]
WEIGHTS = [("LMのみ", 0.0, 1.0), ("牛+LM x1", 1.0, 1.0), ("牛+LM x2", 1.0, 2.0)]

# D: calibrate_camera_from_landmarks.py の Mode3 (full) の出力(参考表示用)
D_PARAM_OVERRIDES = {
    "Area1->Camera_1": [7.5458, 6.2912, -4.9393, -0.0302, -1.113, -0.9927],
    "Camera_1->Camera_2": [13.5447, 0.4388, -0.397, -7.1617, 1.2404, -0.1662],
    "Camera_2->Camera_3": [13.5215, -0.1578, -0.3039, 11.1117, 0.2251, -0.2347],
}
D_XI_ALPHA = {"Camera_1": (-0.065738, 0.519379), "Camera_2": (-0.093895, 0.518379), "Camera_3": (-0.060158, 0.500745)}


def adjust(base, cows, lms, variant, w_cow, w_lm):
    _n, free, rot, tr, pxi, pal = variant
    old = dict(b1.PRIOR)
    b1.PRIOR.update(rot_deg=rot, trans_m=tr, xi=pxi, alpha=pal)
    try:
        return b2.bundle_adjust2(base, cows, lms, w_cow, w_lm, free_intr=free)[0]
    finally:
        b1.PRIOR.update(old)


def folds_for(n_lm, lm_cams, n_cow, rng, k=K_FOLDS):
    lm_fold = b2.make_folds(n_lm, k, rng, strata=lm_cams)
    cow_fold = b2.make_folds(n_cow, k, rng)
    return lm_fold, cow_fold


def boot_p(e_a, e_b, rng, n=N_BOOT):
    """ランドマークを復元抽出して RMSE(a) < RMSE(b) となる割合"""
    idx = rng.integers(0, len(e_a), size=(n, len(e_a)))
    return float(np.mean(np.sqrt(np.mean(e_a[idx] ** 2, axis=1)) < np.sqrt(np.mean(e_b[idx] ** 2, axis=1))))


def rmse(a):
    return float(np.sqrt(np.mean(np.asarray(a) ** 2)))


def cross_validate(base, cows, landmarks, rng, repeats=REPEATS, k=K_FOLDS):
    keys = [(v[0], w[0]) for v in VARIANTS for w in WEIGHTS]
    e_lm = {key: np.zeros(len(landmarks)) for key in keys}       # 繰り返しで平均した、ランドマークごとの検証誤差
    rp = {key: [] for key in keys}
    lm_cams = [lm["cam"] for lm in landmarks]
    for _ in range(repeats):
        lm_fold, cow_fold = folds_for(len(landmarks), lm_cams, len(cows), rng, k)
        for f in range(k):
            tr_l = [lm for lm, fo in zip(landmarks, lm_fold) if fo != f]
            te_idx = [i for i, fo in enumerate(lm_fold) if fo == f]
            tr_c = [c for c, fo in zip(cows, cow_fold) if fo != f]
            te_c = [c for c, fo in zip(cows, cow_fold) if fo == f]
            for v in VARIANTS:
                for wn, w_cow, w_lm in WEIGHTS:
                    cams = adjust(base, tr_c, tr_l, v, w_cow, w_lm)
                    e = b2.landmark_errors(cams, [landmarks[i] for i in te_idx])
                    e_lm[(v[0], wn)][te_idx] += e / repeats
                    rp[(v[0], wn)].append(b1.evaluate_cows(cams, te_c)["reproj_px"])
    return e_lm, {key: float(np.median(np.concatenate(v))) for key, v in rp.items()}


def main():
    import phase3_bev_multicam_gt_zonepose_ver1 as g
    from Mootrack_core import (DS_CX, DS_CY, DS_FX, DS_FY, ROT_FMT, apply_camera_overrides, get_intrinsics,
                               load_floorplan, resolve_poses)
    import refine_calibration_ba_ver3 as b3
    from evaluate_refined_calib_on_all_landmarks_ver3 import LANDMARKS, find_region_vertex

    out_dir = g.OUTPUT_DIR / "bev_calibration_refined"
    out_dir.mkdir(parents=True, exist_ok=True)
    regions, cameras_json, _w, _h = load_floorplan(g.FLOORPLAN_JSON_PATH)
    poses = resolve_poses(apply_camera_overrides(cameras_json), root="Area1", fmt=ROT_FMT)
    cam_ids = list(g.CAM_IDS)
    base = {c: (poses[c].R.copy(), poses[c].t.copy(), np.array(get_intrinsics(c), dtype=np.float64)) for c in cam_ids}
    landmarks = b2.build_landmarks(regions, LANDMARKS, find_region_vertex)
    for lm, d in zip(landmarks, LANDMARKS):
        lm["name"] = f'{d["region"]}#{d["vertex_idx"]}'

    # 牛(B で品質フィルタ。b3.load_data は 'sys.argv' のフレームを読む)
    _g, _cj, _ci, _base, _lms, cows, frames, _od = b3.load_data()
    print(f"フレーム {frames}: 牛 {len(cows)}頭 / ランドマーク {len(landmarks)}点 "
          f"({ {c: sum(l['cam'] == c for l in landmarks) for c in cam_ids} })")
    if len(cows) < 2 * K_FOLDS:
        raise SystemExit("牛の数が少なすぎます")
    rng = np.random.default_rng(SEED)

    # B と D(参考)
    e_B = b2.landmark_errors(base, landmarks)
    poses_D = resolve_poses(apply_camera_overrides(cameras_json, D_PARAM_OVERRIDES), root="Area1", fmt=ROT_FMT)
    cams_D = {c: (poses_D[c].R.copy(), poses_D[c].t.copy(),
                  np.array([DS_FX, DS_FY, DS_CX, DS_CY, *D_XI_ALPHA[c]], dtype=np.float64)) for c in cam_ids}
    e_D = b2.landmark_errors(cams_D, landmarks)
    rp_B = float(np.median(b1.evaluate_cows(base, cows)["reproj_px"]))
    rp_D = float(np.median(b1.evaluate_cows(cams_D, cows)["reproj_px"]))

    e_lm, rp = cross_validate(base, cows, landmarks, rng)

    print(f"\n=== {K_FOLDS}分割クロスバリデーション ×{REPEATS}回: 検証に外したランドマーク43点の床誤差[m] ===")
    print(f"B 現在値 (ランドマーク未使用)      : RMSE {rmse(e_B):.3f} / 牛再投影(全牛) {rp_B:.2f}px")
    print(f"D Mode3 (43点で学習済み。検証ではない参考): RMSE {rmse(e_D):.3f} / 牛再投影(全牛, 未使用データ) {rp_D:.2f}px")
    print(f"\n{'動かし方':<16}{'重み':<12}{'検証LM RMSE':>12}{'中央値':>8}{'最大':>7}{'P(Bに勝つ)':>11}{'P(LMのみに勝つ)':>16}{'牛再投影[px]':>13}")
    rows = []
    for v in VARIANTS:
        for wn, _wc, _wl in WEIGHTS:
            key = (v[0], wn)
            e = e_lm[key]
            pb = boot_p(e, e_B, rng)
            pl = None if wn == "LMのみ" else boot_p(e, e_lm[(v[0], "LMのみ")], rng)
            rows.append((key, rmse(e), pb, pl, rp[key]))
            print(f"{v[0]:<16}{wn:<12}{rmse(e):>12.3f}{np.median(e):>8.3f}{np.max(e):>7.3f}{pb:>11.0%}"
                  f"{'-' if pl is None else format(pl, '.0%'):>16}{rp[key]:>13.2f}")

    print("\n--- カメラ別の検証RMSE[m] ---")
    print(f"{'':<28}" + "".join(f"{c:>11}" for c in cam_ids))
    for label, e in [("B 現在値", e_B)] + [(f"{k[0]}/{k[1]}", e_lm[k]) for k, *_ in rows]:
        print(f"{label:<28}" + "".join(
            f"{rmse(e[[i for i, lm in enumerate(landmarks) if lm['cam'] == c]]):>11.3f}" for c in cam_ids))

    with open(out_dir / "calibration_cv_ver7.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant", "weights", "heldout_lm_rmse", "p_beat_B", "p_beat_lm_only", "cow_reproj_px"])
        w.writerow(["B現在値", "", f"{rmse(e_B):.4f}", "", "", f"{rp_B:.3f}"])
        for key, r, pb, pl, c in rows:
            w.writerow([key[0], key[1], f"{r:.4f}", f"{pb:.3f}", "" if pl is None else f"{pl:.3f}", f"{c:.3f}"])

    best = min(rows, key=lambda r: r[1])
    key, r, pb, pl, c = best
    print(f"\n--- 判定 ---\n検証LM RMSEが最小の手順: {key[0]} / {key[1]}  RMSE {r:.3f} m (B: {rmse(e_B):.3f} m), P(Bに勝つ) {pb:.0%}")
    lm_only_best = min((x for x in rows if x[0][1] == "LMのみ"), key=lambda x: x[1])
    cow_best = min((x for x in rows if x[0][1] != "LMのみ"), key=lambda x: x[1])
    print(f"LMのみの最良: {lm_only_best[0][0]}  RMSE {lm_only_best[1]:.3f} m / 牛を足した最良: {cow_best[0][0]}/{cow_best[0][1]} "
          f"RMSE {cow_best[1]:.3f} m (P(LMのみに勝つ) {cow_best[3]:.0%})")

    variant = next(v for v in VARIANTS if v[0] == key[0])
    _n, w_cow, w_lm = next(x for x in WEIGHTS if x[0] == key[1])
    cams = adjust(base, cows, landmarks, variant, w_cow, w_lm)
    e_in = b2.landmark_errors(cams, landmarks)
    print(f"\n全データ(43点+全牛)で最終調整 {key[0]}/{key[1]}: 43点の誤差(学習済み) RMSE {rmse(e_in):.3f} m")
    ov = b1.overrides_from_cams(cameras_json, cams)
    intr_txt = {c: {nm: round(float(cams[c][2][i]), 6) for i, nm in enumerate(b1.INTR_NAMES)} for c in cam_ids}
    snippet = ["# Mootrack_core.py に貼る内容(ver7)", f"# 手順: {key[0]} / {key[1]}", "CAMERA_PARAM_OVERRIDES = {"] + \
              [f'    "{a}": {b},' for a, b in ov.items()] + ["}", "", "CAM_INTRINSICS = {"] + \
              [f'    "{cc}": {x},' for cc, x in intr_txt.items()] + ["}"]
    (out_dir / "calibration_refined_cv_ver7.txt").write_text("\n".join(snippet), encoding="utf-8")
    (out_dir / "calibration_refined_cv_ver7.json").write_text(json.dumps({
        "frames": frames, "variant": key[0], "weights": key[1],
        "cameras": {c: {"R": cams[c][0].tolist(), "t": cams[c][1].tolist(),
                        "intrinsics": dict(zip(b1.INTR_NAMES, map(float, cams[c][2])))} for c in cam_ids},
        "overrides": ov}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"出力: {out_dir}")


if __name__ == "__main__":
    main()