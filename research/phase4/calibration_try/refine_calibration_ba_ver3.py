"""
refine_calibration_ba_ver3.py

ver2 との違い: ランドマークを「調整用 約11点」と「検証用 残り約21点」に分ける検証方式。
  - 調整用は、各カメラの画像上で散らばるように選ぶ(最遠点サンプリング。誤差の大きさでは選ばない)
  - [固定分割] 散らばり基準で選んだ1通りの分割で、設定ごとに検証用ランドマークの床誤差を表示(点ごとの表つき)
  - [ランダム分割] 調整用をランダムに選ぶ分割をN_REPEATS回繰り返し、誤差の平均・ばらつき・改善した割合を表示
  牛も毎回COW_TEST_FRACだけ検証用に取っておく(牛の整合が悪化していないかの確認用)
  - 調整用に回した点の調整後の誤差も出す(大きいままの点は、ランドマークの指定ミスの疑い)
  - 最良の設定で、全牛+全ランドマークを使って最終調整し、json/txtを出力(Mootrack_core.pyは変更しない)

配置: research/phase4/calibration/ (ver1, ver2, evaluate_refined_calib_on_32landmarks_ver1.py と同じ場所)
使い方: python refine_calibration_ba_ver3.py [フレーム ...]
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

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
N_TRAIN_PER_CAM = {"Camera_1": 3, "Camera_2": 4, "Camera_3": 4}   # 調整用に使うランドマーク数(合計11点)
N_REPEATS = 20            # ランダム分割の回数
COW_TEST_FRAC = 0.25
SEED = 0
SETTINGS = b2.SETTINGS    # (名前, w_cow, w_lm)
SUSPICIOUS_TRAIN_ERR_M = 1.0


# ---------------------------------------------------------------------------
# 分割
# ---------------------------------------------------------------------------
def spread_pick(landmarks, cam, n):
    """カメラ cam のランドマークから、画像上で散らばるようにn点選ぶ(最遠点サンプリング)"""
    idx = [i for i, lm in enumerate(landmarks) if lm["cam"] == cam]
    uv = np.array([landmarks[i]["uv"] for i in idx])
    chosen = [int(np.argmin(np.linalg.norm(uv - uv.mean(0), axis=1)))]      # 重心に最も近い点から開始
    while len(chosen) < min(n, len(idx)):
        dmin = np.min(np.linalg.norm(uv[:, None] - uv[chosen][None], axis=2), axis=1)
        dmin[chosen] = -1
        chosen.append(int(np.argmax(dmin)))
    return [idx[c] for c in chosen]


def random_pick(landmarks, rng):
    out = []
    for cam, n in N_TRAIN_PER_CAM.items():
        idx = [i for i, lm in enumerate(landmarks) if lm["cam"] == cam]
        out += list(rng.choice(idx, size=min(n, len(idx)), replace=False))
    return sorted(int(i) for i in out)


def fixed_pick(landmarks):
    out = []
    for cam, n in N_TRAIN_PER_CAM.items():
        out += spread_pick(landmarks, cam, n)
    return sorted(out)


def run_split(base, cows, landmarks, train_lm, train_cow, settings=SETTINGS):
    """1通りの分割で、調整なし+各設定の結果を返す。{設定名: dict}"""
    tr_l = [landmarks[i] for i in train_lm]
    te_idx = [i for i in range(len(landmarks)) if i not in set(train_lm)]
    te_l = [landmarks[i] for i in te_idx]
    tr_c = [cows[i] for i in train_cow]
    te_c = [cows[i] for i in range(len(cows)) if i not in set(train_cow)]
    res = {}
    ev0 = b1.evaluate_cows(base, te_c)
    res["調整なし(現在の校正)"] = dict(te_idx=te_idx, e_l=b2.landmark_errors(base, te_l), tr_l=b2.landmark_errors(base, tr_l),
                                     rp=ev0["reproj_px"], rs=ev0["resid_m"], cams=base)
    for name, w_cow, w_lm in settings:
        cams, _ = b2.bundle_adjust2(base, tr_c, tr_l, w_cow, w_lm)
        ev = b1.evaluate_cows(cams, te_c)
        res[name] = dict(te_idx=te_idx, e_l=b2.landmark_errors(cams, te_l), tr_l=b2.landmark_errors(cams, tr_l),
                         rp=ev["reproj_px"], rs=ev["resid_m"], cams=cams)
    return res


def score_of(r, r0):
    return b2.rmse(r["e_l"]) / b2.rmse(r0["e_l"]) + np.median(r["rp"]) / np.median(r0["rp"])


# ---------------------------------------------------------------------------
def load_data():
    import phase3_bev_multicam_gt_zonepose_ver1 as g
    from Mootrack_core import (apply_camera_overrides, get_intrinsics, load_floorplan, resolve_poses)
    from evaluate_refined_calib_on_32landmarks_ver1 import LANDMARKS, find_region_vertex

    in_dir = g.OUTPUT_DIR / "bev_gt_triangulated"
    out_dir = g.OUTPUT_DIR / "bev_calibration_refined"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = [int(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else \
        sorted(int(p.name[:6]) for p in in_dir.glob("*_tri_gt_ver1.json"))
    if not frames:
        raise SystemExit(f"三角測量GTのjsonが見つかりません: {in_dir}")
    regions, cameras_json, _w, _h = load_floorplan(g.FLOORPLAN_JSON_PATH)
    poses = resolve_poses(apply_camera_overrides(cameras_json), root="Area1", fmt=g.ROT_FMT)
    cam_ids = list(g.CAM_IDS)
    base = {c: (poses[c].R.copy(), poses[c].t.copy(), np.array(get_intrinsics(c), dtype=np.float64)) for c in cam_ids}
    landmarks = b2.build_landmarks(regions, LANDMARKS, find_region_vertex)
    for lm, d in zip(landmarks, LANDMARKS):
        lm["name"] = f'{d["region"]}#{d["vertex_idx"]}'
    cows = []
    for fr in frames:
        data = json.loads((in_dir / f"{fr:06d}_tri_gt_ver1.json").read_text(encoding="utf-8"))
        for c in data["cows"]:
            clicks = {k: v for k, v in c.get("clicks", {}).items() if v is not None and k in cam_ids}
            if len(clicks) < 2:
                continue
            cow = {"frame": fr, "id": c["id"], "clicks": clicks}
            X, _r = b1.triangulate_cow(base, clicks)
            if X is None:
                continue
            e = b1.evaluate_cows(base, [cow])
            if (e["angle_deg"][0] >= b1.MIN_ANGLE_USE_DEG and e["reproj_px"].max() <= b1.MAX_INIT_REPROJ_PX
                    and b1.HEIGHT_USE_RANGE_M[0] <= e["height_m"][0] <= b1.HEIGHT_USE_RANGE_M[1]):
                cows.append(cow)
    return g, cameras_json, cam_ids, base, landmarks, cows, frames, out_dir


def main():
    g, cameras_json, cam_ids, base, landmarks, cows, frames, out_dir = load_data()
    print(f"フレーム {frames}: 牛(品質フィルタ後){len(cows)}頭 / ランドマーク {len(landmarks)}点")
    if len(cows) < 8:
        raise SystemExit(f"牛の数が少なすぎます({len(cows)}頭)")
    rng = np.random.default_rng(SEED)
    n_te_cow = max(2, int(round(len(cows) * COW_TEST_FRAC)))

    def cow_train_idx():
        te = set(rng.choice(len(cows), size=n_te_cow, replace=False).tolist())
        return [i for i in range(len(cows)) if i not in te]

    # ---- [固定分割] ----
    train_lm = fixed_pick(landmarks)
    res = run_split(base, cows, landmarks, train_lm, cow_train_idx())
    print(f"\n=== [固定分割] 調整用 {len(train_lm)}点(画像上で散らばるように選択) / 検証用 {len(landmarks) - len(train_lm)}点 ===")
    print("調整用: " + ", ".join(f'{landmarks[i]["cam"][-1]}:{landmarks[i]["name"]}' for i in train_lm))
    r0 = res["調整なし(現在の校正)"]
    print(f"\n{'設定':<24}{'検証LM RMSE[m]':>15}{'平均[m]':>9}{'最大[m]':>9}{'牛再投影中央値[px]':>18}{'スコア':>8}")
    scores = {}
    for name, r in res.items():
        sc = score_of(r, r0)
        scores[name] = sc
        print(f"{name:<24}{b2.rmse(r['e_l']):>15.3f}{np.nanmean(r['e_l']):>9.3f}{np.nanmax(r['e_l']):>9.3f}"
              f"{np.median(r['rp']):>18.2f}{sc:>8.2f}")

    best_fixed = min((n for n in res if n != "調整なし(現在の校正)"), key=lambda n: scores[n])
    rb = res[best_fixed]
    print(f"\n--- 検証用ランドマークごとの誤差[m]: 調整なし -> {best_fixed} ---")
    for k, i in enumerate(r0["te_idx"]):
        lm = landmarks[i]
        print(f'  {lm["cam"]} {lm["name"]:<22} {r0["e_l"][k]:6.3f} -> {rb["e_l"][k]:6.3f}  ({rb["e_l"][k] - r0["e_l"][k]:+.3f})')
    print(f"\n--- 調整用に回した点の、調整後の誤差[m] ({best_fixed}) ---")
    for k, i in enumerate(train_lm):
        lm = landmarks[i]
        flag = "  <- 大きいまま。ランドマークの指定ミスの疑い" if rb["tr_l"][k] > SUSPICIOUS_TRAIN_ERR_M else ""
        print(f'  {lm["cam"]} {lm["name"]:<22} {rb["tr_l"][k]:6.3f}{flag}')

    # ---- [ランダム分割] ----
    names = [n for n, _, _ in SETTINGS]
    agg = {n: {"rmse": [], "rp": [], "score": []} for n in names}
    base_rmse = []
    for _ in range(N_REPEATS):
        tl = random_pick(landmarks, rng)
        rs = run_split(base, cows, landmarks, tl, cow_train_idx())
        r00 = rs["調整なし(現在の校正)"]
        base_rmse.append(b2.rmse(r00["e_l"]))
        for n in names:
            agg[n]["rmse"].append(b2.rmse(rs[n]["e_l"]))
            agg[n]["rp"].append(float(np.median(rs[n]["rp"])))
            agg[n]["score"].append(score_of(rs[n], r00))
    print(f"\n=== [ランダム分割 {N_REPEATS}回] 検証用ランドマークRMSE[m] (調整なし 平均 {np.mean(base_rmse):.3f}) ===")
    print(f"{'設定':<24}{'平均':>8}{'標準偏差':>9}{'調整なしより良かった割合':>22}{'スコア平均':>11}")
    for n in names:
        better = np.mean(np.array(agg[n]["rmse"]) < np.array(base_rmse))
        print(f"{n:<24}{np.mean(agg[n]['rmse']):>8.3f}{np.std(agg[n]['rmse']):>9.3f}{better:>22.0%}{np.mean(agg[n]['score']):>11.2f}")
    best = min(names, key=lambda n: np.mean(agg[n]["score"]))
    print(f"\n最良の設定(ランダム分割のスコア平均で選択): {best}  (調整なし=2.00)")

    with open(out_dir / "calibration_cv_ver3.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["setting", "fixed_lm_rmse", "fixed_score", "random_lm_rmse_mean", "random_lm_rmse_std", "random_score_mean"])
        for n in names:
            w.writerow([n, f"{b2.rmse(res[n]['e_l']):.4f}", f"{scores[n]:.3f}", f"{np.mean(agg[n]['rmse']):.4f}",
                        f"{np.std(agg[n]['rmse']):.4f}", f"{np.mean(agg[n]['score']):.3f}"])

    # ---- 最終調整(全牛 + 全ランドマーク) ----
    _n, w_cow, w_lm = next(s for s in SETTINGS if s[0] == best)
    cams, info = b2.bundle_adjust2(base, cows, landmarks, w_cow, w_lm)
    print(f"\n最良の設定で全データ({len(cows)}頭 + {len(landmarks)}点)を使って最終調整")
    print("--- パラメータの変化 ---")
    for c in cam_ids:
        R0, t0, i0 = base[c]
        R1, t1, i1 = cams[c]
        rot_deg = float(np.degrees(np.linalg.norm(Rot.from_matrix(R0.T @ R1).as_rotvec())))
        print(f"  {c}: 向き {rot_deg:.3f}°  位置 {np.round(t1 - t0, 4).tolist()} m")
    ov = b1.overrides_from_cams(cameras_json, cams)
    intr_txt = {c: {nm: round(float(cams[c][2][k]), 6) for k, nm in enumerate(b1.INTR_NAMES)} for c in cam_ids}
    snippet = ["# Mootrack_core.py に貼る内容(ver3: 牛+ランドマーク)", f"# 設定: {best} (w_cow={w_cow}, w_lm={w_lm})",
               "CAMERA_PARAM_OVERRIDES = {"] + [f'    "{k}": {v},' for k, v in ov.items()] + ["}", "", "CAM_INTRINSICS = {"] + \
              [f'    "{c}": {v},' for c, v in intr_txt.items()] + ["}"]
    (out_dir / "calibration_refined_cv_ver3.txt").write_text("\n".join(snippet), encoding="utf-8")
    (out_dir / "calibration_refined_cv_ver3.json").write_text(json.dumps({
        "frames": frames, "n_cows": len(cows), "n_landmarks": len(landmarks), "setting": best, "w_cow": w_cow, "w_lm": w_lm,
        "cameras": {c: {"R": cams[c][0].tolist(), "t": cams[c][1].tolist(),
                        "intrinsics": dict(zip(b1.INTR_NAMES, map(float, cams[c][2])))} for c in cam_ids},
        "overrides": ov, "random_score_mean": float(np.mean(agg[best]["score"]))}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\n出力: {out_dir}")
    if np.mean(agg[best]["score"]) >= 2.0:
        print("※調整なしより良くなっていません。調整後の値は採用しないでください。")
    print("最終調整は全ランドマークを使っているため、その誤差は検証には使えません(検証は上の分割の結果で見ること)。")


if __name__ == "__main__":
    main()