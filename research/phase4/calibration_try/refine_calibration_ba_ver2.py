"""
refine_calibration_ba_ver2.py

refine_calibration_ba_ver1.py(牛の背中のクリックだけを使うバンドル調整)に、
32点の床のランドマーク(床面座標が分かっている点)を拘束として加え、
クロスバリデーションで、重みと効果を確かめる。

  牛のクリック(同じ背中の点を複数カメラで指定)  : カメラ間の整合(密に、広い範囲で効く)
  床のランドマーク(画素 と 床面座標(x, y, Z=0))  : 絶対的な位置・向き・大きさ(全体の基準)
  → 牛だけだと決まらない全体の位置・大きさを、ランドマークが固定する。ランドマークだけだと
    点が32個しか無く、画像の端などで不安定になるが、牛の点が補う。

クロスバリデーション(K分割):
  牛とランドマークを、それぞれK個のグループに分ける(ランドマークはカメラごとに均等に)。
  グループkを検証用に取っておき、残りで調整 -> 検証用の牛とランドマークの誤差を測る。
  これをk=1..Kで繰り返し、すべての検証結果をまとめる。使っていない点の誤差なので、過剰に合わせて
  いないかが分かる。ランドマークの重み w_lm(牛に対する倍率)をいくつか試して比べる。

  検証の指標:
    ランドマーク : 床面の位置誤差[m]  (絶対的な精度。evaluate_*_on_32landmarks と同じ指標)
    牛           : 再投影誤差[擬似px]と光線の残差[m]  (カメラ間の整合)
  調整なし(現在の校正)の値も同じ検証用の点で出して、比べる。

出力(OUTPUT_DIR / bev_calibration_refined /):
  calibration_cv_ver2.csv          : 設定ごとの検証結果
  calibration_refined_cv_ver2.json : 選んだ設定で、全データを使って調整した結果
  calibration_refined_cv_ver2.txt  : Mootrack_core.py に貼れる形式
※ 元のファイルは変更しない。反映は自分で決める(反映前後で評価を取り直すこと)。

refine_calibration_ba_ver1.py / evaluate_refined_calib_on_32landmarks_ver1.py と同じフォルダに置く。

使い方:
    python refine_calibration_ba_ver2.py            # 三角測量GTのjsonがある全フレーム
    python refine_calibration_ba_ver2.py 100 200
"""

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
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
from Mootrack_core import get_intrinsics, unproject_to_floor

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
K_FOLDS = 4
SEED = 0
# 試す設定: (名前, 牛の重み w_cow, ランドマークの重み w_lm)。w_cow=0 は「ランドマークだけ」
SETTINGS = [
    ("牛のみ(ver1と同じ)", 1.0, 0.0),
    ("牛+ランドマーク x0.5", 1.0, 0.5),
    ("牛+ランドマーク x1", 1.0, 1.0),
    ("牛+ランドマーク x2", 1.0, 2.0),
    ("牛+ランドマーク x4", 1.0, 4.0),
    ("ランドマークのみ", 0.0, 1.0),
]
SIGMA_LANDMARK_PX = 5.0     # ランドマークの画素指定のずれの見積もり(擬似ピクセル)。牛のクリックより少し大きめ
LANDMARK_Z = 0.0            # ランドマークは床の点


# ---------------------------------------------------------------------------
# ランドマーク
# ---------------------------------------------------------------------------
def build_landmarks(regions, landmarks_def, find_vertex):
    out = []
    for lm in landmarks_def:
        x, y = find_vertex(regions, lm["region"], lm["vertex_idx"])
        out.append({"cam": lm["camera"], "uv": np.array(lm["pixel"], dtype=np.float64),
                    "X": np.array([x, y, LANDMARK_Z], dtype=np.float64), "label": lm.get("label", "")})
    return out


def landmark_errors(cams, landmarks):
    """床面(Z=0)に投影したときの位置誤差[m]"""
    errs = []
    for lm in landmarks:
        R, t, intr = cams[lm["cam"]]
        xy, valid = unproject_to_floor(lm["uv"].reshape(1, 2), R, t, *intr, floor_z=LANDMARK_Z)
        errs.append(float(np.linalg.norm(xy[0] - lm["X"][:2])) if valid[0] else np.nan)
    return np.array(errs)


# ---------------------------------------------------------------------------
# 牛 + ランドマークのバンドル調整
# ---------------------------------------------------------------------------
def bundle_adjust2(base, cows, landmarks, w_cow, w_lm, free_intr=b1.FREE_INTRINSICS,
                   fix_cam=b1.FIX_EXTRINSICS_CAMERA, verbose=0):
    cam_ids = list(base.keys())
    layout = b1.Layout(cam_ids, free_intr, fix_cam)
    nc = layout.n_cam_params
    F = {c: 0.5 * (base[c][2][0] + base[c][2][1]) for c in cam_ids}
    sig = b1.prior_sigmas(layout, base)

    # 牛(w_cow>0 のときだけ使う)
    X0, used = [], []
    if w_cow > 0:
        for k, cow in enumerate(cows):
            X, _r = b1.triangulate_cow(base, cow["clicks"])
            if X is not None:
                X0.append(X)
                used.append(k)
    X0 = np.array(X0).reshape(-1, 3)
    n_pts = len(used)
    obs = {c: {"pt": [], "uv": []} for c in cam_ids}
    for pi, k in enumerate(used):
        for cid, uv in cows[k]["clicks"].items():
            if cid in obs and uv is not None:
                obs[cid]["pt"].append(pi)
                obs[cid]["uv"].append(uv)
    for c in cam_ids:
        obs[c]["pt"] = np.array(obs[c]["pt"], dtype=int)
        obs[c]["uv"] = np.array(obs[c]["uv"], dtype=np.float64).reshape(-1, 2)
    # ランドマーク(w_lm>0 のときだけ使う)
    lms = {c: [lm for lm in landmarks if lm["cam"] == c] for c in cam_ids} if w_lm > 0 else {c: [] for c in cam_ids}

    def fun(x):
        xc, P = x[:nc], x[nc:].reshape(-1, 3)
        out = []
        for c in cam_ids:
            if len(obs[c]["pt"]) == 0 and not lms[c]:
                continue
            R, t, intr = b1.camera_from_x(layout, xc, base, c)
            if len(obs[c]["pt"]):
                d_obs, ok = b1.rays_world(R, t, intr, obs[c]["uv"])
                v = P[obs[c]["pt"]] - t
                d_pred = v / np.linalg.norm(v, axis=1, keepdims=True)
                r = w_cow * F[c] * (d_obs - d_pred) / b1.SIGMA_CLICK_PX
                r[~ok] = 0.0
                out.append(r.ravel())
            if lms[c]:
                uv = np.array([lm["uv"] for lm in lms[c]])
                XL = np.array([lm["X"] for lm in lms[c]])
                d_obs, ok = b1.rays_world(R, t, intr, uv)
                v = XL - t
                d_pred = v / np.linalg.norm(v, axis=1, keepdims=True)
                r = w_lm * F[c] * (d_obs - d_pred) / SIGMA_LANDMARK_PX
                r[~ok] = 0.0
                out.append(r.ravel())
        out.append(xc / sig)
        return np.concatenate(out)

    # ヤコビアンの疎構造(fun と同じ順序)
    n_obs_rows = sum(3 * len(obs[c]["pt"]) + 3 * len(lms[c]) for c in cam_ids)
    S = lil_matrix((n_obs_rows + nc, nc + 3 * n_pts), dtype=int)
    row = 0
    for c in cam_ids:
        cols = layout.cam_columns(c)
        for pi in obs[c]["pt"]:
            pcols = [nc + 3 * pi, nc + 3 * pi + 1, nc + 3 * pi + 2]
            for r in range(3):
                for col in cols + pcols:
                    S[row + r, col] = 1
            row += 3
        for _lm in lms[c]:
            for r in range(3):
                for col in cols:
                    S[row + r, col] = 1
            row += 3
    for j in range(nc):
        S[n_obs_rows + j, j] = 1

    x0 = np.concatenate([np.zeros(nc), X0.ravel()])
    res = least_squares(fun, x0, jac_sparsity=S, loss="soft_l1", f_scale=b1.LOSS_F_SCALE,
                        x_scale="jac", max_nfev=b1.MAX_NFEV, verbose=verbose)
    cams = {c: b1.camera_from_x(layout, res.x[:nc], base, c) for c in cam_ids}
    return cams, {"n_points": n_pts, "cost": float(res.cost), "nfev": int(res.nfev)}


# ---------------------------------------------------------------------------
# クロスバリデーション
# ---------------------------------------------------------------------------
def make_folds(n, k, rng, strata=None):
    fold = np.zeros(n, dtype=int)
    if strata is None:
        fold[rng.permutation(n)] = np.arange(n) % k
    else:
        for s in sorted(set(strata)):
            idx = np.array([i for i in range(n) if strata[i] == s])
            fold[rng.permutation(idx)] = np.arange(len(idx)) % k
    return fold


def cross_validate(base, cows, landmarks, settings=SETTINGS, k=K_FOLDS, seed=SEED):
    rng = np.random.default_rng(seed)
    cow_fold = make_folds(len(cows), k, rng)
    lm_fold = make_folds(len(landmarks), k, rng, strata=[lm["cam"] for lm in landmarks])
    rows = []
    # 調整なし(現在の校正)
    ev_c0 = b1.evaluate_cows(base, cows)
    e_l0 = landmark_errors(base, landmarks)
    rows.append(("調整なし(現在の校正)", None, None, e_l0, ev_c0["reproj_px"], ev_c0["resid_m"]))
    for name, w_cow, w_lm in settings:
        e_l, e_rp, e_rs = [], [], []
        for f in range(k):
            tr_c = [c for c, fo in zip(cows, cow_fold) if fo != f]
            te_c = [c for c, fo in zip(cows, cow_fold) if fo == f]
            tr_l = [lm for lm, fo in zip(landmarks, lm_fold) if fo != f]
            te_l = [lm for lm, fo in zip(landmarks, lm_fold) if fo == f]
            cams, _info = bundle_adjust2(base, tr_c, tr_l, w_cow, w_lm)
            e_l.append(landmark_errors(cams, te_l))
            ev = b1.evaluate_cows(cams, te_c)
            e_rp.append(ev["reproj_px"])
            e_rs.append(ev["resid_m"])
        rows.append((name, w_cow, w_lm, np.concatenate(e_l), np.concatenate(e_rp), np.concatenate(e_rs)))
    return rows


def rmse(a):
    return float(np.sqrt(np.nanmean(a ** 2))) if len(a) else float("nan")


def main():
    import phase3_bev_multicam_gt_zonepose_ver1 as g
    from Mootrack_core import apply_camera_overrides, load_floorplan, load_floorplan_cameras, resolve_poses
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
    landmarks = build_landmarks(regions, LANDMARKS, find_region_vertex)

    cows_all = []
    for fr in frames:
        data = json.loads((in_dir / f"{fr:06d}_tri_gt_ver1.json").read_text(encoding="utf-8"))
        for c in data["cows"]:
            clicks = {k: v for k, v in c.get("clicks", {}).items() if v is not None and k in cam_ids}
            if len(clicks) >= 2:
                cows_all.append({"frame": fr, "id": c["id"], "clicks": clicks})
    cows = []
    for cow in cows_all:
        X, _r = b1.triangulate_cow(base, cow["clicks"])
        if X is None:
            continue
        e = b1.evaluate_cows(base, [cow])
        if (e["angle_deg"][0] >= b1.MIN_ANGLE_USE_DEG and e["reproj_px"].max() <= b1.MAX_INIT_REPROJ_PX
                and b1.HEIGHT_USE_RANGE_M[0] <= e["height_m"][0] <= b1.HEIGHT_USE_RANGE_M[1]):
            cows.append(cow)
    print(f"フレーム {frames}: 牛 {len(cows_all)}頭 -> 品質フィルタ後 {len(cows)}頭 / ランドマーク {len(landmarks)}点")
    if len(cows) < 2 * K_FOLDS:
        raise SystemExit(f"牛の数が少なすぎます({len(cows)}頭)。もっとクリックしてから実行してください。")

    rows = cross_validate(base, cows, landmarks)
    base_lm, base_rp, base_rs = rmse(rows[0][3]), float(np.median(rows[0][4])), float(np.median(rows[0][5]))
    print(f"\n=== {K_FOLDS}分割クロスバリデーション(すべて、調整に使っていない点での誤差) ===")
    print(f"{'設定':<24} {'ランドマーク床誤差RMSE[m]':>24} {'同 平均[m]':>11} {'牛の再投影 中央値[px]':>20} {'牛の残差 中央値[m]':>18} {'スコア':>7}")
    scores = []
    for name, w_cow, w_lm, e_l, e_rp, e_rs in rows:
        r_l, m_l = rmse(e_l), float(np.nanmean(e_l))
        m_rp, m_rs = float(np.median(e_rp)), float(np.median(e_rs))
        score = r_l / base_lm + m_rp / base_rp          # 調整なしを1+1=2として、小さいほど良い
        scores.append(score)
        print(f"{name:<24} {r_l:>24.3f} {m_l:>11.3f} {m_rp:>20.2f} {m_rs:>18.4f} {score:>7.2f}")
    best = int(np.argmin(scores[1:])) + 1
    print(f"\nスコア = (ランドマークRMSE / 調整なし) + (牛の再投影 / 調整なし)。調整なし=2.00。小さいほど良い。")
    print(f"最良の設定: {rows[best][0]} (スコア {scores[best]:.2f})")

    with open(out_dir / "calibration_cv_ver2.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["setting", "w_cow", "w_lm", "landmark_rmse_m", "landmark_mean_m", "cow_reproj_median_px",
                    "cow_resid_median_m", "score"])
        for (name, w_cow, w_lm, e_l, e_rp, e_rs), sc in zip(rows, scores):
            w.writerow([name, w_cow, w_lm, f"{rmse(e_l):.4f}", f"{np.nanmean(e_l):.4f}",
                        f"{np.median(e_rp):.3f}", f"{np.median(e_rs):.5f}", f"{sc:.3f}"])

    # 最良の設定で、全データを使って調整
    _n, w_cow, w_lm = rows[best][0], rows[best][1], rows[best][2]
    cams, info = bundle_adjust2(base, cows, landmarks, w_cow, w_lm, verbose=0)
    print(f"\n最良の設定で全データ({len(cows)}頭 + {len(landmarks)}点)を使って調整: nfev={info['nfev']}")
    print("\n--- パラメータの変化(調整前 -> 調整後) ---")
    for c in cam_ids:
        R0, t0, i0 = base[c]
        R1, t1, i1 = cams[c]
        rot_deg = float(np.degrees(np.linalg.norm(Rot.from_matrix(R0.T @ R1).as_rotvec())))
        print(f"  {c}: 向きの変化 {rot_deg:.3f}°  位置の変化 {np.round(t1 - t0, 4).tolist()} m")
        print("      内部: " + "  ".join(f"{nm} {i0[k]:.4f}->{i1[k]:.4f}" for k, nm in enumerate(b1.INTR_NAMES)
                                         if nm in b1.FREE_INTRINSICS))
    e_all = landmark_errors(cams, landmarks)
    print(f"\n(参考。調整に使った点を含む全32点) ランドマーク床誤差 RMSE: 調整なし {rmse(rows[0][3]):.3f}m -> {rmse(e_all):.3f}m")

    ov = b1.overrides_from_cams(cameras_json, cams)
    intr_txt = {c: {nm: round(float(cams[c][2][k]), 6) for k, nm in enumerate(b1.INTR_NAMES)} for c in cam_ids}
    snippet = ["# Mootrack_core.py に貼る内容(牛+ランドマーク、クロスバリデーションで選んだ設定)",
               f"# 設定: {rows[best][0]} (w_cow={w_cow}, w_lm={w_lm})", "CAMERA_PARAM_OVERRIDES = {"]
    snippet += [f'    "{k}": {v},' for k, v in ov.items()]
    snippet += ["}", "", "CAM_INTRINSICS = {"] + [f'    "{c}": {v},' for c, v in intr_txt.items()] + ["}"]
    (out_dir / "calibration_refined_cv_ver2.txt").write_text("\n".join(snippet), encoding="utf-8")
    (out_dir / "calibration_refined_cv_ver2.json").write_text(json.dumps({
        "frames": frames, "n_cows": len(cows), "n_landmarks": len(landmarks), "setting": rows[best][0],
        "w_cow": w_cow, "w_lm": w_lm,
        "cameras": {c: {"R": cams[c][0].tolist(), "t": cams[c][1].tolist(),
                        "intrinsics": dict(zip(b1.INTR_NAMES, map(float, cams[c][2])))} for c in cam_ids},
        "overrides": ov, "cv_score": scores[best], "cv_score_no_adjust": scores[0],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n出力: {out_dir}")
    if scores[best] >= scores[0]:
        print("※どの設定も、調整なしより良くなっていません。調整後の値は採用しないでください。")
    print("反映する場合は、evaluate_refined_calib_on_32landmarks_ver1.py の REFINED_JSON_PATH を "
          "calibration_refined_cv_ver2.json に変えて、床の誤差を確認すること。")


if __name__ == "__main__":
    main()