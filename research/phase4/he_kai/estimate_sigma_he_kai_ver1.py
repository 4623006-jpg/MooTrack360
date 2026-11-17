"""
estimate_sigma_he_kai_ver1.py

HE(改)の σ を、三角測量GTの牛の「カメラ間の食い違い」から最尤推定する。
  - σ_px を画像の半径 r の関数にする(const / linear / bins の3モデルを比べる)
  - 高さの誤差は 平均μ_h(姿勢ごと) と ばらつきσ_h(姿勢ごと) を同時に推定する
      (μ_h は有効な想定高さのずれ: h_eff = h_a + μ_h。σ_h にバイアスが混ざるのを防ぐ)
  - 牛単位の交差検証で、保持した牛の食い違いがχ²の95%/99%点に収まる割合(カバレッジ)と、
    保持データの負の対数尤度を出し、いちばん良いモデルを選ぶ
  - 選んだモデルについて、牛単位のブートストラップで95%区間を出す

モデル(同じ牛の2カメラ i, j の食い違い d = P_i - P_j。P は bbox中心を想定高さ h_a で床に投影した位置):
    d ~ N( g_i μ_i - g_j μ_j ,  σ_px(r_i)² J_i J_iᵀ + σ_px(r_j)² J_j J_jᵀ + σ_h,i² g_i g_iᵀ + σ_h,j² g_j g_jᵀ )
    J: 画素 -> 床位置のヤコビアン(HE)。 g = ρ/(H-h_a) · u (高さ1mぶんの半径方向のずれ)

使い方: python estimate_sigma_he_kai_ver1.py [フレーム番号 ...]   (省略時はGTのある全フレーム)
環境変数: N_BINS(既定4) H_A_STANDING(0.75) H_A_LYING(0.5) N_BOOT(100)
出力: <OUTPUT_DIR>/bev_calibration_refined/sigma_he_kai_ver1.json
      (phase3_he_kai_matching_ver2.py が読む)
校正は Mootrack_core.py の現在値(MOOTRACK_CALIB で切り替え可)。
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# 配置: research/phase4/he_kai/(このファイル) など。research 直下に他のフォルダ
RESEARCH_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
PHASE4_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
MOOTRACK_CODE_DIR = (os.path.abspath(os.path.join(RESEARCH_DIR, "calibration")))
for d in (RESEARCH_DIR, PHASE4_DIR, os.path.join(PHASE4_DIR, "web_annotate"),
          os.path.join(PHASE4_DIR, "calibration"), os.path.join(PHASE4_DIR, "calibration_try"),
          os.path.join(PHASE4_DIR, "he_kai"),
          MOOTRACK_CODE_DIR,
          os.path.join(RESEARCH_DIR, "analyze"), os.path.join(RESEARCH_DIR, "annotate"),
          os.path.join(RESEARCH_DIR, "phase3_multi_camera"), THIS_DIR):
    if os.path.isdir(d) and d not in sys.path:
        sys.path.insert(0, d)

POSES = ("standing", "lying")
N_BINS = int(os.environ.get("N_BINS", 4))
H_A = {"standing": float(os.environ.get("H_A_STANDING", 0.75)), "lying": float(os.environ.get("H_A_LYING", 0.5))}
N_BOOT = int(os.environ.get("N_BOOT", 100))
N_FOLDS = 5
SEED = 0
CHI2_95, CHI2_99 = 5.99, 9.21
SIG_PX_BOUNDS = (2.0, 200.0)
SIG_H_BOUNDS = (0.02, 1.5)
MU_H_BOUNDS = (-0.6, 1.0)


# ---------------------------------------------------------------------------
# モデル
# ---------------------------------------------------------------------------
class Spec:
    """σ_px(r) の形。パラメータ ts の並びは kind ごとに違う"""

    def __init__(self, kind, knots=None):
        self.kind, self.knots = kind, (None if knots is None else np.asarray(knots, dtype=np.float64))

    @property
    def n_sigma(self):
        return {"const": 1, "linear": 2}.get(self.kind, len(self.knots) if self.knots is not None else 0)

    def sigma_px(self, ts, r):
        if self.kind == "const":
            return np.full(np.shape(r), ts[0])
        if self.kind == "linear":
            return np.maximum(ts[0] + ts[1] * np.asarray(r) / 1000.0, 1.0)
        return np.interp(r, self.knots, ts)

    def x0(self):
        if self.kind == "linear":
            return [16.0, 0.0]
        return [16.0] * self.n_sigma

    def bounds(self):
        if self.kind == "linear":
            return [SIG_PX_BOUNDS, (-100.0, 200.0)]
        return [SIG_PX_BOUNDS] * self.n_sigma


def split_theta(theta, spec):
    ns = spec.n_sigma
    return theta[:ns], theta[ns:ns + 2], theta[ns + 2:ns + 4]


def pair_stats(theta, spec, data, sel):
    """選んだペアの (平均を引いた差 d, 共分散 Sxx, Sxy, Syy)"""
    ts, sh, mu = split_theta(theta, spec)
    i, j = data["i"][sel], data["j"][sel]
    pi, pj = data["pidx"][i], data["pidx"][j]
    spi, spj = spec.sigma_px(ts, data["r"][i]), spec.sigma_px(ts, data["r"][j])
    gi, gj = data["g"][i], data["g"][j]
    sxx = spi ** 2 * data["jxx"][i] + spj ** 2 * data["jxx"][j] + sh[pi] ** 2 * gi[:, 0] ** 2 + sh[pj] ** 2 * gj[:, 0] ** 2
    sxy = spi ** 2 * data["jxy"][i] + spj ** 2 * data["jxy"][j] + sh[pi] ** 2 * gi[:, 0] * gi[:, 1] + sh[pj] ** 2 * gj[:, 0] * gj[:, 1]
    syy = spi ** 2 * data["jyy"][i] + spj ** 2 * data["jyy"][j] + sh[pi] ** 2 * gi[:, 1] ** 2 + sh[pj] ** 2 * gj[:, 1] ** 2
    d = data["P"][i] - data["P"][j] - (gi * mu[pi][:, None] - gj * mu[pj][:, None])
    return d, sxx, sxy, syy


def nll_terms(theta, spec, data, sel):
    d, sxx, sxy, syy = pair_stats(theta, spec, data, sel)
    det = np.maximum(sxx * syy - sxy ** 2, 1e-12)
    m2 = (syy * d[:, 0] ** 2 - 2 * sxy * d[:, 0] * d[:, 1] + sxx * d[:, 1] ** 2) / det
    return 0.5 * (np.log(det) + m2), m2


def fit(spec, data, sel, x_init=None):
    x0 = np.array(spec.x0() + [0.4, 0.3] + [0.0, 0.0]) if x_init is None else np.asarray(x_init, dtype=np.float64)
    bounds = spec.bounds() + [SIG_H_BOUNDS] * 2 + [MU_H_BOUNDS] * 2
    res = minimize(lambda th: nll_terms(th, spec, data, sel)[0].sum(), x0, method="L-BFGS-B", bounds=bounds)
    return res.x, float(res.fun)


# ---------------------------------------------------------------------------
# データ準備
# ---------------------------------------------------------------------------
def prepare(raw, poses, get_intrinsics, fh):
    from Mootrack_core import unproject_to_floor
    n = len(raw)
    h_a = np.array([H_A[s["pose"]] for s in raw])
    P, g = np.zeros((n, 2)), np.zeros((n, 2))
    jxx, jxy, jyy, r = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    ok = np.zeros(n, dtype=bool)
    for cid in {s["cam"] for s in raw}:
        idx = np.array([i for i, s in enumerate(raw) if s["cam"] == cid])
        m = len(idx)
        uv = np.array([raw[i]["uv"] for i in idx], dtype=np.float64)
        fx, fy, cx, cy, xi, alpha = get_intrinsics(cid)
        cp = poses[cid]
        pts = np.concatenate([uv, uv + np.array([1.0, 0.0]), uv + np.array([0.0, 1.0])])
        xy, v = unproject_to_floor(pts, cp.R, cp.t, fx, fy, cx, cy, xi, alpha, floor_z=-np.tile(h_a[idx], 3))
        xy0, xy1, xy2 = xy[:m], xy[m:2 * m], xy[2 * m:]
        a, b = xy1 - xy0, xy2 - xy0
        cam_xy, H = np.asarray(cp.t[:2], dtype=np.float64), float(-cp.t[2])
        dd = xy0 - cam_xy
        rho = np.maximum(np.linalg.norm(dd, axis=1), 1e-6)
        P[idx], ok[idx] = xy0, v[:m] & v[m:2 * m] & v[2 * m:]
        jxx[idx], jxy[idx], jyy[idx] = a[:, 0] ** 2 + b[:, 0] ** 2, a[:, 0] * a[:, 1] + b[:, 0] * b[:, 1], a[:, 1] ** 2 + b[:, 1] ** 2
        g[idx] = (rho / np.maximum(H - h_a[idx], 0.3))[:, None] * (dd / rho[:, None])
        r[idx] = np.hypot(uv[:, 0] - cx, uv[:, 1] - cy)
    pairs = [(a, b) for a, b in fh.group_pairs(raw) if ok[a] and ok[b]]
    data = dict(P=P, g=g, jxx=jxx, jxy=jxy, jyy=jyy, r=r,
                pidx=np.array([POSES.index(s["pose"]) for s in raw]),
                i=np.array([p[0] for p in pairs]), j=np.array([p[1] for p in pairs]),
                cow=np.array([hash((raw[p[0]]["frame"], raw[p[0]]["cow"])) for p in pairs]))
    return data


def make_specs(data):
    used = np.unique(np.concatenate([data["i"], data["j"]]))
    rs = np.sort(data["r"][used])
    groups = np.array_split(rs, N_BINS)
    knots = [float(np.median(gp)) for gp in groups if len(gp)]
    return {"const": Spec("const"), "linear": Spec("linear"), f"bins{len(knots)}": Spec("bins", knots)}


# ---------------------------------------------------------------------------
# 交差検証 / ブートストラップ
# ---------------------------------------------------------------------------
def cow_folds(data, rng, n_folds=N_FOLDS):
    cows = np.unique(data["cow"])
    rng.shuffle(cows)
    fold_of = {c: k % n_folds for k, c in enumerate(cows)}
    return np.array([fold_of[c] for c in data["cow"]])


def cross_validate(spec, data, rng):
    fold = cow_folds(data, rng)
    nll_sum, m2_all = 0.0, []
    for k in range(N_FOLDS):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        if len(tr) < 10 or len(te) == 0:
            continue
        th, _ = fit(spec, data, tr)
        t, m2 = nll_terms(th, spec, data, te)
        nll_sum += float(t.sum())
        m2_all.append(m2)
    m2_all = np.concatenate(m2_all)
    return nll_sum, float((m2_all <= CHI2_95).mean()), float((m2_all <= CHI2_99).mean())


def bootstrap(spec, data, theta0, rng, n_boot=N_BOOT):
    cows = np.unique(data["cow"])
    by_cow = {c: np.where(data["cow"] == c)[0] for c in cows}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(cows, size=len(cows), replace=True)
        sel = np.concatenate([by_cow[c] for c in pick])
        th, _ = fit(spec, data, sel, x_init=theta0)
        out.append(th)
    return np.array(out)


# ---------------------------------------------------------------------------
def entry_from_theta(spec, theta, extra):
    ts, sh, mu = split_theta(theta, spec)
    e = dict(kind=spec.kind, sigma_h=dict(zip(POSES, map(float, sh))), mu_h=dict(zip(POSES, map(float, mu))),
             h_a=dict(H_A), **extra)
    if spec.kind == "const":
        e["value"] = float(ts[0])
    elif spec.kind == "linear":
        e["a"], e["b"] = float(ts[0]), float(ts[1])
    else:
        e["r_knots"], e["values"] = [float(x) for x in spec.knots], [float(x) for x in ts]
    return e


def run_estimation(data, rng):
    specs = make_specs(data)
    sel_all = np.arange(len(data["i"]))
    print(f"\n{'モデル':<10}{'パラメータ数':>10}{'nll(全体)':>12}{'AIC':>10}{'CV nll':>10}{'CV 95%点内':>11}{'CV 99%点内':>11}")
    results = {}
    for name, spec in specs.items():
        theta, nll = fit(spec, data, sel_all)
        k = spec.n_sigma + 4
        cv_nll, cov95, cov99 = cross_validate(spec, data, rng)
        results[name] = dict(spec=spec, theta=theta, nll=nll, aic=2 * nll + 2 * k, cv_nll=cv_nll, cov95=cov95, cov99=cov99, k=k)
        print(f"{name:<10}{k:>10}{nll:>12.2f}{2 * nll + 2 * k:>10.1f}{cv_nll:>10.2f}{cov95 * 100:>10.1f}%{cov99 * 100:>10.1f}%")
    return results


def main():
    import phase3_bev_multicam_gt_zonepose_ver1 as g
    import Mootrack_core as mc
    import analyze_projection_error_vs_tri_gt_ver1 as an
    import fit_assumed_height_vs_tri_gt_ver1 as fh
    from Mootrack_core import ROT_FMT, apply_camera_overrides, get_intrinsics, load_floorplan_cameras, resolve_poses

    in_dir = g.OUTPUT_DIR / "bev_gt_triangulated"
    out_dir = g.OUTPUT_DIR / "bev_calibration_refined"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = [int(a) for a in sys.argv[1:]] or sorted(int(p.name[:6]) for p in in_dir.glob("*_tri_gt_ver1.json"))
    cam_ids = list(g.CAM_IDS)
    poses = resolve_poses(apply_camera_overrides(load_floorplan_cameras(g.FLOORPLAN_JSON_PATH)), root="Area1", fmt=ROT_FMT)
    print(f"校正セット: {getattr(mc, 'CALIBRATION_SET', 'Mootrack_core.pyの現在値')} / フレーム {frames} / 想定高さ h_a={H_A}")

    raw = []
    for fr in frames:
        xp, tp = Path(an.XML_DIR) / f"{fr:06d}.xml", in_dir / f"{fr:06d}_tri_gt_ver1.json"
        if xp.exists() and tp.exists():
            raw += fh.build_raw(json.loads(tp.read_text(encoding="utf-8"))["cows"], an.parse_xml(xp), cam_ids, an, fr)
        else:
            print(f"見つかりません: {xp if not xp.exists() else tp}")
    data = prepare(raw, poses, get_intrinsics, fh)
    n_pairs = len(data["i"])
    print(f"サンプル {len(raw)}件 / 牛 {len(np.unique(data['cow']))}頭 / 2カメラの組 {n_pairs}")
    for k, p in enumerate(POSES):
        n_p = int(((data["pidx"][data["i"]] == k) | (data["pidx"][data["j"]] == k)).sum())
        print(f"  {p}: 関わるペア {n_p}")
    if n_pairs < 20:
        print("ペアが少なすぎて推定できません")
        return

    rng = np.random.default_rng(SEED)
    results = run_estimation(data, rng)
    best = min(results, key=lambda n: results[n]["cv_nll"])
    print(f"\n選択(CV nll が最小): {best}   ※ 差が小さいモデルどうしは、パラメータの少ない const を優先してよい")

    spec, theta = results[best]["spec"], results[best]["theta"]
    print(f"\nブートストラップ({N_BOOT}回, 牛単位) 中 ...")
    boots = bootstrap(spec, data, theta, rng)
    lo, hi = np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)
    labels = ([f"σ_px(r={spec.knots[k]:.0f}px)" for k in range(spec.n_sigma)] if spec.kind == "bins"
              else {"const": ["σ_px"], "linear": ["σ_px: a", "σ_px: b(/1000px)"]}[spec.kind])
    labels += ["σ_h standing[m]", "σ_h lying[m]", "μ_h standing[m]", "μ_h lying[m]"]
    print(f"{'パラメータ':<22}{'推定値':>10}{'95%区間':>22}")
    for lab, v, a, b in zip(labels, theta, lo, hi):
        print(f"{lab:<22}{v:>10.3f}{('[%.3f, %.3f]' % (a, b)):>22}")
    ns = spec.n_sigma
    for k, p in enumerate(POSES):
        print(f"  h_eff {p}: {H_A[p] + theta[ns + 2 + k]:.3f} m  (h_a {H_A[p]} + μ_h {theta[ns + 2 + k]:+.3f})")

    models = {}
    for name, r_ in results.items():
        models[name] = entry_from_theta(r_["spec"], r_["theta"], dict(
            nll=r_["nll"], aic=r_["aic"], cv_nll=r_["cv_nll"], cv_cov95=r_["cov95"], cv_cov99=r_["cov99"], n_pairs=n_pairs))
    out = out_dir / "sigma_he_kai_ver1.json"
    out.write_text(json.dumps(dict(selected=best, frames=frames, models=models), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nJSON: {out}")
    print("※ 学習も検証も同じフレームの牛。別フレーム/YOLO検出での確認が必要。CV は牛単位で分けた保持データでの値。")


if __name__ == "__main__":
    main()