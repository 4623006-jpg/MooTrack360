"""
estimate_sigma_he_kai_ver4.py

ver4 (ver3 からの変更): 三角測量GT を使わず、YOLOの検出 + XMLの牛の名前 で推定する
  - サンプル: 対象フレーム(既定 0..2000 を 100刻み)でYOLOを回し、検出を XML の GT bbox と IoU で対応づけて、その牛の名前を付ける
      bbox中心 = YOLO検出の中心 (実際のマッチングの入力と同じ)、姿勢 = YOLO検出の姿勢
      同じフレームで同じ牛の姿勢がカメラ間で食い違う / 同じカメラに同じ名前が2つある牛は除く
  - 交差検証・ブートストラップは『牛の名前ごと』に分ける (同じ牛が複数フレームに出るため)
  - 外れ値: 初期推定で食い違いの裾(マハラノビス距離²)を表示。TRIM_CHI2>0 ならそれを超えるペアを除いて推定し直す (ID取り違え対策)
  - 出力: sigma_he_kai_ver4.json
  - 環境変数: ANN_XML_DIR (既定 .../annotation_process/xml-1fps, 牛の名前がカメラをまたいで同じもの) / HE_FRAME_START,END,STEP /
              IOU_THR (0.3) / TRIM_CHI2 (0=除かない) / N_BOOT

ver3 (ver2 からの変更): 出力 JSON を phase3 / he_kai_sigma_model_ver1.py が読める形にする。
  - kind="cam" は models に入れない (別キー "cam_experimental" に保存)。selected は読めるモデルの中で CV nll が最小のもの
  - 出力は sigma_he_kai_ver4.json。phase3 から使うときは HE_SIGMA_JSON=<このjson> を指定する (新しい校正で推定し直した σ を使うため)

ver2 (ver1 からの追加): σ_px をカメラごとの値にするモデル "cam" を追加し、共通の const と比べる。
  - cam: σ_px,c (c = Camera_1..3) を1回の最適化でまとめて推定する (σ_h, μ_h は姿勢ごとのまま)
  - 比較: AIC / 牛単位の交差検証 NLL / CV のカバレッジ / σ_px,c のブートストラップ95%区間 / σ_px,3 > σ_px,1 となる割合
  - 整合性チェック: カメラのペアごと(1-2, 2-3)に const モデルを別々に推定し、cam モデルから予測される値 √((σa²+σb²)/2) と比べる
  - 出力: sigma_he_kai_ver4.json (ver1 の json は上書きしない。kind="cam" は he_kai_sigma_model_ver1.py では未対応)

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

使い方: uv run python ./phase4/he_kai/estimate_sigma_he_kai_ver4.py
環境変数: N_BINS(既定4) H_A_STANDING(0.75) H_A_LYING(0.5) N_BOOT(100)
出力: <OUTPUT_DIR>/bev_calibration_refined/sigma_he_kai_ver4.json
校正は Mootrack_core.py の現在値(MOOTRACK_CALIB で切り替え可)。
"""

import json
from collections import Counter, defaultdict
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

    def __init__(self, kind, knots=None, n_cam=0):
        self.kind, self.knots = kind, (None if knots is None else np.asarray(knots, dtype=np.float64))
        self.n_cam = n_cam

    @property
    def n_sigma(self):
        if self.kind == "cam":
            return self.n_cam
        return {"const": 1, "linear": 2}.get(self.kind, len(self.knots) if self.knots is not None else 0)

    def sigma_px(self, ts, r, cam=None):
        if self.kind == "cam":
            return np.asarray(ts, dtype=np.float64)[np.asarray(cam)]
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
    spi, spj = spec.sigma_px(ts, data["r"][i], data["cam"][i]), spec.sigma_px(ts, data["r"][j], data["cam"][j])
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
def prepare(raw, poses, get_intrinsics, fh, cam_ids):
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
    data = dict(P=P, g=g, jxx=jxx, jxy=jxy, jyy=jyy, r=r, cam=np.array([cam_ids.index(s["cam"]) for s in raw]),
                pidx=np.array([POSES.index(s["pose"]) for s in raw]),
                i=np.array([p[0] for p in pairs]), j=np.array([p[1] for p in pairs]),
                cow=np.array([hash((raw[p[0]]["frame"], raw[p[0]]["cow"])) for p in pairs]))
    return data


_N_CAM = 3


def make_specs(data):
    used = np.unique(np.concatenate([data["i"], data["j"]]))
    rs = np.sort(data["r"][used])
    groups = np.array_split(rs, N_BINS)
    knots = [float(np.median(gp)) for gp in groups if len(gp)]
    return {"const": Spec("const"), "cam": Spec("cam", n_cam=_N_CAM), "linear": Spec("linear"), f"bins{len(knots)}": Spec("bins", knots)}


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
    if spec.kind == "cam":
        e["values"] = [float(x) for x in ts]
    elif spec.kind == "const":
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


def cmp_cam_vs_const(results, data, cam_ids, rng):
    """カメラ別 σ_px (cam) と共通 σ_px (const) の比較 + カメラのペアごとの整合性チェック"""
    rc, rm = results["const"], results["cam"]
    print(f"\n{'=' * 70}\n■ 共通 σ_px (const) と カメラ別 σ_px (cam) の比較\n{'=' * 70}")
    print(f"AIC          : const {rc['aic']:.1f} / cam {rm['aic']:.1f}   (cam - const = {rm['aic'] - rc['aic']:+.1f}; 負なら cam が良い)")
    print(f"CV nll       : const {rc['cv_nll']:.2f} / cam {rm['cv_nll']:.2f}   (cam - const = {rm['cv_nll'] - rc['cv_nll']:+.2f}; 負なら cam が良い)")
    print(f"CV 95/99%内  : const {rc['cov95'] * 100:.1f}/{rc['cov99'] * 100:.1f}% / cam {rm['cov95'] * 100:.1f}/{rm['cov99'] * 100:.1f}%  (目標 95/99%)")
    ts = rm["theta"][:len(cam_ids)]
    print("cam の σ_px  : " + " / ".join(f"{c} {v:.1f}px" for c, v in zip(cam_ids, ts)) + f"   (const: {rc['theta'][0]:.1f}px)")
    # サンプル数 (カメラのペアごと)
    ci, cj = data["cam"][data["i"]], data["cam"][data["j"]]
    pair_key = np.array([tuple(sorted((a, b))) for a, b in zip(ci, cj)])
    print("\nカメラのペアごとの組数:")
    uniq = sorted({tuple(k) for k in pair_key})
    for a, b in uniq:
        n = int(((pair_key[:, 0] == a) & (pair_key[:, 1] == b)).sum())
        print(f"  {cam_ids[a]} - {cam_ids[b]}: {n}組")
    # ブートストラップ (cam)
    print(f"\nブートストラップ({N_BOOT}回, 牛単位) で cam の σ_px,c の区間を計算中 ...")
    boots = bootstrap(rm["spec"], data, rm["theta"], rng)[:, :len(cam_ids)]
    lo, hi = np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)
    for k, c in enumerate(cam_ids):
        print(f"  σ_px {c}: {ts[k]:.1f}px  95%区間 [{lo[k]:.1f}, {hi[k]:.1f}]")
    for a in range(len(cam_ids)):
        for b in range(a + 1, len(cam_ids)):
            print(f"  P(σ_px {cam_ids[b]} > σ_px {cam_ids[a]}) = {(boots[:, b] > boots[:, a]).mean() * 100:.0f}%")
    print("  (区間が互いに大きく重なり、割合が 50% 付近なら、カメラ別の違いは検出できていない = 共通でよい)")
    # ペアごとの整合性チェック
    print(f"\n■ カメラのペアごとに const を別々に推定 (整合性チェック)")
    print("  ペアの σ_px は『2台の平均的な値』。cam の予測 √((σa²+σb²)/2) と近ければ cam モデルは整合的")
    print(f"  {'ペア':<22}{'組数':>6}{'σ_px(ペア別)':>14}{'cam から予測':>14}{'σ_h立位':>9}{'σ_h横臥':>9}{'μ_h立位':>9}{'μ_h横臥':>9}")
    for a, b in uniq:
        sel = np.where((pair_key[:, 0] == a) & (pair_key[:, 1] == b))[0]
        if len(sel) < 15:
            print(f"  {cam_ids[a]}-{cam_ids[b]}: {len(sel)}組 (少なすぎるので省略)")
            continue
        th, _ = fit(rc["spec"], data, sel)
        pred = float(np.sqrt((ts[a] ** 2 + ts[b] ** 2) / 2.0))
        print(f"  {cam_ids[a] + '-' + cam_ids[b]:<22}{len(sel):>6}{th[0]:>13.1f}p{pred:>13.1f}p{th[1]:>9.2f}{th[2]:>9.2f}{th[3]:>+9.2f}{th[4]:>+9.2f}")



# ---------------------------------------------------------------------------
# ver4: YOLO の検出 + XML の牛の名前
# ---------------------------------------------------------------------------
import importlib
import xml.etree.ElementTree as ET

ANN_XML_DIR = os.environ.get("ANN_XML_DIR", "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/annotation_process/xml-1fps")
FRAME_START = int(os.environ.get("HE_FRAME_START", 0))
FRAME_END = int(os.environ.get("HE_FRAME_END", 2000))
FRAME_STEP = int(os.environ.get("HE_FRAME_STEP", 100))
IOU_THR = float(os.environ.get("IOU_THR", 0.3))
TRIM_CHI2 = float(os.environ.get("TRIM_CHI2", 0))
YOLO_MODULE_CANDIDATES = ["phase3_bev_multicam_yolo_check_zonepose_ver6", "phase3_bev_multicam_yolo_check_zonepose_ver3",
                          "phase3_bev_multicam_yolo_check_zonepose_ver1"]


def load_yolo_module():
    last = None
    for name in YOLO_MODULE_CANDIDATES:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as e:
            last = e
    raise last


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def assign_names(gts, dets, thr=IOU_THR):
    """GT bbox(タイル内座標)とYOLO検出を、IoUの大きい順に1対1で対応づける。戻り値: {gt番号: det番号}"""
    cand = sorted(((iou(g, d["bbox_local"]), gi, di) for gi, g in enumerate(gts) for di, d in enumerate(dets)), reverse=True)
    used_g, used_d, out = set(), set(), {}
    for v, gi, di in cand:
        if v < thr:
            break
        if gi in used_g or di in used_d:
            continue
        used_g.add(gi)
        used_d.add(di)
        out[gi] = di
    return out


def read_gt(path):
    """XML -> [(牛の名前, [xmin, ymin, xmax, ymax])]"""
    out = []
    for obj in ET.parse(path).getroot().findall("object"):
        ne, bnd = obj.find("name"), obj.find("bndbox")
        if ne is None or bnd is None or not ne.text:
            continue
        out.append((ne.text.strip(), [float(bnd.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")]))
    return out


def build_raw_yolo(frames, ym, model, poses, cam_ids):
    """YOLO検出に牛の名前を付けたサンプル。同じフレームで、姿勢が食い違う牛・同じカメラに2つある牛は除く"""
    tile_w = ym.IMAGE_WIDTH_PX / len(cam_ids)
    raw, info = [], Counter()
    for fr in frames:
        xml = Path(ANN_XML_DIR) / f"{ym.frame_name(fr)}.xml"
        img = Path(ym.IMAGE_DIR) / f"{ym.frame_name(fr)}.jpg"
        if not (xml.exists() and img.exists()):
            print(f"  見つかりません: {xml if not xml.exists() else img}")
            continue
        gts = read_gt(xml)
        per_cam = ym.yolo_world_points_per_camera_with_pose(model, img, poses)
        samples = []
        for ci, cid in enumerate(cam_ids):
            g_list = [(nm, b) for nm, b in gts if max(0, min(int((b[0] + b[2]) / 2.0 // tile_w), len(cam_ids) - 1)) == ci]
            dets = per_cam.get(cid, [])
            info["gt_bbox"] += len(g_list)
            m = assign_names([[b[0] - ci * tile_w, b[1], b[2] - ci * tile_w, b[3]] for _, b in g_list], dets)
            info["matched"] += len(m)
            for gi, di in m.items():
                x1, y1, x2, y2 = dets[di]["bbox_local"]
                samples.append(dict(frame=fr, cow=g_list[gi][0], cam=cid, pose=dets[di].get("pose"),
                                    P_gt=np.zeros(2), uv=((x1 + x2) / 2.0, (y1 + y2) / 2.0)))
        by = defaultdict(list)
        for sm in samples:
            by[sm["cow"]].append(sm)
        for nm, lst in by.items():
            if any(sm["pose"] not in POSES for sm in lst):
                info["姿勢なし/不明で除外"] += 1
            elif len({sm["pose"] for sm in lst}) > 1:
                info["姿勢がカメラ間で食い違い -> 除外"] += 1
            elif len({sm["cam"] for sm in lst}) < len(lst):
                info["同じカメラに同名2つ -> 除外"] += 1
            else:
                raw += lst
                info["使った牛(フレームごと)"] += 1
        print(f"  frame {fr}: GT {len(gts)} / 使ったサンプル累計 {len(raw)}")
    return raw, info


def main():
    import phase3_bev_multicam_gt_zonepose_ver1 as g
    import Mootrack_core as mc
    import fit_assumed_height_vs_tri_gt_ver1 as fh
    from ultralytics import YOLO
    from Mootrack_core import ROT_FMT, apply_camera_overrides, get_intrinsics, load_floorplan_cameras, resolve_poses

    out_dir = g.OUTPUT_DIR / "bev_calibration_refined"
    out_dir.mkdir(parents=True, exist_ok=True)
    ym = load_yolo_module()
    frames = list(range(FRAME_START, FRAME_END + 1, FRAME_STEP))
    cam_ids = list(g.CAM_IDS)
    poses = resolve_poses(apply_camera_overrides(load_floorplan_cameras(g.FLOORPLAN_JSON_PATH)), root="Area1", fmt=ROT_FMT)
    print(f"校正セット: {getattr(mc, 'CALIBRATION_SET', 'Mootrack_core.pyの現在値')} / フレーム {len(frames)}個 ({frames[0]}..{frames[-1]}) / 想定高さ h_a={H_A}")
    print(f"牛の名前: {ANN_XML_DIR}")
    model = YOLO(ym.YOLO_WEIGHTS_PATH)

    raw, info = build_raw_yolo(frames, ym, model, poses, cam_ids)
    print("\n■ サンプル作成の内訳: " + " / ".join(f"{k} {v}" for k, v in info.items()))
    global _N_CAM
    _N_CAM = len(cam_ids)
    data = prepare(raw, poses, get_intrinsics, fh, cam_ids)
    names = sorted({s["cow"] for s in raw})
    name_id = {n: k for k, n in enumerate(names)}                         # 牛の名前 -> 整数 (交差検証・ブートストラップはこの単位)
    data["cow"] = np.array([name_id[raw[a]["cow"]] for a in data["i"]])
    n_pairs = len(data["i"])
    print(f"サンプル {len(raw)}件 / 牛の名前 {len(np.unique(data['cow']))}頭 / 2カメラの組 {n_pairs}")
    ci, cj = data["cam"][data["i"]], data["cam"][data["j"]]
    for a in range(len(cam_ids)):
        for b in range(a + 1, len(cam_ids)):
            n = int((((ci == a) & (cj == b)) | ((ci == b) & (cj == a))).sum())
            if n:
                print(f"  {cam_ids[a]}-{cam_ids[b]}: {n}組")
    for k, p in enumerate(POSES):
        n_p = int(((data["pidx"][data["i"]] == k) | (data["pidx"][data["j"]] == k)).sum())
        print(f"  {p}: 関わるペア {n_p}")
    if n_pairs < 20:
        print("ペアが少なすぎて推定できません")
        return

    # 外れ値(食い違いの裾)の確認
    th0, _ = fit(Spec("const"), data, np.arange(n_pairs))
    _, m2 = nll_terms(th0, Spec("const"), data, np.arange(n_pairs))
    print(f"\n■ 初期推定(const)のマハラノビス距離²: 中央値 {np.median(m2):.2f}(理想 1.39) / 90%点 {np.percentile(m2, 90):.1f}(理想 4.6)"
          f" / 99%超(>{CHI2_99}): {(m2 > CHI2_99).mean() * 100:.1f}% / >13.8(99.9%): {(m2 > 13.8).mean() * 100:.1f}% / 最大 {m2.max():.1f}")
    if TRIM_CHI2 > 0:
        keep = np.where(m2 <= TRIM_CHI2)[0]
        print(f"  TRIM_CHI2={TRIM_CHI2}: {n_pairs - len(keep)}組を除いて推定し直す")
        for key in ("i", "j", "cow"):
            data[key] = data[key][keep]
        n_pairs = len(keep)

    rng = np.random.default_rng(SEED)
    results = run_estimation(data, rng)

    best = min(results, key=lambda n: results[n]["cv_nll"])
    print(f"\n選択(CV nll が最小): {best}   ※ 差が小さいモデルどうしは、パラメータの少ない const を優先してよい")
    cmp_cam_vs_const(results, data, cam_ids, rng)

    spec, theta = results[best]["spec"], results[best]["theta"]
    print(f"\nブートストラップ({N_BOOT}回, 牛単位) 中 ...")
    boots = bootstrap(spec, data, theta, rng)
    lo, hi = np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)
    labels = ([f"σ_px(r={spec.knots[k]:.0f}px)" for k in range(spec.n_sigma)] if spec.kind == "bins"
              else [f"σ_px {c}" for c in cam_ids] if spec.kind == "cam"
              else {"const": ["σ_px"], "linear": ["σ_px: a", "σ_px: b(/1000px)"]}[spec.kind])
    labels += ["σ_h standing[m]", "σ_h lying[m]", "μ_h standing[m]", "μ_h lying[m]"]
    print(f"{'パラメータ':<22}{'推定値':>10}{'95%区間':>22}")
    for lab, v, a, b in zip(labels, theta, lo, hi):
        print(f"{lab:<22}{v:>10.3f}{('[%.3f, %.3f]' % (a, b)):>22}")
    ns = spec.n_sigma
    for k, p in enumerate(POSES):
        print(f"  h_eff {p}: {H_A[p] + theta[ns + 2 + k]:.3f} m  (h_a {H_A[p]} + μ_h {theta[ns + 2 + k]:+.3f})")

    models = {}
    cam_entry = None
    for name, r_ in results.items():
        if r_["spec"].kind == "cam":
            cam_entry = entry_from_theta(r_["spec"], r_["theta"], dict(
                nll=r_["nll"], aic=r_["aic"], cv_nll=r_["cv_nll"], cv_cov95=r_["cov95"], cv_cov99=r_["cov99"], n_pairs=n_pairs, cam_ids=cam_ids))
            continue
        models[name] = entry_from_theta(r_["spec"], r_["theta"], dict(
            nll=r_["nll"], aic=r_["aic"], cv_nll=r_["cv_nll"], cv_cov95=r_["cov95"], cv_cov99=r_["cov99"], n_pairs=n_pairs))
    out = out_dir / "sigma_he_kai_ver4.json"
    best_readable = min(models, key=lambda n: models[n]["cv_nll"])
    print(f"JSON の selected(phase3 が読めるモデルの中で CV nll 最小): {best_readable}")
    out.write_text(json.dumps(dict(selected=best_readable, frames=frames, models=models, cam_experimental=cam_entry),
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nJSON: {out}")
    print("※ YOLO検出 + XMLの牛の名前で推定。CV・ブートストラップは牛の名前単位。姿勢はYOLO(カメラ間で一致したもののみ)。")


if __name__ == "__main__":
    main()