"""
heatmap_sigma_he_kai_ver2.py

HE(改)の位置誤差 e = sqrt(λmax(Σ)) [m] を、カメラごとに BEV(牛舎の床)上のヒートマップにする。
  Σ = σ_px² J Jᵀ + σ_h² g gᵀ   (J: 画素->床のヤコビアン, g: 高さ1mの違いによる床上のずれ(放射方向))
  色: e [m] (小さいほど信頼できる)
  等高線: k·e = ε  (k = 1, 2, 3 ...) つまり e = ε/k の線。既定は ε=0.425 m, k=1,2,3 -> e = 0.425 / 0.2125 / 0.1417 m
          (k=1 の線の内側が、今の有効範囲(maha_h_valid)と同じ)
  楕円: 格子状の数か所に、その位置の Σ の楕円(1σ)を重ねる(向きと形を見る)。

使い方(research/ で):
    uv run python ./phase4/he_kai/heatmap_sigma_he_kai_ver2.py
環境変数:
    HE_SIGMA_JSON / HE_SIGMA_MODEL : σ の JSON とモデル(既定: sigma_he_kai_ver4.json の const)
    HE_EPS_M          : ε [m] (既定 0.425)
    HE_HEAT_KS        : 等高線に使う k のカンマ区切り (既定 "1,2,3")
    HE_HEAT_TERM      : total / px / h  (既定 total。px=画素項だけ、h=高さ項だけ。等高線は選んだ項の値で引く)
    HE_HEAT_VMAX      : 色の上限 [m] (既定 1.5)
    HE_HEAT_ELLIPSE   : 1=楕円を重ねる(既定) / 0=重ねない
    HE_ELLIPSE_SCALE  : 楕円の見た目の拡大率(既定 2.0。1σ の楕円は牛舎に対して小さいので拡大して描く)
    HE_ELLIPSE_STEP_M : 楕円を置く間隔 [m] (既定 3.0)
    HE_HEAT_NANG      : 方向の数 (既定 360)
出力: <OUTPUT_DIR>/he_kai_heatmap_ver2/heatmap_<term>_eps<ε>.png
配置: research/phase4/he_kai/
"""

import os
import sys
from pathlib import Path

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESEARCH_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
PHASE4_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
MOOTRACK_CODE_DIR = os.path.abspath(os.path.join(RESEARCH_DIR, "calibration"))
for d in (RESEARCH_DIR, PHASE4_DIR, os.path.join(PHASE4_DIR, "web_annotate"), os.path.join(PHASE4_DIR, "calibration"),
          os.path.join(PHASE4_DIR, "calibration_try"), os.path.join(PHASE4_DIR, "he_kai"),
          MOOTRACK_CODE_DIR,
          os.path.join(RESEARCH_DIR, "analyze"), os.path.join(RESEARCH_DIR, "annotate"),
          os.path.join(RESEARCH_DIR, "phase3_multi_camera"), THIS_DIR):
    if os.path.isdir(d) and d not in sys.path:
        sys.path.insert(0, d)

# σ のJSONとモデルの既定(環境変数で上書きできる)
os.environ.setdefault("HE_SIGMA_MODEL", "const")
DEFAULT_SIGMA_JSON = os.path.join(RESEARCH_DIR, "idf1_check", "bev_calibration_refined", "sigma_he_kai_ver4.json")

EPS_M = float(os.environ.get("HE_EPS_M", 0.425))
KS = [float(x) for x in os.environ.get("HE_HEAT_KS", "1,2,3").split(",") if x.strip()]
TERM = os.environ.get("HE_HEAT_TERM", "total")            # total / px / h
VMAX = float(os.environ.get("HE_HEAT_VMAX", 1.5))
DRAW_ELLIPSE = os.environ.get("HE_HEAT_ELLIPSE", "1") == "1"
ELLIPSE_SCALE = float(os.environ.get("HE_ELLIPSE_SCALE", 2.0))
ELLIPSE_STEP_M = float(os.environ.get("HE_ELLIPSE_STEP_M", 3.0))
N_ANG = int(os.environ.get("HE_HEAT_NANG", 360))
R_STEP_PX = 10


def lam_max(sxx, sxy, syy):
    tr = sxx + syy
    det = sxx * syy - sxy * sxy
    return tr / 2.0 + np.sqrt(np.maximum(tr * tr / 4.0 - det, 0.0))


def compute_field(pose, intr, model, unproject_to_floor, height_m, max_r_px, n_ang=N_ANG):
    """カメラ1台分の誤差の場。画像の (r, φ) の格子を床に逆投影し、各点で Σ を作る。
    戻り値: dict(X, Y: 床座標の格子 (n_r, n_ang+1), e_total/e_px/e_h: 標準偏差[m], sxx/sxy/syy: Σ(total))"""
    fx, fy, cx, cy, xi, alpha = intr
    rs = np.arange(R_STEP_PX, max_r_px + 1, R_STEP_PX, dtype=np.float64)
    phis = np.linspace(0.0, 2.0 * np.pi, n_ang + 1)               # 端を重ねて一周を閉じる
    RR, PP = np.meshgrid(rs, phis, indexing="ij")
    uv = np.stack([cx + RR.ravel() * np.cos(PP.ravel()), cy + RR.ravel() * np.sin(PP.ravel())], axis=1)
    n = len(uv)
    pts = np.concatenate([uv, uv + np.array([1.0, 0.0]), uv + np.array([0.0, 1.0])])
    xy, v = unproject_to_floor(pts, pose.R, pose.t, fx, fy, cx, cy, xi, alpha, floor_z=-height_m)
    xy0, xy1, xy2 = xy[:n], xy[n:2 * n], xy[2 * n:]
    ok = v[:n] & v[n:2 * n] & v[2 * n:]
    a, b = xy1 - xy0, xy2 - xy0                                    # J の2列 (u方向, v方向)
    jxx, jxy, jyy = a[:, 0] ** 2 + b[:, 0] ** 2, a[:, 0] * a[:, 1] + b[:, 0] * b[:, 1], a[:, 1] ** 2 + b[:, 1] ** 2
    cam_xy = np.asarray(pose.t[:2], dtype=np.float64)
    H = float(-pose.t[2])
    d = xy0 - cam_xy
    rho = np.maximum(np.linalg.norm(d, axis=1), 1e-6)
    u = d / rho[:, None]
    gmag = rho / max(H - height_m, 0.3)
    sp = model.sigma_px(RR.ravel())
    sh = model.sigma_h_max()
    pxx, pxy, pyy = sp ** 2 * jxx, sp ** 2 * jxy, sp ** 2 * jyy
    hxx, hxy, hyy = sh ** 2 * (gmag * u[:, 0]) ** 2, sh ** 2 * gmag ** 2 * u[:, 0] * u[:, 1], sh ** 2 * (gmag * u[:, 1]) ** 2
    out = dict(X=xy0[:, 0], Y=xy0[:, 1], cam_xy=cam_xy)
    for key, (sxx, sxy, syy) in dict(total=(pxx + hxx, pxy + hxy, pyy + hyy), px=(pxx, pxy, pyy), h=(hxx, hxy, hyy)).items():
        e = np.sqrt(lam_max(sxx, sxy, syy))
        e[~ok] = np.nan
        out["e_" + key] = e
        out["s_" + key] = (sxx, sxy, syy)
    shape = RR.shape
    for k in list(out):
        if k in ("cam_xy",) or k.startswith("s_"):
            continue
        out[k] = out[k].reshape(shape)
    out["ok"] = ok.reshape(shape)
    return out


def ellipse_params(sxx, sxy, syy):
    """Σ の楕円: (長軸の標準偏差, 短軸の標準偏差, 長軸の角度[deg])"""
    lam = np.linalg.eigvalsh(np.array([[sxx, sxy], [sxy, syy]], dtype=np.float64))
    ang = 0.5 * np.degrees(np.arctan2(2.0 * sxy, sxx - syy))
    return float(np.sqrt(max(lam[1], 0.0))), float(np.sqrt(max(lam[0], 0.0))), float(ang)


def plot_fields(fields, real_w, real_h, eps_m, ks, term, vmax, out_path, title_suffix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Ellipse, Rectangle

    cids = list(fields)
    fig, axes = plt.subplots(len(cids), 1, figsize=(13, 4.2 * len(cids)), squeeze=False)
    colors = ["white", "black", "magenta", "cyan", "red", "lime"]
    for ax, cid in zip(axes[:, 0], cids):
        f = fields[cid]
        E = f["e_" + term]
        X, Y = f["X"].reshape(E.shape), f["Y"].reshape(E.shape)
        m = ax.contourf(X, Y, np.ma.masked_invalid(E), levels=np.linspace(0.0, vmax, 31), cmap="turbo", extend="max")
        levels = sorted(eps_m / k for k in ks)
        handles = []
        for lv, col in zip(levels, colors):
            k = eps_m / lv
            Ez = np.where(np.isfinite(E), E, np.inf)
            try:
                ax.contour(X, Y, Ez, levels=[lv], colors=[col], linewidths=1.8)
            except Exception:
                pass
            handles.append(Line2D([0], [0], color=col, lw=2, label=f"k={k:g}: e={lv:.3f} m"))
        if DRAW_ELLIPSE:
            s_xx, s_xy, s_yy = f["s_" + term]
            Xf, Yf, ok = f["X"].ravel(), f["Y"].ravel(), f["ok"].ravel()
            for gx in np.arange(ELLIPSE_STEP_M / 2, real_w, ELLIPSE_STEP_M):
                for gy in np.arange(ELLIPSE_STEP_M / 2, real_h, ELLIPSE_STEP_M):
                    dd = np.hypot(Xf - gx, Yf - gy)
                    dd[~ok] = np.inf
                    i = int(np.argmin(dd))
                    if dd[i] > 0.5 * ELLIPSE_STEP_M:
                        continue
                    a1, a2, ang = ellipse_params(s_xx[i], s_xy[i], s_yy[i])
                    if not np.isfinite(a1) or a1 <= 0:
                        continue
                    ax.add_patch(Ellipse((Xf[i], Yf[i]), 2 * a1 * ELLIPSE_SCALE, 2 * a2 * ELLIPSE_SCALE, angle=ang,
                                         fill=False, ec="black", lw=0.8))
        ax.add_patch(Rectangle((0, 0), real_w, real_h, fill=False, ec="black", lw=2))
        cx_, cy_ = f["cam_xy"]
        ax.plot([cx_], [cy_], marker="^", color="white", mec="black", ms=11, zorder=5)
        ax.text(cx_, cy_, " " + cid, color="black", fontsize=10, weight="bold", zorder=6)
        ax.set_xlim(-1, real_w + 1)
        ax.set_ylim(-1, real_h + 1)
        ax.set_aspect("equal")
        ax.set_title(f"{cid}   e = sqrt(lambda_max) [{term}]   eps={eps_m} m {title_suffix}")
        ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9, facecolor="lightgray")
        fig.colorbar(m, ax=ax, fraction=0.025, pad=0.01, label="std of position error e [m]")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    import importlib
    from Mootrack_core import (CAM_IDS, ROT_FMT, apply_camera_overrides, get_intrinsics, load_floorplan_cameras,
                               load_floorplan_dimensions, resolve_poses, unproject_to_floor)
    from he_kai_sigma_model_ver1 import load_model
    try:
        from compute_acceptable_fisheye_radius import MAX_RADIUS_PX
    except Exception:
        MAX_RADIUS_PX = 1400
    try:
        from curved_zone_utils_ver1 import ZONE_HEIGHT_M
    except Exception:
        ZONE_HEIGHT_M = 0.75
    hk = importlib.import_module(os.environ.get("HE_MATCHING_MODULE", "phase3_he_kai_matching_ver10"))
    ym = hk.load_yolo_module()

    json_path = os.environ.get("HE_SIGMA_JSON") or DEFAULT_SIGMA_JSON
    model = load_model(json_path)
    print(model.describe())
    print(f"ε={EPS_M} m / k={KS} / 項={TERM} / 高さ仮定(床の高さ)={ZONE_HEIGHT_M} m / 方向数={N_ANG}")

    cams = apply_camera_overrides(load_floorplan_cameras(ym.FLOORPLAN_JSON_PATH))
    poses = resolve_poses(cams, root="Area1", fmt=ROT_FMT)
    real_w, real_h = load_floorplan_dimensions(ym.FLOORPLAN_JSON_PATH)

    fields = {}
    for cid in CAM_IDS:
        fields[cid] = compute_field(poses[cid], get_intrinsics(cid), model, unproject_to_floor, ZONE_HEIGHT_M, MAX_RADIUS_PX)
        E = fields[cid]["e_" + TERM]
        ok = np.isfinite(E)
        inside = ok & (fields[cid]["X"].reshape(E.shape) >= 0) & (fields[cid]["X"].reshape(E.shape) <= real_w) \
            & (fields[cid]["Y"].reshape(E.shape) >= 0) & (fields[cid]["Y"].reshape(E.shape) <= real_h)
        print(f"  {cid}: e の範囲(牛舎内) {np.nanmin(E[inside]):.3f}〜{np.nanmax(E[inside]):.3f} m / 中央値 {np.nanmedian(E[inside]):.3f} m")

    out_dir = Path(ym.OUTPUT_DIR) / "he_kai_heatmap_ver2"
    out_path = out_dir / f"heatmap_{TERM}_eps{EPS_M:g}.png"
    plot_fields(fields, real_w, real_h, EPS_M, KS, TERM, VMAX, out_path)
    print(f"保存: {out_path}")


if __name__ == "__main__":
    main()