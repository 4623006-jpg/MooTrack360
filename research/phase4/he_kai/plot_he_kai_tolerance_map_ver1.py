"""
plot_he_kai_tolerance_map_ver1.py

HE(改)の許容範囲のヒートマップ(BEV上)。牛舎の床の各地点に牛がいたときの、カメラごとの位置の不確かさと、
隣接カメラ間で同じ牛とみなすときの許容半径(マッチングの許容範囲)を、色で表す。

  位置の不確かさ Σ(カメラc, 地点P, 姿勢) = σ_px² · J Jᵀ + σ_h(姿勢)² · g gᵀ
      J : 画素 -> 床 のヤコビアン [m/px]    g = ρ/(H-h_a) · (カメラから見た向き)  (高さ1mの違いで半径方向にずれる量)
  図1 (he_kai_uncertainty_single_ver1.png): カメラごと・姿勢ごとの 1σ(長軸) [m]
  図2 (he_kai_tolerance_pair_ver1.png)    : 隣接カメラペアの許容半径 = sqrt(χ²·λmax(Σ_i+Σ_j)) [m]
      左: 許容半径[m]  右: 許容半径 / 現在の固定閾値(1.5 m)  (>1: ユークリッド1.5 mでは取りこぼす / <1: 緩すぎて誤統合のリスク)
      重ね描き: 三角測量GTの牛で、実際に観測した2カメラの食い違い[m] (許容半径と同じ色スケール)
  いずれも 牛舎の外・カメラから見えない場所は白。

出力: bev_calibration_refined/he_kai_*.png
配置: research/phase4/he_kai/ (analyze_projection_error_vs_tri_gt_ver3.py は research/phase4/calibration/ にあること)
使い方: python plot_he_kai_tolerance_map_ver1.py [フレーム ...]   (フレームは重ね描きに使う三角測量GT。省略時は100)
校正は Mootrack_core.py の現在値。
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# 配置: research/phase4/he_kai/(このファイル) など。research 直下に他のフォルダ
RESEARCH_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
PHASE4_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
MOOTRACK_CODE_DIR = (os.path.abspath(os.path.join(RESEARCH_DIR, "calibration")))
for d in (RESEARCH_DIR, PHASE4_DIR, MOOTRACK_CODE_DIR, os.path.join(PHASE4_DIR, "calibration"),
          os.path.join(RESEARCH_DIR, "analyze"), os.path.join(RESEARCH_DIR, "annotate"),
          os.path.join(RESEARCH_DIR, "phase3_multi_camera"), THIS_DIR):
    if os.path.isdir(d) and d not in sys.path:
        sys.path.insert(0, d)

SIGMA_PX = 16.2
SIGMA_H = {"lying": 0.27, "standing": 0.455}
H_ASSUMED = {"lying": 0.5, "standing": 0.75}
GATE_CHI2 = 9.21
EUCLID_THR_M = 1.5
TILE_PX = 2944
STRIDE_PX = 4
CELL_M = 0.25
RHO_MAX_M = 40.0
FILL_MAX_CELLS = 8      # 遠方の穴(画素間隔が床で粗くなる)を埋める最大距離[格子]


def covariance_cells(pose_obj, intr, h_a, sig_px, sig_h, real_w, real_h, cell=CELL_M, stride=STRIDE_PX):
    """BEV格子(cell[m])ごとの Σ の成分 (xx, xy, yy)。カメラから見えない・牛舎の外はnan。戻り値: dict"""
    from Mootrack_core import unproject_to_floor
    us = np.arange(0, TILE_PX, stride, dtype=np.float64)
    vs = np.arange(0, TILE_PX, stride, dtype=np.float64)
    U, V = np.meshgrid(us, vs)
    pts = np.stack([U.ravel(), V.ravel()], axis=1)
    xy, ok = unproject_to_floor(pts, pose_obj.R, pose_obj.t, *intr, floor_z=-h_a)
    P = xy.reshape(U.shape + (2,))
    ok = ok.reshape(U.shape)
    cam_xy = np.asarray(pose_obj.t[:2], dtype=np.float64)
    Hc = float(-pose_obj.t[2])
    Pu = np.gradient(P, stride, axis=1)           # dP/du
    Pv = np.gradient(P, stride, axis=0)           # dP/dv
    JJt_xx = Pu[..., 0] ** 2 + Pv[..., 0] ** 2
    JJt_xy = Pu[..., 0] * Pu[..., 1] + Pv[..., 0] * Pv[..., 1]
    JJt_yy = Pu[..., 1] ** 2 + Pv[..., 1] ** 2
    d = P - cam_xy
    rho = np.linalg.norm(d, axis=-1)
    u = d / np.maximum(rho, 1e-9)[..., None]
    gk = rho / max(Hc - h_a, 0.3)
    gx, gy = gk * u[..., 0], gk * u[..., 1]
    Sxx = sig_px ** 2 * JJt_xx + sig_h ** 2 * gx * gx
    Sxy = sig_px ** 2 * JJt_xy + sig_h ** 2 * gx * gy
    Syy = sig_px ** 2 * JJt_yy + sig_h ** 2 * gy * gy
    # 有効な画素: 逆投影が有効、牛舎の内側、距離が妥当、ヤコビアンが有限
    good = (ok & (rho < RHO_MAX_M) & np.isfinite(Sxx) & (P[..., 0] >= 0) & (P[..., 0] <= real_w)
            & (P[..., 1] >= 0) & (P[..., 1] <= real_h))
    nx, ny = int(np.ceil(real_w / cell)), int(np.ceil(real_h / cell))
    ix = np.clip((P[..., 0] / cell).astype(int), 0, nx - 1)
    iy = np.clip((P[..., 1] / cell).astype(int), 0, ny - 1)
    out = {}
    cnt = np.zeros((ny, nx))
    np.add.at(cnt, (iy[good], ix[good]), 1.0)
    from scipy import ndimage
    hit = cnt > 0
    # 遠方は画素の間隔が床で粗くなり、格子に穴(縞)が空く。近い有効格子の値で、近傍(FILL_MAX_CELLS以内)の穴を埋める
    dist, (ri, ci) = ndimage.distance_transform_edt(~hit, return_indices=True)
    fill = (~hit) & (dist <= FILL_MAX_CELLS)
    for name, arr in (("xx", Sxx), ("xy", Sxy), ("yy", Syy)):
        acc = np.zeros((ny, nx))
        np.add.at(acc, (iy[good], ix[good]), arr[good])
        m = np.where(hit, acc / np.maximum(cnt, 1), np.nan)
        m[fill] = m[ri[fill], ci[fill]]
        out[name] = m
    return out


def lam_max(Sxx, Sxy, Syy):
    tr, det = Sxx + Syy, Sxx * Syy - Sxy ** 2
    return tr / 2 + np.sqrt(np.maximum(tr ** 2 / 4 - det, 0.0))


def draw_outline(ax, regions, cam_xy_by_id, real_w, real_h):
    for r in regions:
        poly = np.vstack([r.polygon_m, r.polygon_m[:1]])
        ax.plot(poly[:, 0], poly[:, 1], color="0.55", lw=0.5)
    for cid, xy in cam_xy_by_id.items():
        ax.plot(*xy, marker="^", color="k", ms=7)
        ax.annotate(cid.replace("Camera_", "C"), xy, xytext=(3, 3), textcoords="offset points", fontsize=8)
    ax.set_xlim(0, real_w)
    ax.set_ylim(real_h, 0)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import phase3_bev_multicam_gt_zonepose_ver1 as g
    import Mootrack_core as mc
    from Mootrack_core import ROT_FMT, apply_camera_overrides, get_intrinsics, load_floorplan, resolve_poses

    out_dir = g.OUTPUT_DIR / "bev_calibration_refined"
    out_dir.mkdir(parents=True, exist_ok=True)
    regions, cameras_json, real_w, real_h = load_floorplan(g.FLOORPLAN_JSON_PATH)
    poses = resolve_poses(apply_camera_overrides(cameras_json), root="Area1", fmt=ROT_FMT)
    cam_ids = list(g.CAM_IDS)
    cam_xy = {c: poses[c].t[:2] for c in cam_ids}
    calib = getattr(mc, "CALIBRATION_SET", "Mootrack_core現在値")
    print(f"校正セット: {calib} / σ_px={SIGMA_PX}px σ_h={SIGMA_H} / χ²={GATE_CHI2} / 床 {real_w}x{real_h} m")

    cells = {(c, p): covariance_cells(poses[c], get_intrinsics(c), H_ASSUMED[p], SIGMA_PX, SIGMA_H[p], real_w, real_h)
             for c in cam_ids for p in ("standing", "lying")}

    # ---- 図1: カメラごと・姿勢ごとの 1σ(長軸) ----
    fig, axes = plt.subplots(2, len(cam_ids), figsize=(6.2 * len(cam_ids), 4.0 * 2))
    for r, p in enumerate(("standing", "lying")):
        for k, c in enumerate(cam_ids):
            S = cells[(c, p)]
            sigma = np.sqrt(lam_max(S["xx"], S["xy"], S["yy"]))
            ax = axes[r, k]
            im = ax.imshow(sigma, origin="upper", extent=[0, real_w, real_h, 0], cmap="viridis_r", vmin=0, vmax=2.0)
            draw_outline(ax, regions, cam_xy, real_w, real_h)
            ax.set_title(f"{c} / {p}: 1-sigma major axis [m]")
            fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.suptitle(f"Position uncertainty of a detection (HE-kai), calibration={calib}", y=0.995)
    fig.tight_layout()
    p1 = out_dir / "he_kai_uncertainty_single_ver1.png"
    fig.savefig(p1, dpi=110)
    plt.close(fig)

    # ---- 重ね描き用: 三角測量GTの牛の、2カメラの食い違い ----
    frames = [int(a) for a in sys.argv[1:]] or [100]
    overlay = []
    try:
        import analyze_projection_error_vs_tri_gt_ver3 as an
        for fr in frames:
            xp, tp = Path(an.XML_DIR) / f"{fr:06d}.xml", g.OUTPUT_DIR / "bev_gt_triangulated" / f"{fr:06d}_tri_gt_ver1.json"
            if xp.exists() and tp.exists():
                s, _ = an.build_samples(json.loads(tp.read_text(encoding="utf-8"))["cows"], an.parse_xml(xp), poses,
                                        get_intrinsics, cam_ids)
                for x in s:
                    x["frame"] = fr
                for si, sj in an._group_pairs(s):
                    overlay.append((0.5 * (si["P_est"] + sj["P_est"]), float(np.linalg.norm(si["P_est"] - sj["P_est"])), si["pose"]))
    except Exception as e:
        print(f"重ね描きをスキップ: {e}")

    # ---- 図2: 隣接ペアの許容半径 ----
    pairs = [(cam_ids[i], cam_ids[i + 1]) for i in range(len(cam_ids) - 1)]
    fig, axes = plt.subplots(len(pairs), 2, figsize=(13, 4.3 * len(pairs)))
    for r, (ci, cj) in enumerate(pairs):
        Si, Sj = cells[(ci, "standing")], cells[(cj, "standing")]
        tol = np.sqrt(GATE_CHI2 * lam_max(Si["xx"] + Sj["xx"], Si["xy"] + Sj["xy"], Si["yy"] + Sj["yy"]))
        ax = axes[r, 0]
        im = ax.imshow(tol, origin="upper", extent=[0, real_w, real_h, 0], cmap="viridis_r", vmin=0, vmax=6.0)
        draw_outline(ax, regions, cam_xy, real_w, real_h)
        if overlay:
            pts = np.array([o[0] for o in overlay])
            ax.scatter(pts[:, 0], pts[:, 1], c=[o[1] for o in overlay], cmap="viridis_r", vmin=0, vmax=6.0,
                       s=26, edgecolors="r", linewidths=0.8, zorder=5)
        ax.set_title(f"{ci}-{cj} (standing): matching tolerance radius [m]  (dots: observed disagreement)")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        ax = axes[r, 1]
        ratio = tol / EUCLID_THR_M
        im = ax.imshow(ratio, origin="upper", extent=[0, real_w, real_h, 0], cmap="RdBu_r", vmin=0, vmax=2.0)
        draw_outline(ax, regions, cam_xy, real_w, real_h)
        ax.set_title(f"{ci}-{cj}: tolerance / {EUCLID_THR_M} m   (>1: fixed gate too strict, <1: too loose)")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        v = ratio[np.isfinite(ratio)]
        if len(v):
            print(f"{ci}-{cj}: 両方から見える格子 {len(v)}個 / 許容半径 中央値 {np.nanmedian(tol):.2f} m / "
                  f"1.5 m より広い格子の割合 {np.mean(v > 1):.0%} / 狭い格子の割合 {np.mean(v < 1):.0%}")
    fig.suptitle(f"Matching tolerance (chi2={GATE_CHI2}), calibration={calib}", y=0.995)
    fig.tight_layout()
    p2 = out_dir / "he_kai_tolerance_pair_ver1.png"
    fig.savefig(p2, dpi=110)
    plt.close(fig)
    print(f"保存: {p1}\n      {p2}")
    if overlay:
        d = np.array([o[1] for o in overlay])
        print(f"重ね描き: 観測した2カメラの食い違い {len(d)}ペア / 中央値 {np.median(d):.2f} m / 90%点 {np.percentile(d, 90):.2f} m")


if __name__ == "__main__":
    main()