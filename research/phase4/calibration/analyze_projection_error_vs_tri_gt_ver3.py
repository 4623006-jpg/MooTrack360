"""
analyze_projection_error_vs_tri_gt_ver3.py

ver2 からの変更: 姿勢が未設定の牛(三角測量ツールで姿勢を選んでいない牛。ほとんどがこれ)は、
  三角測量した背中の高さ(result.height_m)から推定する。POSE_HEIGHT_THR_M より低ければ lying、高ければ standing。
  (背中の高さは、standing で 1.2〜1.5 m、lying で 0.4〜0.8 m 程度)。姿勢の内訳は実行時に表示する。

ver2 の変更点(ver1 からの変更):
  - 姿勢: XMLのposeは "Unspecified" なので使えない。三角測量ツールで付けた姿勢(standing/lying)を使う。
  - [追加] カメラ間の食い違い(同じ牛の、2カメラのbbox中心投影の差)から σ_px, σ_h を最尤推定する。
    マッチングのコストが実際に見ているのはこの食い違いなので、不確かさモデルの較正にはこちらを使う。
    推定後に、マハラノビス距離²が χ²(2) の95%/99%点に収まる割合(較正が合っていれば95%/99%)と、
    ユークリッド距離1.5 m以内の割合を比べる。

HE(改)の手順1: 三角測量GTを正解として、現在の方法(GTのbbox中心を、姿勢から想定した高さで床に投影)
の位置誤差の構造を測り、不確かさモデルのσを決める。

  サンプル : (牛, カメラ) ごと。三角測量GT(レベル1, 交差角が十分, 残差が小さい)の位置 P_gt を正解とし、
             その牛のそのカメラでの bbox中心 を 想定高さ h_a で床に投影した位置 P_est との差 e = P_est - P_gt。
             bbox は、三角測量GTのクリックが入っているXMLのbboxを使う(名前の対応は不要)。
  分解     : カメラから見た向きで e を 視線方向(半径方向) e_r と 垂直方向 e_t に分ける。
             h_eff = h_a + e_r (H-h_a)/ρ : bbox中心の光線が正解位置を通る高さ(想定高さ h_a との差が高さ仮定の誤差)
  モデル   : Var(e_t) = σ_px² · vᵀ J Jᵀ v                      (J = 画素→床の2×2ヤコビアン[m/px])
             Var(e_r) = σ_px² · uᵀ J Jᵀ u + σ_h² · g²,  g = ρ/(H-h_a)   (g = 高さ1mの違いで半径方向にずれる量)
             → σ_px[px], σ_h[m](姿勢別)を推定する。バイアス(平均のずれ)は別に報告する。
  注意     : 正解はクリックした背中の点で、bbox中心とは別の点。そのずれ(体の大きさ分)も σ_px に含まれる
             (マッチングで実際に効くのは、この「実効」の誤差のため、そのまま使ってよい)。
             YOLOのbboxはGTのbboxよりぶれるので、YOLOでのσ_pxは、これより大きい可能性がある。

出力: bev_calibration_refined/projection_error_vs_tri_gt_ver1.csv (サンプルごと)

使い方: python analyze_projection_error_vs_tri_gt_ver1.py [フレーム ...]   (省略時は保存済みの全フレーム)
配置: research/phase4/calibration/
"""

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# 配置: research/phase4/calibration/(このファイル), research/phase4/web_annotate/, research/ 直下に他のフォルダ
RESEARCH_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
PHASE4_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
MOOTRACK_CODE_DIR = (os.path.abspath(os.path.join(RESEARCH_DIR, "calibration")))
for d in (RESEARCH_DIR, PHASE4_DIR, os.path.join(PHASE4_DIR, "web_annotate"),
          os.path.join(PHASE4_DIR, "calibration_try"),
          MOOTRACK_CODE_DIR,
          os.path.join(RESEARCH_DIR, "analyze"), os.path.join(RESEARCH_DIR, "annotate"),
          os.path.join(RESEARCH_DIR, "phase3_multi_camera"), THIS_DIR):
    if os.path.isdir(d) and d not in sys.path:
        sys.path.insert(0, d)

# phase3_bev_multicam_yolo_check_zonepose_ver3.py と同じ設定
XML_DIR = "/mnt/vmlqnap02/dataset/MooTrack360/1-hour-farm1-area1-sample-sequence/xml-1fps"
IMAGE_WIDTH_PX = 8832
LYING_HEIGHT_M = 0.5
STANDING_HEIGHT_M = 0.75
DEFAULT_HEIGHT_M = 0.8

# 使う三角測量GT(信頼できるもの)
POSE_HEIGHT_THR_M = 0.95     # 三角測量した背中の高さがこれ未満なら lying
MIN_ANGLE_DEG = 45.0
MAX_RESID_M = 0.30
DIST_BINS_M = [0, 5, 10, 15, 100]
N_CAMS = 3


def height_for(pose):
    return {"lying": LYING_HEIGHT_M, "standing": STANDING_HEIGHT_M}.get(pose, DEFAULT_HEIGHT_M)


def parse_xml(path):
    import xml.etree.ElementTree as ET
    objs = []
    for obj in ET.parse(path).getroot().findall("object"):
        bnd = obj.find("bndbox")
        pe = obj.find("pose")
        objs.append({"name": obj.find("name").text,
                     "pose": pe.text.strip().lower() if pe is not None and pe.text else None,
                     "box": [float(bnd.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")]})
    return objs


def ground_jacobian(cam_pose, intr, uv, h_a):
    """画素(u,v)を高さh_aの平面に投影した床位置の、画素に対する2×2ヤコビアン[m/px]"""
    from Mootrack_core import unproject_to_floor
    pts = np.array([uv, [uv[0] + 1.0, uv[1]], [uv[0], uv[1] + 1.0]], dtype=np.float64)
    xy, ok = unproject_to_floor(pts, cam_pose.R, cam_pose.t, *intr, floor_z=-h_a)
    if not ok.all():
        return None, None
    return np.stack([xy[1] - xy[0], xy[2] - xy[0]], axis=1), xy[0]


def build_samples(tri_cows, xml_objs, poses, get_intrinsics, cam_ids):
    """tri_cows: 三角測量GTのjsonの cows。戻り値: サンプルのリスト"""
    tile_w = IMAGE_WIDTH_PX / N_CAMS
    out, n_skip = [], {"低品質": 0, "bbox無し": 0}
    for cow in tri_cows:
        r = cow.get("result", {})
        if r.get("level") != 1 or r.get("X") is None:
            n_skip["低品質"] += 1
            continue
        if r.get("max_angle_deg", 0) < MIN_ANGLE_DEG or r.get("resid_m", 9) > MAX_RESID_M:
            n_skip["低品質"] += 1
            continue
        P_gt = np.array(r["X"][:2], dtype=np.float64)
        for k, cid in enumerate(cam_ids):
            uv = cow.get("clicks", {}).get(cid)
            if uv is None:
                continue
            x_full = uv[0] + k * tile_w
            hit = [o for o in xml_objs if o["box"][0] <= x_full <= o["box"][2] and o["box"][1] <= uv[1] <= o["box"][3]]
            if not hit:
                n_skip["bbox無し"] += 1
                continue
            o = min(hit, key=lambda o: (o["box"][2] - o["box"][0]) * (o["box"][3] - o["box"][1]))
            pose = o["pose"] if o["pose"] in ("standing", "lying") else (cow.get("pose") or None)
            if pose not in ("standing", "lying"):
                hb = r.get("height_m")
                pose = None if hb is None else ("lying" if hb < POSE_HEIGHT_THR_M else "standing")
                n_skip["姿勢を高さから推定"] = n_skip.get("姿勢を高さから推定", 0) + 1
            h_a = height_for(pose)
            cx = (o["box"][0] + o["box"][2]) / 2.0 - k * tile_w
            cy = (o["box"][1] + o["box"][3]) / 2.0
            J, P_est = ground_jacobian(poses[cid], get_intrinsics(cid), (cx, cy), h_a)
            if J is None:
                continue
            out.append(sample_from_geometry(cid, pose, h_a, P_est, P_gt, poses[cid].t, J, cow.get("name", ""), cow["id"]))
    return out, n_skip


def sample_from_geometry(cid, pose, h_a, P_est, P_gt, cam_t, J, name="", cow_id=-1):
    cam_xy = np.asarray(cam_t[:2], dtype=np.float64)
    rho_gt = float(np.linalg.norm(P_gt - cam_xy))
    u = (P_gt - cam_xy) / rho_gt
    v = np.array([-u[1], u[0]])
    e = P_est - P_gt
    H = float(-cam_t[2])
    rho_est = float(np.linalg.norm(P_est - cam_xy))
    JJt = J @ J.T
    u_est = (P_est - cam_xy) / rho_est
    return dict(cam=cid, pose=pose or "unknown", h_a=h_a, cow=cow_id, name=name, rho=rho_gt, H=H,
                P_est=P_est, JJt=JJt, gvec=(rho_est / (H - h_a)) * u_est,
                e_r=float(e @ u), e_t=float(e @ v), g=rho_est / (H - h_a), h_eff=h_a + float(e @ u) * (H - h_a) / rho_est,
                a_r=float(u @ JJt @ u), a_t=float(v @ JJt @ v))


def fit_sigmas(samples, remove_bias=False):
    """σ_px, σ_h を推定。remove_bias=True なら、平均のずれ(系統誤差)を引いたあとのばらつきから推定"""
    if not samples:
        return None
    e_t = np.array([s["e_t"] for s in samples])
    e_r = np.array([s["e_r"] for s in samples])
    a_t = np.array([s["a_t"] for s in samples])
    a_r = np.array([s["a_r"] for s in samples])
    g = np.array([s["g"] for s in samples])
    if remove_bias:
        e_t, e_r = e_t - e_t.mean(), e_r - e_r.mean()
    sig_px2 = float(np.mean(e_t ** 2 / a_t))
    sig_h2 = float(np.mean(np.maximum(e_r ** 2 - sig_px2 * a_r, 0.0) / g ** 2))
    return dict(sigma_px=float(np.sqrt(sig_px2)), sigma_h=float(np.sqrt(sig_h2)))


def _group_pairs(samples):
    groups = {}
    for s in samples:
        groups.setdefault((s.get("frame", 0), s["cow"]), []).append(s)
    pairs = []
    for g in groups.values():
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                pairs.append((g[i], g[j]))
    return pairs


def _pair_cov(si, sj, sig_px, sig_h):
    S = np.zeros((2, 2))
    for s in (si, sj):
        S += sig_px ** 2 * s["JJt"] + sig_h.get(s["pose"], sig_h["_default"]) ** 2 * np.outer(s["gvec"], s["gvec"])
    return S


def fit_from_pairs(samples):
    """同じ牛の2カメラの食い違い d = P_est_i - P_est_j が N(0, Σ_i+Σ_j) に従うとして σ_px と姿勢別σ_hを最尤推定"""
    from scipy.optimize import minimize
    pairs = _group_pairs(samples)
    if len(pairs) < 10:
        return None
    poses = sorted({s["pose"] for s in samples})
    D = [pi["P_est"] - pj["P_est"] for pi, pj in pairs]

    def unpack(th):
        sig_h = {p: float(np.exp(th[1 + k])) for k, p in enumerate(poses)}
        sig_h["_default"] = float(np.mean(list(sig_h.values())))
        return float(np.exp(th[0])), sig_h

    def nll(th):
        sig_px, sig_h = unpack(th)
        tot = 0.0
        for (pi, pj), d in zip(pairs, D):
            S = _pair_cov(pi, pj, sig_px, sig_h)
            tot += 0.5 * (d @ np.linalg.solve(S, d) + np.log(np.linalg.det(S)))
        return tot

    th0 = np.log([10.0] + [0.2] * len(poses))
    res = minimize(nll, th0, method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-4, maxiter=600))
    sig_px, sig_h = unpack(res.x)
    m2 = np.array([d @ np.linalg.solve(_pair_cov(pi, pj, sig_px, sig_h), d) for (pi, pj), d in zip(pairs, D)])
    dist = np.array([np.linalg.norm(d) for d in D])
    return dict(sigma_px=sig_px, sigma_h={k: v for k, v in sig_h.items() if k != "_default"}, n_pairs=len(pairs),
                frac95=float(np.mean(m2 <= 5.991)), frac99=float(np.mean(m2 <= 9.210)), frac_euclid=float(np.mean(dist <= 1.5)),
                med_m2=float(np.median(m2)), dist_med=float(np.median(dist)), dist_p90=float(np.percentile(dist, 90)),
                frac99_euclid_equiv=float(np.mean(dist <= 1.5)))


def summarize(samples):
    print(f"サンプル数(牛×カメラ): {len(samples)}")
    poses_present = sorted({s["pose"] for s in samples})
    print(f"\n{'姿勢':<10}{'距離[m]':<10}{'n':>5}{'e_r平均':>9}{'e_r標準偏差':>11}{'e_t平均':>9}{'e_t標準偏差':>11}{'h_eff平均[m]':>12}{'h_eff標準偏差':>13}")
    for pose in poses_present + ["全体"]:
        for lo, hi in zip(DIST_BINS_M[:-1], DIST_BINS_M[1:]):
            sel = [s for s in samples if (pose == "全体" or s["pose"] == pose) and lo <= s["rho"] < hi]
            if len(sel) < 3:
                continue
            er, et = np.array([s["e_r"] for s in sel]), np.array([s["e_t"] for s in sel])
            he = np.array([s["h_eff"] for s in sel])
            print(f"{pose:<10}{f'{lo}-{hi}':<10}{len(sel):>5}{er.mean():>9.2f}{er.std():>11.2f}{et.mean():>9.2f}{et.std():>11.2f}"
                  f"{he.mean():>12.2f}{he.std():>13.2f}")
    print("\n--- σの推定 (e_r, e_t は m) ---")
    print(f"{'対象':<14}{'n':>5}{'σ_px[px]':>10}{'σ_h[m]':>9}   (バイアスを引いた場合: σ_px / σ_h)")
    for pose in poses_present + ["全体"]:
        sel = [s for s in samples if pose == "全体" or s["pose"] == pose]
        a, b = fit_sigmas(sel), fit_sigmas(sel, remove_bias=True)
        if a:
            print(f"{pose:<14}{len(sel):>5}{a['sigma_px']:>10.2f}{a['sigma_h']:>9.3f}   ({b['sigma_px']:.2f} / {b['sigma_h']:.3f})")
    print("\nカメラ別 (全姿勢):")
    for cid in sorted({s["cam"] for s in samples}):
        sel = [s for s in samples if s["cam"] == cid]
        a = fit_sigmas(sel)
        er, et = np.array([s["e_r"] for s in sel]), np.array([s["e_t"] for s in sel])
        print(f"  {cid}: n={len(sel)} σ_px={a['sigma_px']:.2f}px σ_h={a['sigma_h']:.3f}m / 平均のずれ e_r={er.mean():+.2f}m e_t={et.mean():+.2f}m")


def main():
    import phase3_bev_multicam_gt_zonepose_ver1 as g
    import Mootrack_core as mc
    from Mootrack_core import ROT_FMT, apply_camera_overrides, get_intrinsics, load_floorplan_cameras, resolve_poses

    in_dir = g.OUTPUT_DIR / "bev_gt_triangulated"
    out_dir = g.OUTPUT_DIR / "bev_calibration_refined"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = [int(a) for a in sys.argv[1:]] or sorted(int(p.name[:6]) for p in in_dir.glob("*_tri_gt_ver1.json"))
    cam_ids = list(g.CAM_IDS)
    poses = resolve_poses(apply_camera_overrides(load_floorplan_cameras(g.FLOORPLAN_JSON_PATH)), root="Area1", fmt=ROT_FMT)
    print(f"校正セット: {getattr(mc, 'CALIBRATION_SET', 'Mootrack_core.pyの現在値')} / フレーム {frames}")

    samples = []
    for fr in frames:
        xp = Path(XML_DIR) / f"{fr:06d}.xml"
        tp = in_dir / f"{fr:06d}_tri_gt_ver1.json"
        if not xp.exists() or not tp.exists():
            print(f"見つかりません: {xp if not xp.exists() else tp}")
            continue
        s, skip = build_samples(json.loads(tp.read_text(encoding="utf-8"))["cows"], parse_xml(xp), poses, get_intrinsics, cam_ids)
        for x in s:
            x["frame"] = fr
        samples += s
        print(f"  フレーム {fr}: サンプル {len(s)}件 (除外: {skip})")
    if len(samples) < 10:
        raise SystemExit("サンプルが少なすぎます")
    summarize(samples)

    pf = fit_from_pairs(samples)
    print("\n=== カメラ間の食い違いから推定 (マッチングが実際に見る量) ===")
    if pf is None:
        print("  2カメラ以上に写る牛のペアが足りません")
    else:
        print(f"  ペア数 {pf['n_pairs']} / 食い違い距離: 中央値 {pf['dist_med']:.2f} m, 90%点 {pf['dist_p90']:.2f} m")
        print(f"  σ_px = {pf['sigma_px']:.1f} px / σ_h = " + ", ".join(f"{k}: {v:.3f} m" for k, v in pf["sigma_h"].items()))
        print(f"  較正の確認: マハラノビス距離² <= χ²(2)の95%点(5.99) の割合 {pf['frac95']:.0%} (理想95%), 99%点(9.21) {pf['frac99']:.0%} (理想99%)")
        print(f"  比較: ユークリッド距離 <= 1.5 m の割合 {pf['frac_euclid']:.0%}")

    a = fit_sigmas(samples)
    sp = {p: fit_sigmas([s for s in samples if s["pose"] == p]) for p in sorted({s["pose"] for s in samples})}
    print("\n=== (参考) 正解(背中の点)との差から推定した値 ===")
    print(f"SIGMA_PX_EFF = {a['sigma_px']:.2f}   # 画素の実効誤差[px] (bbox中心 vs 背中の点のずれ含む)")
    print("SIGMA_H_BY_POSE = {" + ", ".join(f'"{p}": {v["sigma_h"]:.3f}' for p, v in sp.items() if v) + "}   # 高さ仮定の誤差[m]")

    with open(out_dir / "projection_error_vs_tri_gt_ver1.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        keys = ["frame", "cow", "name", "cam", "pose", "h_a", "rho", "e_r", "e_t", "h_eff", "g", "a_r", "a_t"]
        w.writerow(keys)
        for s in samples:
            w.writerow([f"{s[k]:.4f}" if isinstance(s[k], float) else s[k] for k in keys])
    print(f"\nCSV: {out_dir / 'projection_error_vs_tri_gt_ver1.csv'}")


if __name__ == "__main__":
    main()