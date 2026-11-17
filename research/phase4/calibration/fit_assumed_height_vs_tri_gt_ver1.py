"""
fit_assumed_height_vs_tri_gt_ver1.py

姿勢別の「bbox中心を床に投影するときの想定高さ」を、三角測量GTで最適化する。
(以前の高さのグリッドサーチは、手作業のBEV GT=おおまかな位置で行ったので、三角測量GTでやり直す)

  サンプル : (牛, カメラ)。三角測量GT(レベル1, 交差角>=45°, 残差<=0.3 m)の位置 P_gt を正解とし、
             その牛のそのカメラのbbox中心を、想定高さhで床に投影した位置 P_est(h) を比べる。
             bboxは、三角測量GTのクリックが入っているXMLのbbox。姿勢はツールで入力した姿勢(無ければ背中の高さから推定)。
  評価     : 姿勢ごとに h を 0.30〜1.60 m で振って、
               - RMS誤差 = sqrt(mean |P_est(h) - P_gt|²)    (正解との差)
               - 食い違い = 同じ牛の2カメラの P_est(h) の距離の 中央値 / 90%点 / 1.5 m以内の割合 (マッチングが見る量)
             を出し、それぞれの最適な h と、牛を復元抽出(ブートストラップ)した95%区間を出す。
  追加     : h を距離の一次式 h(ρ) = h0 + h1·ρ にした場合のRMS誤差(距離とともに増えるずれが、これで消えるか)。
  現在の想定高さ(standing 0.75 / lying 0.5)での値を基準として併記する。

出力: bev_calibration_refined/fit_assumed_height_ver1.csv
配置: research/phase4/calibration/ (analyze_projection_error_vs_tri_gt_ver3.py と同じ場所)
使い方: python fit_assumed_height_vs_tri_gt_ver1.py [フレーム ...]   (省略時は保存済みの全フレーム)
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

H_GRID = np.round(np.arange(0.30, 1.601, 0.05), 3)
H0_GRID = np.round(np.arange(0.30, 1.501, 0.10), 3)
H1_GRID = np.round(np.arange(-0.02, 0.0401, 0.005), 4)
CURRENT_H = {"standing": 0.75, "lying": 0.5}
N_BOOT = 500
SEED = 0


def build_raw(tri_cows, xml_objs, cam_ids, an, frame):
    """(牛,カメラ)ごとに、bbox中心の画素・正解位置・姿勢を返す"""
    tile_w = an.IMAGE_WIDTH_PX / an.N_CAMS
    raw = []
    for cow in tri_cows:
        r = cow.get("result", {})
        if r.get("level") != 1 or r.get("X") is None:
            continue
        if r.get("max_angle_deg", 0) < an.MIN_ANGLE_DEG or r.get("resid_m", 9) > an.MAX_RESID_M:
            continue
        P_gt = np.array(r["X"][:2], dtype=np.float64)
        for k, cid in enumerate(cam_ids):
            uv = cow.get("clicks", {}).get(cid)
            if uv is None:
                continue
            x_full = uv[0] + k * tile_w
            hit = [o for o in xml_objs if o["box"][0] <= x_full <= o["box"][2] and o["box"][1] <= uv[1] <= o["box"][3]]
            if not hit:
                continue
            o = min(hit, key=lambda o: (o["box"][2] - o["box"][0]) * (o["box"][3] - o["box"][1]))
            pose = o["pose"] if o["pose"] in ("standing", "lying") else (cow.get("pose") or None)
            if pose not in ("standing", "lying"):
                hb = r.get("height_m")
                pose = None if hb is None else ("lying" if hb < an.POSE_HEIGHT_THR_M else "standing")
            if pose is None:
                continue
            raw.append(dict(frame=frame, cow=cow["id"], cam=cid, pose=pose, P_gt=P_gt,
                            uv=((o["box"][0] + o["box"][2]) / 2.0 - k * tile_w, (o["box"][1] + o["box"][3]) / 2.0)))
    return raw


def project(raw, poses, get_intrinsics, h):
    """h: スカラー or 各サンプルの高さ配列。戻り値: P_est (n,2), 有効フラグ"""
    from Mootrack_core import unproject_to_floor
    n = len(raw)
    h = np.broadcast_to(np.asarray(h, dtype=np.float64), (n,))
    P, ok = np.zeros((n, 2)), np.zeros(n, dtype=bool)
    for cid in {s["cam"] for s in raw}:
        idx = np.array([i for i, s in enumerate(raw) if s["cam"] == cid])
        pts = np.array([raw[i]["uv"] for i in idx], dtype=np.float64)
        xy, v = unproject_to_floor(pts, poses[cid].R, poses[cid].t, *get_intrinsics(cid), floor_z=-h[idx])
        P[idx], ok[idx] = xy, v
    return P, ok


def group_pairs(raw):
    groups = {}
    for i, s in enumerate(raw):
        groups.setdefault((s["frame"], s["cow"]), []).append(i)
    return [(g[a], g[b]) for g in groups.values() for a in range(len(g)) for b in range(a + 1, len(g))]


def sweep(raw, poses, get_intrinsics, h_grid=H_GRID):
    """姿勢一定のサンプル集合 raw で h を振る。戻り値: rms, 食い違いの中央値/90%点/1.5m以内, 二乗誤差行列(n×H)"""
    Pgt = np.array([s["P_gt"] for s in raw])
    pairs = group_pairs(raw)
    rms, dmed, dp90, d15, sq = [], [], [], [], []
    for h in h_grid:
        P, ok = project(raw, poses, get_intrinsics, h)
        e2 = np.where(ok, np.sum((P - Pgt) ** 2, axis=1), np.nan)
        sq.append(e2)
        rms.append(float(np.sqrt(np.nanmean(e2))))
        if pairs:
            d = np.array([np.linalg.norm(P[a] - P[b]) if ok[a] and ok[b] else np.nan for a, b in pairs])
            d = d[~np.isnan(d)]
            dmed.append(float(np.median(d)))
            dp90.append(float(np.percentile(d, 90)))
            d15.append(float(np.mean(d <= 1.5)))
        else:
            dmed.append(np.nan); dp90.append(np.nan); d15.append(np.nan)
    return np.array(rms), np.array(dmed), np.array(dp90), np.array(d15), np.array(sq).T


def bootstrap_best_h(raw, sq, h_grid, rng, n_boot=N_BOOT):
    """牛を復元抽出して、RMS誤差が最小になる h の分布"""
    cows = sorted({(s["frame"], s["cow"]) for s in raw})
    idx_by_cow = {c: [i for i, s in enumerate(raw) if (s["frame"], s["cow"]) == c] for c in cows}
    best = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(cows), len(cows))
        ii = np.concatenate([idx_by_cow[cows[k]] for k in pick])
        best.append(h_grid[int(np.nanargmin(np.sqrt(np.nanmean(sq[ii], axis=0))))])
    return float(np.percentile(best, 2.5)), float(np.percentile(best, 97.5))


def fit_linear_in_distance(raw, poses, get_intrinsics):
    """h(ρ)=h0+h1·ρ (ρ=カメラから正解位置までの水平距離) のRMS誤差が最小の (h0,h1)"""
    Pgt = np.array([s["P_gt"] for s in raw])
    rho = np.array([np.linalg.norm(s["P_gt"] - poses[s["cam"]].t[:2]) for s in raw])
    best = (None, None, np.inf)
    for h0 in H0_GRID:
        for h1 in H1_GRID:
            h = np.clip(h0 + h1 * rho, 0.2, 2.0)
            P, ok = project(raw, poses, get_intrinsics, h)
            e = np.sqrt(np.nanmean(np.where(ok, np.sum((P - Pgt) ** 2, axis=1), np.nan)))
            if e < best[2]:
                best = (float(h0), float(h1), float(e))
    return best


def main():
    import phase3_bev_multicam_gt_zonepose_ver1 as g
    import Mootrack_core as mc
    import analyze_projection_error_vs_tri_gt_ver1 as an
    from Mootrack_core import ROT_FMT, apply_camera_overrides, get_intrinsics, load_floorplan_cameras, resolve_poses

    in_dir = g.OUTPUT_DIR / "bev_gt_triangulated"
    out_dir = g.OUTPUT_DIR / "bev_calibration_refined"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = [int(a) for a in sys.argv[1:]] or sorted(int(p.name[:6]) for p in in_dir.glob("*_tri_gt_ver1.json"))
    cam_ids = list(g.CAM_IDS)
    poses = resolve_poses(apply_camera_overrides(load_floorplan_cameras(g.FLOORPLAN_JSON_PATH)), root="Area1", fmt=ROT_FMT)
    print(f"校正セット: {getattr(mc, 'CALIBRATION_SET', 'Mootrack_core.pyの現在値')} / フレーム {frames}")

    raw = []
    for fr in frames:
        xp, tp = Path(an.XML_DIR) / f"{fr:06d}.xml", in_dir / f"{fr:06d}_tri_gt_ver1.json"
        if xp.exists() and tp.exists():
            raw += build_raw(json.loads(tp.read_text(encoding="utf-8"))["cows"], an.parse_xml(xp), cam_ids, an, fr)
        else:
            print(f"見つかりません: {xp if not xp.exists() else tp}")
    rng = np.random.default_rng(SEED)
    rows = []
    summary = {}
    for pose in ("standing", "lying"):
        sub = [s for s in raw if s["pose"] == pose]
        n_cows = len({(s["frame"], s["cow"]) for s in sub})
        if len(sub) < 8:
            print(f"\n[{pose}] サンプルが少ない({len(sub)}件)のでスキップ")
            continue
        rms, dmed, dp90, d15, sq = sweep(sub, poses, get_intrinsics)
        ib, im, i15 = int(np.argmin(rms)), int(np.nanargmin(dmed)), int(np.nanargmax(d15)) if np.isfinite(d15).any() else 0
        icur = int(np.argmin(np.abs(H_GRID - CURRENT_H[pose])))
        lo, hi = bootstrap_best_h(sub, sq, H_GRID, rng)
        print(f"\n=== {pose}: サンプル {len(sub)}件 / 牛 {n_cows}頭 / 2カメラの組 {len(group_pairs(sub))} ===")
        print(f"{'h[m]':>6}{'RMS誤差[m]':>12}{'食い違い中央値':>14}{'90%点':>8}{'<=1.5m':>8}")
        for k, h in enumerate(H_GRID):
            mark = " <- 現在" if k == icur else (" <- RMS最小" if k == ib else "")
            if k % 2 == 0 or k in (ib, icur, im):
                print(f"{h:>6.2f}{rms[k]:>12.3f}{dmed[k]:>14.3f}{dp90[k]:>8.2f}{d15[k]:>8.0%}{mark}")
        print(f"RMS誤差が最小の h = {H_GRID[ib]:.2f} m (95%区間 {lo:.2f}〜{hi:.2f} m) / RMS {rms[ib]:.3f} m (現在 {CURRENT_H[pose]} m では {rms[icur]:.3f} m)")
        print(f"食い違い(中央値)が最小の h = {H_GRID[im]:.2f} m / 中央値 {dmed[im]:.3f} m (現在: {dmed[icur]:.3f} m)")
        h0, h1, e = fit_linear_in_distance(sub, poses, get_intrinsics)
        print(f"距離の一次式 h(ρ)={h0:.2f}+{h1:.3f}·ρ にすると RMS {e:.3f} m (定数 h={H_GRID[ib]:.2f} では {rms[ib]:.3f} m)")
        summary[pose] = dict(best_h=float(H_GRID[ib]), ci=(lo, hi), rms_best=float(rms[ib]), rms_cur=float(rms[icur]), lin=(h0, h1, e))
        for k, h in enumerate(H_GRID):
            rows.append([pose, f"{h:.2f}", f"{rms[k]:.4f}", f"{dmed[k]:.4f}", f"{dp90[k]:.4f}", f"{d15[k]:.4f}"])

    print("\n=== 提案 (phase3の LYING_HEIGHT_M / STANDING_HEIGHT_M に入れる値) ===")
    for pose, s in summary.items():
        print(f"{pose}: {s['best_h']:.2f} m  (95%区間 {s['ci'][0]:.2f}〜{s['ci'][1]:.2f}, RMS {s['rms_cur']:.3f} -> {s['rms_best']:.3f} m)")
    print("※ 1フレームの牛のみ。区間が広い・牛の数が少ない場合は、別フレームの牛を足して確かめること。")
    with open(out_dir / "fit_assumed_height_ver1.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pose", "h_m", "rms_err_m", "disagree_median_m", "disagree_p90_m", "disagree_within_1p5m"])
        w.writerows(rows)
    print(f"CSV: {out_dir / 'fit_assumed_height_ver1.csv'}")


if __name__ == "__main__":
    main()