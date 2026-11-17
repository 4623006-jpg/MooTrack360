"""
tri_gt_position_error_he_kai_ver2.py

カメラの有効範囲を決めるための「位置の総合誤差」を、三角測量GTを真の位置として直接測る。
(マッチングで使う Σ(最尤推定, sigma_he_kai_ver1.json)とは別。マッチングは食い違いの相対的な広がり、
 こちらは検出単体が真の位置からどれだけずれるか)

  サンプル : (牛, カメラ)。三角測量GT(レベル1, 交差角>=45°, 残差<=0.3 m)の位置 P_gt を真の位置とし、
             その牛のそのカメラの bbox中心を 有効高さ h_eff(姿勢) で床に投影した位置 P_est との差 e = P_est - P_gt。
             (fit_assumed_height_vs_tri_gt_ver1.py の build_raw と同じサンプル。h_eff は固定値: 立位1.4 m / 横臥0.5 m。環境変数 H_STANDING / H_LYING で変更)
  集計     : 画像中心からの半径 r で分位数ビンに分け、ビンごとに
               |e| [m]                 の中央値と q分位点(既定 q=0.9)
               |e| / ‖J‖₂ [px相当]     の q分位点   (‖J‖₂ = 1px動かしたときの床上のずれの最大値 [m/px])
             を出す。「px相当」は、位置の総合誤差を画素の大きさに直した値で、
             有効範囲の判定  ‖J(r,方位)‖₂ × e_px(r) <= ε  (従来の m/px × 画素誤差 <= 許容誤差 と同じ形)  に使う。
             半径が大きいほど誤差が減ることは無い、として、ビンの値の累積最大(包絡)も保存する。
  追加     : 放射方向/円周方向の誤差の平均(偏り)と標準偏差も出す。

出力: <OUTPUT_DIR>/bev_calibration_refined/tri_gt_error_table_he_kai_ver2.json
      (phase3_he_kai_matching_ver3.py の maha_h_tri で使うときは HE_TRI_TABLE にこのパスを指定する)
使い方: uv run python ./phase4/he_kai/tri_gt_position_error_he_kai_ver2.py [フレーム ...]   (省略時は保存済みの全フレーム)
環境変数: N_BINS(既定5) / Q(既定0.9) / EPS_M(既定0.425=体幅0.85mの半分。結果の目安表示に使う) / H_STANDING(1.4) / H_LYING(0.5)
注意: 三角測量GTは同じクリックと同じ校正から作られているので、校正が全カメラで同じ向きにずれていれば見えない。
      フレームが少ない(100番のみ)と、分位点が粗くなる。YOLOの検出ノイズ(YOLOのbbox中心とGTのbbox中心の差)は入らない。
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
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

N_BINS = int(os.environ.get("N_BINS", 5))
Q = float(os.environ.get("Q", 0.9))
EPS_M = float(os.environ.get("EPS_M", 0.425))
TABLE_NAME = "tri_gt_error_table_he_kai_ver2.json"
H_STANDING = float(os.environ.get("H_STANDING", 1.08))   # 立位の牛の背中の高さ [m]
H_LYING = float(os.environ.get("H_LYING", 0.7))         # 横臥の牛の高さ [m]
H_BY_POSE = {"standing": H_STANDING, "lying": H_LYING}


def sample_stats(raw, poses, get_intrinsics, h_by_pose):
    """サンプルごとの (r, |e|, ‖J‖₂, 放射/円周の誤差, カメラ) を返す。投影できないサンプルは除く"""
    from Mootrack_core import unproject_to_floor
    n = len(raw)
    h = np.array([h_by_pose[s["pose"]] for s in raw], dtype=np.float64)
    P_gt = np.array([s["P_gt"] for s in raw], dtype=np.float64)
    r, e_abs, s_spec = np.zeros(n), np.zeros(n), np.zeros(n)
    e_rad, e_tan = np.zeros(n), np.zeros(n)
    ok = np.zeros(n, dtype=bool)
    for cid in sorted({s["cam"] for s in raw}):
        idx = np.array([i for i, s in enumerate(raw) if s["cam"] == cid])
        m = len(idx)
        uv = np.array([raw[i]["uv"] for i in idx], dtype=np.float64)
        fx, fy, cx, cy, xi, alpha = get_intrinsics(cid)
        cp = poses[cid]
        pts = np.concatenate([uv, uv + np.array([1.0, 0.0]), uv + np.array([0.0, 1.0])])
        xy, v = unproject_to_floor(pts, cp.R, cp.t, fx, fy, cx, cy, xi, alpha, floor_z=-np.tile(h[idx], 3))
        xy0, xy1, xy2 = xy[:m], xy[m:2 * m], xy[2 * m:]
        a, b = xy1 - xy0, xy2 - xy0
        jxx, jxy, jyy = (a ** 2).sum(1), (a * b).sum(1), (b ** 2).sum(1)      # J の列 a, b -> JᵀJ
        tr, det = jxx + jyy, jxx * jyy - jxy ** 2
        s_spec[idx] = np.sqrt(tr / 2.0 + np.sqrt(np.maximum(tr * tr / 4.0 - det, 0.0)))
        e = xy0 - P_gt[idx]
        e_abs[idx] = np.linalg.norm(e, axis=1)
        cam_xy = np.asarray(cp.t[:2], dtype=np.float64)
        dd = xy0 - cam_xy
        rho = np.maximum(np.linalg.norm(dd, axis=1), 1e-6)
        u = dd / rho[:, None]
        perp = np.stack([-u[:, 1], u[:, 0]], axis=1)
        e_rad[idx], e_tan[idx] = (e * u).sum(1), (e * perp).sum(1)
        r[idx] = np.hypot(uv[:, 0] - cx, uv[:, 1] - cy)
        ok[idx] = v[:m] & v[m:2 * m] & v[2 * m:]
    cams = np.array([s["cam"] for s in raw])
    return dict(r=r[ok], e=e_abs[ok], s=s_spec[ok], e_rad=e_rad[ok], e_tan=e_tan[ok], cam=cams[ok])


def make_table(st, n_bins=N_BINS, q=Q):
    """半径の分位数ビンごとの表。e_px = |e| / ‖J‖₂ の q分位点と、その累積最大(包絡)を含む"""
    r = st["r"]
    n_bins = int(max(1, min(n_bins, len(r) // 8)))
    edges = np.quantile(r, np.linspace(0.0, 1.0, n_bins + 1))
    edges[-1] += 1e-6
    rows = []
    for k in range(n_bins):
        sel = (r >= edges[k]) & (r < edges[k + 1])
        if not sel.any():
            continue
        e, s = st["e"][sel], st["s"][sel]
        rows.append(dict(
            r_lo=float(edges[k]), r_hi=float(edges[k + 1]), r_med=float(np.median(r[sel])), n=int(sel.sum()),
            e_med_m=float(np.median(e)), e_q_m=float(np.quantile(e, q)),
            e_px_q=float(np.quantile(e / np.maximum(s, 1e-9), q)),
            rad_mean=float(st["e_rad"][sel].mean()), rad_sd=float(st["e_rad"][sel].std()),
            tan_mean=float(st["e_tan"][sel].mean()), tan_sd=float(st["e_tan"][sel].std())))
    env_m = np.maximum.accumulate([x["e_q_m"] for x in rows])
    env_px = np.maximum.accumulate([x["e_px_q"] for x in rows])
    for x, a, b in zip(rows, env_m, env_px):
        x["e_q_m_env"], x["e_px_q_env"] = float(a), float(b)
    return rows


def table_to_json(rows, q, frames, h_eff, n_samples):
    return dict(q=q, frames=frames, h_eff=h_eff, n_samples=n_samples,
                r_knots=[x["r_med"] for x in rows],
                e_q_m=[x["e_q_m_env"] for x in rows], e_q_px=[x["e_px_q_env"] for x in rows],
                bins=rows)


def main():
    import phase3_bev_multicam_gt_zonepose_ver1 as g
    import Mootrack_core as mc
    import analyze_projection_error_vs_tri_gt_ver3 as an
    import fit_assumed_height_vs_tri_gt_ver1 as fh
    from he_kai_sigma_model_ver1 import POSES, load_model
    from Mootrack_core import ROT_FMT, apply_camera_overrides, get_intrinsics, load_floorplan_cameras, resolve_poses

    in_dir = g.OUTPUT_DIR / "bev_gt_triangulated"
    out_dir = g.OUTPUT_DIR / "bev_calibration_refined"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = [int(a) for a in sys.argv[1:]] or sorted(int(p.name[:6]) for p in in_dir.glob("*_tri_gt_ver1.json"))
    cam_ids = list(g.CAM_IDS)
    poses = resolve_poses(apply_camera_overrides(load_floorplan_cameras(g.FLOORPLAN_JSON_PATH)), root="Area1", fmt=ROT_FMT)
    # ver2: 想定高さは固定値(牛の背中=立位1.4 m / 横臥0.5 m)。σのJSONの h_eff は使わない
    h_eff = {p: H_BY_POSE[p] for p in POSES}
    print(f"校正セット: {getattr(mc, 'CALIBRATION_SET', 'Mootrack_core.pyの現在値')} / フレーム {frames} / h_eff={ {p: round(v, 3) for p, v in h_eff.items()} }")

    raw = []
    for fr in frames:
        xp, tp = Path(an.XML_DIR) / f"{fr:06d}.xml", in_dir / f"{fr:06d}_tri_gt_ver1.json"
        if xp.exists() and tp.exists():
            raw += fh.build_raw(json.loads(tp.read_text(encoding="utf-8"))["cows"], an.parse_xml(xp), cam_ids, an, fr)
        else:
            print(f"見つかりません: {xp if not xp.exists() else tp}")
    st = sample_stats(raw, poses, get_intrinsics, h_eff)
    n = len(st["r"])
    print(f"サンプル {len(raw)}件 -> 投影できた {n}件 / 牛 {len({(s['frame'], s['cow']) for s in raw})}頭")
    if n < 16:
        print("サンプルが少なすぎます")
        return

    print(f"\n【全体】|e| [m] (真の位置=三角測量GT との差): 中央値 {np.median(st['e']):.3f} / {int(Q * 100)}%点 {np.quantile(st['e'], Q):.3f} / 最大 {st['e'].max():.3f}")
    print(f"        放射方向: 平均 {st['e_rad'].mean():+.3f} 標準偏差 {st['e_rad'].std():.3f} / 円周方向: 平均 {st['e_tan'].mean():+.3f} 標準偏差 {st['e_tan'].std():.3f}")
    rows = make_table(st)
    print(f"\n【半径ビンごと】(q={Q}, 許容 ε={EPS_M} m)")
    print(f"{'半径[px]':>14}{'n':>5}{'|e|中央値[m]':>13}{f'|e|{int(Q * 100)}%[m]':>11}{'包絡[m]':>9}{f'e_px{int(Q * 100)}%[px相当]':>16}{'包絡[px]':>10}{'放射偏り':>9}{'放射sd':>8}{'円周sd':>8}")
    for x in rows:
        print(f"{x['r_lo']:>7.0f}-{x['r_hi']:<6.0f}{x['n']:>5}{x['e_med_m']:>13.3f}{x['e_q_m']:>11.3f}{x['e_q_m_env']:>9.3f}"
              f"{x['e_px_q']:>16.1f}{x['e_px_q_env']:>10.1f}{x['rad_mean']:>+9.3f}{x['rad_sd']:>8.3f}{x['tan_sd']:>8.3f}")
    over = [x for x in rows if x["e_q_m_env"] > EPS_M]
    print(f"\nメートルで見ると: {int(Q * 100)}%点の誤差が {EPS_M} m を超える最初の半径ビン: "
          + (f"{over[0]['r_lo']:.0f}-{over[0]['r_hi']:.0f} px" if over else "なし(全ビンで以内)"))
    print("\n【カメラごと】(全半径)")
    for cid in sorted(set(st["cam"])):
        sel = st["cam"] == cid
        print(f"  {cid}: n={int(sel.sum())} |e|中央値 {np.median(st['e'][sel]):.3f} m / {int(Q * 100)}%点 {np.quantile(st['e'][sel], Q):.3f} m / 半径の範囲 {st['r'][sel].min():.0f}-{st['r'][sel].max():.0f} px")

    out = out_dir / TABLE_NAME
    out.write_text(json.dumps(table_to_json(rows, Q, frames, h_eff, n), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n保存: {out}")
    print("※ 三角測量GTと同じ校正・クリックで作った正解との差。YOLOの検出ノイズは含まない。フレームが少ないほど表は粗い。")


if __name__ == "__main__":
    main()