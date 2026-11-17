"""
nn_distance_tri_gt_ver1.py

三角測量GTの牛について、「最近傍の牛までの距離」の分布を出す。
目的: マッチングの許容半径(マハラノビスのゲート)や、カメラ有効範囲のε(牛の体の半分など)が、
      牛どうしの間隔に比べて大きすぎないか(誤統合が起きないか)を見る基準にする。

位置の取り方を2通り出して比べる:
  (A) 三角測量した位置 X の (x, y)。クリックした点から直接決まるので、高さの仮定は不要。
  (B) 検出が出す位置に近いもの: 各カメラのbbox中心を想定高さ(立ち1.4m / 寝0.5m)で床に投影し、
      カメラ間で平均した位置。想定高さは ASSUMED_H で変更できる。
最近傍は、フレームごとに牛どうしで求める(別フレームの牛とは比べない)。
姿勢の組み合わせ別(立ち-立ち, 寝-寝, 立ち-寝)にも集計する。

使い方: python nn_distance_tri_gt_ver1.py [フレーム番号 ...]   (省略時はGTのある全フレーム)
出力: <OUTPUT_DIR>/bev_calibration_refined/nn_distance_tri_gt_ver1.csv
校正は Mootrack_core.py の現在値(MOOTRACK_CALIB で切り替え可)。(B)のみ校正に依存。
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

ASSUMED_H = {"standing": 1.4, "lying": 0.5}     # (B)で使う想定高さ[m]
PERCENTILES = (0, 5, 10, 25, 50, 75)
THRESHOLDS_M = (0.5, 1.0, 1.5, 2.0, 3.0)         # 「最近傍がこれ未満の牛」の割合を出す


def nearest_neighbor(points):
    """points: (n,2)。各点の最近傍までの距離と、その相手のindexを返す"""
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    j = np.argmin(d, axis=1)
    return d[np.arange(len(points)), j], j


def report(name, rows):
    """rows: dict(nn, pair) のリスト。全体と姿勢の組み合わせ別に分布を表示する"""
    print(f"\n=== {name}: 牛 {len(rows)}頭 ===")
    groups = {"全体": rows}
    for key in ("standing-standing", "lying-lying", "standing-lying"):
        groups[key] = [r for r in rows if r["pair"] == key]
    head = "".join(f"{('p%d' % p):>8}" for p in PERCENTILES)
    print(f"{'組み合わせ':<20}{'頭数':>6}{head}" + "".join(f"{('<%.1fm' % t):>8}" for t in THRESHOLDS_M))
    for key, rs in groups.items():
        if not rs:
            continue
        v = np.array([r["nn"] for r in rs])
        pcs = "".join(f"{np.percentile(v, p):>8.2f}" for p in PERCENTILES)
        frac = "".join(f"{(v < t).mean() * 100:>7.0f}%" for t in THRESHOLDS_M)
        print(f"{key:<20}{len(rs):>6}{pcs}{frac}")


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
    print(f"校正セット: {getattr(mc, 'CALIBRATION_SET', 'Mootrack_core.pyの現在値')} / フレーム {frames} / (B)の想定高さ {ASSUMED_H}")

    raw = []
    for fr in frames:
        xp, tp = Path(an.XML_DIR) / f"{fr:06d}.xml", in_dir / f"{fr:06d}_tri_gt_ver1.json"
        if xp.exists() and tp.exists():
            raw += fh.build_raw(json.loads(tp.read_text(encoding="utf-8"))["cows"], an.parse_xml(xp), cam_ids, an, fr)
        else:
            print(f"見つかりません: {xp if not xp.exists() else tp}")
    if not raw:
        print("サンプルがありません")
        return

    # (B) 想定高さでbbox中心を床に投影 -> 牛ごとにカメラ平均
    h = np.array([ASSUMED_H[s["pose"]] for s in raw])
    P_est, ok = fh.project(raw, poses, get_intrinsics, h)
    cows = {}
    for i, s in enumerate(raw):
        c = cows.setdefault((s["frame"], s["cow"]), dict(frame=s["frame"], cow=s["cow"], pose=s["pose"],
                                                         P_gt=s["P_gt"], est=[]))
        if ok[i]:
            c["est"].append(P_est[i])

    csv_rows = []
    for source, label in (("A", "(A) 三角測量した位置(クリックから直接。高さの仮定なし)"),
                          ("B", f"(B) bbox中心を想定高さ {ASSUMED_H} で投影した位置(カメラ平均)")):
        result = []
        for fr in frames:
            members = [c for c in cows.values() if c["frame"] == fr and (source == "A" or c["est"])]
            if len(members) < 2:
                continue
            pts = np.array([c["P_gt"] if source == "A" else np.mean(c["est"], axis=0) for c in members])
            nn, j = nearest_neighbor(pts)
            for k, c in enumerate(members):
                pair = "-".join(sorted((c["pose"], members[j[k]]["pose"]), reverse=True))
                result.append(dict(nn=float(nn[k]), pair=pair))
                csv_rows.append([source, fr, c["cow"], c["pose"], members[j[k]]["cow"], members[j[k]]["pose"], f"{nn[k]:.3f}"])
        report(label, result)

    path = out_dir / "nn_distance_tri_gt_ver1.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["position_source", "frame", "cow", "pose", "nn_cow", "nn_pose", "nn_distance_m"])
        w.writerows(csv_rows)
    print(f"\nCSV: {path}")
    print("※ 許容半径(マハラノビスのゲート半径)が、この最近傍距離の下位(p5〜p10)より小さければ、誤統合はまれ。")
    print("※ 1フレームの牛のみ。フレームを増やすと分布が安定する。")


if __name__ == "__main__":
    main()