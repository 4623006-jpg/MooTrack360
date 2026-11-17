"""
coverage_gap_check_he_kai_ver1.py

有効範囲(多角形)に「隙間」があるかを、GTの牛の位置で調べる (YOLOは使わない)。
  範囲の外の検出を捨てる使い方(maha_h_valid / maha_h_tri など)では、どのカメラの範囲にも入らない牛は FN になる。

  各 (フレーム, 牛) について、その牛が写っている各カメラの bbox中心を h_eff(姿勢) で床に投影し、
    - 投影位置が そのカメラの範囲の中か
    - 担当カメラ(牛の平均位置にいちばん近いカメラ)の範囲の中か
  を調べ、次を出す:
    どのカメラの範囲にも入らない牛 (= 範囲を捨てる使い方なら FN になる牛) の割合
    担当カメラの範囲に入らない牛の割合
    入らない牛の位置(x,y)の分布 (どの辺りに隙間があるか) とCSV

使い方: uv run python ./phase4/he_kai/coverage_gap_check_he_kai_ver1.py [orig] [valid] [tri]   (省略時は orig valid tri)
環境変数: HE_FRAME_START / HE_FRAME_END / HE_FRAME_STEP (既定 0 / 2600 / 10), HE_EPS_M / HE_K_SIGMA (valid),
          HE_TRI_TABLE / HE_TRI_EPS_M / HE_TRI_KIND (tri), HE_SIGMA_JSON, HE_OUT_DIR
配置: research/phase4/he_kai/  (phase3_he_kai_matching_ver3.py 以降と同じ場所)
"""

import csv
import os
import sys
from collections import defaultdict
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

FRAME_START = int(os.environ.get("HE_FRAME_START", 0))
FRAME_END = int(os.environ.get("HE_FRAME_END", 2600))
FRAME_STEP = int(os.environ.get("HE_FRAME_STEP", 10))


def main():
    import phase3_he_kai_matching_ver3 as hk
    import gt_ignore_ver1 as gi
    from he_kai_sigma_model_ver1 import build_valid_polygons, load_model
    from zone_overlap_filter_he_kai_ver1 import point_in_polygon, zone_owner
    from Mootrack_core import (CAM_IDS, ROT_FMT, apply_camera_overrides, get_intrinsics, load_floorplan,
                               resolve_poses, unproject_to_floor)

    which = sys.argv[1:] or ["orig", "valid", "tri"]
    ym = hk.load_yolo_module()
    gi.apply(ym)
    regions, cameras_json, real_w, real_h = load_floorplan(ym.FLOORPLAN_JSON_PATH)
    poses = resolve_poses(apply_camera_overrides(cameras_json), root="Area1", fmt=ROT_FMT)
    json_path = os.environ.get("HE_SIGMA_JSON") or str(Path(ym.OUTPUT_DIR) / "bev_calibration_refined" / "sigma_he_kai_ver4.json")
    model = load_model(json_path)
    hk._MODEL = model
    cam_xy = {c: np.asarray(poses[c].t[:2], dtype=np.float64) for c in CAM_IDS}

    polys = {}
    for w in which:
        if w == "orig":
            polys[w] = ym.build_camera_polygons(poses, CAM_IDS, get_intrinsics, height_m=ym.ZONE_HEIGHT_M, n_angles=ym.N_ANGLES)
        elif w == "valid":
            polys[w] = build_valid_polygons(poses, CAM_IDS, get_intrinsics, model, hk.EPS_VALID_M, hk.K_SIGMA,
                                            height_m=0.8, n_angles=144, verbose=True)
        elif w == "tri":
            polys[w] = hk.build_tri_polygons(poses, CAM_IDS, get_intrinsics, height_m=0.8)
        else:
            raise SystemExit(f"不明: {w} (orig / valid / tri)")

    tile_w = ym.IMAGE_WIDTH_PX / len(CAM_IDS)
    cows = defaultdict(list)                  # (frame, name) -> [(cam, xy)]
    for fr in range(FRAME_START, FRAME_END + 1, FRAME_STEP):
        xml = Path(ym.XML_DIR) / f"{ym.frame_name(fr)}.xml"
        if not xml.exists():
            continue
        for o in ym.parse_voc_annotation(xml):
            cx_full = (o["xmin"] + o["xmax"]) / 2.0
            ci = max(0, min(int(cx_full // tile_w), len(CAM_IDS) - 1))
            cid = CAM_IDS[ci]
            uv = (cx_full - ci * tile_w, (o["ymin"] + o["ymax"]) / 2.0)
            h = model.h_eff(o["pose"]) if o["pose"] in ("standing", "lying") else 0.8
            fx, fy, cx, cy, xi, alpha = get_intrinsics(cid)
            xy, ok = unproject_to_floor(np.array([uv], dtype=np.float64), poses[cid].R, poses[cid].t,
                                        fx, fy, cx, cy, xi, alpha, floor_z=-h)
            if ok.all():
                cows[(fr, o["name"])].append((cid, xy[0]))
    print(f"\n対象: (フレーム,牛) {len(cows)}組")

    out_dir = Path(os.environ.get("HE_OUT_DIR") or Path(PHASE4_DIR) / "diagnose" / "analyze_check")
    out_dir.mkdir(parents=True, exist_ok=True)
    for w in which:
        P = polys[w]
        n_any = n_owner = n_two = 0
        unc, unc_owner = [], []
        by_owner = defaultdict(lambda: [0, 0])
        for (fr, name), obs in cows.items():
            mean_xy = np.mean([xy for _, xy in obs], axis=0)
            owner = zone_owner(mean_xy, cam_xy)
            inside = [c for c, xy in obs if point_in_polygon(xy, P[c])]
            by_owner[owner][0] += 1
            if inside:
                n_any += 1
            else:
                unc.append((fr, name, owner, mean_xy))
            if owner in inside:
                n_owner += 1
                by_owner[owner][1] += 1
            else:
                unc_owner.append((fr, name, owner, mean_xy))
            if len(inside) >= 2:
                n_two += 1
        N = len(cows)
        print(f"\n{'=' * 60}\n■ 範囲: {w}\n{'=' * 60}")
        print(f"どれかのカメラの範囲に入る牛     : {n_any / N * 100:6.1f}%   <- 範囲外を捨てる使い方での上限(これ以外は FN)")
        print(f"  入らない牛(= FNになる牛)       : {len(unc)}組 ({len(unc) / N * 100:.1f}%)")
        print(f"担当カメラの範囲に入る牛         : {n_owner / N * 100:6.1f}%")
        print(f"2台以上の範囲に入る牛(統合可能)  : {n_two / N * 100:6.1f}%")
        for c in CAM_IDS:
            t, k = by_owner[c]
            print(f"  担当 {c}: {t}組 / 担当の範囲に入る {k / max(t, 1) * 100:.1f}%")
        if unc:
            xs = np.array([u[3] for u in unc])
            print(f"範囲に入らない牛の位置: x {xs[:, 0].min():.1f}〜{xs[:, 0].max():.1f} m (中央値 {np.median(xs[:, 0]):.1f}) / "
                  f"y {xs[:, 1].min():.1f}〜{xs[:, 1].max():.1f} m (中央値 {np.median(xs[:, 1]):.1f})")
            hist, edges = np.histogram(xs[:, 0], bins=8, range=(0, real_w))
            print("  x方向の分布: " + " ".join(f"{edges[i]:.0f}-{edges[i + 1]:.0f}m:{int(hist[i])}" for i in range(len(hist))))
        out = out_dir / f"coverage_gap_{w}_he_kai_ver1.csv"
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            wr = csv.writer(f)
            wr.writerow(["frame", "cow", "owner", "x", "y", "any_camera_inside"])
            for fr, name, owner, mxy in unc:
                wr.writerow([fr, name, owner, round(float(mxy[0]), 3), round(float(mxy[1]), 3), 0])
        print(f"CSV(入らない牛): {out}")
    print("\n※ GTのbbox中心を h_eff で床に投影した位置で判定。YOLOの検出位置のぶれは含まない。")


if __name__ == "__main__":
    main()