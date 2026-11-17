"""
diagnose_cow_region_he_kai_ver1.py   (置き場所: research/phase4/diagnose/)

特定の牛(例: cow029)の周りで、フレームごとに「統合前の検出」と「統合後の出力」をモードごとに並べて見る診断。
  - euclid と euclid_h で、同じ牛の出力が分かれたり入れ替わったりする理由を調べるために作った。
  - 牛の位置は、その牛の GT(XML)を、そのモードの想定高さで投影した位置を中心にする(環境変数で座標を直接指定も可)。
  - 中心から半径 DIAG_RADIUS_M [m] 以内の、各カメラの検出(位置・姿勢)と、
    隣接カメラの検出ペアの距離(ユークリッド、姿勢が同じか、マハラノビス d²)と、統合後の出力(位置・何台で統合したか)を出す。

使い方(research/ で):
    HE_POSE_CONSTRAINT=1 uv run python ./phase4/diagnose/diagnose_cow_region_he_kai_ver1.py euclid euclid_h
環境変数:
    DIAG_COW        : 調べる牛の名前 (既定 cow029)
    DIAG_FRAMES     : 調べるフレーム (カンマ区切り。既定 1710,1720,1730,1740,1750,1760,1770)
    DIAG_CENTER     : "x,y" で中心を直接指定(GT の牛を使わないとき)
    DIAG_RADIUS_M   : 中心からの半径 [m] (既定 2.5)
    HE_SIGMA_JSON / HE_SIGMA_MODEL / HE_EPS_M / HE_GATE_CHI2 / HE_POSE_CONSTRAINT : phase3_he_kai_matching と同じ
    HE_MATCHING_MODULE : 既定 phase3_he_kai_matching_ver10
出力: 画面 と phase4/he_kai_out/diagnose_cow_ver1/<牛>_<モード>.csv
注意: トラッカー(ID)は動かさない。統合後の出力が「何本あって、どこにあるか」を見る(track ID の入れ替わりは、その2本の出力のどちらが GT に近いかで決まる)。
"""

import csv
import importlib
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

os.environ.setdefault("HE_SIGMA_MODEL", "const")
DEFAULT_SIGMA_JSON = os.path.join(RESEARCH_DIR, "idf1_check", "bev_calibration_refined", "sigma_he_kai_ver4.json")
OUT_DIR = Path(os.environ.get("HE_OUT_DIR", os.path.join(PHASE4_DIR, "he_kai_out"))) / "diagnose_cow_ver1"
COW = os.environ.get("DIAG_COW", "cow029")
FRAMES = [int(x) for x in os.environ.get("DIAG_FRAMES", "1710,1720,1730,1740,1750,1760,1770").split(",") if x.strip()]
RADIUS_M = float(os.environ.get("DIAG_RADIUS_M", 2.5))
CENTER_ENV = os.environ.get("DIAG_CENTER")


def fused_list(result):
    """融合関数の戻り値を、[(位置(2,), n_cams)] にする。(融合リスト, 却下数) のタプルにも対応"""
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], (list, tuple)):
        result = result[0]
    out = []
    for f in result:
        n = None
        if isinstance(f, dict):
            n = f.get("n_cams")
            f = f.get("world_xy", f.get("xy"))
        out.append((np.asarray(f, dtype=np.float64).ravel()[:2], n))
    return out


def near(xy, center, radius):
    return float(np.linalg.norm(np.asarray(xy, dtype=np.float64) - center)) <= radius


def main():
    from ultralytics import YOLO
    import render_bev_frames_yolo_zonepose_ver1 as rf_old
    hk = importlib.import_module(os.environ.get("HE_MATCHING_MODULE", "phase3_he_kai_matching_ver10"))
    from curved_zone_utils_ver1 import N_ANGLES, ZONE_HEIGHT_M, build_camera_polygons, filter_by_curved_zone, get_adjacent_pairs
    from he_kai_sigma_model_ver1 import build_valid_polygons, load_model
    from Mootrack_core import (CAM_IDS, ROT_FMT, apply_camera_overrides, filter_outside_floorplan_bounds, get_intrinsics,
                               load_floorplan, resolve_poses)

    ym = hk.load_yolo_module()

    class _Proxy:
        def __init__(self, primary, fallback):
            object.__setattr__(self, "_p", primary)
            object.__setattr__(self, "_f", fallback)

        def __getattr__(self, name):
            try:
                return getattr(self._p, name)
            except AttributeError:
                return getattr(self._f, name)

        def __setattr__(self, name, value):
            setattr(self._p, name, value)

    rf = _Proxy(ym, rf_old)
    modes = sys.argv[1:] or ["euclid", "euclid_h"]
    bad = [m for m in modes if m not in hk.MODES]
    if bad:
        raise SystemExit(f"不明なモード: {bad} / 使えるモード: {list(hk.MODES)}")
    if "HE_POSE_CONSTRAINT" in os.environ:
        rf.USE_POSE_CONSTRAINT = os.environ["HE_POSE_CONSTRAINT"] == "1"
    print(f"姿勢別統合制約: {'ON' if rf.USE_POSE_CONSTRAINT else 'OFF'} / 牛 {COW} / フレーム {FRAMES} / 半径 {RADIUS_M} m / モード {modes}")

    regions, cameras_json, real_w, real_h = load_floorplan(rf.FLOORPLAN_JSON_PATH)
    poses = resolve_poses(apply_camera_overrides(cameras_json), root="Area1", fmt=ROT_FMT)
    model = load_model(os.environ.get("HE_SIGMA_JSON") or DEFAULT_SIGMA_JSON)
    hk._MODEL, hk._YM = model, ym
    hk.setup_poses(poses)
    orig_heights = (rf.STANDING_HEIGHT_M, rf.LYING_HEIGHT_M)

    polygons = {}
    for m in modes:
        kind = hk.MODES[m]["polygons"]
        if kind == "he_kai":
            polygons[m] = build_valid_polygons(poses, CAM_IDS, get_intrinsics, model, hk.EPS_VALID_M, hk.K_SIGMA,
                                               height_m=ZONE_HEIGHT_M, n_angles=N_ANGLES, verbose=False)
        elif kind == "tri_gt":
            polygons[m] = hk.build_tri_polygons(poses, CAM_IDS, get_intrinsics, height_m=ZONE_HEIGHT_M, n_angles=N_ANGLES)
        else:
            polygons[m] = build_camera_polygons(poses, CAM_IDS, get_intrinsics, height_m=ZONE_HEIGHT_M, n_angles=N_ANGLES, verbose=False)
    adjacent_pairs = get_adjacent_pairs(CAM_IDS)
    yolo_model = YOLO(rf.YOLO_WEIGHTS_PATH)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_rows = {m: [] for m in modes}
    for fr in FRAMES:
        ipath = Path(rf.IMAGE_DIR) / f"{rf.frame_name(fr)}.jpg"
        xpath = Path(rf.XML_DIR) / f"{rf.frame_name(fr)}.xml"
        if not ipath.exists():
            print(f"\n[frame {fr}] 画像がありません: {ipath}")
            continue
        print(f"\n{'=' * 90}\n[frame {fr}]")
        raw_cache = {}
        for m in modes:
            cfg = hk.MODES[m]
            if cfg["heights"] == "fit":
                rf.STANDING_HEIGHT_M, rf.LYING_HEIGHT_M = model.h_eff("standing"), model.h_eff("lying")
            else:
                rf.STANDING_HEIGHT_M, rf.LYING_HEIGHT_M = orig_heights

            # 中心: 指定座標、または、その牛の GT(XML)をこのモードの高さで投影した位置(カメラ平均)
            center = None
            if CENTER_ENV:
                center = np.array([float(v) for v in CENTER_ENV.split(",")], dtype=np.float64)
            elif xpath.exists():
                objs = rf.parse_voc_annotation(xpath)
                gt_per_cam = rf.gt_world_points_per_camera(objs, poses, rf.IMAGE_WIDTH_PX)
                pts = [np.asarray(it["world_xy"], dtype=np.float64) for items in gt_per_cam.values() for it in items if it["name"] == COW]
                if pts:
                    center = np.mean(pts, axis=0)
            if center is None:
                print(f"  [{m}] {COW} の GT が見つかりません(DIAG_CENTER で座標を指定できます)")
                continue

            if cfg["heights"] not in raw_cache:
                raw_cache[cfg["heights"]] = rf.yolo_world_points_per_camera_with_pose(yolo_model, ipath, poses)
            raw = {cid: [dict(it) for it in items] for cid, items in raw_cache[cfg["heights"]].items()}
            deduped = rf.deduplicate_per_camera(raw, rf.DEDUP_IOU_THRESHOLD)
            filtered = filter_by_curved_zone(deduped, polygons[m])
            filtered = filter_outside_floorplan_bounds(filtered, real_w, real_h, margin_m=0.0, verbose=False)

            print(f"\n  --- [{m}] 想定高さ standing={rf.STANDING_HEIGHT_M:.3f} lying={rf.LYING_HEIGHT_M:.3f} / {COW} の GT 位置 ({center[0]:.2f}, {center[1]:.2f}) ---")
            print("  統合前の検出(中心から半径内):")
            idx = {}
            for cid in CAM_IDS:
                for k, it in enumerate(filtered.get(cid, [])):
                    xy = np.asarray(it["world_xy"], dtype=np.float64)
                    if near(xy, center, RADIUS_M):
                        idx[(cid, k)] = it
                        print(f"    {cid} #{k}: 位置 ({xy[0]:.2f}, {xy[1]:.2f}) 姿勢 {it.get('pose')}  GTから {np.linalg.norm(xy - center):.2f} m")
                        csv_rows[m].append(dict(frame=fr, kind="det", cam=cid, k=k, x=round(float(xy[0]), 3), y=round(float(xy[1]), 3),
                                                pose=it.get("pose"), n_cams="", d_euclid="", same_pose="", d2=""))
            print("  隣接カメラの検出ペア:")
            any_pair = False
            for ci, cj in adjacent_pairs:
                for (c1, a), ia in idx.items():
                    if c1 != ci:
                        continue
                    for (c2, b), ib in idx.items():
                        if c2 != cj:
                            continue
                        any_pair = True
                        d = np.asarray(ia["world_xy"], dtype=np.float64) - np.asarray(ib["world_xy"], dtype=np.float64)
                        euc = float(np.linalg.norm(d))
                        same = (ia.get("pose") == ib.get("pose")) if (ia.get("pose") is not None and ib.get("pose") is not None) else None
                        d2 = None
                        try:
                            Sa, Sb = hk.det_covariance(ci, ia, ym.height_for), hk.det_covariance(cj, ib, ym.height_for)
                            if Sa is not None and Sb is not None:
                                d2 = float(d @ np.linalg.solve(Sa + Sb, d))
                        except Exception:
                            pass
                        flag = ""
                        if cfg["match"] == "euclid":
                            ok_dist = euc <= rf.DIST_THR_FUSION_M
                            flag = f"距離しきい値{rf.DIST_THR_FUSION_M:g}m以内:{'○' if ok_dist else '×'}"
                        elif d2 is not None:
                            flag = f"ゲート{hk.GATE_CHI2:g}以内:{'○' if d2 <= hk.GATE_CHI2 else '×'}"
                        pose_s = {True: "同じ", False: "違う", None: "不明"}[same]
                        print(f"    {ci}#{a} - {cj}#{b}: 距離 {euc:.2f} m / 姿勢 {pose_s} ({ia.get('pose')},{ib.get('pose')})"
                              + (f" / d²={d2:.1f}" if d2 is not None else "") + f" / {flag}")
                        csv_rows[m].append(dict(frame=fr, kind="pair", cam=f"{ci}-{cj}", k=f"{a}-{b}", x="", y="", pose="",
                                                n_cams="", d_euclid=round(euc, 3), same_pose=same, d2=None if d2 is None else round(d2, 2)))
            if not any_pair:
                print("    (なし)")

            if cfg["match"] == "maha":
                fused = hk.fuse_mahalanobis(filtered, adjacent_pairs, rf.DIST_THR_FUSION_M, rf.USE_POSE_CONSTRAINT)
            else:
                fused = rf.fuse_frame_zone_pose(filtered, adjacent_pairs, rf.DIST_THR_FUSION_M, rf.USE_POSE_CONSTRAINT)
            outs = [(xy, n) for xy, n in fused_list(fused) if near(xy, center, RADIUS_M)]
            print(f"  統合後の出力(中心から半径内): {len(outs)}本")
            for xy, n in outs:
                print(f"    位置 ({xy[0]:.2f}, {xy[1]:.2f}) 統合したカメラ数 {n}  GTから {np.linalg.norm(xy - center):.2f} m")
                csv_rows[m].append(dict(frame=fr, kind="fused", cam="", k="", x=round(float(xy[0]), 3), y=round(float(xy[1]), 3),
                                        pose="", n_cams=n, d_euclid="", same_pose="", d2=""))
        rf.STANDING_HEIGHT_M, rf.LYING_HEIGHT_M = orig_heights

    for m in modes:
        path = OUT_DIR / f"{COW}_{m}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["frame", "kind", "cam", "k", "x", "y", "pose", "n_cams", "d_euclid", "same_pose", "d2"])
            w.writeheader()
            w.writerows(csv_rows[m])
    print(f"\n保存: {OUT_DIR}")


if __name__ == "__main__":
    main()