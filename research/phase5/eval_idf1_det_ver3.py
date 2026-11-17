"""
eval_idf1_det_ver3.py  (配置: research/phase5/)  ※ver2 に「重なり範囲 / 単眼範囲」の層別を追加し、動きの層は牛単位だけにした

検出単位の IDF1(誤統合・統合し損ねが IDFP / IDFN に出る IDF1)を、層ごとに計算する。
IDF1 の定義・統合・追跡・名前付けは ver1 と同じ(ver1 の関数を import して使う)。1回の実行で全層・全モードを出す。

層(GT の牛の BEV 位置だけから決める。手法には依存しない):
  全体            : すべて(ver1 の結果と一致する確認用)
  動いた牛 / 動かない牛 : 牛単位。全期間で GT の位置が STRAT_ACTIVE_M [m] 以上動いた牛 / それ以外
  重なり範囲 / 単眼範囲 : フレーム単位。その牛の GT 位置が 2 台以上のカメラの有効範囲に入る / 1 台以下
                         (有効範囲は STRAT_RANGE の多角形。全モード共通で固定。eval_stratified_ver2.py と同じ)
  上の組み合わせ  : 動いた牛×重なり範囲 / 動いた牛×単眼範囲 / 動かない牛×重なり範囲 / 動かない牛×単眼範囲
  (ver2 のフレーム単位の「動く牛 / 静止牛」は外した)

層の作り方: 各層で (フレーム, 牛の名前) が層に入るものだけを GT に残し、仮説側は「名前が付いた検出のうち、その名前が層に入るもの」だけを残して、
  ver1 と同じく motmetrics で IDF1 を計算し直す。名前が付かなかった検出(YOLO の誤検出など)は「全体」にだけ数える。
読み方の注意:
  ・層をまたぐ誤統合(例: 重なり範囲の牛と単眼範囲の牛が同じクラスタ)は、各層では片方の牛の検出しか残らないため、どちらの層にも出ない(「全体」にだけ出る)。
    層の IDFP / IDFN の合計は全体と一致しない。
  ・フレーム単位の層(重なり / 単眼)は、牛の軌跡が層の出入りで途切れるので、IDF1 は残ったフレームだけで計算される。
  ・GT延べ数が小さい層は値が不安定。手法ごとの有効範囲が違うので、GT延べ数は手法で違う(全手法で揃える対応は未実装)。

使い方(research/ で):
    HE_POSE_CONSTRAINT=0 uv run python ./phase5/eval_idf1_det_ver3.py iou@0.10 euclid maha_h_valid
環境変数: ver1 と同じ + STRAT_ACTIVE_M(既定2.0) / STRAT_RANGE(重なり判定の有効範囲 orig(既定) / he_kai)
出力: research/phase5/baseline_out/idf1_det_ver3/idf1_det_strata.csv
"""

import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import baseline_iou_ver1 as bl
import render_bev_baseline_ver1 as rb
from eval_idf1_det_ver1 import IdMap, add_cam_frame, fuse_points, track_ids_for

OUT_ROOT = Path(os.environ.get("IDF1DET_OUT_DIR", os.path.join(THIS_DIR, "baseline_out", "idf1_det_ver3")))
ACTIVE_M = float(os.environ.get("STRAT_ACTIVE_M", 2.0))
STRAT_RANGE = os.environ.get("STRAT_RANGE", "orig")
METRICS = ["idf1", "idp", "idr", "idtp", "idfp", "idfn", "num_objects", "num_predictions"]
STRATA_ORDER = ["全体", "動いた牛", "動かない牛", "重なり範囲", "単眼範囲",
                "動いた牛×重なり", "動いた牛×単眼", "動かない牛×重なり", "動かない牛×単眼"]


def gt_series(frame_indices, rf, poses, xml_dir, image_dir):
    """全フレームの GT の牛の位置 {名前: [(サンプル番号, フレーム番号, xy)]}(処理するフレームだけ。YOLO は使わない)"""
    series = defaultdict(list)
    k = 0
    for fi in frame_indices:
        xml_path = xml_dir / f"{rf.frame_name(fi)}.xml"
        image_path = image_dir / f"{rf.frame_name(fi)}.jpg"
        if not (image_path.exists() and xml_path.exists()):
            continue
        per_cam = rf.gt_world_points_per_camera(rf.parse_voc_annotation(xml_path), poses, rf.IMAGE_WIDTH_PX)
        for items in per_cam.values():
            for it in items:
                series[it["name"]].append((k, fi, np.asarray(it["world_xy"], dtype=np.float64)))
        k += 1
    return series


def build_flags(series, strat_polys):
    """overlap[(フレーム番号, 名前)] = GT 位置が 2 台以上のカメラの有効範囲に入るか / active = 動いた牛の名前の集合 / all_names"""
    from curved_zone_utils_ver1 import point_in_polygon
    overlap, active = {}, set()
    for name, ser in series.items():
        pts = np.array([p for _, _, p in ser])
        if float(np.max(np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1))) >= ACTIVE_M:
            active.add(name)
        for k, fi, p in ser:
            overlap[(fi, name)] = sum(point_in_polygon(p, poly) for poly in strat_polys.values()) >= 2
    return overlap, active, set(series)


def main():
    import motmetrics as mm
    import pandas as pd
    from ultralytics import YOLO

    jobs = bl.parse_args(sys.argv[1:] or ["iou@0.10", "euclid", "maha_h_valid"])
    import render_bev_trajectory_he_kai_ver6 as rv
    rf, ym, hk = rv.rf, rv.ym, bl.hk
    from Mootrack_core import (CAM_IDS, ROT_FMT, apply_camera_overrides, filter_outside_floorplan_bounds, load_floorplan,
                               get_intrinsics, resolve_poses, union_find_clusters)
    from curved_zone_utils_ver1 import N_ANGLES, ZONE_HEIGHT_M, build_camera_polygons, filter_by_curved_zone, get_adjacent_pairs
    from he_kai_sigma_model_ver1 import build_valid_polygons, load_model

    if "HE_POSE_CONSTRAINT" in os.environ:
        rf.USE_POSE_CONSTRAINT = os.environ["HE_POSE_CONSTRAINT"] == "1"
    print(f"姿勢別統合制約: {'ON' if rf.USE_POSE_CONSTRAINT else 'OFF'} / フレーム {rv.FRAME_START}〜{rv.FRAME_END} 刻み {rv.FRAME_STEP} / B の方式: {bl.BASE_STYLE}")
    print(f"層分け: 動いた牛 = 全期間で {ACTIVE_M} m 以上 / 重なり範囲 = 2台以上の有効範囲({STRAT_RANGE})に入る")

    import gt_ignore_ver1 as gi
    gt_ignore_state = gi.apply(ym)

    os.environ.setdefault("HE_SIGMA_MODEL", "const")
    json_path = os.environ.get("HE_SIGMA_JSON") or str(Path(rf.OUTPUT_DIR) / "bev_calibration_refined" / "sigma_he_kai_ver4.json")
    model = load_model(json_path)
    print(model.describe())

    regions, cameras_json, real_w, real_h = load_floorplan(rf.FLOORPLAN_JSON_PATH)
    poses = resolve_poses(apply_camera_overrides(cameras_json), root="Area1", fmt=ROT_FMT)
    hk._YM, hk._MODEL = ym, model
    hk.setup_poses(poses)
    bl.MAP_WH = (real_w, real_h)
    orig_heights = (rf.STANDING_HEIGHT_M, rf.LYING_HEIGHT_M)

    adjacent_pairs = get_adjacent_pairs(CAM_IDS)
    poly_orig = build_camera_polygons(poses, CAM_IDS, get_intrinsics, height_m=ZONE_HEIGHT_M, n_angles=N_ANGLES)
    poly_he = None
    exclude = getattr(ym, "EXCLUDE_GT_NAMES", set())

    plans = []
    for name, thr, hk_mode in jobs:
        if hk_mode is None:
            plans.append(dict(name=name, kind="iou", thr=thr, heights=bl.BASE_HEIGHTS, polygons="orig"))
        else:
            cfg = hk.MODES[hk_mode]
            if cfg.get("zone") or cfg.get("gate") or cfg.get("rescue"):
                raise SystemExit(f"{hk_mode} は非対応(zone/gate/rescue 系)")
            plans.append(dict(name=name, kind=cfg["match"], thr=None, heights=cfg["heights"], polygons=cfg["polygons"]))
    polys = {}
    for p in plans:
        if p["polygons"] == "he_kai":
            if poly_he is None:
                poly_he = build_valid_polygons(poses, CAM_IDS, get_intrinsics, model, hk.EPS_VALID_M, hk.K_SIGMA, height_m=ZONE_HEIGHT_M, n_angles=N_ANGLES)
            polys[p["name"]] = poly_he
        elif p["polygons"] == "orig":
            polys[p["name"]] = poly_orig
        else:
            raise SystemExit(f"{p['name']}: 有効範囲 {p['polygons']} は非対応")

    xml_dir, image_dir = Path(rf.XML_DIR), Path(rf.IMAGE_DIR)
    frame_indices = list(range(rv.FRAME_START, rv.FRAME_END + 1, rv.FRAME_STEP))

    # 先に全フレームの GT の位置から、動き別の層を決める(手法に依存しない)
    series = gt_series(frame_indices, rf, poses, xml_dir, image_dir)
    if STRAT_RANGE == "he_kai":
        if poly_he is None:
            poly_he = build_valid_polygons(poses, CAM_IDS, get_intrinsics, model, hk.EPS_VALID_M, hk.K_SIGMA, height_m=ZONE_HEIGHT_M, n_angles=N_ANGLES)
        strat_polys = poly_he
    else:
        strat_polys = poly_orig
    overlap, active, all_names = build_flags(series, strat_polys)
    n_ov = sum(1 for v in overlap.values() if v)
    print(f"GT の牛 {len(all_names)} 頭(動いた牛 {len(active)} 頭) / 牛×フレーム {len(overlap)} 件(重なり範囲 {n_ov} / 単眼範囲 {len(overlap) - n_ov}、範囲={STRAT_RANGE})")
    is_act = lambda n: n in active
    is_ov = lambda fi, n: overlap.get((fi, n), False)
    known = lambda fi, n: (fi, n) in overlap
    preds = {
        "全体":             lambda fi, n: True,
        "動いた牛":         lambda fi, n: is_act(n),
        "動かない牛":       lambda fi, n: n in all_names and not is_act(n),
        "重なり範囲":       lambda fi, n: is_ov(fi, n),
        "単眼範囲":         lambda fi, n: known(fi, n) and not is_ov(fi, n),
        "動いた牛×重なり":   lambda fi, n: is_act(n) and is_ov(fi, n),
        "動いた牛×単眼":     lambda fi, n: is_act(n) and known(fi, n) and not is_ov(fi, n),
        "動かない牛×重なり": lambda fi, n: n in all_names and not is_act(n) and is_ov(fi, n),
        "動かない牛×単眼":   lambda fi, n: n in all_names and not is_act(n) and known(fi, n) and not is_ov(fi, n),
    }

    yolo_model = YOLO(rf.YOLO_WEIGHTS_PATH)
    cache = {}
    accs = {(p["name"], s): mm.MOTAccumulator(auto_id=True) for p in plans for s in STRATA_ORDER}
    gt_maps = {p["name"]: IdMap(1_000_000) for p in plans}
    hyp_maps = {p["name"]: IdMap(10_000_000) for p in plans}
    trackers = {p["name"]: rv.make_tracker(real_w, real_h) for p in plans}
    stats = {p["name"]: defaultdict(int) for p in plans}
    n_frames = 0

    for frame_idx in frame_indices:
        xml_path = xml_dir / f"{rf.frame_name(frame_idx)}.xml"
        image_path = image_dir / f"{rf.frame_name(frame_idx)}.jpg"
        if not (image_path.exists() and xml_path.exists()):
            continue
        gt_objects = rf.parse_voc_annotation(xml_path)
        gt_per_cam_all = rf.gt_world_points_per_camera(gt_objects, poses, rf.IMAGE_WIDTH_PX)
        n_frames += 1

        for p in plans:
            rv.set_heights(p["heights"], orig_heights, model)
            key = (frame_idx, round(rf.STANDING_HEIGHT_M, 4), round(rf.LYING_HEIGHT_M, 4))
            if key not in cache:
                cache[key] = rf.yolo_world_points_per_camera_with_pose(yolo_model, image_path, poses)
            per_camera = {cid: [dict(it) for it in items] for cid, items in cache[key].items()}
            filtered = filter_by_curved_zone(rf.deduplicate_per_camera(per_camera, rf.DEDUP_IOU_THRESHOLD), polys[p["name"]])
            filtered = filter_outside_floorplan_bounds(filtered, real_w, real_h, margin_m=0.0, verbose=False)
            gt_in_zone = filter_by_curved_zone(gt_per_cam_all, polys[p["name"]])

            covs = None
            if p["kind"] == "iou":
                clusters = rb.fuse_iou_detail(filtered, adjacent_pairs, p["thr"], rf.USE_POSE_CONSTRAINT, ym.height_for)["clusters"]
            else:
                if p["kind"] == "maha":
                    covs = {(cid, i): hk.det_covariance(cid, it, ym.height_for) for cid, items in filtered.items() for i, it in enumerate(items)}
                    accepted, _ = hk.match_mahalanobis(filtered, adjacent_pairs, covs, rf.USE_POSE_CONSTRAINT)
                else:
                    accepted = rf.match_zone_pose_constrained(filtered, adjacent_pairs, rf.DIST_THR_FUSION_M, rf.USE_POSE_CONSTRAINT)
                    if isinstance(accepted, tuple) and len(accepted) == 2 and isinstance(accepted[1], int):
                        accepted = accepted[0]
                accepted = [(a[0], a[1]) for a in accepted]
                keys = [(cid, i) for cid, items in filtered.items() for i in range(len(items))]
                groups = defaultdict(list)
                if keys:
                    roots = union_find_clusters(keys, accepted)
                    for k in keys:
                        groups[roots[k]].append(k)
                clusters = list(groups.values())

            fused = fuse_points(clusters, filtered, covs)
            outputs = trackers[p["name"]].update(fused)
            tids = track_ids_for(fused, outputs)
            hyp_of = {}
            for ci_, mem in enumerate(clusters):
                for k in mem:
                    hyp_of[k] = tids[ci_] if tids[ci_] is not None else f"noid:{frame_idx}:{ci_}"
            names = rb.detection_names(gt_objects, filtered, CAM_IDS, rf.IMAGE_WIDTH_PX, exclude)

            for cid in CAM_IDS:
                n_det = len(filtered.get(cid, []))
                gt_names = [it["name"] for it in gt_in_zone.get(cid, [])]
                det_names = [names.get((cid, i)) for i in range(n_det)]
                hyp_ids = [hyp_of[(cid, i)] for i in range(n_det)]
                for s in STRATA_ORDER:
                    pr = preds[s]
                    g = [n for n in gt_names if pr(frame_idx, n)]
                    keep = [i for i, n in enumerate(det_names) if s == "全体" or (n is not None and pr(frame_idx, n))]
                    add_cam_frame(accs[(p["name"], s)], g, [det_names[i] for i in keep], [hyp_ids[i] for i in keep],
                                  gt_maps[p["name"]], hyp_maps[p["name"]])
            st = stats[p["name"]]
            st["clusters"] += len(clusters)
            st["mixed_clusters"] += sum(1 for mem in clusters if len({names[k] for k in mem if names.get(k)}) >= 2)
        rv.set_heights("orig", orig_heights, model)
        if n_frames % 20 == 0:
            print(f"  {n_frames} フレーム処理済み (直近 {rf.frame_name(frame_idx)})")

    mh = mm.metrics.create()
    rows = []
    for p in plans:
        for s in STRATA_ORDER:
            r = mh.compute(accs[(p["name"], s)], metrics=METRICS, name=p["name"]).iloc[0].to_dict()
            rows.append(dict(mode=p["name"], 層=s, **{k: float(v) for k, v in r.items()}))
    df = pd.DataFrame(rows)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_ROOT / "idf1_det_strata.csv", index=False)

    print(f"\n{'=' * 70}\n■ 検出単位の IDF1 の層別 (大きいほど良い、{n_frames} フレーム)\n{'=' * 70}")
    print(df.pivot(index="層", columns="mode", values="idf1").reindex(STRATA_ORDER)[[p["name"] for p in plans]].to_string(float_format=lambda v: f"{v:.4f}"))
    print("\n■ 層ごとの GT延べ数(検出単位) と (IDFP / IDFN)")
    for s in STRATA_ORDER:
        sub = df[df["層"] == s]
        print(f"{s:<8} GT延べ(手法別) " + "/".join(str(int(v)) for v in sub["num_objects"]) + "   "
              + "  ".join(f"{r['mode']}:{int(r['idfp'])}/{int(r['idfn'])}" for _, r in sub.iterrows()))
    print("\n誤統合クラスタ(全体): " + "  ".join(f"{p['name']}:{stats[p['name']]['mixed_clusters']}/{stats[p['name']]['clusters']}" for p in plans))
    print(f"保存: {OUT_ROOT / 'idf1_det_strata.csv'}")
    gi.report(gt_ignore_state)


if __name__ == "__main__":
    main()