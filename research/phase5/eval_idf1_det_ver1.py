"""
eval_idf1_det_ver1.py  (配置: research/phase5/)

検出単位の IDF1: カメラ間の誤統合・統合し損ねを IDFP / IDFN に反映する IDF1。

今の IDF1 は、統合後の BEV の点と GT の位置を距離(3 m 以内)で対応づけるので、誤統合の点も近くの GT に対応づいて減点されない。
ここでは、評価の単位を「各カメラの検出」、対応づけの基準を「牛の名前」に変えて、IDF1 を計算する。式は IDF1 のまま(motmetrics をそのまま使う)。

手順(1フレームごと):
  1. 各カメラの検出に、GT の牛の名前を付ける(eval_integration_ver1.py と同じ。画像上の bbox の IoU ≥ 0.3)。
  2. 統合(B / euclid / maha)でクラスタを作り、クラスタごとの統合点を ReIDBevTracker に渡して、追跡 ID を付ける(通常の評価と同じトラッカー)。
  3. クラスタの中の各検出が、その追跡 ID を引き継ぐ。
  4. 「(フレーム, カメラ)」ごとに、GT 側 = その手法の有効範囲内の GT の牛の名前、仮説側 = そのカメラの検出(持っている追跡 ID)として、
     名前が同じ検出と牛を距離 0、それ以外を対応づけ不可にして motmetrics に渡す。
  5. 牛 1 頭につき追跡 ID を 1 本選ぶ(motmetrics の全体最適割り当て)。選ばれなかった組が IDFN / IDFP になる。

減点される例:
  誤統合: 1 つの ID に牛 A と牛 B の検出が入る → ID は片方の牛にしか使えず、もう一方の牛の検出が IDFN、その ID の出力が IDFP。
  統合し損ね: 1 頭の牛の検出に ID が 2 つ付く → 片方の ID の出力が IDFP。
  同じカメラの検出が 2 つ同じクラスタに入った場合は、2 つ目を別の仮説(重複)として数える。

GT 側は、tracking の評価と同じく、各手法の有効範囲(多角形)の外の GT を除く。
検出器(YOLO)の見落としや誤検出も、IDFN / IDFP に入る(全手法で共通)。

使い方(research/ で):
    HE_POSE_CONSTRAINT=0 uv run python ./phase5/eval_idf1_det_ver1.py iou@0.10 euclid maha_h_valid
環境変数: baseline_iou_ver1.py と同じ (IOU_THRS, BASE_STYLE, HE_FRAME_*, HE_POSE_CONSTRAINT, HE_SIGMA_JSON など)
出力: research/phase5/baseline_out/idf1_det_ver1/idf1_det_all.csv
注意: 名前付けは IoU ≥ 0.3 の割り当てなので、牛が密集した場所では取り違えが起きうる(全手法で共通)。
      BEV 上の距離で評価する通常の IDF1(論文の IDF1 など)とは別の値として扱う。
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

OUT_ROOT = Path(os.environ.get("IDF1DET_OUT_DIR", os.path.join(THIS_DIR, "baseline_out", "idf1_det_ver1")))
METRICS = ["idf1", "idp", "idr", "idtp", "idfp", "idfn", "num_objects", "num_predictions"]


class IdMap:
    """motmetrics は ID を数値として扱うので、牛の名前や重複用の ID を一意な整数に直す。
    整数の追跡 ID はそのまま使い、それ以外(牛の名前、重複、ID 無し)には大きな整数を割り当てる"""

    def __init__(self, base=10_000_000):
        self.base, self.table = base, {}

    def __call__(self, key):
        if isinstance(key, (int, np.integer)):
            return int(key)
        if key not in self.table:
            self.table[key] = self.base + len(self.table)
        return self.table[key]

    def fresh(self):
        """毎回違う ID(同じカメラの検出が同じクラスタに 2 つ入ったときの、2 つ目以降の仮説用)"""
        k = ("fresh", len(self.table))
        return self(k)


def add_cam_frame(acc, gt_names, det_names, hyp_ids, gt_map, hyp_map):
    """1 つの(フレーム, カメラ)を motmetrics に渡す。
    gt_names : そのカメラで有効範囲内の GT の牛の名前のリスト
    det_names: 各検出に付いた牛の名前(付かなければ None)のリスト
    hyp_ids  : 各検出が持つ追跡 ID のリスト(同じ ID が重複したら、2 つ目以降は別の仮説にする)
    gt_map / hyp_map: IdMap(牛の名前、追跡 ID を整数に直す。全フレームで同じものを使う)"""
    names = list(dict.fromkeys(gt_names))
    seen, hids = set(), []
    for h in hyp_ids:
        if h in seen:
            hids.append(hyp_map.fresh())
        else:
            seen.add(h)
            hids.append(hyp_map(h))
    oids = [gt_map(n) for n in names]
    dist = np.full((len(oids), len(hids)), np.nan)
    for gi, o in enumerate(names):
        for hi, n in enumerate(det_names):
            if n is not None and n == o:
                dist[gi, hi] = 0.0
    acc.update(oids, hids, dist)


def fuse_points(clusters, per_camera, covs=None):
    """クラスタごとの統合点。maha は共分散の逆行列で重みづけ(hk.fuse_mahalanobis と同じ)、それ以外は平均"""
    out = []
    for mem in clusters:
        pts = np.array([per_camera[c][i]["world_xy"] for c, i in mem], dtype=np.float64)
        xy = pts.mean(axis=0)
        if covs is not None and len(mem) > 1:
            Ws = [np.linalg.inv(covs[m]) if covs.get(m) is not None else None for m in mem]
            if all(W is not None for W in Ws):
                xy = np.linalg.solve(sum(Ws), sum(W @ p for W, p in zip(Ws, pts)))
        out.append({"world_xy": xy, "n_cams": len(mem)})
    return out


def track_ids_for(fused, outputs):
    """tracker.update の戻り値 [(tid, xy)] から、各統合点の追跡 ID を求める(座標の一致で対応づけ)"""
    ids = []
    for fp in fused:
        tid = None
        for t, xy in outputs:
            if xy is fp["world_xy"] or np.array_equal(np.asarray(xy), np.asarray(fp["world_xy"])):
                tid = int(t)
                break
        ids.append(tid)
    return ids


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

    yolo_model = YOLO(rf.YOLO_WEIGHTS_PATH)
    xml_dir, image_dir = Path(rf.XML_DIR), Path(rf.IMAGE_DIR)
    frame_indices = list(range(rv.FRAME_START, rv.FRAME_END + 1, rv.FRAME_STEP))
    cache = {}
    accs = {p["name"]: mm.MOTAccumulator(auto_id=True) for p in plans}
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
                add_cam_frame(accs[p["name"]], gt_names, det_names, hyp_ids, gt_maps[p["name"]], hyp_maps[p["name"]])
            st = stats[p["name"]]
            st["clusters"] += len(clusters)
            st["mixed_clusters"] += sum(1 for mem in clusters if len({names[k] for k in mem if names.get(k)}) >= 2)
        rv.set_heights("orig", orig_heights, model)
        if n_frames % 20 == 0:
            print(f"  {n_frames} フレーム処理済み (直近 {rf.frame_name(frame_idx)})")

    mh = mm.metrics.create()
    rows = []
    for p in plans:
        r = mh.compute(accs[p["name"]], metrics=METRICS, name=p["name"]).iloc[0].to_dict()
        rows.append(dict(mode=p["name"], **{k: float(v) for k, v in r.items()},
                         誤統合クラスタ=stats[p["name"]]["mixed_clusters"], クラスタ数=stats[p["name"]]["clusters"]))
    df = pd.DataFrame(rows)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_ROOT / "idf1_det_all.csv", index=False)

    print(f"\n{'=' * 70}\n■ 検出単位の IDF1 (カメラ間の誤統合・統合し損ねを反映、{n_frames} フレーム)\n{'=' * 70}")
    show = df[["mode", "idf1", "idp", "idr", "idtp", "idfp", "idfn", "num_objects", "num_predictions", "誤統合クラスタ", "クラスタ数"]].copy()
    for c in ["idtp", "idfp", "idfn", "num_objects", "num_predictions"]:
        show[c] = show[c].astype(int)
    print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nidtp/idfp/idfn は検出(カメラ×フレーム)の数。num_objects = GT の延べ数、num_predictions = 検出の延べ数。")
    print(f"保存: {OUT_ROOT / 'idf1_det_all.csv'}")
    gi.report(gt_ignore_state)


if __name__ == "__main__":
    main()