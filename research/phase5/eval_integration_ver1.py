"""
eval_integration_ver1.py  (配置: research/phase5/)

カメラ間の「統合」そのものの精度を、追跡(トラッカー・motmetrics)を通さずに、フレームごとに直接数える。
追跡の FN/FP は、誤統合を見逃すことがある(誤統合しても、両方の牛の近く(3 m 以内)に出力が残れば FN も FP も出ない)ため。

3つの段階(混ぜない):
  検出(フレーム) : 誤検出・見落とし            <- このスクリプトの対象外
  統合(カメラ間) : 誤統合 / 統合し損ね          <- このスクリプトの対象
  追跡(時間)     : ID の入れ替わりなど          <- 従来通り(IDF1, SW)

数え方(1フレームごとに数えて、全フレームを合計する):
  各カメラの検出に、GT の牛の名前を付ける(検出 bbox と GT bbox の IoU ≥ 0.3。σ の推定と同じ。名前が付かない検出は数えない)。
  TP  : 本来同じ牛(同じ名前)の、隣接カメラの検出ペアが、同じクラスタに入った
  FN  : 本来同じ牛なのに、別のクラスタに分かれた(統合し損ね。追跡では重複 FP になりやすい)
  FP  : 別の牛(名前が違う)の検出が、同じクラスタに入った(誤統合。クラスタ内の全ペアを数える。連鎖で入ったものも含む)
  適合率 = TP/(TP+FP)  再現率 = TP/(TP+FN)  F1
  ほかに、誤統合のあったクラスタの数と割合、誤統合のあったフレームの割合。
  層別: 牛の位置(GT)が、2台以上のカメラの有効範囲(HE のみの円)に入っているか(重なり範囲 / 単眼範囲)。FP ペアは、どちらかの牛が重なり範囲なら重なり範囲。

対象のモード: iou / iou@<しきい値> (論文方式 B。BASE_STYLE=paper が既定)、euclid, euclid_h, maha, maha_h, maha_h_valid など(zone/gate 系は非対応)。
  iou は IOU_THRS で掃引。評価は baseline_iou_ver1.py / eval_stratified_ver1.py と同じ検出・校正・フレーム範囲・姿勢制約。

使い方(research/ で):
    HE_POSE_CONSTRAINT=0 uv run python ./phase5/eval_integration_ver1.py iou@0.2 euclid maha_h_valid
環境変数: baseline_iou_ver1.py と同じ (IOU_THRS, BASE_STYLE, BASE_HEIGHTS, HE_FRAME_*, HE_POSE_CONSTRAINT, HE_SIGMA_JSON など)
出力: research/phase5/baseline_out/integration_ver1/integration_all.csv
注意: 名前の付け方(IoU ≥ 0.3)は、牛が密集した場所で隣の牛と取り違えることがある。全手法で同じ名前を使うので、比較には影響しにくいが、絶対値は参考。
"""

import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import baseline_iou_ver1 as bl
import render_bev_baseline_ver1 as rb          # 名前付け・クラスタの状態判定・B の統合(途中経過つき)を借りる

OUT_ROOT = Path(os.environ.get("INT_OUT_DIR", os.path.join(THIS_DIR, "baseline_out", "integration_ver1")))
STRATA = ("全体", "重なり範囲", "単眼範囲")


def count_pairs(clusters, names, overlap_of, adjacent):
    """クラスタ(キーのリスト)と、検出の名前 {キー: 牛の名前} から、層ごとの TP/FN/FP(ペア数)を返す。
    adjacent: 隣接カメラの組の集合(frozenset)。同じ牛のペアは、隣接カメラのものだけ数える。"""
    root = {k: ci for ci, mem in enumerate(clusters) for k in mem}
    named = [k for k in root if names.get(k)]
    res = {s: Counter() for s in STRATA}

    def add(kind, ov):
        res["全体"][kind] += 1
        res["重なり範囲" if ov else "単眼範囲"][kind] += 1

    for i in range(len(named)):
        for j in range(i + 1, len(named)):
            ka, kb = named[i], named[j]
            na, nb = names[ka], names[kb]
            if na == nb:
                if ka[0] == kb[0] or frozenset((ka[0], kb[0])) not in adjacent:
                    continue
                add("tp" if root[ka] == root[kb] else "fn", overlap_of.get(na, False))
            elif root[ka] == root[kb]:
                add("fp", overlap_of.get(na, False) or overlap_of.get(nb, False))
    return res


def clusters_from_accepted(per_camera, accepted):
    from Mootrack_core import union_find_clusters
    keys = [(cid, i) for cid, items in per_camera.items() for i in range(len(items))]
    if not keys:
        return []
    accepted = [(a[0], a[1]) for a in accepted]          # maha は (キー, キー, 距離...) の形で返すことがある
    roots = union_find_clusters(keys, accepted)
    groups = defaultdict(list)
    for k in keys:
        groups[roots[k]].append(k)
    return list(groups.values())


def prf(c):
    tp, fn, fp = c["tp"], c["fn"], c["fp"]
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * p * r / (p + r) if p == p and r == r and p + r > 0 else float("nan")
    return p, r, f


def main():
    import pandas as pd
    from ultralytics import YOLO

    jobs = bl.parse_args(sys.argv[1:] or ["iou@0.2", "euclid", "maha_h_valid"])
    import render_bev_trajectory_he_kai_ver6 as rv
    rf, ym, hk = rv.rf, rv.ym, bl.hk              # hk は baseline_iou_ver1 と同じもの(評価と同じ phase3_he_kai_matching)
    from Mootrack_core import (CAM_IDS, ROT_FMT, apply_camera_overrides, filter_outside_floorplan_bounds, get_intrinsics,
                               load_floorplan, resolve_poses)
    from curved_zone_utils_ver1 import (N_ANGLES, ZONE_HEIGHT_M, build_camera_polygons, filter_by_curved_zone, get_adjacent_pairs,
                                        point_in_polygon)
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
    adjacent = {frozenset(p) for p in adjacent_pairs}
    poly_orig = build_camera_polygons(poses, CAM_IDS, get_intrinsics, height_m=ZONE_HEIGHT_M, n_angles=N_ANGLES)   # 層分け用(全モード共通)
    poly_he = None
    exclude = getattr(ym, "EXCLUDE_GT_NAMES", set())

    # モードごとの設定
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
    tot = {p["name"]: {s: Counter() for s in STRATA} for p in plans}
    ext = {p["name"]: Counter() for p in plans}            # 誤統合クラスタ数、統合クラスタ数、フレーム数、誤統合のあったフレーム数、名前付き検出数、検出数
    n_frames = 0

    for frame_idx in frame_indices:
        xml_path = xml_dir / f"{rf.frame_name(frame_idx)}.xml"
        image_path = image_dir / f"{rf.frame_name(frame_idx)}.jpg"
        if not (image_path.exists() and xml_path.exists()):
            continue
        gt_objects = rf.parse_voc_annotation(xml_path)
        gt_per_cam = rf.gt_world_points_per_camera(gt_objects, poses, rf.IMAGE_WIDTH_PX)
        by_cow = defaultdict(list)
        for c, items in gt_per_cam.items():
            for it in items:
                by_cow[it["name"]].append(np.asarray(it["world_xy"], dtype=float))
        overlap_of = {n: sum(point_in_polygon(np.mean(v, axis=0), poly_orig[c]) for c in CAM_IDS) >= 2 for n, v in by_cow.items()}
        n_frames += 1

        for p in plans:
            rv.set_heights(p["heights"], orig_heights, model)
            key = (frame_idx, round(rf.STANDING_HEIGHT_M, 4), round(rf.LYING_HEIGHT_M, 4))
            if key not in cache:
                cache[key] = rf.yolo_world_points_per_camera_with_pose(yolo_model, image_path, poses)
            per_camera = {cid: [dict(it) for it in items] for cid, items in cache[key].items()}
            filtered = filter_by_curved_zone(rf.deduplicate_per_camera(per_camera, rf.DEDUP_IOU_THRESHOLD), polys[p["name"]])
            filtered = filter_outside_floorplan_bounds(filtered, real_w, real_h, margin_m=0.0, verbose=False)

            if p["kind"] == "iou":
                bl._IOU_THR = p["thr"]
                clusters = rb.fuse_iou_detail(filtered, adjacent_pairs, p["thr"], rf.USE_POSE_CONSTRAINT, ym.height_for)["clusters"]
            elif p["kind"] == "maha":
                covs = {(cid, i): hk.det_covariance(cid, it, ym.height_for) for cid, items in filtered.items() for i, it in enumerate(items)}
                accepted, _ = hk.match_mahalanobis(filtered, adjacent_pairs, covs, rf.USE_POSE_CONSTRAINT)
                clusters = clusters_from_accepted(filtered, accepted)
            else:                                                      # euclid
                accepted = rf.match_zone_pose_constrained(filtered, adjacent_pairs, rf.DIST_THR_FUSION_M, rf.USE_POSE_CONSTRAINT)
                if isinstance(accepted, tuple) and len(accepted) == 2 and isinstance(accepted[1], int):
                    accepted = accepted[0]                    # phase3 版は (ペア, 姿勢で除外した数) を返す
                clusters = clusters_from_accepted(filtered, accepted)

            names = rb.detection_names(gt_objects, filtered, CAM_IDS, rf.IMAGE_WIDTH_PX, exclude)
            res = count_pairs(clusters, names, overlap_of, adjacent)
            for s in STRATA:
                tot[p["name"]][s].update(res[s])
            states = rb.classify_clusters(clusters, names)
            e = ext[p["name"]]
            e["frames"] += 1
            e["false_merge_clusters"] += states.count("false_merge")
            e["merged_clusters"] += sum(1 for st, mem in zip(states, clusters) if len(mem) >= 2)
            e["frames_with_false_merge"] += int("false_merge" in states)
            e["dets"] += sum(len(v) for v in filtered.values())
            e["named_dets"] += len(names)
        rv.set_heights("orig", orig_heights, model)
        if n_frames % 20 == 0:
            print(f"  {n_frames} フレーム処理済み (直近 {rf.frame_name(frame_idx)})")

    rows = []
    for p in plans:
        for s in STRATA:
            c = tot[p["name"]][s]
            pr, rc, f1 = prf(c)
            rows.append(dict(mode=p["name"], 層=s, TP=c["tp"], FN_統合し損ね=c["fn"], FP_誤統合=c["fp"], 適合率=pr, 再現率=rc, F1=f1))
    df = pd.DataFrame(rows)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_ROOT / "integration_all.csv", index=False)

    print(f"\n{'=' * 70}\n■ 統合の精度(カメラ間ペア単位、{n_frames} フレームの合計)\n{'=' * 70}")
    for s in STRATA:
        print(f"\n[{s}]")
        print(df[df["層"] == s].drop(columns=["層"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n■ 誤統合クラスタ(2検出以上のクラスタのうち、別の牛が入っているもの)")
    print(f"{'mode':<16}{'誤統合クラスタ':>14}{'統合クラスタ':>12}{'割合%':>8}{'誤統合のあったフレーム%':>22}{'名前付き検出%':>14}")
    for p in plans:
        e = ext[p["name"]]
        print(f"{p['name']:<16}{e['false_merge_clusters']:>14}{e['merged_clusters']:>12}{e['false_merge_clusters'] / max(e['merged_clusters'], 1) * 100:>8.1f}"
              f"{e['frames_with_false_merge'] / max(e['frames'], 1) * 100:>22.1f}{e['named_dets'] / max(e['dets'], 1) * 100:>14.1f}")
    print(f"\n保存: {OUT_ROOT / 'integration_all.csv'}")
    gi.report(gt_ignore_state)


if __name__ == "__main__":
    main()