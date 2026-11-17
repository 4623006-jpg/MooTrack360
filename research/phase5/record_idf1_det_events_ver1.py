"""
record_idf1_det_events_ver1.py  (配置: research/phase5/)

検出単位の IDF1(eval_idf1_det_ver5.py と同じ処理)で IDFP / IDFN になった「検出」と「GT の牛」を、
フレーム・カメラ・bbox・理由つきで記録する。画像を作る処理は render_idf1_det_events_ver1.py(記録を読んで強調画像を出す)。

やること:
  1. ver5 と同じ流れ(YOLO → 有効範囲で絞る → 統合 → 追跡 ID → 名前付け)を回し、(フレーム, カメラ) ごとの記録を全部ためる。
  2. 全フレームを見て、IDF1 と同じ基準で「牛 ↔ ID」の 1 対 1 の対応を決める(TP の数が最大になる割り当て、ハンガリー法)。
  3. その対応を使って、各 (フレーム, カメラ) の検出を TP / IDFP、GT の牛を TP / IDFN に分け、理由を付ける。
  4. 検算: 自分で数えた idtp / idfp / idfn が、motmetrics の値と一致するか確認して表示する(一致しなければ警告)。

理由(code):
  IDFP(検出側)
    NONAME_NOGT   名前なし。そのカメラの GT bbox と重ならない(IoU = 0)
    NONAME_LOWIOU 名前なし。GT と少し重なるが IoU 0.3 未満(bbox のずれ、牛の一部、画像の端など)
    NONAME_TAKEN  名前なし。IoU 0.3 以上の GT があるが、その GT は別の検出に割り当て済み(二重検出)
    DUP           同じクラスタに同じカメラの検出が複数あり、2 つ目以降(重複)
    GT_OUT_ZONE   名前はあるが、その牛の GT が有効範囲の外(検出だけが範囲内)
    OTHER_COW_ID  この ID は別の牛に対応づけられた(誤統合の疑い)
    SPLIT_ID      この牛は別の ID に対応づけられた(統合し損ね・ID の切替)
    NO_ID_COW     この牛にはどの ID も対応づけられなかった
  IDFN(GT 側)
    MISSED        未検出。GT bbox と重なる検出がない
    LOWIOU        重なる検出はあるが IoU 0.3 未満で、名前が付かなかった
    TAKEN         IoU 0.3 以上の検出はあるが、別の GT に割り当て済みで名前が付かなかった
    MERGE_ID      検出はあるが、その ID は別の牛に対応づけられている(誤統合の疑い)
    SPLIT_ID      検出はあるが、牛の対応 ID と違う ID が付いている(統合し損ね・ID の切替)
    NO_ID_COW     検出はあるが、この牛にはどの ID も対応づけられなかった
  IDFN には「他のカメラでは検出できていたか(seen_other_cam)」も付ける。

使い方(research/ で。ver5 と同じ引数・環境変数):
    HE_POSE_CONSTRAINT=0 uv run python ./phase5/record_idf1_det_events_ver1.py iou@0.10 euclid maha_h_valid
環境変数: ver5 と同じ(IDF1DET_RANGE: px_var 既定 / he_kai / per_mode, IDF1DET_ORIG_AS など)
  IDF1DET_EVENTS_DIR で出力先を変えられる。
出力: baseline_out/idf1_det_events_ver1_<range>/
    events_<mode>.json  メタ情報(画像フォルダ、タイル幅など)と、イベントの一覧
    events_<mode>.csv   同じ内容の表(フレーム、カメラ、種別、理由、牛、ID、bbox など)
    summary.csv         手法 × 種別 × 理由の件数
"""

import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import baseline_iou_ver1 as bl
import render_bev_baseline_ver1 as rb
from eval_idf1_det_ver1 import IdMap, add_cam_frame, fuse_points, track_ids_for
from eval_idf1_det_ver5 import COMMON_RANGE, ORIG_AS, PxOnlyModel

OUT_ROOT = Path(os.environ.get("IDF1DET_EVENTS_DIR", os.path.join(THIS_DIR, "baseline_out", f"idf1_det_events_ver1_{COMMON_RANGE}")))

REASON_JA = {
    "NONAME_NOGT": "名前なし: GT と重ならない(近くの GT の牛も無い)",
    "NONAME_GT_ELSEWHERE": "名前なし: このカメラに GT は無いが、近くに GT の牛がいる(別のカメラ側にだけ GT がある疑い)",
    "NONAME_LOWIOU": "名前なし: IoU 不足(0.3 未満)",
    "NONAME_TAKEN": "名前なし: GT は別の検出に割り当て済み(二重検出)",
    "DUP": "同じクラスタ内の同じカメラの重複",
    "GT_OUT_ZONE": "名前はあるが、その牛の GT が有効範囲の外",
    "OTHER_COW_ID": "この ID は別の牛に対応(誤統合の疑い)",
    "SPLIT_ID": "牛は別の ID に対応(統合し損ね・ID の切替)",
    "NO_ID_COW": "この牛にはどの ID も対応しなかった",
    "MISSED": "未検出(重なる検出なし)",
    "LOWIOU": "IoU 不足で名前が付かなかった",
    "TAKEN": "検出は別の GT に割り当て済みで名前が付かなかった",
    "MERGE_ID": "検出の ID は別の牛に対応(誤統合の疑い)",
}


NEAR_GT_M = float(os.environ.get("NEAR_GT_M", 1.5))     # 名前なし検出の床座標が、GT の牛の床座標からこの距離以内なら「近くに GT の牛がいる」とみなす [m]


def _f(v):
    return None if v is None else float(v)


def nearest_gt_world(xy, gt_world):
    """検出の床座標 xy に最も近い GT の牛(全カメラの GT の牛ごとの平均位置)と距離 [m]"""
    best, nm = float("inf"), None
    for n, p in gt_world.items():
        d = float(np.hypot(xy[0] - p[0], xy[1] - p[1]))
        if d < best:
            best, nm = d, n
    return nm, best


def gt_boxes_by_cam(gt_objects, cam_ids, tile_w, exclude):
    """{cam_id: [(名前, [x0, y0, x1, y1](タイル内座標))]}。名前付け(rb.detection_names)と同じ振り分け"""
    out = {cid: [] for cid in cam_ids}
    for o in gt_objects:
        if o["name"] in exclude:
            continue
        cxm = (o["xmin"] + o["xmax"]) / 2.0
        ci = max(0, min(int(cxm // tile_w), len(cam_ids) - 1))
        out[cam_ids[ci]].append((o["name"], [o["xmin"] - ci * tile_w, o["ymin"], o["xmax"] - ci * tile_w, o["ymax"]]))
    return out


def best_iou(box, gt_list):
    """box と、同じカメラの GT bbox との最大 IoU と、その GT の名前"""
    best, nm = 0.0, None
    for n, g in gt_list:
        v = rb._iou_box(box, g)
        if v > best:
            best, nm = v, n
    return best, nm


def solve_assignment(records):
    """IDF1 と同じ基準の「牛 ↔ ID」の 1 対 1 対応(同時に出た検出の数が最大になる割り当て)。(牛 → ID, ID → 牛)"""
    from scipy.optimize import linear_sum_assignment
    cnt = Counter()
    for r in records:
        for d in r["det"]:
            if d["name"] is not None and d["name"] in r["zone"] and not d["dup"]:
                cnt[(d["name"], d["hyp"])] += 1
    names = sorted({k[0] for k in cnt}, key=str)
    hyps = sorted({k[1] for k in cnt}, key=str)
    if not names:
        return {}, {}
    ni, hi = {n: i for i, n in enumerate(names)}, {h: i for i, h in enumerate(hyps)}
    m = np.zeros((len(names), len(hyps)))
    for (n, h), c in cnt.items():
        m[ni[n], hi[h]] = c
    rows, cols = linear_sum_assignment(-m)
    cow2id, id2cow = {}, {}
    for r_, c_ in zip(rows, cols):
        if m[r_, c_] > 0:
            cow2id[names[r_]] = hyps[c_]
            id2cow[hyps[c_]] = names[r_]
    return cow2id, id2cow


def label_events(records, cow2id, id2cow):
    """各 (フレーム, カメラ) の記録を TP / IDFP / IDFN に分けて、理由を付ける。(events, 件数)"""
    events, tot, tiles = [], Counter(), {}
    named_by_frame = defaultdict(lambda: defaultdict(set))     # フレーム → カメラ → その検出に付いた名前の集合(他のカメラで検出できていたかの判定用)
    for r in records:
        for d in r["det"]:
            if d["name"] is not None:
                named_by_frame[r["frame"]][r["cam"]].add(d["name"])
    for r in records:
        gt_list = r["gt"]
        tp_names = set()
        n_before = len(events)
        tp_dets = set()
        for d in r["det"]:
            n, h = d["name"], d["hyp"]
            ok = n is not None and n in r["zone"] and not d["dup"] and cow2id.get(n) == h
            if ok:
                tp_dets.add(d["i"])
                tot["idtp"] += 1
                tp_names.add(n)
                continue
            tot["idfp"] += 1
            iou, near = best_iou(d["bbox"], gt_list)
            owner = id2cow.get(h)
            near_name, near_m = nearest_gt_world(d["xy"], r["gt_world"]) if r["gt_world"] else (None, float("inf"))
            if n is None:
                if iou >= rb.NAME_IOU:
                    code = "NONAME_TAKEN"
                elif iou > 0:
                    code = "NONAME_LOWIOU"
                else:
                    code = "NONAME_GT_ELSEWHERE" if near_m <= NEAR_GT_M else "NONAME_NOGT"
            elif d["dup"]:
                code = "DUP"
            elif n not in r["zone"]:
                code = "GT_OUT_ZONE"
            elif owner is not None and owner != n:
                code = "OTHER_COW_ID"
            elif n in cow2id:
                code = "SPLIT_ID"
            else:
                code = "NO_ID_COW"
            events.append(dict(
                kind="IDFP", reason=code, reason_ja=REASON_JA[code], frame=r["frame"], frame_name=r["frame_name"], cam=r["cam"], cam_idx=r["cam_idx"],
                name=n, hyp=h, owner=owner, det_idx=d["i"], bbox=d["bbox"], pose=d["pose"], conf=d["conf"], xy=d["xy"],
                best_iou=round(iou, 3), near_gt=near, near_gt_world=near_name, near_gt_world_m=(None if near_name is None else round(near_m, 2)),
                cluster=d["cluster"], cluster_members=d["members"], cluster_names=d["cluster_names"],
                mixed_cluster=len(set(d["cluster_names"])) >= 2))
        gt_boxes = {n: b for n, b in gt_list}
        for n in sorted(r["zone"], key=str):
            if n in tp_names:
                continue
            tot["idfn"] += 1
            box = gt_boxes.get(n)
            dets_n = [d for d in r["det"] if d["name"] == n]
            iou, _ = best_iou(box, [(d["i"], d["bbox"]) for d in r["det"]]) if (box is not None and r["det"]) else (0.0, None)
            det_i, hyp = None, None
            if dets_n:
                d0 = next((d for d in dets_n if not d["dup"]), dets_n[0])
                det_i, hyp = d0["i"], d0["hyp"]
                owner = id2cow.get(hyp)
                code = "MERGE_ID" if (owner is not None and owner != n) else ("SPLIT_ID" if n in cow2id else "NO_ID_COW")
            else:
                code = "TAKEN" if iou >= rb.NAME_IOU else ("LOWIOU" if iou > 0 else "MISSED")
            seen_other = any(n in names_ for c, names_ in named_by_frame[r["frame"]].items() if c != r["cam"])
            events.append(dict(
                kind="IDFN", reason=code, reason_ja=REASON_JA[code], frame=r["frame"], frame_name=r["frame_name"], cam=r["cam"], cam_idx=r["cam_idx"],
                name=n, hyp=hyp, owner=id2cow.get(hyp) if hyp is not None else None, det_idx=det_i, bbox=box, assigned_id=cow2id.get(n),
                best_iou=round(iou, 3), seen_other_cam=bool(seen_other)))
        if len(events) > n_before:      # イベントのあったタイルだけ、検出と GT の全体像(TP も含む)を残す。画像の灰色・緑の枠に使う
            tiles[f"{r['frame']}|{r['cam']}"] = dict(
                det=[dict(i=d["i"], bbox=d["bbox"], name=d["name"], hyp=d["hyp"], tp=d["i"] in tp_dets) for d in r["det"]],
                gt=[dict(name=n_, bbox=b_, in_zone=n_ in r["zone"], tp=n_ in tp_names) for n_, b_ in gt_list])
    return events, tot, tiles


def main():
    import motmetrics as mm
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
    print(f"姿勢別統合制約: {'ON' if rf.USE_POSE_CONSTRAINT else 'OFF'} / フレーム {rv.FRAME_START}〜{rv.FRAME_END} 刻み {rv.FRAME_STEP} / 範囲: {COMMON_RANGE}")

    import gt_ignore_ver1 as gi
    gt_ignore_state = gi.apply(ym)

    os.environ.setdefault("HE_SIGMA_MODEL", "const")
    json_path = os.environ.get("HE_SIGMA_JSON") or str(Path(rf.OUTPUT_DIR) / "bev_calibration_refined" / "sigma_he_kai_ver4.json")
    model = load_model(json_path)

    regions, cameras_json, real_w, real_h = load_floorplan(rf.FLOORPLAN_JSON_PATH)
    poses = resolve_poses(apply_camera_overrides(cameras_json), root="Area1", fmt=ROT_FMT)
    hk._YM, hk._MODEL = ym, model
    hk.setup_poses(poses)
    bl.MAP_WH = (real_w, real_h)
    orig_heights = (rf.STANDING_HEIGHT_M, rf.LYING_HEIGHT_M)

    adjacent_pairs = get_adjacent_pairs(CAM_IDS)
    poly_cache = {}

    def get_poly(kind):
        if kind not in poly_cache:
            if kind == "orig":
                poly_cache[kind] = build_camera_polygons(poses, CAM_IDS, get_intrinsics, height_m=ZONE_HEIGHT_M, n_angles=N_ANGLES)
            elif kind == "he_kai":
                poly_cache[kind] = build_valid_polygons(poses, CAM_IDS, get_intrinsics, model, hk.EPS_VALID_M, hk.K_SIGMA,
                                                        height_m=ZONE_HEIGHT_M, n_angles=N_ANGLES)
            elif kind == "px_var":
                poly_cache[kind] = build_valid_polygons(poses, CAM_IDS, get_intrinsics, PxOnlyModel(model), hk.EPS_VALID_M, hk.K_SIGMA,
                                                        height_m=ZONE_HEIGHT_M, n_angles=N_ANGLES)
            else:
                raise SystemExit(f"有効範囲 {kind} は非対応")
        return poly_cache[kind]

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
        if COMMON_RANGE in ("he_kai", "px_var"):
            polys[p["name"]] = get_poly(COMMON_RANGE)
        elif p["polygons"] in ("he_kai", "orig"):
            polys[p["name"]] = get_poly("he_kai" if p["polygons"] == "he_kai" else ORIG_AS)
        else:
            raise SystemExit(f"{p['name']}: 有効範囲 {p['polygons']} は非対応")

    xml_dir, image_dir = Path(rf.XML_DIR), Path(rf.IMAGE_DIR)
    frame_indices = list(range(rv.FRAME_START, rv.FRAME_END + 1, rv.FRAME_STEP))
    tile_w = rf.IMAGE_WIDTH_PX / len(CAM_IDS)

    yolo_model = YOLO(rf.YOLO_WEIGHTS_PATH)
    cache = {}
    accs = {p["name"]: mm.MOTAccumulator(auto_id=True) for p in plans}
    gt_maps = {p["name"]: IdMap(1_000_000) for p in plans}
    hyp_maps = {p["name"]: IdMap(10_000_000) for p in plans}
    trackers = {p["name"]: rv.make_tracker(real_w, real_h) for p in plans}
    records = {p["name"]: [] for p in plans}
    n_frames = 0

    for frame_idx in frame_indices:
        xml_path = xml_dir / f"{rf.frame_name(frame_idx)}.xml"
        image_path = image_dir / f"{rf.frame_name(frame_idx)}.jpg"
        if not (image_path.exists() and xml_path.exists()):
            continue
        gt_objects = rf.parse_voc_annotation(xml_path)
        gt_per_cam_all = rf.gt_world_points_per_camera(gt_objects, poses, rf.IMAGE_WIDTH_PX)
        gt_boxes = gt_boxes_by_cam(gt_objects, CAM_IDS, tile_w, exclude)
        _acc = defaultdict(list)
        for _items in gt_per_cam_all.values():
            for _it in _items:
                if _it["name"] not in exclude:
                    _acc[_it["name"]].append(np.asarray(_it["world_xy"], dtype=float))
        gt_world = {n_: [float(v) for v in np.mean(ps, axis=0)] for n_, ps in _acc.items()}      # 全カメラの GT から作った、牛ごとの床座標(評価の GT と同じ作り方)
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
            hyp_of, cl_of = {}, {}
            for ci_, mem in enumerate(clusters):
                for k in mem:
                    hyp_of[k] = tids[ci_] if tids[ci_] is not None else f"noid:{frame_idx}:{ci_}"
                    cl_of[k] = ci_
            names = rb.detection_names(gt_objects, filtered, CAM_IDS, rf.IMAGE_WIDTH_PX, exclude)

            for ci, cid in enumerate(CAM_IDS):
                items = filtered.get(cid, [])
                n_det = len(items)
                gt_names = [it["name"] for it in gt_in_zone.get(cid, [])]
                det_names = [names.get((cid, i)) for i in range(n_det)]
                hyp_ids = [hyp_of[(cid, i)] for i in range(n_det)]
                add_cam_frame(accs[p["name"]], gt_names, det_names, hyp_ids, gt_maps[p["name"]], hyp_maps[p["name"]])

                seen, dets = set(), []
                for i in range(n_det):
                    h = hyp_ids[i]
                    dup = h in seen
                    seen.add(h)
                    mem = clusters[cl_of[(cid, i)]]
                    dets.append(dict(
                        i=i, bbox=[float(v) for v in items[i]["bbox_local"]], pose=items[i].get("pose"), conf=_f(items[i].get("conf")),
                        name=det_names[i], hyp=h, dup=dup, cluster=cl_of[(cid, i)], members=[[c, j] for c, j in mem],
                        cluster_names=[names.get(m_) for m_ in mem if names.get(m_) is not None],
                        xy=[float(v) for v in np.asarray(items[i]["world_xy"], dtype=float)]))
                records[p["name"]].append(dict(
                    frame=frame_idx, frame_name=rf.frame_name(frame_idx), cam=cid, cam_idx=ci,
                    gt=gt_boxes[cid], zone=set(gt_names), gt_world=gt_world, det=dets))
        rv.set_heights("orig", orig_heights, model)
        if n_frames % 20 == 0:
            print(f"  {n_frames} フレーム処理済み (直近 {rf.frame_name(frame_idx)})")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    mh = mm.metrics.create()
    summary_rows = []
    for p in plans:
        nm = p["name"]
        cow2id, id2cow = solve_assignment(records[nm])
        events, tot, tiles = label_events(records[nm], cow2id, id2cow)
        ref = mh.compute(accs[nm], metrics=["idtp", "idfp", "idfn"], name=nm).iloc[0].to_dict()
        ok = all(int(ref[k]) == tot[k] for k in ("idtp", "idfp", "idfn"))
        print(f"\n[{nm}] 検算: 自分の集計 idtp/idfp/idfn = {tot['idtp']}/{tot['idfp']}/{tot['idfn']}  motmetrics = "
              f"{int(ref['idtp'])}/{int(ref['idfp'])}/{int(ref['idfn'])}  → {'一致' if ok else '不一致(割り当ての違いか、重複の扱いを確認)'}")
        cnt = Counter((e["kind"], e["reason"]) for e in events)
        for (kind, reason), c in sorted(cnt.items(), key=lambda kv: (kv[0][0], -kv[1])):
            print(f"  {kind} {reason:<14} {c:>6}  {REASON_JA[reason]}")
            summary_rows.append(dict(mode=nm, kind=kind, reason=reason, count=c, reason_ja=REASON_JA[reason]))
        idfn_miss = [e for e in events if e["kind"] == "IDFN" and e["reason"] in ("MISSED", "LOWIOU", "TAKEN")]
        if idfn_miss:
            seen_n = sum(1 for e in idfn_miss if e["seen_other_cam"])
            print(f"  IDFN のうち検出なし/名前なし {len(idfn_miss)} 件の内、他のカメラでは検出できていた: {seen_n} 件")

        safe = nm.replace("@", "_")
        meta = dict(mode=nm, range=COMMON_RANGE, image_dir=str(image_dir), tile_w=tile_w, cam_ids=list(CAM_IDS),
                    image_width=float(rf.IMAGE_WIDTH_PX), totals=dict(tot), reasons=REASON_JA)
        with open(OUT_ROOT / f"events_{safe}.json", "w", encoding="utf-8") as fp:
            json.dump(dict(meta=meta, events=events, tiles=tiles), fp, ensure_ascii=False)
        cols = ["kind", "reason", "reason_ja", "frame", "frame_name", "cam", "name", "hyp", "owner", "assigned_id", "det_idx", "best_iou",
                "near_gt", "near_gt_world", "near_gt_world_m", "seen_other_cam", "mixed_cluster", "bbox", "xy"]
        with open(OUT_ROOT / f"events_{safe}.csv", "w", newline="", encoding="utf-8-sig") as fp:
            w = csv.DictWriter(fp, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for e in events:
                w.writerow({k: (json.dumps(e[k]) if isinstance(e.get(k), (list, dict)) else e.get(k)) for k in cols})
    with open(OUT_ROOT / "summary.csv", "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.DictWriter(fp, fieldnames=["mode", "kind", "reason", "count", "reason_ja"])
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\n保存: {OUT_ROOT}")
    gi.report(gt_ignore_state)


if __name__ == "__main__":
    main()