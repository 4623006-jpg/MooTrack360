"""
diagnose_fn_fp_he_kai_ver1.py

diagnose_fn_fp_per_frame_ver1.py の発展版。phase3_he_kai_matching_ver2.py のモード
(euclid / euclid_h / maha / maha_h / maha_h_valid)ごとに同じ検出・融合・追跡パイプラインを走らせ、
  1) FN(MISS)/FP がどのフレームで何件か
  2) そのFN/FPが「どの段階で」起きたか(原因の推定)
  3) モードの違いで、FN/FP がどのフレームで増減したか
を出す。

原因の推定(GTごと/trackごとに、近傍 NEAR_M = 2*DIST_THR_TRACK_M [m] 以内に点があるかを段階ごとに見る):
  FN(GTを拾えなかった):
    YOLO検出なし -> 重複除去(IoU)で消えた -> カメラ有効範囲(曲線ゾーン)で除外 -> 牛舎範囲外フィルタで除外
    -> 統合で位置がずれた/誤統合 -> 追跡で落ちた -> 近くにtrackはあるが別のGTに割り当てられた
    (最初に近傍の点が無くなった段階が原因)
  FP(GTに対応しないtrack出力):
    分裂(同じ牛にtrackが複数 = 統合失敗) / 片側カメラだけ(統合失敗) / 近くのGTは未対応(競合)
    / GT近傍だがゲート外(位置ずれ) / GTなし(幽霊 or GT未アノテーション)

整合性: 全フレーム合計の FN/FP は、同じモードで phase3_he_kai_matching_ver2.py を実行したときの
        num_misses / num_false_positives と一致するはず。

使い方: python diagnose_fn_fp_he_kai_ver1.py                          # euclid と maha_h_valid を比べる
        python diagnose_fn_fp_he_kai_ver1.py euclid maha_h            # モードを指定
        python diagnose_fn_fp_he_kai_ver1.py --end 1400               # 終了フレームを指定
        python diagnose_fn_fp_he_kai_ver1.py --start 600 --end 900 --step 10 euclid maha_h
        python diagnose_fn_fp_he_kai_ver1.py --out /path/to/dir       # CSVの出力先を指定
  フレーム範囲の優先順位: 引数(--start/--end/--step) > 環境変数(HE_FRAME_START/END/STEP) > phase3モジュールの値
環境変数: HE_YOLO_MODULE (既定 phase3_bev_multicam_yolo_check_zonepose_ver6) / HE_OUT_DIR
          HE_SIGMA_JSON / HE_SIGMA_MODEL / HE_EPS_M / HE_K_SIGMA / HE_GATE_CHI2 (ver2 と同じ)
出力(既定): phase4/diagnose/analyze_check/fn_fp_per_frame_<モード>.csv  (全フレーム)
            phase4/diagnose/analyze_check/fn_fp_events_<モード>.csv     (FN/FP 1件ごとの位置・原因)
問題フレームを見るには: HE_FRAME_START=<f> HE_FRAME_END=<f> python render_bev_trajectory_he_kai_ver1.py
"""

import csv
import importlib
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESEARCH_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
PHASE4_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
MOOTRACK_CODE_DIR = (os.path.abspath(os.path.join(RESEARCH_DIR, "calibration")))
for d in (RESEARCH_DIR, PHASE4_DIR, os.path.join(PHASE4_DIR, "calibration"),
          os.path.join(PHASE4_DIR, "calibration_try"), os.path.join(PHASE4_DIR, "he_kai"),
          MOOTRACK_CODE_DIR,
          os.path.join(RESEARCH_DIR, "analyze"), os.path.join(RESEARCH_DIR, "annotate"),
          os.path.join(RESEARCH_DIR, "phase3_multi_camera"), THIS_DIR):
    if os.path.isdir(d) and d not in sys.path:
        sys.path.insert(0, d)

import phase3_he_kai_matching_ver2 as hk
from curved_zone_utils_ver1 import N_ANGLES, ZONE_HEIGHT_M, build_camera_polygons, filter_by_curved_zone, get_adjacent_pairs
from he_kai_sigma_model_ver1 import build_valid_polygons, load_model
from Mootrack_core import (
    CAM_IDS, ROT_FMT, ReIDBevTracker, apply_camera_overrides, filter_outside_floorplan_bounds,
    get_intrinsics, load_floorplan_cameras, load_floorplan_dimensions, resolve_poses,
)

DEFAULT_MODES = ["euclid", "maha_h_valid"]
YOLO_MODULE = os.environ.get("HE_YOLO_MODULE", "phase3_bev_multicam_yolo_check_zonepose_ver6")
# CSVの出力先(既定: phase4/diagnose/analyze_check。--out か 環境変数 HE_OUT_DIR で変更できる)
DEFAULT_OUT_DIR = os.environ.get("HE_OUT_DIR") or os.path.join(PHASE4_DIR, "diagnose", "analyze_check")
# このスクリプトが phase3 モジュールから使う名前(ver6 などに無い場合は実行の最初に分かるようにする)
REQUIRED_ATTRS = [
    "FRAME_START", "FRAME_END", "FRAME_STEP", "XML_DIR", "IMAGE_DIR", "YOLO_WEIGHTS_PATH", "FLOORPLAN_JSON_PATH",
    "IMAGE_WIDTH_PX", "frame_name", "parse_voc_annotation", "gt_world_points_this_frame", "name_to_gt_id",
    "yolo_world_points_per_camera_with_pose", "deduplicate_per_camera", "DEDUP_IOU_THRESHOLD", "fuse_frame_zone_pose",
    "DIST_THR_FUSION_M", "USE_POSE_CONSTRAINT", "DIST_THR_TRACK_M", "MAX_MISSED_FRAMES", "EDGE_MARGIN_M",
    "REID_DIST_THR_M", "LOST_MAX_AGE", "STANDING_HEIGHT_M", "LYING_HEIGHT_M", "height_for",
]
SLIGHTLY_OUT_M = 3.0     # GT近傍だがゲート外、とみなす上限距離


def flat(per_camera):
    return [np.asarray(it["world_xy"], dtype=np.float64) for items in per_camera.values() for it in items]


def nearest(p, pts):
    return float(np.min(np.linalg.norm(np.asarray(pts) - p, axis=1))) if len(pts) else float("inf")


def fn_reason(d, near):
    if d["raw"] > near:
        return "YOLO検出なし"
    if d["dedup"] > near:
        return "重複除去(IoU)で消えた"
    if d["zone"] > near:
        return "カメラ有効範囲(曲線ゾーン)で除外"
    if d["bounds"] > near:
        return "牛舎範囲外フィルタで除外"
    if d["fused"] > near:
        return "統合で位置がずれた/誤統合"
    if d["track"] > near:
        return "追跡で落ちた(ID未発行/見失い)"
    return "近くにtrackあり(別のGTに割当)"


def fp_reason(h, matched_gt_idx, near):
    if h["d_gt"] <= near and h["gt_idx"] in matched_gt_idx:
        return "片側カメラだけ(統合失敗)" if h["n_cams"] == 1 else "分裂: 同じ牛にtrackが複数"
    if h["d_gt"] <= near:
        return "近くのGTは未対応(競合)"
    if h["d_gt"] <= SLIGHTLY_OUT_M:
        return "GT近傍だがゲート外(位置ずれ)"
    return "GTなし(幽霊 or GT未アノテーション)"


FALLTHROUGH = "近くにtrackあり(別のGTに割当)"
MERGE_M = 1.2            # 『2頭が1本のtrackに統合』とみなす、trackからそれぞれのGTまでの距離の上限
CAND_M = 4.0             # 統合されなかった理由を調べるとき、相手カメラの検出を探す範囲


def refine_fn(gg, ctx, hid_to_oid, oid_to_gt):
    """『近くにtrackあり』のFNを、実際にその近くのtrackがどのGTに割り当てられているかで細分する"""
    hyps = ctx["hyps"]
    if not hyps:
        return FALLTHROUGH, {}
    dists = [float(np.linalg.norm(h["xy"] - gg["xy"])) for h in hyps]
    k = int(np.argmin(dists))
    h, d_self = hyps[k], dists[k]
    extra = dict(near_track=h["tid"], near_track_dist=round(d_self, 2), near_track_n_cams=h["n_cams"])
    oid = hid_to_oid.get(h["tid"])
    if oid is None:
        return "近くのtrackはどのGTにも未対応(割当ゲート/最適割当の問題)", extra
    other = ctx["gts"][oid_to_gt[oid]]
    d_other = float(np.linalg.norm(h["xy"] - other["xy"]))
    gg_gg = float(np.linalg.norm(gg["xy"] - other["xy"]))
    extra.update(near_track_matched_to=other["name"], near_track_to_other_gt=round(d_other, 2), gt_gt_dist=round(gg_gg, 2))
    if d_self <= MERGE_M and d_other <= MERGE_M:
        return "2頭が1本のtrackに統合(誤統合の疑い)", extra
    return "自分のtrackが無い(隣のGTのtrackが2m以内にあるだけ)", extra


def find_item(bounds, xy, tol=1e-3):
    for cid, items in bounds.items():
        for it in items:
            if np.linalg.norm(np.asarray(it["world_xy"]) - xy) < tol:
                return cid, it
    return None, None


def unmerged_reason(m, ym, cid, it, bounds, adj_set):
    """片側カメラだけのtrack(FP)が、なぜ他カメラの検出と統合されなかったかを、最も近い他カメラの検出で説明する"""
    best = None
    for cj, items in bounds.items():
        if cj == cid:
            continue
        for jt in items:
            euc = float(np.linalg.norm(np.asarray(it["world_xy"]) - np.asarray(jt["world_xy"])))
            if euc <= CAND_M and (best is None or euc < best[0]):
                best = (euc, cj, jt)
    if best is None:
        return f"相手カメラに{CAND_M:.0f}m以内の検出なし", {}
    euc, cj, jt = best
    adj = (cid, cj) in adj_set or (cj, cid) in adj_set
    mism = it["pose"] is not None and jt["pose"] is not None and it["pose"] != jt["pose"]
    info = dict(partner_cam=cj, partner_dist=round(euc, 2), partner_adjacent=adj, pose_a=it["pose"], pose_b=jt["pose"])
    if not adj:
        return f"最も近い検出は隣接でないカメラ({cid}-{cj}, {euc:.2f}m)", info
    if mism and ym.USE_POSE_CONSTRAINT:
        return f"姿勢クラスの食い違いで拒否({it['pose']} vs {jt['pose']}, {euc:.2f}m)", info
    if hk.MODES[m]["match"] == "maha":
        Sa, Sb = hk.det_covariance(cid, it, ym.height_for), hk.det_covariance(cj, jt, ym.height_for)
        if Sa is None or Sb is None:
            return "共分散を計算できない(投影が無効)", info
        d = np.asarray(it["world_xy"]) - np.asarray(jt["world_xy"])
        m2 = float(d @ np.linalg.solve(Sa + Sb, d))
        info["partner_d2"] = round(m2, 2)
        if euc > hk.MAX_EUCLID_M:
            return f"ユークリッド上限({hk.MAX_EUCLID_M}m)超え({euc:.2f}m)", info
        if m2 > hk.GATE_CHI2:
            return f"マハラノビスのゲート外(d²={m2:.1f} > {hk.GATE_CHI2}, {euc:.2f}m)", info
    elif euc > ym.DIST_THR_FUSION_M:
        return f"ユークリッドのゲート外({euc:.2f}m > {ym.DIST_THR_FUSION_M}m)", info
    return f"ゲートも姿勢も満たす(他の検出とのペアとの競合, {euc:.2f}m)", info


def main():
    import motmetrics as mm
    from ultralytics import YOLO

    import argparse
    ap = argparse.ArgumentParser(description="FN/FPの診断(モードごと)")
    ap.add_argument("modes", nargs="*", help=f"モード(既定 {DEFAULT_MODES}): {list(hk.MODES)}")
    ap.add_argument("--start", type=int, default=None, help="開始フレーム(既定: phase3モジュールのFRAME_START)")
    ap.add_argument("--end", type=int, default=None, help="終了フレーム(既定: phase3モジュールのFRAME_END)")
    ap.add_argument("--step", type=int, default=None, help="刻み(既定: phase3モジュールのFRAME_STEP)")
    ap.add_argument("--out", default=None, help=f"CSVの出力先(既定 {DEFAULT_OUT_DIR})")
    args = ap.parse_args()
    modes = args.modes or DEFAULT_MODES
    bad = [m for m in modes if m not in hk.MODES]
    if bad:
        raise SystemExit(f"不明なモード: {bad} / 使えるモード: {list(hk.MODES)}")
    ym = importlib.import_module(YOLO_MODULE)
    missing = [n for n in REQUIRED_ATTRS if not hasattr(ym, n)]
    if missing:
        raise SystemExit(f"{YOLO_MODULE} に次の名前がありません: {missing}\n"
                         f"(HE_YOLO_MODULE で別のphase3モジュールを指定するか、名前を確認してください)")
    if "HE_POSE_CONSTRAINT" in os.environ:                 # 姿勢別の統合制約 1=ON / 0=OFF
        ym.USE_POSE_CONSTRAINT = os.environ["HE_POSE_CONSTRAINT"] == "1"
    print(f"姿勢別統合制約: {'ON' if ym.USE_POSE_CONSTRAINT else 'OFF'}")
    # GTミス(『BBOXのラベル間違いや位置ミス』)のBBOXを評価のGTから除く。無効にするには HE_GT_IGNORE=0
    import gt_ignore_ver1 as gi
    gt_ignore_state = gi.apply(ym)
    # 優先順位: コマンドライン引数 > 環境変数 > phase3モジュールの値
    frame_start = args.start if args.start is not None else int(os.environ.get("HE_FRAME_START", ym.FRAME_START))
    frame_end = args.end if args.end is not None else int(os.environ.get("HE_FRAME_END", ym.FRAME_END))
    frame_step = args.step if args.step is not None else int(os.environ.get("HE_FRAME_STEP", ym.FRAME_STEP))

    cameras_json = load_floorplan_cameras(ym.FLOORPLAN_JSON_PATH)
    real_w, real_h = load_floorplan_dimensions(ym.FLOORPLAN_JSON_PATH)
    poses = resolve_poses(apply_camera_overrides(cameras_json), root="Area1", fmt=ROT_FMT)
    json_path = os.environ.get("HE_SIGMA_JSON") or str(Path(ym.OUTPUT_DIR) / "bev_calibration_refined" / "sigma_he_kai_ver1.json")
    model = load_model(json_path)
    print(f"phase3モジュール: {ym.__name__} / フレーム {frame_start}〜{frame_end} 刻み {frame_step}")
    print(model.describe())
    hk._MODEL, hk._YM = model, ym
    hk.setup_poses(poses)
    orig_heights = (ym.STANDING_HEIGHT_M, ym.LYING_HEIGHT_M)

    polygons = {}
    for m in modes:
        if hk.MODES[m]["polygons"] == "he_kai":
            print(f"[{m}] HE(改)の有効範囲 eps={hk.EPS_VALID_M}m k={hk.K_SIGMA}")
            polygons[m] = build_valid_polygons(poses, CAM_IDS, get_intrinsics, model, hk.EPS_VALID_M, hk.K_SIGMA,
                                               height_m=ZONE_HEIGHT_M, n_angles=N_ANGLES)
        else:
            print(f"[{m}] 既存の曲線ゾーン")
            polygons[m] = build_camera_polygons(poses, CAM_IDS, get_intrinsics, height_m=ZONE_HEIGHT_M, n_angles=N_ANGLES)
    adjacent_pairs = get_adjacent_pairs(CAM_IDS)

    yolo_model = YOLO(ym.YOLO_WEIGHTS_PATH)
    near = 2.0 * ym.DIST_THR_TRACK_M                      # 評価のGT-track対応付けのゲートと同じ
    trackers = {m: ReIDBevTracker(dist_thr=ym.DIST_THR_TRACK_M, max_missed=ym.MAX_MISSED_FRAMES, real_w=real_w, real_h=real_h,
                                  edge_margin_m=ym.EDGE_MARGIN_M, reid_dist_thr=ym.REID_DIST_THR_M,
                                  lost_max_age=ym.LOST_MAX_AGE) for m in modes}
    accs = {m: mm.MOTAccumulator(auto_id=True) for m in modes}
    ctxs = {m: [] for m in modes}
    xml_dir, image_dir = Path(ym.XML_DIR), Path(ym.IMAGE_DIR)

    n_done = 0
    for frame_idx in range(frame_start, frame_end + 1, frame_step):
        xml_path = xml_dir / f"{ym.frame_name(frame_idx)}.xml"
        image_path = image_dir / f"{ym.frame_name(frame_idx)}.jpg"
        if not xml_path.exists() or not image_path.exists():
            continue
        gt_objects = ym.parse_voc_annotation(xml_path)
        raw_cache = {}
        for m in modes:
            cfg = hk.MODES[m]
            if cfg["heights"] == "fit":
                ym.STANDING_HEIGHT_M, ym.LYING_HEIGHT_M = model.h_eff("standing"), model.h_eff("lying")
            else:
                ym.STANDING_HEIGHT_M, ym.LYING_HEIGHT_M = orig_heights
            gt_world = ym.gt_world_points_this_frame(gt_objects, poses, ym.IMAGE_WIDTH_PX)
            gt_names = list(gt_world.keys())
            gt_ids = [ym.name_to_gt_id(n) for n in gt_names]
            gt_pts = [np.mean(np.array(gt_world[n]), axis=0) for n in gt_names]

            if cfg["heights"] not in raw_cache:
                raw_cache[cfg["heights"]] = ym.yolo_world_points_per_camera_with_pose(yolo_model, image_path, poses)
            raw = {cid: [dict(it) for it in items] for cid, items in raw_cache[cfg["heights"]].items()}
            deduped = ym.deduplicate_per_camera(raw, ym.DEDUP_IOU_THRESHOLD)
            zone = filter_by_curved_zone(deduped, polygons[m])
            bounds = filter_outside_floorplan_bounds(zone, real_w, real_h, margin_m=0.0, verbose=False)
            if cfg["match"] == "maha":
                fused = hk.fuse_mahalanobis(bounds, adjacent_pairs, ym.DIST_THR_FUSION_M, ym.USE_POSE_CONSTRAINT)[0]
            else:
                fused = ym.fuse_frame_zone_pose(bounds, adjacent_pairs, ym.DIST_THR_FUSION_M, ym.USE_POSE_CONSTRAINT)[0]
            track_outputs = trackers[m].update(fused)
            hyp_ids = [tid for tid, _ in track_outputs]
            hyp_pts = [np.asarray(xy, dtype=np.float64) for _, xy in track_outputs]

            stages = dict(raw=flat(raw), dedup=flat(deduped), zone=flat(zone), bounds=flat(bounds),
                          fused=[np.asarray(f["world_xy"], dtype=np.float64) for f in fused], track=hyp_pts)
            gts = [dict(name=gt_names[k], id=gt_ids[k], xy=gt_pts[k], n_cam=len(gt_world[gt_names[k]]),
                        **{"d": {s: nearest(gt_pts[k], p) for s, p in stages.items()}}) for k in range(len(gt_names))]
            hyps = []
            for k, (tid, xy) in enumerate(zip(hyp_ids, hyp_pts)):
                dg = [float(np.linalg.norm(xy - g)) for g in gt_pts]
                n_cams = next((f["n_cams"] for f in fused if np.linalg.norm(np.asarray(f["world_xy"]) - xy) < 1e-3), None)
                hyps.append(dict(tid=tid, xy=xy, n_cams=n_cams, d_gt=min(dg) if dg else float("inf"),
                                 gt_idx=int(np.argmin(dg)) if dg else -1))
            if gt_pts and hyp_pts:
                dist = mm.distances.norm2squared_matrix(np.array(gt_pts), np.array(hyp_pts), max_d2=near ** 2)
            else:
                dist = np.empty((len(gt_pts), len(hyp_pts)))
            accs[m].update(gt_ids, hyp_ids, dist)
            ctxs[m].append(dict(frame_idx=frame_idx, gts=gts, hyps=hyps, bounds=bounds))
        ym.STANDING_HEIGHT_M, ym.LYING_HEIGHT_M = orig_heights
        n_done += 1
        if n_done % 20 == 0:
            print(f"  {n_done} フレーム処理済み (直近 {ym.frame_name(frame_idx)})")
    print(f"\n処理フレーム数: {n_done}")

    out_dir = Path(args.out or DEFAULT_OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"CSVの出力先: {out_dir.resolve()}")
    per_frame, all_events = {}, {}
    adj_set = set(adjacent_pairs)
    for m in modes:
        # 共分散(det_covariance)は想定高さに依存するので、そのモードの想定高さに戻して計算する
        if hk.MODES[m]["heights"] == "fit":
            ym.STANDING_HEIGHT_M, ym.LYING_HEIGHT_M = model.h_eff("standing"), model.h_eff("lying")
        else:
            ym.STANDING_HEIGHT_M, ym.LYING_HEIGHT_M = orig_heights
        ev = accs[m].mot_events
        groups = {fid: g for fid, g in ev.groupby(level="FrameId")}
        rows, events = [], []
        for seq, ctx in enumerate(ctxs[m]):
            g = groups.get(seq)
            miss_ids = set(g[g["Type"] == "MISS"]["OId"]) if g is not None else set()
            fp_hids = set(g[g["Type"] == "FP"]["HId"]) if g is not None else set()
            matched_hid = {}
            if g is not None:
                for _, e in g[g["Type"].isin(["MATCH", "SWITCH"])].iterrows():
                    matched_hid[e["OId"]] = e["HId"]
            gt_id_to_idx = {gg["id"]: k for k, gg in enumerate(ctx["gts"])}
            matched_gt_idx = {gt_id_to_idx[o] for o in matched_hid if o in gt_id_to_idx}
            hid_to_oid = {h_: o for o, h_ in matched_hid.items()}
            n_fn = n_fp = 0
            for oid in miss_ids:
                gg = ctx["gts"][gt_id_to_idx[oid]]
                reason, extra = fn_reason(gg["d"], near), {}
                if reason == FALLTHROUGH:
                    reason, extra = refine_fn(gg, ctx, hid_to_oid, gt_id_to_idx)
                events.append(dict(frame=ctx["frame_idx"], type="FN", name=gg["name"], x=round(float(gg["xy"][0]), 2),
                                   y=round(float(gg["xy"][1]), 2), n_cams=gg["n_cam"], reason=reason,
                                   **{f"d_{s}": round(v, 2) for s, v in gg["d"].items()}, **extra))
                n_fn += 1
            for h in ctx["hyps"]:
                if h["tid"] in fp_hids:
                    reason, extra = fp_reason(h, matched_gt_idx, near), {}
                    if reason.startswith("片側カメラだけ"):
                        cid, it = find_item(ctx["bounds"], h["xy"])
                        if it is not None:
                            why, info = unmerged_reason(m, ym, cid, it, ctx["bounds"], adj_set)
                            extra = dict(cam=cid, unmerged=why, **info)
                    events.append(dict(frame=ctx["frame_idx"], type="FP", name=f"track{h['tid']}", x=round(float(h["xy"][0]), 2),
                                       y=round(float(h["xy"][1]), 2), n_cams=h["n_cams"], reason=reason,
                                       d_gt=round(h["d_gt"], 2), **extra))
                    n_fp += 1
            rows.append(dict(frame_idx=ctx["frame_idx"], n_gt=len(ctx["gts"]), n_hyp=len(ctx["hyps"]), n_fn=n_fn, n_fp=n_fp))
        per_frame[m], all_events[m] = rows, events
        ym.STANDING_HEIGHT_M, ym.LYING_HEIGHT_M = orig_heights
        for name, data in ((f"fn_fp_per_frame_{m}.csv", rows), (f"fn_fp_events_{m}.csv", events)):
            if data:
                keys = list({k: None for r in data for k in r})
                with open(out_dir / name, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=keys)
                    w.writeheader()
                    w.writerows(data)

    # ---- 表示 ----
    for m in modes:
        rows, events = per_frame[m], all_events[m]
        print(f"\n{'=' * 78}\n■ モード {m}  {hk.MODES[m]}\n{'=' * 78}")
        print(f"合計: FN={sum(r['n_fn'] for r in rows)}  FP={sum(r['n_fp'] for r in rows)}  (GT延べ {sum(r['n_gt'] for r in rows)})")
        for typ in ("FN", "FP"):
            c = Counter(e["reason"] for e in events if e["type"] == typ)
            if c:
                print(f"  {typ} の原因:")
                for reason, n in c.most_common():
                    print(f"    {n:>4}  {reason}")
        why = Counter(e["unmerged"].split("(")[0] for e in events if e["type"] == "FP" and "unmerged" in e)
        if why:
            print("  FP『片側カメラだけ』が統合されなかった理由(最も近い他カメラの検出で判定):")
            for reason, n in why.most_common():
                print(f"    {n:>4}  {reason}")
            for e in [e for e in events if e["type"] == "FP" and "unmerged" in e][:8]:
                print(f"       frame {e['frame']:>5} {e['name']:<9} {e['cam']}: {e['unmerged']}")
        fn_sub = [e for e in events if e["type"] == "FN" and "near_track_dist" in e]
        if fn_sub:
            print("  FN『近くにtrackあり』の内訳の距離(m)  [自GT-track / track-別GT / GT-GT]:")
            for reason in sorted({e["reason"] for e in fn_sub}):
                es = [e for e in fn_sub if e["reason"] == reason]
                f3 = lambda key: (f"{np.median([e[key] for e in es if key in e]):.2f}" if any(key in e for e in es) else "-")
                print(f"    {len(es):>3}件 {reason}: {f3('near_track_dist')} / {f3('near_track_to_other_gt')} / {f3('gt_gt_dist')}")
            close = sum(1 for e in fn_sub if e.get("gt_gt_dist", 9) <= 1.0)
            print(f"    (GTどうしが1m以内の近接ペア: {close}/{len(fn_sub)}件)")
        bad_rows = sorted([r for r in rows if r["n_fn"] or r["n_fp"]], key=lambda r: -(r["n_fn"] + r["n_fp"]))
        print(f"  FN/FPのあるフレーム: {len(bad_rows)}/{len(rows)}   (上位10)")
        print(f"    {'frame':>6} {'GT':>4} {'hyp':>4} {'FN':>3} {'FP':>3}")
        for r in bad_rows[:10]:
            print(f"    {r['frame_idx']:>6} {r['n_gt']:>4} {r['n_hyp']:>4} {r['n_fn']:>3} {r['n_fp']:>3}")

    if len(modes) >= 2:
        print(f"\n{'=' * 78}\n■ モード間でFN/FPが違うフレーム\n{'=' * 78}")
        head = f"{'frame':>6}" + "".join(f"{m + ' FN/FP':>22}" for m in modes)
        print(head)
        n_diff = 0
        for k in range(len(per_frame[modes[0]])):
            vals = [(per_frame[m][k]["n_fn"], per_frame[m][k]["n_fp"]) for m in modes]
            if len(set(vals)) > 1:
                n_diff += 1
                print(f"{per_frame[modes[0]][k]['frame_idx']:>6}" + "".join(f"{f'{a}/{b}':>22}" for a, b in vals))
        if n_diff == 0:
            print("(全フレームで同じ)")
    print(f"\nCSV: {out_dir}")
    gi.report(gt_ignore_state)
    print("※ 原因は『GTの近傍(%.1fm)に点があるか』を段階ごとに見た推定。GT未アノテーションの牛は『GTなし』のFPになる。" % near)


if __name__ == "__main__":
    main()