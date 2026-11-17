"""
phase3_bev_multicam_yolo_check_epipolar_ver2.py

phase3_bev_multicam_yolo_check_epipolar.py(同じフォルダに置く)の
検出・重複除去・match_epipolar をそのまま使い、「統合後の座標」の決め方だけを
差し替えて比較する。同じ検出結果から、複数の設定(VARIANTS)を同時に評価する。

座標の決め方(2カメラのクラスタのみ):
    w = clip((交差角 - blend_min) / (blend_full - blend_min), 0, 1)
    xy = (1 - w) * 高さ仮定投影の平均 + w * 三角測量点のxy
  交差角が blend_min 未満なら w=0(従来通り)。blend_full 以上で w=1(三角測量点のみ)。
  3カメラ以上のクラスタ、1カメラのクラスタは従来通り。

assoc_min_angle(任意): 交差角がこの値未満のペアは統合自体を取り消す(有効な統合範囲を絞る)。

GTは EXCLUDE_GT_NAMES(アノテーションミスと判断した個体)を除外して評価する。

出力: OUTPUT_DIR / phase3_bev_multicam_yolo_epipolar_ver2_summary.csv(設定ごとのMOT指標)
使い方:
    python phase3_bev_multicam_yolo_check_epipolar_ver2.py
"""

from pathlib import Path
from typing import Dict, List

import numpy as np

import phase3_bev_multicam_yolo_check_epipolar as ep

EXCLUDE_GT_NAMES = {"cow022", "cow109", "cow115", "cow098"}

# 比較する設定。baselineは従来版と同じ結果になるはず(確認用)。
VARIANTS: List[dict] = [
    {"name": "baseline",            "blend_min": None, "blend_full": None, "assoc_min_angle": 0.0},
    {"name": "blend_60_90",         "blend_min": 60.0, "blend_full": 90.0, "assoc_min_angle": 0.0},
    {"name": "blend_60_90_assoc60", "blend_min": 60.0, "blend_full": 90.0, "assoc_min_angle": 60.0},
    {"name": "assoc60_only",        "blend_min": None, "blend_full": None, "assoc_min_angle": 60.0},
    # エピポーラで統合されなかった隣接カメラの検出同士も、高さ仮定の投影位置が
    # fb_merge_thr_m 以内なら統合する(距離ベースの統合を併用)。
    {"name": "baseline_fb1m",       "blend_min": None, "blend_full": None, "assoc_min_angle": 0.0,
     "fb_merge_thr_m": 1.0},
    {"name": "blend_60_90_fb1m",    "blend_min": 60.0, "blend_full": 90.0, "assoc_min_angle": 0.0,
     "fb_merge_thr_m": 1.0},
]


def fb_distance_pairs(per_camera, taken_pairs, thr: float):
    """エピポーラで未統合の検出同士を、隣接カメラ間で高さ仮定の投影位置の距離が
    thr以内ならHungarianで1対1に統合する(姿勢制約は元コードの設定に従う)。"""
    from scipy.optimize import linear_sum_assignment
    used = set()
    for k in taken_pairs:
        used.update(k)
    out = []
    for ci, cj in ep.ADJACENT_PAIRS:
        ii = [a for a in range(len(per_camera[ci])) if (ci, a) not in used]
        jj = [b for b in range(len(per_camera[cj])) if (cj, b) not in used]
        if not ii or not jj:
            continue
        cost = np.full((len(ii), len(jj)), 1e6)
        for x, a in enumerate(ii):
            for y, b in enumerate(jj):
                A, B = per_camera[ci][a], per_camera[cj][b]
                if (ep.USE_POSE_CONSTRAINT and A["pose"] is not None
                        and B["pose"] is not None and A["pose"] != B["pose"]):
                    continue
                d = float(np.linalg.norm(A["world_xy_fallback"] - B["world_xy_fallback"]))
                if d <= thr:
                    cost[x, y] = d
        for x, y in zip(*linear_sum_assignment(cost)):
            if cost[x, y] <= thr:
                out.append(((ci, ii[x]), (cj, jj[y])))
                used.add((ci, ii[x]))
                used.add((cj, jj[y]))
    return out


def ray_angle_deg(it_i: dict, it_j: dict) -> float:
    d1 = it_i["dir_world"] / np.linalg.norm(it_i["dir_world"])
    d2 = it_j["dir_world"] / np.linalg.norm(it_j["dir_world"])
    return float(np.degrees(np.arccos(np.clip(np.dot(d1, d2), -1.0, 1.0))))


def fuse_variant(per_camera, accepted, tri, var) -> List[dict]:
    angles = {}
    kept = []
    for key in accepted:
        (ci, a), (cj, b) = key
        ang = ray_angle_deg(per_camera[ci][a], per_camera[cj][b])
        angles[key] = ang
        if ang >= var["assoc_min_angle"]:
            kept.append(key)

    all_keys = [(cid, idx) for cid, items in per_camera.items() for idx in range(len(items))]
    if not all_keys:
        return []
    extra = fb_distance_pairs(per_camera, kept, var["fb_merge_thr_m"]) if var.get("fb_merge_thr_m") else []
    roots = ep.union_find_clusters(all_keys, kept + extra)
    clusters: Dict = {}
    for k in all_keys:
        clusters.setdefault(roots[k], []).append(k)
    pair_by_members = {frozenset(k): k for k in kept}

    fused = []
    for members in clusters.values():
        fb = np.array([per_camera[c][i]["world_xy_fallback"] for c, i in members])
        xy = fb.mean(axis=0)
        if var["blend_min"] is not None and len(members) == 2:
            key = pair_by_members.get(frozenset(members))
            if key is not None:
                bmin, bfull = var["blend_min"], var["blend_full"]
                w = float(np.clip((angles[key] - bmin) / (bfull - bmin), 0.0, 1.0))
                xy = (1.0 - w) * xy + w * tri[key][:2]
        fused.append({"world_xy": xy, "n_cams": len(members), "members": members})
    return fused


def main():
    import motmetrics as mm
    from ultralytics import YOLO

    cameras_json = ep.apply_camera_overrides(ep.load_floorplan_cameras(ep.FLOORPLAN_JSON_PATH))
    poses = ep.resolve_poses(cameras_json, root="Area1", fmt=ep.ROT_FMT)
    yolo_model = YOLO(ep.YOLO_WEIGHTS_PATH)

    xml_dir, image_dir = Path(ep.XML_DIR), Path(ep.IMAGE_DIR)
    states = {
        v["name"]: {
            "tracker": ep.SimpleBevTracker(ep.DIST_THR_TRACK_M, ep.MAX_MISSED_FRAMES),
            "acc": mm.MOTAccumulator(auto_id=True),
            "frames": [],
        } for v in VARIANTS
    }

    for frame_idx in range(ep.FRAME_START, ep.FRAME_END + 1, ep.FRAME_STEP):
        xml_path = xml_dir / f"{ep.frame_name(frame_idx)}.xml"
        image_path = image_dir / f"{ep.frame_name(frame_idx)}.jpg"
        if not xml_path.exists() or not image_path.exists():
            continue

        gt_objects = [o for o in ep.parse_voc_annotation(xml_path) if o["name"] not in EXCLUDE_GT_NAMES]
        gt_world = ep.gt_world_points_this_frame(gt_objects, poses, ep.IMAGE_WIDTH_PX)
        gt_names = list(gt_world.keys())
        gt_ids = [ep.name_to_gt_id(n) for n in gt_names]
        gt_points = [np.mean(np.array(gt_world[n]), axis=0) for n in gt_names]

        per_camera = ep.deduplicate_per_camera(
            ep.yolo_detections_per_camera(yolo_model, image_path, poses), ep.DEDUP_IOU_THRESHOLD
        )
        accepted, tri = ep.match_epipolar(
            per_camera, poses, ep.ADJACENT_PAIRS, ep.RAY_DIST_THR_M,
            ep.HEIGHT_RANGE_Z, ep.USE_POSE_CONSTRAINT, ep.MARGIN_GAP_M,
        )

        line = [f"  {ep.frame_name(frame_idx)}: GT={len(gt_points)} 採用ペア={len(accepted)}"]
        for var in VARIANTS:
            st = states[var["name"]]
            fused = fuse_variant(per_camera, accepted, tri, var)
            outs = st["tracker"].update(fused)
            hyp_ids = [tid for tid, _xy, _m in outs]
            hyp_points = [xy for _tid, xy, _m in outs]
            if gt_points and hyp_points:
                dist = mm.distances.norm2squared_matrix(
                    np.array(gt_points), np.array(hyp_points), max_d2=(2.0 * ep.DIST_THR_TRACK_M) ** 2)
            else:
                dist = np.empty((len(gt_points), len(hyp_points)))
            st["acc"].update(gt_ids, hyp_ids, dist)
            st["frames"].append({"gt_names": gt_names, "gt_points": gt_points,
                                 "hyp_ids": hyp_ids, "hyp_points": hyp_points})
            line.append(f"{var['name']}={len(outs)}")
        print(" ".join(line))

    mh = mm.metrics.create()
    names = [v["name"] for v in VARIANTS]
    summary = mh.compute_many(
        [states[n]["acc"] for n in names], names=names,
        metrics=["idf1", "idp", "idr", "mota", "motp", "num_switches",
                 "num_objects", "num_matches", "num_false_positives", "num_misses"],
    )
    print("\n" + mm.io.render_summary(summary, formatters=mh.formatters,
                                      namemap=mm.io.motchallenge_metric_names))

    dist_cap_m = 2.0 * ep.DIST_THR_TRACK_M
    print(f"\n=== HOTA (TrackEval, 距離上限 {dist_cap_m}m) ===")
    for n in names:
        h = ep.compute_hota_trackeval(states[n]["frames"], dist_cap_m=dist_cap_m)
        print(f"{n:>22}: HOTA={np.mean(h['HOTA']):.3f} DetA={np.mean(h['DetA']):.3f} "
              f"AssA={np.mean(h['AssA']):.3f} LocA={np.mean(h['LocA']):.3f}")

    out = ep.OUTPUT_DIR / "phase3_bev_multicam_yolo_epipolar_ver2_summary.csv"
    summary.to_csv(out)
    print(f"\nCSV保存: {out}")


if __name__ == "__main__":
    main()