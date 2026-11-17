"""
grid_search_overlap_half_width_zonepose.py

phase3_bev_multicam_yolo_check_zonepose.pyのOVERLAP_HALF_WIDTH_M
(隣接カメラ境界線の片側の幅[m]。合計の重なり幅は2倍になる)をグリッド
サーチするスクリプト。

【OVERLAP_HALF_WIDTH_Mとは】
隣接カメラの境界(2台のカメラのworld x座標の中点)から片側にこの幅だけ
広げた範囲を「両カメラの重なり範囲」とみなし、各カメラの「有効範囲」
(=そのカメラの検出として採用してよいx座標の範囲)を決める
(get_adjacent_zones/get_camera_valid_ranges)。この幅が狭すぎると、
境界付近の同一個体が統合を試みる前に有効範囲外として片側だけ消えてしまい
統合漏れ(FN)の原因になる。逆に広すぎると、無関係な奥の個体まで重なり
範囲に入ってきて誤統合(過剰統合、FN)の原因になりうる。

【このスクリプトでやること】
YOLO検出・重複除去・GT座標の計算(重い処理)はOVERLAP_HALF_WIDTH_Mの値に
依存しないため、フレームごとに1回だけ計算してキャッシュする。
OVERLAP_HALF_WIDTH_Mの各候補値について、有効範囲フィルタ
(filter_out_of_range_detections)+統合(fuse_frame_zone_pose)+
追跡(SimpleBevTracker)+MOTメトリクス計算だけをやり直す。

--- 使い方 ---
    python grid_search_overlap_half_width_zonepose.py
"""

import os
import sys
import csv as csv_module
from pathlib import Path
from typing import List

import numpy as np

MOOTRACK_CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calibration")
if MOOTRACK_CODE_DIR not in sys.path:
    sys.path.insert(0, MOOTRACK_CODE_DIR)

# 同じ出力フォルダにあるphase3_bev_multicam_yolo_check_zonepose.pyから
# 必要な関数・定数をそのまま再利用する(ロジックの二重管理を避けるため)。
from phase3_bev_multicam_yolo_check_zonepose import (
    XML_DIR, IMAGE_DIR, FLOORPLAN_JSON_PATH, OUTPUT_DIR,
    YOLO_WEIGHTS_PATH, FRAME_START, FRAME_END, FRAME_STEP, IMAGE_WIDTH_PX,
    DIST_THR_FUSION_M, DIST_THR_TRACK_M, MAX_MISSED_FRAMES,
    USE_POSE_CONSTRAINT, DEDUP_IOU_THRESHOLD,
    frame_name, parse_voc_annotation, gt_world_points_this_frame, name_to_gt_id,
    yolo_world_points_per_camera_with_pose, deduplicate_per_camera,
    get_adjacent_zones, get_camera_valid_ranges, filter_out_of_range_detections,
    fuse_frame_zone_pose, SimpleBevTracker, compute_hota_trackeval,
)
from Mootrack_core import (
    CAM_IDS, ROT_FMT,
    load_floorplan_cameras, apply_camera_overrides, resolve_poses,
)

# グリッドサーチする境界片側幅の候補[m](合計の重なり幅はこの2倍)
OVERLAP_HALF_WIDTH_GRID: List[float] = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


def build_frame_cache(poses) -> List[dict]:
    """YOLO検出・重複除去・GT座標など、OVERLAP_HALF_WIDTH_Mに依存しない
    重い処理をフレームごとに1回だけ計算してキャッシュする。"""
    from ultralytics import YOLO

    print(f"YOLOモデル読み込み中: {YOLO_WEIGHTS_PATH}")
    yolo_model = YOLO(YOLO_WEIGHTS_PATH)

    xml_dir = Path(XML_DIR)
    image_dir = Path(IMAGE_DIR)
    frame_indices = list(range(FRAME_START, FRAME_END + 1, FRAME_STEP))

    cache: List[dict] = []
    for frame_idx in frame_indices:
        xml_path = xml_dir / f"{frame_name(frame_idx)}.xml"
        image_path = image_dir / f"{frame_name(frame_idx)}.jpg"
        if not xml_path.exists() or not image_path.exists():
            continue

        gt_objects = parse_voc_annotation(xml_path)
        gt_world = gt_world_points_this_frame(gt_objects, poses, IMAGE_WIDTH_PX)
        gt_names = list(gt_world.keys())
        gt_ids = [name_to_gt_id(name) for name in gt_names]
        gt_points = [np.mean(np.array(gt_world[name]), axis=0) for name in gt_names]

        per_camera_raw = yolo_world_points_per_camera_with_pose(yolo_model, image_path, poses)
        per_camera_deduped = deduplicate_per_camera(per_camera_raw, DEDUP_IOU_THRESHOLD)

        cache.append({
            "frame_idx": frame_idx,
            "gt_names": gt_names,
            "gt_ids": gt_ids,
            "gt_points": gt_points,
            "per_camera_deduped": per_camera_deduped,
        })
        print(f"  キャッシュ済み: {frame_name(frame_idx)} "
              f"GT個体={len(gt_points)} YOLO検出(重複除去後)="
              f"{sum(len(v) for v in per_camera_deduped.values())}")

    return cache


def run_one_config(frame_cache: List[dict], poses, half_width: float) -> dict:
    import motmetrics as mm

    zones = get_adjacent_zones(poses, CAM_IDS, half_width)
    valid_ranges = get_camera_valid_ranges(CAM_IDS, zones)

    tracker = SimpleBevTracker(dist_thr=DIST_THR_TRACK_M, max_missed=MAX_MISSED_FRAMES)
    acc = mm.MOTAccumulator(auto_id=True)
    frames_data: List[dict] = []

    for fr in frame_cache:
        per_camera_filtered = filter_out_of_range_detections(fr["per_camera_deduped"], valid_ranges)

        fused_points, _n_pose_rejected = fuse_frame_zone_pose(
            per_camera_filtered, CAM_IDS, zones, DIST_THR_FUSION_M, USE_POSE_CONSTRAINT
        )
        track_outputs = tracker.update(fused_points)
        hyp_ids = [tid for tid, _xy in track_outputs]
        hyp_points = [xy for _tid, xy in track_outputs]

        gt_points = fr["gt_points"]
        if gt_points and hyp_points:
            dist = mm.distances.norm2squared_matrix(
                np.array(gt_points), np.array(hyp_points), max_d2=(2.0 * DIST_THR_TRACK_M) ** 2
            )
        else:
            dist = np.empty((len(gt_points), len(hyp_points)))

        acc.update(fr["gt_ids"], hyp_ids, dist)
        frames_data.append({
            "gt_names": fr["gt_names"], "gt_points": gt_points,
            "hyp_ids": hyp_ids, "hyp_points": hyp_points,
        })

    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=["idf1", "idp", "idr", "mota", "motp", "num_switches",
                 "num_objects", "num_matches", "num_false_positives", "num_misses"],
        name=str(half_width),
    )
    row = summary.iloc[0]

    dist_cap_m = 2.0 * DIST_THR_TRACK_M
    hota_res = compute_hota_trackeval(frames_data, dist_cap_m=dist_cap_m)

    return {
        "overlap_half_width_m": half_width,
        "idf1": row["idf1"], "idp": row["idp"], "idr": row["idr"],
        "mota": row["mota"], "motp": row["motp"],
        "num_switches": row["num_switches"],
        "num_objects": row["num_objects"], "num_matches": row["num_matches"],
        "num_false_positives": row["num_false_positives"], "num_misses": row["num_misses"],
        "hota": float(np.mean(hota_res["HOTA"])),
        "deta": float(np.mean(hota_res["DetA"])),
        "assa": float(np.mean(hota_res["AssA"])),
    }


def main():
    print("floorplan.json 読み込み中...")
    cameras_json = load_floorplan_cameras(FLOORPLAN_JSON_PATH)
    cameras_json = apply_camera_overrides(cameras_json)
    poses = resolve_poses(cameras_json, root="Area1", fmt=ROT_FMT)
    for cid in CAM_IDS:
        print(f"  {cid}: t={poses[cid].t.round(2)}")

    print(f"\n境界片側幅の候補: {OVERLAP_HALF_WIDTH_GRID}\n")

    frame_cache = build_frame_cache(poses)
    print(f"\nキャッシュしたフレーム数: {len(frame_cache)}\n")

    results = []
    for half_width in OVERLAP_HALF_WIDTH_GRID:
        print(f"--- OVERLAP_HALF_WIDTH_M={half_width} で評価中 ---")
        res = run_one_config(frame_cache, poses, half_width)
        results.append(res)
        print(f"  IDF1={res['idf1']*100:.1f}% IDP={res['idp']*100:.1f}% IDR={res['idr']*100:.1f}% "
              f"MOTA={res['mota']*100:.1f}% MOTP={res['motp']:.3f} IDs={res['num_switches']} "
              f"FP={res['num_false_positives']} FN={res['num_misses']} HOTA={res['hota']:.3f}")

    csv_path = OUTPUT_DIR / "grid_search_overlap_half_width_zonepose.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv_module.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        for res in results:
            writer.writerow(res)
    print(f"\nCSV保存: {csv_path}")


if __name__ == "__main__":
    main()