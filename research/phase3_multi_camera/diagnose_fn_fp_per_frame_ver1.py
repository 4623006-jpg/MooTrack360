"""
diagnose_fn_fp_per_frame_ver1.py

phase3_bev_multicam_yolo_check_zonepose_ver1.py(曲線境界版)と全く同じ検出・融合・
追跡パイプラインを実行し、motmetrics.MOTAccumulatorが内部で記録している
フレームごとのイベント(mot_events)から、FN(MISS=GTを検出できなかった)と
FP(=GTに対応しない余分なtrack出力)がどのフレームで何件発生しているかを
一覧にする。

phase3側のmain()は全フレームを1つのMOTAccumulatorに積んでから最終的な
IDF1/MOTA/HOTAのサマリだけを表示するため、「全体でFN=32件」までは分かっても
「具体的にどのフレームで何件」までは分からなかった。本スクリプトはこれを
フレーム単位に分解し、FNまたはFPが発生しているフレームの一覧をCSVと
標準出力に出す。問題フレームが分かれば、render_bev_frames_yolo_zonepose_ver1.py
が出力する {frame_name}.png (bev_frames_yolo_zonepose_curved/ 以下)を直接開いて
目視確認できる。

判定ロジック(distance-gateによるGT-track対応付け)はphase3_ver1のmain()内で
使っているmm.distances.norm2squared_matrix + acc.update()と完全に同じものを
再利用しているため、ここで出るFN/FP件数の合計は、phase3_ver1を実行したときの
サマリのnum_misses/num_false_positivesと一致するはず(整合性チェックとしても使える)。

出力:
  - 標準出力: FNまたはFPが発生したフレームの一覧(フレーム番号・GT数・
    track出力数・FN件数・FP件数)
  - CSV: OUTPUT_DIR/fn_fp_per_frame_curved_{posecon|noposecon}.csv
    (全フレーム分、FN=FP=0の行も含む)

--- 使い方 ---
    python diagnose_fn_fp_per_frame_ver1.py
"""

import csv
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

MOOTRACK_CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calibration")
if MOOTRACK_CODE_DIR not in sys.path:
    sys.path.insert(0, MOOTRACK_CODE_DIR)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

ANALYZE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analyze")
if ANALYZE_DIR not in sys.path:
    sys.path.insert(0, ANALYZE_DIR)

from Mootrack_core import (
    CAM_IDS, ROT_FMT, get_intrinsics,
    load_floorplan_cameras, load_floorplan_dimensions, apply_camera_overrides, resolve_poses,
    filter_outside_floorplan_bounds,
    ReIDBevTracker,
)

from curved_zone_utils_ver1 import (
    ZONE_HEIGHT_M, N_ANGLES,
    build_camera_polygons, filter_by_curved_zone, get_adjacent_pairs,
)

# phase3_bev_multicam_yolo_check_zonepose_ver1.py(曲線境界版)側の検出・融合ロジック・
# 設定値をそのまま再利用(ロジックの二重管理を避けるため)
from phase3_bev_multicam_yolo_check_zonepose_ver4 import (
    FLOORPLAN_JSON_PATH, OUTPUT_DIR, XML_DIR, IMAGE_DIR,
    YOLO_WEIGHTS_PATH, IMAGE_WIDTH_PX,
    FRAME_START, FRAME_END, FRAME_STEP,
    DIST_THR_FUSION_M, DIST_THR_TRACK_M, MAX_MISSED_FRAMES,
    EDGE_MARGIN_M, REID_DIST_THR_M, LOST_MAX_AGE,
    USE_POSE_CONSTRAINT, DEDUP_IOU_THRESHOLD,
    frame_name, parse_voc_annotation, gt_world_points_this_frame, name_to_gt_id,
    yolo_world_points_per_camera_with_pose, deduplicate_per_camera,
    fuse_frame_zone_pose,
)


def main():
    import motmetrics as mm
    from ultralytics import YOLO

    print("floorplan.json 読み込み中...")
    cameras_json = load_floorplan_cameras(FLOORPLAN_JSON_PATH)
    real_w, real_h = load_floorplan_dimensions(FLOORPLAN_JSON_PATH)
    cameras_json = apply_camera_overrides(cameras_json)
    poses = resolve_poses(cameras_json, root="Area1", fmt=ROT_FMT)

    print(f"各カメラの曲線境界(許容半径R_max)を計算中(高さ仮定={ZONE_HEIGHT_M}m)...")
    cam_polygons = build_camera_polygons(poses, CAM_IDS, get_intrinsics, height_m=ZONE_HEIGHT_M, n_angles=N_ANGLES)
    adjacent_pairs = get_adjacent_pairs(CAM_IDS)

    print(f"YOLOモデル読み込み中: {YOLO_WEIGHTS_PATH}")
    yolo_model = YOLO(YOLO_WEIGHTS_PATH)

    xml_dir = Path(XML_DIR)
    image_dir = Path(IMAGE_DIR)
    frame_indices = list(range(FRAME_START, FRAME_END + 1, FRAME_STEP))

    tracker = ReIDBevTracker(
        dist_thr=DIST_THR_TRACK_M, max_missed=MAX_MISSED_FRAMES,
        real_w=real_w, real_h=real_h, edge_margin_m=EDGE_MARGIN_M,
        reid_dist_thr=REID_DIST_THR_M, lost_max_age=LOST_MAX_AGE,
    )
    acc = mm.MOTAccumulator(auto_id=True)

    # acc.update()を呼んだ順番(sequential FrameId)と、実際のフレーム番号・
    # そのフレームのGT数/track出力数を対応付けるためのリスト
    frame_order: List[dict] = []

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
        per_camera_filtered = filter_by_curved_zone(per_camera_deduped, cam_polygons)
        per_camera_filtered = filter_outside_floorplan_bounds(
            per_camera_filtered, real_w, real_h, margin_m=0.0, verbose=False
        )

        fused_points, _n_pose_rejected = fuse_frame_zone_pose(
            per_camera_filtered, adjacent_pairs, DIST_THR_FUSION_M, USE_POSE_CONSTRAINT
        )

        track_outputs = tracker.update(fused_points)
        hyp_ids = [tid for tid, _xy in track_outputs]
        hyp_points = [xy for _tid, xy in track_outputs]

        if gt_points and hyp_points:
            dist = mm.distances.norm2squared_matrix(
                np.array(gt_points), np.array(hyp_points), max_d2=(2.0 * DIST_THR_TRACK_M) ** 2
            )
        else:
            dist = np.empty((len(gt_points), len(hyp_points)))

        acc.update(gt_ids, hyp_ids, dist)
        frame_order.append({
            "frame_idx": frame_idx, "n_gt": len(gt_points), "n_hyp": len(hyp_points),
        })

        if len(frame_order) % 50 == 0:
            print(f"  {len(frame_order)}/{len(frame_indices)} フレーム処理済み(直近: {frame_name(frame_idx)})")

    print(f"\n処理フレーム数: {len(frame_order)}/{len(frame_indices)}")

    # mot_eventsはacc.update()を呼んだ順のsequential FrameId(0,1,2,...)で
    # 記録されているので、frame_order[FrameId]で実際のフレーム番号に対応付ける
    events = acc.mot_events
    miss_counts = events[events["Type"] == "MISS"].groupby(level="FrameId").size()
    fp_counts = events[events["Type"] == "FP"].groupby(level="FrameId").size()

    rows = []
    for seq_id, info in enumerate(frame_order):
        n_fn = int(miss_counts.get(seq_id, 0))
        n_fp = int(fp_counts.get(seq_id, 0))
        rows.append({
            "frame_idx": info["frame_idx"],
            "frame_name": frame_name(info["frame_idx"]),
            "n_gt": info["n_gt"],
            "n_hyp": info["n_hyp"],
            "n_fn": n_fn,
            "n_fp": n_fp,
        })

    total_fn = sum(r["n_fn"] for r in rows)
    total_fp = sum(r["n_fp"] for r in rows)
    print(f"\n全フレーム合計: FN(MISS)={total_fn}  FP={total_fp}")
    print("(この合計値は、phase3_bev_multicam_yolo_check_zonepose_ver1.pyを実行したときの"
          "サマリのnum_misses/num_false_positivesと一致するはず。一致しなければ設定値や"
          "乱数的な処理が無いか確認すること)")

    problem_rows = [r for r in rows if r["n_fn"] > 0 or r["n_fp"] > 0]
    problem_rows.sort(key=lambda r: (r["n_fn"] + r["n_fp"]), reverse=True)

    print(f"\n=== FNまたはFPが発生したフレーム: {len(problem_rows)}/{len(rows)} ===")
    print(f"{'frame':>8} | {'GT数':>5} | {'hyp数':>6} | {'FN':>3} | {'FP':>3}")
    for r in problem_rows:
        print(f"{r['frame_name']:>8} | {r['n_gt']:>5} | {r['n_hyp']:>6} | {r['n_fn']:>3} | {r['n_fp']:>3}")

    suffix = "posecon" if USE_POSE_CONSTRAINT else "noposecon"
    csv_path = OUTPUT_DIR / f"fn_fp_per_frame_curved_{suffix}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_idx", "frame_name", "n_gt", "n_hyp", "n_fn", "n_fp"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV保存(全フレーム分): {csv_path}")
    print("FNまたはFPが発生したフレームのframe_nameを使えば、"
          "render_bev_frames_yolo_zonepose_ver1.pyが出力する"
          "bev_frames_yolo_zonepose_curved/{frame_name}.png を直接開いて目視確認できる"
          "(未実行ならそちらも先に実行しておくこと)。")


if __name__ == "__main__":
    main()