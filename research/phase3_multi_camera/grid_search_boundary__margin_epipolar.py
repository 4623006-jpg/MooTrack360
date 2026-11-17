"""
grid_search_boundary_margin_epipolar.py

phase3_bev_multicam_yolo_check_epipolar.pyのmatch_epipolarに追加した
BOUNDARY_MARGIN_M(隣接カメラの中間境界線から何メートル以内の検出だけを
統合候補にするか)をグリッドサーチするスクリプト。

【背景】match_epipolarはこれまでシーン全体の検出同士(カメラiの全検出 x
カメラjの全検出)からレイの最近接距離が閾値以下の候補を探していた。
これだと、本来は境界を跨いで統合すべきでない、離れた場所にいる個体同士
までもが偶然レイの最近接距離が近くなり、統合相手の候補として扱われて
しまうことがある(実際、統合相手の一意性マージンをどれだけ小さくしても
FPが激増する現象が確認されており、シーン全体を候補にすること自体が
組み合わせ爆発的にノイズを生んでいる可能性が高い)。
境界付近だけに候補を絞れば、この種の偶然の近い値を大幅に減らせるはず。

【このスクリプトでやること】
YOLO検出・重複除去・GT座標の計算(重い処理)はBOUNDARY_MARGIN_Mの値に
依存しないため、フレームごとに1回だけ計算してキャッシュし、
BOUNDARY_MARGIN_Mの各候補値について、統合(fuse_frame_epipolar)+
追跡(SimpleBevTracker)+MOTメトリクス計算だけをやり直す
(phase3_bev_multicam_gt_check_zonepose.py等で使ったのと同じ
キャッシュ方式)。

なお、このグリッドサーチでは統合相手の一意性マージン(MARGIN_GAP_M)は
0(=事実上無効)に固定し、BOUNDARY_MARGIN_Mの効果だけを切り分けて見る。
マージン方式は前回の検証で「密集地帯の判定不能」ではなく「同一カメラ内の
重複検出漏れ」を主に拾ってしまい、狙った効果が出ていないことが分かって
いるため。

--- 使い方 ---
    python grid_search_boundary_margin_epipolar.py
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

MOOTRACK_CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calibration")
if MOOTRACK_CODE_DIR not in sys.path:
    sys.path.insert(0, MOOTRACK_CODE_DIR)

# 同じ出力フォルダにあるphase3_bev_multicam_yolo_check_epipolar.pyから
# 必要な関数・定数をそのまま再利用する(ロジックの二重管理を避けるため)。
from phase3_bev_multicam_yolo_check_epipolar import (
    XML_DIR, IMAGE_DIR, FLOORPLAN_JSON_PATH, OUTPUT_DIR,
    YOLO_WEIGHTS_PATH, FRAME_START, FRAME_END, FRAME_STEP, IMAGE_WIDTH_PX,
    ADJACENT_PAIRS, HEIGHT_RANGE_Z, RAY_DIST_THR_M, USE_POSE_CONSTRAINT,
    DEDUP_IOU_THRESHOLD, DIST_THR_TRACK_M, MAX_MISSED_FRAMES,
    frame_name, parse_voc_annotation, gt_world_points_this_frame, name_to_gt_id,
    yolo_detections_per_camera, deduplicate_per_camera, fuse_frame_epipolar,
    SimpleBevTracker, compute_hota_trackeval,
)
from Mootrack_core import (
    CAM_IDS, ROT_FMT,
    load_floorplan_cameras, apply_camera_overrides, resolve_poses,
)

# グリッドサーチする境界マージン候補[m]。Noneは「制限なし(従来通り全体から
# 候補を探す)」のベースライン。
BOUNDARY_MARGIN_GRID: List[Optional[float]] = [1.0, 2.0, 3.0, 4.0, 5.0, None]

# このグリッドサーチでは一意性マージンを無効化して、境界マージンの効果だけを見る
MARGIN_GAP_FOR_GRID = 0.0


def build_frame_cache(poses) -> List[dict]:
    """YOLO検出・重複除去・GT座標など、BOUNDARY_MARGIN_Mに依存しない
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

        per_camera_raw = yolo_detections_per_camera(yolo_model, image_path, poses)
        per_camera_deduped = deduplicate_per_camera(per_camera_raw, DEDUP_IOU_THRESHOLD)

        cache.append({
            "frame_idx": frame_idx,
            "gt_ids": gt_ids,
            "gt_points": gt_points,
            "per_camera_deduped": per_camera_deduped,
        })
        print(f"  キャッシュ済み: {frame_name(frame_idx)} "
              f"GT個体={len(gt_points)} YOLO検出(重複除去後)="
              f"{sum(len(v) for v in per_camera_deduped.values())}")

    return cache


def run_one_config(frame_cache: List[dict], poses, boundary_margin: Optional[float]) -> dict:
    import motmetrics as mm

    tracker = SimpleBevTracker(dist_thr=DIST_THR_TRACK_M, max_missed=MAX_MISSED_FRAMES)
    acc = mm.MOTAccumulator(auto_id=True)
    frames_data: List[dict] = []

    for fr in frame_cache:
        fused_points = fuse_frame_epipolar(
            fr["per_camera_deduped"], poses, ADJACENT_PAIRS, RAY_DIST_THR_M, HEIGHT_RANGE_Z,
            USE_POSE_CONSTRAINT, MARGIN_GAP_FOR_GRID, boundary_margin,
        )
        track_outputs = tracker.update(fused_points)
        hyp_ids = [tid for tid, _xy, _members in track_outputs]
        hyp_points = [xy for _tid, xy, _members in track_outputs]

        gt_points = fr["gt_points"]
        if gt_points and hyp_points:
            dist = mm.distances.norm2squared_matrix(
                np.array(gt_points), np.array(hyp_points), max_d2=(2.0 * DIST_THR_TRACK_M) ** 2
            )
        else:
            dist = np.empty((len(gt_points), len(hyp_points)))

        acc.update(fr["gt_ids"], hyp_ids, dist)
        frames_data.append({
            "gt_names": [str(i) for i in fr["gt_ids"]], "gt_points": gt_points,
            "hyp_ids": hyp_ids, "hyp_points": hyp_points,
        })

    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=["idf1", "idp", "idr", "mota", "motp", "num_switches",
                 "num_objects", "num_matches", "num_false_positives", "num_misses"],
        name=str(boundary_margin),
    )
    row = summary.iloc[0]

    dist_cap_m = 2.0 * DIST_THR_TRACK_M
    hota_res = compute_hota_trackeval(frames_data, dist_cap_m=dist_cap_m)

    return {
        "boundary_margin_m": boundary_margin if boundary_margin is not None else "なし",
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

    print(f"\n一意性マージンはこのグリッドサーチ中 {MARGIN_GAP_FOR_GRID}(無効)に固定")
    print(f"境界マージン候補: {BOUNDARY_MARGIN_GRID}\n")

    frame_cache = build_frame_cache(poses)
    print(f"\nキャッシュしたフレーム数: {len(frame_cache)}\n")

    results = []
    for boundary_margin in BOUNDARY_MARGIN_GRID:
        print(f"--- boundary_margin={boundary_margin} で評価中 ---")
        res = run_one_config(frame_cache, poses, boundary_margin)
        results.append(res)
        print(f"  IDF1={res['idf1']*100:.1f}% IDP={res['idp']*100:.1f}% IDR={res['idr']*100:.1f}% "
              f"MOTA={res['mota']*100:.1f}% MOTP={res['motp']:.3f} IDs={res['num_switches']} "
              f"FP={res['num_false_positives']} FN={res['num_misses']} HOTA={res['hota']:.3f}")

    csv_path = OUTPUT_DIR / "grid_search_boundary_margin_epipolar.csv"
    import csv as csv_module
    with open(csv_path, "w", newline="") as f:
        writer = csv_module.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        for res in results:
            writer.writerow(res)
    print(f"\nCSV保存: {csv_path}")


if __name__ == "__main__":
    main()