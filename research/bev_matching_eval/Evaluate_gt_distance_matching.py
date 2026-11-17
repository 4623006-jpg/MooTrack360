"""
アノテーションデータ(VOC形式XML、poseタグ付き)を使って、
「bboxの中心を姿勢(standing/lying)別の高さ平面に逆投影して1点にし、
カメラ間の距離が閾値以下なら同一個体とみなす」という手法(center方式)の精度を
検証するスクリプト。

共通のカメラキャリブレーション・幾何変換・Hungarian割当・Union-Find・評価指標の
計算は mootrack_core.py に集約されているので、ここでは
  - このスクリプト固有の設定(パス、pose別の高さ、距離閾値のスイープ範囲)
  - GTのbbox中心をworld座標の1点に変換する処理(build_gt_world_points)
  - 実行フロー(main)
だけを持つ。

YOLOは使わない。アノテーション済みのbbox+個体名(name)+pose(standing/lying)を
正解として使う。時系列ではなく1フレーム内のカメラ間統合精度の検証なので、
対象は1つのアノテーションXMLのみ。
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from Mootrack_core import (
    NUM_CAMERAS, CAM_IDS, ROT_FMT,
    Pose, get_intrinsics,
    load_floorplan_cameras, apply_camera_overrides, resolve_poses,
    unproject_to_floor, parse_voc_annotation,
    compute_camera_assignments_by_distance,
    build_keys_and_name_lookup, evaluate_merge_accuracy,
)


# ---------------------------------------------------------------------------
# このスクリプト固有の設定
# ---------------------------------------------------------------------------

FLOORPLAN_JSON_PATH = "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/calibration/Farm2_Area_1_Artboard_config_.json"
ANNOTATION_XML_PATH = "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/annotation_process/3600_frames_result/annotations/000000.xml"

# ★ 接地点計算(center方式)のパラメータ ★
# gt_center_height_gridsearch.py によるグリッドサーチでF1が最大になった組み合わせ。
COW_HEIGHT_M = 1.45
CENTER_HEIGHT_M = COW_HEIGHT_M / 2.0  # poseが不明な場合のフォールバック高さ

POSE_HEIGHT_M = {
    "lying": 0.9,
    "standing": 1.3,
}

# 検証したい距離閾値の一覧(単位: m。感度分析のため複数試す)
DIST_THRESHOLDS_M = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]


# ---------------------------------------------------------------------------
# bbox(アノテーション) -> カメラ振り分け -> World座標の1点(center方式)
# ---------------------------------------------------------------------------

def build_gt_world_points(
    objects: List[dict],
    poses: Dict[str, Pose],
    image_width: float,
    pose_height_m: Dict[str, float] = POSE_HEIGHT_M,
    default_height_m: float = CENTER_HEIGHT_M,
) -> Dict[str, List[Tuple[str, np.ndarray]]]:
    """
    各GTのbbox中心を、姿勢(pose: standing/lying)に応じた高さ平面に逆投影する。
    pose_height_m に無いposeの値(未設定、typoなど)は default_height_m にフォールバックする。
    """
    tile_w = image_width / NUM_CAMERAS
    per_camera: Dict[str, List[Tuple[str, np.ndarray]]] = {cid: [] for cid in CAM_IDS}

    for obj in objects:
        center_x_full = (obj["xmin"] + obj["xmax"]) / 2.0
        center_y = (obj["ymin"] + obj["ymax"]) / 2.0
        cam_idx = int(center_x_full // tile_w)
        cam_idx = max(0, min(cam_idx, NUM_CAMERAS - 1))
        cam_id = CAM_IDS[cam_idx]

        fx, fy, cx, cy, xi, alpha = get_intrinsics(cam_id)
        local_x = center_x_full - cam_idx * tile_w
        pts_px = np.array([[local_x, center_y]], dtype=np.float64)

        height_m = pose_height_m.get(obj.get("pose"), default_height_m)

        cam_pose = poses[cam_id]
        world_xy, valid = unproject_to_floor(
            pts_px, cam_pose.R, cam_pose.t, fx, fy, cx, cy, xi, alpha,
            floor_z=-height_m,
        )
        if not valid[0]:
            continue

        per_camera[cam_id].append((obj["name"], world_xy[0]))

    return per_camera


def main():
    print("floorplan.json 読み込み中...")
    cameras_json = load_floorplan_cameras(FLOORPLAN_JSON_PATH)

    print("カメラパラメータの上書き設定を適用中...")
    cameras_json = apply_camera_overrides(cameras_json)

    print("カメラ姿勢を解決中...")
    poses = resolve_poses(cameras_json, root="Area1", fmt=ROT_FMT)

    xml_path = Path(ANNOTATION_XML_PATH)
    if not xml_path.exists():
        raise FileNotFoundError(f"アノテーションXMLが見つかりません: {xml_path}")
    print(f"評価対象: {xml_path}")
    print(f"接地点計算方式: center方式 (pose別高さ: {POSE_HEIGHT_M}, フォールバック={CENTER_HEIGHT_M})")

    width, _height, objects = parse_voc_annotation(xml_path)

    from collections import defaultdict
    pose_counts: Dict[str, int] = defaultdict(int)
    for obj in objects:
        pose_counts[obj.get("pose") or "(none)"] += 1
    print(f"  pose内訳: {dict(pose_counts)}")

    per_camera = build_gt_world_points(objects, poses, width)
    assignments, gt_same_pairs = compute_camera_assignments_by_distance(per_camera)

    for cam_id in CAM_IDS:
        print(f"  {cam_id}: {len(per_camera[cam_id])}頭 (アノテーション)")

    total_same_gt = len(gt_same_pairs)
    print(f"\nこのフレームで、複数カメラに跨って映っている個体ペア(正解=同一個体): {total_same_gt}件")
    print(f"Hungarian法による割当ペア総数(閾値フィルタ前、参考値): {len(assignments)}件")

    all_keys, name_lookup = build_keys_and_name_lookup(per_camera)

    print("\n=== 距離閾値ごとの精度 (Hungarian割当 + Union-Findによるクラスタ統合) ===")
    print(f"{'距離閾値(m)':>10} | {'統合成功率(recall)':>20} | {'誤統合precision':>16} | {'F1':>8}")
    for thr in DIST_THRESHOLDS_M:
        accepted_pairs = [
            ((a.cam_i, a.idx_i), (a.cam_j, a.idx_j))
            for a in assignments if a.dist <= thr
        ]
        r = evaluate_merge_accuracy(all_keys, gt_same_pairs, accepted_pairs, name_lookup)

        print(f"{thr:>10.2f} | {r['recall']:>10.3f} ({r['tp']}/{r['tp']+r['fn']}) | "
              f"{r['precision']:>10.3f} (FP={r['fp']}) | {r['f1']:>8.3f}")


if __name__ == "__main__":
    main()