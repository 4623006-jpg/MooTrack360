"""
measure_yolo_bbox_pixel_error.py

「YOLOの検出が実際に何ピクセルくらいブレるか」を、仮定(1pxや10px)ではなく
GTアノテーションとの比較から実測する。

やり方:
  各フレーム・各カメラで、GTのbbox(voc_objects_local_by_camera、ローカル
  ピクセル座標)とYOLOのbbox(yolo_world_points_per_camera_with_pose、同じ
  ローカル座標系)を、bbox中心同士の距離でHungarianマッチングする
  (パイプラインの実際の投影点はbboxの中心((xmin+xmax)/2, (ymin+ymax)/2)
  なので、それと同じ点で比較する)。マッチした(=同じ個体を指していると
  みなせる)ペアについて、GT中心とYOLO中心のピクセル距離を「検出の
  ピクセル誤差」として記録する。

これを多数のフレームで集計すれば、平均・中央値・90/95パーセンタイル・最大値
といった実測の誤差分布が得られる。これをcompute_acceptable_fisheye_radius.py
のPIXEL_UNCERTAINTY_PXに使えば、仮定ではなく実測に基づいた許容範囲が求まる。

さらに、画像中心からの半径ごとにも誤差を集計する(周辺部ほどYOLO自体の
検出精度が落ちているかどうかも、この実測データから分かる)。

--- 使い方 ---
    python measure_yolo_bbox_pixel_error.py
"""

import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

MOOTRACK_CODE_DIR = os.path.join(THIS_DIR, "..", "calibration")
if MOOTRACK_CODE_DIR not in sys.path:
    sys.path.insert(0, MOOTRACK_CODE_DIR)

# render_bev_frames_yolo_zonepose.pyが実際に置かれているフォルダ。
# このスクリプト(measure_yolo_bbox_pixel_error.py)は専用の"analyze"フォルダなど、
# render_bev_frames_yolo_zonepose.pyとは別の場所に置かれることがあるため、
# 明示的にそのフォルダをsys.pathへ追加する。実際のパスに合わせて書き換えること。
RENDER_SCRIPT_DIR = os.path.join(THIS_DIR, "..", "phase3_multi_camera")
if RENDER_SCRIPT_DIR not in sys.path:
    sys.path.insert(0, RENDER_SCRIPT_DIR)

if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from Mootrack_core import CAM_IDS, ROT_FMT, get_intrinsics, load_floorplan, apply_camera_overrides, resolve_poses

from render_bev_frames_yolo_zonepose import (
    XML_DIR, IMAGE_DIR, FLOORPLAN_JSON_PATH, YOLO_WEIGHTS_PATH, IMAGE_WIDTH_PX,
    frame_name, parse_voc_annotation, voc_objects_local_by_camera,
    yolo_world_points_per_camera_with_pose,
)

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------

FRAME_START = 0
FRAME_END = 450
FRAME_STEP = 10

# GT中心とYOLO中心の距離がこれ以内なら「同じ個体の検出」とみなす
# (これより遠い場合は誤検出/検出漏れとして、誤差の統計には含めない)
MATCH_DIST_THR_PX = 100.0

# 画像中心からの半径をこの幅でビンに区切って集計する
RADIUS_BIN_PX = 300


def bbox_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def match_gt_to_yolo(
    gt_items: List[dict], yolo_items: List[dict], dist_thr: float,
) -> List[Tuple[int, int, float]]:
    """GTとYOLOのbbox中心同士をHungarianマッチングする。
    戻り値は[(gt_idx, yolo_idx, 距離px), ...](dist_thr以内のペアのみ)。"""
    if not gt_items or not yolo_items:
        return []
    n_g, n_y = len(gt_items), len(yolo_items)
    cost = np.zeros((n_g, n_y))
    for i, g in enumerate(gt_items):
        gx, gy = bbox_center(g["bbox_local"])
        for j, y in enumerate(yolo_items):
            yx, yy = bbox_center(y["bbox_local"])
            cost[i, j] = float(np.hypot(gx - yx, gy - yy))
    row_idx, col_idx = linear_sum_assignment(cost)
    return [(i, j, float(cost[i, j])) for i, j in zip(row_idx, col_idx) if cost[i, j] <= dist_thr]


def summarize(label: str, errors: List[float]) -> None:
    if not errors:
        print(f"{label}: データ無し")
        return
    arr = np.array(errors)
    print(f"{label}: n={len(arr)}  平均={arr.mean():.2f}px  中央値={np.median(arr):.2f}px  "
          f"標準偏差={arr.std():.2f}px  90%ile={np.percentile(arr, 90):.2f}px  "
          f"95%ile={np.percentile(arr, 95):.2f}px  最大={arr.max():.2f}px")


def main():
    from ultralytics import YOLO

    print("floorplan.json 読み込み中...")
    _regions, cameras_json, _real_w, _real_h = load_floorplan(FLOORPLAN_JSON_PATH)
    cameras_json = apply_camera_overrides(cameras_json)
    poses = resolve_poses(cameras_json, root="Area1", fmt=ROT_FMT)

    print(f"YOLOモデル読み込み中: {YOLO_WEIGHTS_PATH}")
    yolo_model = YOLO(YOLO_WEIGHTS_PATH)

    xml_dir = Path(XML_DIR)
    image_dir = Path(IMAGE_DIR)
    frame_indices = list(range(FRAME_START, FRAME_END + 1, FRAME_STEP))

    all_errors: List[float] = []
    errors_by_camera: Dict[str, List[float]] = defaultdict(list)
    errors_by_radius_bin: Dict[int, List[float]] = defaultdict(list)
    n_frames_used = 0

    for frame_idx in frame_indices:
        xml_path = xml_dir / f"{frame_name(frame_idx)}.xml"
        image_path = image_dir / f"{frame_name(frame_idx)}.jpg"
        if not xml_path.exists() or not image_path.exists():
            continue

        gt_objects = parse_voc_annotation(xml_path)
        gt_per_camera = voc_objects_local_by_camera(gt_objects, IMAGE_WIDTH_PX)
        yolo_per_camera = yolo_world_points_per_camera_with_pose(yolo_model, image_path, poses)

        n_frames_used += 1
        for cam_id in CAM_IDS:
            gt_items = gt_per_camera.get(cam_id, [])
            yolo_items = yolo_per_camera.get(cam_id, [])
            matches = match_gt_to_yolo(gt_items, yolo_items, MATCH_DIST_THR_PX)

            _fx, _fy, cx, cy, _xi, _alpha = get_intrinsics(cam_id)

            for gt_idx, _yolo_idx, dist_px in matches:
                all_errors.append(dist_px)
                errors_by_camera[cam_id].append(dist_px)

                gx, gy = bbox_center(gt_items[gt_idx]["bbox_local"])
                radius = float(np.hypot(gx - cx, gy - cy))
                radius_bin = int(radius // RADIUS_BIN_PX) * RADIUS_BIN_PX
                errors_by_radius_bin[radius_bin].append(dist_px)

        if n_frames_used % 10 == 0:
            print(f"  {n_frames_used}フレーム処理済み (直近: {frame_name(frame_idx)}, "
                  f"累積マッチ数={len(all_errors)})")

    print(f"\n処理フレーム数: {n_frames_used}  総マッチ数: {len(all_errors)}\n")

    print("=== 全体 ===")
    summarize("全カメラ合計", all_errors)

    print("\n=== カメラ別 ===")
    for cam_id in CAM_IDS:
        summarize(cam_id, errors_by_camera.get(cam_id, []))

    print(f"\n=== 画像中心からの半径別({RADIUS_BIN_PX}px刻み) ===")
    for radius_bin in sorted(errors_by_radius_bin.keys()):
        summarize(f"半径[{radius_bin}, {radius_bin + RADIUS_BIN_PX})px", errors_by_radius_bin[radius_bin])

    if all_errors:
        arr = np.array(all_errors)
        print(f"\n参考: compute_acceptable_fisheye_radius.pyのPIXEL_UNCERTAINTY_PXに使うなら、"
              f"平均({arr.mean():.1f}px)より90%ile({np.percentile(arr, 90):.1f}px)や"
              f"95%ile({np.percentile(arr, 95):.1f}px)の方が、外れ値も考慮した安全側の見積もりになる。")


if __name__ == "__main__":
    main()