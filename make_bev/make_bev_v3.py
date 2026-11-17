"""
make_bev_pose_height.py

make_bev.py(gamma/relief/center方式を切り替えられる旧式スクリプト)を、
Mootrack_core.py の共通関数を使うように書き直した版。

接地点の求め方は「center方式」に固定し、高さ仮定はGTの<pose>タグ
(lying/standing)に応じて切り替える(gt_center_height_gridsearch.py /
phase3_bev_multicam_gt_check.py で採用した、再キャリブレーション後の
ベスト値と同じ):
    lying    -> LYING_HEIGHT_M    = 0.6m
    standing -> STANDING_HEIGHT_M = 1.0m
    (poseタグが無い個体は DEFAULT_HEIGHT_M = 0.8m にフォールバック)

bbox中心ピクセルを、その高さの水平面(z = -height_m)にDouble Sphereモデルで
逆投影する(gamma方式の経験的曲線補正やrelief方式の2隅平均は使わない、
最もシンプルな center方式)。

出力(OUTPUT_DIR):
  - bev_cow_positions.png            : 複数カメラ平均後のBEV(フロアプラン+牛の位置)
  - bev_cow_positions.csv            : 上記の座標一覧
  - bev_cow_positions_per_camera.png : カメラ別の生投影結果(平均前)+ズレの可視化
  - bev_cow_positions_per_camera.csv : カメラ別の座標とズレ量

--- 使い方 ---
    python make_bev_pose_height.py
"""

import csv
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import cv2

# ---------------------------------------------------------------------------
# Mootrack_core.py の読み込み
# このスクリプトが calibration/ の「兄弟フォルダ」に置かれている前提の
# 自己位置基準パス(プロジェクトの標準パターン)。
# 実際の配置がこれと違う場合は MOOTRACK_CODE_DIR を直接書き換えること。
# ---------------------------------------------------------------------------
MOOTRACK_CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calibration")
if MOOTRACK_CODE_DIR not in sys.path:
    sys.path.insert(0, MOOTRACK_CODE_DIR)

from Mootrack_core import (
    NUM_CAMERAS, CAM_IDS, ROT_FMT,
    get_intrinsics,
    load_floorplan, apply_camera_overrides, resolve_poses,
    unproject_to_floor,
    parse_voc_annotation,
)

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
# ※ このパスは過去に FileNotFoundError で問題になったパスと同じもの。
#   エラーが出る場合は `find ... -iname "Farm2_Area_1_Artboard_config_.json"`
#   で現在の正しい場所を調べて書き換えること。
FLOORPLAN_JSON_PATH = "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/calibration/Farm2_Area_1_Artboard_config_.json"
ANNOTATION_XML_PATH = "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/annotation_process/3600_frames_result/annotations/000000.xml"
OUTPUT_DIR = Path("/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/calibration")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 姿勢別の高さ仮定(再キャリブレーション後のセンター方式ベスト値)
LYING_HEIGHT_M = 0.4
STANDING_HEIGHT_M = 0.7
DEFAULT_HEIGHT_M = 0.8  # poseタグが無い場合のフォールバック

TYPE_COLORS_BGR = {
    "wall":            (0, 200, 0),
    "cow_beds":        (0, 140, 255),
    "feeding_bay":     (255, 140, 0),
    "water":           (255, 255, 0),
    "gate":            (0, 0, 255),
    "milking_station": (255, 0, 255),
    "hay_area":        (0, 255, 255),
    "cow_brushes":     (255, 0, 180),
}

CAM_COLORS_BGR = {
    "Camera_1": (255, 80, 80),
    "Camera_2": (80, 220, 80),
    "Camera_3": (80, 180, 255),
}


def height_for(pose) -> float:
    if pose == "lying":
        return LYING_HEIGHT_M
    if pose == "standing":
        return STANDING_HEIGHT_M
    return DEFAULT_HEIGHT_M


# ---------------------------------------------------------------------------
# center方式(bbox中心 -> 姿勢別の高さ平面に逆投影)でworld座標を計算
# ---------------------------------------------------------------------------

def compute_world_points(objects: List[dict], poses, img_w: float):
    """
    objects: Mootrack_core.parse_voc_annotation() が返す dict のリスト
             (name, pose, xmin, ymin, xmax, ymax)
    戻り値: name -> [(cam_id, world_xy), ...] のグルーピング済み辞書
    """
    tile_w = img_w / NUM_CAMERAS
    by_cam_objs: Dict[str, List[dict]] = defaultdict(list)
    for obj in objects:
        center_x_full = (obj["xmin"] + obj["xmax"]) / 2.0
        cam_idx = int(center_x_full // tile_w)
        cam_idx = max(0, min(cam_idx, NUM_CAMERAS - 1))
        cam_id = CAM_IDS[cam_idx]
        by_cam_objs[cam_id].append(obj)

    grouped: Dict[str, List[Tuple[str, np.ndarray, dict]]] = defaultdict(list)
    n_valid, n_total = 0, 0

    for cam_id, objs in by_cam_objs.items():
        cam_idx = CAM_IDS.index(cam_id)
        fx, fy, cx, cy, xi, alpha = get_intrinsics(cam_id)
        cam_pose = poses[cam_id]

        pts_px = []
        heights = []
        for obj in objs:
            local_x = (obj["xmin"] + obj["xmax"]) / 2.0 - cam_idx * tile_w
            local_y = (obj["ymin"] + obj["ymax"]) / 2.0
            pts_px.append([local_x, local_y])
            heights.append(height_for(obj["pose"]))
        pts_px = np.array(pts_px, dtype=np.float64)
        floor_z = -np.array(heights, dtype=np.float64)

        world_xy, valid = unproject_to_floor(
            pts_px, cam_pose.R, cam_pose.t, fx, fy, cx, cy, xi, alpha, floor_z=floor_z
        )

        for obj, xy, ok in zip(objs, world_xy, valid):
            n_total += 1
            if not ok:
                continue
            n_valid += 1
            grouped[obj["name"]].append((cam_id, xy, obj))

    print(f"  床面に投影できた検出数: {n_valid} / {n_total}")
    return grouped


# ---------------------------------------------------------------------------
# BEV描画
# ---------------------------------------------------------------------------

def draw_regions(canvas, regions, scale: float):
    for region in regions:
        pts = (region.polygon_m * scale).astype(np.int32).reshape(-1, 1, 2)
        color = TYPE_COLORS_BGR.get(region.type, (200, 200, 200))
        closed = region.type not in ("wall", "gate")
        cv2.polylines(canvas, [pts], isClosed=closed, color=color,
                      thickness=2, lineType=cv2.LINE_AA)


def draw_legend(canvas, x0, y0, items):
    for i, (label, color) in enumerate(items):
        y = y0 + i * 22
        cv2.rectangle(canvas, (x0, y), (x0 + 16, y + 16), color, -1)
        cv2.putText(canvas, label, (x0 + 24, y + 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    print("floorplan.json 読み込み中...")
    regions, cameras_json, real_w, real_h = load_floorplan(FLOORPLAN_JSON_PATH)

    print("カメラパラメータの上書き設定を適用中...")
    cameras_json = apply_camera_overrides(cameras_json)

    print("カメラ姿勢を解決中...")
    poses = resolve_poses(cameras_json, root="Area1", fmt=ROT_FMT)
    for cid in CAM_IDS:
        p = poses[cid]
        fx, fy, cx, cy, xi, alpha = get_intrinsics(cid)
        print(f"  {cid}: t={p.t.round(2)}  xi={xi} alpha={alpha}")

    print("annotation.xml 読み込み中...")
    img_w, img_h, objects = parse_voc_annotation(ANNOTATION_XML_PATH)
    print(f"  画像サイズ: {img_w}x{img_h}px  検出数: {len(objects)}")
    print(f"  高さ仮定: lying={LYING_HEIGHT_M}m, standing={STANDING_HEIGHT_M}m, "
          f"pose不明={DEFAULT_HEIGHT_M}m")

    grouped = compute_world_points(objects, poses, img_w)
    print(f"  ユニークな牛の数: {len(grouped)}")

    merged_points: Dict[str, np.ndarray] = {}
    for name, items in grouped.items():
        xy = np.array([it[1] for it in items])
        merged_points[name] = xy.mean(axis=0)

    # --- 平均後のBEV ---
    scale = 40
    margin = 220
    w_px = int(real_w * scale) + margin
    h_px = int(real_h * scale) + 40
    canvas = np.full((h_px, w_px, 3), (30, 30, 30), dtype=np.uint8)
    draw_regions(canvas, regions, scale)

    for cid in CAM_IDS:
        p = poses[cid]
        px, py = int(p.t[0] * scale), int(p.t[1] * scale)
        if 0 <= px < w_px and 0 <= py < h_px:
            cv2.circle(canvas, (px, py), 9, (0, 0, 255), -1)
            cv2.circle(canvas, (px, py), 11, (255, 255, 255), 2)
            cv2.putText(canvas, cid, (px + 14, py - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)

    for name, xy in merged_points.items():
        px, py = int(xy[0] * scale), int(xy[1] * scale)
        if 0 <= px < w_px and 0 <= py < h_px:
            cv2.circle(canvas, (px, py), 6, (255, 255, 255), -1)
            cv2.circle(canvas, (px, py), 7, (0, 0, 0), 1)
            cv2.putText(canvas, name.replace("cow", ""), (px + 6, py + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1, cv2.LINE_AA)

    legend_items = list(TYPE_COLORS_BGR.items()) + [("camera", (0, 0, 255)), ("cow (detected)", (255, 255, 255))]
    draw_legend(canvas, int(real_w * scale) + 15, 20, legend_items)

    out_path = OUTPUT_DIR / "bev_cow_positions.png"
    cv2.imwrite(str(out_path), canvas)
    print(f"保存しました: {out_path}")

    csv_path = OUTPUT_DIR / "bev_cow_positions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["cow_name", "world_x_m", "world_y_m", "num_camera_views"])
        for name, xy in sorted(merged_points.items()):
            writer.writerow([name, round(float(xy[0]), 3), round(float(xy[1]), 3), len(grouped[name])])
    print(f"保存しました: {csv_path}")

    # --- カメラ別BEV(平均前の生座標 + ズレの可視化) ---
    print("\nカメラ別BEVを作成中...")
    canvas_per_cam = np.full((h_px, w_px, 3), (30, 30, 30), dtype=np.uint8)
    draw_regions(canvas_per_cam, regions, scale)

    for cid in CAM_IDS:
        p = poses[cid]
        px, py = int(p.t[0] * scale), int(p.t[1] * scale)
        if 0 <= px < w_px and 0 <= py < h_px:
            cv2.circle(canvas_per_cam, (px, py), 9, (0, 0, 255), -1)
            cv2.circle(canvas_per_cam, (px, py), 11, (255, 255, 255), 2)
            cv2.putText(canvas_per_cam, cid, (px + 14, py - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)

    csv_rows = []
    for name, items in sorted(grouped.items()):
        mean_xy = merged_points[name]
        mpx, mpy = int(mean_xy[0] * scale), int(mean_xy[1] * scale)

        for cam_id, xy, obj in sorted(items, key=lambda it: CAM_IDS.index(it[0])):
            color = CAM_COLORS_BGR.get(cam_id, (200, 200, 200))
            px, py = int(xy[0] * scale), int(xy[1] * scale)
            if 0 <= px < w_px and 0 <= py < h_px:
                if 0 <= mpx < w_px and 0 <= mpy < h_px:
                    cv2.line(canvas_per_cam, (mpx, mpy), (px, py), color, 1, cv2.LINE_AA)
                cv2.circle(canvas_per_cam, (px, py), 5, color, -1)
                cv2.circle(canvas_per_cam, (px, py), 6, (0, 0, 0), 1)

            dist = float(np.linalg.norm(xy - mean_xy))
            csv_rows.append([
                name, cam_id, round(float(xy[0]), 3), round(float(xy[1]), 3),
                round(float(mean_xy[0]), 3), round(float(mean_xy[1]), 3), round(dist, 3),
                obj["pose"], height_for(obj["pose"]),
                round(obj["xmin"], 1), round(obj["ymin"], 1), round(obj["xmax"], 1), round(obj["ymax"], 1),
            ])

        if 0 <= mpx < w_px and 0 <= mpy < h_px:
            cv2.drawMarker(canvas_per_cam, (mpx, mpy), (255, 255, 255),
                           markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
            cv2.putText(canvas_per_cam, name.replace("cow", ""), (mpx + 8, mpy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1, cv2.LINE_AA)

    legend_items = [(cid, CAM_COLORS_BGR[cid]) for cid in CAM_IDS] + [("mean (x mark)", (255, 255, 255))]
    draw_legend(canvas_per_cam, int(real_w * scale) + 15, 20, legend_items)

    out_path_per_cam = OUTPUT_DIR / "bev_cow_positions_per_camera.png"
    cv2.imwrite(str(out_path_per_cam), canvas_per_cam)
    print(f"保存しました: {out_path_per_cam}")

    csv_path_per_cam = OUTPUT_DIR / "bev_cow_positions_per_camera.csv"
    with open(csv_path_per_cam, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cow_name", "camera_id", "world_x_m", "world_y_m",
            "mean_x_m", "mean_y_m", "dist_from_mean_m", "pose", "assumed_height_m",
            "bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax",
        ])
        writer.writerows(csv_rows)
    print(f"保存しました: {csv_path_per_cam}")

    print("\n--- ズレ(平均からの距離)が大きい検出 TOP10 ---")
    for row in sorted(csv_rows, key=lambda r: -r[6])[:10]:
        name, cam_id, wx, wy, mx, my, dist = row[:7]
        print(f"  {name:10s} {cam_id:10s} world=({wx:.2f},{wy:.2f}) mean=({mx:.2f},{my:.2f}) dist={dist:.3f}m")


if __name__ == "__main__":
    main()