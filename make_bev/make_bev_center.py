"""
make_bev_center.py

MooTrack360 (Farm1-Area1) の3台の魚眼カメラ合成画像に含まれる牛のバウンディング
ボックスを、bboxの中心を姿勢(standing/lying)別の高さ平面に逆投影する「center方式」
でフロアプラン(floorplan.json)の座標系(メートル単位)に変換し、鳥瞰図(BEV)として
描画・保存するスクリプト。

共通のカメラキャリブレーション・幾何変換(Double Sphere逆投影)・アノテーション
読み込みは Mootrack_core.py に集約されているので、ここでは
  - このスクリプト固有の設定(パス、pose別の高さ、描画色、高さスイープ範囲)
  - CowDetectionへの変換・カメラ振り分け・center方式の接地点計算
  - 高さ仮定スイープ(比較表示専用)
  - BEV描画(全体平均版 / カメラ別・ズレ可視化版)と結果CSV出力
だけを持つ。個体の統合(Hungarian割当+Union-Find)は行わない
(それは evaluate_gt_distance_matching.py / evaluate_yolo_distance_matching.py の役割)。
このスクリプトはあくまで「center方式で投影した結果を目で確認する」ための可視化用。

--- 備考: 接地点の高さ仮定(pose別) ---
ここでの POSE_HEIGHT_M はグリッドサーチ前の初期値(lying=0.4, standing=0.85)。
gt_center_height_gridsearch.py によるGTベースのグリッドサーチでは
lying=0.9, standing=1.3 (F1=0.694) がより良い結果だったので、精度検証目的なら
そちらの値に揃えることを推奨する(このスクリプトは可視化用のため、あえて元の値の
ままにしてある)。
"""

import csv
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

import numpy as np
import cv2

from Mootrack_core import (
    NUM_CAMERAS, CAM_IDS, ROT_FMT,
    Pose, Region, get_intrinsics,
    load_floorplan, apply_camera_overrides, resolve_poses,
    unproject_to_floor, parse_voc_annotation,
)

# ---------------------------------------------------------------------------
# このスクリプト固有の設定
# ---------------------------------------------------------------------------

FLOORPLAN_JSON_PATH = "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/calibration/Farm2_Area_1_Artboard_config_.json"
ANNOTATION_XML_PATH = "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/annotation_process/3600_frames_result/annotations/000000.xml"
OUTPUT_DIR = Path("/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/calibration")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ★ 接地点計算(center方式)のパラメータ ★
COW_HEIGHT_M = 1
CENTER_HEIGHT_M = COW_HEIGHT_M / 2.0  # pose不明時のフォールバック高さ

POSE_HEIGHT_M = {
    "lying": 0.4,
    "standing": 0.85,
}

# 高さ仮定を振って比較表示するための値リスト(比較表示のみに使用)
CENTER_HEIGHT_VALUES = [0.50, 0.60, 0.70, 0.725, 0.80, 0.90, 1.00]

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

# カメラごとの点の色(BGR) — per-camera BEVで各視点を区別するため
CAM_COLORS_BGR = {
    "Camera_1": (255, 80, 80),    # 青系
    "Camera_2": (80, 220, 80),    # 緑系
    "Camera_3": (80, 180, 255),   # 橙系
}


# ---------------------------------------------------------------------------
# データ構造(このスクリプト固有: 可視化に必要な情報を1牛1件で持つ)
# ---------------------------------------------------------------------------

@dataclass
class CowDetection:
    name: str
    cam_index: int
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    pose: Optional[str] = None
    world_xy: Optional[np.ndarray] = None
    valid: bool = False


# ---------------------------------------------------------------------------
# アノテーション読み込み(Mootrack_core.parse_voc_annotationをCowDetectionに変換)
# ---------------------------------------------------------------------------

def load_annotation(path: str) -> Tuple[List[CowDetection], int, int]:
    img_w, img_h, objects = parse_voc_annotation(path)
    detections = [
        CowDetection(
            name=obj["name"], cam_index=-1,
            xmin=obj["xmin"], ymin=obj["ymin"], xmax=obj["xmax"], ymax=obj["ymax"],
            pose=obj.get("pose"),
        )
        for obj in objects
    ]
    return detections, int(img_w), int(img_h)


def assign_cameras_and_footpoints(detections: List[CowDetection], img_w: int, num_cams: int) -> float:
    tile_w = img_w / num_cams
    for d in detections:
        cx_full = (d.xmin + d.xmax) / 2.0
        cam_idx = int(cx_full // tile_w)
        cam_idx = min(max(cam_idx, 0), num_cams - 1)
        d.cam_index = cam_idx
    return tile_w


# ---------------------------------------------------------------------------
# center方式による接地点計算(bbox中心をpose別の高さ平面に逆投影)
# ---------------------------------------------------------------------------

def compute_footpoints_center_batch(
    detections_idx: List[int], detections: List[CowDetection],
    cam_idx: int, tile_w: float, pose: Pose,
    fx, fy, cx, cy, xi, alpha,
    center_height_m: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    center_height_m が指定された場合は、その固定値を全検出に一律適用する
    (CENTER_HEIGHT_VALUESを振って比較するスイープ用)。
    指定されない(None)場合は、検出ごとのpose(lying/standing)に応じて
    POSE_HEIGHT_M から高さを取得する(POSE_HEIGHT_Mに無いposeはCENTER_HEIGHT_Mに
    フォールバック)。
    """
    pts_px = []
    heights = []
    for i in detections_idx:
        d = detections[i]
        local_xmin = d.xmin - cam_idx * tile_w
        local_xmax = d.xmax - cam_idx * tile_w
        box_center_x = (local_xmin + local_xmax) / 2.0
        box_center_y = (d.ymin + d.ymax) / 2.0
        pts_px.append([box_center_x, box_center_y])

        if center_height_m is not None:
            heights.append(center_height_m)
        else:
            heights.append(POSE_HEIGHT_M.get(d.pose, CENTER_HEIGHT_M))

    pts_px = np.array(pts_px, dtype=np.float64)
    floor_z = -np.array(heights, dtype=np.float64)
    world_xy, valid = unproject_to_floor(pts_px, pose.R, pose.t, fx, fy, cx, cy, xi, alpha,
                                          floor_z=floor_z)
    return world_xy, valid


# ---------------------------------------------------------------------------
# centerスイープ(bbox中心を仮定する高さを振って比較。比較表示専用)
# ---------------------------------------------------------------------------

def run_one_center_stats(center_height_m: float, poses: Dict[str, Pose],
                          detections_src: List[CowDetection], tile_w: float) -> dict:
    detections = [
        CowDetection(d.name, d.cam_index, d.xmin, d.ymin, d.xmax, d.ymax, d.pose)
        for d in detections_src
    ]

    by_cam = defaultdict(list)
    for i, d in enumerate(detections):
        by_cam[d.cam_index].append(i)

    for cam_idx, idxs in by_cam.items():
        cam_id = CAM_IDS[cam_idx]
        pose = poses[cam_id]
        fx, fy, cx, cy, xi, alpha = get_intrinsics(cam_id)

        world_xy, valid = compute_footpoints_center_batch(
            idxs, detections, cam_idx, tile_w, pose, fx, fy, cx, cy, xi, alpha, center_height_m
        )
        for j, i in enumerate(idxs):
            detections[i].world_xy = world_xy[j]
            detections[i].valid = bool(valid[j])

    grouped: Dict[str, List[CowDetection]] = defaultdict(list)
    for d in detections:
        if d.valid:
            grouped[d.name].append(d)

    dists = []
    for name, dets in grouped.items():
        if len(dets) < 2:
            continue
        xy = np.array([d.world_xy for d in dets])
        mean_xy = xy.mean(axis=0)
        for d in dets:
            dists.append(float(np.linalg.norm(d.world_xy - mean_xy)))

    dists = np.array(dists)
    return {
        "center_height_m": center_height_m,
        "n_pairs": len(dists),
        "sum": float(dists.sum()) if len(dists) else 0.0,
        "mean": float(dists.mean()) if len(dists) else 0.0,
        "median": float(np.median(dists)) if len(dists) else 0.0,
        "max": float(dists.max()) if len(dists) else 0.0,
    }


def print_center_sweep(poses: Dict[str, Pose], detections: List[CowDetection], tile_w: float):
    print(f"\n{'h(m)':>6} | {'n_pairs':>8} | {'sum(m)':>9} | {'mean(m)':>8} | {'median(m)':>10} | {'max(m)':>8}")
    print("-" * 62)
    results = []
    for h in CENTER_HEIGHT_VALUES:
        res = run_one_center_stats(h, poses, detections, tile_w)
        results.append(res)
        print(f"{res['center_height_m']:>6.3f} | {res['n_pairs']:>8d} | {res['sum']:>9.3f} | "
              f"{res['mean']:>8.4f} | {res['median']:>10.4f} | {res['max']:>8.3f}")

    best = min(results, key=lambda r: r["mean"])
    print(f"\n最も平均ズレが小さい高さ仮定(固定値版): {best['center_height_m']}m (mean={best['mean']:.4f}m)")
    print(f"※ 実際の描画には POSE_HEIGHT_M = {POSE_HEIGHT_M} (pose別) を使用中"
          f" / フォールバック CENTER_HEIGHT_M = {CENTER_HEIGHT_M}")

    out_csv = OUTPUT_DIR / "center_sweep_results.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["center_height_m", "n_pairs", "sum", "mean", "median", "max"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"保存しました: {out_csv}")
    return results


# ---------------------------------------------------------------------------
# BEV描画
# ---------------------------------------------------------------------------

def draw_regions(canvas, regions: List[Region], scale: float):
    for region in regions:
        pts = (region.polygon_m * scale).astype(np.int32).reshape(-1, 1, 2)
        color = TYPE_COLORS_BGR.get(region.type, (200, 200, 200))
        closed = region.type not in ("wall", "gate")
        cv2.polylines(canvas, [pts], isClosed=closed, color=color,
                      thickness=2, lineType=cv2.LINE_AA)


def draw_legend(canvas, x0, y0):
    items = list(TYPE_COLORS_BGR.items()) + [("camera", (0, 0, 255)), ("cow (detected)", (255, 255, 255))]
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
    detections, img_w, img_h = load_annotation(ANNOTATION_XML_PATH)
    tile_w = assign_cameras_and_footpoints(detections, img_w, NUM_CAMERAS)
    print(f"  画像サイズ: {img_w}x{img_h} -> タイル幅 {tile_w:.1f}px x {NUM_CAMERAS}台")
    print(f"  検出数: {len(detections)}")
    n_unknown_pose = sum(1 for d in detections if d.pose not in POSE_HEIGHT_M)
    print(f"  接地点計算方式: center方式 (pose別 POSE_HEIGHT_M={POSE_HEIGHT_M}, "
          f"フォールバック CENTER_HEIGHT_M={CENTER_HEIGHT_M}, pose不明/未対応の検出数={n_unknown_pose})")

    print("\n[比較] center方式(固定高さを振った場合。実際の描画はpose別POSE_HEIGHT_Mを使用):")
    print_center_sweep(poses, detections, tile_w)

    by_cam = defaultdict(list)
    for i, d in enumerate(detections):
        by_cam[d.cam_index].append(i)

    for cam_idx, idxs in by_cam.items():
        cam_id = CAM_IDS[cam_idx]
        pose = poses[cam_id]
        fx, fy, cx, cy, xi, alpha = get_intrinsics(cam_id)

        world_xy, valid = compute_footpoints_center_batch(
            idxs, detections, cam_idx, tile_w, pose, fx, fy, cx, cy, xi, alpha
        )
        for j, i in enumerate(idxs):
            detections[i].world_xy = world_xy[j]
            detections[i].valid = bool(valid[j])

    n_valid = sum(d.valid for d in detections)
    print(f"  床面に投影できた検出数: {n_valid} / {len(detections)}")

    grouped: Dict[str, List[CowDetection]] = defaultdict(list)
    for d in detections:
        if d.valid:
            grouped[d.name].append(d)

    merged_points: Dict[str, np.ndarray] = {
        name: np.array([d.world_xy for d in dets]).mean(axis=0)
        for name, dets in grouped.items()
    }
    print(f"  ユニークな牛の数: {len(merged_points)}")

    # ------------------------------------------------------------------
    # BEVキャンバス描画(平均化後)
    # ------------------------------------------------------------------
    scale = 40  # px per meter
    margin = 220
    w_px = int(real_w * scale) + margin
    h_px = int(real_h * scale) + 40
    canvas = np.zeros((h_px, w_px, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30)

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

    draw_legend(canvas, int(real_w * scale) + 15, 20)

    out_path = OUTPUT_DIR / "bev_cow_positions.png"
    cv2.imwrite(str(out_path), canvas)
    print(f"保存しました: {out_path}")

    csv_path = OUTPUT_DIR / "bev_cow_positions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["cow_name", "world_x_m", "world_y_m", "num_camera_views"])
        for name, xy in sorted(merged_points.items()):
            n_views = len(grouped[name])
            writer.writerow([name, round(float(xy[0]), 3), round(float(xy[1]), 3), n_views])
    print(f"保存しました: {csv_path}")

    # ------------------------------------------------------------------
    # カメラ別BEV描画 + CSV出力
    # ------------------------------------------------------------------
    print("\nカメラ別BEVを作成中...")

    canvas_per_cam = np.zeros((h_px, w_px, 3), dtype=np.uint8)
    canvas_per_cam[:] = (30, 30, 30)
    draw_regions(canvas_per_cam, regions, scale)

    for cid in CAM_IDS:
        p = poses[cid]
        px, py = int(p.t[0] * scale), int(p.t[1] * scale)
        if 0 <= px < w_px and 0 <= py < h_px:
            cv2.circle(canvas_per_cam, (px, py), 9, (0, 0, 255), -1)
            cv2.circle(canvas_per_cam, (px, py), 11, (255, 255, 255), 2)
            cv2.putText(canvas_per_cam, cid, (px + 14, py - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)

    for name, dets in grouped.items():
        mean_xy = merged_points[name]
        mpx, mpy = int(mean_xy[0] * scale), int(mean_xy[1] * scale)

        for d in dets:
            cam_id = CAM_IDS[d.cam_index]
            color = CAM_COLORS_BGR.get(cam_id, (200, 200, 200))
            px, py = int(d.world_xy[0] * scale), int(d.world_xy[1] * scale)

            if 0 <= px < w_px and 0 <= py < h_px:
                if 0 <= mpx < w_px and 0 <= mpy < h_px:
                    cv2.line(canvas_per_cam, (mpx, mpy), (px, py), color, 1, cv2.LINE_AA)
                cv2.circle(canvas_per_cam, (px, py), 5, color, -1)
                cv2.circle(canvas_per_cam, (px, py), 6, (0, 0, 0), 1)

        if 0 <= mpx < w_px and 0 <= mpy < h_px:
            cv2.drawMarker(canvas_per_cam, (mpx, mpy), (255, 255, 255),
                           markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
            cv2.putText(canvas_per_cam, name.replace("cow", ""), (mpx + 8, mpy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1, cv2.LINE_AA)

    legend_items = [(cid, CAM_COLORS_BGR[cid]) for cid in CAM_IDS] + [("mean (x mark)", (255, 255, 255))]
    x0, y0 = int(real_w * scale) + 15, 20
    for i, (label, color) in enumerate(legend_items):
        y = y0 + i * 22
        cv2.rectangle(canvas_per_cam, (x0, y), (x0 + 16, y + 16), color, -1)
        cv2.putText(canvas_per_cam, label, (x0 + 24, y + 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)

    out_path_per_cam = OUTPUT_DIR / "bev_cow_positions_per_camera.png"
    cv2.imwrite(str(out_path_per_cam), canvas_per_cam)
    print(f"保存しました: {out_path_per_cam}")

    csv_path_per_cam = OUTPUT_DIR / "bev_cow_positions_per_camera.csv"
    with open(csv_path_per_cam, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cow_name", "camera_id", "pose", "world_x_m", "world_y_m",
            "mean_x_m", "mean_y_m", "dist_from_mean_m",
            "bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"
        ])
        for name, dets in sorted(grouped.items()):
            mean_xy = merged_points[name]
            for d in sorted(dets, key=lambda x: x.cam_index):
                cam_id = CAM_IDS[d.cam_index]
                dist = float(np.linalg.norm(d.world_xy - mean_xy))
                writer.writerow([
                    name, cam_id, d.pose,
                    round(float(d.world_xy[0]), 3), round(float(d.world_xy[1]), 3),
                    round(float(mean_xy[0]), 3), round(float(mean_xy[1]), 3),
                    round(dist, 3),
                    round(d.xmin, 1), round(d.ymin, 1), round(d.xmax, 1), round(d.ymax, 1),
                ])
    print(f"保存しました: {csv_path_per_cam}")

    print("\n--- ズレ(平均からの距離)が大きい検出 TOP10 ---")
    rows = []
    for name, dets in grouped.items():
        mean_xy = merged_points[name]
        for d in dets:
            cam_id = CAM_IDS[d.cam_index]
            dist = float(np.linalg.norm(d.world_xy - mean_xy))
            rows.append((dist, name, cam_id, d.world_xy, mean_xy))
    rows.sort(key=lambda r: -r[0])
    for dist, name, cam_id, wxy, mxy in rows[:10]:
        print(f"  {name:10s} {cam_id:10s} world=({wxy[0]:.2f},{wxy[1]:.2f}) "
              f"mean=({mxy[0]:.2f},{mxy[1]:.2f}) dist={dist:.3f}m")


if __name__ == "__main__":
    main()