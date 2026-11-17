"""
make_bev_sam_centroid.py

sam_centroid_pilot.py が出力した sam_centroid_results.csv(各牛のSAMマスク重心)を、
Mootrack_core.py の Double Sphere逆投影で floorplan.json の座標系(メートル単位)に
変換し、鳥瞰図(BEV)として描画する。

make_bev_center.py の「center方式(bbox中心)」を「SAMマスク重心」に置き換えた版。
高さ仮定は gt_center_height_gridsearch.py でのグリッドサーチ結果
(lying=0.9m, standing=1.3m, F1=0.694)に合わせている
(center方式のベースラインと条件を揃えて比較しやすくするため)。

--- 事前準備 ---
Mootrack_core.py をこのスクリプトと同じディレクトリに置いておくこと
(Mootrack_core.py自身のdocstringにある通り、素朴な `from Mootrack_core import ...`
 で読み込む前提)。

--- 使い方 ---
    python make_bev_sam_centroid.py
(パスは全て下の設定セクションに直書きしてあるので、引数なしで実行する)
"""

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from Mootrack_core import (
    NUM_CAMERAS, CAM_IDS, ROT_FMT,
    Pose, Region, get_intrinsics,
    load_floorplan, apply_camera_overrides, resolve_poses,
    unproject_to_floor,
)

# ---------------------------------------------------------------------------
# 設定(パスはここを書き換える)
# ---------------------------------------------------------------------------
SAM_CSV_PATH = "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/make_bev/sam_centroid_test/sam_centroid_results.csv"
FLOORPLAN_JSON_PATH = "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/calibration/Farm2_Area_1_Artboard_config_.json"
IMAGE_WIDTH_PX = 8832  # 000000.jpg (3カメラ分ステッチ済み画像)の幅
OUTPUT_DIR = "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/make_bev/sam_centroid_test"

# 生の重心ではなく、脚を除去した胴体重心(torso_centroid_x/y)を使う場合はTrueにする
USE_TORSO_CENTROID = False

# center方式のベースライン(gt_center_height_gridsearch.pyでのグリッドサーチ結果、
# F1=0.694)と条件を揃えるため、同じ高さ仮定を使う。
# sam_centroid_height_gridsearch.py の結果に合わせて更新してもよい。
POSE_HEIGHT_M = {
    "lying": 0.5,
    "standing": 1.0193,
}
FALLBACK_HEIGHT_M = 1.0

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


@dataclass
class SamDetection:
    name: str
    pose: Optional[str]
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    centroid_x: float
    centroid_y: float
    score: float
    cam_index: int = -1
    world_xy: Optional[np.ndarray] = None
    valid: bool = False


def load_sam_results(csv_path: Path, use_torso: bool = USE_TORSO_CENTROID) -> List[SamDetection]:
    x_col = "torso_centroid_x" if use_torso else "centroid_x"
    y_col = "torso_centroid_y" if use_torso else "centroid_y"
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    dets = []
    for r in rows:
        dets.append(SamDetection(
            name=r["name"],
            pose=(r["pose"] or None),
            xmin=float(r["xmin"]), ymin=float(r["ymin"]),
            xmax=float(r["xmax"]), ymax=float(r["ymax"]),
            centroid_x=float(r[x_col]), centroid_y=float(r[y_col]),
            score=float(r["score"]),
        ))
    return dets


def assign_cameras(dets: List[SamDetection], img_w: int, num_cams: int) -> float:
    tile_w = img_w / num_cams
    for d in dets:
        # camera割り当てはbbox中心基準(元画像上での位置。make_bev_center.pyと同じ考え方)
        cx_full = (d.xmin + d.xmax) / 2.0
        cam_idx = int(cx_full // tile_w)
        cam_idx = min(max(cam_idx, 0), num_cams - 1)
        d.cam_index = cam_idx
    return tile_w


def project_sam_centroids(dets: List[SamDetection], poses: Dict[str, Pose], tile_w: float) -> None:
    by_cam = defaultdict(list)
    for i, d in enumerate(dets):
        by_cam[d.cam_index].append(i)

    for cam_idx, idxs in by_cam.items():
        cam_id = CAM_IDS[cam_idx]
        pose = poses[cam_id]
        fx, fy, cx, cy, xi, alpha = get_intrinsics(cam_id)

        pts_px = []
        heights = []
        for i in idxs:
            d = dets[i]
            local_x = d.centroid_x - cam_idx * tile_w
            local_y = d.centroid_y
            pts_px.append([local_x, local_y])
            heights.append(POSE_HEIGHT_M.get(d.pose, FALLBACK_HEIGHT_M))

        pts_px = np.array(pts_px, dtype=np.float64)
        floor_z = -np.array(heights, dtype=np.float64)
        world_xy, valid = unproject_to_floor(
            pts_px, pose.R, pose.t, fx, fy, cx, cy, xi, alpha, floor_z=floor_z
        )
        for j, i in enumerate(idxs):
            dets[i].world_xy = world_xy[j]
            dets[i].valid = bool(valid[j])


def draw_regions(canvas, regions: List[Region], scale: float) -> None:
    for region in regions:
        pts = (region.polygon_m * scale).astype(np.int32).reshape(-1, 1, 2)
        color = TYPE_COLORS_BGR.get(region.type, (200, 200, 200))
        closed = region.type not in ("wall", "gate")
        cv2.polylines(canvas, [pts], isClosed=closed, color=color, thickness=2, lineType=cv2.LINE_AA)


def draw_legend(canvas, x0: int, y0: int, items) -> None:
    for i, (label, color) in enumerate(items):
        y = y0 + i * 22
        cv2.rectangle(canvas, (x0, y), (x0 + 16, y + 16), color, -1)
        cv2.putText(canvas, label, (x0 + 24, y + 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)


def draw_camera_markers(canvas, poses: Dict[str, Pose], scale: float, w_px: int, h_px: int) -> None:
    for cid in CAM_IDS:
        p = poses[cid]
        px, py = int(p.t[0] * scale), int(p.t[1] * scale)
        if 0 <= px < w_px and 0 <= py < h_px:
            cv2.circle(canvas, (px, py), 9, (0, 0, 255), -1)
            cv2.circle(canvas, (px, py), 11, (255, 255, 255), 2)
            cv2.putText(canvas, cid, (px + 14, py - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)


def main() -> None:
    sam_csv_p = Path(SAM_CSV_PATH)
    out_dir_p = Path(OUTPUT_DIR)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    img_w = IMAGE_WIDTH_PX

    print("floorplan.json 読み込み中...")
    regions, cameras_json, real_w, real_h = load_floorplan(FLOORPLAN_JSON_PATH)
    cameras_json = apply_camera_overrides(cameras_json)
    poses = resolve_poses(cameras_json, root="Area1", fmt=ROT_FMT)

    print(f"使用する点: {'胴体重心(脚除去後)' if USE_TORSO_CENTROID else '生のマスク重心'}")
    print("SAM重心の結果を読み込み中...")
    dets = load_sam_results(sam_csv_p)
    tile_w = assign_cameras(dets, img_w, NUM_CAMERAS)
    print(f"  検出数: {len(dets)} (タイル幅 {tile_w:.1f}px x {NUM_CAMERAS}台)")

    project_sam_centroids(dets, poses, tile_w)
    n_valid = sum(d.valid for d in dets)
    print(f"  床面に投影できた検出数: {n_valid} / {len(dets)}")

    grouped: Dict[str, List[SamDetection]] = defaultdict(list)
    for d in dets:
        if d.valid:
            grouped[d.name].append(d)
    merged_points = {name: np.array([d.world_xy for d in ds]).mean(axis=0) for name, ds in grouped.items()}
    print(f"  ユニークな牛の数: {len(merged_points)}")

    scale = 40  # px per meter
    margin = 220
    w_px = int(real_w * scale) + margin
    h_px = int(real_h * scale) + 40

    # ------------------------------------------------------------------
    # 全体平均版
    # ------------------------------------------------------------------
    canvas = np.zeros((h_px, w_px, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30)
    draw_regions(canvas, regions, scale)
    draw_camera_markers(canvas, poses, scale, w_px, h_px)

    for name, xy in merged_points.items():
        px, py = int(xy[0] * scale), int(xy[1] * scale)
        if 0 <= px < w_px and 0 <= py < h_px:
            cv2.circle(canvas, (px, py), 6, (255, 255, 255), -1)
            cv2.circle(canvas, (px, py), 7, (0, 0, 0), 1)
            cv2.putText(canvas, name.replace("cow", ""), (px + 6, py + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1, cv2.LINE_AA)

    draw_legend(canvas, int(real_w * scale) + 15, 20,
                list(TYPE_COLORS_BGR.items()) + [("camera", (0, 0, 255)), ("cow (SAM centroid)", (255, 255, 255))])

    out_path = out_dir_p / "bev_sam_centroid.png"
    cv2.imwrite(str(out_path), canvas)
    print(f"保存しました: {out_path}")

    # ------------------------------------------------------------------
    # カメラ別 + ズレ可視化版
    # ------------------------------------------------------------------
    canvas2 = np.zeros((h_px, w_px, 3), dtype=np.uint8)
    canvas2[:] = (30, 30, 30)
    draw_regions(canvas2, regions, scale)
    draw_camera_markers(canvas2, poses, scale, w_px, h_px)

    for name, ds in grouped.items():
        mean_xy = merged_points[name]
        mpx, mpy = int(mean_xy[0] * scale), int(mean_xy[1] * scale)
        for d in ds:
            cam_id = CAM_IDS[d.cam_index]
            color = CAM_COLORS_BGR.get(cam_id, (200, 200, 200))
            px, py = int(d.world_xy[0] * scale), int(d.world_xy[1] * scale)
            if 0 <= px < w_px and 0 <= py < h_px:
                if 0 <= mpx < w_px and 0 <= mpy < h_px:
                    cv2.line(canvas2, (mpx, mpy), (px, py), color, 1, cv2.LINE_AA)
                cv2.circle(canvas2, (px, py), 5, color, -1)
                cv2.circle(canvas2, (px, py), 6, (0, 0, 0), 1)
        if 0 <= mpx < w_px and 0 <= mpy < h_px:
            cv2.drawMarker(canvas2, (mpx, mpy), (255, 255, 255),
                           markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
            cv2.putText(canvas2, name.replace("cow", ""), (mpx + 8, mpy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1, cv2.LINE_AA)

    legend_items = [(cid, CAM_COLORS_BGR[cid]) for cid in CAM_IDS] + [("mean (x mark)", (255, 255, 255))]
    draw_legend(canvas2, int(real_w * scale) + 15, 20, legend_items)

    out_path2 = out_dir_p / "bev_sam_centroid_per_camera.png"
    cv2.imwrite(str(out_path2), canvas2)
    print(f"保存しました: {out_path2}")

    # ------------------------------------------------------------------
    # ズレ(平均からの距離)ランキング + CSV出力
    # ------------------------------------------------------------------
    print("\n--- ズレ(平均からの距離)が大きい検出 TOP10 ---")
    rows = []
    for name, ds in grouped.items():
        mean_xy = merged_points[name]
        for d in ds:
            cam_id = CAM_IDS[d.cam_index]
            dist = float(np.linalg.norm(d.world_xy - mean_xy))
            rows.append((dist, name, cam_id, d.world_xy, mean_xy))
    rows.sort(key=lambda r: -r[0])
    for dist, name, cam_id, wxy, mxy in rows[:10]:
        print(f"  {name:10s} {cam_id:10s} world=({wxy[0]:.2f},{wxy[1]:.2f}) "
              f"mean=({mxy[0]:.2f},{mxy[1]:.2f}) dist={dist:.3f}m")

    csv_path = out_dir_p / "bev_sam_centroid_positions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["cow_name", "world_x_m", "world_y_m", "num_camera_views"])
        for name, xy in sorted(merged_points.items()):
            writer.writerow([name, round(float(xy[0]), 3), round(float(xy[1]), 3), len(grouped[name])])
    print(f"保存しました: {csv_path}")


if __name__ == "__main__":
    main()