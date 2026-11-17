"""
visualize_curved_valid_zones.py

compute_acceptable_fisheye_radius.pyで求める「許容できる魚眼カメラの範囲
(画像上の半径R_max)」を使って、各カメラの「信頼できる範囲」を床面(世界座標)上の
曲線として描画し、既存のget_adjacent_zones(隣接カメラのワールドx座標の中点を
基準にした直線境界)とどれくらい形が違うかを可視化する。

今までの境界は「ワールドx座標の中点 ± OVERLAP_HALF_WIDTH_M」という、x軸に垂直な
帯(直線)だった。これはシンプルだが、実際の検出誤差は画像上の半径(光学中心からの
距離)に依存して非線形に増幅される(周辺部ほど誤差が大きい)ので、本来の
「信頼できる範囲」は、各カメラの画像上の円(半径R_max)が、傾いたカメラ姿勢を
通して床に投影された形になり、直線ではなく曲線(おおよそ楕円に近い形)になるはず。

このスクリプトはまだ本番の融合パイプラインには組み込まず、まず「直線境界」と
「曲線境界」の違いを可視化で確認するためのもの
(本番組み込みは、この結果を見てから判断する)。

出力: OUTPUT_DIR/curved_valid_zones_vs_straight_boundary.png

--- 使い方 ---
    python visualize_curved_valid_zones.py
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np

MOOTRACK_CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calibration")
if MOOTRACK_CODE_DIR not in sys.path:
    sys.path.insert(0, MOOTRACK_CODE_DIR)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

RENDER_SCRIPT_DIR = os.path.join(THIS_DIR, "..", "phase3_multi_camera")
if RENDER_SCRIPT_DIR not in sys.path:
    sys.path.insert(0, RENDER_SCRIPT_DIR)

from Mootrack_core import (
    CAM_IDS, ROT_FMT, get_intrinsics,
    load_floorplan, apply_camera_overrides, resolve_poses,
    unproject_to_floor,
)

# compute_acceptable_fisheye_radius.py側の許容誤差の判定ロジック・設定値を再利用
# (ロジックの二重管理を避けるため)
from compute_acceptable_fisheye_radius import (
    HEIGHT_M, ACCEPTABLE_WORLD_ERROR_M,
    pixel_error_for_radius, unproject_one,
    RADIUS_STEP_PX, MAX_RADIUS_PX,
)

# render_bev_frames_yolo_zonepose.py側のフロアプラン描画・既存の直線境界ロジック・
# 設定値をそのまま再利用
from render_bev_frames_yolo_zonepose import (
    FLOORPLAN_JSON_PATH, OUTPUT_DIR,
    OVERLAP_HALF_WIDTH_M,
    get_adjacent_zones, draw_regions,
    CAM_ZONE_COLORS_BGR,
)

# 曲線境界を何分割でサンプリングするか(多いほど滑らかな曲線になる)
N_ANGLES = 144


def find_acceptable_radius(cam_pose, fx, fy, cx, cy, xi, alpha) -> float:
    """compute_acceptable_fisheye_radius.pyと同じ判定ロジックで、画像中心から
    外側に向かって最初にNG(許容誤差超え)になる半径[px]を求める。
    横方向のプラス側・マイナス側それぞれで求め、厳しい方(小さい方)を採用する
    (円として一律に使うための単純化)。"""
    floor_z = -HEIGHT_M

    def ok_at(r_signed: float) -> bool:
        base_x, base_y = cx + r_signed, cy
        p_base, v_base = unproject_one(cam_pose, fx, fy, cx, cy, xi, alpha, floor_z, base_x, base_y)
        p_next, v_next = unproject_one(cam_pose, fx, fy, cx, cy, xi, alpha, floor_z, base_x + 1.0, base_y)
        if not (v_base and v_next):
            return False
        m_per_px = float(np.linalg.norm(p_next - p_base))
        pixel_error_px, _ = pixel_error_for_radius(abs(r_signed))
        expected_error_m = m_per_px * pixel_error_px
        return expected_error_m <= ACCEPTABLE_WORLD_ERROR_M

    r_pos_limit = float(MAX_RADIUS_PX)
    for r in range(0, MAX_RADIUS_PX + 1, RADIUS_STEP_PX):
        if not ok_at(r):
            r_pos_limit = float(r)
            break

    r_neg_limit = float(MAX_RADIUS_PX)
    for r in range(0, MAX_RADIUS_PX + 1, RADIUS_STEP_PX):
        if not ok_at(-r):
            r_neg_limit = float(r)
            break

    return min(r_pos_limit, r_neg_limit)


def camera_valid_zone_curve(cam_pose, fx, fy, cx, cy, xi, alpha, r_max_px: float, n_angles: int):
    """画像上で半径r_max_pxの円を一周サンプリングし、それぞれを床面に逆投影した
    世界座標の点列(閉曲線)を返す。"""
    floor_z = -HEIGHT_M
    pts_px = np.array([
        [cx + r_max_px * np.cos(2 * np.pi * i / n_angles),
         cy + r_max_px * np.sin(2 * np.pi * i / n_angles)]
        for i in range(n_angles)
    ], dtype=np.float64)
    world_xy, valid = unproject_to_floor(
        pts_px, cam_pose.R, cam_pose.t, fx, fy, cx, cy, xi, alpha, floor_z=floor_z,
    )
    return world_xy, valid


def main():
    print("floorplan.json 読み込み中...")
    regions, cameras_json, real_w, real_h = load_floorplan(FLOORPLAN_JSON_PATH)
    cameras_json = apply_camera_overrides(cameras_json)
    poses = resolve_poses(cameras_json, root="Area1", fmt=ROT_FMT)

    zones = get_adjacent_zones(poses, CAM_IDS, OVERLAP_HALF_WIDTH_M)

    scale = 40
    margin = 40
    w_px = int(real_w * scale) + margin
    h_px = int(real_h * scale) + margin
    if w_px % 2 != 0:
        w_px += 1
    if h_px % 2 != 0:
        h_px += 1
    canvas = np.full((h_px, w_px, 3), (30, 30, 30), dtype=np.uint8)
    draw_regions(canvas, regions, scale)

    # 既存の直線境界(帯)をオレンジで薄く描く(比較対象)
    for (cam_i, cam_j), (lo, hi) in zones.items():
        x0, x1 = int(lo * scale), int(hi * scale)
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x0, 0), (x1, h_px), (0, 165, 255), -1)
        cv2.addWeighted(overlay, 0.12, canvas, 0.88, 0, canvas)

    for cid in CAM_IDS:
        p = poses[cid]
        px, py = int(p.t[0] * scale), int(p.t[1] * scale)
        fx, fy, cx, cy, xi, alpha = get_intrinsics(cid)

        r_max = find_acceptable_radius(p, fx, fy, cx, cy, xi, alpha)
        print(f"{cid}: 許容半径(円として採用)={r_max:.0f}px")

        world_xy, valid = camera_valid_zone_curve(p, fx, fy, cx, cy, xi, alpha, r_max, N_ANGLES)
        color = CAM_ZONE_COLORS_BGR.get(cid, (255, 255, 255))
        poly_pts = [
            (int(x * scale), int(y * scale))
            for (x, y), v in zip(world_xy, valid) if v
        ]
        if len(poly_pts) >= 3:
            pts_arr = np.array(poly_pts, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts_arr], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
        else:
            print(f"  警告: {cid}の曲線境界が有効点不足で描けなかった(valid点={len(poly_pts)})")

        if 0 <= px < w_px and 0 <= py < h_px:
            cv2.circle(canvas, (px, py), 7, (0, 0, 255), -1)
            cv2.circle(canvas, (px, py), 9, (255, 255, 255), 2)
            cv2.putText(canvas, cid, (px + 12, py - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)

    out_path = OUTPUT_DIR / "curved_valid_zones_vs_straight_boundary.png"
    cv2.imwrite(str(out_path), canvas)
    print(f"\n保存しました: {out_path}")
    print("オレンジの帯=既存の直線境界(OVERLAP_HALF_WIDTH_M)、色付きの輪郭線=各カメラの曲線境界(許容半径R_max)")


if __name__ == "__main__":
    main()