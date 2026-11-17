"""
analyze_pixel_to_world_sensitivity.py

魚眼カメラの光学中心(cx, cy)を通る水平な直線上に、一定間隔(RADIUS_STEP_PX)の
点群を用意し、各点で「その場所から1ピクセルだけ動かしたときに、世界座標(床面)上で
どれだけズレるか[m/px]」を計算する。

魚眼は画素→実世界の角度の対応が非線形で、画像中心付近と周辺部とで
「1ピクセルが表す実世界の距離」が大きく変わる。これを定量化することで、
「YOLOのbbox検出のわずかな画素誤差(1〜数px)が、画像内のどの位置でどれだけ
世界座標の誤差に増幅されるか」を具体的な数値として示せる
(周辺部ほどm/pxが大きいなら、周辺部の検出はわずかな画素誤差でも大きな
世界座標誤差になりやすい、ということが言える)。

横方向(光学中心を通る水平線に沿った方向、動径方向にほぼ相当)と縦方向
(動径に直交する方向におおよそ相当)の両方でm/pxを計算して並べて見られるので、
歪みが動径方向にだけ強いのか、接線方向にも出ているのかの手がかりにもなる。

--- 使い方 ---
    1. CAMERA_ID/HEIGHT_Mを調べたい条件に変更
    2. python analyze_pixel_to_world_sensitivity.py
    3. 標準出力の表、またはOUTPUT_CSVを確認
"""

import csv
import os
import sys
from pathlib import Path

import numpy as np

MOOTRACK_CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calibration")
if MOOTRACK_CODE_DIR not in sys.path:
    sys.path.insert(0, MOOTRACK_CODE_DIR)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from Mootrack_core import (
    ROT_FMT, get_intrinsics,
    load_floorplan_cameras, apply_camera_overrides, resolve_poses,
    unproject_to_floor,
)

FLOORPLAN_JSON_PATH = "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/calibration/Farm2_Area_1_Artboard_config_.json"

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------

CAMERA_ID = "Camera_2"
HEIGHT_M = 0.75  # 床から何m上を見ているかの仮定(standing個体相当、floor_z=-HEIGHT_M)

# 光学中心(cx,cy)を通る水平線上を、この間隔[px]でサンプリングする
RADIUS_STEP_PX = 50
# 中心からどこまで(片側)サンプリングするか[px]。画像の半分弱くらいが目安。
MAX_RADIUS_PX = 1400

OUTPUT_CSV = Path(__file__).parent / f"pixel_to_world_sensitivity_{CAMERA_ID}_h{HEIGHT_M}.csv"


def main():
    print("floorplan.json(cameras部分)読み込み中...")
    cameras_json = load_floorplan_cameras(FLOORPLAN_JSON_PATH)
    cameras_json = apply_camera_overrides(cameras_json)
    poses = resolve_poses(cameras_json, root="Area1", fmt=ROT_FMT)
    cam_pose = poses[CAMERA_ID]
    fx, fy, cx, cy, xi, alpha = get_intrinsics(CAMERA_ID)

    floor_z = -HEIGHT_M

    print(f"カメラ: {CAMERA_ID}  fx={fx} fy={fy} cx={cx} cy={cy} xi={xi} alpha={alpha}")
    print(f"仮定高さ: {HEIGHT_M}m (floor_z={floor_z})")
    print(f"サンプリング: 中心(cx,cy)を通る水平線上、半径-{MAX_RADIUS_PX}px〜+{MAX_RADIUS_PX}px、"
          f"{RADIUS_STEP_PX}px刻み\n")

    print(f"{'半径[px]':>10} | {'m/px(横=動径方向)':>18} | {'m/px(縦=接線方向)':>18} | 備考")

    rows = []
    radii = list(range(-MAX_RADIUS_PX, MAX_RADIUS_PX + 1, RADIUS_STEP_PX))
    for r in radii:
        base_x, base_y = cx + r, cy

        # [基準点, 横に1pxずらした点, 縦に1pxずらした点] をまとめて逆投影する
        pts_px = np.array([
            [base_x, base_y],
            [base_x + 1.0, base_y],
            [base_x, base_y + 1.0],
        ], dtype=np.float64)

        world_xy, valid = unproject_to_floor(
            pts_px, cam_pose.R, cam_pose.t, fx, fy, cx, cy, xi, alpha, floor_z=floor_z,
        )

        if not (valid[0] and valid[1] and valid[2]):
            print(f"{r:>10} | {'--':>18} | {'--':>18} | 魚眼の有効範囲外(逆投影失敗)")
            rows.append({"radius_px": r, "m_per_px_radial": None, "m_per_px_tangential": None, "note": "invalid"})
            continue

        base_world = world_xy[0]
        m_per_px_radial = float(np.linalg.norm(world_xy[1] - base_world))
        m_per_px_tangential = float(np.linalg.norm(world_xy[2] - base_world))

        print(f"{r:>10} | {m_per_px_radial:>18.5f} | {m_per_px_tangential:>18.5f} |")
        rows.append({
            "radius_px": r, "m_per_px_radial": m_per_px_radial,
            "m_per_px_tangential": m_per_px_tangential, "note": "",
        })

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["radius_px", "m_per_px_radial", "m_per_px_tangential", "note"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV保存: {OUTPUT_CSV}")
    print("\n見方: m/pxの値が大きいほど、その画像位置では1pxの検出誤差が大きな世界座標誤差に"
          "増幅されることを意味する。中心(半径0)付近より周辺部(|半径|が大きい)の方が"
          "値が大きければ、魚眼の周辺部ほど検出誤差の影響を受けやすい、という今回の仮説が"
          "数値で裏付けられる。")


if __name__ == "__main__":
    main()