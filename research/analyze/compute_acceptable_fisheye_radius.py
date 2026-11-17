"""
compute_acceptable_fisheye_radius.py

analyze_pixel_to_world_sensitivity.pyで求めた「1pxずれたときに世界座標(床面)上で
どれだけズレるか(m/px)」と、measure_yolo_bbox_pixel_error.pyで実測した
「YOLO検出が実際に何pxずれているか(半径帯ごと)」を組み合わせて、乳牛の実際の
体格(体長・体幅)から決めた「許容できる世界座標誤差」と比較することで、
「魚眼カメラのどこまでの範囲(半径)なら、検出位置のブレを許容できるか」を
定量的に求める。

考え方:
  1. ACCEPTABLE_WORLD_ERROR_M(許容できる世界座標誤差)を、牛の体格から決める。
     デフォルトは体幅の半分(ERROR_FRACTION_OF_BODY_WIDTH=0.5 × 体幅0.85m中央値
     ≒ 0.425m)。これは「位置推定の誤差が、牛の体の半分の幅を超えたら、
     どのくらいズレて牛を見ているか怪しくなってくる」という考え方。
  2. 検出のピクセル単位での誤差。2つのモードがある:
       USE_MEASURED_PIXEL_ERROR = True  → measure_yolo_bbox_pixel_error.pyで
         実測した「半径帯ごとのピクセル誤差(90%ile)」をそのまま使う
         (PIXEL_ERROR_BY_RADIUS_BIN_PX、下記)。半径によって誤差自体の大きさが
         変わることを反映できる、より実態に近いモード。
       USE_MEASURED_PIXEL_ERROR = False → PIXEL_UNCERTAINTY_PXという固定値を
         全ての半径で使う、以前のシンプルなモード。
  3. ある半径rでのm/px(1pxずれたときの世界座標上のズレ)に対して、
       expected_error_m = m_per_px(r) * pixel_error_px(r)
     を計算する(pixel_error_px(r)はモードに応じて実測値 or 固定値)。
  4. expected_error_mがACCEPTABLE_WORLD_ERROR_Mを超える半径を「信頼できない範囲」
     とみなし、超えない最大の半径を「許容できる魚眼カメラの範囲」として求める。

PIXEL_ERROR_BY_RADIUS_BIN_PXの値は、measure_yolo_bbox_pixel_error.pyの実測結果
(全カメラ合計、半径帯ごとの90%ile)をそのまま使っている。カメラ別・半径帯別の
クロス集計ではなく全カメラ合計の値なので、カメラごとの違い(例えばCamera_3が
他より悪い)はここには反映されていない。精度を上げたい場合は、measure_yolo_
bbox_pixel_error.py側をカメラ×半径帯のクロス集計に拡張してから、ここの
辞書を置き換えるとよい。

--- 使い方 ---
    1. CAMERA_ID/HEIGHT_M/ACCEPTABLE_WORLD_ERROR_M/USE_MEASURED_PIXEL_ERROR/
       PIXEL_UNCERTAINTY_PXを調べたい条件に変更
    2. python compute_acceptable_fisheye_radius.py
"""

import csv
import os
import sys
from pathlib import Path
from typing import Tuple

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
HEIGHT_M = 0.75  # 床から何m上を見ているかの仮定(floor_z=-HEIGHT_M)

# 乳牛の一般的な体格(上から見た占有範囲の目安)
COW_BODY_LENGTH_M = 2.25   # 体長(頭〜尻)の中央値、約2.0〜2.5mの中間
COW_BODY_WIDTH_M = 0.85    # 体幅(胴体の横幅)の中央値、約0.7〜1.0mの中間

# 許容できる世界座標誤差[m]。デフォルトは体幅の半分
# (誤差が牛の体の半分の幅を超えたら、位置推定として怪しくなる、という考え方)
ERROR_FRACTION_OF_BODY_WIDTH = 0.5
ACCEPTABLE_WORLD_ERROR_M = COW_BODY_WIDTH_M * ERROR_FRACTION_OF_BODY_WIDTH

# True: measure_yolo_bbox_pixel_error.pyの実測値(半径帯ごと)を使う
# False: PIXEL_UNCERTAINTY_PXという固定値を全ての半径で使う(単純な比較用)
USE_MEASURED_PIXEL_ERROR = True

# 固定値モード(USE_MEASURED_PIXEL_ERROR=False)のときに使う想定ピクセル誤差
PIXEL_UNCERTAINTY_PX = 1.0

# 実測モード(USE_MEASURED_PIXEL_ERROR=True)で使う、半径帯ごとのピクセル誤差[px]。
# measure_yolo_bbox_pixel_error.pyの実測結果(全カメラ合計、90%ile)をそのまま使用。
# キーはビンの開始半径[px](RADIUS_BIN_PX刻み)、値はそのビンのピクセル誤差[px]。
RADIUS_BIN_PX = 300
PIXEL_ERROR_BY_RADIUS_BIN_PX = {
    0: 32.50,
    300: 11.76,
    600: 13.13,
    900: 11.60,
    1200: 68.87,
}

RADIUS_STEP_PX = 20
MAX_RADIUS_PX = 1400

OUTPUT_CSV = Path(__file__).parent / f"acceptable_fisheye_radius_{CAMERA_ID}_h{HEIGHT_M}.csv"


def unproject_one(cam_pose, fx, fy, cx, cy, xi, alpha, floor_z, x: float, y: float):
    pts_px = np.array([[x, y]], dtype=np.float64)
    world_xy, valid = unproject_to_floor(
        pts_px, cam_pose.R, cam_pose.t, fx, fy, cx, cy, xi, alpha, floor_z=floor_z,
    )
    return world_xy[0], bool(valid[0])


def pixel_error_for_radius(r_abs: float) -> Tuple[float, bool]:
    """半径(絶対値)に対応するピクセル誤差[px]と、実測範囲外で外挿したかどうかを返す。"""
    if not USE_MEASURED_PIXEL_ERROR:
        return PIXEL_UNCERTAINTY_PX, False

    bin_starts = sorted(PIXEL_ERROR_BY_RADIUS_BIN_PX.keys())
    max_bin_start = bin_starts[-1]
    if r_abs >= max_bin_start + RADIUS_BIN_PX:
        # 実測範囲の外側。最後のビンの値をそのまま使う(保守的な外挿)
        return PIXEL_ERROR_BY_RADIUS_BIN_PX[max_bin_start], True

    for start in reversed(bin_starts):
        if r_abs >= start:
            return PIXEL_ERROR_BY_RADIUS_BIN_PX[start], False
    return PIXEL_ERROR_BY_RADIUS_BIN_PX[bin_starts[0]], False


def main():
    print("floorplan.json(cameras部分)読み込み中...")
    cameras_json = load_floorplan_cameras(FLOORPLAN_JSON_PATH)
    cameras_json = apply_camera_overrides(cameras_json)
    poses = resolve_poses(cameras_json, root="Area1", fmt=ROT_FMT)
    cam_pose = poses[CAMERA_ID]
    fx, fy, cx, cy, xi, alpha = get_intrinsics(CAMERA_ID)
    floor_z = -HEIGHT_M

    mode_label = "実測(半径帯ごと)" if USE_MEASURED_PIXEL_ERROR else f"固定値({PIXEL_UNCERTAINTY_PX}px)"
    print(f"カメラ: {CAMERA_ID}  fx={fx} fy={fy} cx={cx} cy={cy} xi={xi} alpha={alpha}")
    print(f"仮定高さ: {HEIGHT_M}m")
    print(f"牛の体格: 体長={COW_BODY_LENGTH_M}m 体幅={COW_BODY_WIDTH_M}m")
    print(f"許容できる世界座標誤差: {ACCEPTABLE_WORLD_ERROR_M:.3f}m "
          f"(体幅×{ERROR_FRACTION_OF_BODY_WIDTH})")
    print(f"ピクセル誤差モード: {mode_label}\n")
    print(f"{'半径[px]':>10} | {'px誤差':>8} | {'m/px':>10} | {'想定誤差[m]':>14} | {'許容比[%]':>10} | 判定")

    rows = []
    radii = list(range(-MAX_RADIUS_PX, MAX_RADIUS_PX + 1, RADIUS_STEP_PX))

    for r in radii:
        base_x, base_y = cx + r, cy

        p_base, v_base = unproject_one(cam_pose, fx, fy, cx, cy, xi, alpha, floor_z, base_x, base_y)
        p_next, v_next = unproject_one(cam_pose, fx, fy, cx, cy, xi, alpha, floor_z, base_x + 1.0, base_y)

        if not (v_base and v_next):
            print(f"{r:>10} | {'--':>8} | {'--':>10} | {'--':>14} | {'--':>10} | 魚眼の有効範囲外")
            rows.append({"radius_px": r, "pixel_error_px": None, "m_per_px": None,
                         "expected_error_m": None, "ratio_pct": None, "judgement": "invalid"})
            continue

        m_per_px = float(np.linalg.norm(p_next - p_base))
        pixel_error_px, extrapolated = pixel_error_for_radius(abs(r))
        expected_error_m = m_per_px * pixel_error_px
        ratio_pct = expected_error_m / ACCEPTABLE_WORLD_ERROR_M * 100.0

        judgement = "OK" if expected_error_m <= ACCEPTABLE_WORLD_ERROR_M else "NG(誤差過大)"
        if extrapolated:
            judgement += "(外挿)"
        print(f"{r:>10} | {pixel_error_px:>8.2f} | {m_per_px:>10.5f} | {expected_error_m:>14.5f} | "
              f"{ratio_pct:>10.1f} | {judgement}")
        rows.append({"radius_px": r, "pixel_error_px": pixel_error_px, "m_per_px": m_per_px,
                     "expected_error_m": expected_error_m, "ratio_pct": ratio_pct, "judgement": judgement})

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["radius_px", "pixel_error_px", "m_per_px",
                                                "expected_error_m", "ratio_pct", "judgement"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV保存: {OUTPUT_CSV}")

    # 中心(半径0)から外側に向かって、最初にNGになる半径を左右それぞれ探す
    valid_rows = [r for r in rows if r["ratio_pct"] is not None]
    center_idx = next((i for i, r in enumerate(valid_rows) if r["radius_px"] == 0), None)
    if center_idx is not None:
        r_pos_limit = None
        for r in valid_rows[center_idx:]:
            if not r["judgement"].startswith("OK"):
                r_pos_limit = r["radius_px"]
                break
        r_neg_limit = None
        for r in reversed(valid_rows[:center_idx + 1]):
            if not r["judgement"].startswith("OK"):
                r_neg_limit = r["radius_px"]
                break

        print("\n=== 許容できる魚眼カメラの範囲(中心からの半径) ===")
        if r_neg_limit is not None:
            print(f"  負方向: 半径{r_neg_limit}px付近から許容誤差({ACCEPTABLE_WORLD_ERROR_M:.3f}m)を超える")
        else:
            print(f"  負方向: 探索範囲内(-{MAX_RADIUS_PX}px)では常にOK")
        if r_pos_limit is not None:
            print(f"  正方向: 半径{r_pos_limit}px付近から許容誤差({ACCEPTABLE_WORLD_ERROR_M:.3f}m)を超える")
        else:
            print(f"  正方向: 探索範囲内(+{MAX_RADIUS_PX}px)では常にOK")
        print(f"  (許容誤差={ACCEPTABLE_WORLD_ERROR_M:.3f}m[体幅{COW_BODY_WIDTH_M}m×{ERROR_FRACTION_OF_BODY_WIDTH}], "
              f"ピクセル誤差モード={mode_label}, HEIGHT_M={HEIGHT_M}mを前提とした結果)")


if __name__ == "__main__":
    main()