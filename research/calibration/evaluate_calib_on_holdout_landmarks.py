"""
evaluate_calib_on_holdout_landmarks.py

Mootrack_core.py側に既に組み込まれている較正済み上書き値
(apply_camera_overrides / get_intrinsics)を使って、Camera_1/2/3すべてを
較正済みパラメータで解決し、入力したピクセル座標を世界座標に変換した結果と、
floorplan.json上の正解座標とのズレ[m]を求める。
phase3_bev_multicam_*.py系のスクリプトと同じ較正の通し方(load_floorplan_cameras
-> apply_camera_overrides -> resolve_poses、get_intrinsicsで内部パラメータ取得)
をそのまま使っている。

--- 使い方 ---
1. LANDMARKSに、評価したい点を追加・編集する。
   region/vertex_idxでfloorplan.json上の正解座標を指定し、
   camera(例:"Camera_1")とpixel(そのカメラのタイル画像上の座標)を書く。
"""

import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

MOOTRACK_CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calibration")
if MOOTRACK_CODE_DIR not in sys.path:
    sys.path.insert(0, MOOTRACK_CODE_DIR)

from Mootrack_core import (
    ROT_FMT, get_intrinsics,
    load_floorplan, apply_camera_overrides, resolve_poses,
    unproject_to_floor,
)

FLOORPLAN_JSON_PATH = "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/calibration/Farm2_Area_1_Artboard_config_.json"


def find_region_vertex(regions, region_name: str, vertex_idx: int) -> Tuple[float, float]:
    for r in regions:
        if r.name == region_name:
            x, y = r.polygon_m[vertex_idx]
            return float(x), float(y)
    raise KeyError(f"region=\"{region_name}\" がfloorplan.jsonに見つかりません")


# ---------------------------------------------------------------------------
# 評価したいランドマーク。region/vertex_idxで正解座標(floorplan.json由来)を、
# camera/pixelで画像上の対応点を指定する。好きに追加・編集していい。
# ---------------------------------------------------------------------------
LANDMARKS: List[dict] = [
    {"region": "Feeding_Bay_2", "vertex_idx": 1, "camera": "Camera_1", "pixel": (2654,1141), "label": "Feeding_Bay_2右上(x=0.0)"},
    {"region": "Wall_10", "vertex_idx": 1, "camera": "Camera_1", "pixel": (2635,1309), "label": "Wall_10上"},
    {"region": "Wall_10", "vertex_idx": 0, "camera": "Camera_1", "pixel": (2621,1650), "label": "Wall_10下"},
    {"region": "Wall_2", "vertex_idx": 0, "camera": "Camera_2", "pixel": (2552,1432), "label": "Wall_2上"},
    {"region": "Wall_2", "vertex_idx": 1, "camera": "Camera_2", "pixel": (2417,2052), "label": "Wall_2下"},
    {"region": "Wall_6", "vertex_idx": 0, "camera": "Camera_2", "pixel": (2239,1809), "label": "Wall_6上"},
    {"region": "Wall_6", "vertex_idx": 1, "camera": "Camera_2", "pixel": (2303,1496) ,"label": "Wall_6下"},
    {"region": "Feeding_Bay_2", "vertex_idx": 1, "camera": "Camera_2", "pixel": (2346,1015), "label": "Feeding_Bay_2右上(x=30.90)"},
    {"region": "Wall_6", "vertex_idx": 1, "camera": "Camera_3", "pixel": (869,1494), "label": "Wall_6上"},
    {"region": "Wall_1", "vertex_idx": 0, "camera": "Camera_3", "pixel": (1117,1843) ,"label": "Wall_1下"},
    {"region": "Feeding_Bay_2", "vertex_idx": 1, "camera": "Camera_3", "pixel": (986,863), "label": "Feeding_Bay_1右上"},


    {"camera": "Camera_1", "region": "Wall_9", "vertex_idx": 1, "pixel": (1100, 1090), "label": "wall_9上側"},
    {"camera": "Camera_1", "region": "Wall_11", "vertex_idx": 1, "pixel": (887, 2065), "label": "wall_11上側"},
    {"camera": "Camera_1", "region": "Gate_1", "vertex_idx": 1, "pixel": (801, 2044), "label": "Gate_1左側"},
    {"camera": "Camera_1", "region": "Feeding_Bay_2", "vertex_idx": 0, "pixel": (777, 839), "label": "Feeding_Bay_2の左上側"},
    {"camera": "Camera_1", "region": "Wall_9", "vertex_idx": 0, "pixel": (1093, 1859), "label": "Wall_9下側"},
    {"camera": "Camera_1", "region": "Feeding_Bay_3", "vertex_idx": 1, "pixel": (2760, 1231), "label": "Feeding_Bay_3の右上"},
    {"camera": "Camera_1", "region": "Wall_12", "vertex_idx": 1, "pixel": (2665, 1755), "label": "Wall_12の上"},
    {"camera": "Camera_1", "region": "Wall_7", "vertex_idx": 0, "pixel": (2726, 1768), "label": "Wall_7の右"},
 
    {"camera": "Camera_2", "region": "Feeding_Bay_2", "vertex_idx": 0, "pixel": (370, 948), "label": "Feeding_Bay_2の左上"},
    {"camera": "Camera_2", "region": "Wall_9", "vertex_idx": 1, "pixel": (382, 1130), "label": "Wall_9の上"},
    {"camera": "Camera_2", "region": "Gate_1", "vertex_idx": 1, "pixel": (272, 1637), "label": "Gate_1の左"},
    {"camera": "Camera_2", "region": "Gate_1", "vertex_idx": 0, "pixel": (289, 1646), "label": "Gate_1の右"},
    {"camera": "Camera_2", "region": "Wall_11", "vertex_idx": 0, "pixel": (297, 1779), "label": "Wall_11の下"},
    {"camera": "Camera_2", "region": "Wall_10", "vertex_idx": 1, "pixel": (2148, 1232), "label": "Wall_10の上"},
    {"camera": "Camera_2", "region": "Wall_10", "vertex_idx": 0, "pixel": (2042, 1842), "label": "Wall_10の下"},
    {"camera": "Camera_2", "region": "Wall_12", "vertex_idx": 1, "pixel": (2234, 2041), "label": "Wall_12の上"},
    {"camera": "Camera_2", "region": "Wall_12", "vertex_idx": 0, "pixel": (2163, 2213), "label": "Wall_12の下"},
    {"camera": "Camera_2", "region": "Feeding_Bay_3", "vertex_idx": 1, "pixel": (2643, 1282), "label": "Feeding_Bay_3の右上"},
    {"camera": "Camera_2", "region": "Wall_3", "vertex_idx": 1, "pixel": (2467, 1383), "label": "Wall_3左"},
    {"camera": "Camera_2", "region": "Wall_4", "vertex_idx": 1, "pixel": (2307, 2082), "label": "Wall_4左"},
 
    {"camera": "Camera_3", "region": "Feeding_Bay_2", "vertex_idx": 0, "pixel": (185, 1476), "label": "Feeding_Bay_2の左上"},
    {"camera": "Camera_3", "region": "Gate_1", "vertex_idx": 1, "pixel": (269, 1931), "label": "Gate_1の左"},
    {"camera": "Camera_3", "region": "Wall_10", "vertex_idx": 1, "pixel": (682, 1350), "label": "Wall_10の上"},
    {"camera": "Camera_3", "region": "Wall_10", "vertex_idx": 0, "pixel": (791, 1896), "label": "Wall_10の下"},
    {"camera": "Camera_3", "region": "Wall_12", "vertex_idx": 1, "pixel": (1222, 2128), "label": "Wall_12の上"},
    {"camera": "Camera_3", "region": "Wall_12", "vertex_idx": 0, "pixel": (1302, 2303), "label": "Wall_12の下"},
    {"camera": "Camera_3", "region": "Wall_3", "vertex_idx": 1, "pixel": (1265, 1133), "label": "Wall_3左"},
    {"camera": "Camera_3", "region": "Wall_4", "vertex_idx": 1, "pixel": (1515, 2217), "label": "Wall_4左"},
    {"camera": "Camera_3", "region": "Feeding_Bay_3", "vertex_idx": 1, "pixel": (2109, 779), "label": "Feeding_Bay_3の右上"},
    {"camera": "Camera_3", "region": "Wall_14", "vertex_idx": 0, "pixel": (1692, 731), "label": "Wall_14の上"},
    {"camera": "Camera_3", "region": "Milking_Station_2", "vertex_idx": 0, "pixel": (2141, 959), "label": "Milking_Station_2の左上"},
    {"camera": "Camera_3", "region": "Milking_Station_2", "vertex_idx": 3, "pixel": (2293, 1838), "label": "Milking_Station_2の左下"},
]


def main():
    print("floorplan.json 読み込み中...")
    regions, cameras_json_raw, real_w, real_h = load_floorplan(FLOORPLAN_JSON_PATH)
    print(f"床面サイズ: {real_w}m x {real_h}m / 区画数: {len(regions)}")

    # Mootrack_core.py側に組み込まれている較正済み上書き値をCamera_1/2/3すべてに適用
    cameras_json = apply_camera_overrides(cameras_json_raw)
    poses = resolve_poses(cameras_json, root="Area1", fmt=ROT_FMT)

    print("\nカメラ姿勢(較正済み上書き値適用後):")
    for cid in ["Camera_1", "Camera_2", "Camera_3"]:
        R, t = poses[cid].R, poses[cid].t
        fx, fy, cx, cy, xi, alpha = get_intrinsics(cid)
        print(f"  {cid}: t={t.round(3)}  xi={xi}  alpha={alpha}")

    print(f"\n=== ランドマークごとの誤差 ===")
    print(f"  {'camera':>8} | {'region/vertex':>24} | {'誤差[m]':>9} | 備考")

    rows = []
    for lm in LANDMARKS:
        cam_id = lm["camera"]
        cam_pose = poses[cam_id]
        fx, fy, cx, cy, xi, alpha = get_intrinsics(cam_id)
        pts_px = np.array([lm["pixel"]], dtype=np.float64)
        world_gt = np.array([find_region_vertex(regions, lm["region"], lm["vertex_idx"])], dtype=np.float64)

        world_xy, valid = unproject_to_floor(
            pts_px, cam_pose.R, cam_pose.t, fx, fy, cx, cy, xi, alpha, floor_z=0.0
        )
        err = float(np.linalg.norm(world_xy[0] - world_gt[0])) if valid[0] else float("nan")

        rows.append({"camera": cam_id, "region": lm["region"], "vertex_idx": lm["vertex_idx"], "err": err})
        region_vertex = f"{lm['region']}#{lm['vertex_idx']}"
        print(f"  {cam_id:>8} | {region_vertex:>24} | {err:>9.3f} | {lm['label']}")

    errs = np.array([r["err"] for r in rows])
    print(f"\n=== 全体サマリー ===")
    print(f"件数: {len(errs)} / 平均誤差: {np.nanmean(errs):.3f}m / 中央値: {np.nanmedian(errs):.3f}m / "
          f"最大: {np.nanmax(errs):.3f}m / RMSE: {np.sqrt(np.nanmean(errs ** 2)):.3f}m")

    print(f"\n=== カメラ別サマリー ===")
    by_cam: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        by_cam[r["camera"]].append(r["err"])
    for cid in ["Camera_1", "Camera_2", "Camera_3"]:
        if cid not in by_cam:
            continue
        e = np.array(by_cam[cid])
        print(f"{cid}: 件数={len(e)} 平均={np.nanmean(e):.3f}m 最大={np.nanmax(e):.3f}m "
              f"RMSE={np.sqrt(np.nanmean(e ** 2)):.3f}m")


if __name__ == "__main__":
    main()