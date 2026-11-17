"""
sam_centroid_height_gridsearch.py

SAM(Segment Anything)のbboxプロンプト由来マスク重心(sam_centroid_results.csv)を
footpointとして使う場合の、lying/standingそれぞれの高さ仮定と距離閾値をグリッド
サーチし、F1が最大になる組み合わせを探すスクリプト。

gt_center_height_gridsearch.py の「center方式(bbox中心)」と全く同じ評価の枠組み
(Mootrack_core.pyのHungarian割当+Union-Find+F1)を使い、入力点だけをbbox中心から
SAMマスク重心に差し替えている。ベースライン(center方式、lying=0.9/standing=1.3、
F1=0.694)とそのまま比較できる。

--- 事前準備 ---
sam_centroid_pilot.py を先に実行し、sam_centroid_results.csv を作っておくこと。

--- 使い方 ---
    python sam_centroid_height_gridsearch.py
(パスや胴体重心を使うかどうかは、全て下の設定セクションに直書きしてあるので、
 引数なしで実行する)
"""

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from Mootrack_core import (
    NUM_CAMERAS, CAM_IDS, ROT_FMT,
    Pose, get_intrinsics,
    load_floorplan_cameras, apply_camera_overrides, resolve_poses,
    unproject_to_floor,
    compute_camera_assignments_by_distance,
    build_keys_and_name_lookup, evaluate_merge_accuracy,
)

# ---------------------------------------------------------------------------
# 設定 (パスが違う場合はここを書き換える。gt_center_height_gridsearch.pyと同じ値)
# ---------------------------------------------------------------------------

SAM_CSV_PATH = "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/make_bev/sam_centroid_test/sam_centroid_results.csv"
FLOORPLAN_JSON_PATH = "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/calibration/Farm2_Area_1_Artboard_config_.json"
IMAGE_WIDTH_PX = 8832  # 000000.jpg (3カメラ分ステッチ済み画像)の幅

# 生の重心ではなく、脚を除去した胴体重心(torso_centroid_x/y)を使う場合はTrueにする
USE_TORSO_CENTROID = False

# ---------------------------------------------------------------------------
# ★ グリッドサーチの探索範囲(center方式のグリッドサーチと同じ範囲) ★
# ---------------------------------------------------------------------------
HEIGHT_CANDIDATES_M = np.round(np.arange(0.10, 1.51, 0.10), 2).tolist()
DIST_THRESHOLDS_M = np.round(np.arange(0.10, 3.01, 0.10), 2).tolist()

TOP_N_RESULTS = 15


def load_sam_results(csv_path: Path, use_torso: bool) -> List[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    x_col = "torso_centroid_x" if use_torso else "centroid_x"
    y_col = "torso_centroid_y" if use_torso else "centroid_y"
    if use_torso and x_col not in rows[0]:
        raise KeyError(
            f"{x_col} 列がCSVにありません。torso_centroidに対応した"
            f"新しいsam_centroid_pilot.pyで再生成したCSVを使ってください。"
        )
    objects = []
    for r in rows:
        objects.append({
            "name": r["name"],
            "pose": r["pose"] or None,
            "xmin": float(r["xmin"]), "ymin": float(r["ymin"]),
            "xmax": float(r["xmax"]), "ymax": float(r["ymax"]),
            "centroid_x": float(r[x_col]), "centroid_y": float(r[y_col]),
        })
    return objects


def build_sam_world_points(
    objects: List[dict],
    poses: Dict[str, Pose],
    image_width: float,
    lying_height_m: float,
    standing_height_m: float,
) -> Dict[str, List[Tuple[str, np.ndarray]]]:
    tile_w = image_width / NUM_CAMERAS
    per_camera: Dict[str, List[Tuple[str, np.ndarray]]] = {cid: [] for cid in CAM_IDS}

    for obj in objects:
        # カメラ振り分けはbbox中心基準(center方式・make_bev_sam_centroid.pyと同じ考え方)
        center_x_full = (obj["xmin"] + obj["xmax"]) / 2.0
        cam_idx = int(center_x_full // tile_w)
        cam_idx = max(0, min(cam_idx, NUM_CAMERAS - 1))
        cam_id = CAM_IDS[cam_idx]

        fx, fy, cx, cy, xi, alpha = get_intrinsics(cam_id)
        # 実際に投影する点はbbox中心ではなくSAMマスク重心
        local_x = obj["centroid_x"] - cam_idx * tile_w
        local_y = obj["centroid_y"]
        pts_px = np.array([[local_x, local_y]], dtype=np.float64)

        height_m = lying_height_m if obj.get("pose") == "lying" else standing_height_m

        cam_pose = poses[cam_id]
        world_xy, valid = unproject_to_floor(
            pts_px, cam_pose.R, cam_pose.t, fx, fy, cx, cy, xi, alpha,
            floor_z=-height_m,
        )
        if not valid[0]:
            continue

        per_camera[cam_id].append((obj["name"], world_xy[0]))

    return per_camera


def evaluate_one_config(
    objects: List[dict],
    poses: Dict[str, Pose],
    image_width: float,
    lying_h: float,
    standing_h: float,
) -> List[dict]:
    per_camera = build_sam_world_points(objects, poses, image_width, lying_h, standing_h)
    assignments, gt_same_pairs = compute_camera_assignments_by_distance(per_camera)
    all_keys, name_lookup = build_keys_and_name_lookup(per_camera)

    results = []
    for thr in DIST_THRESHOLDS_M:
        accepted_pairs = [
            ((a.cam_i, a.idx_i), (a.cam_j, a.idx_j))
            for a in assignments if a.dist <= thr
        ]
        r = evaluate_merge_accuracy(all_keys, gt_same_pairs, accepted_pairs, name_lookup)
        results.append({
            "lying_h": lying_h, "standing_h": standing_h, "dist_thr": thr,
            "tp": r["tp"], "fn": r["fn"], "fp": r["fp"],
            "recall": r["recall"], "precision": r["precision"], "f1": r["f1"],
        })
    return results


def main():
    print("floorplan.json 読み込み中...")
    cameras_json = load_floorplan_cameras(FLOORPLAN_JSON_PATH)
    cameras_json = apply_camera_overrides(cameras_json)
    poses = resolve_poses(cameras_json, root="Area1", fmt=ROT_FMT)

    csv_path = Path(SAM_CSV_PATH)
    if not csv_path.exists():
        raise FileNotFoundError(f"SAM結果CSVが見つかりません: {csv_path}")
    print(f"使用する点: {'胴体重心(脚除去後)' if USE_TORSO_CENTROID else '生のマスク重心'}")
    objects = load_sam_results(csv_path, USE_TORSO_CENTROID)

    n_combos = len(HEIGHT_CANDIDATES_M) ** 2
    print(f"評価対象: {csv_path} ({len(objects)}件)")
    print(f"高さ候補: {HEIGHT_CANDIDATES_M}")
    print(f"距離閾値候補: {len(DIST_THRESHOLDS_M)}件")
    print(f"探索する(lying_h, standing_h)の組み合わせ数: {n_combos}件 "
          f"(× 距離閾値{len(DIST_THRESHOLDS_M)}件 = 総評価数 {n_combos * len(DIST_THRESHOLDS_M)}件)")
    print("グリッドサーチ中...\n")

    all_results: List[dict] = []
    done = 0
    for lying_h in HEIGHT_CANDIDATES_M:
        for standing_h in HEIGHT_CANDIDATES_M:
            all_results.extend(evaluate_one_config(objects, poses, IMAGE_WIDTH_PX, lying_h, standing_h))
            done += 1
            if done % 20 == 0:
                print(f"  進捗: {done}/{n_combos} 組み合わせ完了")

    valid_results = [r for r in all_results if r["f1"] == r["f1"]]  # NaN != NaN を利用
    valid_results.sort(key=lambda r: -r["f1"])

    print(f"\n=== F1上位{TOP_N_RESULTS}件 ===")
    print(f"{'lying_h':>8} | {'standing_h':>10} | {'距離閾値':>8} | {'recall':>18} | {'precision':>16} | {'F1':>8}")
    for r in valid_results[:TOP_N_RESULTS]:
        print(f"{r['lying_h']:>8.2f} | {r['standing_h']:>10.2f} | {r['dist_thr']:>8.2f} | "
              f"{r['recall']:>10.3f} ({r['tp']}/{r['tp']+r['fn']}) | "
              f"{r['precision']:>10.3f} (FP={r['fp']}) | {r['f1']:>8.3f}")

    best = valid_results[0]
    print(f"\n最良の組み合わせ: lying_h={best['lying_h']}, standing_h={best['standing_h']}, "
          f"距離閾値={best['dist_thr']} -> F1={best['f1']:.3f}")
    print("(center方式のベースライン: lying_h=0.9, standing_h=1.3, 距離閾値=2.8 -> F1=0.694)")

    uniform_results = [r for r in valid_results if r["lying_h"] == r["standing_h"]]
    if uniform_results:
        best_uniform = uniform_results[0]
        print(f"参考(姿勢区別なし・単一高さの中での最良): height={best_uniform['lying_h']}, "
              f"距離閾値={best_uniform['dist_thr']} -> F1={best_uniform['f1']:.3f}")


if __name__ == "__main__":
    main()