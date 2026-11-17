"""
evaluate_gt_distance_matching.py の「center方式(bbox中心1点)+ 距離閾値」の
パラメータ(lying/standingそれぞれの高さ、および距離閾値)をグリッドサーチし、
F1が最大になる組み合わせを探すスクリプト。

共通のカメラキャリブレーション・幾何変換・Hungarian割当・Union-Find・評価指標の
計算は mootrack_core.py に集約されているので、ここではグリッドサーチ固有の
ロジック(高さ候補の総当たり、結果の集計・表示)だけを持つ。

GTのbboxとpose(standing/lying)、および個体名(name)を使う。YOLOは使わない。
1フレーム(ANNOTATION_XML_PATH)のみを対象にした、カメラ間統合精度の検証。
"""

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

import sys
MOOTRACK_CODE_DIR = "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/research/calibration"
if MOOTRACK_CODE_DIR not in sys.path:
    sys.path.insert(0, MOOTRACK_CODE_DIR)

from Mootrack_core import (
    NUM_CAMERAS, CAM_IDS, ROT_FMT,
    Pose, get_intrinsics,
    load_floorplan_cameras, apply_camera_overrides, resolve_poses,
    unproject_to_floor, parse_voc_annotation,
    compute_camera_assignments_by_distance,
    build_keys_and_name_lookup, evaluate_merge_accuracy,
)


# ---------------------------------------------------------------------------
# 設定 (evaluate_gt_distance_matching.py と同じ値)
# ---------------------------------------------------------------------------

FLOORPLAN_JSON_PATH = "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/calibration/Farm2_Area_1_Artboard_config_.json"
ANNOTATION_XML_PATH = "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/annotation_process/3600_frames_result/annotations/000000.xml"

# ---------------------------------------------------------------------------
# ★ グリッドサーチの探索範囲 ★
# ---------------------------------------------------------------------------
HEIGHT_CANDIDATES_M = np.round(np.arange(0.75, 0.85, 0.10), 2).tolist()
DIST_THRESHOLDS_M = np.round(np.arange(0.50, 2.01, 0.10), 2).tolist()

TOP_N_RESULTS = 15  # 上位何件を表示するか


# ---------------------------------------------------------------------------
# bbox(アノテーション) -> カメラ振り分け -> World座標の1点(center方式)
# ---------------------------------------------------------------------------

def build_gt_world_points(
    objects: List[dict],
    poses: Dict[str, Pose],
    image_width: float,
    lying_height_m: float,
    standing_height_m: float,
) -> Dict[str, List[Tuple[str, np.ndarray]]]:
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

        if obj.get("pose") == "lying":
            height_m = lying_height_m
        else:
            # standing、または不明な場合はstanding側にフォールバック
            height_m = standing_height_m

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
    """
    1つの(lying_h, standing_h)について、全ての距離閾値でのrecall/precision/F1を計算して返す。
    """
    per_camera = build_gt_world_points(objects, poses, image_width, lying_h, standing_h)
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

    xml_path = Path(ANNOTATION_XML_PATH)
    if not xml_path.exists():
        raise FileNotFoundError(f"アノテーションXMLが見つかりません: {xml_path}")

    width, _height, objects = parse_voc_annotation(xml_path)

    n_combos = len(HEIGHT_CANDIDATES_M) ** 2
    print(f"評価対象: {xml_path}")
    print(f"高さ候補: {HEIGHT_CANDIDATES_M}")
    print(f"距離閾値候補: {len(DIST_THRESHOLDS_M)}件")
    print(f"探索する(lying_h, standing_h)の組み合わせ数: {n_combos}件 "
          f"(× 距離閾値{len(DIST_THRESHOLDS_M)}件 = 総評価数 {n_combos * len(DIST_THRESHOLDS_M)}件)")
    print("グリッドサーチ中...\n")

    all_results: List[dict] = []
    done = 0
    # for lying_h in HEIGHT_CANDIDATES_M:
    #     for standing_h in HEIGHT_CANDIDATES_M:
    #         all_results.extend(evaluate_one_config(objects, poses, width, lying_h, standing_h))
    #         done += 1
    #         if done % 20 == 0:
    #             print(f"  進捗: {done}/{n_combos} 組み合わせ完了")
    lying_h = 0.4
    standing_h = 0.75
    all_results.extend(evaluate_one_config(objects, poses, width, lying_h, standing_h))

    # NaNを除外してF1でソート
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

    # 参考: lying_h == standing_h (姿勢区別なしの単一高さ)の中での最良値
    uniform_results = [r for r in valid_results if r["lying_h"] == r["standing_h"]]
    if uniform_results:
        best_uniform = uniform_results[0]
        print(f"参考(姿勢区別なし・単一高さの中での最良): height={best_uniform['lying_h']}, "
              f"距離閾値={best_uniform['dist_thr']} -> F1={best_uniform['f1']:.3f}")
        if best["f1"] > best_uniform["f1"] + 1e-9:
            print("-> 姿勢別に高さを分けた方が良い結果です。")
        else:
            print("-> 姿勢を区別しない単一高さでも同等以上の結果です。posture分けの効果は限定的かもしれません。")


if __name__ == "__main__":
    main()