"""
make_bev_zone_pose_fusion.py

make_bev_pose_height.py(姿勢別高さでbboxを鳥瞰図に投影するだけのスクリプト)に、
エピポーラ拘束は使わずに以下3つの制約を追加した「個体統合」ロジックを組み込んだ版。

  1. 有効範囲外の除外:
     各カメラには「専属範囲+隣接カメラとの重なり範囲」までの有効範囲を割り当てる
     (例: Camera_1は(-inf, Camera_1-2重なりの右端], Camera_2は両側の重なりの間,
     Camera_3は[Camera_2-3重なりの左端, +inf))。魚眼は視野が広いため、あるカメラが
     自分の有効範囲を大きく超えた遠方の牛をたまたま検出してしまうことがあるが、
     そのような検出は周辺視野の歪みで誤差が大きく信頼できないため、統合処理に入れる前に
     除外する。
  2. 隣接カメラ制約:
     Camera_1-Camera_2、Camera_2-Camera_3の隣接ペアだけを統合対象とし、
     物理的に重なりが無いはずのCamera_1-Camera_3は候補から除外する。
     (位置による絞り込みはしない。魚眼視野が広いため、境界から離れた位置の牛でも
     隣接カメラ両方に正しく映り込んでいることがあり、それらも統合対象にする)
  3. 姿勢クラス制約:
     lying/standingのラベルが一致する検出同士だけを統合候補にする。
     どちらかのpose情報が無い場合は、この制約はスキップする(過剰に絞り込まないため)。

これら3つの制約で候補を絞り込んだ上で、Hungarian割当(1-距離をコストにした最適割当)+
Union-Findで個体クラスタを作る、という枠組み自体はgt_center_height_gridsearch.pyや
Mootrack_core.pyと同じ。評価はMootrack_core.evaluate_merge_accuracy()をそのまま使い、
recall/precision/F1を算出する(以前の制約無し版=F1=0.626, recall=50%との比較用)。

高さ仮定はmake_bev_pose_height.pyと同じ(lying=0.6m, standing=1.0m)。
エピポーラ拘束(レイの交差判定)は含まない。

出力(OUTPUT_DIR):
  - bev_zone_pose_fusion.png       : 統合後の個体位置(カメラ専属範囲は各カメラの色、
                                      重なり範囲での統合結果はグレー)+ 重なり範囲の帯を描画
  - bev_zone_pose_fusion_result.csv: 統合クラスタごとの座標・メンバー内訳・所属ゾーン
  - コンソールにrecall/precision/F1、tp/fn/fpの内訳、ゾーン別頭数を出力

--- 使い方 ---
    python make_bev_zone_pose_fusion.py
"""

import csv
import os
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

MOOTRACK_CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calibration")
if MOOTRACK_CODE_DIR not in sys.path:
    sys.path.insert(0, MOOTRACK_CODE_DIR)

from Mootrack_core import (
    NUM_CAMERAS, CAM_IDS, ROT_FMT,
    get_intrinsics,
    load_floorplan, apply_camera_overrides, resolve_poses,
    unproject_to_floor,
    parse_voc_annotation,
    union_find_clusters,
    evaluate_merge_accuracy,
)

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
FLOORPLAN_JSON_PATH = "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/calibration/Farm2_Area_1_Artboard_config_.json"
ANNOTATION_XML_PATH = "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/annotation_process/3600_frames_result/annotations/000000.xml"
OUTPUT_DIR = Path("/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/calibration")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LYING_HEIGHT_M = 0.5
STANDING_HEIGHT_M = 0.75
DEFAULT_HEIGHT_M = 0.8

DIST_THR_FUSION_M = 1.4          # 統合を認める距離閾値(gt_center_height_gridsearch.pyのベスト値)
OVERLAP_HALF_WIDTH_M = 1.5       # 隣接カメラ境界からの片側の幅(合計3m幅になる)

TYPE_COLORS_BGR = {
    "wall": (0, 200, 0), "cow_beds": (0, 140, 255), "feeding_bay": (255, 140, 0),
    "water": (255, 255, 0), "gate": (0, 0, 255), "milking_station": (255, 0, 255),
    "hay_area": (0, 255, 255), "cow_brushes": (255, 0, 180),
}

# カメラごとの「専属範囲」の色分け(重なり範囲の外側で、そのカメラ単独の担当域)
CAM_ZONE_COLORS_BGR = {
    "Camera_1": (255, 120, 60),   # 青系
    "Camera_2": (60, 200, 60),    # 緑系
    "Camera_3": (60, 140, 255),   # 橙系
}
OVERLAP_ZONE_COLOR_BGR = (220, 220, 220)  # 重なり範囲内の点はグレーで区別


def height_for(pose: Optional[str]) -> float:
    if pose == "lying":
        return LYING_HEIGHT_M
    if pose == "standing":
        return STANDING_HEIGHT_M
    return DEFAULT_HEIGHT_M


# ---------------------------------------------------------------------------
# 1. 各カメラの検出をワールド座標へ(姿勢別高さで逆投影)
# ---------------------------------------------------------------------------

def compute_world_points_with_pose(objects: List[dict], poses, img_w: float) -> Dict[str, List[dict]]:
    """cam_id -> [{"name":..., "pose":..., "world_xy":...}, ...]"""
    tile_w = img_w / NUM_CAMERAS
    by_cam_objs: Dict[str, List[dict]] = defaultdict(list)
    for obj in objects:
        center_x_full = (obj["xmin"] + obj["xmax"]) / 2.0
        cam_idx = int(center_x_full // tile_w)
        cam_idx = max(0, min(cam_idx, NUM_CAMERAS - 1))
        by_cam_objs[CAM_IDS[cam_idx]].append(obj)

    per_camera: Dict[str, List[dict]] = {cid: [] for cid in CAM_IDS}
    for cam_id, objs in by_cam_objs.items():
        cam_idx = CAM_IDS.index(cam_id)
        fx, fy, cx, cy, xi, alpha = get_intrinsics(cam_id)
        cam_pose = poses[cam_id]

        pts_px, heights = [], []
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
            if not ok:
                continue
            per_camera[cam_id].append({"name": obj["name"], "pose": obj["pose"], "world_xy": xy})

    return per_camera


# ---------------------------------------------------------------------------
# 2. 隣接カメラの重なり範囲(境界±OVERLAP_HALF_WIDTH_M)を求める
# ---------------------------------------------------------------------------

def get_adjacent_zones(poses, cam_ids: List[str], half_width: float) -> Dict[Tuple[str, str], Tuple[float, float]]:
    zones = {}
    for cam_i, cam_j in zip(cam_ids[:-1], cam_ids[1:]):
        x_i = poses[cam_i].t[0]
        x_j = poses[cam_j].t[0]
        boundary = (x_i + x_j) / 2.0
        zones[(cam_i, cam_j)] = (boundary - half_width, boundary + half_width)
        print(f"  {cam_i}-{cam_j} 境界: x={boundary:.2f}m -> 重なり範囲 [{boundary - half_width:.2f}, {boundary + half_width:.2f}]")
    return zones


# ---------------------------------------------------------------------------
# 2a. 各カメラの「有効範囲」(専属範囲+隣接する重なり範囲)を求め、
#     それより外側に投影された検出(魚眼の遠い周辺視野で誤差が大きい低品質な検出)を除外する
# ---------------------------------------------------------------------------

def get_camera_valid_ranges(cam_ids: List[str], zones: Dict[Tuple[str, str], Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
    """cam_id -> (lo, hi)。そのカメラの専属範囲+隣接する重なり範囲までを有効範囲とする。
    例: Camera_1は(-inf, Camera_1-2重なりの右端]、Camera_2は[Camera_1-2重なりの左端, Camera_2-3重なりの右端]、
    Camera_3は[Camera_2-3重なりの左端, +inf)。"""
    ordered_pairs = list(zip(cam_ids[:-1], cam_ids[1:]))
    ranges: Dict[str, Tuple[float, float]] = {}
    ranges[cam_ids[0]] = (-np.inf, zones[ordered_pairs[0]][1])
    ranges[cam_ids[-1]] = (zones[ordered_pairs[-1]][0], np.inf)
    for i in range(1, len(cam_ids) - 1):
        lo = zones[ordered_pairs[i - 1]][0]
        hi = zones[ordered_pairs[i]][1]
        ranges[cam_ids[i]] = (lo, hi)
    return ranges


def filter_out_of_range_detections(
    per_camera: Dict[str, List[dict]], valid_ranges: Dict[str, Tuple[float, float]],
) -> Dict[str, List[dict]]:
    """各カメラの検出のうち、そのカメラの有効範囲外に投影されたものを除外する。
    魚眼の広い視野でたまたま遠くの牛が写り込んでも、そのカメラの担当範囲を大きく
    超えて投影される場合は信頼できない検出として捨てる。"""
    filtered: Dict[str, List[dict]] = {cid: [] for cid in per_camera}
    n_dropped = 0
    for cam_id, items in per_camera.items():
        lo, hi = valid_ranges[cam_id]
        for it in items:
            if lo <= it["world_xy"][0] <= hi:
                filtered[cam_id].append(it)
            else:
                n_dropped += 1
    print(f"  有効範囲外として除外した検出数: {n_dropped}")
    return filtered


# ---------------------------------------------------------------------------
# 2b. ワールドx座標から「どのカメラの専属範囲か / どの重なり範囲か」を判定する
# ---------------------------------------------------------------------------

def classify_exclusive_zone(
    x: float, cam_ids: List[str], zones: Dict[Tuple[str, str], Tuple[float, float]],
) -> str:
    """cam_idsの並び順に沿って、xがどのカメラの専属範囲(カメラ名を返す)か、
    あるいはどの重なり範囲(overlap_CamA_CamB を返す)に入っているかを判定する。"""
    ordered_pairs = list(zip(cam_ids[:-1], cam_ids[1:]))
    band_ranges = [zones[pair] for pair in ordered_pairs]

    if x < band_ranges[0][0]:
        return cam_ids[0]
    for i, (lo, hi) in enumerate(band_ranges):
        if lo <= x <= hi:
            return f"overlap_{ordered_pairs[i][0]}_{ordered_pairs[i][1]}"
        if i < len(band_ranges) - 1 and hi < x < band_ranges[i + 1][0]:
            return cam_ids[i + 1]
    if x > band_ranges[-1][1]:
        return cam_ids[-1]
    return "unknown"


# ---------------------------------------------------------------------------
# 3. 隣接カメラ制約 + 重なり範囲制約 + 姿勢クラス制約で統合候補を絞り込み、
#    Hungarian割当(1-距離をコストに)で1対1のペアを選ぶ
# ---------------------------------------------------------------------------

def match_zone_pose_constrained(
    per_camera: Dict[str, List[dict]],
    poses,
    cam_ids: List[str],
    zones: Dict[Tuple[str, str], Tuple[float, float]],
    dist_thr: float,
) -> List[Tuple[Tuple[str, int], Tuple[str, int]]]:
    """隣接カメラペア(Camera_1-Camera_2, Camera_2-Camera_3)だけを比較対象にする。
    魚眼は視野が広く、境界から離れた位置の牛でも隣接カメラ両方に映り込むことがあるため、
    位置(重なり範囲3m以内かどうか)では候補を絞り込まない。zonesは可視化用の帯表示にのみ使う。"""
    accepted_pairs: List[Tuple[Tuple[str, int], Tuple[str, int]]] = []

    for cam_i, cam_j in zones.keys():
        items_i = per_camera[cam_i]
        items_j = per_camera[cam_j]
        if not items_i or not items_j:
            continue

        n_i, n_j = len(items_i), len(items_j)
        cost = np.full((n_i, n_j), 1e6)
        for a, it_i in enumerate(items_i):
            for b, it_j in enumerate(items_j):
                if it_i["pose"] is not None and it_j["pose"] is not None and it_i["pose"] != it_j["pose"]:
                    continue  # 姿勢クラス不一致は候補から除外
                d = float(np.linalg.norm(it_i["world_xy"] - it_j["world_xy"]))
                if d <= dist_thr:
                    cost[a, b] = d
                # 閾値を超える組は生の距離を入れず1e6のままにする。生の距離(数m〜十数m)を
                # 入れてしまうと、Hungarianが行列全体の合計コストを最小化しようとして、
                # 本来ペアであるはずの組(距離が近い)を差し置いて無関係な遠い組を選んでしまう
                # ことがある(閾値判定をHungarianの後段でやると起きる典型的な不具合)。

        row_idx, col_idx = linear_sum_assignment(cost)
        for a, b in zip(row_idx, col_idx):
            if cost[a, b] <= dist_thr:
                accepted_pairs.append(((cam_i, a), (cam_j, b)))

    return accepted_pairs


# ---------------------------------------------------------------------------
# 4. 評価用: 全カメラペア(隣接に限らず)でnameが一致する組を正解として列挙
# ---------------------------------------------------------------------------

def build_gt_same_pairs(per_camera: Dict[str, List[dict]]) -> List[Tuple[str, str, int, int, str]]:
    gt_same_pairs = []
    for cam_i, cam_j in combinations(per_camera.keys(), 2):
        for a, it_i in enumerate(per_camera[cam_i]):
            for b, it_j in enumerate(per_camera[cam_j]):
                if it_i["name"] == it_j["name"]:
                    gt_same_pairs.append((cam_i, cam_j, a, b, it_i["name"]))
    return gt_same_pairs


# ---------------------------------------------------------------------------
# デバッグ用: 隣接カメラ間で「本来同一個体のはず」のペアについて、
# 実際の距離・姿勢ラベルの一致状況・統合できたかどうかを直接表示する。
# 統合数が0件/少ない場合に、原因が(a)距離が閾値を超えている(キャリブレーション/
# 高さ仮定の誤差)のか、(b)姿勢ラベルの不一致で候補から弾かれているのか、
# (c)そもそもGTペア自体が隣接カメラ間に存在しないのか、を切り分けるためのもの。
# ---------------------------------------------------------------------------

def debug_print_adjacent_gt_pairs(
    per_camera: Dict[str, List[dict]],
    zones: Dict[Tuple[str, str], Tuple[float, float]],
    dist_thr: float,
) -> None:
    """match_zone_pose_constrained()と同じHungarian割当を内部でもう一度計算し、
    本来同一個体のはずのペアが、(a)距離超過、(b)姿勢不一致、(c)距離・姿勢は問題ないが
    Hungarianの1対1割当で他の候補に競り負けた、のどれで統合NGになっているかを
    区別して表示する。"""
    print("\n=== デバッグ: 隣接カメラ間のGT同一個体ペアの詳細 ===")
    n_shown = 0
    for cam_i, cam_j in zones.keys():
        items_i = per_camera[cam_i]
        items_j = per_camera[cam_j]
        if not items_i or not items_j:
            continue

        n_i, n_j = len(items_i), len(items_j)
        cost = np.full((n_i, n_j), 1e6)
        for a, it_i in enumerate(items_i):
            for b, it_j in enumerate(items_j):
                if it_i["pose"] is not None and it_j["pose"] is not None and it_i["pose"] != it_j["pose"]:
                    continue
                d = float(np.linalg.norm(it_i["world_xy"] - it_j["world_xy"]))
                if d <= dist_thr:
                    cost[a, b] = d  # match_zone_pose_constrained()と同じく、閾値超えは1e6のまま
        row_idx, col_idx = linear_sum_assignment(cost)
        assigned_b_for_a = {a: b for a, b in zip(row_idx, col_idx)}

        for a, it_i in enumerate(items_i):
            for b, it_j in enumerate(items_j):
                if it_i["name"] != it_j["name"]:
                    continue
                dist = float(np.linalg.norm(it_i["world_xy"] - it_j["world_xy"]))
                pose_match = (it_i["pose"] is None or it_j["pose"] is None or it_i["pose"] == it_j["pose"])
                won_slot = assigned_b_for_a.get(a) == b
                accepted = won_slot and pose_match and dist <= dist_thr

                if accepted:
                    status = "統合OK"
                elif not pose_match:
                    status = "統合NG(姿勢不一致)"
                elif dist > dist_thr:
                    status = "統合NG(距離超過)"
                elif not won_slot:
                    rival_b = assigned_b_for_a.get(a)
                    rival_dist = cost[a, rival_b] if rival_b is not None else float("nan")
                    rival_name = items_j[rival_b]["name"] if rival_b is not None else "?"
                    if rival_dist >= 1e5:
                        status = f"統合NG({cam_i}#{a}は有効な相手が無く未割当のまま)"
                    else:
                        status = (f"統合NG(Hungarianで{cam_j}#{rival_b}[{rival_name}]"
                                  f"(dist={rival_dist:.3f}m)に競り負け)")
                else:
                    status = "統合NG(不明)"

                print(f"  {it_i['name']:10s} {cam_i}#{a}({it_i['pose']}) - {cam_j}#{b}({it_j['pose']}) "
                      f"dist={dist:.3f}m thr={dist_thr}m pose_match={pose_match} -> {status}")
                n_shown += 1

    if n_shown == 0:
        print("  隣接カメラ間でnameが一致するGTペアが1件も見つかりませんでした"
              "(カメラへの振り分け・有効範囲フィルタ・annotation内容を確認してください)")


# ---------------------------------------------------------------------------
# BEV描画
# ---------------------------------------------------------------------------

def draw_regions(canvas, regions, scale: float):
    for region in regions:
        pts = (region.polygon_m * scale).astype(np.int32).reshape(-1, 1, 2)
        color = TYPE_COLORS_BGR.get(region.type, (200, 200, 200))
        closed = region.type not in ("wall", "gate")
        cv2.polylines(canvas, [pts], isClosed=closed, color=color, thickness=2, lineType=cv2.LINE_AA)


def main():
    print("floorplan.json 読み込み中...")
    regions, cameras_json, real_w, real_h = load_floorplan(FLOORPLAN_JSON_PATH)
    cameras_json = apply_camera_overrides(cameras_json)
    poses = resolve_poses(cameras_json, root="Area1", fmt=ROT_FMT)
    for cid in CAM_IDS:
        print(f"  {cid}: t={poses[cid].t.round(2)}")

    print("\n隣接カメラの重なり範囲(±{:.1f}m)を計算中...".format(OVERLAP_HALF_WIDTH_M))
    zones = get_adjacent_zones(poses, CAM_IDS, OVERLAP_HALF_WIDTH_M)

    print("\nannotation.xml 読み込み中...")
    img_w, img_h, objects = parse_voc_annotation(ANNOTATION_XML_PATH)
    per_camera = compute_world_points_with_pose(objects, poses, img_w)
    for cid in CAM_IDS:
        print(f"  {cid}: {len(per_camera[cid])}件(除外前)")

    print("\n各カメラの有効範囲(専属範囲+隣接する重なり範囲)外の検出を除外中...")
    valid_ranges = get_camera_valid_ranges(CAM_IDS, zones)
    for cid in CAM_IDS:
        lo, hi = valid_ranges[cid]
        print(f"  {cid} 有効範囲: [{lo:.2f}, {hi:.2f}]")
    per_camera = filter_out_of_range_detections(per_camera, valid_ranges)
    for cid in CAM_IDS:
        print(f"  {cid}: {len(per_camera[cid])}件(除外後)")

    print("\n隣接カメラ制約 + 姿勢クラス制約でマッチング中...")
    accepted_pairs = match_zone_pose_constrained(per_camera, poses, CAM_IDS, zones, DIST_THR_FUSION_M)
    print(f"  採用されたペア数: {len(accepted_pairs)}")

    debug_print_adjacent_gt_pairs(per_camera, zones, DIST_THR_FUSION_M)

    all_keys = [(cam_id, idx) for cam_id, items in per_camera.items() for idx in range(len(items))]
    name_lookup = {(cam_id, idx): items[idx]["name"] for cam_id, items in per_camera.items() for idx in range(len(items))}
    gt_same_pairs = build_gt_same_pairs(per_camera)

    result = evaluate_merge_accuracy(all_keys, gt_same_pairs, accepted_pairs, name_lookup)
    print(f"\n=== 評価結果(有効範囲外の除外+隣接カメラ制約+姿勢クラス制約、エピポーラなし) ===")
    print(f"  recall(統合率)   = {result['recall']:.3f} ({result['tp']}/{result['tp'] + result['fn']})")
    print(f"  precision        = {result['precision']:.3f} (FP={result['fp']})")
    print(f"  F1               = {result['f1']:.3f}")
    print(f"  (参考: 制約無し版のベスト F1=0.626, recall=50.0%, lying=0.6/standing=1.0/dist_thr=1.4)")

    # --- クラスタごとに融合した点を作る(可視化・CSV用) ---
    roots = result["roots"]
    clusters: Dict[Tuple[str, int], List[Tuple[str, int]]] = defaultdict(list)
    for k in all_keys:
        clusters[roots[k]].append(k)

    fused_points = []
    for root, members in clusters.items():
        xy = np.array([per_camera[cam_id][idx]["world_xy"] for cam_id, idx in members])
        names = [per_camera[cam_id][idx]["name"] for cam_id, idx in members]
        majority_name = max(set(names), key=names.count)
        mean_xy = xy.mean(axis=0)
        zone = classify_exclusive_zone(float(mean_xy[0]), CAM_IDS, zones)
        fused_points.append({
            "cluster_id": root, "world_xy": mean_xy,
            "n_members": len(members), "majority_name": majority_name,
            "member_names": names, "zone": zone,
        })

    print(f"  統合後の個体数: {len(fused_points)}(元の検出総数: {len(all_keys)})")

    # --- BEV描画 ---
    scale = 40
    margin = 110  # 右側にカメラ専属範囲の凡例を表示するための余白
    w_px = int(real_w * scale) + margin
    h_px = int(real_h * scale) + 40
    canvas = np.full((h_px, w_px, 3), (30, 30, 30), dtype=np.uint8)
    draw_regions(canvas, regions, scale)

    # 重なり範囲を帯として描画
    for (cam_i, cam_j), (lo, hi) in zones.items():
        x0, x1 = int(lo * scale), int(hi * scale)
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x0, 0), (x1, h_px), (0, 165, 255), -1)
        cv2.addWeighted(overlay, 0.15, canvas, 0.85, 0, canvas)
        cv2.putText(canvas, f"{cam_i}-{cam_j}", (x0 + 4, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 165, 255), 1, cv2.LINE_AA)

    for cid in CAM_IDS:
        p = poses[cid]
        px, py = int(p.t[0] * scale), int(p.t[1] * scale)
        if 0 <= px < w_px and 0 <= py < h_px:
            cv2.circle(canvas, (px, py), 9, (0, 0, 255), -1)
            cv2.circle(canvas, (px, py), 11, (255, 255, 255), 2)
            cv2.putText(canvas, cid, (px + 14, py - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)

    for fp in fused_points:
        xy = fp["world_xy"]
        px, py = int(xy[0] * scale), int(xy[1] * scale)
        if 0 <= px < w_px and 0 <= py < h_px:
            # カメラの専属範囲ならそのカメラの色、重なり範囲ならグレーで描画
            color = CAM_ZONE_COLORS_BGR.get(fp["zone"], OVERLAP_ZONE_COLOR_BGR)
            # 複数カメラのメンバーで統合できた点は白枠、単一カメラだけの点は黒枠で区別
            border = (255, 255, 255) if fp["n_members"] > 1 else (0, 0, 0)
            cv2.circle(canvas, (px, py), 6, color, -1)
            cv2.circle(canvas, (px, py), 7, border, 2 if fp["n_members"] > 1 else 1)
            label = fp["majority_name"].replace("cow", "")
            cv2.putText(canvas, label, (px + 6, py + 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, (0, 255, 0), 1, cv2.LINE_AA)

    # 凡例(カメラ専属範囲の色 + 重なり範囲の色)
    legend_items = [(cid, CAM_ZONE_COLORS_BGR[cid]) for cid in CAM_IDS] + [("overlap", OVERLAP_ZONE_COLOR_BGR)]
    for i, (label, color) in enumerate(legend_items):
        y = 20 + i * 22
        cv2.rectangle(canvas, (w_px - margin + 5, y), (w_px - margin + 21, y + 16), color, -1)
        cv2.putText(canvas, label, (w_px - margin + 26, y + 13), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 255), 1, cv2.LINE_AA)

    out_path = OUTPUT_DIR / "bev_zone_pose_fusion.png"
    cv2.imwrite(str(out_path), canvas)
    print(f"\n保存しました: {out_path}")

    csv_path = OUTPUT_DIR / "bev_zone_pose_fusion_result.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster_id", "world_x_m", "world_y_m", "zone", "n_members", "majority_name", "member_names"])
        for fp in sorted(fused_points, key=lambda r: str(r["cluster_id"])):
            writer.writerow([
                str(fp["cluster_id"]), round(float(fp["world_xy"][0]), 3), round(float(fp["world_xy"][1]), 3),
                fp["zone"], fp["n_members"], fp["majority_name"], "|".join(fp["member_names"]),
            ])
    print(f"保存しました: {csv_path}")

    zone_counts = defaultdict(int)
    for fp in fused_points:
        zone_counts[fp["zone"]] += 1
    print("\nゾーン別の個体数内訳:")
    for zone, count in sorted(zone_counts.items()):
        print(f"  {zone}: {count}頭")


if __name__ == "__main__":
    main()