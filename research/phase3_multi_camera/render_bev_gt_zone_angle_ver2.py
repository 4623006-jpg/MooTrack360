"""
render_bev_gt_zone_angle_ver7.py

YOLOなし(GTのbboxを検出として入力)で、1フレーム分のBEVを1枚だけ描く。

統合の流れ(2段階):
  0. 各カメラの曲線境界の外/牛舎の矩形範囲外の検出を除外(USE_ZONE_FILTER / USE_FLOORPLAN_FILTER)
  1. 【1段目】隣接カメラ間で、高さ仮定の投影位置の距離が DIST_THR_FUSION_M 以内 かつ 姿勢一致
     のペアをHungarianで先に統合する。
  2. 【2段目】1段目で統合されなかった検出だけを対象に、epipolar判定で統合する:
       姿勢一致 + 光線の交差角 >= MIN_RAY_ANGLE_DEG + 光線の最近接距離 <= RAY_DIST_THR_M
       (最近接点が前方で、高さが範囲内) + 次点との差 >= MARGIN_GAP_M
  3. Union-Findでクラスタ化
  4. 統合後の位置: 2カメラ以上のクラスタは、クラスタ内の交差角がすべて MIN_RAY_ANGLE_DEG 以上なら
     三角測量(光線への垂直距離の二乗和の最小解)、そうでなければ高さ仮定の投影位置の平均。
     1カメラのクラスタは高さ仮定。

BEVの凡例:
  カメラ色の小さい丸 : 各カメラのbboxの高さ仮定投影位置(統合後の位置と細線で結ぶ)
  灰の×             : 曲線境界/牛舎範囲の外で除外された検出
  大きい丸(青/灰)+牛番号 : 統合後の位置(青=2カメラ、灰=1カメラ)
  白い菱形           : 三角測量で位置を決めたクラスタの、高さ仮定投影の平均位置(丸との距離=三角測量による移動量)
  マゼンタ輪 MERGE   : 異なる牛を誤って統合
  赤輪 SPLIT         : 同じ牛なのに別クラスタに分裂
  色つきの細い曲線    : 各カメラの曲線境界

設置場所: curved_zone_utils_ver1.py と同じフォルダ。
使い方:
    python render_bev_gt_zone_angle_ver7.py
"""

import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

MOOTRACK_CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calibration")
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for d in (MOOTRACK_CODE_DIR, THIS_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

ANALYZE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analyze")
if ANALYZE_DIR not in sys.path:
    sys.path.insert(0, ANALYZE_DIR)

from Mootrack_core import (
    NUM_CAMERAS, CAM_IDS, ROT_FMT,
    get_intrinsics,
    load_floorplan, apply_camera_overrides, resolve_poses,
    unproject_ds, unproject_to_floor,
    union_find_clusters,
    filter_outside_floorplan_bounds,
)
from curved_zone_utils_ver1 import (
    ZONE_HEIGHT_M, N_ANGLES,
    build_camera_polygons, filter_by_curved_zone, get_adjacent_pairs,
)

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
XML_DIR = "/mnt/vmlqnap02/dataset/MooTrack360/1-hour-farm1-area1-sample-sequence/xml-1fps"
# 注意: このFLOORPLAN_JSON_PATHは以前から要検証(値が古い可能性あり)と指摘済みのファイル
FLOORPLAN_JSON_PATH = "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/calibration/Farm2_Area_1_Artboard_config_.json"
OUTPUT_DIR = Path(
    "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/research/idf1_check"
)
FRAME_IDX = 0
IMAGE_WIDTH_PX = 8832

EXCLUDE_GT_NAMES = {"cow022", "cow109", "cow115", "cow098"}   # アノテーションミスと判断した個体

LYING_HEIGHT_M = 0.5
STANDING_HEIGHT_M = 0.75
DEFAULT_HEIGHT_M = 0.8

MIN_RAY_ANGLE_DEG = 60.0     # 交差角の下限(度)。2段目の統合条件と、三角測量を使う条件。None = 条件なし
USE_ZONE_FILTER = True      # True: 各カメラの曲線境界の外の検出を除外(元のepipolar版は無し)
USE_FLOORPLAN_FILTER = True  # True: 牛舎の矩形範囲外の検出を除外(元のepipolar版は無し)
MARGIN_GAP_M = 0.001         # 最良候補と次点候補のコスト差がこれ未満なら統合を見送る(元のepipolar版と同じ)
USE_POSE_CONSTRAINT = True
DIST_THR_FUSION_M = 0.7      # 1段目: 高さ仮定の投影位置の距離の上限(m)
RAY_DIST_THR_M = 1.0         # 2本の光線の最近接距離の上限(統合判定はこれだけ)
USE_TRIANGULATION = True     # 2カメラ以上のクラスタの位置を三角測量で決める
HEIGHT_RANGE_Z = (-1.6, 0.1)  # 三角測量点の高さ(ワールドz。床が概ね0、高いほど負)の許容範囲

SCALE = 40
TYPE_COLORS_BGR = {
    "wall": (0, 200, 0), "cow_beds": (0, 140, 255), "feeding_bay": (255, 140, 0),
    "water": (255, 255, 0), "gate": (0, 0, 255), "milking_station": (255, 0, 255),
    "hay_area": (0, 255, 255), "cow_brushes": (255, 0, 180),
}
CAM_COLORS_BGR = {"Camera_1": (255, 120, 60), "Camera_2": (60, 200, 60), "Camera_3": (60, 140, 255)}
NAME_COLOR = (0, 255, 255)


def height_for(pose):
    return {"lying": LYING_HEIGHT_M, "standing": STANDING_HEIGHT_M}.get(pose, DEFAULT_HEIGHT_M)


def parse_voc_annotation(xml_path):
    objects = []
    for obj in ET.parse(xml_path).getroot().findall("object"):
        pose_el = obj.find("pose")
        bnd = obj.find("bndbox")
        objects.append({
            "name": obj.find("name").text,
            "pose": pose_el.text.strip().lower() if pose_el is not None and pose_el.text else None,
            "xmin": float(bnd.find("xmin").text), "ymin": float(bnd.find("ymin").text),
            "xmax": float(bnd.find("xmax").text), "ymax": float(bnd.find("ymax").text),
        })
    return objects


def gt_items_per_camera(objects, poses):
    """GTのbboxを、カメラごとの検出リスト(world_xy, 光線方向, 姿勢, 牛名)に変換する。"""
    tile_w = IMAGE_WIDTH_PX / NUM_CAMERAS
    per_camera = {cid: [] for cid in CAM_IDS}
    uid = 0
    for o in objects:
        cx_full = (o["xmin"] + o["xmax"]) / 2.0
        cam_idx = max(0, min(int(cx_full // tile_w), NUM_CAMERAS - 1))
        cam_id = CAM_IDS[cam_idx]
        px = np.array([[cx_full - cam_idx * tile_w, (o["ymin"] + o["ymax"]) / 2.0]], dtype=np.float64)
        fx, fy, cx, cy, xi, alpha = get_intrinsics(cam_id)
        pose = poses[cam_id]
        xy, v1 = unproject_to_floor(px, pose.R, pose.t, fx, fy, cx, cy, xi, alpha,
                                    floor_z=-height_for(o["pose"]))
        dirs_cam, v2 = unproject_ds(px, fx, fy, cx, cy, xi, alpha)
        if v1[0] and v2[0]:
            d = (pose.R @ dirs_cam.T).T[0]
            uid += 1
            per_camera[cam_id].append({
                "uid": uid, "name": o["name"], "pose": o["pose"], "world_xy": xy[0],
                "center": pose.t.copy(),
                "dir_world": d / np.linalg.norm(d),
            })
    return per_camera


def triangulate(items):
    """光線(カメラ中心c, 方向d)への垂直距離の二乗和を最小にする3D点。
    (sum(I - d d^T)) X = sum((I - d d^T) c) を解く。条件を満たさなければNone。"""
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for it in items:
        d, c = it["dir_world"], it["center"]
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ c
    try:
        X = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    if any(np.dot(X - it["center"], it["dir_world"]) <= 0 for it in items):
        return None
    if not (HEIGHT_RANGE_Z[0] <= X[2] <= HEIGHT_RANGE_Z[1]):
        return None
    return X


def ray_closest_distance(a, b):
    """2本の半直線(カメラ中心+方向)の (最近接距離, 最近接点の中点)。どちらかが
    カメラの後ろ側で最近接する/ほぼ平行な場合は (inf, None)。"""
    d1, d2, p1, p2 = a["dir_world"], b["dir_world"], a["center"], b["center"]
    w0 = p1 - p2
    bb, d, e = float(d1 @ d2), float(d1 @ w0), float(d2 @ w0)
    denom = 1.0 - bb * bb
    if abs(denom) < 1e-9:
        return float("inf"), None
    t1 = (bb * e - d) / denom
    t2 = (e - bb * d) / denom
    if t1 <= 0 or t2 <= 0:
        return float("inf"), None
    q1, q2 = p1 + t1 * d1, p2 + t2 * d2
    return float(np.linalg.norm(q1 - q2)), (q1 + q2) / 2.0


def ray_angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(a["dir_world"], b["dir_world"]), -1.0, 1.0))))


def match_adjacent(per_camera, adjacent_pairs, exclude=frozenset()):
    """隣接カメラ間、光線の最近接距離 + 姿勢一致 + 光線の交差角 でHungarian割当。"""
    pairs = []
    for ci, cj in adjacent_pairs:
        items_i, items_j = per_camera[ci], per_camera[cj]
        if not items_i or not items_j:
            continue
        cost = np.full((len(items_i), len(items_j)), 1e6)
        for a, A in enumerate(items_i):
            for b, B in enumerate(items_j):
                if (ci, a) in exclude or (cj, b) in exclude:
                    continue
                if USE_POSE_CONSTRAINT and A["pose"] and B["pose"] and A["pose"] != B["pose"]:
                    continue
                if MIN_RAY_ANGLE_DEG is not None and ray_angle_deg(A, B) < MIN_RAY_ANGLE_DEG:
                    continue
                dist, mid = ray_closest_distance(A, B)
                if mid is None or not (HEIGHT_RANGE_Z[0] <= mid[2] <= HEIGHT_RANGE_Z[1]):
                    continue   # カメラの後ろ側 / 牛としてありえない高さ
                if dist <= RAY_DIST_THR_M:
                    cost[a, b] = dist
        for a, b in zip(*linear_sum_assignment(cost)):
            if cost[a, b] > RAY_DIST_THR_M:
                continue
            row_alt = np.delete(cost[a, :], b)
            col_alt = np.delete(cost[:, b], a)
            alt_best = min(row_alt.min() if row_alt.size else np.inf,
                           col_alt.min() if col_alt.size else np.inf)
            if alt_best - cost[a, b] < MARGIN_GAP_M:
                continue   # 僅差で区別がつかない -> 統合を見送る
            pairs.append(((ci, a), (cj, b)))
    return pairs


def match_adjacent_floor(per_camera, adjacent_pairs):
    """【1段目】隣接カメラ間、高さ仮定の投影位置の距離 <= DIST_THR_FUSION_M かつ 姿勢一致。"""
    pairs = []
    for ci, cj in adjacent_pairs:
        items_i, items_j = per_camera[ci], per_camera[cj]
        if not items_i or not items_j:
            continue
        cost = np.full((len(items_i), len(items_j)), 1e6)
        for a, A in enumerate(items_i):
            for b, B in enumerate(items_j):
                if USE_POSE_CONSTRAINT and A["pose"] and B["pose"] and A["pose"] != B["pose"]:
                    continue
                d = float(np.linalg.norm(A["world_xy"] - B["world_xy"]))
                if d <= DIST_THR_FUSION_M:
                    cost[a, b] = d
        for a, b in zip(*linear_sum_assignment(cost)):
            if cost[a, b] <= DIST_THR_FUSION_M:
                pairs.append(((ci, a), (cj, b)))
    return pairs


def min_pairwise_angle(items):
    ang = [ray_angle_deg(items[i], items[j]) for i in range(len(items)) for j in range(i + 1, len(items))]
    return min(ang) if ang else 180.0


def evaluate(per_camera, fused_members):
    name_of = {(c, i): it["name"] for c, items in per_camera.items() for i, it in enumerate(items)}
    cluster_of = {m: k for k, ms in enumerate(fused_members) for m in ms}
    keys = list(name_of)
    tp = fn = fp = 0
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            if keys[a][0] == keys[b][0]:
                continue
            same_name = name_of[keys[a]] == name_of[keys[b]]
            same_cl = cluster_of[keys[a]] == cluster_of[keys[b]]
            tp += same_name and same_cl
            fn += same_name and not same_cl
            fp += (not same_name) and same_cl
    wrong = [k for k, ms in enumerate(fused_members) if len({name_of[m] for m in ms}) > 1]
    clusters_of_name = defaultdict(set)
    for m, k in cluster_of.items():
        clusters_of_name[name_of[m]].add(k)
    split = [n for n, s in clusters_of_name.items() if len(s) > 1]
    return tp, fn, fp, wrong, split, name_of


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    regions, cameras_json, real_w, real_h = load_floorplan(FLOORPLAN_JSON_PATH)
    poses = resolve_poses(apply_camera_overrides(cameras_json), root="Area1", fmt=ROT_FMT)
    cam_polygons = build_camera_polygons(poses, CAM_IDS, get_intrinsics, height_m=ZONE_HEIGHT_M, n_angles=N_ANGLES)
    adjacent_pairs = get_adjacent_pairs(CAM_IDS)

    name = f"{FRAME_IDX:06d}"
    objs = [o for o in parse_voc_annotation(Path(XML_DIR) / f"{name}.xml") if o["name"] not in EXCLUDE_GT_NAMES]
    raw = gt_items_per_camera(objs, poses)
    kept = filter_by_curved_zone(raw, cam_polygons) if USE_ZONE_FILTER else raw
    if USE_FLOORPLAN_FILTER:
        kept = filter_outside_floorplan_bounds(kept, real_w, real_h, margin_m=0.0, verbose=False)

    pairs1 = match_adjacent_floor(kept, adjacent_pairs)
    used1 = {k for pr in pairs1 for k in pr}
    pairs2 = match_adjacent(kept, adjacent_pairs, exclude=used1)
    pairs = pairs1 + pairs2
    all_keys = [(c, i) for c, items in kept.items() for i in range(len(items))]
    roots = union_find_clusters(all_keys, pairs)
    groups = defaultdict(list)
    for k in all_keys:
        groups[roots[k]].append(k)
    members = list(groups.values())
    fb_mean = [np.mean([kept[c][i]["world_xy"] for c, i in ms], axis=0) for ms in members]
    fused, tri_used, shifts = [], set(), []
    for k, ms in enumerate(members):
        its = [kept[c][i] for c, i in ms]
        ok_angle = MIN_RAY_ANGLE_DEG is None or (len(ms) >= 2 and min_pairwise_angle(its) >= MIN_RAY_ANGLE_DEG)
        X = triangulate(its) if (USE_TRIANGULATION and len(ms) >= 2 and ok_angle) else None
        if X is not None:
            fused.append(X[:2])
            tri_used.add(k)
            shifts.append(float(np.linalg.norm(X[:2] - fb_mean[k])))
        else:
            fused.append(fb_mean[k])
    tp, fn, fp, wrong, split, name_of = evaluate(kept, members)

    # --- 描画 ---
    margin = 110
    w_px = int(real_w * SCALE) + margin
    h_px = int(real_h * SCALE) + 40
    w_px += w_px % 2
    h_px += h_px % 2
    canvas = np.full((h_px, w_px, 3), (30, 30, 30), dtype=np.uint8)
    for region in regions:
        pts = (region.polygon_m * SCALE).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], isClosed=region.type not in ("wall", "gate"),
                      color=TYPE_COLORS_BGR.get(region.type, (200, 200, 200)), thickness=2, lineType=cv2.LINE_AA)
    for cid, poly in cam_polygons.items():
        if poly is not None and len(poly) >= 3:
            cv2.polylines(canvas, [(poly * SCALE).astype(np.int32).reshape(-1, 1, 2)], True,
                          CAM_COLORS_BGR.get(cid, (255, 255, 255)), 1, cv2.LINE_AA)
    for cid in CAM_IDS:
        px, py = int(poses[cid].t[0] * SCALE), int(poses[cid].t[1] * SCALE)
        cv2.circle(canvas, (px, py), 9, (0, 0, 255), -1)
        cv2.circle(canvas, (px, py), 11, (255, 255, 255), 2)
        cv2.putText(canvas, cid, (px + 14, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    kept_ids = {it["uid"] for items in kept.values() for it in items}
    for c, items in raw.items():                      # 除外された検出
        for it in items:
            if it["uid"] not in kept_ids:
                cv2.drawMarker(canvas, (int(it["world_xy"][0] * SCALE), int(it["world_xy"][1] * SCALE)),
                               (120, 120, 120), markerType=cv2.MARKER_TILTED_CROSS, markerSize=8, thickness=1)

    for k, ms in enumerate(members):
        fx_, fy_ = int(fused[k][0] * SCALE), int(fused[k][1] * SCALE)
        for c, i in ms:
            q = kept[c][i]["world_xy"]
            col = CAM_COLORS_BGR.get(c, (255, 255, 255))
            cv2.line(canvas, (int(q[0] * SCALE), int(q[1] * SCALE)), (fx_, fy_), col, 1, cv2.LINE_AA)
            cv2.circle(canvas, (int(q[0] * SCALE), int(q[1] * SCALE)), 4, col, -1)
    for k, ms in enumerate(members):
        px, py = int(fused[k][0] * SCALE), int(fused[k][1] * SCALE)
        multi = len(ms) >= 2
        if k in tri_used:
            cv2.drawMarker(canvas, (int(fb_mean[k][0] * SCALE), int(fb_mean[k][1] * SCALE)), (255, 255, 255),
                           markerType=cv2.MARKER_DIAMOND, markerSize=10, thickness=1)
        cv2.circle(canvas, (px, py), 7, (255, 160, 60) if multi else (170, 170, 170), -1)
        cv2.circle(canvas, (px, py), 9, (255, 255, 255) if multi else (0, 0, 0), 2 if multi else 1)
        label = "/".join(sorted({name_of[m].replace("cow", "") for m in ms}))
        cv2.putText(canvas, label, (px + 10, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, NAME_COLOR, 1, cv2.LINE_AA)
        if k in wrong:
            cv2.circle(canvas, (px, py), 15, (255, 0, 255), 2)
            cv2.putText(canvas, "MERGE", (px + 8, py + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1, cv2.LINE_AA)
    for n in split:
        pts = [fused[k] for k, ms in enumerate(members) if any(name_of[m] == n for m in ms)]
        p = np.mean(pts, axis=0)
        cv2.circle(canvas, (int(p[0] * SCALE), int(p[1] * SCALE)), 18, (0, 0, 255), 2)
        cv2.putText(canvas, "SPLIT", (int(p[0] * SCALE) + 8, int(p[1] * SCALE) + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

    n_raw = sum(len(v) for v in raw.values())
    n_kept = sum(len(v) for v in kept.values())
    header = (f"{name} GTのみ入力  床距離<={DIST_THR_FUSION_M}m 交差角{('>=%.0fdeg' % MIN_RAY_ANGLE_DEG) if MIN_RAY_ANGLE_DEG is not None else 'なし'} 光線距離<={RAY_DIST_THR_M}m  1段目{len(pairs1)}組+2段目{len(pairs2)}組  "
              f"bbox {n_raw}->範囲内{n_kept}  クラスタ={len(members)}(三角測量{len(tri_used)}, 平均移動{(np.mean(shifts) if shifts else 0):.2f}m)  ペア TP={tp} FN(統合漏れ)={fn} FP(誤統合)={fp}")
    cv2.putText(canvas, header, (8, h_px - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    for k, cid in enumerate(CAM_IDS):
        y = 20 + k * 22
        cv2.rectangle(canvas, (w_px - margin + 5, y), (w_px - margin + 21, y + 16), CAM_COLORS_BGR[cid], -1)
        cv2.putText(canvas, cid, (w_px - margin + 26, y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    out = OUTPUT_DIR / f"bev_gt_zone_angle_ver7_{name}.png"
    cv2.imwrite(str(out), canvas)
    print(f"bbox {n_raw} -> 範囲内 {n_kept} / クラスタ {len(members)} / TP={tp} FN={fn} FP={fp}")
    print(f"保存: {out}")


if __name__ == "__main__":
    main()