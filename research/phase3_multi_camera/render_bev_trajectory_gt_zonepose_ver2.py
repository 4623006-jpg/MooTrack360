"""
render_bev_trajectory_gt_zonepose_ver4.py

render_bev_trajectory_yolo_zonepose_ver3.py(YOLO入力の軌跡動画)との差分は1点のみ:
検出をYOLOではなくGT(XML)のbboxにした。YOLOの見逃し・誤検出を除いた状態で、
「曲線境界での除外 + 隣接カメラ+姿勢制約+距離閾値の統合 + トラッカー」だけの挙動を
動画で確認する。

検出・融合・トラッカー・設定値は phase3_bev_multicam_gt_zonepose_ver1.py(g)のものを使い、
描画(壁・エリア、曲線境界、track_idごとの色、直近TRAIL_LENGTHフレームの尾、GTの十字)と
動画化は render_bev_frames_yolo_zonepose_ver5.py の関数を再利用する。
GTの十字は、牛名ごとの平均位置(評価で使うGT位置)。

ver4: SPLITが出たフレームで、その牛の各bboxの情報(カメラ・姿勢・投影位置・範囲内か)と、
      bboxペアごとに統合されなかった理由(同一カメラ/隣接でない/姿勢不一致/距離超過/割当競合)を
      ターミナルに出力する。
ver3: 現在位置の丸のラベルを、トラックID(ID12)から牛番号(001、複数が統合されていれば001/045)にした。
      丸の色は従来通りトラックIDごと(同じ丸の色=同じトラック)。

ver1との差分: 牛名で判定した統合の誤りをBEV上に表示する。
  マゼンタ輪 MERGE+牛番号 : 異なる牛を誤って同じクラスタに統合した
  赤輪 SPLIT+牛番号       : 同じ牛なのに複数のクラスタに分かれている(未統合)
ヘッダの TP/FN/FP は、牛名で判定した統合ペア数(正しい統合/統合漏れ/誤統合)、
M/S はそのフレームのMERGE/SPLITの数。

出力先: OUTPUT_DIR/bev_trajectory_frames_gt_zonepose_explain_{reid|noreid}/{frame}.png
       + OUTPUT_DIR/bev_trajectory_gt_zonepose_explain_{reid|noreid}.mp4

このファイルは phase3_bev_multicam_gt_zonepose_ver1.py / render_bev_frames_yolo_zonepose_ver5.py /
curved_zone_utils_ver1.py と同じフォルダに置くこと。

--- 使い方 ---
    python render_bev_trajectory_gt_zonepose_ver4.py
"""

import os
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for d in (os.path.join(THIS_DIR, "..", "calibration"), os.path.join(THIS_DIR, "..", "analyze"), THIS_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

import phase3_bev_multicam_gt_zonepose_ver1 as g
from Mootrack_core import load_floorplan, union_find_clusters
from render_bev_frames_yolo_zonepose_ver5 import (
    CAM_ZONE_COLORS_BGR, GT_MARKER_COLOR_BGR, draw_regions, make_video_from_frames,
)

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
FRAME_START = 0
FRAME_END = 1400
FRAME_STEP = g.FRAME_STEP

TRAIL_LENGTH = 30       # 軌跡として残す直近フレーム数
MAKE_VIDEO = True
VIDEO_FPS = 2
BG_COLOR_BGR = (30, 30, 30)

_TRACKER_SUFFIX = "reid" if g.USE_REID_TRACKER else "noreid"
OUTPUT_DIR = g.OUTPUT_DIR
FRAMES_OUT_DIR = OUTPUT_DIR / f"bev_trajectory_frames_gt_zonepose_explain_{_TRACKER_SUFFIX}"
FRAMES_OUT_DIR.mkdir(parents=True, exist_ok=True)


def track_color(track_id: int) -> Tuple[int, int, int]:
    hue = (track_id * 47) % 180
    bgr = cv2.cvtColor(np.uint8([[[hue, 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_trail(canvas, points, scale, base_color, bg_color=BG_COLOR_BGR):
    n = len(points)
    if n < 2:
        return
    h_px, w_px = canvas.shape[:2]
    for i in range(n - 1):
        p0 = (int(points[i][0] * scale), int(points[i][1] * scale))
        p1 = (int(points[i + 1][0] * scale), int(points[i + 1][1] * scale))
        if not (0 <= p0[0] < w_px and 0 <= p0[1] < h_px and 0 <= p1[0] < w_px and 0 <= p1[1] < h_px):
            continue
        alpha = (i + 1) / n
        color = tuple(int(bg_color[c] * (1 - alpha) + base_color[c] * alpha) for c in range(3))
        cv2.line(canvas, p0, p1, color, max(1, int(round(1 + alpha * 2))), cv2.LINE_AA)


def explain_split(frame_idx, name, per_camera_raw, per_camera_kept, adjacent_pairs):
    """SPLITした牛 name について、各bboxの情報とペアごとの未統合の理由を出力する。"""
    def key(cam, it):
        return (cam, it["name"], round(float(it["world_xy"][0]), 4), round(float(it["world_xy"][1]), 4))

    kept_keys = {key(c, it) for c, items in per_camera_kept.items() for it in items}
    boxes = []
    for cam, items in per_camera_raw.items():
        for it in items:
            if it["name"] == name:
                boxes.append((cam, it, key(cam, it) in kept_keys))
    print(f"  [SPLIT解析] frame={frame_idx:06d} {name}: bbox数={len(boxes)}")
    for cam, it, kept in boxes:
        print(f"      {cam} pose={it['pose']} 投影位置=({it['world_xy'][0]:.2f},{it['world_xy'][1]:.2f}) "
              f"範囲内={'はい' if kept else 'いいえ(除外された)'}")
    adj = {frozenset(p) for p in adjacent_pairs}
    for a in range(len(boxes)):
        for b in range(a + 1, len(boxes)):
            (ca, ia, ka), (cb, ib, kb) = boxes[a], boxes[b]
            if not (ka and kb):
                continue    # 範囲外で除外された方はSPLITの数に入らない
            d = float(np.linalg.norm(ia["world_xy"] - ib["world_xy"]))
            if ca == cb:
                why = "同一カメラ内に同名bboxが複数(アノテーション重複の疑い)。同一カメラ内は統合の対象外"
            elif frozenset((ca, cb)) not in adj:
                why = "隣接カメラではない(統合の対象外)"
            elif g.USE_POSE_CONSTRAINT and ia["pose"] and ib["pose"] and ia["pose"] != ib["pose"]:
                why = f"姿勢が不一致 ({ia['pose']} vs {ib['pose']}) で姿勢制約により除外"
            elif d > g.DIST_THR_FUSION_M:
                why = f"距離 {d:.2f}m > 閾値 {g.DIST_THR_FUSION_M}m"
            else:
                why = "条件はすべて満たす -> 1対1割当の競合(近くの別個体と取り合い)の可能性"
            print(f"      {ca} と {cb}: 距離={d:.2f}m -> {why}")


def fuse_with_errors(per_camera, adjacent_pairs):
    """g.fuse_frame_zone_pose と同じ統合を行い、牛名による誤り(MERGE/SPLIT)の情報も返す。
    戻り値: fused, pair_stats(tp,fn,fp), merge_info[(xy, 牛名リスト)], split_info[(xy, 牛名)]"""
    pairs, _ = g.match_zone_pose_constrained(per_camera, adjacent_pairs, g.DIST_THR_FUSION_M, g.USE_POSE_CONSTRAINT)
    all_keys = [(c, i) for c, items in per_camera.items() for i in range(len(items))]
    if not all_keys:
        return [], (0, 0, 0), [], []
    roots = union_find_clusters(all_keys, pairs)
    groups = defaultdict(list)
    for k in all_keys:
        groups[roots[k]].append(k)
    members = list(groups.values())
    fused = [{"world_xy": np.mean([per_camera[c][i]["world_xy"] for c, i in ms], axis=0), "n_cams": len(ms)}
             for ms in members]
    name_of = {k: per_camera[k[0]][k[1]]["name"] for k in all_keys}
    cluster_of = {m: ci for ci, ms in enumerate(members) for m in ms}

    tp = fn = fp = 0
    for a in range(len(all_keys)):
        for b in range(a + 1, len(all_keys)):
            ka, kb = all_keys[a], all_keys[b]
            if ka[0] == kb[0]:
                continue
            same_name = name_of[ka] == name_of[kb]
            same_cl = cluster_of[ka] == cluster_of[kb]
            tp += same_name and same_cl
            fn += same_name and not same_cl
            fp += (not same_name) and same_cl

    merge_info = []
    clusters_of_name = defaultdict(set)
    for ci, ms in enumerate(members):
        names = sorted({name_of[m] for m in ms})
        if len(names) > 1:
            merge_info.append((fused[ci]["world_xy"], names))
        for n in names:
            clusters_of_name[n].add(ci)
    for ci, ms in enumerate(members):
        fused[ci]["names"] = sorted({name_of[m] for m in ms})
    split_info = [(np.mean([fused[ci]["world_xy"] for ci in cis], axis=0), n)
                  for n, cis in clusters_of_name.items() if len(cis) > 1]
    return fused, (tp, fn, fp), merge_info, split_info


def draw_frame_trajectory(frame_idx, regions, poses, cam_polygons, real_w, real_h,
                          gt_points: Dict[str, np.ndarray], track_outputs, track_histories, pair_stats,
                          merge_info, split_info, label_of):
    scale = 40
    margin = 110
    w_px = int(real_w * scale) + margin
    h_px = int(real_h * scale) + 40
    w_px += w_px % 2     # mp4は幅/高さが偶数でないと崩れることがある
    h_px += h_px % 2
    canvas = np.full((h_px, w_px, 3), BG_COLOR_BGR, dtype=np.uint8)
    draw_regions(canvas, regions, scale)

    for cid in g.CAM_IDS:
        poly = cam_polygons.get(cid)
        if poly is not None and len(poly) >= 3:
            cv2.polylines(canvas, [(poly * scale).astype(np.int32).reshape(-1, 1, 2)], True,
                          CAM_ZONE_COLORS_BGR.get(cid, (255, 255, 255)), 1, cv2.LINE_AA)
    for cid in g.CAM_IDS:
        px, py = int(poses[cid].t[0] * scale), int(poses[cid].t[1] * scale)
        cv2.circle(canvas, (px, py), 9, (0, 0, 255), -1)
        cv2.circle(canvas, (px, py), 11, (255, 255, 255), 2)
        cv2.putText(canvas, cid, (px + 14, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    for xy in gt_points.values():            # GT(牛名ごとの平均位置)
        px, py = int(xy[0] * scale), int(xy[1] * scale)
        if 0 <= px < w_px and 0 <= py < h_px:
            cv2.drawMarker(canvas, (px, py), GT_MARKER_COLOR_BGR, markerType=cv2.MARKER_CROSS,
                           markerSize=8, thickness=1)

    for tid, _xy in track_outputs:           # 軌跡の尾
        hist = track_histories.get(tid)
        if hist:
            draw_trail(canvas, list(hist), scale, track_color(tid))

    for tid, xy in track_outputs:            # 現在位置(track_idで色分け)
        px, py = int(xy[0] * scale), int(xy[1] * scale)
        if 0 <= px < w_px and 0 <= py < h_px:
            color = track_color(tid)
            cv2.circle(canvas, (px, py), 7, color, -1)
            cv2.circle(canvas, (px, py), 9, (255, 255, 255), 1)
            cv2.putText(canvas, label_of.get(tid, f"ID{tid}"), (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

    for xy, names in merge_info:             # 誤統合: マゼンタ輪 + 牛番号
        px, py = int(xy[0] * scale), int(xy[1] * scale)
        cv2.circle(canvas, (px, py), 15, (255, 0, 255), 2)
        cv2.putText(canvas, "MERGE " + "/".join(n.replace("cow", "") for n in names), (px + 10, py + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 0, 255), 1, cv2.LINE_AA)
    for xy, name in split_info:              # 未統合: 赤輪 + 牛番号
        px, py = int(xy[0] * scale), int(xy[1] * scale)
        cv2.circle(canvas, (px, py), 19, (0, 0, 255), 2)
        cv2.putText(canvas, "SPLIT " + name.replace("cow", ""), (px + 10, py - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1, cv2.LINE_AA)

    tp, fn, fp = pair_stats
    header = (f"{g.frame_name(frame_idx)}  GT入力  GT牛={len(gt_points)} track={len(track_outputs)}  "
              f"統合ペア TP={tp} FN(統合漏れ)={fn} FP(誤統合)={fp}  MERGE={len(merge_info)} SPLIT={len(split_info)}")
    cv2.putText(canvas, header, (8, h_px - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def main():
    regions, cameras_json, real_w, real_h = load_floorplan(g.FLOORPLAN_JSON_PATH)
    poses = g.resolve_poses(g.apply_camera_overrides(cameras_json), root="Area1", fmt=g.ROT_FMT)
    cam_polygons = g.build_camera_polygons(poses, g.CAM_IDS, g.get_intrinsics,
                                           height_m=g.ZONE_HEIGHT_M, n_angles=g.N_ANGLES)
    adjacent_pairs = g.get_adjacent_pairs(g.CAM_IDS)

    if g.USE_REID_TRACKER:
        tracker = g.ReIDBevTracker(
            dist_thr=g.DIST_THR_TRACK_M, max_missed=g.MAX_MISSED_FRAMES,
            real_w=real_w, real_h=real_h, edge_margin_m=g.EDGE_MARGIN_M,
            reid_dist_thr=g.REID_DIST_THR_M, lost_max_age=g.LOST_MAX_AGE)
    else:
        tracker = g.SimpleBevTracker(g.DIST_THR_TRACK_M, g.MAX_MISSED_FRAMES)
    track_histories: Dict[int, deque] = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))

    saved_paths: List[Path] = []
    frame_indices = list(range(FRAME_START, FRAME_END + 1, FRAME_STEP))
    for frame_idx in frame_indices:
        xml_path = Path(g.XML_DIR) / f"{g.frame_name(frame_idx)}.xml"
        if not xml_path.exists():
            continue
        gt_objects = g.parse_voc_annotation(xml_path)
        gt_world = g.gt_world_points_this_frame(gt_objects, poses, g.IMAGE_WIDTH_PX)
        gt_points = {n: np.mean(np.array(v), axis=0) for n, v in gt_world.items()}

        per_camera = g.gt_detections_per_camera(gt_objects, poses)
        per_camera_raw = per_camera       # 範囲の除外の前(SPLIT解析用)
        if g.USE_ZONE_FILTER:
            per_camera = g.filter_by_curved_zone(per_camera, cam_polygons)
        if g.USE_FLOORPLAN_FILTER:
            per_camera = g.filter_outside_floorplan_bounds(per_camera, real_w, real_h, margin_m=0.0, verbose=False)
        fused_points, pair_stats, merge_info, split_info = fuse_with_errors(per_camera, adjacent_pairs)
        for _xy, split_name in split_info:
            explain_split(frame_idx, split_name, per_camera_raw, per_camera, adjacent_pairs)

        track_outputs = tracker.update(fused_points)
        label_of = {}
        for tid, xy in track_outputs:
            track_histories[tid].append((float(xy[0]), float(xy[1])))
            for f in fused_points:
                if np.linalg.norm(np.asarray(xy) - f["world_xy"]) < 1e-6:
                    label_of[tid] = "/".join(n.replace("cow", "") for n in f.get("names", []))
                    break

        canvas = draw_frame_trajectory(frame_idx, regions, poses, cam_polygons, real_w, real_h,
                                       gt_points, track_outputs, track_histories, pair_stats,
                                       merge_info, split_info, label_of)
        out_path = FRAMES_OUT_DIR / f"{g.frame_name(frame_idx)}.png"
        cv2.imwrite(str(out_path), canvas)
        saved_paths.append(out_path)
        if len(saved_paths) % 20 == 0:
            print(f"  {len(saved_paths)}/{len(frame_indices)} 処理済み (直近 {g.frame_name(frame_idx)}, track={len(track_outputs)})")

    print(f"\n{len(saved_paths)}枚のBEV軌跡画像を保存しました: {FRAMES_OUT_DIR}")
    if MAKE_VIDEO and saved_paths:
        video_path = OUTPUT_DIR / f"bev_trajectory_gt_zonepose_explain_{_TRACKER_SUFFIX}.mp4"
        make_video_from_frames(FRAMES_OUT_DIR, video_path, VIDEO_FPS, saved_paths)


if __name__ == "__main__":
    main()