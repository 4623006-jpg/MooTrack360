"""
render_bev_trajectory_yolo_zonepose_ver3.py

render_bev_trajectory_yolo_zonepose.py(直線境界版)との差分は1点のみ:
検出・融合に使う隣接カメラの有効範囲の区切りを、「ワールドx座標の中点 ±
OVERLAP_HALF_WIDTH_M」という直線(x軸に垂直な帯)から、curved_zone_utils_ver1.py
が計算する「許容できる魚眼カメラの範囲(画像上の半径R_max)」に基づく曲線
(カメラごとの多角形)に置き換えた。

検出・融合ロジック自体はrender_bev_frames_yolo_zonepose_ver1.py(曲線境界版)を
再利用し、フレーム間の個体ID継続(ReIDBevTracker/SimpleBevTracker)・軌跡描画
(直近TRAIL_LENGTHフレームの尾をフェードさせる方式)はrender_bev_trajectory_
yolo_zonepose.pyと同一。

出力先: OUTPUT_DIR/bev_trajectory_frames_yolo_zonepose_curved_{reid|noreid}/{frame}.png
       + OUTPUT_DIR/bev_trajectory_yolo_zonepose_curved_{reid|noreid}.mp4 (MAKE_VIDEO=Trueの場合)

注意: FRAME_END=35990(デフォルト)はFRAME_STEP=10フレーム刻みでも約3600フレーム分の
YOLO推論が必要になる。実行時間がかなり長くなることを想定しておくこと。

このファイルは render_bev_frames_yolo_zonepose_ver1.py / curved_zone_utils_ver1.py
と同じフォルダに置くこと(plain importで読み込むため)。

バージョン管理方針: この先さらに変更する場合は、このファイルを直接書き換えず
_ver2.py を新しく作ること。

--- 使い方 ---
    python render_bev_trajectory_yolo_zonepose_ver1.py
"""

import os
import sys
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, List, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

MOOTRACK_CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calibration")
if MOOTRACK_CODE_DIR not in sys.path:
    sys.path.insert(0, MOOTRACK_CODE_DIR)


ANALYZE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analyze")
if ANALYZE_DIR not in sys.path:
    sys.path.insert(0, ANALYZE_DIR)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from Mootrack_core import (
    CAM_IDS, ROT_FMT, get_intrinsics,
    load_floorplan, apply_camera_overrides, resolve_poses,
    filter_outside_floorplan_bounds,
    ReIDBevTracker,
)

from curved_zone_utils_ver1 import (
    ZONE_HEIGHT_M, N_ANGLES,
    build_camera_polygons, filter_by_curved_zone, get_adjacent_pairs,
)

# render_bev_frames_yolo_zonepose_ver1.py側(曲線境界版)の検出・融合ロジック・
# 設定値・配色をそのまま再利用(ロジックの二重管理を避けるため)
from render_bev_frames_yolo_zonepose_ver5 import (
    FLOORPLAN_JSON_PATH, OUTPUT_DIR, XML_DIR, IMAGE_DIR,
    YOLO_WEIGHTS_PATH, IMAGE_WIDTH_PX,
    FRAME_STEP,
    DIST_THR_FUSION_M, USE_POSE_CONSTRAINT, DEDUP_IOU_THRESHOLD,
    CAM_ZONE_COLORS_BGR,
    frame_name, parse_voc_annotation, gt_world_points_per_camera,
    yolo_world_points_per_camera_with_pose, deduplicate_per_camera,
    fuse_frame_zone_pose, draw_regions,
    GT_MARKER_COLOR_BGR,
    make_video_from_frames,
)

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------

FRAME_START = 0
FRAME_END = 4800  # ここまでの全フレームを処理する(FRAME_STEPはrender script側の値を再利用)

# Trueなら見失いリストからのReID(Mootrack_core.ReIDBevTracker)を使う。
# Falseなら以前の単純な実装(SimpleBevTracker、見失ったら完全削除し、
# 再出現時は新しいIDを振る)に戻す。同じ設定値(dist_thr/max_missed)のまま
# ReIDの有無だけを切り替えて動画を見比えたいときに使う。
USE_REID_TRACKER = True

# ReIDあり/なしでPNG・mp4を別フォルダ・別ファイル名に分けておく
# (同じフレーム番号のPNGを上書きし合って比較できなくなるのを防ぐため)
_TRACKER_SUFFIX = "reid" if USE_REID_TRACKER else "noreid"
TRAJ_FRAMES_OUT_DIR = OUTPUT_DIR / f"bev_trajectory_frames_yolo_zonepose_curved_{_TRACKER_SUFFIX}"
TRAJ_FRAMES_OUT_DIR.mkdir(parents=True, exist_ok=True)

# 軌跡として残す直近フレーム数(フェードする尾の長さ)。
# FRAME_STEP=10(1fpsデータで10フレーム刻み)なら、30フレーム分の尾=約5分間の移動を表示。
TRAIL_LENGTH = 30

# フレーム間追跡のパラメータ(ReIDBevTracker/SimpleBevTracker共通)
DIST_THR_TRACK_M = 1.5
MAX_MISSED_FRAMES = 2
# 牛舎の外周(境界)からこの距離以内で現れた未紐付け検出は、外から入ってきた
# 個体として無条件に新規ID発行の対象にする。これより内側で現れた場合は、
# 新規個体とはみなさず、まずlost_tracks(見失いリスト)の中からReIDを試みる。
# (USE_REID_TRACKER=Trueのときのみ使う)
EDGE_MARGIN_M = 0.7
# ReID時の距離閾値(見失っている間の移動を考慮し、通常のDIST_THR_TRACK_Mより緩める)
REID_DIST_THR_M = 3.0
# lost_tracksを保持するフレーム数の上限(Noneなら無期限に保持し続ける)
LOST_MAX_AGE = 10

MAKE_VIDEO = True
VIDEO_FPS = 30

BG_COLOR_BGR = (30, 30, 30)


# ---------------------------------------------------------------------------
# フレーム間追跡(ReIDなし旧版。USE_REID_TRACKER=Falseのときに使う)
# ---------------------------------------------------------------------------

class SimpleBevTracker:
    def __init__(self, dist_thr: float, max_missed: int):
        self.dist_thr = dist_thr
        self.max_missed = max_missed
        self.tracks: Dict[int, dict] = {}
        self.next_id = 1

    def update(self, fused_points: List[dict]) -> List[Tuple[int, np.ndarray]]:
        track_ids = list(self.tracks.keys())
        n_trk, n_det = len(track_ids), len(fused_points)
        matched_trk, matched_det = set(), set()

        if n_trk > 0 and n_det > 0:
            cost = np.zeros((n_trk, n_det))
            for i, tid in enumerate(track_ids):
                for j, fp in enumerate(fused_points):
                    cost[i, j] = float(np.linalg.norm(self.tracks[tid]["xy"] - fp["world_xy"]))
            row_idx, col_idx = linear_sum_assignment(cost)
            for i, j in zip(row_idx, col_idx):
                if cost[i, j] <= self.dist_thr:
                    tid = track_ids[i]
                    self.tracks[tid]["xy"] = fused_points[j]["world_xy"]
                    self.tracks[tid]["missed"] = 0
                    matched_trk.add(tid)
                    matched_det.add(j)

        for tid in track_ids:
            if tid not in matched_trk:
                self.tracks[tid]["missed"] += 1

        self.tracks = {tid: t for tid, t in self.tracks.items() if t["missed"] <= self.max_missed}

        for j, fp in enumerate(fused_points):
            if j not in matched_det:
                self.tracks[self.next_id] = {"xy": fp["world_xy"], "missed": 0}
                self.next_id += 1

        return [(tid, t["xy"]) for tid, t in self.tracks.items() if t["missed"] == 0]


# ---------------------------------------------------------------------------
# 軌跡描画
# ---------------------------------------------------------------------------

def track_color(track_id: int) -> Tuple[int, int, int]:
    """track_idごとに安定した色を割り当てる(色相をID依存でずらすだけなので、
    同じIDなら常に同じ色になる)。"""
    hue = (track_id * 47) % 180  # 47は180に対してずらし幅が偏りにくい値
    hsv = np.uint8([[[hue, 200, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_trail(
    canvas: np.ndarray, points: List[Tuple[float, float]], scale: float,
    base_color: Tuple[int, int, int], bg_color: Tuple[int, int, int] = BG_COLOR_BGR,
) -> None:
    """points(古い->新しいの順の世界座標xy)を結ぶ線を、古いほど背景色に近づけて
    フェードさせながら描く(直近Nフレームだけの尾を表現)。"""
    n = len(points)
    if n < 2:
        return
    h_px, w_px = canvas.shape[0], canvas.shape[1]
    for i in range(n - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        p0 = (int(x0 * scale), int(y0 * scale))
        p1 = (int(x1 * scale), int(y1 * scale))
        if not (0 <= p0[0] < w_px and 0 <= p0[1] < h_px and 0 <= p1[0] < w_px and 0 <= p1[1] < h_px):
            continue
        alpha = (i + 1) / n  # 古い(0に近い)ほど薄く、新しい(1)ほど濃くなる
        color = tuple(int(bg_color[c] * (1 - alpha) + base_color[c] * alpha) for c in range(3))
        thickness = max(1, int(round(1 + alpha * 2)))
        cv2.line(canvas, p0, p1, color, thickness, cv2.LINE_AA)


def draw_frame_trajectory(
    frame_idx: int, regions, poses, cam_polygons: Dict[str, np.ndarray], real_w: float, real_h: float,
    gt_raw: List[dict], track_outputs: List[Tuple[int, np.ndarray]],
    track_histories: Dict[int, "deque"], has_gt: bool,
) -> np.ndarray:
    scale = 40
    margin = 110
    w_px = int(real_w * scale) + margin
    h_px = int(real_h * scale) + 40
    # mp4(H.264/mp4v, yuv420p)はw/hが奇数だとchromaのアライメントが崩れて
    # 緑一色など壊れた映像になることがあるため、偶数に揃える
    if w_px % 2 != 0:
        w_px += 1
    if h_px % 2 != 0:
        h_px += 1
    canvas = np.full((h_px, w_px, 3), BG_COLOR_BGR, dtype=np.uint8)
    draw_regions(canvas, regions, scale)

    # 曲線境界(カメラごとの許容半径R_maxを床面に逆投影した多角形)を
    # 色分けした輪郭線で描く(直線境界版のオレンジの帯の代わり)
    for cid in CAM_IDS:
        poly = cam_polygons.get(cid)
        if poly is None or len(poly) < 3:
            continue
        pts = (poly * scale).astype(np.int32).reshape(-1, 1, 2)
        color = CAM_ZONE_COLORS_BGR.get(cid, (255, 255, 255))
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)

    for cid in CAM_IDS:
        p = poses[cid]
        px, py = int(p.t[0] * scale), int(p.t[1] * scale)
        if 0 <= px < w_px and 0 <= py < h_px:
            cv2.circle(canvas, (px, py), 9, (0, 0, 255), -1)
            cv2.circle(canvas, (px, py), 11, (255, 255, 255), 2)
            cv2.putText(canvas, cid, (px + 14, py - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # GT(アノテーションがあるフレームのみ、参考として薄く表示)
    if has_gt:
        for item in gt_raw:
            xy = item["world_xy"]
            px, py = int(xy[0] * scale), int(xy[1] * scale)
            if 0 <= px < w_px and 0 <= py < h_px:
                cv2.drawMarker(canvas, (px, py), GT_MARKER_COLOR_BGR, markerType=cv2.MARKER_CROSS,
                               markerSize=8, thickness=1)

    # 軌跡(直近TRAIL_LENGTHフレームの尾)
    for tid, _xy in track_outputs:
        hist = track_histories.get(tid)
        if hist:
            draw_trail(canvas, list(hist), scale, track_color(tid))

    # 現在位置(track_idで色分け、ラベル付き)
    for tid, xy in track_outputs:
        px, py = int(xy[0] * scale), int(xy[1] * scale)
        if 0 <= px < w_px and 0 <= py < h_px:
            color = track_color(tid)
            cv2.circle(canvas, (px, py), 7, color, -1)
            cv2.circle(canvas, (px, py), 9, (255, 255, 255), 1)
            cv2.putText(canvas, f"ID{tid}", (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, color, 1, cv2.LINE_AA)

    gt_label = "" if has_gt else "  (GT無し)"
    header = f"{frame_name(frame_idx)}  track={len(track_outputs)}{gt_label}  [曲線境界]"
    cv2.putText(canvas, header, (8, h_px - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas


def main():
    from ultralytics import YOLO

    print("floorplan.json 読み込み中...")
    regions, cameras_json, real_w, real_h = load_floorplan(FLOORPLAN_JSON_PATH)
    cameras_json = apply_camera_overrides(cameras_json)
    poses = resolve_poses(cameras_json, root="Area1", fmt=ROT_FMT)

    print(f"各カメラの曲線境界(許容半径R_max)を計算中(高さ仮定={ZONE_HEIGHT_M}m)...")
    cam_polygons = build_camera_polygons(poses, CAM_IDS, get_intrinsics, height_m=ZONE_HEIGHT_M, n_angles=N_ANGLES)
    adjacent_pairs = get_adjacent_pairs(CAM_IDS)

    print(f"YOLOモデル読み込み中: {YOLO_WEIGHTS_PATH}")
    yolo_model = YOLO(YOLO_WEIGHTS_PATH)

    xml_dir = Path(XML_DIR)
    image_dir = Path(IMAGE_DIR)
    frame_indices = list(range(FRAME_START, FRAME_END + 1, FRAME_STEP))
    print(f"対象フレーム数: {len(frame_indices)} "
          f"(frame_start={FRAME_START}, frame_end={FRAME_END}, step={FRAME_STEP})")
    print("※ フレーム数が多いため、実行には時間がかかる想定")

    if USE_REID_TRACKER:
        print(f"ReIDBevTracker使用(edge_margin_m={EDGE_MARGIN_M}, reid_dist_thr={REID_DIST_THR_M}, "
              f"lost_max_age={LOST_MAX_AGE})")
        tracker = ReIDBevTracker(
            dist_thr=DIST_THR_TRACK_M, max_missed=MAX_MISSED_FRAMES,
            real_w=real_w, real_h=real_h, edge_margin_m=EDGE_MARGIN_M,
            reid_dist_thr=REID_DIST_THR_M, lost_max_age=LOST_MAX_AGE,
        )
    else:
        print("SimpleBevTracker使用(ReIDなし旧版)")
        tracker = SimpleBevTracker(dist_thr=DIST_THR_TRACK_M, max_missed=MAX_MISSED_FRAMES)
    track_histories: Dict[int, deque] = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))

    saved_paths = []
    n_skipped_no_image = 0
    for frame_idx in frame_indices:
        xml_path = xml_dir / f"{frame_name(frame_idx)}.xml"
        image_path = image_dir / f"{frame_name(frame_idx)}.jpg"
        if not image_path.exists():
            n_skipped_no_image += 1
            continue

        # GTアノテーション(xml)が無いフレームでも軌跡画像は生成できるようにする
        has_gt = xml_path.exists()
        if has_gt:
            gt_objects = parse_voc_annotation(xml_path)
            gt_per_camera_raw = gt_world_points_per_camera(gt_objects, poses, IMAGE_WIDTH_PX)
            gt_per_camera_filtered = filter_by_curved_zone(gt_per_camera_raw, cam_polygons)
            gt_raw = [
                {"name": it["name"], "cam_id": cam_id, "world_xy": it["world_xy"]}
                for cam_id, items in gt_per_camera_filtered.items() for it in items
            ]
        else:
            gt_raw = []

        per_camera_raw = yolo_world_points_per_camera_with_pose(yolo_model, image_path, poses)
        per_camera_deduped = deduplicate_per_camera(per_camera_raw, DEDUP_IOU_THRESHOLD)
        per_camera_filtered = filter_by_curved_zone(per_camera_deduped, cam_polygons)
        # 牛舎(floorplan)の矩形範囲の外側に投影された検出(固定的な幽霊検出の主因)を除外
        per_camera_filtered = filter_outside_floorplan_bounds(
            per_camera_filtered, real_w, real_h, margin_m=0.0, verbose=False
        )
        fused_points = fuse_frame_zone_pose(
            per_camera_filtered, adjacent_pairs, DIST_THR_FUSION_M, USE_POSE_CONSTRAINT
        )

        # このフレームで実際にマッチした(=missed==0)track_idとその位置だけが返る
        track_outputs = tracker.update(fused_points)
        for tid, xy in track_outputs:
            track_histories[tid].append((float(xy[0]), float(xy[1])))

        canvas = draw_frame_trajectory(
            frame_idx, regions, poses, cam_polygons, real_w, real_h,
            gt_raw, track_outputs, track_histories, has_gt,
        )
        out_path = TRAJ_FRAMES_OUT_DIR / f"{frame_name(frame_idx)}.png"
        cv2.imwrite(str(out_path), canvas)
        saved_paths.append(out_path)

        if len(saved_paths) % 50 == 0:
            print(f"  {len(saved_paths)}/{len(frame_indices)} フレーム処理済み "
                  f"(直近: {frame_name(frame_idx)}, 現在track数={len(track_outputs)})")

    print(f"\n{len(saved_paths)}枚のBEV軌跡画像を保存しました: {TRAJ_FRAMES_OUT_DIR}")
    if n_skipped_no_image:
        print(f"(画像が見つからずスキップしたフレーム数: {n_skipped_no_image})")

    if MAKE_VIDEO and saved_paths:
        video_path = OUTPUT_DIR / f"bev_trajectory_yolo_zonepose_curved_{_TRACKER_SUFFIX}.mp4"
        make_video_from_frames(TRAJ_FRAMES_OUT_DIR, video_path, VIDEO_FPS, saved_paths)


if __name__ == "__main__":
    main()