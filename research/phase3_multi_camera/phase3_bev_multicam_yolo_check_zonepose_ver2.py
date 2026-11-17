"""
phase3_bev_multicam_yolo_check_zonepose.py

phase3_bev_multicam_yolo_check.py(YOLO検出+compute_camera_assignments_by_distance
による融合)に対応する、フレーム内融合ロジックだけを新しい3制約版
(make_bev_zone_pose_fusion.py / phase3_bev_multicam_gt_check_zonepose.pyと同じ:
 有効範囲外の除外 + 隣接カメラ制約 + 姿勢クラス制約、Hungarianコストのバグ修正込み、
 エピポーラ拘束なし)に差し替えたもの。

GT版(phase3_bev_multicam_gt_check_zonepose.py)ではXMLアノテーションにpose
(lying/standing)タグが無く姿勢クラス制約が実質スキップされていたが、
YOLOモデルは2クラス(0:cow/lying, 1:cow/standing)を出力するため、
ここでは姿勢クラス制約が実際に効く。GT版で有効範囲フィルタ+隣接カメラ制約
+Hungarian修正だけでも精度が上がったことを踏まえると、YOLOでは姿勢クラス制約が
さらに上乗せで効く可能性がある。

入力元・評価ロジックはphase3_bev_multicam_yolo_check.pyと同じ:
  - 各カメラのタイル画像でYOLO推論(カメラ別クロップに対して推論)
  - 評価用のGT(個体名・真の位置)は別途XMLから読み込み、YOLO由来のtrack出力と
    distance-gateで突き合わせてIDF1/MOTA/HOTAを算出する

--- 使い方 ---
    python phase3_bev_multicam_yolo_check_zonepose.py
"""

import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

MOOTRACK_CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calibration")
if MOOTRACK_CODE_DIR not in sys.path:
    sys.path.insert(0, MOOTRACK_CODE_DIR)

from Mootrack_core import (
    NUM_CAMERAS, CAM_IDS, ROT_FMT,
    Pose, get_intrinsics,
    load_floorplan_cameras, load_floorplan_dimensions, apply_camera_overrides, resolve_poses,
    unproject_to_floor,
    union_find_clusters,
    filter_outside_floorplan_bounds,
)

# ---------------------------------------------------------------------------
# 設定(phase3_bev_multicam_yolo_check.pyと共通の値はそのまま揃える)
# ---------------------------------------------------------------------------
XML_DIR = "/mnt/vmlqnap02/dataset/MooTrack360/1-hour-farm1-area1-sample-sequence/xml-1fps"
IMAGE_DIR = "/mnt/vmlqnap02/dataset/MooTrack360/1-hour-farm1-area1-sample-sequence/1-hour-farm1-area1-frames"
# 注意: このFLOORPLAN_JSON_PATHは以前から要検証(値が古い可能性あり)と指摘済みのファイル
FLOORPLAN_JSON_PATH = "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/calibration/Farm2_Area_1_Artboard_config_.json"
OUTPUT_DIR = Path(
    "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/research/idf1_check"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

YOLO_WEIGHTS_PATH = "/mnt/vmlqnap02/dataset/MooTrack360/Trained models/yolo11x/train_ultralytics/weights/best.pt"
YOLO_CONF_THRESH = 0.33

FRAME_START = 0
FRAME_END = 1470
FRAME_STEP = 10

IMAGE_WIDTH_PX = 8832

LYING_HEIGHT_M = 0.75   # 暫定値(一般的な牛のbbox中心の目安)。後でグリッドサーチ予定
STANDING_HEIGHT_M = 0.75
DEFAULT_HEIGHT_M = 0.8  # GT側でpose不明の場合、およびYOLOクラスが解決できない場合のフォールバック

DIST_THR_FUSION_M = 1.5     # カメラ間融合の距離閾値(既存版と同じ)
OVERLAP_HALF_WIDTH_M = 1.5    # 隣接カメラ境界からの片側の幅(合計3m幅)

DIST_THR_TRACK_M = 1.5
MAX_MISSED_FRAMES = 2

# アブレーション用スイッチ: Falseにすると姿勢クラス制約(lying/standing不一致での
# 統合除外)を無効化する。姿勢ラベルの一瞬のブレがID切り替わりやFPにどれだけ
# 寄与しているかを、この値だけ変えて再実行して比較する。
USE_POSE_CONSTRAINT = True

# 同一カメラ内の重複検出除去: IoUがこの閾値以上のペアを同一個体とみなし、
# bboxが大きい方だけを残す(小さい方=大抵は誤検出やズレた重複箱を捨てる)。
DEDUP_IOU_THRESHOLD = 0.5

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# YOLOモデルのクラスID -> pose (`m.names`で確認済み: {0: 'cow/lying', 1: 'cow/standing'})
YOLO_CLASS_TO_POSE = {0: "lying", 1: "standing"}


def frame_name(idx: int) -> str:
    return f"{idx:06d}"


def parse_voc_annotation(xml_path) -> List[dict]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        pose_el = obj.find("pose")
        pose = pose_el.text.strip().lower() if pose_el is not None and pose_el.text else None
        bnd = obj.find("bndbox")
        objects.append({
            "name": name,
            "pose": pose,
            "xmin": float(bnd.find("xmin").text),
            "ymin": float(bnd.find("ymin").text),
            "xmax": float(bnd.find("xmax").text),
            "ymax": float(bnd.find("ymax").text),
        })
    return objects


def height_for(pose: Optional[str]) -> float:
    if pose == "lying":
        return LYING_HEIGHT_M
    if pose == "standing":
        return STANDING_HEIGHT_M
    return DEFAULT_HEIGHT_M


def name_to_gt_id(name: str) -> int:
    m = re.search(r"\d+", name)
    if not m:
        raise ValueError(f"nameから数値IDを抽出できません: {name}")
    return int(m.group())


def gt_world_points_this_frame(objects: List[dict], poses: Dict[str, "Pose"], image_width: float) -> Dict[str, List[np.ndarray]]:
    """評価用: GT個体名 -> このフレームでの世界座標リスト(複数カメラに映っていれば複数)"""
    tile_w = image_width / NUM_CAMERAS
    result: Dict[str, List[np.ndarray]] = defaultdict(list)
    for obj in objects:
        center_x_full = (obj["xmin"] + obj["xmax"]) / 2.0
        cam_idx = int(center_x_full // tile_w)
        cam_idx = max(0, min(cam_idx, NUM_CAMERAS - 1))
        cam_id = CAM_IDS[cam_idx]

        fx, fy, cx, cy, xi, alpha = get_intrinsics(cam_id)
        local_x = center_x_full - cam_idx * tile_w
        local_y = (obj["ymin"] + obj["ymax"]) / 2.0
        pts_px = np.array([[local_x, local_y]], dtype=np.float64)

        height_m = height_for(obj["pose"])
        cam_pose = poses[cam_id]
        world_xy, valid = unproject_to_floor(
            pts_px, cam_pose.R, cam_pose.t, fx, fy, cx, cy, xi, alpha, floor_z=-height_m,
        )
        if valid[0]:
            result[obj["name"]].append(world_xy[0])
    return result


# ---------------------------------------------------------------------------
# YOLO検出を各カメラのタイルクロップで推論し、姿勢ラベル付きでワールド座標へ
# (name=Noneだが、poseはmatch_zone_pose_constrainedの姿勢クラス制約で使うので残す)
# ---------------------------------------------------------------------------

def yolo_world_points_per_camera_with_pose(
    yolo_model, image_path: Path, poses: Dict[str, "Pose"],
) -> Dict[str, List[dict]]:
    """cam_id -> [{"name": None, "pose": "lying"/"standing", "world_xy": ndarray}, ...]"""
    full_img = cv2.imread(str(image_path))
    per_camera: Dict[str, List[dict]] = {cid: [] for cid in CAM_IDS}
    if full_img is None:
        return per_camera

    tile_w = IMAGE_WIDTH_PX / NUM_CAMERAS

    for cam_idx, cam_id in enumerate(CAM_IDS):
        x0 = int(cam_idx * tile_w)
        x1_edge = int((cam_idx + 1) * tile_w)
        frame_img = full_img[:, x0:x1_edge]

        results = yolo_model.predict(
            frame_img, conf=YOLO_CONF_THRESH, device=DEVICE, verbose=False
        )[0]
        if len(results.boxes) == 0:
            continue

        fx, fy, cx, cy, xi, alpha = get_intrinsics(cam_id)
        pts_px, poses_list, bboxes = [], [], []
        for box in results.boxes:
            bx1, by1, bx2, by2 = box.xyxy[0].tolist()
            pts_px.append([(bx1 + bx2) / 2.0, (by1 + by2) / 2.0])
            poses_list.append(YOLO_CLASS_TO_POSE.get(int(box.cls[0])))
            bboxes.append((bx1, by1, bx2, by2))
        pts_px = np.array(pts_px, dtype=np.float64)
        heights = np.array([height_for(p) for p in poses_list], dtype=np.float64)
        floor_z = -heights

        cam_pose = poses[cam_id]
        world_xy, valid = unproject_to_floor(
            pts_px, cam_pose.R, cam_pose.t, fx, fy, cx, cy, xi, alpha,
            floor_z=floor_z,
        )
        for j in range(len(pts_px)):
            if valid[j]:
                per_camera[cam_id].append({
                    "name": None, "pose": poses_list[j],
                    "bbox_local": bboxes[j], "world_xy": world_xy[j],
                })

    return per_camera


def iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _bbox_area(bbox: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def deduplicate_by_iou(items: List[dict], iou_thr: float) -> List[dict]:
    """同一カメラ内の検出について、bbox IoUがiou_thr以上のペアを同一個体とみなし、
    bboxが大きい方だけを残す(NMSと似ているが、confでなくbbox面積の大小で選ぶ)。
    姿勢クラスが違っても対象にする(同じ牛がlying/standing両方で誤って検出される
    ケースをカバーするため、クラスを跨いで比較する)。"""
    order = sorted(range(len(items)), key=lambda i: _bbox_area(items[i]["bbox_local"]), reverse=True)
    kept: List[dict] = []
    for i in order:
        it = items[i]
        is_dup = any(iou_xyxy(it["bbox_local"], k["bbox_local"]) >= iou_thr for k in kept)
        if not is_dup:
            kept.append(it)
    return kept


def deduplicate_per_camera(per_camera: Dict[str, List[dict]], iou_thr: float) -> Dict[str, List[dict]]:
    return {cam_id: deduplicate_by_iou(items, iou_thr) for cam_id, items in per_camera.items()}


# ---------------------------------------------------------------------------
# 隣接カメラの重なり範囲・各カメラの有効範囲(フレームに依存しないので1回だけ計算)
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


def get_camera_valid_ranges(cam_ids: List[str], zones: Dict[Tuple[str, str], Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
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
    filtered: Dict[str, List[dict]] = {cid: [] for cid in per_camera}
    for cam_id, items in per_camera.items():
        lo, hi = valid_ranges[cam_id]
        for it in items:
            if lo <= it["world_xy"][0] <= hi:
                filtered[cam_id].append(it)
    return filtered


# ---------------------------------------------------------------------------
# 隣接カメラ制約 + 姿勢クラス制約でマッチング(Hungarianのコストは
# 閾値以内の組にしか実距離を入れない = バグ修正済み版)
# ---------------------------------------------------------------------------

def match_zone_pose_constrained(
    per_camera: Dict[str, List[dict]],
    cam_ids: List[str],
    zones: Dict[Tuple[str, str], Tuple[float, float]],
    dist_thr: float,
    use_pose_constraint: bool = True,
) -> Tuple[List[Tuple[Tuple[str, int], Tuple[str, int]]], int]:
    """戻り値は (accepted_pairs, n_pose_rejected)。
    n_pose_rejectedは「距離的には閾値内(=距離だけなら統合候補になれた)だったのに、
    姿勢クラス制約によって弾かれたペアの件数」。use_pose_constraintの効果を
    True/Falseで切り替えたときにこの数字が0になるかどうかで、制約が実際に
    機能しているかを直接確認できる。"""
    accepted_pairs: List[Tuple[Tuple[str, int], Tuple[str, int]]] = []
    n_pose_rejected = 0

    for cam_i, cam_j in zones.keys():
        items_i = per_camera[cam_i]
        items_j = per_camera[cam_j]
        if not items_i or not items_j:
            continue

        n_i, n_j = len(items_i), len(items_j)
        cost = np.full((n_i, n_j), 1e6)
        for a, it_i in enumerate(items_i):
            for b, it_j in enumerate(items_j):
                d = float(np.linalg.norm(it_i["world_xy"] - it_j["world_xy"]))
                pose_mismatch = (
                    it_i["pose"] is not None and it_j["pose"] is not None
                    and it_i["pose"] != it_j["pose"]
                )
                if use_pose_constraint and pose_mismatch:
                    if d <= dist_thr:
                        # 距離的にはOKだったのに、姿勢制約(有効時のみ)で弾いた候補。
                        # use_pose_constraint=Falseなら、この分岐自体に入らないので
                        # 必ず0件になる(=切り替えが実際に効いているかの直接証拠)。
                        n_pose_rejected += 1
                    continue
                if d <= dist_thr:
                    cost[a, b] = d
                # 閾値超えは生の距離を入れず1e6のままにする(Hungarianの誤選択を防ぐ)

        row_idx, col_idx = linear_sum_assignment(cost)
        for a, b in zip(row_idx, col_idx):
            if cost[a, b] <= dist_thr:
                accepted_pairs.append(((cam_i, a), (cam_j, b)))

    return accepted_pairs, n_pose_rejected


def fuse_frame_zone_pose(
    per_camera_filtered: Dict[str, List[dict]],
    cam_ids: List[str],
    zones: Dict[Tuple[str, str], Tuple[float, float]],
    dist_thr: float,
    use_pose_constraint: bool = True,
) -> Tuple[List[dict], int]:
    """有効範囲フィルタ済みのper_cameraから、zone+pose制約でマッチングし、
    Union-Findでクラスタ化して融合点リストを返す。
    YOLO検出にはnameが無いので、各要素は {"world_xy": ndarray, "n_cams": int} のみ。
    戻り値は (fused, n_pose_rejected)。"""
    accepted_pairs, n_pose_rejected = match_zone_pose_constrained(
        per_camera_filtered, cam_ids, zones, dist_thr, use_pose_constraint
    )
    all_keys = [(cam_id, idx) for cam_id, items in per_camera_filtered.items() for idx in range(len(items))]
    if not all_keys:
        return [], n_pose_rejected
    roots = union_find_clusters(all_keys, accepted_pairs)

    clusters: Dict[Tuple[str, int], List[np.ndarray]] = defaultdict(list)
    for key in all_keys:
        cam_id, idx = key
        clusters[roots[key]].append(per_camera_filtered[cam_id][idx]["world_xy"])

    fused = []
    for members in clusters.values():
        xy_mean = np.mean(np.array(members), axis=0)
        fused.append({"world_xy": xy_mean, "n_cams": len(members)})
    return fused, n_pose_rejected


# ---------------------------------------------------------------------------
# フレーム間追跡(既存版と同一)
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


TRACKEVAL_DIR = "/home/ara/myproject/project2/TrackEval"
if TRACKEVAL_DIR not in sys.path:
    sys.path.insert(0, TRACKEVAL_DIR)


def compute_hota_trackeval(frames_data: List[dict], dist_cap_m: float) -> dict:
    """本家TrackEvalのtrackeval.metrics.HOTAを直接呼び出す(既存版と同一ロジック)。"""
    import trackeval

    gt_name_to_idx: Dict[str, int] = {}
    trk_id_to_idx: Dict[int, int] = {}
    for fr in frames_data:
        for name in fr["gt_names"]:
            if name not in gt_name_to_idx:
                gt_name_to_idx[name] = len(gt_name_to_idx)
        for tid in fr["hyp_ids"]:
            if tid not in trk_id_to_idx:
                trk_id_to_idx[tid] = len(trk_id_to_idx)

    gt_ids_per_frame, tracker_ids_per_frame, similarity_per_frame = [], [], []
    num_gt_dets, num_tracker_dets = 0, 0

    for fr in frames_data:
        gt_idx = np.array([gt_name_to_idx[n] for n in fr["gt_names"]], dtype=int)
        trk_idx = np.array([trk_id_to_idx[t] for t in fr["hyp_ids"]], dtype=int)
        gt_ids_per_frame.append(gt_idx)
        tracker_ids_per_frame.append(trk_idx)
        num_gt_dets += len(gt_idx)
        num_tracker_dets += len(trk_idx)

        if len(gt_idx) == 0 or len(trk_idx) == 0:
            similarity_per_frame.append(np.zeros((len(gt_idx), len(trk_idx))))
            continue

        gt_pts = np.array(fr["gt_points"])
        hyp_pts = np.array(fr["hyp_points"])
        dist = np.linalg.norm(gt_pts[:, None, :] - hyp_pts[None, :, :], axis=-1)
        sim = np.clip(1.0 - dist / dist_cap_m, 0.0, 1.0)
        similarity_per_frame.append(sim)

    data = {
        "num_gt_ids": len(gt_name_to_idx),
        "num_tracker_ids": len(trk_id_to_idx),
        "num_gt_dets": num_gt_dets,
        "num_tracker_dets": num_tracker_dets,
        "gt_ids": gt_ids_per_frame,
        "tracker_ids": tracker_ids_per_frame,
        "similarity_scores": similarity_per_frame,
    }

    hota_metric = trackeval.metrics.HOTA()
    res = hota_metric.eval_sequence(data)
    return res


def main():
    import motmetrics as mm
    from ultralytics import YOLO

    print("floorplan.json 読み込み中...")
    cameras_json = load_floorplan_cameras(FLOORPLAN_JSON_PATH)
    real_w, real_h = load_floorplan_dimensions(FLOORPLAN_JSON_PATH)
    cameras_json = apply_camera_overrides(cameras_json)
    poses = resolve_poses(cameras_json, root="Area1", fmt=ROT_FMT)
    for cid in CAM_IDS:
        print(f"  {cid}: t={poses[cid].t.round(2)}")

    print("\n隣接カメラの重なり範囲(±{:.1f}m)を計算中...".format(OVERLAP_HALF_WIDTH_M))
    zones = get_adjacent_zones(poses, CAM_IDS, OVERLAP_HALF_WIDTH_M)
    valid_ranges = get_camera_valid_ranges(CAM_IDS, zones)
    for cid in CAM_IDS:
        lo, hi = valid_ranges[cid]
        print(f"  {cid} 有効範囲: [{lo:.2f}, {hi:.2f}]")

    print(f"\n姿勢クラス制約: {'有効' if USE_POSE_CONSTRAINT else '無効(アブレーション)'}")

    print(f"\nYOLOモデル読み込み中: {YOLO_WEIGHTS_PATH}")
    yolo_model = YOLO(YOLO_WEIGHTS_PATH)

    xml_dir = Path(XML_DIR)
    image_dir = Path(IMAGE_DIR)
    frame_indices = list(range(FRAME_START, FRAME_END + 1, FRAME_STEP))

    tracker = SimpleBevTracker(dist_thr=DIST_THR_TRACK_M, max_missed=MAX_MISSED_FRAMES)
    acc = mm.MOTAccumulator(auto_id=True)
    n_processed = 0
    frames_data: List[dict] = []
    total_pose_rejected = 0

    for frame_idx in frame_indices:
        xml_path = xml_dir / f"{frame_name(frame_idx)}.xml"
        image_path = image_dir / f"{frame_name(frame_idx)}.jpg"
        if not xml_path.exists() or not image_path.exists():
            continue

        gt_objects = parse_voc_annotation(xml_path)
        gt_world = gt_world_points_this_frame(gt_objects, poses, IMAGE_WIDTH_PX)
        gt_names = list(gt_world.keys())
        gt_ids = [name_to_gt_id(name) for name in gt_names]
        gt_points = [np.mean(np.array(gt_world[name]), axis=0) for name in gt_names]

        per_camera_raw = yolo_world_points_per_camera_with_pose(yolo_model, image_path, poses)
        n_yolo_raw = sum(len(v) for v in per_camera_raw.values())
        per_camera_deduped = deduplicate_per_camera(per_camera_raw, DEDUP_IOU_THRESHOLD)
        n_yolo_deduped = sum(len(v) for v in per_camera_deduped.values())
        per_camera_filtered = filter_out_of_range_detections(per_camera_deduped, valid_ranges)
        # 牛舎(floorplan)の矩形範囲の外側に投影された検出(固定的な幽霊検出の主因)を除外
        per_camera_filtered = filter_outside_floorplan_bounds(
            per_camera_filtered, real_w, real_h, margin_m=0.0, verbose=False
        )
        n_yolo_filtered = sum(len(v) for v in per_camera_filtered.values())

        fused_points, n_pose_rejected = fuse_frame_zone_pose(
            per_camera_filtered, CAM_IDS, zones, DIST_THR_FUSION_M, USE_POSE_CONSTRAINT
        )
        total_pose_rejected += n_pose_rejected

        track_outputs = tracker.update(fused_points)
        hyp_ids = [tid for tid, _xy in track_outputs]
        hyp_points = [xy for _tid, xy in track_outputs]

        if gt_points and hyp_points:
            dist = mm.distances.norm2squared_matrix(
                np.array(gt_points), np.array(hyp_points), max_d2=(2.0 * DIST_THR_TRACK_M) ** 2
            )
        else:
            dist = np.empty((len(gt_points), len(hyp_points)))

        acc.update(gt_ids, hyp_ids, dist)
        frames_data.append({
            "gt_names": gt_names, "gt_points": gt_points,
            "hyp_ids": hyp_ids, "hyp_points": hyp_points,
        })
        n_processed += 1
        print(f"  {frame_name(frame_idx)}: GT個体={len(gt_points)} YOLO検出={n_yolo_raw}"
              f"(重複除去後{n_yolo_deduped}, 有効範囲除外後{n_yolo_filtered}) "
              f"融合点={len(fused_points)} track出力={len(track_outputs)} "
              f"姿勢制約で除外={n_pose_rejected}")

    print(f"\n処理フレーム数: {n_processed}/{len(frame_indices)}")
    print(f"姿勢クラス制約(USE_POSE_CONSTRAINT={USE_POSE_CONSTRAINT})により、"
          f"距離的にはOKだったのに除外された候補ペア合計: {total_pose_rejected}")
    print("  (この値はUSE_POSE_CONSTRAINT=Falseなら実装上必ず0になる。"
          "Trueにしたときに0より大きい値が出れば、制約が実際にペアを弾いている証拠。"
          "0のままなら『配線は正しいが、このデータでは姿勢不一致が発生していない』ことを意味する)")

    suffix = "posecon" if USE_POSE_CONSTRAINT else "noposecon"

    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=["idf1", "idp", "idr", "mota", "motp", "num_switches",
                 "num_objects", "num_matches", "num_false_positives", "num_misses"],
        name=f"multicam_bev_yolo_zonepose_{suffix}",
    )
    strsummary = mm.io.render_summary(
        summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names
    )
    print(f"\n{strsummary}")

    dist_cap_m = 2.0 * DIST_THR_TRACK_M
    hota_res = compute_hota_trackeval(frames_data, dist_cap_m=dist_cap_m)
    print(f"\n=== HOTA(本家TrackEval, alpha=0.05〜0.95を距離閾値<= (1-alpha)*{dist_cap_m}mに対応させて計算) ===")
    print(f"HOTA(平均)  = {np.mean(hota_res['HOTA']):.3f}")
    print(f"DetA(平均)  = {np.mean(hota_res['DetA']):.3f}")
    print(f"AssA(平均)  = {np.mean(hota_res['AssA']):.3f}")
    print(f"LocA(平均)  = {np.mean(hota_res['LocA']):.3f}")
    print(f"HOTA(0)     = {hota_res['HOTA(0)']:.3f}  (alpha->0での極限値、参考)")

    print(f"\n(参考: phase3_bev_multicam_yolo_check.py [従来の距離Hungarian融合版] と比較すること。"
          f"GTベースzone+pose版の結果、単一カメラPhase2実運用 IDF1=86.4%も参照)")

    csv_path = OUTPUT_DIR / f"phase3_bev_multicam_yolo_zonepose_{suffix}_summary.csv"
    summary.to_csv(csv_path)
    hota_csv_path = csv_path.with_name(f"phase3_bev_multicam_yolo_zonepose_{suffix}_hota.csv")
    with open(hota_csv_path, "w") as f:
        f.write("alpha,HOTA,DetA,AssA,LocA\n")
        for a, alpha in enumerate(np.arange(0.05, 0.99, 0.05)):
            f.write(f"{alpha:.2f},{hota_res['HOTA'][a]},{hota_res['DetA'][a]},"
                    f"{hota_res['AssA'][a]},{hota_res['LocA'][a]}\n")
    print(f"\nCSV保存: {csv_path}")
    print(f"HOTA内訳CSV保存: {hota_csv_path}")


if __name__ == "__main__":
    main()