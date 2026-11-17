"""
phase3_bev_multicam_yolo_check_epipolar.py

phase3_bev_multicam_yolo_check_zonepose.py(有効範囲外の除外+隣接カメラ制約+
姿勢クラス制約、距離閾値ベースの統合)に対応する、統合ロジックをエピポーラ幾何学
(レイの最近接距離)ベースに差し替えた版。

【これまでの経緯・なぜ差し替えるか】
zonepose版は「特定の高さ(lying=0.5m/standing=0.75m)を仮定して床面に投影した
2D座標」同士の距離で統合を判定していた。この方式には構造的な弱点があり、
実際に以下の不具合が確認された:
  - 高さ仮定の誤差により、境界(隣接カメラの重なり範囲)にいる同一個体の投影が
    片方のカメラの「有効範囲」からわずかにはみ出し、統合を試みる前に片割れが
    消えてしまうケースがある
  - 距離閾値を緩めると、今度は境界付近に密集する別々の個体を誤って統合して
    しまう(過剰統合、FN急増)

エピポーラ幾何学は、検出ピクセルを「特定の高さへ投影した2D点」に潰さず、
カメラ中心から伸びる3Dの視線(レイ)のまま扱う。2つのカメラの視線が実空間で
どれだけ近づくか(最近接距離)で同一個体かどうかを判定するため、高さを先に
決め打ちする必要が無く、上記の構造的弱点を持たない。

【ハイブリッド方式(2026-09追記)】当初は統合成立時の最終座標も2本のレイの
最近接点(の中点)=生の三角測量結果をそのまま採用していたが、これは
レイの交差角が浅い場面で検出ノイズが大きく増幅され、フレーム間で座標が
跳ねてIDスイッチが急増する(IDs=41〜53)問題を引き起こすことが視覚化に
より判明した。そこで、「統合するかどうかの判定」にはエピポーラ(レイの
最近接距離)をそのまま使い続け、「統合成立後の最終座標」は生の三角測量点
ではなく、統合された各カメラの高さ仮定ベースの投影位置(world_xy_fallback)
の平均に差し替えた。境界での統合漏れ解消というエピポーラ本来の利点は
判定ロジックに残したまま、座標の安定性はfallback方式に近い値で確保する。

有効範囲フィルタ(filter_out_of_range_detections)や重なり範囲(OVERLAP_HALF_WIDTH_M)
の概念は、zonepose版特有の弱点を回避するための対症療法だったため、この版では
使わない。残すのは「隣接カメラ制約(物理的に重ならないCamera_1-Camera_3は除外)」
「姿勢クラス制約(任意)」「同一カメラ内の重複除去(IoUベース)」の3つ。
単独カメラでしか検出されなかった個体(統合相手がいない)は、レイ単体では
奥行きが定まらないので、従来通り姿勢別高さ仮定(unproject_to_floor)で
位置を決める。

--- 使い方 ---
    python phase3_bev_multicam_yolo_check_epipolar.py
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
    load_floorplan_cameras, apply_camera_overrides, resolve_poses,
    unproject_ds, unproject_to_floor,
    union_find_clusters,
)

# ---------------------------------------------------------------------------
# 設定(zonepose版と共通の値はそのまま揃える)
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
FRAME_END = 290
FRAME_STEP = 10

IMAGE_WIDTH_PX = 8832

# 単独カメラでしか検出されなかった個体(三角測量できない)の位置決めにのみ使う、
# 従来通りの高さ仮定。エピポーラでの統合判定そのものには使わない。
LYING_HEIGHT_M = 0.5
STANDING_HEIGHT_M = 0.75
DEFAULT_HEIGHT_M = 0.8

# レイの最近接距離の閾値(この距離以下なら同一個体とみなす)。
# zoneposeのDIST_THR_FUSION_Mとは全く別物の量(床面2D距離ではなく3D空間でのレイの
# すれ違い距離)なので、意味も適正値も別に検討・チューニングする必要がある。
RAY_DIST_THR_M = 0.5

# 「統合相手の一意性マージン」。最良候補と次点候補のレイ最近接距離の差が
# この値未満の場合、僅差すぎて区別がつかない(=密集地帯で別々の候補が
# ほぼ同じコストになっている)とみなし、統合を見送って単独カメラの
# fallback位置に任せる。可視化デバッグで、密集地帯の一部の個体が
# フレームごとに統合相手を取り違えて座標が跳ね、ID切り替えの原因になって
# いることが確認できたための対策。値は要チューニング。
MARGIN_GAP_M = 0.001

# 三角測量した点の高さがこの範囲(床から0〜1.6m)に収まっていなければ、
# 「牛としてありえない高さ」として候補から除外する。floor_z=-heightの符号規則に
# 合わせて、ここはワールドz座標(床が概ね0、高さが上がるほど負)で表す。
HEIGHT_RANGE_Z = (-1.6, 0.1)

USE_POSE_CONSTRAINT = True
DEDUP_IOU_THRESHOLD = 0.5

DIST_THR_TRACK_M = 1.0
MAX_MISSED_FRAMES = 2

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
YOLO_CLASS_TO_POSE = {0: "lying", 1: "standing"}

ADJACENT_PAIRS: List[Tuple[str, str]] = list(zip(CAM_IDS[:-1], CAM_IDS[1:]))


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
    """評価用: GT個体名 -> このフレームでの世界座標リスト(zonepose版と同一ロジック)"""
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
# YOLO検出 -> ワールド座標系でのレイ(方向ベクトル)+ フォールバック用の
# 高さ仮定ベースの2D位置(単独カメラ検出の場合にのみ使う)
# ---------------------------------------------------------------------------

def unproject_to_ray(pts_px, R, fx, fy, cx, cy, xi, alpha):
    """unproject_dsの結果(カメラローカルの方向ベクトル)をワールド座標系の
    方向ベクトルに変換する。カメラ中心はPose.tそのものなので別途扱う。"""
    dirs_cam, valid = unproject_ds(pts_px, fx, fy, cx, cy, xi, alpha)
    dirs_world = (R @ dirs_cam.T).T
    return dirs_world, valid


def yolo_detections_per_camera(
    yolo_model, image_path: Path, poses: Dict[str, "Pose"],
) -> Dict[str, List[dict]]:
    """cam_id -> [{"pose":..., "bbox_local":..., "dir_world": ndarray,
                   "world_xy_fallback": ndarray}, ...]"""
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
        cam_pose = poses[cam_id]

        pts_px, poses_list, bboxes = [], [], []
        for box in results.boxes:
            bx1, by1, bx2, by2 = box.xyxy[0].tolist()
            pts_px.append([(bx1 + bx2) / 2.0, (by1 + by2) / 2.0])
            poses_list.append(YOLO_CLASS_TO_POSE.get(int(box.cls[0])))
            bboxes.append((bx1, by1, bx2, by2))
        pts_px = np.array(pts_px, dtype=np.float64)

        dirs_world, valid_ray = unproject_to_ray(pts_px, cam_pose.R, fx, fy, cx, cy, xi, alpha)

        heights = np.array([height_for(p) for p in poses_list], dtype=np.float64)
        world_xy_fb, valid_fb = unproject_to_floor(
            pts_px, cam_pose.R, cam_pose.t, fx, fy, cx, cy, xi, alpha, floor_z=-heights,
        )

        for j in range(len(pts_px)):
            if valid_ray[j] and valid_fb[j]:
                per_camera[cam_id].append({
                    "pose": poses_list[j],
                    "bbox_local": bboxes[j],
                    "dir_world": dirs_world[j],
                    "world_xy_fallback": world_xy_fb[j],
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
# エピポーラ(レイの最近接距離)による統合
# ---------------------------------------------------------------------------

def closest_points_between_rays(
    p1: np.ndarray, d1: np.ndarray, p2: np.ndarray, d2: np.ndarray,
) -> Tuple[float, float, np.ndarray, np.ndarray, float]:
    """2本の半直線(レイ) p1+t1*d1 (t1>=0), p2+t2*d2 (t2>=0) の最近接点を求める。
    d1, d2は単位ベクトル。
    戻り値: (t1, t2, point1, point2, dist)。
    t1<=0またはt2<=0はカメラの後ろ側で最近接する(=幾何的にありえない)ことを示す。"""
    w0 = p1 - p2
    a = float(np.dot(d1, d1))
    b = float(np.dot(d1, d2))
    c = float(np.dot(d2, d2))
    d = float(np.dot(d1, w0))
    e = float(np.dot(d2, w0))
    denom = a * c - b * b

    if abs(denom) < 1e-9:
        # ほぼ平行なレイ(このカメラ配置ではまず起きないはずだが念のため)
        return -1.0, -1.0, p1, p2, float("inf")

    t1 = (b * e - c * d) / denom
    t2 = (a * e - b * d) / denom
    point1 = p1 + t1 * d1
    point2 = p2 + t2 * d2
    dist = float(np.linalg.norm(point1 - point2))
    return t1, t2, point1, point2, dist


def match_epipolar(
    per_camera: Dict[str, List[dict]],
    poses: Dict[str, "Pose"],
    adjacent_pairs: List[Tuple[str, str]],
    ray_dist_thr: float,
    height_range: Tuple[float, float],
    use_pose_constraint: bool = True,
    margin_gap: float = MARGIN_GAP_M,
) -> Tuple[List[Tuple[Tuple[str, int], Tuple[str, int]]], Dict[Tuple[Tuple[str, int], Tuple[str, int]], np.ndarray]]:
    """隣接カメラペアについて、レイの最近接距離が閾値以下の候補をHungarianで
    1対1割当する(閾値超えは1e6にしてから割当 = zonepose版と同じバグ対策)。
    さらに、採用されたペアについて「同じ行(カメラiの検出)・同じ列(カメラjの
    検出)にある次点候補」とのコスト差がmargin_gap未満なら、僅差すぎて
    区別がつかない(密集地帯で別の個体が候補になっている)とみなし採用を
    取り消す(=単独カメラのfallbackに任せる)。
    戻り値: (accepted_pairs, triangulated) triangulatedはペアごとの3D三角測量点。"""
    accepted_pairs: List[Tuple[Tuple[str, int], Tuple[str, int]]] = []
    triangulated: Dict[Tuple[Tuple[str, int], Tuple[str, int]], np.ndarray] = {}

    for cam_i, cam_j in adjacent_pairs:
        items_i = per_camera[cam_i]
        items_j = per_camera[cam_j]
        if not items_i or not items_j:
            continue

        p_i, p_j = poses[cam_i].t, poses[cam_j].t
        n_i, n_j = len(items_i), len(items_j)
        cost = np.full((n_i, n_j), 1e6)
        cand_points: Dict[Tuple[int, int], np.ndarray] = {}

        for a, it_i in enumerate(items_i):
            for b, it_j in enumerate(items_j):
                if (use_pose_constraint and it_i["pose"] is not None
                        and it_j["pose"] is not None and it_i["pose"] != it_j["pose"]):
                    continue
                t1, t2, pt1, pt2, dist = closest_points_between_rays(
                    p_i, it_i["dir_world"], p_j, it_j["dir_world"]
                )
                if t1 <= 0 or t2 <= 0:
                    continue  # カメラの後ろ側 = 幾何的にありえない
                mid = (pt1 + pt2) / 2.0
                if not (height_range[0] <= mid[2] <= height_range[1]):
                    continue  # 牛としてありえない高さ
                if dist <= ray_dist_thr:
                    cost[a, b] = dist
                    cand_points[(a, b)] = mid
                # 閾値超えは生の距離を入れず1e6のままにする(Hungarianの誤選択を防ぐ)

        row_idx, col_idx = linear_sum_assignment(cost)
        for a, b in zip(row_idx, col_idx):
            if cost[a, b] > ray_dist_thr:
                continue
            best = cost[a, b]
            # 同じ行(カメラiの検出a)・同じ列(カメラjの検出b)にある、この
            # ペア以外の候補のうち最小コスト = 次点候補
            row_alt = np.delete(cost[a, :], b)
            col_alt = np.delete(cost[:, b], a)
            alt_best = min(
                row_alt.min() if row_alt.size else np.inf,
                col_alt.min() if col_alt.size else np.inf,
            )
            if alt_best - best < margin_gap:
                continue  # 僅差で区別がつかない -> 統合を見送る
            key = ((cam_i, a), (cam_j, b))
            accepted_pairs.append(key)
            triangulated[key] = cand_points[(a, b)]

    return accepted_pairs, triangulated


def fuse_frame_epipolar(
    per_camera: Dict[str, List[dict]],
    poses: Dict[str, "Pose"],
    adjacent_pairs: List[Tuple[str, str]],
    ray_dist_thr: float,
    height_range: Tuple[float, float],
    use_pose_constraint: bool = True,
    margin_gap: float = MARGIN_GAP_M,
) -> List[dict]:
    """エピポーラ統合(=「同一個体かどうか」の判定にのみ使う)+Union-Findで
    クラスタ化し、融合点リストを返す。

    【ハイブリッド方式】判定はエピポーラ(レイの最近接距離)のロバストさを
    そのまま活かす一方、最終的な2D座標には生の三角測量点(pt1,pt2の中点)を
    使わない。三角測量点はレイの交差角が浅いと検出ノイズが大きく増幅され、
    フレーム間で座標が跳ねてトラッカーのID安定性を壊すことが確認されたため。
    代わりに、クラスタに属する各カメラの「高さ仮定ベースの投影位置
    (world_xy_fallback)」の平均を採用する。単独カメラのみのクラスタは
    メンバーが1つだけなので、この式は従来のfallback方式と自然に一致する
    (=平均を取る対象が1点だけになるだけ)。

    【一意性マージン】密集地帯では、統合相手の候補が複数僅差で存在し、
    フレームごとにどちらが選ばれるか入れ替わって座標が跳ねる(視覚化デバッグ
    で確認済み)。match_epipolar側で、最良候補と次点候補の差がmargin_gap未満
    の場合は統合を見送るようにして、この入れ替わりを抑える。"""
    accepted_pairs, _triangulated = match_epipolar(
        per_camera, poses, adjacent_pairs, ray_dist_thr, height_range, use_pose_constraint, margin_gap
    )
    all_keys = [(cam_id, idx) for cam_id, items in per_camera.items() for idx in range(len(items))]
    if not all_keys:
        return []
    roots = union_find_clusters(all_keys, accepted_pairs)

    clusters: Dict[Tuple[str, int], List[Tuple[str, int]]] = defaultdict(list)
    for k in all_keys:
        clusters[roots[k]].append(k)

    fused = []
    for root, members in clusters.items():
        fallback_pts = np.array(
            [per_camera[cam_id][idx]["world_xy_fallback"] for cam_id, idx in members]
        )
        xy = fallback_pts.mean(axis=0)
        # membersを残しておく: 「統合相手がフレームごとに入れ替わっていないか」を
        # 追跡側で検証するためのデバッグ情報(cam_id, per_camera内でのindex)。
        fused.append({"world_xy": xy, "n_cams": len(members), "members": members})
    return fused


# ---------------------------------------------------------------------------
# フレーム間追跡(zonepose版と同一)
# ---------------------------------------------------------------------------

class SimpleBevTracker:
    def __init__(self, dist_thr: float, max_missed: int):
        self.dist_thr = dist_thr
        self.max_missed = max_missed
        self.tracks: Dict[int, dict] = {}
        self.next_id = 1

    def update(self, fused_points: List[dict]) -> List[Tuple[int, np.ndarray, List[Tuple[str, int]]]]:
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
                    self.tracks[tid]["members"] = fused_points[j].get("members", [])
                    matched_trk.add(tid)
                    matched_det.add(j)

        for tid in track_ids:
            if tid not in matched_trk:
                self.tracks[tid]["missed"] += 1

        self.tracks = {tid: t for tid, t in self.tracks.items() if t["missed"] <= self.max_missed}

        for j, fp in enumerate(fused_points):
            if j not in matched_det:
                self.tracks[self.next_id] = {
                    "xy": fp["world_xy"], "missed": 0, "members": fp.get("members", []),
                }
                self.next_id += 1

        return [
            (tid, t["xy"], t.get("members", []))
            for tid, t in self.tracks.items() if t["missed"] == 0
        ]


TRACKEVAL_DIR = "/home/ara/myproject/project2/TrackEval"
if TRACKEVAL_DIR not in sys.path:
    sys.path.insert(0, TRACKEVAL_DIR)


def compute_hota_trackeval(frames_data: List[dict], dist_cap_m: float) -> dict:
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
    cameras_json = apply_camera_overrides(cameras_json)
    poses = resolve_poses(cameras_json, root="Area1", fmt=ROT_FMT)
    for cid in CAM_IDS:
        print(f"  {cid}: t={poses[cid].t.round(2)}")

    print(f"\n隣接カメラペア: {ADJACENT_PAIRS}")
    print(f"姿勢クラス制約: {'有効' if USE_POSE_CONSTRAINT else '無効'}")
    print(f"レイ最近接距離の閾値: {RAY_DIST_THR_M}m / 高さ範囲チェック: {HEIGHT_RANGE_Z}")
    print(f"統合相手の一意性マージン: {MARGIN_GAP_M}m")

    print(f"\nYOLOモデル読み込み中: {YOLO_WEIGHTS_PATH}")
    yolo_model = YOLO(YOLO_WEIGHTS_PATH)

    xml_dir = Path(XML_DIR)
    image_dir = Path(IMAGE_DIR)
    frame_indices = list(range(FRAME_START, FRAME_END + 1, FRAME_STEP))

    tracker = SimpleBevTracker(dist_thr=DIST_THR_TRACK_M, max_missed=MAX_MISSED_FRAMES)
    acc = mm.MOTAccumulator(auto_id=True)
    n_processed = 0
    frames_data: List[dict] = []
    # 「統合相手がフレームごとに入れ替わっていないか」を確認するためのデバッグ用状態。
    # track_id -> 前フレームでの統合相手の {cam_id: bbox中心のローカルピクセル座標}
    prev_partner_sig: Dict[int, Dict[str, Tuple[float, float]]] = {}

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

        per_camera_raw = yolo_detections_per_camera(yolo_model, image_path, poses)
        n_yolo_raw = sum(len(v) for v in per_camera_raw.values())
        per_camera_deduped = deduplicate_per_camera(per_camera_raw, DEDUP_IOU_THRESHOLD)
        n_yolo_deduped = sum(len(v) for v in per_camera_deduped.values())

        fused_points = fuse_frame_epipolar(
            per_camera_deduped, poses, ADJACENT_PAIRS, RAY_DIST_THR_M, HEIGHT_RANGE_Z, USE_POSE_CONSTRAINT,
            MARGIN_GAP_M
        )
        n_merged = sum(1 for fp in fused_points if fp["n_cams"] > 1)

        track_outputs = tracker.update(fused_points)
        hyp_ids = [tid for tid, _xy, _members in track_outputs]
        hyp_points = [xy for _tid, xy, _members in track_outputs]

        # --- デバッグ: 統合相手(n_cams=2のペア)がフレームごとに入れ替わっていないか ---
        # track_id -> {cam_id: (bbox中心のローカルピクセル座標)}
        curr_partner_px: Dict[int, Dict[str, Tuple[float, float]]] = {}
        for tid, xy, members in track_outputs:
            if len(members) != 2:
                continue
            per_cam_px = {}
            for cam_id, idx in members:
                bx1, by1, bx2, by2 = per_camera_deduped[cam_id][idx]["bbox_local"]
                per_cam_px[cam_id] = ((bx1 + bx2) / 2.0, (by1 + by2) / 2.0)
            curr_partner_px[tid] = per_cam_px
            prev_px = prev_partner_sig.get(tid)
            if prev_px is None:
                print(f"    [merge-track] id={tid} xy={xy.round(2)} cams={sorted(per_cam_px.keys())} (新規)")
            else:
                if set(prev_px.keys()) != set(per_cam_px.keys()):
                    print(f"    [merge-track] id={tid} xy={xy.round(2)} ⚠ カメラ構成が変化: "
                          f"{sorted(prev_px.keys())} -> {sorted(per_cam_px.keys())}")
                else:
                    moved = {
                        cam_id: round(float(np.linalg.norm(
                            np.array(per_cam_px[cam_id]) - np.array(prev_px[cam_id])
                        )), 1)
                        for cam_id in per_cam_px
                    }
                    print(f"    [merge-track] id={tid} xy={xy.round(2)} "
                          f"各カメラ内でのbbox中心ピクセル移動量={moved}")
        prev_partner_sig = curr_partner_px

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
              f"(重複除去後{n_yolo_deduped}) 融合点={len(fused_points)}(うち2カメラ統合={n_merged}) "
              f"track出力={len(track_outputs)}")

    print(f"\n処理フレーム数: {n_processed}/{len(frame_indices)}")

    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=["idf1", "idp", "idr", "mota", "motp", "num_switches",
                 "num_objects", "num_matches", "num_false_positives", "num_misses"],
        name="multicam_bev_yolo_epipolar",
    )
    strsummary = mm.io.render_summary(
        summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names
    )
    print(f"\n{strsummary}")
    print(f"\n(参考: phase3_bev_multicam_yolo_check_zonepose.py [距離閾値+ゾーン制約版] "
          f"と比較すること)")

    dist_cap_m = 2.0 * DIST_THR_TRACK_M
    hota_res = compute_hota_trackeval(frames_data, dist_cap_m=dist_cap_m)
    print(f"\n=== HOTA(本家TrackEval, alpha=0.05〜0.95を距離閾値<= (1-alpha)*{dist_cap_m}mに対応させて計算) ===")
    print(f"HOTA(平均)  = {np.mean(hota_res['HOTA']):.3f}")
    print(f"DetA(平均)  = {np.mean(hota_res['DetA']):.3f}")
    print(f"AssA(平均)  = {np.mean(hota_res['AssA']):.3f}")
    print(f"LocA(平均)  = {np.mean(hota_res['LocA']):.3f}")
    print(f"HOTA(0)     = {hota_res['HOTA(0)']:.3f}  (alpha->0での極限値、参考)")

    csv_path = OUTPUT_DIR / "phase3_bev_multicam_yolo_epipolar_summary.csv"
    summary.to_csv(csv_path)
    hota_csv_path = csv_path.with_name("phase3_bev_multicam_yolo_epipolar_hota.csv")
    with open(hota_csv_path, "w") as f:
        f.write("alpha,HOTA,DetA,AssA,LocA\n")
        for a, alpha in enumerate(np.arange(0.05, 0.99, 0.05)):
            f.write(f"{alpha:.2f},{hota_res['HOTA'][a]},{hota_res['DetA'][a]},"
                    f"{hota_res['AssA'][a]},{hota_res['LocA'][a]}\n")
    print(f"\nCSV保存: {csv_path}")
    print(f"HOTA内訳CSV保存: {hota_csv_path}")


if __name__ == "__main__":
    main()