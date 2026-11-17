"""
アノテーションデータ(VOC形式XML)を使って、
「bboxをワールド座標へ変換し、カメラ間のworld polygon IOUが閾値以上なら同一個体とみなす」
という従来手法(bev_matching_with_visualization.py の手法)の精度を検証するスクリプト。

YOLOによる検出は使わない。アノテーション済みのbbox+個体名(name)を正解として使うため。
時系列(複数フレーム)の追跡精度ではなく、1フレーム内でのカメラ間の個体統合精度が
知りたいものなので、対象は1つのアノテーションXMLのみとする。

検証ロジック:
1. 対象フレームのXMLから bbox(xmin,ymin,xmax,ymax) と個体名(name) を読み込む
2. bboxの中心x座標からどのカメラタイルに属すか判定し、Double Sphereモデルで
   ワールド座標(床平面)へ逆投影してPolygonを作る
3. カメラペアごとに、コスト行列(1-IOU)を使ったHungarian法(linear_sum_assignment)で
   1対1の最適割当を解く。これは bev_matching_with_visualization.py の
   match_camera_pair と同じロジック。IOUが高い箱が複数あっても、割当は1つだけに絞られる
4. Hungarianで割り当てられ、かつIOU >= 閾値のペアを union_find_merge と同じロジックで
   推移的に統合し、最終的な個体クラスタ(≒個体ID)を作る。
   (Camera1-2とCamera2-3が別々にマッチしていれば、Camera1-3が直接マッチしていなくても
   同じクラスタに統合される。これは実際の従来手法と同じ挙動)
5. 正解: 2つのbboxのnameが同じなら「本来同一個体」、違えば「本来別個体」
   予測: 同じクラスタに属していれば「同一個体と判定(統合)」
6. 複数カメラに映っている個体(name一致ペア)のうち、実際に同じクラスタに統合できた
   割合(recall)と、誤って別個体を同じクラスタに統合してしまった割合(推移的な連鎖
   誤統合も含む)を集計する

出力される「統合成功率」が、知りたかった
「複数カメラに映っている乳牛のうち、正しく個体統合できた割合」に相当する。
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import dataclass
from itertools import combinations
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon

import json


# ---------------------------------------------------------------------------
# 設定 (bev_matching_with_visualization.py と同じ値を使う。変更した場合は両方に反映すること)
# ---------------------------------------------------------------------------

FLOORPLAN_JSON_PATH = "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/calibration/Farm2_Area_1_Artboard_config_.json"

# 検証対象のアノテーションXML(1フレーム分)。カメラ間の個体統合の検証がメインなので
# 時系列(複数フレーム)ではなく1ファイルのみを対象にする。
ANNOTATION_XML_PATH = "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/annotation_process/3600_frames_result/annotations/000000.xml"

NUM_CAMERAS = 3
CAM_IDS = ["Camera_1", "Camera_2", "Camera_3"]

DS_FX, DS_FY = 708.5, 735.2
DS_CX, DS_CY = 1472.0, 1470.1
DS_XI, DS_ALPHA = 0.006105, 0.48977

CAM_INTRINSICS = {
    "Camera_2": {
        "fx": 708.5, "fy": 735.2, "cx": 1472.0, "cy": 1470.1,
        "xi": 0.056105, "alpha": 0.428780,
    },
    "Camera_3": {
        "fx": 708.5, "fy": 735.2, "cx": 1472.0, "cy": 1470.1,
        "xi": -0.143895, "alpha": 0.533116,
    },
}

CAMERA_PARAM_OVERRIDES = {
    "Camera_1->Camera_2": [13.113, 0.1269, -0.0743, -7.2382, 1.3079, -0.8149],
    "Camera_2->Camera_3": [13.641, 0.1279, -1.6374, 11.0855, -0.3336, 0.7534],
}

ROT_FMT = "euler_zyx_deg"
BEV_PLANE_HEIGHT_M = 0.0  # 床面 = 0.0

# 検証したいIOU閾値の一覧(感度分析のため複数試す)
IOU_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


@dataclass
class Pose:
    R: np.ndarray
    t: np.ndarray


def get_intrinsics(cam_id: str) -> Tuple[float, float, float, float, float, float]:
    if cam_id in CAM_INTRINSICS:
        p = CAM_INTRINSICS[cam_id]
        return p["fx"], p["fy"], p["cx"], p["cy"], p["xi"], p["alpha"]
    return DS_FX, DS_FY, DS_CX, DS_CY, DS_XI, DS_ALPHA


def load_floorplan_cameras(path: str):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data.get("cameras", [])


def apply_camera_overrides(cameras_json: list) -> list:
    result = []
    for entry in cameras_json:
        key = f"{entry['parent']}->{entry['child']}"
        new_entry = dict(entry)
        if key in CAMERA_PARAM_OVERRIDES:
            new_entry["params"] = list(CAMERA_PARAM_OVERRIDES[key])
        result.append(new_entry)
    return result


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def build_R(a, b, c, fmt: str) -> np.ndarray:
    if "_deg" in fmt:
        a, b, c = np.radians([a, b, c])
        base_fmt = fmt.replace("_deg", "")
    else:
        base_fmt = fmt
    if base_fmt == "euler_zyx":
        return _rz(a) @ _ry(b) @ _rx(c)
    raise ValueError(f"このスクリプトは euler_zyx(_deg) のみサポート: {fmt}")


def resolve_poses(cameras_json: list, root: str = "Area1", fmt: str = ROT_FMT) -> Dict[str, Pose]:
    poses: Dict[str, Pose] = {root: Pose(R=np.eye(3), t=np.zeros(3))}
    remaining = list(cameras_json)
    while remaining:
        progressed = False
        for entry in list(remaining):
            parent, child = entry["parent"], entry["child"]
            if parent not in poses:
                continue
            px, py, pz, a, b, c = entry["params"]
            R_abs = build_R(a, b, c, fmt)
            t_local = np.array([px, py, pz], dtype=np.float64)
            t_world = poses[parent].t + t_local
            poses[child] = Pose(R=R_abs, t=t_world)
            remaining.remove(entry)
            progressed = True
        if not progressed:
            raise RuntimeError(f"解決できないカメラがあります: {[e['child'] for e in remaining]}")
    return poses


def unproject_ds(pts_px, fx, fy, cx, cy, xi, alpha):
    mx = (pts_px[:, 0] - cx) / fx
    my = (pts_px[:, 1] - cy) / fy
    r2 = mx ** 2 + my ** 2

    if alpha <= 0.5:
        valid = np.ones(len(pts_px), dtype=bool)
    else:
        valid = r2 < 1.0 / (2.0 * alpha - 1.0) - 1e-9

    inner = np.maximum(1.0 - (2.0 * alpha - 1.0) * r2, 0.0)
    mz = (1.0 - alpha ** 2 * r2) / (alpha * np.sqrt(inner) + (1.0 - alpha))

    scale = np.sqrt(mx ** 2 + my ** 2 + mz ** 2)
    safe = valid & (scale > 1e-9)
    dx = np.where(safe, mx, 0.0)
    dy = np.where(safe, my, 0.0)
    dz = np.where(safe, mz - xi * scale, 0.0)

    norm = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    norm = np.where(norm > 1e-9, norm, 1.0)
    dirs = np.stack([dx / norm, dy / norm, dz / norm], axis=-1)
    return dirs, safe


def unproject_to_floor(pts_px, R, t, fx, fy, cx, cy, xi, alpha, floor_z=0.0):
    dirs_cam, valid = unproject_ds(pts_px, fx, fy, cx, cy, xi, alpha)
    dirs_world = (R @ dirs_cam.T).T

    dz = dirs_world[:, 2]
    t_safe = np.abs(dz) > 1e-9
    tt = np.where(t_safe, (floor_z - t[2]) / dz, 0.0)
    valid = valid & t_safe & (tt > 0)

    world_x = np.where(valid, t[0] + tt * dirs_world[:, 0], 0.0)
    world_y = np.where(valid, t[1] + tt * dirs_world[:, 1], 0.0)
    return np.stack([world_x, world_y], axis=-1), valid


def polygon_iou(poly_a: Polygon, poly_b: Polygon) -> float:
    if poly_a is None or poly_b is None:
        return 0.0
    if not poly_a.is_valid or not poly_b.is_valid:
        return 0.0
    inter_area = poly_a.intersection(poly_b).area
    union_area = poly_a.union(poly_b).area
    return inter_area / union_area if union_area > 0 else 0.0


# ---------------------------------------------------------------------------
# アノテーションXML(VOC形式)の読み込み
# save_voc_xml で書き出された想定のタグ構成:
#   <annotation><size><width>..</width><height>..</height></size>
#     <object><name>cow001</name><bndbox><xmin>..</xmin>...</bndbox></object>
#     ...
# タグ名が異なる場合はここを実際のXMLに合わせて修正すること。
# ---------------------------------------------------------------------------

def parse_voc_annotation(xml_path: Path) -> Tuple[float, float, List[dict]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    width = float(size.find("width").text)
    height = float(size.find("height").text)

    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        bnd = obj.find("bndbox")
        objects.append({
            "name": name,
            "xmin": float(bnd.find("xmin").text),
            "ymin": float(bnd.find("ymin").text),
            "xmax": float(bnd.find("xmax").text),
            "ymax": float(bnd.find("ymax").text),
        })
    return width, height, objects


# ---------------------------------------------------------------------------
# bbox(アノテーション) -> カメラ振り分け -> World Polygon
# ---------------------------------------------------------------------------

def build_gt_world_polygons(
    objects: List[dict],
    poses: Dict[str, Pose],
    image_width: float,
) -> Dict[str, List[Tuple[str, Polygon]]]:
    tile_w = image_width / NUM_CAMERAS
    per_camera: Dict[str, List[Tuple[str, Polygon]]] = {cid: [] for cid in CAM_IDS}

    for obj in objects:
        center_x = (obj["xmin"] + obj["xmax"]) / 2.0
        cam_idx = int(center_x // tile_w)
        cam_idx = max(0, min(cam_idx, NUM_CAMERAS - 1))
        cam_id = CAM_IDS[cam_idx]

        fx, fy, cx, cy, xi, alpha = get_intrinsics(cam_id)
        corners_full = np.array([
            [obj["xmin"], obj["ymin"]],
            [obj["xmax"], obj["ymin"]],
            [obj["xmax"], obj["ymax"]],
            [obj["xmin"], obj["ymax"]],
        ])
        corners_local = corners_full.copy()
        corners_local[:, 0] -= cam_idx * tile_w

        pose = poses[cam_id]
        world_xy, valid = unproject_to_floor(
            corners_local, pose.R, pose.t, fx, fy, cx, cy, xi, alpha,
            floor_z=BEV_PLANE_HEIGHT_M,
        )
        if not valid.all():
            continue

        poly = Polygon(world_xy.tolist())
        if not poly.is_valid:
            poly = poly.buffer(0)
        per_camera[cam_id].append((obj["name"], poly))

    return per_camera


# ---------------------------------------------------------------------------
# カメラペアごとにHungarian法(1対1最適割当)で対応付けを解く
# bev_matching_with_visualization.py の match_camera_pair と同じロジック。
# IOUの高い箱が複数あっても、割当は最良の1組だけに絞られる点に注意。
# ---------------------------------------------------------------------------

@dataclass
class Assignment:
    cam_i: str
    cam_j: str
    idx_i: int
    idx_j: int
    iou: float
    name_i: str
    name_j: str


def compute_camera_assignments(
    per_camera: Dict[str, List[Tuple[str, Polygon]]],
) -> Tuple[List[Assignment], List[Tuple[str, str, int, int, str]]]:
    """
    戻り値:
      assignments: Hungarianによって選ばれた割当ペア(閾値でのフィルタ前)
      gt_same_pairs: name が一致する全カメラ間ペア(Hungarianの割当に選ばれたかは問わない、
                     "本来同一個体であるはずの組"の全数)
    """
    cam_ids = list(per_camera.keys())
    assignments: List[Assignment] = []
    gt_same_pairs: List[Tuple[str, str, int, int, str]] = []

    for cam_i, cam_j in combinations(cam_ids, 2):
        pairs_i = per_camera[cam_i]
        pairs_j = per_camera[cam_j]
        n_i, n_j = len(pairs_i), len(pairs_j)

        # 正解側: nameが一致する全ペアを列挙(Hungarianの割当とは独立)
        for a, (name_i, _poly_i) in enumerate(pairs_i):
            for b, (name_j, _poly_j) in enumerate(pairs_j):
                if name_i == name_j:
                    gt_same_pairs.append((cam_i, cam_j, a, b, name_i))

        if n_i == 0 or n_j == 0:
            continue

        iou_matrix = np.zeros((n_i, n_j))
        for a, (_name_i, poly_i) in enumerate(pairs_i):
            for b, (_name_j, poly_j) in enumerate(pairs_j):
                iou_matrix[a, b] = polygon_iou(poly_i, poly_j)

        cost_matrix = 1.0 - iou_matrix
        row_idx, col_idx = linear_sum_assignment(cost_matrix)
        for a, b in zip(row_idx, col_idx):
            assignments.append(Assignment(
                cam_i=cam_i, cam_j=cam_j, idx_i=a, idx_j=b,
                iou=iou_matrix[a, b],
                name_i=pairs_i[a][0], name_j=pairs_j[b][0],
            ))

    return assignments, gt_same_pairs


# ---------------------------------------------------------------------------
# Union-Find による推移的な個体統合
# bev_matching_with_visualization.py の union_find_merge と同じロジック。
# Hungarianで割り当てられ、かつIOU>=閾値のペア(accepted_pairs)だけを統合対象にする。
# ---------------------------------------------------------------------------

def union_find_clusters(
    keys: List[Tuple[str, int]],
    accepted_pairs: List[Tuple[Tuple[str, int], Tuple[str, int]]],
) -> Dict[Tuple[str, int], Tuple[str, int]]:
    """戻り値: 各キー(cam_id, idx) -> そのクラスタの代表元(root)"""
    parent: Dict[Tuple[str, int], Tuple[str, int]] = {k: k for k in keys}

    def find(x: Tuple[str, int]) -> Tuple[str, int]:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: Tuple[str, int], y: Tuple[str, int]) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for k1, k2 in accepted_pairs:
        union(k1, k2)

    return {k: find(k) for k in keys}


def main():
    print("floorplan.json 読み込み中...")
    cameras_json = load_floorplan_cameras(FLOORPLAN_JSON_PATH)

    print("カメラパラメータの上書き設定を適用中...")
    cameras_json = apply_camera_overrides(cameras_json)

    print("カメラ姿勢を解決中...")
    poses = resolve_poses(cameras_json, root="Area1", fmt=ROT_FMT)

    xml_path = Path(ANNOTATION_XML_PATH)
    if not xml_path.exists():
        raise FileNotFoundError(f"アノテーションXMLが見つかりません: {xml_path}")
    print(f"評価対象: {xml_path}")

    width, _height, objects = parse_voc_annotation(xml_path)
    per_camera = build_gt_world_polygons(objects, poses, width)
    assignments, gt_same_pairs = compute_camera_assignments(per_camera)

    for cam_id in CAM_IDS:
        print(f"  {cam_id}: {len(per_camera[cam_id])}頭 (アノテーション)")

    total_same_gt = len(gt_same_pairs)
    print(f"\nこのフレームで、複数カメラに跨って映っている個体ペア(正解=同一個体): {total_same_gt}件")
    print(f"Hungarian法による割当ペア総数(閾値フィルタ前、参考値): {len(assignments)}件")

    # 全検出をキー(cam_id, idx)で列挙し、name引きの辞書も作る
    all_keys: List[Tuple[str, int]] = [
        (cam_id, idx) for cam_id, dets in per_camera.items() for idx in range(len(dets))
    ]
    name_lookup: Dict[Tuple[str, int], str] = {
        (cam_id, idx): dets[idx][0]
        for cam_id, dets in per_camera.items()
        for idx in range(len(dets))
    }

    print("\n=== IOU閾値ごとの精度 (Hungarian割当 + Union-Findによるクラスタ統合) ===")
    print(f"{'IOU閾値':>8} | {'統合成功率(recall)':>20} | {'誤統合precision':>16} | {'F1':>8}")
    for thr in IOU_THRESHOLDS:
        # Hungarianで割り当てられ、かつIOU>=閾値のペアだけを統合対象にする
        accepted_pairs = [
            ((a.cam_i, a.idx_i), (a.cam_j, a.idx_j))
            for a in assignments if a.iou >= thr
        ]
        roots = union_find_clusters(all_keys, accepted_pairs)

        # 正解: 本来同一個体のペアが、最終的に同じクラスタに入っていればTP
        # (Camera1-3が直接マッチしていなくても、Camera2経由で推移的に統合されていればTPになる)
        tp = sum(
            1 for (cam_i, cam_j, idx_i, idx_j, _name) in gt_same_pairs
            if roots[(cam_i, idx_i)] == roots[(cam_j, idx_j)]
        )
        fn = total_same_gt - tp

        # 誤統合: 同じクラスタ内にnameが異なる検出が混ざっている組を数える
        # (直接のペアだけでなく、推移的な連鎖によって誤って同じクラスタに入ったケースも含む)
        clusters: Dict[Tuple[str, int], List[Tuple[str, int]]] = defaultdict(list)
        for k in all_keys:
            clusters[roots[k]].append(k)

        fp = 0
        for members in clusters.values():
            if len(members) < 2:
                continue
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    if name_lookup[members[i]] != name_lookup[members[j]]:
                        fp += 1

        merge_rate = tp / (tp + fn) if (tp + fn) > 0 else float("nan")  # 求めたかった「統合されている割合」
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        f1 = (
            2 * precision * merge_rate / (precision + merge_rate)
            if (precision + merge_rate) > 0 else float("nan")
        )

        print(f"{thr:>8.2f} | {merge_rate:>10.3f} ({tp}/{tp+fn}) | {precision:>10.3f} (FP={fp}) | {f1:>8.3f}")


if __name__ == "__main__":
    main()