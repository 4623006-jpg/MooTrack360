"""
Mootrack_core.py

MooTrack360 の魚眼マルチカメラ牛個体統合パイプラインで共通して使う、
カメラキャリブレーション・幾何変換(Double Sphere逆投影)・アノテーション読み込み・
カメラ間マッチング(Hungarian法)・Union-Findによる推移的統合・評価指標(recall/
precision/F1)の計算をまとめたライブラリ。

各実行スクリプト(evaluate_*.py, *_gridsearch.py, visualize_*.py など)は
このモジュールをimportして使う。フレーム固有のパス設定(ANNOTATION_XML_PATH,
SOURCE_IMAGE_PATH, MODEL_PATHなど)や、パラメータのグリッド、実行フロー(main())は
各スクリプト側に残す。

このファイルを他のスクリプトと同じディレクトリに置いておけば、
    from Mootrack_core import ...
でそのままimportできる。
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import dataclass
from itertools import combinations
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon


# ---------------------------------------------------------------------------
# カメラキャリブレーション設定(現場・フレームによらず共通の値)
# ---------------------------------------------------------------------------

NUM_CAMERAS = 3
CAM_IDS = ["Camera_1", "Camera_2", "Camera_3"]

DS_FX, DS_FY = 708.5, 735.2
DS_CX, DS_CY = 1472.0, 1470.1
DS_XI, DS_ALPHA = 0.006105, 0.48977

CAM_INTRINSICS = {
    "Camera_1": {'fx': 708.5, 'fy': 735.2, 'cx': 1472.0, 'cy': 1470.1, 'xi': -0.142590, 'alpha': 0.562601},
    "Camera_2": {'fx': 708.5, 'fy': 735.2, 'cx': 1472.0, 'cy': 1470.1, 'xi': -0.093895, 'alpha': 0.517550},
    "Camera_3": {'fx': 708.5, 'fy': 735.2, 'cx': 1472.0, 'cy': 1470.1, 'xi': 0.006105, 'alpha': 0.461274},
}

CAMERA_PARAM_OVERRIDES = {
    "Area1->Camera_1": [7.652, 6.2073, -5.1564, 0.0422, -2.3987, -1.7193],
    "Camera_1->Camera_2": [13.4939, 0.4865, -0.2389, -7.1072, 1.2858, -0.174],
    "Camera_2->Camera_3": [13.5157, -0.2291, 0.1482, 11.0752, -0.0369, -0.322],
}

ROT_FMT = "euler_zyx_deg"


def get_intrinsics(cam_id: str) -> Tuple[float, float, float, float, float, float]:
    """カメラIDに対応するDouble Sphereの内部パラメータ(fx,fy,cx,cy,xi,alpha)を返す。
    CAM_INTRINSICSに個別設定があればそちらを、無ければ共通デフォルト値を返す。"""
    if cam_id in CAM_INTRINSICS:
        p = CAM_INTRINSICS[cam_id]
        return p["fx"], p["fy"], p["cx"], p["cy"], p["xi"], p["alpha"]
    return DS_FX, DS_FY, DS_CX, DS_CY, DS_XI, DS_ALPHA


# ---------------------------------------------------------------------------
# データ構造
# ---------------------------------------------------------------------------

@dataclass
class Pose:
    R: np.ndarray
    t: np.ndarray


@dataclass
class Region:
    name: str
    type: str
    polygon_m: np.ndarray


@dataclass
class Assignment:
    """Polygon IOUベースのカメラ間割当結果(evaluate_gt_iou_matching.py系で使用)"""
    cam_i: str
    cam_j: str
    idx_i: int
    idx_j: int
    iou: float
    name_i: str
    name_j: str


@dataclass
class PointAssignment:
    """1点+距離ベースのカメラ間割当結果(evaluate_*_distance_matching.py系で使用)"""
    cam_i: str
    cam_j: str
    idx_i: int
    idx_j: int
    dist: float
    name_i: Optional[str]
    name_j: Optional[str]


# ---------------------------------------------------------------------------
# floorplan.json 読み込み / カメラ姿勢の解決
# ---------------------------------------------------------------------------

def load_floorplan(path: str):
    """semantic_regions と cameras の両方を含むフル読み込み(BEV描画用)。
    戻り値: (regions, cameras_json, real_w, real_h)"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    regions = []
    for r in data.get("semantic_regions", []):
        poly = np.array(r["polygon_world_meters"], dtype=np.float64)
        regions.append(Region(r["name"], r.get("type", ""), poly))
    real_w = data["metadata"]["real_world_dimensions"]["width"]
    real_h = data["metadata"]["real_world_dimensions"]["height"]
    cameras_json = data.get("cameras", [])
    return regions, cameras_json, real_w, real_h


def load_floorplan_cameras(path: str) -> list:
    """camerasだけを読み込む軽量版(評価スクリプト用。regionsの描画が不要な場合)"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data.get("cameras", [])


def apply_camera_overrides(
    cameras_json: list,
    overrides: Dict[str, list] = CAMERA_PARAM_OVERRIDES,
) -> list:
    """overridesに指定されたカメラのparamsを上書きしたコピーを返す。
    元のcameras_json(floorplan.jsonから読み込んだもの)は変更しない。"""
    result = []
    for entry in cameras_json:
        key = f"{entry['parent']}->{entry['child']}"
        new_entry = dict(entry)
        if key in overrides:
            new_entry["params"] = list(overrides[key])
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


def build_R(a, b, c, fmt: str = ROT_FMT) -> np.ndarray:
    if "_deg" in fmt:
        a, b, c = np.radians([a, b, c])
        base_fmt = fmt.replace("_deg", "")
    else:
        base_fmt = fmt
    if base_fmt == "euler_zyx":
        return _rz(a) @ _ry(b) @ _rx(c)
    raise ValueError(f"このモジュールは euler_zyx(_deg) のみサポート: {fmt}")


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


# ---------------------------------------------------------------------------
# Double Sphere 逆投影
# ---------------------------------------------------------------------------

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


def unproject_to_floor(pts_px, R, t, fx, fy, cx, cy, xi, alpha, floor_z: Union[float, np.ndarray] = 0.0):
    """
    floor_z はスカラーでも、pts_px と同じ長さの配列(検出ごとに異なる高さ)でもよい。
    """
    dirs_cam, valid = unproject_ds(pts_px, fx, fy, cx, cy, xi, alpha)
    dirs_world = (R @ dirs_cam.T).T

    floor_z_arr = np.asarray(floor_z, dtype=np.float64)
    if floor_z_arr.ndim == 0:
        floor_z_arr = np.full(len(pts_px), float(floor_z_arr))

    dz = dirs_world[:, 2]
    t_safe = np.abs(dz) > 1e-9
    tt = np.where(t_safe, (floor_z_arr - t[2]) / dz, 0.0)
    valid = valid & t_safe & (tt > 0)

    world_x = np.where(valid, t[0] + tt * dirs_world[:, 0], 0.0)
    world_y = np.where(valid, t[1] + tt * dirs_world[:, 1], 0.0)
    return np.stack([world_x, world_y], axis=-1), valid


# ---------------------------------------------------------------------------
# アノテーションXML(VOC形式)の読み込み
# 想定タグ構成:
#   <annotation><size><width>..</width><height>..</height></size>
#     <object><name>cow001</name><pose>standing|lying</pose>
#       <bndbox><xmin>..</xmin>...</bndbox></object>
#     ...
# タグ名が異なる場合はここを実際のXMLに合わせて修正すること。
# ---------------------------------------------------------------------------

def parse_voc_annotation(xml_path) -> Tuple[float, float, List[dict]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    width = float(size.find("width").text)
    height = float(size.find("height").text)

    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        bnd = obj.find("bndbox")
        pose_tag = obj.find("pose")
        pose = pose_tag.text.strip().lower() if pose_tag is not None and pose_tag.text else None
        objects.append({
            "name": name,
            "pose": pose,  # "standing" / "lying" (未設定の場合はNone)
            "xmin": float(bnd.find("xmin").text),
            "ymin": float(bnd.find("ymin").text),
            "xmax": float(bnd.find("xmax").text),
            "ymax": float(bnd.find("ymax").text),
        })
    return width, height, objects


# ---------------------------------------------------------------------------
# Polygon IOU (evaluate_gt_iou_matching.py / bev_matching_with_visualization.py 系)
# ---------------------------------------------------------------------------

def polygon_iou(poly_a: Polygon, poly_b: Polygon) -> float:
    if poly_a is None or poly_b is None:
        return 0.0
    if not poly_a.is_valid or not poly_b.is_valid:
        return 0.0
    inter_area = poly_a.intersection(poly_b).area
    union_area = poly_a.union(poly_b).area
    return inter_area / union_area if union_area > 0 else 0.0


# ---------------------------------------------------------------------------
# カメラペアごとのHungarian法(1対1最適割当)
# ---------------------------------------------------------------------------

def compute_camera_assignments_by_iou(
    per_camera: Dict[str, List[Tuple[str, Polygon]]],
) -> Tuple[List[Assignment], List[Tuple[str, str, int, int, str]]]:
    """Polygon IOUをコスト(1-IOU)としてHungarian割当を解く。
    戻り値: (assignments, gt_same_pairs)"""
    cam_ids = list(per_camera.keys())
    assignments: List[Assignment] = []
    gt_same_pairs: List[Tuple[str, str, int, int, str]] = []

    for cam_i, cam_j in combinations(cam_ids, 2):
        pairs_i = per_camera[cam_i]
        pairs_j = per_camera[cam_j]
        n_i, n_j = len(pairs_i), len(pairs_j)

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


def compute_camera_assignments_by_distance(
    per_camera: Dict[str, List[Tuple[Optional[str], np.ndarray]]],
) -> Tuple[List[PointAssignment], List[Tuple[str, str, int, int, str]]]:
    """2点間のユークリッド距離をコストとしてHungarian割当を解く(反転不要)。
    nameがNone(YOLO検出でGT対応が取れなかった等)のペアはgt_same_pairsに含めない。
    戻り値: (assignments, gt_same_pairs)"""
    cam_ids = list(per_camera.keys())
    assignments: List[PointAssignment] = []
    gt_same_pairs: List[Tuple[str, str, int, int, str]] = []

    for cam_i, cam_j in combinations(cam_ids, 2):
        pairs_i = per_camera[cam_i]
        pairs_j = per_camera[cam_j]
        n_i, n_j = len(pairs_i), len(pairs_j)

        for a, (name_i, _pt_i) in enumerate(pairs_i):
            for b, (name_j, _pt_j) in enumerate(pairs_j):
                if name_i is not None and name_j is not None and name_i == name_j:
                    gt_same_pairs.append((cam_i, cam_j, a, b, name_i))

        if n_i == 0 or n_j == 0:
            continue

        dist_matrix = np.zeros((n_i, n_j))
        for a, (_name_i, pt_i) in enumerate(pairs_i):
            for b, (_name_j, pt_j) in enumerate(pairs_j):
                dist_matrix[a, b] = float(np.linalg.norm(pt_i - pt_j))

        cost_matrix = dist_matrix
        row_idx, col_idx = linear_sum_assignment(cost_matrix)
        for a, b in zip(row_idx, col_idx):
            assignments.append(PointAssignment(
                cam_i=cam_i, cam_j=cam_j, idx_i=a, idx_j=b,
                dist=dist_matrix[a, b],
                name_i=pairs_i[a][0], name_j=pairs_j[b][0],
            ))

    return assignments, gt_same_pairs


# ---------------------------------------------------------------------------
# Union-Find による推移的な個体統合
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


# ---------------------------------------------------------------------------
# キー一覧 / name引き辞書の組み立て(各評価スクリプトで繰り返し出てくる処理)
# ---------------------------------------------------------------------------

def build_keys_and_name_lookup(
    per_camera: Dict[str, List[Tuple[Optional[str], object]]],
) -> Tuple[List[Tuple[str, int]], Dict[Tuple[str, int], Optional[str]]]:
    all_keys: List[Tuple[str, int]] = [
        (cam_id, idx) for cam_id, items in per_camera.items() for idx in range(len(items))
    ]
    name_lookup: Dict[Tuple[str, int], Optional[str]] = {
        (cam_id, idx): items[idx][0]
        for cam_id, items in per_camera.items()
        for idx in range(len(items))
    }
    return all_keys, name_lookup


# ---------------------------------------------------------------------------
# 評価指標(recall/precision/F1)
# ---------------------------------------------------------------------------

def evaluate_merge_accuracy(
    all_keys: List[Tuple[str, int]],
    gt_same_pairs: List[Tuple[str, str, int, int, str]],
    accepted_pairs: List[Tuple[Tuple[str, int], Tuple[str, int]]],
    name_lookup: Dict[Tuple[str, int], Optional[str]],
) -> dict:
    """
    Union-Findで最終的な個体クラスタを作り、tp/fn/fp/recall/precision/F1を計算する。

    - tp: 本来同一個体(gt_same_pairs)のうち、最終的に同じクラスタに入った組
    - fn: 本来同一個体だが、別クラスタのままだった組
    - fp: 同じクラスタ内でnameが異なる(本来別個体)組。直接のペアだけでなく、
          推移的な連鎖によって誤って同じクラスタに入ったケースも数える。
          nameがNone(GT対応不明)な要素が絡む組はFP集計から除外する。
    """
    roots = union_find_clusters(all_keys, accepted_pairs)

    total_same_gt = len(gt_same_pairs)
    tp = sum(
        1 for (cam_i, cam_j, idx_i, idx_j, _name) in gt_same_pairs
        if roots[(cam_i, idx_i)] == roots[(cam_j, idx_j)]
    )
    fn = total_same_gt - tp

    clusters: Dict[Tuple[str, int], List[Tuple[str, int]]] = defaultdict(list)
    for k in all_keys:
        clusters[roots[k]].append(k)

    fp = 0
    for members in clusters.values():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                name_a = name_lookup[members[i]]
                name_b = name_lookup[members[j]]
                if name_a is None or name_b is None:
                    continue
                if name_a != name_b:
                    fp += 1

    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else float("nan")
    )

    return {
        "tp": tp, "fn": fn, "fp": fp, "total_same_gt": total_same_gt,
        "recall": recall, "precision": precision, "f1": f1,
        "roots": roots,
    }