"""
curved_zone_utils_ver1.py

render_bev_frames_yolo_zonepose.py / phase3_bev_multicam_yolo_check_zonepose.py が
使っていた「隣接カメラのワールドx座標の中点 ± OVERLAP_HALF_WIDTH_M」という
直線(x軸に垂直な帯)でのカメラ有効範囲の区切りを、compute_acceptable_fisheye_
radius.pyで求めた「許容できる魚眼カメラの範囲(画像上の半径R_max)」に基づく
曲線(各カメラの画像上の円を、傾いたカメラ姿勢を通して床に逆投影した多角形)に
置き換えるための共通ロジック。

visualize_curved_valid_zones.py(可視化のみ版)のfind_acceptable_radius /
camera_valid_zone_curveと同じ計算を使うが、こちらは生成した多角形を実際の
検出フィルタリング・ゾーン分類にも使えるようにする(point-in-polygon判定を
追加)。

本番パイプライン側(render_bev_frames_yolo_zonepose_ver1.py /
phase3_bev_multicam_yolo_check_zonepose_ver1.py)はこのモジュールをimportして使う。

バージョン管理方針: このロジックに変更を入れる場合は、このファイルを直接
書き換えるのではなく _ver2, _ver3 ... という新しいファイルを作る
(呼び出し側のimport文も合わせて新しいverに向ける)。
"""

import os
import sys
from typing import Dict, List, Tuple

import cv2
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

MOOTRACK_CODE_DIR = os.path.join(THIS_DIR, "..", "calibration")
if MOOTRACK_CODE_DIR not in sys.path:
    sys.path.insert(0, MOOTRACK_CODE_DIR)

from Mootrack_core import unproject_to_floor

# compute_acceptable_fisheye_radius.py側の許容誤差の判定ロジック・設定値を再利用
# (ロジックの二重管理を避けるため。許容比・ピクセル誤差テーブルを変更したい場合は
# そちらのACCEPTABLE_WORLD_ERROR_M/PIXEL_ERROR_BY_RADIUS_BIN_PX等を直接編集する)
from compute_acceptable_fisheye_radius import (
    HEIGHT_M as ZONE_HEIGHT_M,
    ACCEPTABLE_WORLD_ERROR_M,
    pixel_error_for_radius,
    unproject_one,
    RADIUS_STEP_PX,
    MAX_RADIUS_PX,
)

# 曲線境界を何分割でサンプリングするか(多いほど滑らかな多角形になる)
N_ANGLES = 360


def find_acceptable_radius(cam_pose, fx, fy, cx, cy, xi, alpha, height_m: float = ZONE_HEIGHT_M) -> float:
    """compute_acceptable_fisheye_radius.pyと同じ判定ロジックで、画像中心から
    外側に向かって最初にNG(許容誤差超え)になる半径[px]を求める。
    横方向のプラス側・マイナス側それぞれで求め、厳しい方(小さい方)を採用する
    (円として一律に使うための単純化。本来は全方位で異なりうるが、Double Sphere
    モデルは画像中心からの距離のみに依存するため、水平線上の走査で十分近似できる)。"""
    floor_z = -height_m

    def ok_at(r_signed: float) -> bool:
        base_x, base_y = cx + r_signed, cy
        p_base, v_base = unproject_one(cam_pose, fx, fy, cx, cy, xi, alpha, floor_z, base_x, base_y)
        p_next, v_next = unproject_one(cam_pose, fx, fy, cx, cy, xi, alpha, floor_z, base_x + 1.0, base_y)
        if not (v_base and v_next):
            return False
        m_per_px = float(np.linalg.norm(p_next - p_base))
        pixel_error_px, _ = pixel_error_for_radius(abs(r_signed))
        expected_error_m = m_per_px * pixel_error_px
        return expected_error_m <= ACCEPTABLE_WORLD_ERROR_M

    r_pos_limit = float(MAX_RADIUS_PX)
    for r in range(0, MAX_RADIUS_PX + 1, RADIUS_STEP_PX):
        if not ok_at(r):
            r_pos_limit = float(r)
            break

    r_neg_limit = float(MAX_RADIUS_PX)
    for r in range(0, MAX_RADIUS_PX + 1, RADIUS_STEP_PX):
        if not ok_at(-r):
            r_neg_limit = float(r)
            break

    return min(r_pos_limit, r_neg_limit)


def camera_valid_polygon(cam_pose, fx, fy, cx, cy, xi, alpha, r_max_px: float,
                          height_m: float = ZONE_HEIGHT_M, n_angles: int = N_ANGLES) -> np.ndarray:
    """画像上で半径r_max_pxの円を一周サンプリングし、それぞれを床面に逆投影した
    世界座標の点列(閉多角形、無効点は除外)を返す。shape=(M,2)、M<=n_angles。"""
    floor_z = -height_m
    pts_px = np.array([
        [cx + r_max_px * np.cos(2 * np.pi * i / n_angles),
         cy + r_max_px * np.sin(2 * np.pi * i / n_angles)]
        for i in range(n_angles)
    ], dtype=np.float64)
    world_xy, valid = unproject_to_floor(
        pts_px, cam_pose.R, cam_pose.t, fx, fy, cx, cy, xi, alpha, floor_z=floor_z,
    )
    return world_xy[valid]


def build_camera_polygons(
    poses: dict, cam_ids: List[str], get_intrinsics_fn,
    height_m: float = ZONE_HEIGHT_M, n_angles: int = N_ANGLES, verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """各カメラについて許容半径R_maxを求め、そのR_maxの円を床面に逆投影した
    多角形(世界座標)を返す。cam_id -> ndarray(shape=(M,2))。"""
    polygons: Dict[str, np.ndarray] = {}
    for cid in cam_ids:
        p = poses[cid]
        fx, fy, cx, cy, xi, alpha = get_intrinsics_fn(cid)
        r_max = find_acceptable_radius(p, fx, fy, cx, cy, xi, alpha, height_m)
        poly = camera_valid_polygon(p, fx, fy, cx, cy, xi, alpha, r_max, height_m, n_angles)
        polygons[cid] = poly
        if verbose:
            print(f"  {cid}: 許容半径R_max={r_max:.0f}px -> 曲線境界ポリゴン点数={len(poly)}")
    return polygons


def point_in_polygon(xy: np.ndarray, polygon: np.ndarray) -> bool:
    """世界座標の点xyが、polygon(世界座標の点列、閉多角形とみなす)の内側にあるか。
    cv2.pointPolygonTestはfloat32のピクセル列を想定しているが、スケール不変な
    判定(>0/<0のみ使う)なので、メートル単位の座標をそのまま渡しても問題ない。"""
    if len(polygon) < 3:
        return False
    poly_f32 = polygon.astype(np.float32).reshape(-1, 1, 2)
    pt = (float(xy[0]), float(xy[1]))
    result = cv2.pointPolygonTest(poly_f32, pt, measureDist=False)
    return result >= 0


def classify_curved_zone(
    xy: np.ndarray, cam_ids: List[str], cam_polygons: Dict[str, np.ndarray],
) -> str:
    """世界座標の点xyが、どのカメラの曲線境界ポリゴンの内側にあるかを判定する。
    1台のみの内側 -> そのカメラ名(専属範囲)
    2台以上の内側 -> "overlap_curved_{cam_a}_{cam_b}_..." (隣接判定ではなく
                      実際に重なっているポリゴンの組で判定するため、straight-line版の
                      classify_exclusive_zoneより厳密)
    どのカメラにも属さない -> "unknown" (牛舎内だが、どのカメラの信頼範囲にも
                      入らない=許容誤差を超える場所、として扱う)"""
    containing = [cid for cid in cam_ids if point_in_polygon(xy, cam_polygons[cid])]
    if len(containing) == 0:
        return "unknown"
    if len(containing) == 1:
        return containing[0]
    return "overlap_curved_" + "_".join(containing)


def filter_by_curved_zone(
    per_camera: Dict[str, List[dict]], cam_polygons: Dict[str, np.ndarray],
) -> Dict[str, List[dict]]:
    """各カメラの検出について、そのカメラ自身の曲線境界ポリゴンの内側にあるものだけを残す。
    straight-line版のfilter_out_of_range_detections(x座標の範囲だけで判定)の
    置き換え。2D(x,y)の多角形判定になるため、カメラの視野の傾きも反映される。"""
    filtered: Dict[str, List[dict]] = {cid: [] for cid in per_camera}
    for cam_id, items in per_camera.items():
        polygon = cam_polygons.get(cam_id)
        if polygon is None or len(polygon) < 3:
            continue
        for it in items:
            if point_in_polygon(it["world_xy"], polygon):
                filtered[cam_id].append(it)
    return filtered


def get_adjacent_pairs(cam_ids: List[str]) -> List[Tuple[str, str]]:
    """match_zone_pose_constrained/fuse_frame_zone_pose が「どのカメラペア間で
    統合を試みるか」を決めるのに使う隣接ペアのリスト。straight-line版のzones.keys()
    と同じ役割だが、曲線版では境界の値(lo,hi)を持たないため、ペアのリストだけを返す。"""
    return list(zip(cam_ids[:-1], cam_ids[1:]))