"""
zone_overlap_filter_he_kai_ver1.py

「牛舎をカメラ間の中央で分割 -> 有効範囲は個体統合の領域を決める」ための検出フィルタ。
(ym.filter_by_curved_zone の置き換え。入力・出力の形は同じ: {cam_id: [dict(world_xy=...), ...]})

  担当ゾーン : 床上の点 p は、いちばん近いカメラ(= カメラ間の垂直二等分線で分割)の担当
  残す検出   : カメラ i の検出(位置 p)は、次のどちらかなら残す
                 (a) p が i の担当ゾーン内            -> 有効範囲の外でも常に残す (ここで FN を出さない)
                 (b) p が他カメラの担当ゾーンで、かつ i の有効範囲の中 -> 統合のために残す (重なりの帯)
               それ以外(他カメラの担当ゾーンで、i の有効範囲の外)は捨てる。担当カメラが見ているので FN にならない。
  効果       : 有効範囲が小さいと、統合の帯が狭くなる (FN ではなく、境界の二重検出に効く)。

配置: research/phase4/he_kai/
使い方: from zone_overlap_filter_he_kai_ver1 import filter_zone_overlap
"""

from typing import Dict, List

import numpy as np


def point_in_polygon(pt, poly) -> bool:
    """ray casting。poly: (M,2)"""
    poly = np.asarray(poly, dtype=np.float64)
    if poly.ndim != 2 or len(poly) < 3:
        return False
    x, y = float(pt[0]), float(pt[1])
    inside = False
    x1, y1 = poly[-1]
    for x2, y2 in poly:
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1:
            inside = not inside
        x1, y1 = x2, y2
    return inside


def zone_owner(pt, cam_xy: Dict[str, np.ndarray]) -> str:
    """床上の点 pt を担当するカメラ(いちばん近いカメラ)"""
    return min(cam_xy, key=lambda c: float(np.linalg.norm(np.asarray(pt, dtype=np.float64) - cam_xy[c])))


def filter_zone_overlap(per_camera: Dict[str, List[dict]], cam_polygons: Dict[str, np.ndarray],
                        cam_xy: Dict[str, np.ndarray], stats: dict = None) -> Dict[str, List[dict]]:
    out = {cid: [] for cid in per_camera}
    for cid, items in per_camera.items():
        poly = cam_polygons.get(cid)
        for it in items:
            p = it["world_xy"]
            if zone_owner(p, cam_xy) == cid:
                out[cid].append(it)
                if stats is not None:
                    stats["own"] = stats.get("own", 0) + 1
            elif poly is not None and point_in_polygon(p, poly):
                out[cid].append(it)
                if stats is not None:
                    stats["overlap"] = stats.get("overlap", 0) + 1
            elif stats is not None:
                stats["dropped"] = stats.get("dropped", 0) + 1
    return out