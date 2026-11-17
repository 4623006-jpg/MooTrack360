"""
range_from_tri_gt_he_kai_ver1.py

三角測量GTから直接測った位置の総合誤差(tri_gt_position_error_he_kai_ver1.py が出す表)で、
カメラの有効範囲(床上の多角形)を作る。マッチングの Σ(最尤推定)とは独立。

  判定(kind="px_eq", 既定): ‖J(r, 方位)‖₂ × e_px(r) <= ε
      ‖J‖₂ : その位置で画素を1px動かしたときの床上のずれの最大値 [m/px] (方位ごとに計算。1px前進差分)
      e_px(r) : 表の「px相当の誤差」(位置誤差の q分位点 / ‖J‖₂ を半径ビンごとに集計、累積最大で単調にしたもの)を半径で線形補間
      従来の find_acceptable_radius (m/px × 画素誤差 <= 許容誤差) と同じ形で、画素誤差の表を
      「YOLOとGTの画素ずれ」から「三角測量GTとの位置の総合誤差」に置き換えたもの。方位ごとに境界を持つ。
  判定(kind="meters"): e_m(r) <= ε   (表の位置誤差[m]を半径だけの関数として使う。カメラ・方位の違いは見ない)

  各方位で、画像の中心から外へ向かって最初に ε を超えた所の手前までを有効とする(build_valid_polygons と同じ)。
  ε の既定は 0.425 m (体幅0.85 mの半分, 従来の ACCEPTABLE_WORLD_ERROR_M と同じ)。

使い方: from range_from_tri_gt_he_kai_ver1 import build_valid_polygons_tri, load_table
配置: research/phase4/he_kai/
"""

import glob
import json
import os
from typing import Dict, List

import numpy as np

TABLE_NAME = "tri_gt_error_table_he_kai_ver1.json"


def find_table(path=None) -> str:
    if path and os.path.exists(str(path)):
        return str(path)
    env = os.environ.get("HE_TRI_TABLE")
    if env and os.path.exists(env):
        return env
    research = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    cand = os.path.join(research, "idf1_check", "bev_calibration_refined", TABLE_NAME)
    if os.path.exists(cand):
        return cand
    hits = glob.glob(os.path.join(research, "**", TABLE_NAME), recursive=True)
    if hits:
        return max(hits, key=os.path.getmtime)
    raise FileNotFoundError(f"{TABLE_NAME} が見つかりません。先に tri_gt_position_error_he_kai_ver1.py を実行するか、"
                            f"環境変数 HE_TRI_TABLE にパスを指定してください。")


def load_table(path=None) -> dict:
    p = find_table(path)
    print(f"三角測量GTの誤差表: {p}")
    return json.loads(open(p, encoding="utf-8").read())


def _interp(r, knots, values):
    """半径 r での値(範囲外は端の値で一定)"""
    return float(np.interp(r, np.asarray(knots, dtype=np.float64), np.asarray(values, dtype=np.float64)))


def build_valid_polygons_tri(poses: dict, cam_ids: List[str], get_intrinsics_fn, table: dict,
                             eps_m: float = 0.425, kind: str = "px_eq", height_m: float = 0.8,
                             n_angles: int = 144, verbose: bool = True) -> Dict[str, np.ndarray]:
    """各カメラの有効範囲の多角形(世界座標)を返す。cam_id -> ndarray(M,2)"""
    from Mootrack_core import unproject_to_floor
    try:
        from compute_acceptable_fisheye_radius import MAX_RADIUS_PX, RADIUS_STEP_PX
    except Exception:
        MAX_RADIUS_PX, RADIUS_STEP_PX = 1800, 10
    if kind not in ("px_eq", "meters"):
        raise ValueError(f"kind は px_eq / meters のどちらか: {kind}")
    knots = table["r_knots"]
    vals = table["e_q_px"] if kind == "px_eq" else table["e_q_m"]

    polygons: Dict[str, np.ndarray] = {}
    ang = 2.0 * np.pi * np.arange(n_angles) / n_angles
    for cid in cam_ids:
        p = poses[cid]
        fx, fy, cx, cy, xi, alpha = get_intrinsics_fn(cid)
        alive = np.ones(n_angles, dtype=bool)
        last_xy = np.zeros((n_angles, 2))
        last_r = np.zeros(n_angles)
        for r in range(int(RADIUS_STEP_PX), int(MAX_RADIUS_PX) + 1, int(RADIUS_STEP_PX)):
            uv = np.stack([cx + r * np.cos(ang), cy + r * np.sin(ang)], axis=1)
            pts = np.concatenate([uv, uv + np.array([1.0, 0.0]), uv + np.array([0.0, 1.0])])
            xy, v = unproject_to_floor(pts, p.R, p.t, fx, fy, cx, cy, xi, alpha, floor_z=-height_m)
            xy0, xy1, xy2 = xy[:n_angles], xy[n_angles:2 * n_angles], xy[2 * n_angles:]
            ok = v[:n_angles] & v[n_angles:2 * n_angles] & v[2 * n_angles:]
            e_r = _interp(r, knots, vals)
            if kind == "px_eq":
                a, b = xy1 - xy0, xy2 - xy0
                jxx, jxy, jyy = (a ** 2).sum(1), (a * b).sum(1), (b ** 2).sum(1)
                tr, det = jxx + jyy, jxx * jyy - jxy ** 2
                s = np.sqrt(tr / 2.0 + np.sqrt(np.maximum(tr * tr / 4.0 - det, 0.0)))
                err = s * e_r
            else:
                err = np.full(n_angles, e_r)
            alive &= ok & (err <= eps_m)
            if not alive.any():
                break
            last_xy[alive] = xy0[alive]
            last_r[alive] = r
        keep = last_r > 0
        polygons[cid] = last_xy[keep]
        if verbose:
            rr = last_r[keep]
            msg = f"平均 {rr.mean():.0f}px (最小 {rr.min():.0f} / 最大 {rr.max():.0f})" if keep.any() else "有効範囲なし"
            print(f"  {cid}: 三角測量GT({kind}, q={table.get('q')}, ε={eps_m}m) 有効範囲 半径 {msg} -> 多角形点数={int(keep.sum())}")
    return polygons