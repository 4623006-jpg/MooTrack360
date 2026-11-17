"""
he_kai_sigma_model_ver1.py

HE(改)の「画像の半径で連続に変わる σ_px(r)」と、それを使った「カメラ有効範囲」の共通部品。
  - estimate_sigma_he_kai_ver1.py        が σ をデータから推定して JSON に保存する
  - phase3_he_kai_matching_ver2.py       がこの JSON を読んで、マッチングとカメラ有効範囲に使う

σ_px(r): bbox中心の画素のずれの標準偏差 [px]。r は画像中心からの半径 [px]。
   kind = "const"  : 全域で一定
   kind = "linear" : a + b * r/1000
   kind = "bins"   : 半径のビン中心(r_knots)での値(values)を線形補間(外側は端の値で一定)
σ_h(姿勢): 高さの仮定のまわりの、実際の高さのばらつき [m]
μ_h(姿勢): 高さの仮定の平均的なずれ [m]。有効な想定高さ h_eff = h_a + μ_h

カメラ有効範囲:
  画像中心から各方向(N_ANGLES方向)へ半径を増やしながら、その位置の位置誤差
      e = sqrt( λmax( σ_px(r)² J Jᵀ + σ_h² g gᵀ ) )   [m]   (1σ)
  を計算し、 k·e > eps になる直前の半径までをそのカメラの有効範囲にする。
  各方向の端を床に逆投影した点列が、有効範囲の多角形(curved_zone_utils の polygon と同じ形式)。
  J: 画素 -> 床位置の2×2ヤコビアン(HE)。g: 高さが1m違ったときの床位置のずれ。
"""

import json
import os
from typing import Dict, List, Optional

import numpy as np

POSES = ("standing", "lying")


class SigmaModel:
    def __init__(self, entry: dict, name: str = ""):
        self.name = name
        self.kind = entry["kind"]
        self.entry = entry
        self.sigma_h_by_pose = dict(entry["sigma_h"])
        self.mu_h_by_pose = dict(entry["mu_h"])
        self.h_a_by_pose = dict(entry["h_a"])

    def sigma_px(self, r):
        r = np.asarray(r, dtype=np.float64)
        e = self.entry
        if self.kind == "const":
            return np.full(r.shape, float(e["value"]))
        if self.kind == "linear":
            return np.maximum(float(e["a"]) + float(e["b"]) * r / 1000.0, 1.0)
        if self.kind == "bins":
            return np.interp(r, np.asarray(e["r_knots"], dtype=np.float64), np.asarray(e["values"], dtype=np.float64))
        raise ValueError(f"unknown kind: {self.kind}")

    def sigma_h(self, pose: Optional[str]) -> float:
        if pose in self.sigma_h_by_pose:
            return float(self.sigma_h_by_pose[pose])
        return float(max(self.sigma_h_by_pose.values()))      # 姿勢不明は大きい方(保守的)

    def sigma_h_max(self) -> float:
        return float(max(self.sigma_h_by_pose.values()))

    def h_eff(self, pose: str) -> float:
        return float(self.h_a_by_pose[pose] + self.mu_h_by_pose[pose])

    def describe(self) -> str:
        rs = np.array([0, 500, 1000, 1500, 2000, 2500], dtype=np.float64)
        curve = ", ".join(f"r={int(r)}:{s:.1f}" for r, s in zip(rs, self.sigma_px(rs)))
        return (f"モデル {self.name} ({self.kind}) σ_px(r)[px]: {curve} / σ_h={self.sigma_h_by_pose} / "
                f"μ_h={self.mu_h_by_pose} / h_a={self.h_a_by_pose} -> h_eff="
                f"{ {p: round(self.h_eff(p), 3) for p in POSES} }")


SIGMA_JSON_NAME = "sigma_he_kai_ver1.json"


def find_sigma_json(path=None) -> str:
    """σのJSONの場所を返す。渡されたpathが存在すればそれ。無ければ
       環境変数 HE_SIGMA_JSON -> research/idf1_check/bev_calibration_refined/ -> research 以下を検索(最新)
    の順に探す。呼び出し側のOUTPUT_DIR(スクリプトごとに違うことがある)に依存しないようにするため。"""
    import glob
    if path and os.path.exists(str(path)):
        return str(path)
    env = os.environ.get("HE_SIGMA_JSON")
    if env and os.path.exists(env):
        return env
    research = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    cand = os.path.join(research, "idf1_check", "bev_calibration_refined", SIGMA_JSON_NAME)
    if os.path.exists(cand):
        return cand
    hits = glob.glob(os.path.join(research, "**", SIGMA_JSON_NAME), recursive=True)
    if hits:
        return max(hits, key=os.path.getmtime)
    raise FileNotFoundError(
        f"{SIGMA_JSON_NAME} が見つかりません(指定: {path})。先に estimate_sigma_he_kai_ver1.py を実行するか、"
        f"環境変数 HE_SIGMA_JSON にパスを指定してください。")


def load_model(path, name: Optional[str] = None) -> SigmaModel:
    path = find_sigma_json(path)
    print(f"σのJSON: {path}")
    data = json.loads(open(path, encoding="utf-8").read())
    name = name or os.environ.get("HE_SIGMA_MODEL") or data["selected"]
    return SigmaModel(data["models"][name], name)


def _lam_max(sxx, sxy, syy):
    tr = sxx + syy
    det = sxx * syy - sxy * sxy
    return tr / 2.0 + np.sqrt(np.maximum(tr * tr / 4.0 - det, 0.0))


def build_valid_polygons(poses: dict, cam_ids: List[str], get_intrinsics_fn, model: SigmaModel,
                         eps_m: float, k_sigma: float = 1.0, height_m: float = 0.8,
                         n_angles: int = 360, verbose: bool = True) -> Dict[str, np.ndarray]:
    """各カメラの有効範囲の多角形(世界座標)を返す。cam_id -> ndarray(M,2)。
    k_sigma * e(r, 方向) <= eps_m を満たす最も外側までを有効とする(内側から最初に超えた所で打ち切り)。"""
    from Mootrack_core import unproject_to_floor
    try:
        from compute_acceptable_fisheye_radius import MAX_RADIUS_PX, RADIUS_STEP_PX
    except Exception:
        MAX_RADIUS_PX, RADIUS_STEP_PX = 1800, 10

    polygons: Dict[str, np.ndarray] = {}
    ang = 2.0 * np.pi * np.arange(n_angles) / n_angles
    sig_h = model.sigma_h_max()
    for cid in cam_ids:
        p = poses[cid]
        fx, fy, cx, cy, xi, alpha = get_intrinsics_fn(cid)
        cam_xy = np.asarray(p.t[:2], dtype=np.float64)
        H = float(-p.t[2])
        alive = np.ones(n_angles, dtype=bool)
        last_xy = np.zeros((n_angles, 2))
        last_r = np.zeros(n_angles)
        for r in range(int(RADIUS_STEP_PX), int(MAX_RADIUS_PX) + 1, int(RADIUS_STEP_PX)):
            uv = np.stack([cx + r * np.cos(ang), cy + r * np.sin(ang)], axis=1)
            pts = np.concatenate([uv, uv + np.array([1.0, 0.0]), uv + np.array([0.0, 1.0])])
            xy, v = unproject_to_floor(pts, p.R, p.t, fx, fy, cx, cy, xi, alpha, floor_z=-height_m)
            xy0, xy1, xy2 = xy[:n_angles], xy[n_angles:2 * n_angles], xy[2 * n_angles:]
            ok = v[:n_angles] & v[n_angles:2 * n_angles] & v[2 * n_angles:]
            a, b = xy1 - xy0, xy2 - xy0
            jxx, jxy, jyy = a[:, 0] ** 2 + b[:, 0] ** 2, a[:, 0] * a[:, 1] + b[:, 0] * b[:, 1], a[:, 1] ** 2 + b[:, 1] ** 2
            d = xy0 - cam_xy
            rho = np.maximum(np.linalg.norm(d, axis=1), 1e-6)
            u = d / rho[:, None]
            gmag = rho / max(H - height_m, 0.3)
            sp = float(model.sigma_px(r))
            sxx = sp ** 2 * jxx + sig_h ** 2 * (gmag * u[:, 0]) ** 2
            sxy = sp ** 2 * jxy + sig_h ** 2 * gmag ** 2 * u[:, 0] * u[:, 1]
            syy = sp ** 2 * jyy + sig_h ** 2 * (gmag * u[:, 1]) ** 2
            e = np.sqrt(_lam_max(sxx, sxy, syy))
            alive &= ok & (k_sigma * e <= eps_m)
            if not alive.any():
                break
            last_xy[alive] = xy0[alive]
            last_r[alive] = r
        keep = last_r > 0
        polygons[cid] = last_xy[keep]
        if verbose:
            rr = last_r[keep]
            msg = f"平均 {rr.mean():.0f}px (最小 {rr.min():.0f} / 最大 {rr.max():.0f})" if keep.any() else "有効範囲なし"
            print(f"  {cid}: HE(改)有効範囲 (k·e<={eps_m}m, k={k_sigma}) 半径 {msg} -> 多角形点数={int(keep.sum())}")
    return polygons