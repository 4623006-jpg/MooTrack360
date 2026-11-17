"""
phase3_he_kai_matching_ver1.py

HE(改)の手順2〜4: カメラ間マッチングの距離を、ユークリッド距離から、位置の不確かさを入れた
マハラノビス距離に置き換えて、既存のphase3評価(YOLO -> 統合 -> 追跡 -> IDF1/MOTA/MOTP/HOTA)で比べる。

  各検出の位置の不確かさ: Σ = σ_px² · J Jᵀ + σ_h(姿勢)² · g gᵀ
      J : bbox中心の画素 -> 床位置 のヤコビアン [m/px]  (想定高さh_aで投影)
      g : 高さが1m違ったときの床位置のずれ  = ρ/(H-h_a) · (カメラから見た向きの単位ベクトル)
  マッチング(カメラペアごと): コスト = d² = (P_i - P_j)ᵀ (Σ_i + Σ_j)⁻¹ (P_i - P_j)
      d² <= GATE_CHI2 のペアだけ採用 (Hungarian)。姿勢クラス制約は既存どおり。
      暴走防止に、ユークリッド距離が MAX_EUCLID_M を超えるペアは採用しない。
  融合位置: 逆共分散で重み付けした平均 (不確かさの小さいカメラの位置を重く扱う)。
  比較: HE_MODE=euclid (既存と同じ) / mahalanobis を、同じYOLO検出で順に実行して、結果を並べる。
        YOLO推論は1回だけ行い、キャッシュして2回目に使い回す。

既存のphase3のスクリプト(YOLO_MODULE)の fuse_frame_zone_pose を差し替えるだけで、他は変更しない。
出力: <phase3のOUTPUT_DIR>/he_kai_eval/{euclid,mahalanobis}/ に各モードのCSV。

配置: research/phase4/he_kai/ など (YOLO_MODULEが research/phase3_multi_camera/ にあること)
使い方: python phase3_he_kai_matching_ver1.py           # 両方のモードを実行して比較
        python phase3_he_kai_matching_ver1.py mahalanobis   # 片方だけ
校正は Mootrack_core.py の現在値(MOOTRACK_CALIB で切り替え可)。
"""

import importlib
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# 配置: research/phase4/he_kai/(このファイル) など。research 直下に他のフォルダ
RESEARCH_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
PHASE4_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
MOOTRACK_CODE_DIR = (os.path.abspath(os.path.join(RESEARCH_DIR, "calibration")))
for d in (RESEARCH_DIR, PHASE4_DIR, MOOTRACK_CODE_DIR,
          os.path.join(RESEARCH_DIR, "analyze"), os.path.join(RESEARCH_DIR, "annotate"),
          os.path.join(RESEARCH_DIR, "phase3_multi_camera"), THIS_DIR):
    if os.path.isdir(d) and d not in sys.path:
        sys.path.insert(0, d)

# 比べるphase3のスクリプト(新しい順に試す)
YOLO_MODULE_CANDIDATES = ["phase3_bev_multicam_yolo_check_zonepose_ver6", "phase3_bev_multicam_yolo_check_zonepose_ver3"]

# 不確かさモデル(analyze_projection_error_vs_tri_gt_ver3.py の「カメラ間の食い違いから推定」の値)
SIGMA_PX = 16.2
SIGMA_H = {"lying": 0.27, "standing": 0.455, "unknown": 0.40}
GATE_CHI2 = 9.21          # χ²(2) の99%点。5.99 なら95%点
MAX_EUCLID_M = 6.0        # マハラノビスで通っても、これを超えるペアは採用しない
EPS_COV = 1e-6

_POSES = {}
_YM = None


def setup_poses(poses):
    _POSES.clear()
    _POSES.update(poses)


def det_covariance(cam_id, item, height_for):
    """検出の位置の不確かさ Σ (2×2) を返す"""
    from Mootrack_core import get_intrinsics, unproject_to_floor
    pose = item.get("pose")
    h_a = height_for(pose)
    x1, y1, x2, y2 = item["bbox_local"]
    uv = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    cp = _POSES[cam_id]
    pts = np.array([uv, [uv[0] + 1.0, uv[1]], [uv[0], uv[1] + 1.0]], dtype=np.float64)
    xy, ok = unproject_to_floor(pts, cp.R, cp.t, *get_intrinsics(cam_id), floor_z=-h_a)
    if not ok.all():
        return None
    J = np.stack([xy[1] - xy[0], xy[2] - xy[0]], axis=1)
    cam_xy = np.asarray(cp.t[:2], dtype=np.float64)
    P = np.asarray(item["world_xy"], dtype=np.float64)
    rho = float(np.linalg.norm(P - cam_xy))
    if rho < 1e-6:
        return None
    u = (P - cam_xy) / rho
    g = (rho / max(float(-cp.t[2]) - h_a, 0.3)) * u
    sig_h = SIGMA_H.get(pose if pose in SIGMA_H else "unknown")
    return SIGMA_PX ** 2 * (J @ J.T) + sig_h ** 2 * np.outer(g, g) + EPS_COV * np.eye(2)


def match_mahalanobis(per_camera, adjacent_pairs, covs, use_pose_constraint=True):
    accepted, n_pose_rejected = [], 0
    for ci, cj in adjacent_pairs:
        A, B = per_camera[ci], per_camera[cj]
        if not A or not B:
            continue
        cost = np.full((len(A), len(B)), 1e6)
        for a, ia in enumerate(A):
            for b, ib in enumerate(B):
                d = np.asarray(ia["world_xy"]) - np.asarray(ib["world_xy"])
                euc = float(np.linalg.norm(d))
                mism = ia["pose"] is not None and ib["pose"] is not None and ia["pose"] != ib["pose"]
                Sa, Sb = covs[(ci, a)], covs[(cj, b)]
                if Sa is None or Sb is None or euc > MAX_EUCLID_M:
                    continue
                m2 = float(d @ np.linalg.solve(Sa + Sb, d))
                if m2 > GATE_CHI2:
                    continue
                if use_pose_constraint and mism:
                    n_pose_rejected += 1
                    continue
                cost[a, b] = m2
        r, c = linear_sum_assignment(cost)
        for a, b in zip(r, c):
            if cost[a, b] < 1e5:
                accepted.append(((ci, a), (cj, b)))
    return accepted, n_pose_rejected


def fuse_mahalanobis(per_camera_filtered, adjacent_pairs, dist_thr=None, use_pose_constraint=True, *args, **kwargs):
    """ym.fuse_frame_zone_pose の置き換え。戻り値は (fused, n_pose_rejected)"""
    from Mootrack_core import union_find_clusters
    height_for = _YM.height_for
    covs = {}
    for cid, items in per_camera_filtered.items():
        for idx, it in enumerate(items):
            covs[(cid, idx)] = det_covariance(cid, it, height_for)
    accepted, n_rej = match_mahalanobis(per_camera_filtered, adjacent_pairs, covs, use_pose_constraint)
    keys = [(cid, idx) for cid, items in per_camera_filtered.items() for idx in range(len(items))]
    if not keys:
        return [], n_rej
    roots = union_find_clusters(keys, accepted)
    clusters = defaultdict(list)
    for k in keys:
        clusters[roots[k]].append(k)
    fused = []
    for members in clusters.values():
        pts = np.array([per_camera_filtered[c][i]["world_xy"] for c, i in members], dtype=np.float64)
        Ws = [np.linalg.inv(covs[m]) if covs[m] is not None else None for m in members]
        if len(members) > 1 and all(W is not None for W in Ws):
            Wsum = sum(Ws)
            xy = np.linalg.solve(Wsum, sum(W @ p for W, p in zip(Ws, pts)))
        else:
            xy = pts.mean(axis=0)
        fused.append({"world_xy": xy, "n_cams": len(members)})
    return fused, n_rej


def load_yolo_module():
    last = None
    for name in YOLO_MODULE_CANDIDATES:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as e:
            last = e
    raise last


def run_mode(ym, mode, out_root, cache):
    """mode: euclid / mahalanobis。ym.main() をそのまま呼ぶ(fuse_frame_zone_pose だけ差し替え)"""
    import pandas as pd
    orig_fuse = ym.fuse_frame_zone_pose
    orig_yolo = ym.yolo_world_points_per_camera_with_pose
    orig_out = ym.OUTPUT_DIR
    ym.OUTPUT_DIR = Path(out_root) / mode
    ym.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def cached_yolo(model, image_path, poses):
        key = str(image_path)
        if key not in cache:
            cache[key] = orig_yolo(model, image_path, poses)
        # 浅いコピー(下流がリストを書き換えても、次のモードに影響させない)
        return {cid: [dict(it) for it in items] for cid, items in cache[key].items()}

    ym.yolo_world_points_per_camera_with_pose = cached_yolo
    if mode == "mahalanobis":
        ym.fuse_frame_zone_pose = fuse_mahalanobis
    try:
        print(f"\n{'#' * 70}\n# モード: {mode}\n{'#' * 70}")
        ym.main()
    finally:
        ym.fuse_frame_zone_pose = orig_fuse
        ym.yolo_world_points_per_camera_with_pose = orig_yolo
        out_dir = ym.OUTPUT_DIR
        ym.OUTPUT_DIR = orig_out
    csvs = sorted(Path(out_dir).glob("*_summary.csv"))
    return pd.read_csv(csvs[-1], index_col=0).iloc[0].to_dict() if csvs else {}


def main():
    modes = sys.argv[1:] or ["euclid", "mahalanobis"]
    global _YM
    ym = _YM = load_yolo_module()
    print(f"phase3モジュール: {ym.__name__} / SIGMA_PX={SIGMA_PX} SIGMA_H={SIGMA_H} GATE_CHI2={GATE_CHI2} MAX_EUCLID_M={MAX_EUCLID_M}")
    from Mootrack_core import ROT_FMT, apply_camera_overrides, load_floorplan_cameras, resolve_poses
    setup_poses(resolve_poses(apply_camera_overrides(load_floorplan_cameras(ym.FLOORPLAN_JSON_PATH)), root="Area1", fmt=ROT_FMT))
    out_root = Path(ym.OUTPUT_DIR) / "he_kai_eval"
    cache, results = {}, {}
    for m in modes:
        results[m] = run_mode(ym, m, out_root, cache)

    keys = ["idf1", "idp", "idr", "mota", "motp", "num_switches", "num_objects", "num_matches", "num_false_positives", "num_misses"]
    print(f"\n{'=' * 70}\n■ 比較 (同じYOLO検出・同じ校正。マッチングの距離だけが違う)\n{'=' * 70}")
    print(f"{'指標':<22}" + "".join(f"{m:>16}" for m in results))
    for k in keys:
        row = f"{k:<22}"
        for m in results:
            v = results[m].get(k)
            row += f"{'-':>16}" if v is None else (f"{v:>16.4f}" if isinstance(v, float) and abs(v) < 10 else f"{v:>16.0f}")
        print(row)
    print("\n※ motp は小さいほど良い(距離の2乗)。num_false_positives / num_misses / num_switches は小さいほど良い。")


if __name__ == "__main__":
    main()