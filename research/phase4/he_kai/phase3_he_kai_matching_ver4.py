"""
phase3_he_kai_matching_ver4.py

ver4 (ver3 からの追加): 有効範囲の使い方を変えたモード (zone_overlap_filter_he_kai_ver1.py)
  従来(ver3まで): 有効範囲の外の検出は捨てる -> 範囲が小さいと FN が増える
  *_zone モード : 牛舎をカメラ間の中央で分割し、各カメラは自分の担当ゾーンの検出を常に残す。
                  他カメラの担当ゾーンにはみ出した検出は、自分の有効範囲の中だけ残す(統合の帯)。
                  -> 有効範囲は「どこで個体統合するか」だけを決める。
    maha_h_tri_zone   : 三角測量GTの有効範囲 (HE_TRI_TABLE / HE_TRI_EPS_M / HE_TRI_KIND)
    maha_h_valid_zone : HE(改)の有効範囲 (HE_EPS_M / HE_K_SIGMA)
    maha_h_orig_zone  : 従来の円の有効範囲 (基準)

ver3 (ver2 からの追加):
  maha_h_tri モード: マハラノビス + h_eff + 「三角測量GTで直接測った位置の総合誤差」による有効範囲。
     範囲の作り方は range_from_tri_gt_he_kai_ver1.py (tri_gt_position_error_he_kai_ver1.py が出す表を使う)。
     マッチングの Σ は今まで通り(σのJSON, 最尤推定)。有効範囲の判定だけが違う。
     環境変数: HE_TRI_TABLE (表のJSON) / HE_TRI_EPS_M (既定 0.425 = 体幅の半分) / HE_TRI_KIND (px_eq か meters, 既定 px_eq)
  HE_EPS_M の既定を 1.4 にした(ver2 を手元で書き換えていた値に合わせた)。
  出力先は he_kai_eval_ver4/。

ver1 からの変更点:
  1. σ_px を定数ではなく、画像の半径 r で連続に変わる σ_px(r) にした
     (estimate_sigma_he_kai_ver1.py が推定した sigma_he_kai_ver1.json を読む)
  2. 想定高さを、推定した有効高さ h_eff = h_a + μ_h にできる(σ_h はそのまわりのばらつきになる)
  3. カメラの有効範囲(曲線ゾーン)を、HE(改)の位置誤差で決められる:
        k · sqrt(λmax(σ_px(r)² J Jᵀ + σ_h² g gᵀ)) <= eps_m   を満たす範囲
     (現行の find_acceptable_radius の「HE × 画素誤差(表) <= 許容誤差」を、高さ項と連続な σ_px(r) に置き換えたもの)

  各検出の位置の不確かさ: Σ = σ_px(r)² · J Jᵀ + σ_h(姿勢)² · g gᵀ
  マッチング(カメラペアごと): d² = (P_i - P_j)ᵀ (Σ_i + Σ_j)⁻¹ (P_i - P_j) <= GATE_CHI2 のペアだけ採用 (Hungarian)
  融合位置: 逆共分散で重み付けした平均

比較するモード(同じYOLO検出・同じ校正。何を変えたかだけが違う):
  euclid         : 既存(ユークリッド距離、想定高さは phase3 の値、曲線ゾーンは既存)
  euclid_h       : euclid + 想定高さを h_eff にする                     <- 高さ変更だけの効果
  maha_h         : マハラノビス + h_eff                                  <- マッチングの効果
  maha_h_valid   : マハラノビス + h_eff + HE(改)の有効範囲              <- 有効範囲の効果
  maha           : マハラノビス + 想定高さは phase3 の値のまま (参考。σ_hは h_eff まわりの値なので厳密には不整合)

環境変数:
  HE_SIGMA_JSON  : σのJSON (既定 <OUTPUT_DIR>/bev_calibration_refined/sigma_he_kai_ver1.json)
  HE_SIGMA_MODEL : JSON内のモデル名 (既定は JSON の selected)
  HE_GATE_CHI2   : χ²(2)のしきい値 (既定 9.21 = 99%点。5.99 なら95%点)
  HE_EPS_M       : 有効範囲の許容位置誤差 [m] (既定 1.0)
  HE_K_SIGMA     : 有効範囲で e に掛ける倍率 (既定 1.0 = 1σ。3.03 なら99%の範囲)
  HE_MAX_EUCLID_M: マハラノビスで通っても、これを超えるペアは採用しない (既定 6.0)
  HE_POSE_CONSTRAINT: 姿勢別の統合制約 1=ON / 0=OFF (未指定ならphase3モジュールの USE_POSE_CONSTRAINT)

使い方: python phase3_he_kai_matching_ver4.py                   # 4モードを実行して比較
        python phase3_he_kai_matching_ver4.py euclid maha_h_valid   # 指定したモードだけ
出力: <phase3のOUTPUT_DIR>/he_kai_eval_ver4/{モード名}/ に各モードのCSV。
YOLO推論は「想定高さの組」ごとに1回だけ行ってキャッシュする(想定高さが変わると world_xy が変わるため)。
"""

import importlib
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# 配置: research/phase4/he_kai/(このファイル) など(YOLO_MODULEが research/phase3_multi_camera/ にあること)
RESEARCH_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
PHASE4_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
MOOTRACK_CODE_DIR = (os.path.abspath(os.path.join(RESEARCH_DIR, "calibration")))
for d in (RESEARCH_DIR, PHASE4_DIR, os.path.join(PHASE4_DIR, "calibration"),
          os.path.join(PHASE4_DIR, "calibration_try"), os.path.join(PHASE4_DIR, "he_kai"),
          MOOTRACK_CODE_DIR,
          os.path.join(RESEARCH_DIR, "analyze"), os.path.join(RESEARCH_DIR, "annotate"),
          os.path.join(RESEARCH_DIR, "phase3_multi_camera"), THIS_DIR):
    if os.path.isdir(d) and d not in sys.path:
        sys.path.insert(0, d)

from he_kai_sigma_model_ver1 import POSES, build_valid_polygons, load_model

YOLO_MODULE_CANDIDATES = ["phase3_bev_multicam_yolo_check_zonepose_ver6", "phase3_bev_multicam_yolo_check_zonepose_ver3",
                          "phase3_bev_multicam_yolo_check_zonepose_ver1"]

GATE_CHI2 = float(os.environ.get("HE_GATE_CHI2", 9.21))
EPS_VALID_M = float(os.environ.get("HE_EPS_M", 1.4))
K_SIGMA = float(os.environ.get("HE_K_SIGMA", 1.0))
MAX_EUCLID_M = float(os.environ.get("HE_MAX_EUCLID_M", 6.0))
TRI_EPS_M = float(os.environ.get("HE_TRI_EPS_M", 0.425))        # maha_h_tri の許容位置誤差 [m] (体幅0.85 mの半分)
TRI_KIND = os.environ.get("HE_TRI_KIND", "px_eq")               # px_eq / meters
EPS_COV = 1e-6

# モード -> (マッチング, 想定高さ, 有効範囲)
MODES = {
    "euclid":       dict(match="euclid", heights="orig", polygons="orig"),
    "euclid_h":     dict(match="euclid", heights="fit", polygons="orig"),
    "maha":         dict(match="maha", heights="orig", polygons="orig"),
    "maha_h":       dict(match="maha", heights="fit", polygons="orig"),
    "maha_h_valid": dict(match="maha", heights="fit", polygons="he_kai"),
    "maha_h_tri":   dict(match="maha", heights="fit", polygons="tri_gt"),
    "maha_h_tri_zone":   dict(match="maha", heights="fit", polygons="tri_gt", zone="overlap"),
    "maha_h_valid_zone": dict(match="maha", heights="fit", polygons="he_kai", zone="overlap"),
    "maha_h_orig_zone":  dict(match="maha", heights="fit", polygons="orig", zone="overlap"),
}
DEFAULT_MODES = ["euclid", "euclid_h", "maha_h", "maha_h_valid"]      # maha_h_tri は引数で指定する

_POSES = {}
_YM = None
_MODEL = None
_TRI_TABLE = None


def build_tri_polygons(poses, cam_ids, get_intrinsics_fn, height_m=None, n_angles=144, verbose=True):
    """maha_h_tri の有効範囲(三角測量GTの誤差表から)。動画のスクリプトからも使う"""
    global _TRI_TABLE
    from range_from_tri_gt_he_kai_ver1 import build_valid_polygons_tri, load_table
    if _TRI_TABLE is None:
        _TRI_TABLE = load_table()
    return build_valid_polygons_tri(poses, cam_ids, get_intrinsics_fn, _TRI_TABLE, eps_m=TRI_EPS_M, kind=TRI_KIND,
                                    height_m=height_m if height_m is not None else 0.8, n_angles=n_angles, verbose=verbose)


def setup_poses(poses):
    _POSES.clear()
    _POSES.update(poses)


def det_covariance(cam_id, item, height_for):
    """検出の位置の不確かさ Σ (2×2) を返す。σ_px は bbox中心の画像中心からの半径 r の関数"""
    from Mootrack_core import get_intrinsics, unproject_to_floor
    pose = item.get("pose")
    h_a = height_for(pose)
    x1, y1, x2, y2 = item["bbox_local"]
    uv = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    cp = _POSES[cam_id]
    fx, fy, cx, cy, xi, alpha = get_intrinsics(cam_id)
    pts = np.array([uv, [uv[0] + 1.0, uv[1]], [uv[0], uv[1] + 1.0]], dtype=np.float64)
    xy, ok = unproject_to_floor(pts, cp.R, cp.t, fx, fy, cx, cy, xi, alpha, floor_z=-h_a)
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
    r = float(np.hypot(uv[0] - cx, uv[1] - cy))
    sp = float(_MODEL.sigma_px(r))
    sh = _MODEL.sigma_h(pose)
    return sp ** 2 * (J @ J.T) + sh ** 2 * np.outer(g, g) + EPS_COV * np.eye(2)


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


def run_mode(ym, mode, cfg, out_root, cache, orig_heights):
    """ym.main() をそのまま呼ぶ。差し替えるのは 融合関数・想定高さ・有効範囲の作り方 だけ"""
    import pandas as pd
    orig_fuse, orig_yolo, orig_out = ym.fuse_frame_zone_pose, ym.yolo_world_points_per_camera_with_pose, ym.OUTPUT_DIR
    orig_build = getattr(ym, "build_camera_polygons", None)
    orig_filter = getattr(ym, "filter_by_curved_zone", None)
    zone_stats = {}
    ym.OUTPUT_DIR = Path(out_root) / mode
    ym.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if cfg["heights"] == "fit":
        ym.STANDING_HEIGHT_M, ym.LYING_HEIGHT_M = _MODEL.h_eff("standing"), _MODEL.h_eff("lying")
    else:
        ym.STANDING_HEIGHT_M, ym.LYING_HEIGHT_M = orig_heights

    def cached_yolo(model, image_path, poses):
        key = (str(image_path), round(ym.STANDING_HEIGHT_M, 4), round(ym.LYING_HEIGHT_M, 4))
        if key not in cache:
            cache[key] = orig_yolo(model, image_path, poses)
        return {cid: [dict(it) for it in items] for cid, items in cache[key].items()}

    ym.yolo_world_points_per_camera_with_pose = cached_yolo
    if cfg["match"] == "maha":
        ym.fuse_frame_zone_pose = fuse_mahalanobis
    if cfg["polygons"] == "he_kai":
        if orig_build is None:
            raise RuntimeError("phase3モジュールに build_camera_polygons がありません(曲線ゾーン版を使ってください)")

        def he_kai_polygons(poses, cam_ids, get_intrinsics_fn, height_m=None, n_angles=144, verbose=True):
            return build_valid_polygons(poses, cam_ids, get_intrinsics_fn, _MODEL, EPS_VALID_M, K_SIGMA,
                                        height_m=height_m if height_m is not None else 0.8,
                                        n_angles=n_angles, verbose=verbose)
        ym.build_camera_polygons = he_kai_polygons
    if cfg["polygons"] == "tri_gt":
        if orig_build is None:
            raise RuntimeError("phase3モジュールに build_camera_polygons がありません(曲線ゾーン版を使ってください)")
        ym.build_camera_polygons = build_tri_polygons
    if cfg.get("zone") == "overlap":
        from zone_overlap_filter_he_kai_ver1 import filter_zone_overlap
        cam_xy = {cid: np.asarray(p.t[:2], dtype=np.float64) for cid, p in _POSES.items()}

        def zone_filter(per_camera, cam_polygons):
            return filter_zone_overlap(per_camera, cam_polygons, cam_xy, zone_stats)
        ym.filter_by_curved_zone = zone_filter
    try:
        print(f"\n{'#' * 70}\n# モード: {mode}  {cfg}\n#   想定高さ standing={ym.STANDING_HEIGHT_M:.3f} lying={ym.LYING_HEIGHT_M:.3f}\n{'#' * 70}")
        ym.main()
        if zone_stats:
            tot = sum(zone_stats.values()) or 1
            print(f"  [zone] 担当ゾーン内で保持 {zone_stats.get('own', 0)} / 統合の帯で保持 {zone_stats.get('overlap', 0)} / 捨てた {zone_stats.get('dropped', 0)} (全体の {zone_stats.get('dropped', 0) / tot * 100:.1f}%)")
    finally:
        if orig_filter is not None:
            ym.filter_by_curved_zone = orig_filter
        ym.fuse_frame_zone_pose, ym.yolo_world_points_per_camera_with_pose = orig_fuse, orig_yolo
        if orig_build is not None:
            ym.build_camera_polygons = orig_build
        ym.STANDING_HEIGHT_M, ym.LYING_HEIGHT_M = orig_heights
        out_dir = ym.OUTPUT_DIR
        ym.OUTPUT_DIR = orig_out
    csvs = sorted(Path(out_dir).glob("*_summary.csv"))
    return pd.read_csv(csvs[-1], index_col=0).iloc[0].to_dict() if csvs else {}


def main():
    modes = sys.argv[1:] or DEFAULT_MODES
    bad = [m for m in modes if m not in MODES]
    if bad:
        raise SystemExit(f"不明なモード: {bad} / 使えるモード: {list(MODES)}")
    global _YM, _MODEL
    ym = _YM = load_yolo_module()
    json_path = os.environ.get("HE_SIGMA_JSON") or str(Path(ym.OUTPUT_DIR) / "bev_calibration_refined" / "sigma_he_kai_ver1.json")
    _MODEL = load_model(json_path)
    # 評価に使うフレーム(アノテーションのあるフレーム)。環境変数で上書きできる
    ym.FRAME_START = int(os.environ.get("HE_FRAME_START", ym.FRAME_START))
    ym.FRAME_END = int(os.environ.get("HE_FRAME_END", ym.FRAME_END))
    ym.FRAME_STEP = int(os.environ.get("HE_FRAME_STEP", ym.FRAME_STEP))
    # 姿勢別の統合制約(lyingとstandingは統合しない)のON/OFF。環境変数 HE_POSE_CONSTRAINT=1/0 で上書きできる
    # (未指定ならphase3モジュールの USE_POSE_CONSTRAINT のまま。ユークリッド/マハラノビスの両方に効く)
    if "HE_POSE_CONSTRAINT" in os.environ:
        ym.USE_POSE_CONSTRAINT = os.environ["HE_POSE_CONSTRAINT"] == "1"
    print(f"姿勢別統合制約: {'ON' if ym.USE_POSE_CONSTRAINT else 'OFF'}")
    # GTミス(『BBOXのラベル間違いや位置ミス』)のBBOXを評価のGTから除く。無効にするには HE_GT_IGNORE=0
    import gt_ignore_ver1 as gi
    gt_ignore_state = gi.apply(ym)
    print(f"phase3モジュール: {ym.__name__} / フレーム {ym.FRAME_START}〜{ym.FRAME_END} 刻み {ym.FRAME_STEP}")
    print(f"σのJSON: {json_path}")
    print(_MODEL.describe())
    print(f"GATE_CHI2={GATE_CHI2} MAX_EUCLID_M={MAX_EUCLID_M} / 有効範囲(maha_h_valid): eps={EPS_VALID_M}m k={K_SIGMA}"
          f" / 有効範囲(maha_h_tri): eps={TRI_EPS_M}m kind={TRI_KIND}")

    from Mootrack_core import ROT_FMT, apply_camera_overrides, load_floorplan_cameras, resolve_poses
    setup_poses(resolve_poses(apply_camera_overrides(load_floorplan_cameras(ym.FLOORPLAN_JSON_PATH)), root="Area1", fmt=ROT_FMT))
    orig_heights = (ym.STANDING_HEIGHT_M, ym.LYING_HEIGHT_M)
    out_root = Path(ym.OUTPUT_DIR) / "he_kai_eval_ver4"
    cache, results = {}, {}
    for m in modes:
        results[m] = run_mode(ym, m, MODES[m], out_root, cache, orig_heights)

    keys = ["idf1", "idp", "idr", "mota", "motp", "num_switches", "num_objects", "num_matches", "num_false_positives", "num_misses"]
    print(f"\n{'=' * 70}\n■ 比較 (同じYOLO検出・同じ校正。モードごとに変えた点だけが違う)\n{'=' * 70}")
    print(f"{'指標':<22}" + "".join(f"{m:>16}" for m in results))
    for k in keys:
        row = f"{k:<22}"
        for m in results:
            v = results[m].get(k)
            row += f"{'-':>16}" if v is None else (f"{v:>16.4f}" if isinstance(v, float) and abs(v) < 10 else f"{v:>16.0f}")
        print(row)
    print("\n※ motp は小さいほど良い(距離の2乗)。num_false_positives / num_misses / num_switches は小さいほど良い。")
    gi.report(gt_ignore_state)
    print("※ 追跡指標は飽和していて差が出にくい。差が出ない場合は、再現率/適合率(取りこぼし・誤統合)で比べる。")


if __name__ == "__main__":
    main()