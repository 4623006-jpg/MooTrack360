"""
tri_gt_eval_modes_he_kai_ver2.py

モードごとの「統合した牛の位置」が、三角測量GT(複数カメラのクリックから三角測量した真の位置)にどれだけ近いかを測る。
  - 評価(phase3)の MOTP は、GTのbbox中心を想定高さで投影した位置との差で、真の位置との差ではない。
    ここでは、YOLO検出 -> 各モードの融合(統合した位置) と 三角測量GT の (X, Y) との距離 [m] を直接比べる。
  - 三角測量GTのある各フレーム・各牛について、そのフレームの融合位置のうち最も近いもの(ハンガリアン法、ゲート内)を対応づけ、
    距離を集計する: 対応づけられた割合 / 平均 / 中央値 / RMS / 90%点 / 最大。

使い方(research/ で。引数はフレーム番号。省略すると三角測量GTのあるフレーム全部):
    uv run python ./phase4/he_kai/tri_gt_eval_modes_he_kai_ver2.py euclid_h maha_h maha_h_valid
  (モード名だけ並べる。フレーム番号は環境変数 TRI_EVAL_FRAMES="100,200" で指定)
環境変数:
    HE_SIGMA_JSON / HE_SIGMA_MODEL / HE_EPS_M / HE_K_SIGMA / HE_GATE_CHI2 / HE_POSE_CONSTRAINT : phase3_he_kai_matching と同じ
    TRI_EVAL_GATE_M   : 三角測量GTと融合位置を対応づける距離の上限 [m] (既定 1.5)
    TRI_EVAL_FRAMES   : 評価するフレーム(カンマ区切り。既定: 三角測量GTのある全フレーム)
    HE_MATCHING_MODULE: 読み込む phase3_he_kai_matching のモジュール名 (既定 phase3_he_kai_matching_ver10)
出力: phase4/he_kai_out/tri_gt_eval_ver2/ (牛ごとの誤差 CSV と、モードごとの要約 CSV)
注意: 三角測量GTの牛は、校正(LM+牛)の学習に使った牛と重なる場合、誤差が小さめに出る(循環)。
"""

import csv
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESEARCH_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
PHASE4_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
MOOTRACK_CODE_DIR = os.path.abspath(os.path.join(RESEARCH_DIR, "calibration"))
for d in (RESEARCH_DIR, PHASE4_DIR, os.path.join(PHASE4_DIR, "web_annotate"), os.path.join(PHASE4_DIR, "calibration"),
          os.path.join(PHASE4_DIR, "calibration_try"), os.path.join(PHASE4_DIR, "he_kai"),
          MOOTRACK_CODE_DIR,
          os.path.join(RESEARCH_DIR, "analyze"), os.path.join(RESEARCH_DIR, "annotate"),
          os.path.join(RESEARCH_DIR, "phase3_multi_camera"), THIS_DIR):
    if os.path.isdir(d) and d not in sys.path:
        sys.path.insert(0, d)

os.environ.setdefault("HE_SIGMA_MODEL", "const")
DEFAULT_SIGMA_JSON = os.path.join(RESEARCH_DIR, "idf1_check", "bev_calibration_refined", "sigma_he_kai_ver4.json")
OUT_DIR = Path(os.environ.get("HE_OUT_DIR", os.path.join(PHASE4_DIR, "he_kai_out"))) / "tri_gt_eval_ver2"
GATE_M = float(os.environ.get("TRI_EVAL_GATE_M", 1.5))
DEFAULT_MODES = ["euclid_h", "maha_h", "maha_h_valid"]


def load_tri_cows(path, min_angle, max_resid):
    """三角測量GTの牛: [(牛のid, 真の位置 (X, Y), 高さ)]。品質の悪いものは除く"""
    out = []
    for cow in json.loads(Path(path).read_text(encoding="utf-8"))["cows"]:
        r = cow.get("result", {})
        if r.get("level") != 1 or r.get("X") is None:
            continue
        if r.get("max_angle_deg", 0) < min_angle or r.get("resid_m", 9) > max_resid:
            continue
        out.append((cow.get("id"), np.array(r["X"][:2], dtype=np.float64), r.get("height_m")))
    return out


def fused_to_xy(result):
    """融合関数の戻り値から、融合位置の配列 (n, 2) を作る。
    fuse_frame_zone_pose は (融合リスト, 姿勢で却下した数) のタプルを返すので、その場合は先頭だけ使う。
    各要素は {"world_xy": ...} の辞書、または座標そのもの。"""
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], (list, tuple)):
        result = result[0]
    pts = []
    for f in result:
        if isinstance(f, dict):
            f = f.get("world_xy", f.get("xy"))
        pts.append(np.asarray(f, dtype=np.float64).ravel()[:2])
    return np.array(pts).reshape(-1, 2)


def associate(truth_xy, fused_xy, gate_m):
    """真の位置と融合位置をハンガリアン法で対応づける(距離の合計が最小)。戻り値: 各真の位置の誤差 [m] (対応なし/ゲート外は nan)"""
    err = np.full(len(truth_xy), np.nan)
    if len(truth_xy) == 0 or len(fused_xy) == 0:
        return err
    D = np.linalg.norm(truth_xy[:, None, :] - fused_xy[None, :, :], axis=2)
    cost = np.where(D <= gate_m, D, 1e6)
    rows, cols = linear_sum_assignment(cost)
    for r, c in zip(rows, cols):
        if cost[r, c] < 1e5:
            err[r] = D[r, c]
    return err


def summarize(err_all, n_truth):
    e = err_all[np.isfinite(err_all)]
    if len(e) == 0:
        return dict(n_truth=n_truth, n_found=0)
    return dict(n_truth=n_truth, n_found=len(e), found_pct=100.0 * len(e) / max(n_truth, 1),
                mean=float(e.mean()), median=float(np.median(e)), rms=float(np.sqrt((e ** 2).mean())),
                q90=float(np.quantile(e, 0.9)), max=float(e.max()))


def main():
    from ultralytics import YOLO
    import phase3_bev_multicam_gt_zonepose_ver1 as g
    hk = importlib.import_module(os.environ.get("HE_MATCHING_MODULE", "phase3_he_kai_matching_ver10"))
    import render_bev_frames_yolo_zonepose_ver1 as rf_old
    from curved_zone_utils_ver1 import N_ANGLES, ZONE_HEIGHT_M, build_camera_polygons, filter_by_curved_zone, get_adjacent_pairs
    from he_kai_sigma_model_ver1 import build_valid_polygons, load_model
    from Mootrack_core import (CAM_IDS, ROT_FMT, apply_camera_overrides, filter_outside_floorplan_bounds, get_intrinsics,
                               load_floorplan, resolve_poses)

    ym = hk.load_yolo_module()

    class _Proxy:
        def __init__(self, primary, fallback):
            object.__setattr__(self, "_p", primary)
            object.__setattr__(self, "_f", fallback)

        def __getattr__(self, name):
            try:
                return getattr(self._p, name)
            except AttributeError:
                return getattr(self._f, name)

        def __setattr__(self, name, value):
            setattr(self._p, name, value)

    rf = _Proxy(ym, rf_old)

    modes = [a for a in sys.argv[1:]] or DEFAULT_MODES
    bad = [m for m in modes if m not in hk.MODES]
    if bad:
        raise SystemExit(f"不明なモード: {bad} / 使えるモード: {list(hk.MODES)}")
    if "HE_POSE_CONSTRAINT" in os.environ:
        rf.USE_POSE_CONSTRAINT = os.environ["HE_POSE_CONSTRAINT"] == "1"

    try:
        import analyze_projection_error_vs_tri_gt_ver3 as an
        min_angle, max_resid = an.MIN_ANGLE_DEG, an.MAX_RESID_M
    except Exception:
        min_angle, max_resid = 15.0, 0.5
        print(f"(品質のしきい値: 既定 max_angle>={min_angle}° resid<={max_resid} m を使用)")

    in_dir = g.OUTPUT_DIR / "bev_gt_triangulated"
    env_frames = os.environ.get("TRI_EVAL_FRAMES")
    frames = [int(x) for x in env_frames.split(",")] if env_frames else sorted(int(p.name[:6]) for p in in_dir.glob("*_tri_gt_ver1.json"))
    print(f"三角測量GTのフレーム: {frames} / モード: {modes} / 対応づけゲート {GATE_M} m")

    regions, cameras_json, real_w, real_h = load_floorplan(rf.FLOORPLAN_JSON_PATH)
    poses = resolve_poses(apply_camera_overrides(cameras_json), root="Area1", fmt=ROT_FMT)
    model = load_model(os.environ.get("HE_SIGMA_JSON") or DEFAULT_SIGMA_JSON)
    print(model.describe())
    hk._MODEL, hk._YM = model, ym
    hk.setup_poses(poses)
    orig_heights = (rf.STANDING_HEIGHT_M, rf.LYING_HEIGHT_M)

    polygons = {}
    for m in modes:
        kind = hk.MODES[m]["polygons"]
        if kind == "he_kai":
            polygons[m] = build_valid_polygons(poses, CAM_IDS, get_intrinsics, model, hk.EPS_VALID_M, hk.K_SIGMA,
                                               height_m=ZONE_HEIGHT_M, n_angles=N_ANGLES, verbose=False)
        elif kind == "tri_gt":
            polygons[m] = hk.build_tri_polygons(poses, CAM_IDS, get_intrinsics, height_m=ZONE_HEIGHT_M, n_angles=N_ANGLES)
        else:
            polygons[m] = build_camera_polygons(poses, CAM_IDS, get_intrinsics, height_m=ZONE_HEIGHT_M, n_angles=N_ANGLES, verbose=False)
    adjacent_pairs = get_adjacent_pairs(CAM_IDS)
    yolo_model = YOLO(rf.YOLO_WEIGHTS_PATH)

    errs = {m: [] for m in modes}
    rows = []
    n_truth = 0
    for fr in frames:
        tpath = in_dir / f"{fr:06d}_tri_gt_ver1.json"
        ipath = Path(rf.IMAGE_DIR) / f"{rf.frame_name(fr)}.jpg"
        if not tpath.exists() or not ipath.exists():
            print(f"  見つかりません: {tpath if not tpath.exists() else ipath}")
            continue
        cows = load_tri_cows(tpath, min_angle, max_resid)
        truth_xy = np.array([c[1] for c in cows]).reshape(-1, 2)
        n_truth += len(cows)
        raw_cache = {}
        for m in modes:
            cfg = hk.MODES[m]
            if cfg["heights"] == "fit":
                rf.STANDING_HEIGHT_M, rf.LYING_HEIGHT_M = model.h_eff("standing"), model.h_eff("lying")
            else:
                rf.STANDING_HEIGHT_M, rf.LYING_HEIGHT_M = orig_heights
            if cfg["heights"] not in raw_cache:
                raw_cache[cfg["heights"]] = rf.yolo_world_points_per_camera_with_pose(yolo_model, ipath, poses)
            raw = {cid: [dict(it) for it in items] for cid, items in raw_cache[cfg["heights"]].items()}
            deduped = rf.deduplicate_per_camera(raw, rf.DEDUP_IOU_THRESHOLD)
            filtered = filter_by_curved_zone(deduped, polygons[m])
            filtered = filter_outside_floorplan_bounds(filtered, real_w, real_h, margin_m=0.0, verbose=False)
            if cfg["match"] == "maha":
                fused = hk.fuse_mahalanobis(filtered, adjacent_pairs, rf.DIST_THR_FUSION_M, rf.USE_POSE_CONSTRAINT)[0]
            else:
                fused = rf.fuse_frame_zone_pose(filtered, adjacent_pairs, rf.DIST_THR_FUSION_M, rf.USE_POSE_CONSTRAINT)
            fused_xy = fused_to_xy(fused)
            err = associate(truth_xy, fused_xy, GATE_M)
            errs[m].append(err)
            for (cid, xy, h), e in zip(cows, err):
                rows.append(dict(mode=m, frame=fr, cow=cid, truth_x=round(float(xy[0]), 3), truth_y=round(float(xy[1]), 3),
                                 true_height=h, error_m=None if not np.isfinite(e) else round(float(e), 4)))
        rf.STANDING_HEIGHT_M, rf.LYING_HEIGHT_M = orig_heights

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "per_cow_errors.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["mode", "frame", "cow", "truth_x", "truth_y", "true_height", "error_m"])
        w.writeheader()
        w.writerows(rows)

    print(f"\n三角測量GTの牛(のべ) {n_truth}頭 / 真の位置との距離 [m] (対応づけゲート {GATE_M} m 以内のみ)")
    print(f"{'mode':<18}{'見つかった':>10}{'割合%':>8}{'平均':>8}{'中央値':>8}{'RMS':>8}{'90%点':>8}{'最大':>8}")
    summary = []
    for m in modes:
        s = summarize(np.concatenate(errs[m]) if errs[m] else np.array([]), n_truth)
        summary.append(dict(mode=m, **s))
        if s["n_found"]:
            print(f"{m:<18}{s['n_found']:>10}{s['found_pct']:>8.1f}{s['mean']:>8.3f}{s['median']:>8.3f}{s['rms']:>8.3f}{s['q90']:>8.3f}{s['max']:>8.3f}")
        else:
            print(f"{m:<18}{0:>10}")
    with open(OUT_DIR / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["mode", "n_truth", "n_found", "found_pct", "mean", "median", "rms", "q90", "max"])
        w.writeheader()
        w.writerows(summary)
    print(f"\n保存: {OUT_DIR}")
    print("※ 三角測量GTの牛が校正(LM+牛)の学習に使われている場合、誤差は小さめに出ます。")


if __name__ == "__main__":
    main()