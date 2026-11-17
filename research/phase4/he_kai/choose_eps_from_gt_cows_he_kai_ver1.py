"""
choose_eps_from_gt_cows_he_kai_ver1.py

HE(改)の有効範囲の ε/k を、GTの牛のいる位置から決めるための材料を出す。

考え方:
  有効範囲の条件は  k * sqrt(λmax(Σ)) <= ε  で、効くのは比 ε/k だけ(= 最悪の向きの 1σ 誤差の上限)。
  GTの各牛(フレーム, 牛)について、その牛が写っている各カメラで Σ を計算し、
  一番誤差の小さいカメラの  e = sqrt(λmax(Σ))  を「その牛のベスト誤差」とする。
  ベスト誤差の分布の分位点を ε/k にすれば、「牛のX%が、少なくとも1台のカメラの有効範囲に入る」ようになる。
  ε/k の候補ごとのカバー率(その候補以下の牛の割合)の表も出す。

  Σ は phase3_he_kai_matching_ver2.py の det_covariance と同じ式:
      Σ = σ_px(r)² J Jᵀ + σ_h(姿勢)² g gᵀ      (J: 1px前進差分, g = ρ/(H-h_a)·u, r: 画像中心からの半径)
  想定高さは h_eff(姿勢) = h_a + μ_h (σのJSONの値)。姿勢不明は 0.8 m。
  e_pose : 姿勢別の σ_h を使う(マッチングで使う Σ と同じ)
  e_max  : 全姿勢の σ_h の最大を使う(build_valid_polygons の有効範囲判定と同じ)  <- 有効範囲の判定にはこちらが対応する

使い方:
    uv run python ./phase4/he_kai/choose_eps_from_gt_cows_he_kai_ver1.py
環境変数:
    HE_FRAME_START / HE_FRAME_END / HE_FRAME_STEP : GTを見るフレーム (既定 0 / 2600 / 10)
    HE_SIGMA_JSON / HE_SIGMA_MODEL : σのJSON (他のスクリプトと同じ)
    HE_GT_IGNORE : 0 でGT除外リストを使わない (既定は gt_ignore_fn_ver1.csv があれば適用)
    HE_OUT_DIR : CSVの出力先 (既定 phase4/diagnose/analyze_check)
出力: <HE_OUT_DIR>/eps_from_gt_cows_he_kai_ver1.csv  (GTのBBOXごとの r, ρ, e_pose, e_max)
配置: research/phase4/he_kai/
注意: GTのBBOX中心(YOLOの検出ではない)で計算している。σ もGTから推定した値なので、実際の検出より誤差を小さく見積もる可能性がある。
"""

import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESEARCH_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
PHASE4_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
MOOTRACK_CODE_DIR = (os.path.abspath(os.path.join(RESEARCH_DIR, "calibration")))
for d in (RESEARCH_DIR, PHASE4_DIR, os.path.join(PHASE4_DIR, "web_annotate"),
          os.path.join(PHASE4_DIR, "calibration"), os.path.join(PHASE4_DIR, "calibration_try"),
          os.path.join(PHASE4_DIR, "he_kai"),
          MOOTRACK_CODE_DIR,
          os.path.join(RESEARCH_DIR, "analyze"), os.path.join(RESEARCH_DIR, "annotate"),
          os.path.join(RESEARCH_DIR, "phase3_multi_camera"), THIS_DIR):
    if os.path.isdir(d) and d not in sys.path:
        sys.path.insert(0, d)

FRAME_START = int(os.environ.get("HE_FRAME_START", 0))
FRAME_END = int(os.environ.get("HE_FRAME_END", 2600))
FRAME_STEP = int(os.environ.get("HE_FRAME_STEP", 10))
DEFAULT_HEIGHT_M = 0.8
CANDIDATES = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.25, 1.5, 2.0]
QUANTILES = [50, 75, 90, 95, 99]


def lam_max(S):
    return float(np.linalg.eigvalsh(S)[-1])


def cov_terms(cam_id, uv, h_a, poses):
    """検出1つの (JJᵀ, g gᵀ, r, ρ) を返す。投影できなければ None"""
    from Mootrack_core import get_intrinsics, unproject_to_floor
    cp = poses[cam_id]
    fx, fy, cx, cy, xi, alpha = get_intrinsics(cam_id)
    pts = np.array([uv, [uv[0] + 1.0, uv[1]], [uv[0], uv[1] + 1.0]], dtype=np.float64)
    xy, ok = unproject_to_floor(pts, cp.R, cp.t, fx, fy, cx, cy, xi, alpha, floor_z=-h_a)
    if not ok.all():
        return None
    J = np.stack([xy[1] - xy[0], xy[2] - xy[0]], axis=1)
    cam_xy = np.asarray(cp.t[:2], dtype=np.float64)
    rho = float(np.linalg.norm(xy[0] - cam_xy))
    if rho < 1e-6:
        return None
    u = (xy[0] - cam_xy) / rho
    g = (rho / max(float(-cp.t[2]) - h_a, 0.3)) * u
    r = float(np.hypot(uv[0] - cx, uv[1] - cy))
    return J @ J.T, np.outer(g, g), r, rho


def errors_for(cam_id, uv, pose, poses, model):
    """(e_pose, e_max, r, rho): 最悪の向きの 1σ 誤差 [m]"""
    h_a = model.h_eff(pose) if pose in ("standing", "lying") else DEFAULT_HEIGHT_M
    t = cov_terms(cam_id, uv, h_a, poses)
    if t is None:
        return None
    JJt, ggt, r, rho = t
    sp = float(model.sigma_px(r))
    e_pose = np.sqrt(lam_max(sp ** 2 * JJt + model.sigma_h(pose) ** 2 * ggt))
    e_max = np.sqrt(lam_max(sp ** 2 * JJt + model.sigma_h_max() ** 2 * ggt))
    return float(e_pose), float(e_max), r, rho


def main():
    import render_bev_frames_yolo_zonepose_ver1 as rf
    import gt_ignore_ver1 as gi
    from he_kai_sigma_model_ver1 import load_model
    from Mootrack_core import CAM_IDS, ROT_FMT, apply_camera_overrides, load_floorplan, resolve_poses

    gt_ignore_state = gi.apply(rf)                      # 既知のGTミスを除く(HE_GT_IGNORE=0 で無効)
    regions, cameras_json, real_w, real_h = load_floorplan(rf.FLOORPLAN_JSON_PATH)
    poses = resolve_poses(apply_camera_overrides(cameras_json), root="Area1", fmt=ROT_FMT)
    json_path = os.environ.get("HE_SIGMA_JSON") or str(Path(rf.OUTPUT_DIR) / "bev_calibration_refined" / "sigma_he_kai_ver1.json")
    model = load_model(json_path)
    print(model.describe())

    tile_w = rf.IMAGE_WIDTH_PX / len(CAM_IDS)
    rows = []                                           # GTのBBOXごと
    best = defaultdict(list)                            # (frame, cow) -> [(e_pose, e_max, cam)]
    n_frames = 0
    for fr in range(FRAME_START, FRAME_END + 1, FRAME_STEP):
        xml = Path(rf.XML_DIR) / f"{rf.frame_name(fr)}.xml"
        if not xml.exists():
            continue
        n_frames += 1
        for o in rf.parse_voc_annotation(xml):
            cx_full = (o["xmin"] + o["xmax"]) / 2.0
            cam_idx = max(0, min(int(cx_full // tile_w), len(CAM_IDS) - 1))
            cam_id = CAM_IDS[cam_idx]
            uv = (cx_full - cam_idx * tile_w, (o["ymin"] + o["ymax"]) / 2.0)
            res = errors_for(cam_id, uv, o["pose"], poses, model)
            if res is None:
                continue
            e_pose, e_max, r, rho = res
            rows.append([fr, o["name"], cam_id, o["pose"] or "", round(r, 1), round(rho, 3), round(e_pose, 4), round(e_max, 4)])
            best[(fr, o["name"])].append((e_pose, e_max, cam_id))

    if not rows:
        raise SystemExit("GTのBBOXが1つも計算できませんでした(フレーム範囲・XML_DIRを確認してください)")
    print(f"\n対象: {n_frames}フレーム, GTのBBOX {len(rows)}個, (フレーム,牛) {len(best)}組")

    e_pose_obs = np.array([r[6] for r in rows])
    e_max_obs = np.array([r[7] for r in rows])
    best_pose = np.array([min(x[0] for x in v) for v in best.values()])
    best_max = np.array([min(x[1] for x in v) for v in best.values()])
    n_cams = np.array([len(v) for v in best.values()])
    print(f"2台以上に写っている牛: {(n_cams >= 2).mean() * 100:.1f}%  (1台だけ: {(n_cams == 1).mean() * 100:.1f}%)")

    print("\n【1】ベスト誤差 sqrt(λmax(Σ)) [m] の分布 (牛ごとに一番小さいカメラ)")
    print(f"{'':<22}" + "".join(f"{'p' + str(q):>8}" for q in QUANTILES) + f"{'最大':>8}")
    for label, arr in (("e_max(範囲判定と同じ)", best_max), ("e_pose(姿勢別σ_h)", best_pose)):
        print(f"{label:<22}" + "".join(f"{np.percentile(arr, q):>8.3f}" for q in QUANTILES) + f"{arr.max():>8.3f}")

    print("\n【2】ε/k の候補ごとのカバー率 (ε/k = 最悪の向きの1σ誤差の上限 [m])")
    print(f"{'ε/k':>6}{'牛のカバー率(e_max)':>22}{'牛のカバー率(e_pose)':>22}{'BBOX単位(e_max)':>18}   ※ k=1 なら ε=ε/k, k=3.03 なら ε=3.03×ε/k")
    for c in CANDIDATES:
        print(f"{c:>6.2f}{(best_max <= c).mean() * 100:>21.1f}%{(best_pose <= c).mean() * 100:>21.1f}%{(e_max_obs <= c).mean() * 100:>17.1f}%")

    print("\n【3】カバー率から決める ε/k (e_max)")
    for q in (90, 95, 99):
        v = float(np.percentile(best_max, q))
        print(f"  牛の{q}%をカバー: ε/k = {v:.3f} m  -> k=1 なら ε={v:.3f}, k=3.03 なら ε={3.03 * v:.3f}")

    print("\n【4】どのカメラがベストか (e_max)")
    cnt = defaultdict(int)
    for v in best.values():
        cnt[min(v, key=lambda x: x[1])[2]] += 1
    for cid in CAM_IDS:
        print(f"  {cid}: {cnt[cid]}組 ({cnt[cid] / len(best) * 100:.1f}%)")

    out_dir = Path(os.environ.get("HE_OUT_DIR") or Path(PHASE4_DIR) / "diagnose" / "analyze_check")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "eps_from_gt_cows_he_kai_ver1.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["frame", "cow", "camera", "pose", "r_px", "rho_m", "e_pose_m", "e_max_m"])
        w.writerows(rows)
    print(f"\nCSV: {out}")
    gi.report(gt_ignore_state)
    print("\n注意: GTのBBOX中心とGT由来のσで計算している。YOLOの検出誤差は含まないので、実際の範囲はこれより狭く使うのが安全。")
    print("      カバー率だけでなく、FN/FP/IDF1(phase3_he_kai_matching_ver2.py の HE_EPS_M 掃引)でも確認すること。")


if __name__ == "__main__":
    main()