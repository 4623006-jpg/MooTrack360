"""
eval_bev_yolo_motp_ver2.py   (ver1からの変更: 曲線境界フィルタ等をYOLO評価ファイルに頼らず元モジュールから直接読む。
                              YOLO評価ファイルの名前を YOLO_MODULE で切り替え可能)

eval_bev_gt_motp_ver1.py のYOLO版。手動で作った1フレーム分のBEVのGT(json)を正解として、
YOLOで検出したbboxを高さ仮定でBEVに落とした位置の誤差(MOTP)を出す。
YOLOのクラス(lying / standing)で高さを切り替える(phase3_bev_multicam_yolo_check_zonepose_ver3.py と同じ)。

予測点のモード:
  yolo_fused        : 重複除去 -> 曲線境界/牛舎範囲で除外 -> 隣接カメラ+姿勢制約+距離閾値で統合した位置
                      (YOLO評価で使っている融合結果そのまま。トラッカーは1フレームなので使わない)
  yolo_cam:Camera_N : そのカメラ1台の検出(重複除去まで、曲線境界での除外はしない)の投影位置。
                      カメラごと・姿勢クラスごとの投影誤差を見る用。そのカメラに写っていない牛は
                      「取りこぼし」になる(正解は牛舎全体の牛なので、カメラ別の取りこぼしは多くて当然)。

追加の出力:
  - 姿勢クラス別の誤差 : yolo_cam の対応づけで、YOLOのクラス(lying/standing)ごとの誤差の平均・中央値
  - 高さの掃引         : 対応づけたペアについて、仮定する高さを 0.3〜1.2m で変えたときの平均誤差を、
                         姿勢クラスごとに出す。誤差が最小になる高さが、その姿勢クラスに合う高さの目安
                         (今は lying=0.5 / standing=0.75)。対応づけは現在の高さで行い、固定している。
                         手動GTのクリックのずれや、bbox中心と実際の位置のずれも含まれるので、目安として見る。

関数(対応づけ・集計・描画)は eval_bev_gt_motp_ver1.py のものを再利用する。同じフォルダに置くこと。
YOLOの推論は3カメラ×1フレーム分だけ。モデルの読み込みに少し時間がかかる。

出力(OUTPUT_DIR / bev_yolo_motp /):
  motp_yolo_{frame}_ver1.csv        : モード・閾値ごとの全ペアの距離
  motp_yolo_{frame}_{mode}_ver1.png : 対応づけの図
  height_sweep_{frame}_ver1.csv     : 姿勢×高さごとの平均誤差

使い方:
    python eval_bev_yolo_motp_ver1.py 100
"""

import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYZE_DIR = os.path.join(THIS_DIR, "..", "analyze")
ANNOTATE_DIR = os.path.join(THIS_DIR, "..", "annotate")
PHASE3_DIR = os.path.join(THIS_DIR, "..", "phase3_multi_camera")
for d in (os.path.join(THIS_DIR, "..", "calibration"), ANALYZE_DIR, ANNOTATE_DIR, PHASE3_DIR, THIS_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

import importlib

import eval_bev_gt_motp_ver1 as e
from Mootrack_core import CAM_IDS, filter_outside_floorplan_bounds, get_intrinsics, unproject_to_floor
from curved_zone_utils_ver1 import filter_by_curved_zone

# YOLO推論・重複除去・融合の関数と設定値を持つ評価ファイル名(.pyなし)。使っている最新の版に変える。
# 必要な名前: IMAGE_DIR, FLOORPLAN_JSON_PATH, OUTPUT_DIR, YOLO_WEIGHTS_PATH, YOLO_CONF_THRESH,
#            DEDUP_IOU_THRESHOLD, DIST_THR_FUSION_M, USE_POSE_CONSTRAINT, LYING_HEIGHT_M, STANDING_HEIGHT_M,
#            frame_name, yolo_world_points_per_camera_with_pose, deduplicate_per_camera, fuse_frame_zone_pose
YOLO_MODULE = "phase3_bev_multicam_yolo_check_zonepose_ver6"
y = importlib.import_module(YOLO_MODULE)

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
DEFAULT_FRAME = 100
THRESHOLDS_M = e.THRESHOLDS_M
MODES = ["yolo_fused"] + [f"yolo_cam:{c}" for c in CAM_IDS]
DRAW_MODES = ["yolo_fused"]
DRAW_THR_M = 1.0
POSE_THR_M = 3.0                  # 姿勢別集計・高さ掃引で使う対応づけの閾値
HEIGHT_SWEEP = [round(h, 2) for h in np.arange(0.3, 1.21, 0.1)]
OUT_DIR = y.OUTPUT_DIR / "bev_yolo_motp"


def build_yolo_predictions(frame_idx, model, poses, cam_polygons, adjacent_pairs, real_w, real_h):
    image_path = Path(y.IMAGE_DIR) / f"{y.frame_name(frame_idx)}.jpg"
    if not image_path.exists():
        raise FileNotFoundError(f"画像が見つかりません: {image_path}")
    raw = y.yolo_world_points_per_camera_with_pose(model, image_path, poses)
    deduped = y.deduplicate_per_camera(raw, y.DEDUP_IOU_THRESHOLD)
    filtered = filter_by_curved_zone(deduped, cam_polygons)
    filtered = filter_outside_floorplan_bounds(filtered, real_w, real_h, margin_m=0.0, verbose=False)
    fused = y.fuse_frame_zone_pose(filtered, adjacent_pairs, y.DIST_THR_FUSION_M, y.USE_POSE_CONSTRAINT)[0]  # 戻り値の1つ目が融合点リスト

    n = lambda d: sum(len(v) for v in d.values())
    print(f"YOLO検出: 生={n(raw)} 重複除去後={n(deduped)} 曲線境界+牛舎範囲後={n(filtered)} 融合点={len(fused)}")
    for cid in CAM_IDS:
        poses_count = defaultdict(int)
        for it in deduped[cid]:
            poses_count[it["pose"]] += 1
        print(f"  {cid}: 重複除去後{len(deduped[cid])}件 {dict(poses_count)}")

    preds, meta = {}, {}
    preds["yolo_fused"] = [(f"{f['n_cams']}cam", np.asarray(f["world_xy"], dtype=float)) for f in fused]
    meta["yolo_fused"] = [None] * len(fused)
    for cid in CAM_IDS:
        mode = f"yolo_cam:{cid}"
        preds[mode] = [(it["pose"] or "?", np.asarray(it["world_xy"], dtype=float)) for it in deduped[cid]]
        meta[mode] = [{"cam": cid, "pose": it["pose"] or "?",
                       "px": np.array([(it["bbox_local"][0] + it["bbox_local"][2]) / 2.0,
                                       (it["bbox_local"][1] + it["bbox_local"][3]) / 2.0])}
                      for it in deduped[cid]]
    return preds, meta


def project_at_height(poses, cam_id, px, height_m):
    fx, fy, cx, cy, xi, alpha = get_intrinsics(cam_id)
    cam_pose = poses[cam_id]
    xy, valid = unproject_to_floor(np.array([px], dtype=np.float64), cam_pose.R, cam_pose.t,
                                   fx, fy, cx, cy, xi, alpha, floor_z=np.array([-height_m]))
    return xy[0] if valid[0] else None


def main():
    frame_idx = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FRAME
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO

    regions, cameras_json, real_w, real_h = e.load_floorplan(y.FLOORPLAN_JSON_PATH)
    poses = e.g.resolve_poses(e.g.apply_camera_overrides(cameras_json), root="Area1", fmt=e.g.ROT_FMT)
    cam_polygons = e.g.build_camera_polygons(poses, e.g.CAM_IDS, e.g.get_intrinsics,
                                             height_m=e.g.ZONE_HEIGHT_M, n_angles=e.g.N_ANGLES)
    adjacent_pairs = e.g.get_adjacent_pairs(e.g.CAM_IDS)

    ann = e.load_annotation(frame_idx)
    print(f"YOLOモデル読み込み中: {y.YOLO_WEIGHTS_PATH} (conf>={y.YOLO_CONF_THRESH})")
    model = YOLO(y.YOLO_WEIGHTS_PATH)
    preds, meta = build_yolo_predictions(frame_idx, model, poses, cam_polygons, adjacent_pairs, real_w, real_h)
    base = e.a.build_base(regions, poses, cam_polygons, real_w, real_h)

    print(f"\n=== フレーム {y.frame_name(frame_idx)}  手動BEVGT={len(ann)}点 (YOLO) ===")
    print(f"{'モード':20s} {'閾値':>5s} {'MOTP':>7s} {'中央値':>7s} {'90%点':>7s} {'最大':>7s} "
          f"{'予測数':>6s} {'対応':>4s} {'取りこぼし':>8s} {'余り':>4s}   (距離はm)")
    rows = []
    for mode in MODES:
        pred = preds[mode]
        for thr in THRESHOLDS_M:
            matches, _up, _ua = e.match_points(pred, ann, thr)
            s = e.summarize(matches, len(pred), len(ann))
            print(f"{mode:20s} {thr:5.1f} {e.fmt(s['motp']):>7s} {e.fmt(s['median']):>7s} {e.fmt(s['p90']):>7s} "
                  f"{e.fmt(s['max']):>7s} {s['n_pred']:6d} {s['n_match']:4d} {s['miss']:8d} {s['fp']:4d}")
            for pi, ai, dist in matches:
                rows.append([mode, thr, ann[ai][0], pred[pi][0], f"{dist:.4f}"])

    csv_path = OUT_DIR / f"motp_yolo_{y.frame_name(frame_idx)}_ver1.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mode", "thr_m", "ann_id", "pred_label", "dist_m"])
        w.writerows(rows)
    print(f"\nCSV: {csv_path}")

    # --- 姿勢クラス別の誤差 + 高さ掃引(カメラ別の検出を全カメラ分まとめる) ---
    by_pose = defaultdict(list)                    # pose -> [dist]
    sweep_items = []                               # (pose, cam, px, ann_xy)
    for cid in CAM_IDS:
        mode = f"yolo_cam:{cid}"
        matches, _up, _ua = e.match_points(preds[mode], ann, POSE_THR_M)
        for pi, ai, dist in matches:
            m = meta[mode][pi]
            by_pose[m["pose"]].append(dist)
            sweep_items.append((m["pose"], m["cam"], m["px"], ann[ai][1]))

    print(f"\n--- 姿勢クラス別の誤差 (カメラ別の検出, 閾値{POSE_THR_M}m, 距離はm) ---")
    for pose in ("lying", "standing", "?"):
        d = np.array(by_pose.get(pose, []))
        if len(d):
            print(f"  {pose:9s} 対応数={len(d):3d}  平均={d.mean():.3f}  中央値={np.median(d):.3f}  最大={d.max():.3f}")

    print(f"\n--- 高さの掃引 (姿勢クラスごとの平均誤差[m]。最小の高さが目安。現在: lying={y.LYING_HEIGHT_M}, "
          f"standing={y.STANDING_HEIGHT_M}) ---")
    sweep_rows = []
    header = "  高さ[m] " + " ".join(f"{h:6.2f}" for h in HEIGHT_SWEEP)
    print(header)
    for pose in ("lying", "standing"):
        items = [it for it in sweep_items if it[0] == pose]
        if not items:
            continue
        means = []
        for h in HEIGHT_SWEEP:
            ds = []
            for _p, cam, px, axy in items:
                xy = project_at_height(poses, cam, px, h)
                if xy is not None:
                    ds.append(float(np.linalg.norm(xy - axy)))
            means.append(float(np.mean(ds)) if ds else float("nan"))
            sweep_rows.append([pose, h, len(ds), f"{means[-1]:.4f}"])
        best = HEIGHT_SWEEP[int(np.nanargmin(means))]
        print(f"  {pose:8s}" + " ".join(f"{m:6.3f}" for m in means) + f"   <- 最小: {best}m (n={len(items)})")
    sweep_csv = OUT_DIR / f"height_sweep_{y.frame_name(frame_idx)}_ver1.csv"
    with open(sweep_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pose", "height_m", "n", "mean_dist_m"])
        w.writerows(sweep_rows)
    print(f"CSV: {sweep_csv}")

    for mode in DRAW_MODES:
        matches, unm_pred, unm_ann = e.match_points(preds[mode], ann, DRAW_THR_M)
        s = e.summarize(matches, len(preds[mode]), len(ann))
        title = (f"{y.frame_name(frame_idx)} {mode} thr={DRAW_THR_M}m  MOTP={e.fmt(s['motp']).strip()}m "
                 f"対応={s['n_match']} 取りこぼし(赤輪)={s['miss']} 余り(マゼンタ輪)={s['fp']}")
        img = e.draw_matching(base, ann, preds[mode], matches, unm_pred, unm_ann, title)
        out = OUT_DIR / f"motp_yolo_{y.frame_name(frame_idx)}_{mode.replace(':', '_')}_ver1.png"
        cv2.imwrite(str(out), img)
        print(f"画像: {out}")


if __name__ == "__main__":
    main()