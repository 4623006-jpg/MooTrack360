"""
diagnose_epipolar_triangulation_ver1.py

phase3_bev_multicam_yolo_check_epipolar.py(同じフォルダに置く)の
YOLO検出・重複除去・match_epipolar をそのまま使い、「採用された統合ペア」ごとに
三角測量点が座標に使えそうかを診断する。トラッキング結果・座標は一切変えない。

出力 (OUTPUT_DIR):
  triangulation_diagnosis_pairs_ver1.csv   : 採用ペアごとの指標
  triangulation_diagnosis_summary_ver1.txt : 交差角ビン別・姿勢別の集計

見るべき点:
  - height_m (三角測量点の高さ = -z): lying 0.5 / standing 0.75 付近に集まるか
  - height_err_m: 仮定高さとの差(符号付き)。中央値がずれていればbbox中心ずれ由来のバイアス
  - xy_diff_m: 三角測量点のxy と 高さ仮定投影の平均xy との差
  - fb_gap_m: 2カメラの高さ仮定投影同士の差(高さ仮定の不整合の大きさ)
  - 交差角が大きいビンほど height_err / xy_diff が小さくなるか

使い方:
    python diagnose_epipolar_triangulation_ver1.py
"""
import os
import sys

phase3_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","phase3_multi_camera")
if phase3_DIR not in sys.path:
    sys.path.insert(0, phase3_DIR)

import csv
from pathlib import Path

import numpy as np

import phase3_bev_multicam_yolo_check_epipolar as ep

ANGLE_BINS_DEG = [0, 20, 40, 60, 90, 181]
OUT_CSV = ep.OUTPUT_DIR / "triangulation_diagnosis_pairs_ver1.csv"
OUT_TXT = ep.OUTPUT_DIR / "triangulation_diagnosis_summary_ver1.txt"


def pair_metrics(frame_idx, key, tri_mid, per_camera, poses):
    (cam_i, a), (cam_j, b) = key
    it_i, it_j = per_camera[cam_i][a], per_camera[cam_j][b]
    d1, d2 = it_i["dir_world"], it_j["dir_world"]
    d1 = d1 / np.linalg.norm(d1)
    d2 = d2 / np.linalg.norm(d2)
    angle = float(np.degrees(np.arccos(np.clip(np.dot(d1, d2), -1.0, 1.0))))

    _t1, _t2, _p1, _p2, ray_dist = ep.closest_points_between_rays(
        poses[cam_i].t, d1, poses[cam_j].t, d2
    )
    pose = it_i["pose"] or it_j["pose"]
    assumed_h = ep.height_for(pose)
    height_m = float(-tri_mid[2])

    fb_i, fb_j = it_i["world_xy_fallback"], it_j["world_xy_fallback"]
    fb_mean = (fb_i + fb_j) / 2.0

    return {
        "frame": frame_idx,
        "cam_i": cam_i, "cam_j": cam_j,
        "pose": pose,
        "angle_deg": round(angle, 2),
        "ray_dist_m": round(float(ray_dist), 4),
        "height_m": round(height_m, 3),
        "assumed_height_m": assumed_h,
        "height_err_m": round(height_m - assumed_h, 3),
        "tri_x": round(float(tri_mid[0]), 3), "tri_y": round(float(tri_mid[1]), 3),
        "fb_mean_x": round(float(fb_mean[0]), 3), "fb_mean_y": round(float(fb_mean[1]), 3),
        "xy_diff_m": round(float(np.linalg.norm(tri_mid[:2] - fb_mean)), 3),
        "fb_gap_m": round(float(np.linalg.norm(fb_i - fb_j)), 3),
        "dist_to_cam_i_m": round(float(np.linalg.norm(tri_mid[:2] - poses[cam_i].t[:2])), 2),
        "dist_to_cam_j_m": round(float(np.linalg.norm(tri_mid[:2] - poses[cam_j].t[:2])), 2),
    }


def summarize(rows, title):
    lines = [f"=== {title} (n={len(rows)}) ==="]
    if not rows:
        return lines + ["  (なし)"]

    def stat(sub):
        if not sub:
            return None
        h = np.array([r["height_m"] for r in sub])
        he = np.array([r["height_err_m"] for r in sub])
        xd = np.array([r["xy_diff_m"] for r in sub])
        fg = np.array([r["fb_gap_m"] for r in sub])
        return (len(sub), np.median(h), np.percentile(h, 25), np.percentile(h, 75),
                np.median(he), np.median(np.abs(he)),
                np.median(xd), np.percentile(xd, 90), np.median(fg))

    lines.append("  交差角ビン | n | 高さ中央値[25-75%] | 高さ誤差(符号付)中央値 | |高さ誤差|中央値 | xy差 中央値/90%ile | fb_gap中央値")
    for lo, hi in zip(ANGLE_BINS_DEG[:-1], ANGLE_BINS_DEG[1:]):
        sub = [r for r in rows if lo <= r["angle_deg"] < hi]
        s = stat(sub)
        label = f"{lo}-{hi if hi <= 180 else 180}deg"
        if s is None:
            lines.append(f"  {label:>10} | 0")
            continue
        n, hm, h25, h75, he, hae, xm, x90, fg = s
        lines.append(f"  {label:>10} | {n} | {hm:.2f}[{h25:.2f}-{h75:.2f}] | {he:+.2f} | {hae:.2f} | {xm:.2f}/{x90:.2f} | {fg:.2f}")
    return lines


def main():
    from ultralytics import YOLO

    cameras_json = ep.load_floorplan_cameras(ep.FLOORPLAN_JSON_PATH)
    cameras_json = ep.apply_camera_overrides(cameras_json)
    poses = ep.resolve_poses(cameras_json, root="Area1", fmt=ep.ROT_FMT)

    print(f"YOLOモデル読み込み中: {ep.YOLO_WEIGHTS_PATH}")
    yolo_model = YOLO(ep.YOLO_WEIGHTS_PATH)

    image_dir = Path(ep.IMAGE_DIR)
    rows = []
    n_frames = 0

    for frame_idx in range(ep.FRAME_START, ep.FRAME_END + 1, ep.FRAME_STEP):
        image_path = image_dir / f"{ep.frame_name(frame_idx)}.jpg"
        if not image_path.exists():
            continue
        per_camera = ep.deduplicate_per_camera(
            ep.yolo_detections_per_camera(yolo_model, image_path, poses), ep.DEDUP_IOU_THRESHOLD
        )
        accepted, tri = ep.match_epipolar(
            per_camera, poses, ep.ADJACENT_PAIRS, ep.RAY_DIST_THR_M,
            ep.HEIGHT_RANGE_Z, ep.USE_POSE_CONSTRAINT, ep.MARGIN_GAP_M,
        )
        for key in accepted:
            rows.append(pair_metrics(frame_idx, key, tri[key], per_camera, poses))
        n_frames += 1
        print(f"  {ep.frame_name(frame_idx)}: 採用ペア={len(accepted)}")

    if not rows:
        print("採用ペアが0件でした。")
        return

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    lines = [f"処理フレーム数={n_frames} / 採用ペア総数={len(rows)}", ""]
    lines += summarize(rows, "全体")
    for pose in ("lying", "standing"):
        lines.append("")
        lines += summarize([r for r in rows if r["pose"] == pose], f"姿勢={pose}")
    for cam_i, cam_j in ep.ADJACENT_PAIRS:
        lines.append("")
        lines += summarize([r for r in rows if r["cam_i"] == cam_i and r["cam_j"] == cam_j],
                           f"カメラペア={cam_i}-{cam_j}")
    text = "\n".join(lines)
    print("\n" + text)
    OUT_TXT.write_text(text, encoding="utf-8")
    print(f"\nCSV保存: {OUT_CSV}\n集計保存: {OUT_TXT}")


if __name__ == "__main__":
    main()