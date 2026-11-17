"""
render_yolo_bbox_images_he_kai_ver1.py

指定したフレームの「YOLOのBBOX画像」だけを保存する(BEV画像・融合・追跡は行わない)。
render_bev_frames_yolo_zonepose_ver1.py の save_annotated_camera_images をそのまま使うので、描き方は同じ:
    姿勢で色分け / 太線 = 最後まで残った検出 / 細線 = 消えた検出 (重複除去) (範囲外除外)
どの有効範囲(曲線ゾーン)・想定高さで「消えた/残った」を判定するかは、phase3_he_kai_matching_ver2.py の
モード(HE_MODE)で決める。

使い方: python render_yolo_bbox_images_he_kai_ver1.py 620 560 1320
        HE_MODE=euclid python render_yolo_bbox_images_he_kai_ver1.py 620     # 既存の有効範囲で見る
環境変数: HE_MODE (既定 maha_h_valid) / HE_SIGMA_JSON / HE_SIGMA_MODEL / HE_EPS_M / HE_K_SIGMA (ver2 と同じ)
出力: <OUTPUT_DIR>/yolo_bbox_frames_he_kai_<モード>/{frame}_{cam_id}.jpg
配置: research/phase4/he_kai/ (render_bev_frames_yolo_zonepose_ver1.py と curved_zone_utils_ver1.py が import できること)
"""

import os
import sys
from pathlib import Path

import cv2

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
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

import render_bev_frames_yolo_zonepose_ver1 as rf
import phase3_he_kai_matching_ver2 as hk
from curved_zone_utils_ver1 import N_ANGLES, ZONE_HEIGHT_M, build_camera_polygons, filter_by_curved_zone
from he_kai_sigma_model_ver1 import build_valid_polygons, load_model
from Mootrack_core import (
    CAM_IDS, ROT_FMT, apply_camera_overrides, filter_outside_floorplan_bounds, get_intrinsics, load_floorplan, resolve_poses,
)

MODE = os.environ.get("HE_MODE", "maha_h_valid")


def save_annotated_with_conf(frame_idx, image_path, raw, deduped, filtered, out_dir):
    """rf.save_annotated_camera_images と同じ描き方 + ラベルにYOLOの信頼度(conf)を表示する"""
    full = cv2.imread(str(image_path))
    if full is None:
        return []
    tile_w = rf.IMAGE_WIDTH_PX / rf.NUM_CAMERAS
    saved = []
    for ci, cid in enumerate(CAM_IDS):
        crop = full[:, int(ci * tile_w):int((ci + 1) * tile_w)].copy()
        ids_dedup = {id(it) for it in deduped.get(cid, [])}
        ids_kept = {id(it) for it in filtered.get(cid, [])}
        for it in raw.get(cid, []):
            x1, y1, x2, y2 = [int(round(v)) for v in it["bbox_local"]]
            color = rf.POSE_BOX_COLORS_BGR.get(it["pose"], (180, 180, 180))
            if id(it) not in ids_dedup:
                thick, tag = 1, "(重複除去)"
            elif id(it) not in ids_kept:
                thick, tag = 1, "(範囲外除外)"
            else:
                thick, tag = 2, ""
            cv2.rectangle(crop, (x1, y1), (x2, y2), color, thick, lineType=cv2.LINE_AA)
            conf = it.get("conf")
            label = f"{it['pose']} {conf:.2f}{tag}" if conf is not None else f"{it['pose']}{tag}"
            cv2.putText(crop, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        path = Path(out_dir) / f"{rf.frame_name(frame_idx)}_{cid}.jpg"
        cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        saved.append(path)
    return saved


def main():
    frames = sorted({int(a) for a in sys.argv[1:]})
    if not frames:
        raise SystemExit("フレーム番号を指定してください。例: python render_yolo_bbox_images_he_kai_ver1.py 620 560")
    if MODE not in hk.MODES:
        raise SystemExit(f"不明なモード: {MODE} / 使えるモード: {list(hk.MODES)}")
    cfg = hk.MODES[MODE]
    from ultralytics import YOLO

    regions, cameras_json, real_w, real_h = load_floorplan(rf.FLOORPLAN_JSON_PATH)
    poses = resolve_poses(apply_camera_overrides(cameras_json), root="Area1", fmt=ROT_FMT)

    orig_heights = (rf.STANDING_HEIGHT_M, rf.LYING_HEIGHT_M)
    if cfg["heights"] == "fit" or cfg["polygons"] == "he_kai":
        json_path = os.environ.get("HE_SIGMA_JSON") or str(Path(rf.OUTPUT_DIR) / "bev_calibration_refined" / "sigma_he_kai_ver4.json")
        model = load_model(json_path)
        if cfg["heights"] == "fit":
            rf.STANDING_HEIGHT_M, rf.LYING_HEIGHT_M = model.h_eff("standing"), model.h_eff("lying")
    print(f"モード {MODE} {cfg} / 想定高さ standing={rf.STANDING_HEIGHT_M:.3f} lying={rf.LYING_HEIGHT_M:.3f}")
    if cfg["polygons"] == "he_kai":
        polygons = build_valid_polygons(poses, CAM_IDS, get_intrinsics, model, hk.EPS_VALID_M, hk.K_SIGMA,
                                        height_m=ZONE_HEIGHT_M, n_angles=N_ANGLES)
    else:
        polygons = build_camera_polygons(poses, CAM_IDS, get_intrinsics, height_m=ZONE_HEIGHT_M, n_angles=N_ANGLES)

    out_dir = Path(rf.OUTPUT_DIR) / f"yolo_bbox_frames_he_kai_{MODE}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rf.ANNOTATED_OUT_DIR = out_dir          # save_annotated_camera_images の保存先をここに向ける

    yolo_model = YOLO(rf.YOLO_WEIGHTS_PATH)
    n = 0
    for frame_idx in frames:
        image_path = Path(rf.IMAGE_DIR) / f"{rf.frame_name(frame_idx)}.jpg"
        if not image_path.exists():
            print(f"  {rf.frame_name(frame_idx)}: 画像が無いのでスキップ")
            continue
        raw = rf.yolo_world_points_per_camera_with_pose(yolo_model, image_path, poses)
        deduped = rf.deduplicate_per_camera(raw, rf.DEDUP_IOU_THRESHOLD)
        filtered = filter_by_curved_zone(deduped, polygons)
        filtered = filter_outside_floorplan_bounds(filtered, real_w, real_h, margin_m=0.0, verbose=False)
        saved = save_annotated_with_conf(frame_idx, image_path, raw, deduped, filtered, out_dir)
        n += len(saved)
        print(f"  {rf.frame_name(frame_idx)}: {len(saved)}枚保存 (YOLO {sum(len(v) for v in raw.values())} -> 残った {sum(len(v) for v in filtered.values())})")
    rf.STANDING_HEIGHT_M, rf.LYING_HEIGHT_M = orig_heights
    print(f"\n{n}枚を保存しました: {out_dir}")


if __name__ == "__main__":
    main()