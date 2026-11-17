"""
render_frames_gt_vs_yolo_he_kai_ver1.py

指定したフレームだけ、カメラごとに「GTのBBOX(左)」と「YOLOの検出BBOX(右)」を並べた1枚の画像を出す。
FN/FP の原因を実画像で確認するためのもの(diagnose_fn_fp_he_kai_ver1.py で見つけたフレームを指定する)。

  左 : GTのBBOX(名前・姿勢)。YOLOと重ならない(IoU<IOU_MATCH)GTは赤太線で "FN?" と表示
  右 : YOLOのBBOX(姿勢で色分け)。段階のタグ付き
        太線(タグなし) : 最後まで残った検出
        細線 (重複除去) : IoU重複除去で消えた
        細線 (範囲外)   : カメラ有効範囲(曲線ゾーン)で除外された
        細線 (牛舎外)   : 牛舎範囲外フィルタで除外された
       残った検出のうちGTと重ならないものはマゼンタで "FP?" と表示
  どちらの有効範囲/想定高さを使うかは、phase3_he_kai_matching_ver2.py のモード(HE_MODE)で決める。

使い方: python render_frames_gt_vs_yolo_he_kai_ver1.py 620 560 1320
        HE_MODE=euclid python render_frames_gt_vs_yolo_he_kai_ver1.py 620     # 既存の有効範囲で見る
環境変数: HE_MODE (既定 maha_h_valid) / HE_IMG_SCALE (既定 0.5: 出力の縮小率) / HE_IOU_MATCH (既定 0.3)
          HE_SIGMA_JSON / HE_SIGMA_MODEL / HE_EPS_M / HE_K_SIGMA (ver2 と同じ)
出力: <OUTPUT_DIR>/frames_gt_vs_yolo_he_kai/{frame}_{cam_id}_<モード>.jpg
配置: research/phase4/he_kai/ (render_bev_frames_yolo_zonepose_ver1.py と curved_zone_utils_ver1.py が import できること)
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np

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
    CAM_IDS, ROT_FMT, apply_camera_overrides, filter_outside_floorplan_bounds, load_floorplan, resolve_poses,
)

MODE = os.environ.get("HE_MODE", "maha_h_valid")
IMG_SCALE = float(os.environ.get("HE_IMG_SCALE", 0.5))
IOU_MATCH = float(os.environ.get("HE_IOU_MATCH", 0.3))
RED, MAGENTA = (0, 0, 255), (255, 0, 255)


def put_label(img, text, x, y, color, scale=0.6):
    cv2.putText(img, text, (x, max(14, y)), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def main():
    frames = sorted({int(a) for a in sys.argv[1:]})
    if not frames:
        raise SystemExit("フレーム番号を指定してください。例: python render_frames_gt_vs_yolo_he_kai_ver1.py 620 560")
    if MODE not in hk.MODES:
        raise SystemExit(f"不明なモード: {MODE} / 使えるモード: {list(hk.MODES)}")
    cfg = hk.MODES[MODE]
    from ultralytics import YOLO

    regions, cameras_json, real_w, real_h = load_floorplan(rf.FLOORPLAN_JSON_PATH)
    poses = resolve_poses(apply_camera_overrides(cameras_json), root="Area1", fmt=ROT_FMT)
    json_path = os.environ.get("HE_SIGMA_JSON") or str(Path(rf.OUTPUT_DIR) / "bev_calibration_refined" / "sigma_he_kai_ver1.json")
    model = load_model(json_path)
    orig_heights = (rf.STANDING_HEIGHT_M, rf.LYING_HEIGHT_M)
    if cfg["heights"] == "fit":
        rf.STANDING_HEIGHT_M, rf.LYING_HEIGHT_M = model.h_eff("standing"), model.h_eff("lying")
    print(f"モード {MODE} {cfg} / 想定高さ standing={rf.STANDING_HEIGHT_M:.3f} lying={rf.LYING_HEIGHT_M:.3f}")
    if cfg["polygons"] == "he_kai":
        polygons = build_valid_polygons(poses, CAM_IDS, get_intrinsics_fn(), model, hk.EPS_VALID_M, hk.K_SIGMA,
                                        height_m=ZONE_HEIGHT_M, n_angles=N_ANGLES)
    else:
        polygons = build_camera_polygons(poses, CAM_IDS, get_intrinsics_fn(), height_m=ZONE_HEIGHT_M, n_angles=N_ANGLES)

    yolo_model = YOLO(rf.YOLO_WEIGHTS_PATH)
    out_dir = Path(rf.OUTPUT_DIR) / "frames_gt_vs_yolo_he_kai"
    out_dir.mkdir(parents=True, exist_ok=True)
    tile_w = rf.IMAGE_WIDTH_PX / rf.NUM_CAMERAS
    n_saved = 0

    for frame_idx in frames:
        name = rf.frame_name(frame_idx)
        xml_path, image_path = Path(rf.XML_DIR) / f"{name}.xml", Path(rf.IMAGE_DIR) / f"{name}.jpg"
        if not image_path.exists():
            print(f"  {name}: 画像が無いのでスキップ")
            continue
        full = cv2.imread(str(image_path))
        gt_per_cam = (rf.voc_objects_local_by_camera(rf.parse_voc_annotation(xml_path), rf.IMAGE_WIDTH_PX)
                      if xml_path.exists() else {})
        raw = rf.yolo_world_points_per_camera_with_pose(yolo_model, image_path, poses)
        deduped = rf.deduplicate_per_camera(raw, rf.DEDUP_IOU_THRESHOLD)
        zone = filter_by_curved_zone(deduped, polygons)
        bounds = filter_outside_floorplan_bounds(zone, real_w, real_h, margin_m=0.0, verbose=False)

        for ci, cid in enumerate(CAM_IDS):
            crop = full[:, int(ci * tile_w):int((ci + 1) * tile_w)]
            left, right = crop.copy(), crop.copy()
            gts = gt_per_cam.get(cid, [])
            ylist = raw.get(cid, [])
            ids_dedup = {id(it) for it in deduped.get(cid, [])}
            ids_zone = {id(it) for it in zone.get(cid, [])}
            ids_kept = {id(it) for it in bounds.get(cid, [])}

            n_fn = n_fp = 0
            for g in gts:                                            # 左: GT
                box = [int(round(v)) for v in g["bbox_local"]]
                miss = not any(rf.iou_xyxy(tuple(g["bbox_local"]), tuple(it["bbox_local"])) >= IOU_MATCH for it in ylist)
                col = RED if miss else rf.GT_BOX_COLOR_BGR
                cv2.rectangle(left, (box[0], box[1]), (box[2], box[3]), col, 4 if miss else 2, cv2.LINE_AA)
                put_label(left, f"{g['name']}" + (f"({g['pose']})" if g["pose"] else "") + (" FN?" if miss else ""),
                          box[0], box[1] - 6, col)
                n_fn += int(miss)
            for it in ylist:                                         # 右: YOLO
                box = [int(round(v)) for v in it["bbox_local"]]
                col = rf.POSE_BOX_COLORS_BGR.get(it["pose"], (180, 180, 180))
                if id(it) not in ids_dedup:
                    thick, tag = 1, "(重複除去)"
                elif id(it) not in ids_zone:
                    thick, tag = 1, "(範囲外)"
                elif id(it) not in ids_kept:
                    thick, tag = 1, "(牛舎外)"
                else:
                    thick, tag = 3, ""
                    if not any(rf.iou_xyxy(tuple(g["bbox_local"]), tuple(it["bbox_local"])) >= IOU_MATCH for g in gts):
                        col, tag = MAGENTA, " FP?"
                        n_fp += 1
                cv2.rectangle(right, (box[0], box[1]), (box[2], box[3]), col, thick, cv2.LINE_AA)
                conf = it.get("conf")
                put_label(right, f"{it['pose']} {conf:.2f}{tag}" if conf is not None else f"{it['pose']}{tag}",
                          box[0], box[1] - 6, col)
            put_label(left, f"{name} {cid} GT n={len(gts)} (FN? {n_fn})", 10, 34, rf.GT_BOX_COLOR_BGR, 0.9)
            put_label(right, f"YOLO n={len(ylist)} kept={len(ids_kept)} (FP? {n_fp}) [{MODE}]", 10, 34, (255, 255, 255), 0.9)

            out = np.hstack([left, right])
            if IMG_SCALE != 1.0:
                out = cv2.resize(out, None, fx=IMG_SCALE, fy=IMG_SCALE, interpolation=cv2.INTER_AREA)
            path = out_dir / f"{name}_{cid}_{MODE}.jpg"
            cv2.imwrite(str(path), out, [cv2.IMWRITE_JPEG_QUALITY, 90])
            n_saved += 1
            print(f"  {name} {cid}: GT {len(gts)} / YOLO {len(ylist)} (残った {len(ids_kept)}) / FN? {n_fn} FP? {n_fp}")
    rf.STANDING_HEIGHT_M, rf.LYING_HEIGHT_M = orig_heights
    print(f"\n{n_saved}枚を保存しました: {out_dir}")
    print("※ FN?/FP? はBBOXの重なり(IoU)だけの目安。BEV上の判定(2m以内の対応付け)とは一致しないことがある。")


def get_intrinsics_fn():
    from Mootrack_core import get_intrinsics
    return get_intrinsics


if __name__ == "__main__":
    main()