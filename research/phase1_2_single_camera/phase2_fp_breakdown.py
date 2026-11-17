"""
phase2_fp_breakdown.py

【vml36で実行】

Tentativeトラックの猶予パッチ適用後、FN=195→54まで改善した一方でFP=62→335と
大幅に増えた。増えた273件のFPが「ノイズ」なのか「同じ物理個体に対する重複ID」
なのかを切り分けるための診断スクリプト。

motmetricsのacc.events(Type=='FP'の行)から、FPと判定されたHId(トラックの
track_id)を集計し、そのフレームでの位置(bbox中心)と、最も近いGT個体との距離を
調べる。もしFPの多くが「GTのすぐそば(数十px以内)」に集中していれば、重複ID
(同じ牛を2つ以上のtrack_idで別々に追っている)が濃厚。逆に、GTから遠い場所に
散らばっていれば、単純なノイズ検出(背景・他物体の誤検出)が濃厚。

phase2_yolo_conftrack_check.pyと同じパイプライン(YOLO+ConfTrackTracker、
Camera_1タイル、30フレーム)をそのまま流用し、評価ロジックだけFP側の詳細集計に
差し替えている。

--- 使い方 ---
    python phase2_fp_breakdown.py
"""

import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import yaml

XML_DIR = "/mnt/vmlqnap02/dataset/MooTrack360/1-hour-farm1-area1-sample-sequence/xml-1fps"
IMAGE_DIR = "/mnt/vmlqnap02/dataset/MooTrack360/1-hour-farm1-area1-sample-sequence/1-hour-farm1-area1-frames"
OUTPUT_DIR = Path(
    "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/research/idf1_check"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

YOLO_WEIGHTS_PATH = "/mnt/vmlqnap02/dataset/MooTrack360/Trained models/yolo11x/train_ultralytics/weights/best.pt"
YOLO_CONF_THRESH = 0.1

MOOTRACK_CODE_SRC = "/home/ara/myproject/project2/MooTrack360_source_code/MooTrack360_code/src"
TRACKER_CONFIG_PATH = f"{MOOTRACK_CODE_SRC}/configs/tracker_configs.yaml"

FRAME_START = 0
FRAME_END = 290
FRAME_STEP = 10

IMAGE_WIDTH_PX = 8832
NUM_CAMERAS = 3
CAM_IDS = ["Camera_1", "Camera_2", "Camera_3"]
TARGET_CAMERA = "Camera_1"

DIRTY_NAMES = {"cow024", "cow027"}

# GTとの距離がこれ未満なら「重複ID(同じ牛を別トラックで追っている)」とみなす
DUPLICATE_DIST_THRESH_PX = 80.0

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if MOOTRACK_CODE_SRC not in sys.path:
    sys.path.insert(0, MOOTRACK_CODE_SRC)

from Trackers.conftrack_tracker import ConfTrackTracker  # noqa: E402


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int


def frame_name(idx: int) -> str:
    return f"{idx:06d}"


def parse_voc_annotation(xml_path) -> List[dict]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        bnd = obj.find("bndbox")
        objects.append({
            "name": name,
            "xmin": float(bnd.find("xmin").text),
            "ymin": float(bnd.find("ymin").text),
            "xmax": float(bnd.find("xmax").text),
            "ymax": float(bnd.find("ymax").text),
        })
    return objects


def assign_camera(center_x_full: float, image_width: float) -> str:
    tile_w = image_width / NUM_CAMERAS
    cam_idx = int(center_x_full // tile_w)
    cam_idx = max(0, min(cam_idx, NUM_CAMERAS - 1))
    return CAM_IDS[cam_idx]


def xyxy_to_xywh(x1, y1, x2, y2) -> Tuple[float, float, float, float]:
    return x1, y1, x2 - x1, y2 - y1


def name_to_gt_id(name: str) -> int:
    m = re.search(r"\d+", name)
    if not m:
        raise ValueError(f"nameから数値IDを抽出できません: {name}")
    return int(m.group())


def main():
    import motmetrics as mm
    from ultralytics import YOLO

    xml_dir = Path(XML_DIR)
    image_dir = Path(IMAGE_DIR)
    cam_idx = CAM_IDS.index(TARGET_CAMERA)
    tile_w = IMAGE_WIDTH_PX / NUM_CAMERAS

    print(f"YOLOモデル読み込み中: {YOLO_WEIGHTS_PATH}")
    yolo_model = YOLO(YOLO_WEIGHTS_PATH)

    print(f"ConfTrackTracker初期化中(device={DEVICE})...")
    tracker = ConfTrackTracker(tracker_params_path=TRACKER_CONFIG_PATH, device=DEVICE)

    acc = mm.MOTAccumulator(auto_id=True)

    # フレームごとのGT位置・トラック出力位置を、後でFP行の詳細を引くために保存しておく
    frame_records: List[dict] = []

    frame_indices = list(range(FRAME_START, FRAME_END + 1, FRAME_STEP))

    for frame_idx in frame_indices:
        xml_path = xml_dir / f"{frame_name(frame_idx)}.xml"
        image_path = image_dir / f"{frame_name(frame_idx)}.jpg"
        if not xml_path.exists() or not image_path.exists():
            continue

        full_img = cv2.imread(str(image_path))
        if full_img is None:
            continue

        x0 = int(cam_idx * tile_w)
        x1_edge = int((cam_idx + 1) * tile_w)
        frame_img = full_img[:, x0:x1_edge]

        results = yolo_model.predict(
            frame_img, conf=YOLO_CONF_THRESH, device=DEVICE, verbose=False
        )[0]
        detections = [
            Detection(*box.xyxy[0].tolist(), float(box.conf[0]), int(box.cls[0]))
            for box in results.boxes
        ]
        tracks = tracker.update(detections, frame=frame_img)

        gt_objects = parse_voc_annotation(xml_path)
        gt_ids = []
        gt_boxes_xywh = []
        gt_names = []
        for o in gt_objects:
            cx_full = (o["xmin"] + o["xmax"]) / 2.0
            if assign_camera(cx_full, IMAGE_WIDTH_PX) != TARGET_CAMERA:
                continue
            if o["name"] in DIRTY_NAMES:
                continue
            gt_id = name_to_gt_id(o["name"])
            gt_ids.append(gt_id)
            gt_names.append(o["name"])
            local_xmin = o["xmin"] - cam_idx * tile_w
            local_xmax = o["xmax"] - cam_idx * tile_w
            gt_boxes_xywh.append(xyxy_to_xywh(local_xmin, o["ymin"], local_xmax, o["ymax"]))

        hyp_ids = [t.track_id for t in tracks]
        hyp_boxes_xywh = [xyxy_to_xywh(t.x1, t.y1, t.x2, t.y2) for t in tracks]

        dist = mm.distances.iou_matrix(gt_boxes_xywh, hyp_boxes_xywh, max_iou=0.5)
        acc.update(gt_ids, hyp_ids, dist)

        # このフレームのGT中心・hyp中心を保存(後でFP行の位置照合に使う)
        gt_centers = {
            gt_id: ((gx + gw / 2.0), (gy + gh / 2.0))
            for gt_id, (gx, gy, gw, gh) in zip(gt_ids, gt_boxes_xywh)
        }
        hyp_centers = {
            hid: ((hx + hw / 2.0), (hy + hh / 2.0))
            for hid, (hx, hy, hw, hh) in zip(hyp_ids, hyp_boxes_xywh)
        }
        frame_records.append({
            "frame_idx": frame_idx,
            "gt_centers": gt_centers,
            "hyp_centers": hyp_centers,
        })

    mh = mm.metrics.create()
    summary = mh.compute(
        acc, metrics=["idf1", "idp", "idr", "mota", "num_false_positives"], name="x"
    )
    print(f"\n全体: IDF1={summary['idf1'].iloc[0]:.3f} FP={int(summary['num_false_positives'].iloc[0])}")

    # --- FP行の詳細分析 ---
    events = acc.events
    fp_events = events[events["Type"] == "FP"]

    # frame_idxとacc内部のFrameId(0,1,2,...という連番)を対応させる
    frame_records_by_ordinal = {i: rec for i, rec in enumerate(frame_records)}

    dup_count = 0
    noise_count = 0
    rows = []

    for frame_ordinal, row in fp_events.groupby(level=0):
        rec = frame_records_by_ordinal.get(frame_ordinal)
        if rec is None:
            continue
        for _, ev in row.iterrows():
            hid = ev["HId"]
            if hid != hid:  # NaN
                continue
            hid = int(hid)
            hyp_center = rec["hyp_centers"].get(hid)
            if hyp_center is None:
                continue
            hx, hy = hyp_center

            best_gt = None
            best_dist = None
            for gt_id, (gx, gy) in rec["gt_centers"].items():
                d = float(np.hypot(hx - gx, hy - gy))
                if best_dist is None or d < best_dist:
                    best_dist = d
                    best_gt = gt_id

            is_dup = best_dist is not None and best_dist < DUPLICATE_DIST_THRESH_PX
            if is_dup:
                dup_count += 1
            else:
                noise_count += 1

            rows.append({
                "frame": rec["frame_idx"], "hyp_track_id": hid,
                "hyp_xy": (round(hx, 1), round(hy, 1)),
                "nearest_gt": best_gt, "dist_to_nearest_gt": round(best_dist, 1) if best_dist else None,
                "type": "重複ID疑い" if is_dup else "ノイズ疑い",
            })

    print(f"\nFP={len(rows)}件中: 重複ID疑い(最近傍GTから{DUPLICATE_DIST_THRESH_PX}px未満)={dup_count}件 / "
          f"ノイズ疑い(GTから離れている)={noise_count}件")

    print(f"\n{'frame':>8}{'hyp_id':>10}{'hyp_xy':>20}{'最寄GT':>8}{'距離':>8}  種別")
    for r in rows[:60]:
        print(f"{frame_name(r['frame']):>8}{r['hyp_track_id']:>10}{str(r['hyp_xy']):>20}"
              f"{r['nearest_gt']:>8}{str(r['dist_to_nearest_gt']):>8}  {r['type']}")
    if len(rows) > 60:
        print(f"... 以下省略({len(rows) - 60}件)")

    import csv
    csv_path = OUTPUT_DIR / "phase2_fp_breakdown.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "hyp_track_id", "hyp_xy", "nearest_gt", "dist_to_nearest_gt", "type"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV保存: {csv_path}")


if __name__ == "__main__":
    main()