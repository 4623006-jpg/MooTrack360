"""
phase2_tracker_threshold_gridsearch.py

【vml36で実行】

phase2_yolo_conftrack_check.pyでMOTA/IDRが低かった原因は、tracker_configs.yaml
の track_new_thr / track_confirm_thr (共に0.86) が、実際のYOLOモデルの信頼度分布
(Camera_1タイル推論で平均0.718)に対して高すぎ、GT44〜45頭のうち半分程度しか
新規トラック化されていなかったこと(yolo_detection_debug_viz.pyで確認済み)。

このスクリプトは track_new_thr と track_confirm_thr の組み合わせをグリッドサーチし、
IDF1が最大になる値を探す。

YOLO推論はコストが高いため、30フレーム分の検出結果とGTを先に1回だけ計算して
キャッシュし、閾値の組み合わせごとにはConfTrackTrackerだけを作り直して評価する
(検出自体は使い回す)。tracker_configs.yamlの元ファイルは書き換えず、組み合わせ
ごとに一時yamlファイルを作ってConfTrackTrackerに渡す。

--- 使い方 ---
    python phase2_tracker_threshold_gridsearch.py
(パス・探索範囲は全て下の設定セクションに直書きしてあるので、引数なしで実行する)
"""

import os
import re
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import yaml

# ---------------------------------------------------------------------------
# 設定(phase2_yolo_conftrack_check.pyと同じ)
# ---------------------------------------------------------------------------
XML_DIR = "/mnt/vmlqnap02/dataset/MooTrack360/1-hour-farm1-area1-sample-sequence/xml-1fps"
IMAGE_DIR = "/mnt/vmlqnap02/dataset/MooTrack360/1-hour-farm1-area1-sample-sequence/1-hour-farm1-area1-frames"
OUTPUT_DIR = Path(
    "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/research/idf1_check"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

YOLO_WEIGHTS_PATH = "/mnt/vmlqnap02/dataset/MooTrack360/Trained models/yolo11x/train_ultralytics/weights/best.pt"
YOLO_CONF_THRESH = 0.1

MOOTRACK_CODE_SRC = "/home/ara/myproject/project2/MooTrack360_source_code/MooTrack360_code/src"
BASE_TRACKER_CONFIG_PATH = f"{MOOTRACK_CODE_SRC}/configs/tracker_configs.yaml"

FRAME_START = 0
FRAME_END = 290
FRAME_STEP = 10

IMAGE_WIDTH_PX = 8832
NUM_CAMERAS = 3
CAM_IDS = ["Camera_1", "Camera_2", "Camera_3"]
TARGET_CAMERA = "Camera_1"

DIRTY_NAMES = {"cow024", "cow027"}

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# グリッドサーチ範囲(YOLOの信頼度分布: 平均0.718, 0.86以上が58件中23件だった)
# ---------------------------------------------------------------------------
TRACK_NEW_THR_CANDIDATES = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.86]
TRACK_CONFIRM_THR_CANDIDATES = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.86]

TOP_N_RESULTS = 15


# ---------------------------------------------------------------------------
# ConfTrackTrackerのimport
# ---------------------------------------------------------------------------
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


def load_frame_data() -> List[dict]:
    """YOLO推論とGT読み込みを1回だけ行い、フレームごとの検出・GT・画像をキャッシュする"""
    from ultralytics import YOLO

    print(f"YOLOモデル読み込み中: {YOLO_WEIGHTS_PATH}")
    yolo_model = YOLO(YOLO_WEIGHTS_PATH)

    xml_dir = Path(XML_DIR)
    image_dir = Path(IMAGE_DIR)
    cam_idx = CAM_IDS.index(TARGET_CAMERA)
    tile_w = IMAGE_WIDTH_PX / NUM_CAMERAS

    frame_indices = list(range(FRAME_START, FRAME_END + 1, FRAME_STEP))
    cached = []

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
        boxes = [
            (*box.xyxy[0].tolist(), float(box.conf[0]), int(box.cls[0]))
            for box in results.boxes
        ]

        gt_objects = parse_voc_annotation(xml_path)
        gt_ids = []
        gt_boxes_xywh = []
        for o in gt_objects:
            cx_full = (o["xmin"] + o["xmax"]) / 2.0
            if assign_camera(cx_full, IMAGE_WIDTH_PX) != TARGET_CAMERA:
                continue
            if o["name"] in DIRTY_NAMES:
                continue
            gt_ids.append(name_to_gt_id(o["name"]))
            local_xmin = o["xmin"] - cam_idx * tile_w
            local_xmax = o["xmax"] - cam_idx * tile_w
            gt_boxes_xywh.append(xyxy_to_xywh(local_xmin, o["ymin"], local_xmax, o["ymax"]))

        cached.append({
            "frame_idx": frame_idx,
            "boxes": boxes,
            "frame_img": frame_img,
            "gt_ids": gt_ids,
            "gt_boxes_xywh": gt_boxes_xywh,
        })
        print(f"  {frame_name(frame_idx)}: YOLO検出={len(boxes)}件 GT={len(gt_ids)}件 キャッシュ完了")

    return cached


def evaluate_one_config(cached_frames: List[dict], track_new_thr: float, track_confirm_thr: float) -> dict:
    import motmetrics as mm

    with open(BASE_TRACKER_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    cfg["conftrack"]["track_new_thr"] = track_new_thr
    cfg["conftrack"]["track_confirm_thr"] = track_confirm_thr

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, dir=OUTPUT_DIR
        ) as tmp:
            yaml.safe_dump(cfg, tmp)
            tmp_path = tmp.name

        tracker = ConfTrackTracker(tracker_params_path=tmp_path, device=DEVICE)
        acc = mm.MOTAccumulator(auto_id=True)

        for entry in cached_frames:
            detections = [Detection(*b) for b in entry["boxes"]]
            tracks = tracker.update(detections, frame=entry["frame_img"])
            hyp_ids = [t.track_id for t in tracks]
            hyp_boxes_xywh = [xyxy_to_xywh(t.x1, t.y1, t.x2, t.y2) for t in tracks]
            dist = mm.distances.iou_matrix(entry["gt_boxes_xywh"], hyp_boxes_xywh, max_iou=0.5)
            acc.update(entry["gt_ids"], hyp_ids, dist)

        mh = mm.metrics.create()
        summary = mh.compute(
            acc, metrics=["idf1", "idp", "idr", "mota", "num_switches"], name="x"
        )
        row = summary.iloc[0]
        return {
            "track_new_thr": track_new_thr,
            "track_confirm_thr": track_confirm_thr,
            "idf1": float(row["idf1"]),
            "idp": float(row["idp"]),
            "idr": float(row["idr"]),
            "mota": float(row["mota"]),
            "num_switches": int(row["num_switches"]),
        }
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def main():
    print("=== フレームデータのキャッシュ(YOLO推論・GT読み込みを1回だけ実行) ===")
    cached_frames = load_frame_data()
    print(f"\n{len(cached_frames)}フレームをキャッシュしました\n")

    total = len(TRACK_NEW_THR_CANDIDATES) * len(TRACK_CONFIRM_THR_CANDIDATES)
    print(f"探索する組み合わせ数: {total}\n")

    results = []
    done = 0
    for new_thr in TRACK_NEW_THR_CANDIDATES:
        for confirm_thr in TRACK_CONFIRM_THR_CANDIDATES:
            r = evaluate_one_config(cached_frames, new_thr, confirm_thr)
            results.append(r)
            done += 1
            print(f"  [{done}/{total}] track_new_thr={new_thr:.2f} track_confirm_thr={confirm_thr:.2f} "
                  f"-> IDF1={r['idf1']:.3f} (IDP={r['idp']:.3f}, IDR={r['idr']:.3f}, "
                  f"MOTA={r['mota']:.3f}, IDs={r['num_switches']})")

    results.sort(key=lambda r: -r["idf1"])

    print(f"\n=== IDF1上位{TOP_N_RESULTS}件 ===")
    print(f"{'track_new_thr':>14} | {'track_confirm_thr':>18} | {'IDF1':>8} | {'IDP':>8} | {'IDR':>8} | {'MOTA':>8} | {'IDs':>4}")
    for r in results[:TOP_N_RESULTS]:
        print(f"{r['track_new_thr']:>14.2f} | {r['track_confirm_thr']:>18.2f} | "
              f"{r['idf1']:>8.3f} | {r['idp']:>8.3f} | {r['idr']:>8.3f} | "
              f"{r['mota']:>8.3f} | {r['num_switches']:>4}")

    best = results[0]
    print(f"\n最良の組み合わせ: track_new_thr={best['track_new_thr']}, "
          f"track_confirm_thr={best['track_confirm_thr']} -> IDF1={best['idf1']:.3f}")
    print("(参考: 元のtracker_configs.yaml値 track_new_thr=0.86, track_confirm_thr=0.86 -> IDF1=0.687)")

    import csv
    csv_path = OUTPUT_DIR / "phase2_threshold_gridsearch_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV保存: {csv_path}")


if __name__ == "__main__":
    main()