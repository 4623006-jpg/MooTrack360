"""
phase2_fn_breakdown.py

【vml36で実行】

phase2_yolo_conftrack_check.py で確認したベスト設定(track_new_thr=0.8,
track_confirm_thr=0.5)でIDF1=86.4%を達成したが、残りの誤差はほぼ全てFN
(見逃し=278件)由来だった(IDP=96.0%・IDs=8はPhase1の上限に匹敵する水準)。

このスクリプトは、そのFNが特定の個体に集中しているのか、全体に薄く
広がっているのかを個体別(name)に集計する。motmetricsの詳細イベント
(acc.events, Type=='MISS'の行)を使い、GTのIDごとにMISS回数・出現回数・
見逃し率を算出する。

集中しているようなら、その個体固有の検出弱さ(遮蔽・タイル端に近い位置など)を
疑い、閾値調整ではなく検出モデル側の改善を検討する材料になる。

--- 使い方 ---
    python phase2_fn_breakdown.py
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

# ---------------------------------------------------------------------------
# 設定
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
# track_new_thr=0.8, track_confirm_thr=0.5 に修正済みのyamlをそのまま使う想定
TRACKER_CONFIG_PATH = f"{MOOTRACK_CODE_SRC}/configs/tracker_configs.yaml"

FRAME_START = 0
FRAME_END = 290
FRAME_STEP = 10

IMAGE_WIDTH_PX = 8832
NUM_CAMERAS = 3
CAM_IDS = ["Camera_1", "Camera_2", "Camera_3"]
TARGET_CAMERA = "Camera_1"

DIRTY_NAMES = {"cow024", "cow027"}

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
    with open(TRACKER_CONFIG_PATH) as f:
        cfg_preview = yaml.safe_load(f)
    print(f"  現在のtrack_new_thr={cfg_preview['conftrack'].get('track_new_thr')}, "
          f"track_confirm_thr={cfg_preview['conftrack'].get('track_confirm_thr')}")
    tracker = ConfTrackTracker(tracker_params_path=TRACKER_CONFIG_PATH, device=DEVICE)

    acc = mm.MOTAccumulator(auto_id=True)
    gt_id_to_name: Dict[int, str] = {}
    appearances: Dict[int, int] = defaultdict(int)  # gt_id -> このrangeでの出現フレーム数

    frame_indices = list(range(FRAME_START, FRAME_END + 1, FRAME_STEP))
    n_processed = 0

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
        n_processed += 1

        gt_objects = parse_voc_annotation(xml_path)
        gt_ids = []
        gt_boxes_xywh = []
        for o in gt_objects:
            cx_full = (o["xmin"] + o["xmax"]) / 2.0
            if assign_camera(cx_full, IMAGE_WIDTH_PX) != TARGET_CAMERA:
                continue
            if o["name"] in DIRTY_NAMES:
                continue
            gt_id = name_to_gt_id(o["name"])
            gt_id_to_name[gt_id] = o["name"]
            appearances[gt_id] += 1
            gt_ids.append(gt_id)
            local_xmin = o["xmin"] - cam_idx * tile_w
            local_xmax = o["xmax"] - cam_idx * tile_w
            gt_boxes_xywh.append(xyxy_to_xywh(local_xmin, o["ymin"], local_xmax, o["ymax"]))

        hyp_ids = [t.track_id for t in tracks]
        hyp_boxes_xywh = [xyxy_to_xywh(t.x1, t.y1, t.x2, t.y2) for t in tracks]

        dist = mm.distances.iou_matrix(gt_boxes_xywh, hyp_boxes_xywh, max_iou=0.5)
        acc.update(gt_ids, hyp_ids, dist)

    print(f"\n処理フレーム数: {n_processed}/{len(frame_indices)}")

    # --- 全体サマリー(確認用) ---
    mh = mm.metrics.create()
    summary = mh.compute(
        acc, metrics=["idf1", "idp", "idr", "mota", "num_switches"], name="x"
    )
    print(f"\n全体: IDF1={summary['idf1'].iloc[0]:.3f} IDP={summary['idp'].iloc[0]:.3f} "
          f"IDR={summary['idr'].iloc[0]:.3f} MOTA={summary['mota'].iloc[0]:.3f} "
          f"IDs={int(summary['num_switches'].iloc[0])}")

    # --- 個体別のMISS(FN)内訳 ---
    events = acc.events
    miss_events = events[events["Type"] == "MISS"]
    miss_counts: Dict[int, int] = defaultdict(int)
    for oid in miss_events["OId"]:
        if oid == oid:  # NaN除外
            miss_counts[int(oid)] += 1

    rows = []
    for gt_id, name in sorted(gt_id_to_name.items()):
        n_app = appearances[gt_id]
        n_miss = miss_counts.get(gt_id, 0)
        miss_rate = n_miss / n_app if n_app > 0 else float("nan")
        rows.append((name, n_app, n_miss, miss_rate))

    rows.sort(key=lambda r: -r[3])

    print(f"\n=== 個体別 見逃し(FN)率 上位(降順) ===")
    print(f"{'name':<10}{'出現数':>8}{'見逃し数':>10}{'見逃し率':>10}")
    for name, n_app, n_miss, rate in rows:
        flag = "  <-- 要注目" if rate >= 0.5 else ""
        print(f"{name:<10}{n_app:>8}{n_miss:>10}{rate:>10.1%}{flag}")

    total_miss = sum(r[2] for r in rows)
    n_heavy = sum(1 for r in rows if r[3] >= 0.5)
    print(f"\n見逃し合計: {total_miss}件 / うち見逃し率50%以上の個体: {n_heavy}頭"
          f"(全{len(rows)}頭中)")

    import csv
    csv_path = OUTPUT_DIR / "phase2_fn_breakdown.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "num_appearances", "num_miss", "miss_rate"])
        writer.writerows(rows)
    print(f"\nCSV保存: {csv_path}")


if __name__ == "__main__":
    main()