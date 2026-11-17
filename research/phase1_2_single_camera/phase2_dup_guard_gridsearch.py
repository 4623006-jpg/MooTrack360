"""
phase2_dup_guard_gridsearch.py

【vml36で実行】

Tentativeトラックへの猶予追加(FN改善だがFP急増)→重複トラック防止ガード追加
(FP改善だが、境界エリアの不安定な個体(cow022/036/092等)のFNが逆に悪化)、という
経緯を受けて、重複ガードの適用範囲を絞る`dup_guard_max_staleness`パラメータの
最適値を探すグリッドサーチ。

dup_guard_max_staleness = N は「直近N フレーム以内にマッチできているトラックだけを
重複防止の対象にする(それより古い、マッチできていない“ゾンビ”トラックは対象外に
して、新しい検出のpending pool復活を妨げないようにする)」という意味。

  - 小さい(0に近い)ほど: 重複ガードがほぼ効かなくなる → Tentative猶予のみの状態
    (FN=54, FP=335, IDF1=80.2%)に近づくはず
  - 大きい(30=max_age相当)ほど: 全トラックが対象になる → フルの重複ガード状態
    (FN=124, FP=129, IDF1=87.7%)に近づくはず

YOLO推論はコストが高いため、30フレーム分の検出結果とGTを先に1回だけ計算して
キャッシュし、dup_guard_max_staleness の値ごとにはConfTrackTrackerだけを
作り直して評価する。track_new_thr/track_confirm_thrは現状のベスト値(0.8/0.5)に
固定する。

さらに、IDF1が最良の設定については、cow022/cow036/cow040/cow043/cow045/
cow046/cow047/cow092(境界エリアの問題個体)の見逃し率も個別に出し、
「全体IDF1は良いが狙った個体は救えていない」状態を見逃さないようにする。

--- 使い方 ---
    python phase2_dup_guard_gridsearch.py
"""

import os
import re
import sys
import tempfile
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
BASE_TRACKER_CONFIG_PATH = f"{MOOTRACK_CODE_SRC}/configs/tracker_configs.yaml"

FRAME_START = 0
FRAME_END = 290
FRAME_STEP = 10

IMAGE_WIDTH_PX = 8832
NUM_CAMERAS = 3
CAM_IDS = ["Camera_1", "Camera_2", "Camera_3"]
TARGET_CAMERA = "Camera_1"

DIRTY_NAMES = {"cow024", "cow027"}
PROBLEM_NAMES = ["cow022", "cow036", "cow040", "cow043", "cow045", "cow046", "cow047", "cow092"]

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# dup_guard_max_staleness候補(0=ほぼガード無効, 大きいほどフルガードに近づく)
DUP_GUARD_STALENESS_CANDIDATES = [0, 1, 2, 3, 5, 8, 15, 30]

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
        gt_names = []
        gt_boxes_xywh = []
        for o in gt_objects:
            cx_full = (o["xmin"] + o["xmax"]) / 2.0
            if assign_camera(cx_full, IMAGE_WIDTH_PX) != TARGET_CAMERA:
                continue
            if o["name"] in DIRTY_NAMES:
                continue
            gt_ids.append(name_to_gt_id(o["name"]))
            gt_names.append(o["name"])
            local_xmin = o["xmin"] - cam_idx * tile_w
            local_xmax = o["xmax"] - cam_idx * tile_w
            gt_boxes_xywh.append(xyxy_to_xywh(local_xmin, o["ymin"], local_xmax, o["ymax"]))

        cached.append({
            "frame_idx": frame_idx,
            "boxes": boxes,
            "frame_img": frame_img,
            "gt_ids": gt_ids,
            "gt_names": gt_names,
            "gt_boxes_xywh": gt_boxes_xywh,
        })
        print(f"  {frame_name(frame_idx)}: YOLO検出={len(boxes)}件 GT={len(gt_ids)}件 キャッシュ完了")

    return cached


def evaluate_one_config(cached_frames: List[dict], dup_guard_max_staleness: int) -> dict:
    import motmetrics as mm

    with open(BASE_TRACKER_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    cfg["conftrack"]["dup_guard_max_staleness"] = dup_guard_max_staleness

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, dir=OUTPUT_DIR
        ) as tmp:
            yaml.safe_dump(cfg, tmp)
            tmp_path = tmp.name

        tracker = ConfTrackTracker(tracker_params_path=tmp_path, device=DEVICE)
        acc = mm.MOTAccumulator(auto_id=True)

        gt_id_to_name: Dict[int, str] = {}
        appearances: Dict[int, int] = defaultdict(int)

        for entry in cached_frames:
            detections = [Detection(*b) for b in entry["boxes"]]
            tracks = tracker.update(detections, frame=entry["frame_img"])
            hyp_ids = [t.track_id for t in tracks]
            hyp_boxes_xywh = [xyxy_to_xywh(t.x1, t.y1, t.x2, t.y2) for t in tracks]
            dist = mm.distances.iou_matrix(entry["gt_boxes_xywh"], hyp_boxes_xywh, max_iou=0.5)
            acc.update(entry["gt_ids"], hyp_ids, dist)

            for gid, name in zip(entry["gt_ids"], entry["gt_names"]):
                gt_id_to_name[gid] = name
                appearances[gid] += 1

        mh = mm.metrics.create()
        summary = mh.compute(
            acc, metrics=["idf1", "idp", "idr", "mota", "num_switches",
                          "num_false_positives", "num_misses"], name="x"
        )
        row = summary.iloc[0]

        events = acc.events
        miss_events = events[events["Type"] == "MISS"]
        miss_counts: Dict[int, int] = defaultdict(int)
        for oid in miss_events["OId"]:
            if oid == oid:
                miss_counts[int(oid)] += 1

        problem_miss_rates = {}
        for name in PROBLEM_NAMES:
            gid = name_to_gt_id(name)
            n_app = appearances.get(gid, 0)
            n_miss = miss_counts.get(gid, 0)
            problem_miss_rates[name] = (n_miss / n_app) if n_app > 0 else None

        return {
            "dup_guard_max_staleness": dup_guard_max_staleness,
            "idf1": float(row["idf1"]),
            "idp": float(row["idp"]),
            "idr": float(row["idr"]),
            "mota": float(row["mota"]),
            "num_switches": int(row["num_switches"]),
            "fp": int(row["num_false_positives"]),
            "fn": int(row["num_misses"]),
            "problem_miss_rates": problem_miss_rates,
        }
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def main():
    print("=== フレームデータのキャッシュ(YOLO推論・GT読み込みを1回だけ実行) ===")
    cached_frames = load_frame_data()
    print(f"\n{len(cached_frames)}フレームをキャッシュしました\n")

    results = []
    for staleness in DUP_GUARD_STALENESS_CANDIDATES:
        r = evaluate_one_config(cached_frames, staleness)
        results.append(r)
        print(f"  staleness={staleness:>3} -> IDF1={r['idf1']:.3f} IDP={r['idp']:.3f} IDR={r['idr']:.3f} "
              f"MOTA={r['mota']:.3f} IDs={r['num_switches']} FP={r['fp']} FN={r['fn']}")

    print(f"\n{'staleness':>10} | {'IDF1':>8} | {'IDP':>8} | {'IDR':>8} | {'MOTA':>8} | {'IDs':>4} | {'FP':>5} | {'FN':>5}")
    for r in results:
        print(f"{r['dup_guard_max_staleness']:>10} | {r['idf1']:>8.3f} | {r['idp']:>8.3f} | "
              f"{r['idr']:>8.3f} | {r['mota']:>8.3f} | {r['num_switches']:>4} | {r['fp']:>5} | {r['fn']:>5}")

    best = max(results, key=lambda r: r["idf1"])
    print(f"\n最良のIDF1: staleness={best['dup_guard_max_staleness']} -> IDF1={best['idf1']:.3f}")

    print(f"\n=== 各staleness候補での問題個体8頭の見逃し率 ===")
    header = f"{'staleness':>10}" + "".join(f"{n:>10}" for n in PROBLEM_NAMES)
    print(header)
    for r in results:
        row_str = f"{r['dup_guard_max_staleness']:>10}"
        for name in PROBLEM_NAMES:
            v = r["problem_miss_rates"][name]
            row_str += f"{'-' if v is None else f'{v:.0%}':>10}"
        print(row_str)

    import csv
    csv_path = OUTPUT_DIR / "phase2_dup_guard_gridsearch_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["dup_guard_max_staleness", "idf1", "idp", "idr", "mota", "num_switches", "fp", "fn"] + PROBLEM_NAMES
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: r[k] for k in ["dup_guard_max_staleness", "idf1", "idp", "idr", "mota", "num_switches", "fp", "fn"]}
            row.update(r["problem_miss_rates"])
            writer.writerow(row)
    print(f"\nCSV保存: {csv_path}")


if __name__ == "__main__":
    main()