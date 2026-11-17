"""
phase2_confidence_timeseries.py

【vml36で実行】

phase2_fn_breakdown.py(pending pool適用後)の再評価で、依然として見逃し率が
高い(80〜93%)個体について、pending poolの5フレーム連続一致条件
(pending_init_streak=5)がなぜ満たせないのかを切り分けるための診断スクリプト。

対象30フレーム全てで、GT中心に最も近いYOLO検出(conf>=0.01で全部拾う)について、
  - 信頼度(conf)
  - GT bboxとのIoU(位置がどれだけ合っているか)
  - 1つ前のフレームで選ばれた検出とのIoU(pending_iou_thresh=0.4との比較。
    pending poolのstreak判定を簡易的に模擬)
  - その結果としてのstreak推移
を記録・表示する。

見たいポイント:
  - conf が pending_min_conf(0.35) を頻繁に下回っている
      → pending poolに入る条件自体を満たせていない(閾値をさらに下げないと無理)
  - conf は 0.35〜0.8 にあるが、iou_prev が頻繁に 0.4 を下回る
      → 位置が毎フレーム動きすぎてstreakが切れている
        (pending_iou_threshを緩める/pending_max_missedを増やす方向で対処できる)

比較対象として、pending pool導入後に大きく改善した3頭(cow031, cow044, cow034)も
一緒に出力し、「streak=5に到達できている個体」と「できていない個体」の違いを
見比べられるようにしている。

【注意】ここでのstreak判定は実際のConfTrack内pending poolの完全な再現ではなく、
「毎フレームGT中心に最も近い1検出だけ」を追跡する簡易シミュレーションです。
複数候補の競合や生成順序までは再現していませんが、個体ごとの傾向を掴む診断
としては十分なはずです。

--- 使い方 ---
    python phase2_confidence_timeseries.py
"""

import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

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
YOLO_CONF_THRESH_DEBUG = 0.01  # 何が出ているか全部見る

FRAME_START = 0
FRAME_END = 290
FRAME_STEP = 10

IMAGE_WIDTH_PX = 8832
NUM_CAMERAS = 3
CAM_IDS = ["Camera_1", "Camera_2", "Camera_3"]
TARGET_CAMERA = "Camera_1"

# phase2_fn_breakdown.py(pending pool適用後)で依然見逃し率が高かった8頭
PROBLEM_NAMES = ["cow022", "cow092", "cow036", "cow040", "cow043", "cow045", "cow046", "cow047"]
# 比較用: pending pool導入で大きく改善した3頭(対照群)
IMPROVED_NAMES = ["cow031", "cow044", "cow034"]
ALL_TARGET_NAMES = PROBLEM_NAMES + IMPROVED_NAMES

# tracker_configs.yaml / pending pool実装と揃える
PENDING_MIN_CONF = 0.35
TRACK_NEW_THR = 0.8
PENDING_IOU_THRESH = 0.4

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


def iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def main():
    from ultralytics import YOLO

    xml_dir = Path(XML_DIR)
    image_dir = Path(IMAGE_DIR)
    cam_idx = CAM_IDS.index(TARGET_CAMERA)
    tile_w = IMAGE_WIDTH_PX / NUM_CAMERAS

    print(f"YOLOモデル読み込み中: {YOLO_WEIGHTS_PATH}")
    model = YOLO(YOLO_WEIGHTS_PATH)

    frame_indices = list(range(FRAME_START, FRAME_END + 1, FRAME_STEP))

    prev_box: Dict[str, Optional[Tuple[float, float, float, float]]] = {n: None for n in ALL_TARGET_NAMES}
    streak: Dict[str, int] = {n: 0 for n in ALL_TARGET_NAMES}
    max_streak: Dict[str, int] = {n: 0 for n in ALL_TARGET_NAMES}
    rows_by_name: Dict[str, List[dict]] = {n: [] for n in ALL_TARGET_NAMES}

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

        gt_objects = parse_voc_annotation(xml_path)
        gt_by_name = {}
        for o in gt_objects:
            cx_full = (o["xmin"] + o["xmax"]) / 2.0
            if assign_camera(cx_full, IMAGE_WIDTH_PX) != TARGET_CAMERA:
                continue
            if o["name"] not in ALL_TARGET_NAMES:
                continue
            local = dict(o)
            local["xmin"] -= cam_idx * tile_w
            local["xmax"] -= cam_idx * tile_w
            gt_by_name[o["name"]] = local

        if not gt_by_name:
            continue

        results = model.predict(
            frame_img, conf=YOLO_CONF_THRESH_DEBUG, device=DEVICE, verbose=False
        )[0]
        det_boxes = []
        for box in results.boxes:
            bx1, by1, bx2, by2 = box.xyxy[0].tolist()
            det_boxes.append((bx1, by1, bx2, by2, float(box.conf[0])))

        for name, gt in gt_by_name.items():
            gt_box = (gt["xmin"], gt["ymin"], gt["xmax"], gt["ymax"])
            gt_cx = (gt["xmin"] + gt["xmax"]) / 2.0
            gt_cy = (gt["ymin"] + gt["ymax"]) / 2.0

            best = None
            best_dist = None
            for bx1, by1, bx2, by2, conf in det_boxes:
                bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                d = float(np.hypot(bcx - gt_cx, bcy - gt_cy))
                if best_dist is None or d < best_dist:
                    best_dist = d
                    best = (bx1, by1, bx2, by2, conf)

            if best is None:
                rows_by_name[name].append({
                    "frame": frame_idx, "conf": None, "iou_gt": None,
                    "iou_prev": None, "streak": 0, "note": "検出0件",
                })
                prev_box[name] = None
                streak[name] = 0
                continue

            bx1, by1, bx2, by2, conf = best
            det_box = (bx1, by1, bx2, by2)
            iou_gt = iou_xyxy(det_box, gt_box)

            iou_prev = None
            if prev_box[name] is not None:
                iou_prev = iou_xyxy(det_box, prev_box[name])

            if conf >= TRACK_NEW_THR:
                note = "track_new_thr到達(即トラック化)"
                streak[name] = 0
            elif conf < PENDING_MIN_CONF:
                note = "pending_min_conf未満(即棄却)"
                streak[name] = 0
            else:
                if iou_prev is not None and iou_prev >= PENDING_IOU_THRESH:
                    streak[name] += 1
                    note = "IoU一致・streak継続"
                else:
                    streak[name] = 1
                    note = "IoU不一致・streakリセット(新規pending)"

            max_streak[name] = max(max_streak[name], streak[name])

            rows_by_name[name].append({
                "frame": frame_idx, "conf": conf, "iou_gt": iou_gt,
                "iou_prev": iou_prev, "streak": streak[name], "note": note,
            })
            prev_box[name] = det_box

    for name in ALL_TARGET_NAMES:
        rows = rows_by_name[name]
        group = "問題個体" if name in PROBLEM_NAMES else "改善済み(対照群)"
        print(f"\n=== {name} [{group}] 最大streak到達値: {max_streak[name]} (5必要) ===")
        print(f"{'frame':>8}{'conf':>8}{'iou_gt':>8}{'iou_prev':>10}{'streak':>8}  note")
        for r in rows:
            conf_str = f"{r['conf']:.3f}" if r["conf"] is not None else "-"
            iou_gt_str = f"{r['iou_gt']:.3f}" if r["iou_gt"] is not None else "-"
            iou_prev_str = f"{r['iou_prev']:.3f}" if r["iou_prev"] is not None else "-"
            print(f"{frame_name(r['frame']):>8}{conf_str:>8}{iou_gt_str:>8}"
                  f"{iou_prev_str:>10}{r['streak']:>8}  {r['note']}")

    print("\n=== まとめ ===")
    print(f"{'name':<10}{'group':<16}{'最大streak':>12}{'5到達':>8}")
    for name in ALL_TARGET_NAMES:
        group = "問題個体" if name in PROBLEM_NAMES else "改善済み"
        reached = "YES" if max_streak[name] >= 5 else "no"
        print(f"{name:<10}{group:<16}{max_streak[name]:>12}{reached:>8}")

    csv_path = OUTPUT_DIR / "phase2_confidence_timeseries.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "frame", "conf", "iou_gt", "iou_prev", "streak", "note"])
        for name in ALL_TARGET_NAMES:
            for r in rows_by_name[name]:
                writer.writerow([
                    name, frame_name(r["frame"]), r["conf"], r["iou_gt"],
                    r["iou_prev"], r["streak"], r["note"],
                ])
    print(f"\nCSV保存: {csv_path}")


if __name__ == "__main__":
    main()