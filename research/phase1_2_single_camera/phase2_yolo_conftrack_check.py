"""
phase2_yolo_conftrack_check.py

【vml36で実行】

IDF1評価のPhase 2: 実際のYOLO検出結果をConfTrackTrackerに入力し、Phase 1
(phase1_gt_conftrack_check.py、GT bboxをそのまま入力した場合のトラッカー単体の
上限性能: IDF1=96.6%)と比較して、実運用に近いIDF1を計測する。

Phase 1との違いは検出の入力元だけで、GT読み込み・カメラ振り分け・trackerとの
連携・motmetrics評価のロジックはほぼ共通。

【重要】当初はステッチ画像(8832x2944)全体をそのままYOLOに渡していたが、
yolo_detection_debug_viz.pyでの目視確認の結果、この横長画像をそのまま推論すると
ultralyticsの自動リサイズで牛が小さくなりすぎて検出品質が大きく劣化することが
判明した。そのためこのバージョンでは、先にCamera_1のタイル(2944px幅)を
切り出してから推論する方式に変更している。

--- 事前準備 ---
    uv pip install ultralytics
(未インストールの場合。インストール済みならスキップしてよい)

--- 使い方 ---
    python phase2_yolo_conftrack_check.py
(パス・フレーム範囲は全て下の設定セクションに直書きしてあるので、引数なしで実行する)
"""

import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
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
# ConfTrack自体がdetection_high_thr(0.8)/detection_low_thr(0.3)で高低信頼度を
# 使い分ける設計(ByteTrack系のLCTM)なので、ここでの事前フィルタは緩めにしておき、
# 閾値判断はトラッカー側に任せる
YOLO_CONF_THRESH = 0.1

# ConfTrackTrackerが置かれているプロジェクトのsrcディレクトリ(sys.pathに追加する)
MOOTRACK_CODE_SRC = "/home/ara/myproject/project2/MooTrack360_source_code/MooTrack360_code/src"
TRACKER_CONFIG_PATH = f"{MOOTRACK_CODE_SRC}/configs/tracker_configs.yaml"

FRAME_START = 0
FRAME_END = 290       # Phase 1と同じ範囲(30フレーム)
FRAME_STEP = 10

IMAGE_WIDTH_PX = 8832
NUM_CAMERAS = 3
CAM_IDS = ["Camera_1", "Camera_2", "Camera_3"]
TARGET_CAMERA = "Camera_1"

# check_track_gt_consistency.pyの結果、同一フレーム内に重複が見つかった個体
# (評価対象から除外する。GTとしては使わないが、実シーンとの整合はYOLO検出側が
# 自然に処理する)
DIRTY_NAMES = {"cow024", "cow027"}

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# ConfTrackTrackerのimport(sys.pathにsrcを追加してから)
# ---------------------------------------------------------------------------
if MOOTRACK_CODE_SRC not in sys.path:
    sys.path.insert(0, MOOTRACK_CODE_SRC)

from Trackers.conftrack_tracker import ConfTrackTracker  # noqa: E402


@dataclass
class Detection:
    """ConfTrackTracker.update()が期待するduck-typedな検出オブジェクト"""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int


# ---------------------------------------------------------------------------
# GT XML読み込み・カメラ振り分け(phase1_gt_conftrack_check.pyと同じロジック)
# ---------------------------------------------------------------------------

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
    """'cow001' -> 1 のように、nameから一意な整数IDを取り出す。
    motmetrics(mot.py)の内部実装がOId列をfloatにキャストするため、
    'cow001'のような文字列をそのままacc.update()に渡すとエラーになる。"""
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
    gt_id_to_name: Dict[int, str] = {}

    frame_indices = list(range(FRAME_START, FRAME_END + 1, FRAME_STEP))
    n_processed = 0

    for frame_idx in frame_indices:
        xml_path = xml_dir / f"{frame_name(frame_idx)}.xml"
        image_path = image_dir / f"{frame_name(frame_idx)}.jpg"
        if not xml_path.exists() or not image_path.exists():
            print(f"  {frame_name(frame_idx)}: XMLまたは画像無し、スキップ")
            continue

        full_img = cv2.imread(str(image_path))
        if full_img is None:
            print(f"  {frame_name(frame_idx)}: 画像読み込み失敗、スキップ")
            continue

        # Camera_1タイルだけ切り出す(推論もCMCもこのクロップに対して行う。
        # ステッチ画像全体に推論すると自動リサイズで牛が小さくなりすぎるため)
        x0 = int(cam_idx * tile_w)
        x1_edge = int((cam_idx + 1) * tile_w)
        frame_img = full_img[:, x0:x1_edge]

        # --- YOLO推論(Camera_1タイルのみ) ---
        results = yolo_model.predict(
            frame_img, conf=YOLO_CONF_THRESH, device=DEVICE, verbose=False
        )[0]

        detections = [
            Detection(*box.xyxy[0].tolist(), float(box.conf[0]), int(box.cls[0]))
            for box in results.boxes
        ]

        tracks = tracker.update(detections, frame=frame_img)
        n_processed += 1

        # --- GT側(Camera_1のみ、dirty名は除外) ---
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
            gt_ids.append(gt_id)
            local_xmin = o["xmin"] - cam_idx * tile_w
            local_xmax = o["xmax"] - cam_idx * tile_w
            gt_boxes_xywh.append(xyxy_to_xywh(local_xmin, o["ymin"], local_xmax, o["ymax"]))

        # --- 予測側(トラッカーの出力全部) ---
        hyp_ids = [t.track_id for t in tracks]
        hyp_boxes_xywh = [xyxy_to_xywh(t.x1, t.y1, t.x2, t.y2) for t in tracks]

        dist = mm.distances.iou_matrix(gt_boxes_xywh, hyp_boxes_xywh, max_iou=0.5)
        acc.update(gt_ids, hyp_ids, dist)

        print(f"  {frame_name(frame_idx)}: GT={len(gt_ids)}件 YOLO検出(Camera_1)={len(detections)}件 "
              f"track出力={len(tracks)}件")

    print(f"\n処理フレーム数: {n_processed}/{len(frame_indices)}")

    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=["idf1", "idp", "idr", "mota", "motp", "num_switches",
                 "num_objects", "num_matches", "num_false_positives", "num_misses"],
        name=f"{TARGET_CAMERA}_phase2_yolo",
    )
    strsummary = mm.io.render_summary(
        summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names
    )
    print(f"\n{strsummary}")
    print("\n(Phase 1参考値: GT bboxをそのまま入力した場合の上限性能 "
          "IDF1=96.6%, IDP=93.5%, IDR=99.9%, MOTA=93.1%, IDs=1)")

    csv_path = OUTPUT_DIR / "phase2_yolo_conftrack_summary.csv"
    summary.to_csv(csv_path)
    print(f"\nCSV保存: {csv_path}")


if __name__ == "__main__":
    main()