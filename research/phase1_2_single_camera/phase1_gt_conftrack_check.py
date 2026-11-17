"""
phase1_gt_conftrack_check.py

【vml36で実行】

IDF1評価のPhase 1: YOLO検出のノイズを排除した状態で、ConfTrackTracker自体の
ID一貫性ロジック(Kalman+IoUマッチング)がどれだけの性能を持つかを見る
「上限性能テスト」。

GTのbbox(Camera_1のみ)をそのまま「検出結果」としてConfTrackTrackerに
フレーム順に流し込み、出力されたtrack_idの推移を、GTのname(check_track_gt_
consistency.pyで一貫性を確認済み)と比較してIDF1を計算する。

check_track_gt_consistency.py の結果、000000-000290(30フレーム, Camera_1)の
範囲では48個体中46個体が完全にクリーン(1フレーム1回、移動量50px以内)、
cow024(000180のみ重複)とcow027(000130のみ重複)の2個体だけ問題ありだった。
この2個体はDIRTY_NAMESとして評価対象から除外する
(検出としては入力に含めるが、GTとしては使わない)。

--- 事前準備 ---
    uv pip install motmetrics
(未インストールの場合。インストール済みならスキップしてよい)

--- 使い方 ---
    python phase1_gt_conftrack_check.py
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

# ConfTrackTrackerが置かれているプロジェクトのsrcディレクトリ(sys.pathに追加する)
MOOTRACK_CODE_SRC = "/home/ara/myproject/project2/MooTrack360_source_code/MooTrack360_code/src"
TRACKER_CONFIG_PATH = f"{MOOTRACK_CODE_SRC}/configs/tracker_configs.yaml"

FRAME_START = 0
FRAME_END = 290       # 30フレーム分(check_track_gt_consistency.pyと同じ範囲)
FRAME_STEP = 10

IMAGE_WIDTH_PX = 8832
NUM_CAMERAS = 3
CAM_IDS = ["Camera_1", "Camera_2", "Camera_3"]
TARGET_CAMERA = "Camera_1"

# check_track_gt_consistency.pyの結果、同一フレーム内に重複が見つかった個体
# (評価対象から除外する。検出としては入力するが、GTとしては使わない)
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
# GT XML読み込み・カメラ振り分け(check_track_gt_consistency.pyと同じロジック)
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

    xml_dir = Path(XML_DIR)
    image_dir = Path(IMAGE_DIR)
    cam_idx = CAM_IDS.index(TARGET_CAMERA)
    tile_w = IMAGE_WIDTH_PX / NUM_CAMERAS

    print(f"ConfTrackTracker初期化中(device={DEVICE})...")
    tracker = ConfTrackTracker(tracker_params_path=TRACKER_CONFIG_PATH, device=DEVICE)

    acc = mm.MOTAccumulator(auto_id=True)
    gt_id_to_name: Dict[int, str] = {}

    frame_indices = list(range(FRAME_START, FRAME_END + 1, FRAME_STEP))
    n_processed = 0
    n_excluded_dirty = 0

    for frame_idx in frame_indices:
        xml_path = xml_dir / f"{frame_name(frame_idx)}.xml"
        image_path = image_dir / f"{frame_name(frame_idx)}.jpg"
        if not xml_path.exists():
            print(f"  {frame_name(frame_idx)}: XML無し、スキップ")
            continue

        objects = parse_voc_annotation(xml_path)

        # Camera_1の物体だけ抽出し、ローカル座標(タイル内)に変換
        cam_objects = []
        for obj in objects:
            cx_full = (obj["xmin"] + obj["xmax"]) / 2.0
            if assign_camera(cx_full, IMAGE_WIDTH_PX) != TARGET_CAMERA:
                continue
            local = dict(obj)
            local["xmin"] -= cam_idx * tile_w
            local["xmax"] -= cam_idx * tile_w
            cam_objects.append(local)

        # 検出リスト(全個体。dirtyも含めて実際のシーンとして入力する)
        detections = [
            Detection(o["xmin"], o["ymin"], o["xmax"], o["ymax"], 1.0, 0)
            for o in cam_objects
        ]

        # このフレームのCamera_1タイル画像(CMC用に実画像を渡す)
        frame_img = None
        if image_path.exists():
            full_img = cv2.imread(str(image_path))
            if full_img is not None:
                x0 = int(cam_idx * tile_w)
                x1 = int((cam_idx + 1) * tile_w)
                frame_img = full_img[:, x0:x1]

        tracks = tracker.update(detections, frame=frame_img)
        n_processed += 1

        # GT側(dirty名は除外)。motmetricsはID列をfloatにキャストするため、
        # nameの文字列ではなく数値IDを渡す(name_to_gt_id)
        gt_ids = []
        gt_boxes_xywh = []
        for o in cam_objects:
            if o["name"] in DIRTY_NAMES:
                n_excluded_dirty += 1
                continue
            gt_id = name_to_gt_id(o["name"])
            gt_id_to_name[gt_id] = o["name"]
            gt_ids.append(gt_id)
            gt_boxes_xywh.append(xyxy_to_xywh(o["xmin"], o["ymin"], o["xmax"], o["ymax"]))

        # 予測側(トラッカーの出力全部)
        hyp_ids = [t.track_id for t in tracks]
        hyp_boxes_xywh = [xyxy_to_xywh(t.x1, t.y1, t.x2, t.y2) for t in tracks]

        dist = mm.distances.iou_matrix(gt_boxes_xywh, hyp_boxes_xywh, max_iou=0.5)
        acc.update(gt_ids, hyp_ids, dist)

        print(f"  {frame_name(frame_idx)}: GT={len(gt_ids)}件(除外{len(cam_objects) - len(gt_ids)}件) "
              f"track出力={len(tracks)}件")

    print(f"\n処理フレーム数: {n_processed}/{len(frame_indices)}")

    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=["idf1", "idp", "idr", "mota", "motp", "num_switches",
                 "num_objects", "num_matches", "num_false_positives", "num_misses"],
        name=f"{TARGET_CAMERA}_phase1_gt",
    )
    strsummary = mm.io.render_summary(
        summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names
    )
    print(f"\n{strsummary}")

    csv_path = OUTPUT_DIR / "phase1_gt_conftrack_summary.csv"
    summary.to_csv(csv_path)
    print(f"\nCSV保存: {csv_path}")


if __name__ == "__main__":
    main()