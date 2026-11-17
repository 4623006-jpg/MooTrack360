"""
phase2_missed_cow_visualizer.py

【vml36で実行】

phase2_fn_breakdown.pyで判明した「見逃し率100%(もしくはほぼ100%)の個体」
(PROBLEM_NAMES)について、なぜYOLOが検出できていないのかを目視確認するための
可視化スクリプト。

各個体について、GT XML上でその個体が最初に登場するフレームを探し、
そのGT bbox周辺(マージン込み)をCamera_1タイル画像から切り出す。
同じフレームに対してYOLOを非常に低い信頼度閾値(0.01)で推論し、
GT bboxに一番近い検出があれば、その信頼度と位置を報告した上でオーバーレイに
描画する(近くに何も検出が無ければ「検出候補自体が存在しない」ことがわかる)。

  - 近くに低信頼度の検出がある場合 → 信頼度の閾値調整・キャリブレーションの問題
  - 近くに検出が全く無い場合 → モデルがそもそもその見た目/姿勢/位置を
    「牛」として認識できていない、学習データ不足等を疑う根拠になる

--- 使い方 ---
    python phase2_missed_cow_visualizer.py
"""

from pathlib import Path
from typing import List, Optional, Tuple
import xml.etree.ElementTree as ET

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
YOLO_CONF_THRESH_DEBUG = 0.01  # とにかく何か出ているかを見たいので極端に低くする

FRAME_START = 0
FRAME_END = 290
FRAME_STEP = 10

IMAGE_WIDTH_PX = 8832
NUM_CAMERAS = 3
CAM_IDS = ["Camera_1", "Camera_2", "Camera_3"]
TARGET_CAMERA = "Camera_1"

# phase2_fn_breakdown.pyの結果、見逃し率が高かった個体
PROBLEM_NAMES = [
    "cow022", "cow036", "cow040", "cow043", "cow045",
    "cow046", "cow047", "cow092", "cow031", "cow044", "cow034",
]

CROP_MARGIN_PX = 200  # GT bboxの周囲に付けるマージン

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


def find_first_appearance(name: str, xml_dir: Path, cam_idx: int, tile_w: float) -> Optional[Tuple[int, dict]]:
    """指定した個体名が最初に登場するフレームと、そのローカル座標bboxを返す"""
    frame_indices = list(range(FRAME_START, FRAME_END + 1, FRAME_STEP))
    for frame_idx in frame_indices:
        xml_path = xml_dir / f"{frame_name(frame_idx)}.xml"
        if not xml_path.exists():
            continue
        for obj in parse_voc_annotation(xml_path):
            if obj["name"] != name:
                continue
            cx_full = (obj["xmin"] + obj["xmax"]) / 2.0
            if assign_camera(cx_full, IMAGE_WIDTH_PX) != TARGET_CAMERA:
                continue
            local = dict(obj)
            local["xmin"] -= cam_idx * tile_w
            local["xmax"] -= cam_idx * tile_w
            return frame_idx, local
    return None


def main():
    from ultralytics import YOLO

    xml_dir = Path(XML_DIR)
    image_dir = Path(IMAGE_DIR)
    cam_idx = CAM_IDS.index(TARGET_CAMERA)
    tile_w = IMAGE_WIDTH_PX / NUM_CAMERAS

    print(f"YOLOモデル読み込み中: {YOLO_WEIGHTS_PATH}")
    model = YOLO(YOLO_WEIGHTS_PATH)

    print(f"{'name':<10}{'frame':>8}{'GT中心(x,y)':>18}{'最寄検出信頼度':>16}{'最寄検出距離[px]':>18}")

    summary_rows = []

    for name in PROBLEM_NAMES:
        found = find_first_appearance(name, xml_dir, cam_idx, tile_w)
        if found is None:
            print(f"{name:<10} 対象範囲内でCamera_1に登場するフレームが見つかりません")
            continue
        frame_idx, gt = found

        image_path = image_dir / f"{frame_name(frame_idx)}.jpg"
        full_img = cv2.imread(str(image_path))
        x0 = int(cam_idx * tile_w)
        x1_edge = int((cam_idx + 1) * tile_w)
        frame_img = full_img[:, x0:x1_edge]

        results = model.predict(
            frame_img, conf=YOLO_CONF_THRESH_DEBUG, device=DEVICE, verbose=False
        )[0]

        gt_cx = (gt["xmin"] + gt["xmax"]) / 2.0
        gt_cy = (gt["ymin"] + gt["ymax"]) / 2.0

        # GT中心に最も近い検出を探す
        best_dist = None
        best_conf = None
        best_box = None
        for box in results.boxes:
            bx1, by1, bx2, by2 = box.xyxy[0].tolist()
            bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
            d = float(np.hypot(bcx - gt_cx, bcy - gt_cy))
            if best_dist is None or d < best_dist:
                best_dist = d
                best_conf = float(box.conf[0])
                best_box = (bx1, by1, bx2, by2)

        conf_str = f"{best_conf:.3f}" if best_conf is not None else "検出0件"
        dist_str = f"{best_dist:.1f}" if best_dist is not None else "-"
        print(f"{name:<10}{frame_name(frame_idx):>8}{f'({gt_cx:.0f},{gt_cy:.0f})':>18}"
              f"{conf_str:>16}{dist_str:>18}")
        summary_rows.append((name, frame_idx, best_conf, best_dist))

        # クロップして保存(GT=緑, 最寄り検出=赤)
        cx0 = max(0, int(gt["xmin"] - CROP_MARGIN_PX))
        cy0 = max(0, int(gt["ymin"] - CROP_MARGIN_PX))
        cx1 = min(frame_img.shape[1], int(gt["xmax"] + CROP_MARGIN_PX))
        cy1 = min(frame_img.shape[0], int(gt["ymax"] + CROP_MARGIN_PX))
        crop = frame_img[cy0:cy1, cx0:cx1].copy()

        # GT bboxをクロップ内座標に変換して描画
        gx1, gy1 = int(gt["xmin"] - cx0), int(gt["ymin"] - cy0)
        gx2, gy2 = int(gt["xmax"] - cx0), int(gt["ymax"] - cy0)
        cv2.rectangle(crop, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
        cv2.putText(crop, f"GT:{name}", (gx1, max(0, gy1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        if best_box is not None:
            bx1, by1, bx2, by2 = best_box
            rx1, ry1 = int(bx1 - cx0), int(by1 - cy0)
            rx2, ry2 = int(bx2 - cx0), int(by2 - cy0)
            cv2.rectangle(crop, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)
            cv2.putText(crop, f"YOLO:{best_conf:.2f}", (rx1, max(0, ry1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        out_path = OUTPUT_DIR / f"missed_{name}_frame{frame_name(frame_idx)}.jpg"
        cv2.imwrite(str(out_path), crop)

    print(f"\n保存先: {OUTPUT_DIR}")
    print("(緑=GTのbbox、赤=そのGTに一番近いYOLO検出(conf>=0.01)。赤が無ければ検出候補が皆無)")
    print("\nローカルに転送して、以下を確認してください:")
    print("  - 赤枠が全く無い個体: モデルがその見た目/姿勢/位置を牛として認識できていない")
    print("  - 赤枠はあるが信頼度が低い個体: 位置は合っているが確信度が低いだけ(キャリブレーション寄りの問題)")
    print("  - 複数個体で共通する背景・位置・姿勢のパターンがないか目視で見比べる")


if __name__ == "__main__":
    main()