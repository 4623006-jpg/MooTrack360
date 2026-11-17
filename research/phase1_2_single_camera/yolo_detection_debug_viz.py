"""
yolo_detection_debug_viz.py

【vml36で実行】

phase2_yolo_conftrack_check.pyでYOLO検出数が異常(GT比2倍以上)かつtrack出力が
全フレーム0件だった原因を切り分けるための可視化・診断スクリプト。

000000.jpg 1枚に対して、
  (a) ステッチ画像全体にそのままYOLO推論をかけた場合
  (b) Camera_1のタイルだけを切り出してからYOLO推論をかけた場合
の両方を実行し、検出結果(bbox+信頼度)をオーバーレイ画像として保存する。
あわせて信頼度の分布(min/max/mean、tracker_new_thr=0.86を超えている件数)を
表示し、「track_new_thrを一度も超えていないから追跡が始まらない」という仮説と
「ステッチ画像を丸ごと渡すと縮小されすぎて検出の質が落ちる」という仮説の
どちらが効いているかを確認する。

--- 使い方 ---
    python yolo_detection_debug_viz.py
"""

from pathlib import Path

import cv2
import numpy as np
import torch

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
IMAGE_PATH = "/mnt/vmlqnap02/dataset/MooTrack360/1-hour-farm1-area1-sample-sequence/1-hour-farm1-area1-frames/000000.jpg"
YOLO_WEIGHTS_PATH = "/mnt/vmlqnap02/dataset/MooTrack360/Trained models/yolo11x/train_ultralytics/weights/best.pt"
OUTPUT_DIR = Path(
    "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/research/idf1_check"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

YOLO_CONF_THRESH = 0.1   # phase2と同じ、緩めに全部拾う
TRACK_NEW_THR = 0.86     # tracker_configs.yamlのtrack_new_thrと同じ値(参考表示用)

IMAGE_WIDTH_PX = 8832
NUM_CAMERAS = 3
CAM_IDX = 0  # Camera_1

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def draw_boxes(img: np.ndarray, boxes_conf, thickness=2) -> np.ndarray:
    out = img.copy()
    for x1, y1, x2, y2, conf in boxes_conf:
        color = (0, 255, 0) if conf >= TRACK_NEW_THR else (0, 165, 255) if conf >= 0.3 else (0, 0, 255)
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        cv2.putText(out, f"{conf:.2f}", (int(x1), max(0, int(y1) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def print_stats(label: str, confs: list):
    if not confs:
        print(f"  {label}: 検出0件")
        return
    arr = np.array(confs)
    n_above_new_thr = int((arr >= TRACK_NEW_THR).sum())
    print(f"  {label}: 件数={len(arr)} 信頼度 min={arr.min():.3f} max={arr.max():.3f} "
          f"mean={arr.mean():.3f} / track_new_thr({TRACK_NEW_THR})以上={n_above_new_thr}件")


def main():
    from ultralytics import YOLO

    print(f"YOLOモデル読み込み中: {YOLO_WEIGHTS_PATH}")
    model = YOLO(YOLO_WEIGHTS_PATH)

    full_img = cv2.imread(IMAGE_PATH)
    if full_img is None:
        raise FileNotFoundError(IMAGE_PATH)
    print(f"元画像サイズ: {full_img.shape[1]}x{full_img.shape[0]}")

    tile_w = IMAGE_WIDTH_PX // NUM_CAMERAS
    x0, x1 = CAM_IDX * tile_w, (CAM_IDX + 1) * tile_w
    cam1_img = full_img[:, x0:x1]
    print(f"Camera_1タイルサイズ: {cam1_img.shape[1]}x{cam1_img.shape[0]}")

    # (a) ステッチ画像全体にそのまま推論
    print("\n[a] ステッチ画像全体に推論中...")
    results_full = model.predict(full_img, conf=YOLO_CONF_THRESH, device=DEVICE, verbose=False)[0]
    boxes_full = [(*b.xyxy[0].tolist(), float(b.conf[0])) for b in results_full.boxes]
    print_stats("ステッチ画像全体", [c for *_, c in boxes_full])
    overlay_full = draw_boxes(full_img, boxes_full)
    cv2.imwrite(str(OUTPUT_DIR / "debug_yolo_full_image.jpg"), overlay_full)
    # Camera_1範囲だけ切り出した確認用(全体版だと小さすぎて見えないため)
    cv2.imwrite(str(OUTPUT_DIR / "debug_yolo_full_image_cam1crop.jpg"), overlay_full[:, x0:x1])

    # (b) Camera_1タイルだけ切り出してから推論
    print("\n[b] Camera_1タイルのみに推論中...")
    results_cam1 = model.predict(cam1_img, conf=YOLO_CONF_THRESH, device=DEVICE, verbose=False)[0]
    boxes_cam1 = [(*b.xyxy[0].tolist(), float(b.conf[0])) for b in results_cam1.boxes]
    print_stats("Camera_1タイルのみ", [c for *_, c in boxes_cam1])
    overlay_cam1 = draw_boxes(cam1_img, boxes_cam1)
    cv2.imwrite(str(OUTPUT_DIR / "debug_yolo_cam1_crop_only.jpg"), overlay_cam1)

    print(f"\n保存先: {OUTPUT_DIR}")
    print("  debug_yolo_full_image_cam1crop.jpg : (a)ステッチ画像全体に推論した結果のうちCamera_1部分")
    print("  debug_yolo_cam1_crop_only.jpg      : (b)Camera_1タイルだけを切り出して推論した結果")
    print("(ローカルに転送して両方見比べてください。緑=track_new_thr以上、黄=detection_low_thr以上、赤=それ未満)")


if __name__ == "__main__":
    main()