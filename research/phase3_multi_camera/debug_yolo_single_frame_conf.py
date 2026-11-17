"""
debug_yolo_single_frame_conf.py

鳥瞰図上で特定のフレームだけ消えてしまう個体がいる原因を調べるためのデバッグ用
スクリプト。TARGET_FRAME_IDXで指定した1フレームについて、各カメラのYOLO検出を
「本番のYOLO_CONF_THRESHより大幅に緩い閾値(DEBUG_CONF_THRESH)」まで全部拾って
描画し、bboxの横に信頼度(conf)を表示する。

これにより、消えた個体が
  (a) そもそもYOLOに検出されていない(DEBUG_CONF_THRESH以下でも箱が無い)
  (b) 検出はされているが信頼度が低く、本番のYOLO_CONF_THRESHで弾かれている
      (薄い色・細い枠で「(閾値未満)」と表示される)
  (c) 信頼度は十分あるが、重複除去(IoU)や有効範囲/floorplan範囲外フィルタ、
      あるいは融合(統合)側で消えている
のどれなのかを一目で切り分けられる(cを疑う場合は、既存のsave_annotated_camera_images
や[extra-fused]ログの方を見るのが早い)。

--- 使い方 ---
    1. 下のTARGET_FRAME_IDXを調べたいフレーム番号に書き換える
    2. python debug_yolo_single_frame_conf.py
    3. OUTPUT_DIR/debug_yolo_single_frame/{frame}_{cam_id}_debug_conf.jpg を確認
"""

import os
import sys
from pathlib import Path

import cv2

MOOTRACK_CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calibration")
if MOOTRACK_CODE_DIR not in sys.path:
    sys.path.insert(0, MOOTRACK_CODE_DIR)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from Mootrack_core import CAM_IDS, NUM_CAMERAS

# render_bev_frames_yolo_zonepose.py側の設定・配色をそのまま再利用
# (YOLO_CONF_THRESHなどの値がズレると「どれが閾値未満か」の判定が食い違うため)
from render_bev_frames_yolo_zonepose_ver4 import (
    IMAGE_DIR, OUTPUT_DIR, YOLO_WEIGHTS_PATH,
    YOLO_CONF_THRESH, IMAGE_WIDTH_PX, YOLO_CLASS_TO_POSE,
    frame_name, POSE_BOX_COLORS_BGR, DEVICE,
)

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------

TARGET_FRAME_IDX = 1720  # 調べたいフレーム番号に変更する

# 本番のYOLO_CONF_THRESHより大幅に緩めて、低信頼度の検出まで全部拾う
DEBUG_CONF_THRESH = 0.1

DEBUG_OUT_DIR = OUTPUT_DIR / "debug_yolo_single_frame"
DEBUG_OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    from ultralytics import YOLO

    image_path = Path(IMAGE_DIR) / f"{frame_name(TARGET_FRAME_IDX)}.jpg"
    if not image_path.exists():
        print(f"エラー: 画像が見つかりません: {image_path}")
        return

    print(f"YOLOモデル読み込み中: {YOLO_WEIGHTS_PATH}")
    yolo_model = YOLO(YOLO_WEIGHTS_PATH)

    full_img = cv2.imread(str(image_path))
    if full_img is None:
        print(f"エラー: 画像を読み込めません: {image_path}")
        return

    tile_w = IMAGE_WIDTH_PX / NUM_CAMERAS

    print(f"対象フレーム: {frame_name(TARGET_FRAME_IDX)}  "
          f"本番のYOLO_CONF_THRESH={YOLO_CONF_THRESH}  デバッグ用の緩い閾値={DEBUG_CONF_THRESH}\n")

    for cam_idx, cam_id in enumerate(CAM_IDS):
        x0 = int(cam_idx * tile_w)
        x1 = int((cam_idx + 1) * tile_w)
        crop = full_img[:, x0:x1].copy()

        results = yolo_model.predict(
            crop, conf=DEBUG_CONF_THRESH, device=DEVICE, verbose=False
        )[0]
        n_boxes = len(results.boxes)
        n_pass = 0

        for box in results.boxes:
            bx1, by1, bx2, by2 = [int(round(v)) for v in box.xyxy[0].tolist()]
            conf = float(box.conf[0])
            pose = YOLO_CLASS_TO_POSE.get(int(box.cls[0]))
            base_color = POSE_BOX_COLORS_BGR.get(pose, (180, 180, 180))

            passes_thresh = conf >= YOLO_CONF_THRESH
            if passes_thresh:
                n_pass += 1
                color = base_color
                thickness = 2
                tag = ""
            else:
                # 本番の閾値未満の箱は、色を薄く・枠を細くして区別する
                color = tuple(int(c * 0.5 + 128 * 0.5) for c in base_color)
                thickness = 1
                tag = "(閾値未満)"

            cv2.rectangle(crop, (bx1, by1), (bx2, by2), color, thickness, lineType=cv2.LINE_AA)
            label = f"{pose} conf={conf:.2f}{tag}"
            cv2.putText(crop, label, (bx1, max(0, by1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2, cv2.LINE_AA)

        print(f"  {cam_id}: 検出数(conf>={DEBUG_CONF_THRESH})={n_boxes}  "
              f"うち本番閾値({YOLO_CONF_THRESH})以上={n_pass}")

        out_path = DEBUG_OUT_DIR / f"{frame_name(TARGET_FRAME_IDX)}_{cam_id}_debug_conf.jpg"
        cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"    保存: {out_path}")

    print(f"\n完了。{DEBUG_OUT_DIR} を確認してください。")
    print("色が薄く枠が細い箱は「検出はされているが信頼度が低く本番閾値で弾かれている」個体。")
    print("対象の個体がどのカメラの画像にも全く箱が出ていない場合は、"
          "YOLOがそもそも検出できていない(閾値の問題ではない)ことになる。")


if __name__ == "__main__":
    main()