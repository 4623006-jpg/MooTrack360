"""
render_gt_annotated_camera_images_ver1.py

実画像(カメラごとのクロップ)にGTのbbox(個体名・姿勢ラベル付き)を重ね描きした
画像を生成する。render_bev_frames_yolo_zonepose_ver1.pyのsave_annotated_
camera_images()が生成するYOLO検出重ね描き画像(annotated_frames_yolo_zonepose_
curved/{frame}_{cam_id}.jpg)と同じフレーム・同じカメラの画像を並べて見比べれば、
「GTには写っているのにYOLOが検出していない(FNの原因)」「YOLOが検出した場所に
GTが無い(FPの原因=幽霊検出)」「GT同士が画像上でも重なって密集している
(crowding_suspectの視覚的確認)」などを実画像上で直接確認できる。

phase3_bev_multicam_fn_fp_causes_ver1.pyのFN/FP明細で出てくる具体的なフレーム
番号(例: 000620, 000560, ...)をFRAME_LISTに入れて実行すれば、そのフレームだけ
GT重ね描き画像を生成する(全フレームを処理すると時間がかかるため、
調査したいフレームだけに絞れるようにしている)。

出力先: OUTPUT_DIR/annotated_frames_gt_zonepose_curved/{frame}_{cam_id}_gt.jpg
       (YOLO側のannotated_frames_yolo_zonepose_curved/{frame}_{cam_id}.jpgと
        ファイル名の番号・カメラIDが揃っているので、同じ番号のファイルを
        並べて比較すればよい)

--- 使い方 ---
    1. FRAME_LISTに調べたいフレーム番号を入れる(空ならFRAME_START〜FRAME_END
       を FRAME_STEP刻みで全部処理する。全フレームだとかなりの枚数になるので注意)
    2. python render_gt_annotated_camera_images_ver1.py
"""

import os
import sys
from pathlib import Path
from typing import List

import cv2

MOOTRACK_CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..","..", "calibration")
if MOOTRACK_CODE_DIR not in sys.path:
    sys.path.insert(0, MOOTRACK_CODE_DIR)

HE_KAI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "he_kai")
if HE_KAI_DIR not in sys.path:
    sys.path.insert(0, HE_KAI_DIR)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

# render_bev_frames_yolo_zonepose_ver1.pyが実際に置かれているフォルダ。
# このファイル(render_gt_annotated_camera_images_ver1.py)は"analyze"フォルダなど、
# render_bev_frames_yolo_zonepose_ver1.pyとは別の場所に置かれることがあるため、
# 明示的にそのフォルダをsys.pathへ追加する。実際のパスに合わせて書き換えること
# (同じフォルダに置く場合は不要)。
RENDER_SCRIPT_DIR = os.path.join(THIS_DIR, "..", "phase3_multi_camera")
if RENDER_SCRIPT_DIR not in sys.path:
    sys.path.insert(0, RENDER_SCRIPT_DIR)

# render_bev_frames_yolo_zonepose_ver1.py(曲線境界版)側の設定値・ヘルパーを
# そのまま再利用(ロジックの二重管理を避けるため)
from render_bev_frames_yolo_zonepose_ver1 import (
    XML_DIR, IMAGE_DIR, OUTPUT_DIR, IMAGE_WIDTH_PX, CAM_IDS, NUM_CAMERAS,
    frame_name, parse_voc_annotation, voc_objects_local_by_camera,
    GT_BOX_COLOR_BGR,
)

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------

# 調べたいフレーム番号をここに列挙する(phase3_bev_multicam_fn_fp_causes_ver1.py
# のFN/FP明細に出てきたフレーム番号をそのまま入れる想定)。
# 空リストのままなら、下のFRAME_START/FRAME_END/FRAME_STEPの範囲を全部処理する。
FRAME_LIST: List[int] = [80, 130, 180, 220, 320, 330, 340, 600, 690, 700, 760, 770, 1000, 1080, 1110, 1150, 1370, 1410, 1420, 1430, 1490, 1530, 1550, 1590, 1600, 1640, 1690, 1700, 1720, 1770, 1780, 1800, 1810, 1820, 1830, 1840, 1850, 1860, 1870, 1880, 1890, 1960, 1990, 2010, 2030, 2070, 2080, 2090, 2100, 2130, 2190, 2300, 2340, 2360, 2500, 2580, 2600, 2610, 2620, 2630, 2690]

FRAME_START = 0
FRAME_END = 290
FRAME_STEP = 10

ANNOTATED_GT_OUT_DIR = OUTPUT_DIR / "miss_frame_bbox_all"
ANNOTATED_GT_OUT_DIR.mkdir(parents=True, exist_ok=True)

POSE_LABEL_COLOR_BGR = {"lying": (60, 200, 60), "standing": (60, 140, 255), None: GT_BOX_COLOR_BGR}


def draw_gt_camera_images(frame_idx: int) -> List[Path]:
    xml_path = Path(XML_DIR) / f"{frame_name(frame_idx)}.xml"
    image_path = Path(IMAGE_DIR) / f"{frame_name(frame_idx)}.jpg"
    if not xml_path.exists() or not image_path.exists():
        print(f"  {frame_name(frame_idx)}: xmlまたは画像が無いのでスキップ "
              f"(xml={xml_path.exists()}, image={image_path.exists()})")
        return []

    full_img = cv2.imread(str(image_path))
    if full_img is None:
        print(f"  {frame_name(frame_idx)}: 画像の読み込みに失敗")
        return []

    gt_objects = parse_voc_annotation(xml_path)
    gt_per_camera = voc_objects_local_by_camera(gt_objects, IMAGE_WIDTH_PX)

    tile_w = IMAGE_WIDTH_PX / NUM_CAMERAS
    saved: List[Path] = []

    for cam_idx, cam_id in enumerate(CAM_IDS):
        x0 = int(cam_idx * tile_w)
        x1 = int((cam_idx + 1) * tile_w)
        crop = full_img[:, x0:x1].copy()

        for it in gt_per_camera.get(cam_id, []):
            bx1, by1, bx2, by2 = [int(round(v)) for v in it["bbox_local"]]
            color = POSE_LABEL_COLOR_BGR.get(it["pose"], GT_BOX_COLOR_BGR)
            cv2.rectangle(crop, (bx1, by1), (bx2, by2), color, 2, lineType=cv2.LINE_AA)
            label = f"{it['name']}" + (f"({it['pose']})" if it["pose"] else "")
            cv2.putText(crop, label, (bx1, max(0, by1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2, cv2.LINE_AA)

        n_boxes = len(gt_per_camera.get(cam_id, []))
        cv2.putText(crop, f"GT n={n_boxes}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, GT_BOX_COLOR_BGR, 2, cv2.LINE_AA)

        out_path = ANNOTATED_GT_OUT_DIR / f"{frame_name(frame_idx)}_{cam_id}_gt.jpg"
        cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        saved.append(out_path)

    return saved


def main():
    if FRAME_LIST:
        frame_indices = sorted(set(FRAME_LIST))
        print(f"FRAME_LISTで指定された{len(frame_indices)}フレームを処理します: {frame_indices}")
    else:
        frame_indices = list(range(FRAME_START, FRAME_END + 1, FRAME_STEP))
        print(f"FRAME_LISTが空なので、{FRAME_START}〜{FRAME_END}(step={FRAME_STEP})の"
              f"{len(frame_indices)}フレームを処理します")

    total_saved = 0
    for frame_idx in frame_indices:
        saved = draw_gt_camera_images(frame_idx)
        total_saved += len(saved)
        if saved:
            print(f"  {frame_name(frame_idx)}: {len(saved)}枚保存")

    print(f"\n合計{total_saved}枚のGT重ね描き画像を保存しました: {ANNOTATED_GT_OUT_DIR}")
    print("render_bev_frames_yolo_zonepose_ver1.pyが生成する"
          "annotated_frames_yolo_zonepose_curved/{frame}_{cam_id}.jpg(YOLO側)と"
          "同じフレーム番号・カメラIDのファイルを並べて見比べることで、"
          "FN(GTにあるのにYOLOが検出していない)/FP(YOLOにあるのにGTが無い)/"
          "crowding(GT同士が画像上でも重なって密集している)を視覚的に確認できる。"
          "YOLO側がまだ無い場合は、先にrender_bev_frames_yolo_zonepose_ver1.pyを"
          "同じフレーム範囲で実行しておくこと。")


if __name__ == "__main__":
    main()