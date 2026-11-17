"""
render_gt_label_bbox_ver1.py

指定したフレーム・カメラ・ラベル(牛名)のGT bboxを強調した画像を保存する。
アノテーションの確認用。他のスクリプトに依存しない(このファイル単体で動く)。

TARGETS に (フレーム番号, カメラ, 牛名) を列挙する。
  - カメラ  : "Camera_1" / "Camera_2" / "Camera_3" / None(Noneなら3台すべて)
  - 牛名    : "cow078" や "078" や 78 のどれでもよい / None(Noneならそのカメラの全bboxを強調)
同じカメラに同じ牛名が複数あれば、すべて強調して「同名が複数」と画像に表示する
(アノテーションの重複の確認用)。

出力: OUTPUT_DIR / gt_label_bbox / {frame}_{camera}_{label}.jpg        カメラ全体(強調あり)
                                  {frame}_{camera}_{label}_zoom{k}.jpg  強調したbbox周辺の拡大

使い方:
    1. TARGETS を編集する
    2. python render_gt_label_bbox_ver1.py
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2

XML_DIR = "/mnt/vmlqnap02/dataset/MooTrack360/1-hour-farm1-area1-sample-sequence/xml-1fps"
IMAGE_DIR = "/mnt/vmlqnap02/dataset/MooTrack360/1-hour-farm1-area1-sample-sequence/1-hour-farm1-area1-frames"
OUTPUT_DIR = Path(
    "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/research/idf1_check"
)
OUT_DIR = OUTPUT_DIR / "gt_label_bbox"

IMAGE_WIDTH_PX = 8832
CAM_IDS = ["Camera_1", "Camera_2", "Camera_3"]

# (フレーム番号, カメラ, 牛名)
TARGETS: List[Tuple[int, Optional[str], Union[str, int, None]]] = [
    (1090, "Camera_2", "cow094"),
]

SHOW_OTHERS = True        # 強調対象以外のbboxも細い灰色線+牛番号で描く(位置関係の確認用)
ZOOM_MARGIN_RATIO = 1.0   # 拡大画像で、bboxの幅/高さの何倍ぶん周囲を含めるか
ZOOM_MIN_SIZE = 400       # 拡大画像の最小の一辺(px)

HIGHLIGHT_COLOR = (0, 0, 255)   # 赤
OTHER_COLOR = (170, 170, 170)


def parse_voc_annotation(xml_path):
    objects = []
    for obj in ET.parse(xml_path).getroot().findall("object"):
        pose_el = obj.find("pose")
        bnd = obj.find("bndbox")
        objects.append({
            "name": obj.find("name").text,
            "pose": pose_el.text.strip().lower() if pose_el is not None and pose_el.text else None,
            "xmin": float(bnd.find("xmin").text), "ymin": float(bnd.find("ymin").text),
            "xmax": float(bnd.find("xmax").text), "ymax": float(bnd.find("ymax").text),
        })
    return objects


def normalize_label(label) -> Optional[str]:
    """'cow078' / '078' / 78 -> 'cow078'"""
    if label is None:
        return None
    m = re.search(r"\d+", str(label))
    return f"cow{int(m.group()):03d}" if m else str(label)


def same_label(a: str, b: str) -> bool:
    ma, mb = re.search(r"\d+", a), re.search(r"\d+", b)
    if ma and mb:
        return int(ma.group()) == int(mb.group())
    return a == b


def render_one(frame_idx: int, camera: str, label, objects, full_img) -> List[Path]:
    tile_w = IMAGE_WIDTH_PX / len(CAM_IDS)
    cam_idx = CAM_IDS.index(camera)
    x0, x1 = int(cam_idx * tile_w), int((cam_idx + 1) * tile_w)
    crop = full_img[:, x0:x1].copy()

    in_cam = []
    for o in objects:
        cx = (o["xmin"] + o["xmax"]) / 2.0
        if min(int(cx // tile_w), len(CAM_IDS) - 1) == cam_idx:
            in_cam.append({**o, "bbox": (o["xmin"] - cam_idx * tile_w, o["ymin"],
                                         o["xmax"] - cam_idx * tile_w, o["ymax"])})

    targets = [o for o in in_cam if label is None or same_label(o["name"], label)]
    others = [o for o in in_cam if o not in targets]

    if SHOW_OTHERS:
        for o in others:
            bx1, by1, bx2, by2 = [int(round(v)) for v in o["bbox"]]
            cv2.rectangle(crop, (bx1, by1), (bx2, by2), OTHER_COLOR, 1, cv2.LINE_AA)
            cv2.putText(crop, o["name"].replace("cow", ""), (bx1, max(0, by1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, OTHER_COLOR, 1, cv2.LINE_AA)
    for o in targets:
        bx1, by1, bx2, by2 = [int(round(v)) for v in o["bbox"]]
        cv2.rectangle(crop, (bx1, by1), (bx2, by2), HIGHLIGHT_COLOR, 4, cv2.LINE_AA)
        text = o["name"] + (f"({o['pose']})" if o["pose"] else "")
        cv2.putText(crop, text, (bx1, max(0, by1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, HIGHLIGHT_COLOR, 3, cv2.LINE_AA)

    status = f"frame {frame_idx:06d} {camera} label={normalize_label(label) or 'ALL'}  該当bbox={len(targets)}"
    if label is not None and len(targets) > 1:
        status += "  <同名が複数>"
    if label is not None and len(targets) == 0:
        status += "  <このカメラに該当ラベル無し>"
    cv2.putText(crop, status, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3, cv2.LINE_AA)

    tag = normalize_label(label) or "ALL"
    base = f"{frame_idx:06d}_{camera}_{tag}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    p = OUT_DIR / f"{base}.jpg"
    cv2.imwrite(str(p), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    saved.append(p)

    if label is not None:
        h, w = crop.shape[:2]
        for k, o in enumerate(targets, start=1):
            bx1, by1, bx2, by2 = o["bbox"]
            bw, bh = bx2 - bx1, by2 - by1
            half_w = max(bw * (0.5 + ZOOM_MARGIN_RATIO), ZOOM_MIN_SIZE / 2)
            half_h = max(bh * (0.5 + ZOOM_MARGIN_RATIO), ZOOM_MIN_SIZE / 2)
            cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
            zx1, zx2 = int(max(0, cx - half_w)), int(min(w, cx + half_w))
            zy1, zy2 = int(max(0, cy - half_h)), int(min(h, cy + half_h))
            zp = OUT_DIR / f"{base}_zoom{k}.jpg"
            cv2.imwrite(str(zp), crop[zy1:zy2, zx1:zx2], [cv2.IMWRITE_JPEG_QUALITY, 92])
            saved.append(zp)
    return saved


def main():
    total = 0
    for frame_idx, camera, label in TARGETS:
        xml_path = Path(XML_DIR) / f"{frame_idx:06d}.xml"
        image_path = Path(IMAGE_DIR) / f"{frame_idx:06d}.jpg"
        if not xml_path.exists() or not image_path.exists():
            print(f"{frame_idx:06d}: xmlまたは画像が無いのでスキップ (xml={xml_path.exists()}, image={image_path.exists()})")
            continue
        full_img = cv2.imread(str(image_path))
        if full_img is None:
            print(f"{frame_idx:06d}: 画像の読み込みに失敗")
            continue
        objects = parse_voc_annotation(xml_path)
        cameras = CAM_IDS if camera is None else [camera]
        for cam in cameras:
            saved = render_one(frame_idx, cam, label, objects, full_img)
            total += len(saved)
            print(f"{frame_idx:06d} {cam} {normalize_label(label) or 'ALL'}: {len(saved)}枚保存")
    print(f"\n合計{total}枚を保存しました: {OUT_DIR}")


if __name__ == "__main__":
    main()