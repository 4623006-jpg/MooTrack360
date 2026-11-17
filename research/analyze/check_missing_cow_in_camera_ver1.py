"""
check_missing_cow_in_camera_ver1.py

特例チェック用スクリプト: 「cow094は常にCamera_3に写っているはず」のような、
特定の個体が特定のカメラに恒常的に存在するはずのケースについて、実際には
その個体のアノテーションが(そのカメラ内で)欠落しているフレームを検出する。

check_gt_annotation_consistency_ver*.pyとは別軸のチェック。あちらは
「前フレームとの差分(出現/消失)」を見ていたが、今回は単純に
「各フレームで、指定した個体が指定したカメラ内にアノテーションされて
いるかどうか」を全フレームについて確認する。

ロジック:
  1. フレームごとにXMLをパースし、各<object>をbboxの中心x座標から
     どのカメラに属すか判定する(center_x // tile_w、他スクリプトと同じ方式)。
  2. TARGET_CAM内にTARGET_NAMEという名前のオブジェクトが1つもなければ、
     「欠落」としてそのフレームを記録する。
  3. (任意)TARGET_NAMEがそのフレームのXML全体にも一切存在しない場合は、
     「カメラ3にいないだけでなく、そもそもアノテーション自体が無い」のか
     区別できるように、別フラグで記録する。

出力:
  - 標準出力: 欠落フレームの一覧
  - txt: OUTPUT_DIR/missing_{TARGET_NAME}_in_{TARGET_CAM}.txt
    (Wordに貼り付けてPDF化できる、番号付きの指示文形式)
  - CSV: OUTPUT_DIR/missing_{TARGET_NAME}_in_{TARGET_CAM}.csv

--- 使い方 ---
    1. XML_DIR/FRAME_START/FRAME_END/FRAME_STEP/IMAGE_WIDTH_PXを
       実際の環境に合わせて書き換える
    2. TARGET_NAME/TARGET_CAMを確認したい個体名・カメラ名に書き換える
       (複数チェックしたい場合はCHECKS配列に追加する)
    3. python check_missing_cow_in_camera_ver1.py
"""

import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Set, Tuple

# ---------------------------------------------------------------------------
# 設定(他のphase3_*/check_*スクリプトと同じ値。実際のパスに合わせて書き換える)
# ---------------------------------------------------------------------------

XML_DIR = "/mnt/vmlqnap02/dataset/MooTrack360/1-hour-farm1-area1-sample-sequence/xml-1fps"
OUTPUT_DIR = Path(
    "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/research/idf1_check"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FRAME_START = 0
FRAME_END = 1470
FRAME_STEP = 10

NUM_CAMERAS = 3
CAM_IDS = ["Camera_1", "Camera_2", "Camera_3"]
IMAGE_WIDTH_PX = 8832

# チェックしたい「個体名 x 必ずいるはずのカメラ」の組。今回の特例
# (cow094は必ずCamera_3にいるはず)の他、同様のケースがあれば追加できる。
CHECKS: List[Tuple[str, str]] = [
    ("cow094", "Camera_3"),
]


def frame_name(idx: int) -> str:
    return f"{idx:06d}"


def parse_objects(xml_path: Path) -> List[dict]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = []
    for obj in root.findall("object"):
        bnd = obj.find("bndbox")
        objects.append({
            "name": obj.find("name").text,
            "xmin": float(bnd.find("xmin").text),
            "ymin": float(bnd.find("ymin").text),
            "xmax": float(bnd.find("xmax").text),
            "ymax": float(bnd.find("ymax").text),
        })
    return objects


def assign_camera(obj: dict, tile_w: float) -> str:
    center_x_full = (obj["xmin"] + obj["xmax"]) / 2.0
    cam_idx = int(center_x_full // tile_w)
    cam_idx = max(0, min(cam_idx, NUM_CAMERAS - 1))
    return CAM_IDS[cam_idx]


def main():
    xml_dir = Path(XML_DIR)
    frame_indices = list(range(FRAME_START, FRAME_END + 1, FRAME_STEP))
    tile_w = IMAGE_WIDTH_PX / NUM_CAMERAS

    for target_name, target_cam in CHECKS:
        print(f"\n=== チェック対象: {target_name} が {target_cam} に写っているはず ===")

        missing_rows: List[dict] = []
        instructions: List[str] = []
        n_processed = 0

        for frame_idx in frame_indices:
            xml_path = xml_dir / f"{frame_name(frame_idx)}.xml"
            if not xml_path.exists():
                continue
            n_processed += 1

            objects = parse_objects(xml_path)
            names_in_target_cam: Set[str] = {
                o["name"] for o in objects if assign_camera(o, tile_w) == target_cam
            }
            names_anywhere: Set[str] = {o["name"] for o in objects}

            if target_name not in names_in_target_cam:
                present_elsewhere = target_name in names_anywhere
                missing_rows.append({
                    "frame": frame_name(frame_idx),
                    "target_name": target_name,
                    "target_cam": target_cam,
                    "present_in_other_camera": present_elsewhere,
                })
                if present_elsewhere:
                    text = (f"{frame_idx}フレーム目で{target_name}が{target_cam}に"
                            f"アノテーションされていません(他のカメラには存在しています。"
                            f"カメラの割り当てミスの可能性があります)。")
                else:
                    text = (f"{frame_idx}フレーム目で{target_name}が{target_cam}に"
                            f"アノテーションされていません(どのカメラにも存在しません)。")
                instructions.append(text)
                print(f"  [欠落] {text}")

        print(f"\n処理フレーム数: {n_processed}/{len(frame_indices)}")
        print(f"{target_name}が{target_cam}で欠落しているフレーム数: {len(missing_rows)}件")

        safe_name = target_name.replace(" ", "_")
        safe_cam = target_cam.replace(" ", "_")

        txt_path = OUTPUT_DIR / f"missing_{safe_name}_in_{safe_cam}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"{target_name} が {target_cam} でアノテーション欠落しているフレーム一覧\n\n")
            for i, text in enumerate(instructions, start=1):
                f.write(f"{i}. {text}\n")
        print(f"txt保存: {txt_path}")

        csv_path = OUTPUT_DIR / f"missing_{safe_name}_in_{safe_cam}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["frame", "target_name", "target_cam", "present_in_other_camera"]
            )
            writer.writeheader()
            writer.writerows(missing_rows)
        print(f"CSV保存: {csv_path}")


if __name__ == "__main__":
    main()