"""
check_gt_annotation_consistency_ver4.py

check_gt_annotation_consistency_ver3.pyとの違い: ver3はFN/FPフレーム近傍で
only_in_next/only_in_prev(重複個体を除いた差分)をテキストで一覧表示するだけ
だったが、ver4ではその「差分の個体」のbboxを実画像上に描いた画像も保存する。
これにより、「本当に新しい個体が現れた/消えた」のか「同じ個体に別の名前が
付けられてしまった(タイポ・個体の取り違え)」のかを画像で直接判断できる。

カメラ・YOLO・Mootrack_coreには依存しない完全に独立したスクリプト(パス依存の
問題を避けるため、bbox→カメラの振り分けロジックもこのファイル内に素朴な形で
持たせている)。

やること:
  1. check_gt_annotation_consistency_ver3.pyと同じく、FN/FPフレーム近傍で
     only_in_next/only_in_prevを検出する。
  2. only_in_next(今回だけ出現した個体名)については、そのcur_frame側の画像に
     bboxを描いて保存する。
  3. only_in_prev(前回だけ存在した個体名)については、そのprev_frame側の画像に
     bboxを描いて保存する。
  4. 1個体が複数カメラに映っている(=同じフレーム内に同名で複数bboxがある)
     場合は、その全部の画像を保存する。

出力:
  - 標準出力: ver3と同じ(FN/FP近傍のonly_in_next/only_in_prev一覧)
  - CSV: OUTPUT_DIR/gt_name_nonoverlap_near_fn_fp.csv / gt_name_nonoverlap_all.csv
  - 画像: OUTPUT_DIR/gt_name_diff_bbox_images/{frame}_{cam_id}_{name}_{next|prev}.jpg

--- 使い方 ---
    1. XML_DIR/IMAGE_DIR/FRAME_START/FRAME_END/FRAME_STEP/IMAGE_WIDTH_PXを
       実際の環境に合わせて書き換える
    2. FN_FP_FRAMESを最新のphase3_bev_multicam_frame_zonepose_ver2.pyの
       出力結果に合わせて更新する
    3. python check_gt_annotation_consistency_ver4.py
"""

import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Set

import cv2

# ---------------------------------------------------------------------------
# 設定(他のphase3_*スクリプトと同じ値。実際のパスに合わせて書き換えること)
# ---------------------------------------------------------------------------

XML_DIR = "/mnt/vmlqnap02/dataset/MooTrack360/1-hour-farm1-area1-sample-sequence/xml-1fps"
IMAGE_DIR = "/mnt/vmlqnap02/dataset/MooTrack360/1-hour-farm1-area1-sample-sequence/1-hour-farm1-area1-frames"
OUTPUT_DIR = Path(
    "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/research/idf1_check"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BBOX_IMG_OUT_DIR = OUTPUT_DIR / "gt_name_diff_bbox_images"
BBOX_IMG_OUT_DIR.mkdir(parents=True, exist_ok=True)

FRAME_START = 0
FRAME_END = 1470
FRAME_STEP = 10

NUM_CAMERAS = 3
CAM_IDS = ["Camera_1", "Camera_2", "Camera_3"]
IMAGE_WIDTH_PX = 8832

# phase3_bev_multicam_frame_zonepose_ver2.pyの出力(FN/FPが発生したフレーム一覧)
# をそのまま埋め込んだもの。最新の結果に合わせて更新すること。
FN_FP_FRAMES = {
    620, 560, 1320, 650, 810, 1000, 1380, 100, 130, 140, 220, 270, 280, 340,
    400, 480, 540, 550, 570, 580, 590, 1080, 1390, 80, 110, 120, 150, 160,
    170, 180, 190, 200, 210, 230, 240, 250, 260, 290, 320, 330, 410, 500,
    510, 520, 530, 610, 630, 670, 690, 700, 760, 770, 820, 840, 860, 870,
    880, 890, 910, 990, 1040, 1050, 1110, 1120, 1140, 1150, 1160, 1190,
    1200, 1220, 1260, 1280, 1290, 1340, 1360, 1370,
}

# FN/FPフレームから何フレーム分(FRAME_STEP単位)離れていれば「隣接」とみなすか
NEARBY_STEPS = 1

BBOX_COLOR_BGR = (0, 0, 255)  # 赤(目立たせる)


def frame_name(idx: int) -> str:
    return f"{idx:06d}"


def parse_objects(xml_path: Path) -> List[dict]:
    """XMLの<object>を全部、bbox付きでパースする。"""
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


def is_near_fn_fp(frame_idx: int) -> bool:
    for offset in range(-NEARBY_STEPS, NEARBY_STEPS + 1):
        if (frame_idx + offset * FRAME_STEP) in FN_FP_FRAMES:
            return True
    return False


def save_bbox_images_for_name(frame_idx: int, name: str, objects: List[dict], tag: str) -> int:
    """指定フレーム・指定個体名のbboxを全部(複数カメラに映っていれば全部)
    画像に描いて保存する。tagは"next"/"prev"(どちらの差分で見つかったかの印)。
    戻り値は保存した枚数。"""
    image_path = Path(IMAGE_DIR) / f"{frame_name(frame_idx)}.jpg"
    if not image_path.exists():
        print(f"    (画像が見つからないためスキップ: {image_path})")
        return 0
    full_img = cv2.imread(str(image_path))
    if full_img is None:
        print(f"    (画像の読み込みに失敗: {image_path})")
        return 0

    tile_w = IMAGE_WIDTH_PX / NUM_CAMERAS
    matches = [obj for obj in objects if obj["name"] == name]
    n_saved = 0

    for i, obj in enumerate(matches):
        center_x_full = (obj["xmin"] + obj["xmax"]) / 2.0
        cam_idx = int(center_x_full // tile_w)
        cam_idx = max(0, min(cam_idx, NUM_CAMERAS - 1))
        cam_id = CAM_IDS[cam_idx]
        x0 = int(cam_idx * tile_w)
        x1 = int((cam_idx + 1) * tile_w)
        crop = full_img[:, x0:x1].copy()

        bx1 = int(round(obj["xmin"] - x0))
        by1 = int(round(obj["ymin"]))
        bx2 = int(round(obj["xmax"] - x0))
        by2 = int(round(obj["ymax"]))
        cv2.rectangle(crop, (bx1, by1), (bx2, by2), BBOX_COLOR_BGR, 3, lineType=cv2.LINE_AA)
        label = f"{name} ({'今回だけ' if tag == 'next' else '前回だけ'})"
        cv2.putText(crop, label, (bx1, max(0, by1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, BBOX_COLOR_BGR, 2, cv2.LINE_AA)

        suffix = f"_{i}" if len(matches) > 1 else ""
        out_path = BBOX_IMG_OUT_DIR / f"{frame_name(frame_idx)}_{cam_id}_{name}_{tag}{suffix}.jpg"
        cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        n_saved += 1

    if not matches:
        print(f"    (警告: frame {frame_name(frame_idx)}のXMLに名前'{name}'のbboxが見つからない)")

    return n_saved


def main():
    xml_dir = Path(XML_DIR)
    frame_indices = list(range(FRAME_START, FRAME_END + 1, FRAME_STEP))

    prev_objects: Optional[List[dict]] = None
    prev_frame_idx: Optional[int] = None
    all_rows: List[dict] = []
    near_rows: List[dict] = []
    n_processed = 0
    n_images_saved = 0

    for frame_idx in frame_indices:
        xml_path = xml_dir / f"{frame_name(frame_idx)}.xml"
        if not xml_path.exists():
            continue

        cur_objects = parse_objects(xml_path)
        cur_names: Set[str] = {o["name"] for o in cur_objects}
        n_processed += 1

        if prev_objects is not None:
            prev_names: Set[str] = {o["name"] for o in prev_objects}
            only_in_next = sorted(cur_names - prev_names)
            only_in_prev = sorted(prev_names - cur_names)
            n_common = len(cur_names & prev_names)

            is_relevant = is_near_fn_fp(prev_frame_idx) or is_near_fn_fp(frame_idx)

            row = {
                "prev_frame": frame_name(prev_frame_idx),
                "cur_frame": frame_name(frame_idx),
                "only_in_next": ";".join(only_in_next),
                "only_in_prev": ";".join(only_in_prev),
                "n_common": n_common,
                "near_fn_fp": is_relevant,
            }
            all_rows.append(row)

            if is_relevant and (only_in_next or only_in_prev):
                near_rows.append(row)
                print(f"  {row['prev_frame']} -> {row['cur_frame']}  (共通{n_common}個体は省略)")
                print(f"    only_in_next(今回だけ): {only_in_next}")
                print(f"    only_in_prev(前回だけ): {only_in_prev}")

                for name in only_in_next:
                    n = save_bbox_images_for_name(frame_idx, name, cur_objects, "next")
                    n_images_saved += n
                for name in only_in_prev:
                    n = save_bbox_images_for_name(prev_frame_idx, name, prev_objects, "prev")
                    n_images_saved += n

        prev_objects = cur_objects
        prev_frame_idx = frame_idx

    print(f"\n処理フレーム数: {n_processed}/{len(frame_indices)}")
    print(f"FN/FPフレーム近傍で差分があったペア: {len(near_rows)}件")
    print(f"保存したbbox画像: {n_images_saved}枚 -> {BBOX_IMG_OUT_DIR}")

    n_all_diff = sum(1 for r in all_rows if r["only_in_next"] or r["only_in_prev"])
    if n_all_diff > 0:
        ratio = len(near_rows) / n_all_diff * 100
        print(f"\n(参考: 全差分ペア={n_all_diff}件のうちFN/FP近傍={len(near_rows)}件, {ratio:.1f}%)")

    near_csv_path = OUTPUT_DIR / "gt_name_nonoverlap_near_fn_fp.csv"
    with open(near_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["prev_frame", "cur_frame", "only_in_next", "only_in_prev", "n_common", "near_fn_fp"]
        )
        writer.writeheader()
        writer.writerows(near_rows)
    print(f"\nCSV保存(FN/FP近傍のみ): {near_csv_path}")

    all_csv_path = OUTPUT_DIR / "gt_name_nonoverlap_all.csv"
    with open(all_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["prev_frame", "cur_frame", "only_in_next", "only_in_prev", "n_common", "near_fn_fp"]
        )
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"CSV保存(全ペア): {all_csv_path}")


if __name__ == "__main__":
    main()