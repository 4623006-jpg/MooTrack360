"""
sam_centroid_pilot.py

【vml36で実行】

牛の斑紋テクスチャへの過学習が疑われるキーポイント方式から方向転換し、
YOLO/GTのbboxをプロンプトとしてSAM(Segment Anything)にゼロショットで
インスタンスマスクを出させ、そのマスクの重心(全マスク画素の平均座標)を
footpoint候補として使う案を試すパイロットスクリプト。

模様に依存しないシルエット形状だけの情報になるので、個体ごとの斑紋への
過学習を避けられる。またfootpoint計算で本当に必要なのは「解剖学的に正確な
背骨上の点」ではなく「牛の重心付近の、高さ平面と交差させるのに妥当な点」
なので、マスク重心の方がこの用途に合っている可能性がある。

standing牛は脚がマスクに含まれる分、生の重心(centroid_x/y)が脚側に
引っ張られやすい。そこでモルフォロジー・オープニングで脚のような細い構造を
除去した「胴体重心」(torso_centroid_x/y)も併せて計算・保存する。
確認用画像では 生の重心=赤丸 / 胴体重心=黄丸 で描画される。

追加学習・ラベリングは一切不要(SAMの事前学習済みモデルをそのまま使うだけ)。

--- 事前準備 ---
    pip install transformers accelerate

--- 使い方 ---
    python sam_centroid_pilot.py
(パスは全て下の設定セクションに直書きしてあるので、引数なしで実行する)
"""

import csv
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import SamModel, SamProcessor

# ---------------------------------------------------------------------------
# 設定(パスはここを書き換える)
# ---------------------------------------------------------------------------
ANNOTATION_XML_PATH = "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/annotation_process/3600_frames_result/annotations/000000.xml"
SOURCE_IMAGE_PATH = "/mnt/vmlqnap02/dataset/MooTrack360/1-hour-farm1-area1-sample-sequence/1-hour-farm1-area1-frames/000000.jpg"
OUTPUT_DIR = "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/make_bev/sam_centroid_test"

SAM_MODEL_NAME = "facebook/sam-vit-huge"
DEVICE = "cuda:0"
BATCH_SIZE = 8  # 1回のモデル呼び出しでまとめて処理するbbox数

# 脚のような細い構造をモルフォロジー・オープニングで除去するためのカーネルサイズ(px)。
# 元画像(8832x2944)上のピクセル単位。脚の太さより大きく、胴体の太さより小さい値が目安。
# 除去しすぎ/しなさすぎに見えたら調整する。
LEG_REMOVAL_KERNEL_SIZE = 21


def parse_gt_objects(xml_path: Path) -> list[dict]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        pose = obj.findtext("pose")
        bnd = obj.find("bndbox")
        xmin = int(float(bnd.findtext("xmin")))
        ymin = int(float(bnd.findtext("ymin")))
        xmax = int(float(bnd.findtext("xmax")))
        ymax = int(float(bnd.findtext("ymax")))
        objects.append({
            "name": name, "pose": pose,
            "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
        })
    return objects


def mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def torso_centroid(mask: np.ndarray, kernel_size: int = LEG_REMOVAL_KERNEL_SIZE) -> tuple[float, float] | None:
    """
    脚のような細い突起をモルフォロジー・オープニング(収縮->膨張)で削り取り、
    残った連結成分のうち最大のもの(=胴体)の重心を返す。
    オープニングで全部消えてしまった場合は、生のマスク重心にフォールバックする。
    """
    mask_u8 = mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opened, connectivity=8)
    if n_labels <= 1:  # ラベル0=背景のみ、前景が残らなかった
        return mask_centroid(mask)

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))
    cx, cy = centroids[largest_label]
    return float(cx), float(cy)


def main():
    xml_path = Path(ANNOTATION_XML_PATH)
    image_path = Path(SOURCE_IMAGE_PATH)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    objects = parse_gt_objects(xml_path)
    print(f"{len(objects)}頭ぶんのGT bboxを読み込み")

    pil_image = Image.open(image_path).convert("RGB")
    cv_image = cv2.imread(str(image_path))
    print(f"元画像サイズ: {pil_image.width}x{pil_image.height}")

    print(f"SAM({SAM_MODEL_NAME})をロード中(初回はHuggingFaceからダウンロードが走ります)...")
    processor = SamProcessor.from_pretrained(SAM_MODEL_NAME)
    model = SamModel.from_pretrained(SAM_MODEL_NAME).to(DEVICE)
    model.eval()

    overlay = cv_image.copy()
    results = []

    # 処理中に外部要因(OOM Killer等)で落ちても結果を失わないよう、
    # 1バッチ処理するごとにCSVへ逐次追記する(最後にまとめて書き出すのではなく)
    csv_path = out_dir / "sam_centroid_results.csv"
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "name", "pose", "xmin", "ymin", "xmax", "ymax",
        "centroid_x", "centroid_y",
        "torso_centroid_x", "torso_centroid_y",
        "score",
    ])

    for start in range(0, len(objects), BATCH_SIZE):
        batch = objects[start:start + BATCH_SIZE]
        boxes = [[o["xmin"], o["ymin"], o["xmax"], o["ymax"]] for o in batch]

        inputs = processor(pil_image, input_boxes=[boxes], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs, multimask_output=False)

        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )[0]  # shape: (num_boxes, 1, H, W)
        scores = outputs.iou_scores.cpu()[0]  # shape: (num_boxes, 1)

        for obj, mask_t, score_t in zip(batch, masks, scores):
            mask = mask_t[0].numpy().astype(bool)
            score = float(score_t[0])
            c = mask_centroid(mask)
            if c is None:
                print(f"  {obj['name']} ({obj['pose']}): マスク空、スキップ")
                continue
            cx, cy = c
            tc = torso_centroid(mask)
            tcx, tcy = tc if tc is not None else (cx, cy)

            print(f"  {obj['name']} ({obj['pose']}) "
                  f"bbox=({obj['xmin']},{obj['ymin']},{obj['xmax']},{obj['ymax']}) "
                  f"iou_score={score:.3f} -> 重心=({cx:.1f},{cy:.1f}) 胴体重心=({tcx:.1f},{tcy:.1f})")
            results.append({
                **obj, "centroid_x": cx, "centroid_y": cy,
                "torso_centroid_x": tcx, "torso_centroid_y": tcy, "score": score,
            })
            csv_writer.writerow([
                obj["name"], obj["pose"], obj["xmin"], obj["ymin"], obj["xmax"], obj["ymax"],
                round(cx, 2), round(cy, 2), round(tcx, 2), round(tcy, 2), round(score, 4),
            ])

            # マスク輪郭を薄く重ね、生の重心(赤)と胴体重心(黄)の両方に点を打つ
            mask_u8 = mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
            cv2.circle(overlay, (int(cx), int(cy)), 6, (0, 0, 255), -1)
            cv2.circle(overlay, (int(tcx), int(tcy)), 6, (0, 255, 255), -1)
            cv2.rectangle(overlay, (obj["xmin"], obj["ymin"]), (obj["xmax"], obj["ymax"]), (255, 255, 0), 1)

        csv_file.flush()
        print(f"  ...{min(start + BATCH_SIZE, len(objects))}/{len(objects)}頭 処理済み "
              f"(CSVに逐次保存済み)")

    csv_file.close()
    print(f"CSV保存: {csv_path} (次のBEV変換スクリプトの入力になります)")

    overlay_path = out_dir / "overlay_sam_centroid.jpg"
    cv2.imwrite(str(overlay_path), overlay)
    print(f"\n確認用画像: {overlay_path}")
    print(f"平均iou_score: {np.mean([r['score'] for r in results]):.3f}" if results else "有効な結果なし")
    print("(元画像は大きいので、ローカルに転送してから拡大して見てください)")


if __name__ == "__main__":
    main()