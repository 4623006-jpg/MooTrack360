"""
check_gt_annotation_consistency_ver6.py

check_gt_annotation_consistency_ver5.pyとの違い: ver5の「消失側(prev)」の
指示文は「{prev_frame}フレーム目で{cow}が消えたので、{cur_frame}フレーム目
を確認してください」という向きだったが、これは論理的に逆だった。実際に
「cowが見当たらない(消えている)」と分かるのは、個体がまだ存在していた
prev_frameではなく、存在しなくなったcur_frame(今のフレーム)を見た時点。
ver6では、この向きを修正:

    消失側(prev): 「{cur_frame}フレーム目の{camera}カメラで{cow}が消えて
                   いるので、{prev_frame}フレーム目を確認してください。」
    出現側(next): 「{cur_frame}フレーム目の{camera}カメラで{cow}が現れた
                   ので、{prev_frame}フレーム目を確認してください。」
                   (こちらはver5から向きの変更なし)

どちらのケースも「異変に気づくのはcur_frame(今のフレーム)」「確認で見に
行くのはprev_frame(1つ前のフレーム)」という統一的な向きになる。
消失側のカメラ情報は、cur_frame側にはそもそもbboxが無いため、引き続き
prev_frame側のbboxから取得する(最後に見えていたカメラ=確認すべきカメラ)。

出力:
  - 標準出力: 生成した指示文の一覧
  - txt: OUTPUT_DIR/gt_label_mistake_instructions.txt
  - CSV: OUTPUT_DIR/gt_name_nonoverlap_near_fn_fp.csv / gt_name_nonoverlap_all.csv
    (ver3/ver4/ver5と同じ形式、参考情報として維持)

--- 使い方 ---
    1. XML_DIR/FRAME_START/FRAME_END/FRAME_STEP/IMAGE_WIDTH_PXを
       実際の環境に合わせて書き換える
    2. FN_FP_FRAMESを最新のphase3_bev_multicam_frame_zonepose_ver2.pyの
       出力結果に合わせて更新する
    3. python check_gt_annotation_consistency_ver6.py
"""

import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# 設定(他のphase3_*スクリプトと同じ値。実際のパスに合わせて書き換えること)
# ---------------------------------------------------------------------------

XML_DIR = "/mnt/vmlqnap02/dataset/MooTrack360/1-hour-farm1-area1-sample-sequence/xml-1fps"
OUTPUT_DIR = Path(
    "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/research/analyze_check"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FRAME_START = 0
FRAME_END = 1470
FRAME_STEP = 10

NUM_CAMERAS = 3
CAM_IDS = ["Camera_1", "Camera_2", "Camera_3"]
IMAGE_WIDTH_PX = 8832

# phase3_bev_multicam_frame_zonepose_ver2.pyの出力(FN/FPが発生したフレーム一覧)
# をそのまま埋め込んだもの。最新の結果に合わせて更新すること。
FN_FP_FRAMES = {
    130,620,640,650
}

# FN/FPフレームから何フレーム分(FRAME_STEP単位)離れていれば「隣接」とみなすか
NEARBY_STEPS = 1


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


def assign_camera(obj: dict, tile_w: float) -> str:
    center_x_full = (obj["xmin"] + obj["xmax"]) / 2.0
    cam_idx = int(center_x_full // tile_w)
    cam_idx = max(0, min(cam_idx, NUM_CAMERAS - 1))
    return CAM_IDS[cam_idx]


def cameras_for_name(objects: List[dict], name: str, tile_w: float) -> List[str]:
    """指定した個体名が写っている全カメラ(複数カメラに重複して写っている
    場合は複数返す)。"""
    cams = [assign_camera(o, tile_w) for o in objects if o["name"] == name]
    # 重複除去(順序維持)
    seen: Set[str] = set()
    uniq = []
    for c in cams:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def is_near_fn_fp(frame_idx: int) -> bool:
    for offset in range(-NEARBY_STEPS, NEARBY_STEPS + 1):
        if (frame_idx + offset * FRAME_STEP) in FN_FP_FRAMES:
            return True
    return False


def main():
    xml_dir = Path(XML_DIR)
    frame_indices = list(range(FRAME_START, FRAME_END + 1, FRAME_STEP))
    tile_w = IMAGE_WIDTH_PX / NUM_CAMERAS

    prev_objects: Optional[List[dict]] = None
    prev_frame_idx: Optional[int] = None
    all_rows: List[dict] = []
    near_rows: List[dict] = []
    instructions: List[str] = []
    n_processed = 0

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

                # 出現側(next)・消失側(prev)ともに、異変に気づくのは今回の
                # フレーム(frame_idx)であり、確認のために見に行くのは1つ前の
                # フレーム(prev_frame_idx)という向きで統一する。
                for name in only_in_next:
                    cams = cameras_for_name(cur_objects, name, tile_w)
                    cam_str = "・".join(cams) if cams else "不明なカメラ"
                    text = (f"{frame_idx}フレーム目の{cam_str}カメラで{name}が現れたので、"
                            f"{prev_frame_idx}フレーム目と比較して確認してください。")
                    instructions.append(text)
                    print(f"    [指示] {text}")

                # 消失側(prev): カメラ情報はcur_frame側にbboxが無いため、
                # 最後に見えていたprev_frame側のbboxから取得する
                # (=確認すべきカメラそのもの)。
                for name in only_in_prev:
                    cams = cameras_for_name(prev_objects, name, tile_w)
                    cam_str = "・".join(cams) if cams else "不明なカメラ"
                    text = (f"{frame_idx}フレーム目の{cam_str}カメラで{name}が消えているので、"
                            f"{prev_frame_idx}フレーム目と比較して確認してください。")
                    instructions.append(text)
                    print(f"    [指示] {text}")

        prev_objects = cur_objects
        prev_frame_idx = frame_idx

    print(f"\n処理フレーム数: {n_processed}/{len(frame_indices)}")
    print(f"FN/FPフレーム近傍で差分があったペア: {len(near_rows)}件")
    print(f"生成した指示文: {len(instructions)}件")

    n_all_diff = sum(1 for r in all_rows if r["only_in_next"] or r["only_in_prev"])
    if n_all_diff > 0:
        ratio = len(near_rows) / n_all_diff * 100
        print(f"\n(参考: 全差分ペア={n_all_diff}件のうちFN/FP近傍={len(near_rows)}件, {ratio:.1f}%)")

    # --- txt出力(Wordに貼り付けてPDF化する想定の番号付きリスト) ---
    txt_path = OUTPUT_DIR / "gt_label_mistake_instructions.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("GTラベルミス確認リスト\n")
        f.write("以下の各項目について、指定されたフレームの指定カメラを目視で確認してください。\n\n")
        for i, text in enumerate(instructions, start=1):
            f.write(f"{i}. {text}\n")
    print(f"\ntxt保存(Word貼り付け用): {txt_path}")

    # --- 参考CSV(ver3/ver4と同じ形式) ---
    near_csv_path = OUTPUT_DIR / "gt_name_nonoverlap_near_fn_fp.csv"
    with open(near_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["prev_frame", "cur_frame", "only_in_next", "only_in_prev", "n_common", "near_fn_fp"]
        )
        writer.writeheader()
        writer.writerows(near_rows)
    print(f"CSV保存(FN/FP近傍のみ): {near_csv_path}")

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