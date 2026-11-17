"""
check_track_gt_consistency.py (単一カメラ限定版)

【vml36で実行】

xml-1fps以下のGT XML(VOC形式、1fpsで連続サンプリング済み)を複数フレーム分
読み込み、TARGET_CAMERA(デフォルトCamera_1)に映っている個体だけに絞って、
各名前(例: cow001)がフレーム間で同一個体を指しているかを自動チェックする。

現在の目的は「単一カメラ内でのトラッキング検証(IDF1のPhase 1)」であり、
カメラ間でのID引き継ぎは別の課題(将来的な多カメラ統合の話)なので、
このスクリプトはTARGET_CAMERA以外のnameは最初から集計対象に含めない
(異なるカメラのタイル座標同士を比較して見かけ上の「大ジャンプ」を
誤検出する問題を避けるため)。

  - 各nameが何フレームに登場するか(連続して映っているか、途中で欠落がないか)
  - 連続フレーム間でのbbox中心の移動量(px、同じカメラ内でのみ計算)。
    1fps(1秒間隔)でも牛はほとんど動かないはずなので、異常に大きなジャンプが
    あれば「実は別個体に同じ名前が誤って振られている」等を疑う根拠になる
  - 1フレームに同じnameが2回以上出現していないか(重複ラベルの検出)

--- 使い方 ---
    python check_track_gt_consistency.py
(パス・フレーム範囲・対象カメラは全て下の設定セクションに直書きしてあるので、
 引数なしで実行する)
"""

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from Mootrack_core import parse_voc_annotation, NUM_CAMERAS, CAM_IDS

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
XML_DIR = "/mnt/vmlqnap02/dataset/MooTrack360/1-hour-farm1-area1-sample-sequence/xml-1fps"
OUTPUT_DIR = Path(
    "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/research/idf1_check"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# チェック対象のフレーム範囲(frame番号は10刻み前提。実在しないフレームはスキップする)
FRAME_START = 0
FRAME_END = 290        # 30フレーム分(0,10,...,290)。アノテーション品質が不安な
                        # 範囲(cow024の持続的重複が始まる000310以降)を避けるため縮小
FRAME_STEP = 10

# ステッチ済み結合画像の幅(カメラ振り分け判定に使用。既存スクリプトと同じ値)
IMAGE_WIDTH_PX = 8832

# 今回の検証対象カメラ。これ以外のカメラに映っているnameは最初から除外する
TARGET_CAMERA = "Camera_1"

# この移動量[px]を超えるフレーム間ジャンプがあれば警告表示する
JUMP_WARN_THRESHOLD_PX = 300.0


def frame_name(idx: int) -> str:
    return f"{idx:06d}"


def assign_camera(center_x_full: float, image_width: float) -> str:
    tile_w = image_width / NUM_CAMERAS
    cam_idx = int(center_x_full // tile_w)
    cam_idx = max(0, min(cam_idx, NUM_CAMERAS - 1))
    return CAM_IDS[cam_idx]


def main():
    xml_dir = Path(XML_DIR)
    frame_indices = list(range(FRAME_START, FRAME_END + 1, FRAME_STEP))

    # name -> [(frame_idx, cx, cy, pose), ...] (frame順、TARGET_CAMERAのみ)
    tracks: Dict[str, List[Tuple[int, float, float, str]]] = {}
    # フレーム内重複チェック用: (frame_idx, name) -> 出現回数
    dup_check: Dict[Tuple[int, str], int] = {}

    found_frames = 0
    missing_frames = []

    for frame_idx in frame_indices:
        xml_path = xml_dir / f"{frame_name(frame_idx)}.xml"
        if not xml_path.exists():
            missing_frames.append(frame_idx)
            continue
        found_frames += 1

        width, height, objects = parse_voc_annotation(xml_path)
        for obj in objects:
            cx = (obj["xmin"] + obj["xmax"]) / 2.0
            cy = (obj["ymin"] + obj["ymax"]) / 2.0
            cam = assign_camera(cx, IMAGE_WIDTH_PX)
            if cam != TARGET_CAMERA:
                continue  # 対象カメラ以外は最初から見ない

            key = (frame_idx, obj["name"])
            dup_check[key] = dup_check.get(key, 0) + 1

            tracks.setdefault(obj["name"], []).append(
                (frame_idx, cx, cy, obj["pose"])
            )

    print(f"読み込んだフレーム数: {found_frames}/{len(frame_indices)}"
          f"(範囲: {FRAME_START}〜{FRAME_END}, step={FRAME_STEP})")
    if missing_frames:
        print(f"  存在しなかったフレーム: {missing_frames[:10]}"
              f"{' ...' if len(missing_frames) > 10 else ''}")

    print(f"\n対象カメラ: {TARGET_CAMERA}")
    print(f"登場した個体名(name)の数: {len(tracks)}")

    dups = [(f, n, c) for (f, n), c in dup_check.items() if c > 1]
    if dups:
        print(f"\n★ 同一フレーム内に同じnameが複数回出現: {len(dups)}件(要確認)")
        for f, n, c in sorted(dups)[:20]:
            print(f"    frame={frame_name(f)} name={n} 出現数={c}")

    csv_path = OUTPUT_DIR / "track_gt_consistency.csv"
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "name", "num_frames", "first_frame", "last_frame",
        "max_jump_px", "max_jump_between_frames", "pose_values",
    ])

    print(f"\n{'name':<12}{'出現数':>6}{'区間':>16}{'最大ジャンプ[px]':>18}  備考")

    n_warn = 0
    for name, entries in sorted(tracks.items()):
        entries.sort(key=lambda e: e[0])
        n = len(entries)
        first_f, last_f = entries[0][0], entries[-1][0]
        poses = sorted(set(e[3] for e in entries if e[3]))

        max_jump = 0.0
        max_jump_pair = ""
        for i in range(1, n):
            f0, x0, y0, _ = entries[i - 1]
            f1, x1, y1, _ = entries[i]
            d = float(np.hypot(x1 - x0, y1 - y0))
            if d > max_jump:
                max_jump = d
                max_jump_pair = f"{frame_name(f0)}->{frame_name(f1)}"

        note = ""
        if max_jump > JUMP_WARN_THRESHOLD_PX:
            note += f"大ジャンプ({max_jump_pair})"
            n_warn += 1

        print(f"{name:<12}{n:>6}{f'{frame_name(first_f)}-{frame_name(last_f)}':>16}"
              f"{max_jump:>18.1f}  {note}")

        csv_writer.writerow([
            name, n, frame_name(first_f), frame_name(last_f),
            round(max_jump, 1), max_jump_pair, ",".join(poses),
        ])

    csv_file.close()
    print(f"\nCSV保存: {csv_path}")
    print(f"\n{JUMP_WARN_THRESHOLD_PX}px を超える大ジャンプがあった個体: {n_warn}件"
          f" / {TARGET_CAMERA}内の全個体数: {len(tracks)}件")
    print("(件数が少なければ、このGTのnameをそのままCamera_1のトラックGTとして"
          "信頼して良い、という判断材料になる)")


if __name__ == "__main__":
    main()