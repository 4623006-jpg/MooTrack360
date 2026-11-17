"""
evaluate_gt_distance_matching.py の「center方式(bbox中心1点)+ 距離閾値」による
個体統合アルゴリズムを、GTのbboxではなく実際のYOLO検出結果に対して適用し、精度を
検証するスクリプト。

共通のカメラキャリブレーション・幾何変換・Hungarian割当・Union-Find・評価指標の
計算は mootrack_core.py に集約されているので、ここでは
  - YOLO推論・カメラ別confidenceしきい値によるバケツ分け
  - 姿勢(class_id)別の高さでのworld座標変換
  - YOLO検出へのGT name仮割当(評価専用)
  - 実行フロー(main)
だけを持つ。

evaluate_gt_distance_matching.py との違いはここだけ:
  - bboxをアノテーションXMLからではなく、YOLO(model.predict)の検出結果から得る
  - 高さ(CENTER_HEIGHT_M)を固定値ではなく、検出されたclass_id(姿勢)によって変える
      class_id 0 (lying/座位)    -> 0.9m
      class_id 1 (standing/立位) -> 1.3m
    (gt_center_height_gridsearch.py によるGTベースのグリッドサーチでF1が最大になった
     組み合わせ(lying=0.9, standing=1.3, 距離閾値=2.8, F1=0.694)を採用している。
     class_idの割当が逆だった場合は CLASS_HEIGHT_M を入れ替えること)
  - YOLO検出には個体名(name)が無いので、評価のためだけに、同じカメラタイル内で
    2D IoU(画像平面上のIoU。ワールド座標のPolygon IoUとは別物)が最も高いGTのbboxから
    nameを仮に割り当てる(GT_MATCH_IOU_THR未満なら「対応するGTなし」としてnameはNone)。
    このname割当はあくまで精度評価用であり、実際の統合アルゴリズム(カメラ間の距離に
    よるHungarian割当+Union-Find)には一切使っていない。
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
from ultralytics import YOLO

from Mootrack_core import (
    NUM_CAMERAS, CAM_IDS, ROT_FMT,
    Pose, get_intrinsics,
    load_floorplan_cameras, apply_camera_overrides, resolve_poses,
    unproject_to_floor, parse_voc_annotation,
    compute_camera_assignments_by_distance,
    build_keys_and_name_lookup, evaluate_merge_accuracy,
)


# ---------------------------------------------------------------------------
# このスクリプト固有の設定
# ---------------------------------------------------------------------------

FLOORPLAN_JSON_PATH = "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/calibration/Farm2_Area_1_Artboard_config_.json"
SOURCE_IMAGE_PATH = "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/annotation_process/3600_frames_result/000000.jpg"
ANNOTATION_XML_PATH = "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/annotation_process/3600_frames_result/annotations/000000.xml"
MODEL_PATH = "/mnt/vmlqnap02/dataset/MooTrack360/Trained models/yolo11x/train_ultralytics/weights/best.pt"

YOLO_PREDICT_CONF = 0.01
YOLO_PREDICT_IOU = 0.5
YOLO_PREDICT_IMGSZ = 2560
YOLO_PREDICT_CLASSES = [0, 1]

CAMERA_CONF_THR = {
    "Camera_1": 0.5,
    "Camera_2": 0.5,
    "Camera_3": 0.2,
}

# ★ 姿勢(class_id)ごとの接地点の高さ ★
# gt_center_height_gridsearch.py によるGTベースのグリッドサーチ結果(F1=0.694)を採用。
# 解剖学的な体高とは一致しないが、カメラ較正誤差込みで最も精度が良かった値。
# class_idの割当が逆だった場合はここを入れ替えること。
CLASS_HEIGHT_M = {
    0: 0.9,   # lying / 座位
    1: 1.3,   # standing / 立位
}
DEFAULT_HEIGHT_M = 1.1  # 想定外のclass_idが来た場合のフォールバック(中間的な値)

# YOLO検出にGTのnameを対応付けるための2D IoU閾値(評価用のみ。統合アルゴリズムには不使用)
GT_MATCH_IOU_THR = 0.3

# 検証したい距離閾値の一覧(単位: m)
DIST_THRESHOLDS_M = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]


# ---------------------------------------------------------------------------
# YOLO検出 -> raw_boxes -> カメラ別バケツ分け(カメラ別confidenceしきい値で足切り)
# ---------------------------------------------------------------------------

def yolo_results_to_raw_boxes(results, conf_threshold: float = YOLO_PREDICT_CONF) -> List[dict]:
    boxes = results.boxes
    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    raw_boxes = []
    for (x1, y1, x2, y2), c, cf in zip(xyxy, cls, conf):
        if cf < conf_threshold:
            continue
        raw_boxes.append({
            "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
            "class_id": int(c), "conf": float(cf),
        })
    return raw_boxes


def bucket_detections_by_camera(
    raw_boxes: List[dict],
    tile_w: float,
    cam_ids: List[str] = CAM_IDS,
    cam_conf_thr: Dict[str, float] = CAMERA_CONF_THR,
) -> Dict[str, List[dict]]:
    num_cameras = len(cam_ids)
    bucketed: Dict[str, List[dict]] = {cam_id: [] for cam_id in cam_ids}
    for box in raw_boxes:
        center_x = (box["x1"] + box["x2"]) / 2.0
        cam_idx = int(center_x // tile_w)
        cam_idx = max(0, min(cam_idx, num_cameras - 1))
        cam_id = cam_ids[cam_idx]
        if box["conf"] < cam_conf_thr.get(cam_id, 0.0):
            continue
        bucketed[cam_id].append(box)
    return bucketed


def bucket_gt_by_camera(
    objects: List[dict],
    tile_w: float,
    cam_ids: List[str] = CAM_IDS,
) -> Dict[str, List[dict]]:
    num_cameras = len(cam_ids)
    bucketed: Dict[str, List[dict]] = {cam_id: [] for cam_id in cam_ids}
    for obj in objects:
        center_x = (obj["xmin"] + obj["xmax"]) / 2.0
        cam_idx = int(center_x // tile_w)
        cam_idx = max(0, min(cam_idx, num_cameras - 1))
        bucketed[cam_ids[cam_idx]].append(obj)
    return bucketed


# ---------------------------------------------------------------------------
# YOLO検出へのGT name仮割当(評価専用。2D IoU、フル画像座標系で計算)
# ---------------------------------------------------------------------------

def bbox_iou_2d(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def assign_gt_names(
    bucketed_yolo: Dict[str, List[dict]],
    bucketed_gt: Dict[str, List[dict]],
    iou_threshold: float = GT_MATCH_IOU_THR,
) -> Dict[str, List[dict]]:
    """
    各YOLO検出に対し、同じカメラタイル内で2D IoUが最大のGT bboxのnameを仮割当する。
    iou_threshold未満なら対応するGTなしとして name=None にする(実運用のマッチングには
    使わない。評価のためだけの処理)。
    """
    result: Dict[str, List[dict]] = {}
    for cam_id, dets in bucketed_yolo.items():
        gt_list = bucketed_gt.get(cam_id, [])
        new_dets = []
        for det in dets:
            det_box = (det["x1"], det["y1"], det["x2"], det["y2"])
            best_iou = 0.0
            best_name = None
            for gt in gt_list:
                gt_box = (gt["xmin"], gt["ymin"], gt["xmax"], gt["ymax"])
                iou = bbox_iou_2d(det_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_name = gt["name"]
            new_det = dict(det)
            new_det["gt_name"] = best_name if best_iou >= iou_threshold else None
            new_det["gt_match_iou"] = best_iou
            new_dets.append(new_det)
        result[cam_id] = new_dets
    return result


# ---------------------------------------------------------------------------
# YOLO検出(class_id, gt_name付き) -> World座標の1点(姿勢に応じた高さで逆投影)
# ---------------------------------------------------------------------------

def build_world_points_from_yolo(
    bucketed_with_names: Dict[str, List[dict]],
    poses: Dict[str, Pose],
    tile_w: float,
) -> Dict[str, List[Tuple[Optional[str], np.ndarray]]]:
    per_camera: Dict[str, List[Tuple[Optional[str], np.ndarray]]] = {cid: [] for cid in CAM_IDS}

    for cam_idx, cam_id in enumerate(CAM_IDS):
        dets = bucketed_with_names.get(cam_id, [])
        if not dets:
            continue

        fx, fy, cx, cy, xi, alpha = get_intrinsics(cam_id)
        pose = poses[cam_id]

        for det in dets:
            center_x_full = (det["x1"] + det["x2"]) / 2.0
            center_y = (det["y1"] + det["y2"]) / 2.0
            local_x = center_x_full - cam_idx * tile_w
            height_m = CLASS_HEIGHT_M.get(det["class_id"], DEFAULT_HEIGHT_M)

            pts_px = np.array([[local_x, center_y]], dtype=np.float64)
            world_xy, valid = unproject_to_floor(
                pts_px, pose.R, pose.t, fx, fy, cx, cy, xi, alpha,
                floor_z=-height_m,
            )
            if not valid[0]:
                continue

            per_camera[cam_id].append((det.get("gt_name"), world_xy[0]))

    return per_camera


def main():
    print("floorplan.json 読み込み中...")
    cameras_json = load_floorplan_cameras(FLOORPLAN_JSON_PATH)

    print("カメラパラメータの上書き設定を適用中...")
    cameras_json = apply_camera_overrides(cameras_json)

    print("カメラ姿勢を解決中...")
    poses = resolve_poses(cameras_json, root="Area1", fmt=ROT_FMT)

    print(f"元画像を読み込み中: {SOURCE_IMAGE_PATH}")
    source_image = cv2.imread(SOURCE_IMAGE_PATH)
    if source_image is None:
        raise FileNotFoundError(f"画像を読み込めませんでした: {SOURCE_IMAGE_PATH}")
    img_w = source_image.shape[1]
    tile_w = img_w / NUM_CAMERAS

    print("YOLOモデル読み込み中...")
    model = YOLO(MODEL_PATH)

    print("推論を実行中...")
    results = model.predict(
        source=SOURCE_IMAGE_PATH,
        classes=YOLO_PREDICT_CLASSES,
        conf=YOLO_PREDICT_CONF,
        iou=YOLO_PREDICT_IOU,
        imgsz=YOLO_PREDICT_IMGSZ,
    )
    raw_boxes = yolo_results_to_raw_boxes(results[0])
    bucketed_yolo = bucket_detections_by_camera(raw_boxes, tile_w)

    print("GTアノテーションを読み込み中(評価用のname対応付けのみに使用)...")
    _gt_width, _gt_height, gt_objects = parse_voc_annotation(Path(ANNOTATION_XML_PATH))
    bucketed_gt = bucket_gt_by_camera(gt_objects, tile_w)

    bucketed_with_names = assign_gt_names(bucketed_yolo, bucketed_gt)

    print("\n--- カメラ別の検出数 / GT対応付け状況 ---")
    for cam_id in CAM_IDS:
        dets = bucketed_with_names[cam_id]
        n_matched = sum(1 for d in dets if d["gt_name"] is not None)
        n_unmatched = len(dets) - n_matched
        print(f"  {cam_id}: YOLO検出 {len(dets)}件 (GT対応あり {n_matched}件 / 対応なし(未知) {n_unmatched}件) "
              f"/ GT総数 {len(bucketed_gt[cam_id])}件")

    per_camera = build_world_points_from_yolo(bucketed_with_names, poses, tile_w)
    assignments, gt_same_pairs = compute_camera_assignments_by_distance(per_camera)

    total_same_gt = len(gt_same_pairs)
    print(f"\nYOLO検出のうち、複数カメラに跨って対応付けられた個体ペア(正解=同一個体): {total_same_gt}件")
    print("※ GTの全個体数(102件, 000000.xml)より少なくなるのが普通。YOLOの見逃し・誤検出の影響を含むため。")
    print(f"Hungarian法による割当ペア総数(閾値フィルタ前、参考値): {len(assignments)}件")

    all_keys, name_lookup = build_keys_and_name_lookup(per_camera)

    print("\n=== 距離閾値ごとの精度 (YOLO検出 + 姿勢別高さ + Hungarian + Union-Find) ===")
    print(f"{'距離閾値(m)':>10} | {'統合成功率(recall)':>20} | {'誤統合precision':>16} | {'F1':>8}")
    for thr in DIST_THRESHOLDS_M:
        accepted_pairs = [
            ((a.cam_i, a.idx_i), (a.cam_j, a.idx_j))
            for a in assignments if a.dist <= thr
        ]
        r = evaluate_merge_accuracy(all_keys, gt_same_pairs, accepted_pairs, name_lookup)

        print(f"{thr:>10.2f} | {r['recall']:>14.3f} ({r['tp']:>3}/{r['tp']+r['fn']:>3}) | "
              f"{r['precision']:>13.3f} (FP={r['fp']:>3}) | {r['f1']:>8.3f}")


if __name__ == "__main__":
    main()