"""
render_cylinder_fit_overlay_ver1.py

fit_cow_cylinder_model_ver1.py は単一の検出(TARGET_DETECTIONSで手動指定した
1件)についてのみ、フィッティング結果をbbox付近の切り出し画像に重ねて保存
していた。今回は、鳥瞰図(BEV)ではなく「実際のカメラ画像そのもの」に、
1フレーム分・全カメラ・全個体のフィッティング結果を重ねて見たいという要望
に応えるため、以下の2種類の画像を生成する。

  1. 個体ごとの切り出し画像(fit_cow_cylinder_model_ver1.save_overlay_image
     をそのまま再利用): 観測bbox(白)+フィットした3Dモデルの輪(緑)を、
     bbox周辺を拡大して保存。
  2. カメラ全体画像(新規): 1フレーム・1カメラ全体(フル画角、見やすいよう
     縮小)に、そのカメラに写っている全個体の観測bbox+フィット済み3Dモデルの
     輪をまとめて描画。牛が密集している場面での見え方を一度に確認できる。

ロジック(カメラ割当・円柱モデル・フィッティング・座標系の符号規則)は
render_bev_cylinder_fit_ver1.py / fit_cow_cylinder_model_ver1.py と同じものを
再利用する。

出力:
  - 標準出力: 各個体のフィッティング結果(ベースラインとの差分)
  - 画像: OUTPUT_DIR/cylinder_fit_overlay/
      個体ごと: {frame}_{cam_id}_{name}_cylinder_overlay.jpg (bbox周辺の拡大)
      カメラ全体: {frame}_{cam_id}_all_cylinder_overlay.jpg (フル画角、縮小)

--- 使い方 ---
    1. 設定セクションのパスを実際の環境に合わせて書き換える
    2. FRAME_IDXを確認したいフレーム番号に変更する
    3. python render_cylinder_fit_overlay_ver1.py
"""

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

MOOTRACK_CODE_DIR = os.path.join(THIS_DIR, "..", "calibration")
if MOOTRACK_CODE_DIR not in sys.path:
    sys.path.insert(0, MOOTRACK_CODE_DIR)

from Mootrack_core import ROT_FMT, get_intrinsics, load_floorplan, apply_camera_overrides, resolve_poses, unproject_to_floor

import fit_cow_cylinder_model_ver1 as cyl  # noqa: E402  (円柱モデル・フィッティング・座標系規則を再利用)

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------

XML_DIR = "/mnt/vmlqnap02/dataset/MooTrack360/1-hour-farm1-area1-sample-sequence/xml-1fps"
IMAGE_DIR = "/mnt/vmlqnap02/dataset/MooTrack360/1-hour-farm1-area1-sample-sequence/1-hour-farm1-area1-frames"
FLOORPLAN_JSON_PATH = "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/calibration/Farm2_Area_1_Artboard_config_.json"
OUTPUT_DIR = Path(
    "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/research/idf1_check"
)
OVERLAY_OUT_DIR = OUTPUT_DIR / "cylinder_fit_overlay"
OVERLAY_OUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_WIDTH_PX = 8832
NUM_CAMERAS = 3
CAM_IDS = ["Camera_1", "Camera_2", "Camera_3"]

FRAME_IDX = 100  # 確認したいフレーム番号

FULLCAM_SCALE = 0.5  # カメラ全体画像の縮小率


def frame_name(idx: int) -> str:
    return f"{idx:06d}"


def parse_objects(xml_path: Path) -> List[dict]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        pose_el = obj.find("pose")
        pose = pose_el.text.strip().lower() if pose_el is not None and pose_el.text else "standing"
        bnd = obj.find("bndbox")
        objects.append({
            "name": name, "pose": pose,
            "xmin": float(bnd.find("xmin").text), "ymin": float(bnd.find("ymin").text),
            "xmax": float(bnd.find("xmax").text), "ymax": float(bnd.find("ymax").text),
        })
    return objects


def assign_camera(obj: dict, tile_w: float) -> Tuple[int, str]:
    center_x_full = (obj["xmin"] + obj["xmax"]) / 2.0
    cam_idx = int(center_x_full // tile_w)
    cam_idx = max(0, min(cam_idx, NUM_CAMERAS - 1))
    return cam_idx, CAM_IDS[cam_idx]


def draw_cylinder_wireframe(tile_img: np.ndarray, X_fit: float, Y_fit: float, pose: str,
                             R: np.ndarray, t: np.ndarray, fx: float, fy: float, cx: float, cy: float,
                             xi: float, alpha: float, color_bgr: Tuple[int, int, int]) -> None:
    """フィットした3D円柱/楕円柱モデルの各高さの輪を、与えられた画像に描画する。"""
    _, (rx, ry), full_h = cyl.cylinder_model_params(pose)
    heights = np.linspace(0.0, full_h, cyl.N_HEIGHT_RINGS)
    angles = np.linspace(0.0, 2.0 * np.pi, cyl.N_ANGLES, endpoint=False)

    for h in heights:
        ring_pts = []
        for a in angles:
            px = X_fit + rx * np.cos(a)
            py = Y_fit + ry * np.sin(a)
            p_world = np.array([px, py, cyl.Z_SIGN * h])
            p_cam = cyl.world_to_cam(p_world, R, t)
            u, v, valid = cyl.project_point_fisheye(p_cam, fx, fy, cx, cy, xi, alpha)
            if valid:
                ring_pts.append((int(round(u)), int(round(v))))
        if len(ring_pts) >= 3:
            pts_arr = np.array(ring_pts, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(tile_img, [pts_arr], isClosed=True, color=color_bgr, thickness=2, lineType=cv2.LINE_AA)


def main():
    print("floorplan.json 読み込み中...")
    regions, cameras_json, real_w, real_h = load_floorplan(FLOORPLAN_JSON_PATH)
    cameras_json = apply_camera_overrides(cameras_json)
    poses = resolve_poses(cameras_json, root="Area1", fmt=ROT_FMT)
    tile_w = IMAGE_WIDTH_PX / NUM_CAMERAS

    xml_path = Path(XML_DIR) / f"{frame_name(FRAME_IDX)}.xml"
    if not xml_path.exists():
        print(f"XMLが見つかりません: {xml_path}")
        return

    objects = parse_objects(xml_path)
    print(f"frame={frame_name(FRAME_IDX)}  検出数(重複含む)={len(objects)}")

    image_path = Path(IMAGE_DIR) / f"{frame_name(FRAME_IDX)}.jpg"
    if not image_path.exists():
        print(f"画像が見つかりません: {image_path}")
        return
    full_img = cv2.imread(str(image_path))
    if full_img is None:
        print(f"画像の読み込みに失敗: {image_path}")
        return

    # カメラごとに、そのカメラに写っている個体の(bbox, pose, 名前, fit結果)を集める
    per_cam_results: Dict[str, List[dict]] = {cam_id: [] for cam_id in CAM_IDS}

    for obj in objects:
        cam_idx, cam_id = assign_camera(obj, tile_w)
        x0 = cam_idx * tile_w
        local_bbox = (obj["xmin"] - x0, obj["ymin"], obj["xmax"] - x0, obj["ymax"])
        pose = obj["pose"] if obj["pose"] in ("standing", "lying") else "standing"
        name = obj["name"]

        fx, fy, cx_, cy_, xi, alpha = get_intrinsics(cam_id)
        cam_pose = poses[cam_id]

        local_x = (local_bbox[0] + local_bbox[2]) / 2.0
        local_y = (local_bbox[1] + local_bbox[3]) / 2.0
        center_height, _, _ = cyl.cylinder_model_params(pose)
        pts_px = np.array([[local_x, local_y]], dtype=np.float64)
        baseline_xy, ok = unproject_to_floor(
            pts_px, cam_pose.R, cam_pose.t, fx, fy, cx_, cy_, xi, alpha, floor_z=-center_height,
        )
        if not ok[0]:
            print(f"  [スキップ] {name} ({cam_id}): ベースライン逆投影失敗")
            continue
        X0, Y0 = baseline_xy[0]

        X_fit, Y_fit, _ = cyl.fit_xy_to_bbox(
            local_bbox, pose, cam_pose.R, cam_pose.t, fx, fy, cx_, cy_, xi, alpha, (X0, Y0),
        )
        diff_m = float(np.linalg.norm([X_fit - X0, Y_fit - Y0]))
        print(f"  {name:10s} pose={pose:8s} cam={cam_id}  "
              f"baseline=({X0:.2f},{Y0:.2f})  fit=({X_fit:.2f},{Y_fit:.2f})  diff={diff_m:.3f}m")

        per_cam_results[cam_id].append({
            "name": name, "pose": pose, "local_bbox": local_bbox,
            "X_fit": X_fit, "Y_fit": Y_fit,
            "R": cam_pose.R, "t": cam_pose.t,
            "fx": fx, "fy": fy, "cx": cx_, "cy": cy_, "xi": xi, "alpha": alpha,
        })

        # 1) 個体ごとの切り出し画像(既存関数を再利用)
        overlay_path = cyl.save_overlay_image(
            FRAME_IDX, cam_id, local_bbox, pose, X_fit, Y_fit,
            cam_pose.R, cam_pose.t, fx, fy, cx_, cy_, xi, alpha,
        )
        if overlay_path:
            print(f"    個体別オーバーレイ保存: {overlay_path}")

    # 2) カメラ全体画像(そのカメラに写っている全個体をまとめて描画)
    for cam_id, results in per_cam_results.items():
        if not results:
            continue
        cam_idx = CAM_IDS.index(cam_id)
        x0 = int(cam_idx * tile_w)
        x1 = int((cam_idx + 1) * tile_w)
        tile_img = full_img[:, x0:x1].copy()

        for r in results:
            bx1, by1, bx2, by2 = [int(round(v)) for v in r["local_bbox"]]
            box_color = (255, 255, 255)
            cv2.rectangle(tile_img, (bx1, by1), (bx2, by2), box_color, 3, cv2.LINE_AA)
            model_color = (0, 255, 0) if r["pose"] == "standing" else (0, 140, 255)
            draw_cylinder_wireframe(
                tile_img, r["X_fit"], r["Y_fit"], r["pose"], r["R"], r["t"],
                r["fx"], r["fy"], r["cx"], r["cy"], r["xi"], r["alpha"], model_color,
            )
            cv2.putText(tile_img, r["name"], (bx1, max(0, by1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, model_color, 2, cv2.LINE_AA)

        cv2.putText(tile_img, f"{frame_name(FRAME_IDX)} {cam_id}  "
                               f"(white=observed bbox, green=standing model, orange=lying model)",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)

        if FULLCAM_SCALE != 1.0:
            h, w = tile_img.shape[:2]
            tile_img = cv2.resize(tile_img, (int(w * FULLCAM_SCALE), int(h * FULLCAM_SCALE)))

        out_path = OVERLAY_OUT_DIR / f"{frame_name(FRAME_IDX)}_{cam_id}_all_cylinder_overlay.jpg"
        cv2.imwrite(str(out_path), tile_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"\nカメラ全体オーバーレイ保存: {out_path}")


if __name__ == "__main__":
    main()