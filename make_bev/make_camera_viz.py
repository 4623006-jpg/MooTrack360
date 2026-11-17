"""
make_camera_viz.py (Mootrack_core連携版・較正前後比較)

Farm2_Area1の3台の魚眼カメラを使い、フロアプランJSONの semantic_regions を
カメラ画像に投影して外部パラメータの正しい解釈を目視確認するスクリプト。

カメラモデル: Double Sphere (DS) モデル

--- 位置・姿勢チェーンの解釈 ---
位置(x,y,z)は親の回転を掛けずにそのまま加算(t_world = t_parent + t_local)、
回転(a,b,c)は親のRと合成せずカメラ単独の絶対姿勢として扱う
(R_world = build_R(a,b,c,fmt))。これはMootrack_core.pyのresolve_poses()の
実装と完全に一致している(=このスクリプト側で独自実装していた解釈は、
Mootrack_core.py側にすでに反映済みだった)。

--- 今回の変更点(Mootrack_core.pyから直接パラメータを取得するように変更) ---
以前はこのファイル内にDS内部パラメータのデフォルト値・CAM_INTRINSICS・
CAMERA_PARAM_OVERRIDES・build_R・resolve_poses・apply_camera_overrides・
get_intrinsicsを個別に直書きしていたが、すべてMootrack_core.pyからimportする
ように変更した。

これにより、calibrate_camera_from_landmarks.py 等で求めた新しい較正値を
Mootrack_core.py の CAMERA_PARAM_OVERRIDES / CAM_INTRINSICS に反映するだけで、
このスクリプトを再実行すれば「較正後」オーバーレイに自動で反映される
(このファイル側で較正値を手動コピペし直す必要がなくなった)。

「較正前」= floorplan.json生値 + Mootrack_coreの共通デフォルト内部パラメータ(DS_*)
「較正後」= Mootrack_core.pyの現行CAMERA_PARAM_OVERRIDES / CAM_INTRINSICS適用後
という構図は従来通り。

★注意★ 以前このファイルに直書きされていたCAM_INTRINSICSでは
Camera_2のalpha=0.421186だったが、Mootrack_core.py側は現在0.428780になっている
(値が食い違っていた)。今回の変更でMootrack_core.py側の値がそのまま使われる。
0.421186がローカルでの調整済みの値だった場合は、先にMootrack_core.py側の
CAM_INTRINSICSも更新してから実行すること。

また、build_R/resolve_poses はMootrack_core.pyのものをそのまま使うため、
euler_zyx(_deg)以外の回転順序(euler_xyz, rotvec等)のテストは非対応になった
(このプロジェクトではROT_FMT="euler_zyx_deg"で固定運用されているため、実害は
無いはずだが、念のため明記しておく)。

--- 使い方 ---
    python make_camera_viz.py
(パス・SHOW_AFTERは全て下の設定セクションに直書きしてあるので、引数なしで実行する)
"""

from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import cv2

from Mootrack_core import (
    CAM_IDS, ROT_FMT,
    DS_FX, DS_FY, DS_CX, DS_CY, DS_XI, DS_ALPHA,
    get_intrinsics, build_R, resolve_poses, apply_camera_overrides,
    load_floorplan as _load_floorplan_core, load_floorplan_cameras,
    Pose, Region,
)

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
FLOORPLAN_JSON_PATH = (
    "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/calibration/Farm2_Area_1_Artboard_config_.json"
)
COMBINED_IMAGE_PATH = (
    "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/annotation_process/3600_frames_result/000000.jpg"
)
OUTPUT_DIR = Path(
    "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/calibration"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 「較正後」(Mootrack_core.pyの現行CAMERA_PARAM_OVERRIDES/CAM_INTRINSICS適用後)の
# オーバーレイも生成するかどうか。Falseにすると「較正前」のみ生成する。
SHOW_AFTER = True


# ---------------------------------------------------------------------------
# floorplan.json読み込み(regions部分だけを使う。cameras部分はload_floorplan_camerasで別途取得)
# ---------------------------------------------------------------------------

def load_floorplan_regions(path: str) -> List[Region]:
    regions, _cameras_json, _real_w, _real_h = _load_floorplan_core(path)
    return regions


def load_cameras_json(path: str) -> list:
    return load_floorplan_cameras(path)


# Mootrack_core.Pose には座標変換メソッドが無いので、ここでヘルパーとして用意する
def world_to_cam(pose: Pose, pts_world: np.ndarray) -> np.ndarray:
    return (pose.R.T @ (pts_world - pose.t).T).T


# ---------------------------------------------------------------------------
# Double Sphere (DS) カメラモデルによる順投影(world -> pixel)
# Mootrack_core.pyは逆投影(pixel -> floor)しか提供していないため、
# 可視化に必要な順投影はこのスクリプト固有のロジックとして残す。
# ---------------------------------------------------------------------------

def project_ds(points_cam: np.ndarray,
                fx: float, fy: float, cx: float, cy: float,
                xi: float, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    x, y, z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
    d1 = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    if alpha <= 0.5:
        w1 = alpha / (1.0 - alpha)
    else:
        w1 = (1.0 - alpha) / alpha
    w2 = (w1 + xi) / np.sqrt(2.0 * w1 * xi + xi ** 2 + 1.0)

    valid = z > -w2 * d1 + 1e-6

    denom_inner = xi * d1 + z
    d2 = np.sqrt(x ** 2 + y ** 2 + denom_inner ** 2)
    denom = alpha * d2 + (1.0 - alpha) * denom_inner

    safe = valid & (np.abs(denom) > 1e-9)
    u = np.where(safe, fx * x / denom + cx, 0.0)
    v = np.where(safe, fy * y / denom + cy, 0.0)
    return np.stack([u, v], axis=-1), safe


# ---------------------------------------------------------------------------
# フロアプラン領域をカメラ画像に投影してオーバーレイ
# ---------------------------------------------------------------------------

TYPE_COLORS = {
    "wall": (0, 255, 0),
    "cow_beds": (255, 140, 0),
    "feeding_bay": (0, 140, 255),
    "water": (255, 255, 0),
    "gate": (0, 0, 255),
    "milking_station": (255, 0, 255),
    "hay_area": (0, 255, 255),
    "cow_brushes": (180, 0, 255),
}


def densify_polygon(polygon_m: np.ndarray, steps_per_edge: int = 20) -> np.ndarray:
    pts = []
    n = len(polygon_m)
    for i in range(n):
        p0 = polygon_m[i]
        p1 = polygon_m[(i + 1) % n]
        for t in np.linspace(0, 1, steps_per_edge, endpoint=False):
            pts.append(p0 + t * (p1 - p0))
    return np.array(pts, dtype=np.float64)


def overlay_regions_on_image(
        img: np.ndarray, regions: List[Region],
        pose: Pose,
        fx, fy, cx, cy, xi, alpha,
        only_types: Optional[List[str]] = None,
        floor_z: float = 0.0,
        label: str = "") -> np.ndarray:
    out = img.copy()
    h, w = img.shape[:2]
    for region in regions:
        if only_types and region.type not in only_types:
            continue
        dense = densify_polygon(region.polygon_m, steps_per_edge=30)
        world_xyz = np.hstack([dense, np.full((len(dense), 1), floor_z, dtype=np.float64)])
        cam_xyz = world_to_cam(pose, world_xyz)
        img_pts, valid = project_ds(cam_xyz, fx, fy, cx, cy, xi, alpha)

        in_frame = (img_pts[:, 0] >= 0) & (img_pts[:, 0] < w) & \
                   (img_pts[:, 1] >= 0) & (img_pts[:, 1] < h)
        mask = valid & in_frame
        if mask.sum() < 2:
            continue

        pts = img_pts[mask].astype(np.int32).reshape(-1, 1, 2)
        color = TYPE_COLORS.get(region.type, (200, 200, 200))
        is_closed = region.type not in ("wall", "gate")
        cv2.polylines(out, [pts], isClosed=is_closed,
                      color=color, thickness=3, lineType=cv2.LINE_AA)
        center = img_pts[mask].mean(axis=0).astype(int)
        cv2.putText(out, region.name, tuple(center),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, region.name, tuple(center),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    if label:
        cv2.rectangle(out, (0, 0), (min(w, 260), 40), (0, 0, 0), -1)
        cv2.putText(out, label, (10, 27), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2, cv2.LINE_AA)
    return out


def stack_side_by_side(img_before: np.ndarray, img_after: np.ndarray, gap: int = 6) -> np.ndarray:
    h = max(img_before.shape[0], img_after.shape[0])
    gap_col = np.full((h, gap, 3), 255, dtype=np.uint8)
    return np.hstack([img_before, gap_col, img_after])


# ---------------------------------------------------------------------------
# BEV(俯瞰)マップ
# ---------------------------------------------------------------------------

def create_bev_canvas(real_width: float, real_height: float, scale: int):
    w_px = int(real_width * scale)
    h_px = int(real_height * scale)
    return np.zeros((h_px, w_px, 3), dtype=np.uint8), scale


def draw_bev_map(regions, poses, scale=40, real_w=44.0, real_h=15.0):
    canvas, _ = create_bev_canvas(real_w, real_h, scale)
    for region in regions:
        pts = (region.polygon_m * scale).astype(np.int32).reshape(-1, 1, 2)
        color = TYPE_COLORS.get(region.type, (200, 200, 200))
        cv2.polylines(canvas, [pts], isClosed=(region.type not in ("wall", "gate")),
                      color=color, thickness=2)
    h_canvas, w_canvas = canvas.shape[:2]
    for cam_id, pose in poses.items():
        if cam_id == "Area1":
            continue
        px = int(pose.t[0] * scale)
        py = int(pose.t[1] * scale)
        if 0 <= px < w_canvas and 0 <= py < h_canvas:
            cv2.circle(canvas, (px, py), 10, (0, 0, 255), -1)
            cv2.circle(canvas, (px, py), 12, (255, 255, 255), 2)
            cv2.putText(canvas, cam_id, (px + 14, py + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return canvas


# ---------------------------------------------------------------------------
# メイン: 較正前・較正後の両方を生成して比較しやすくする
# ---------------------------------------------------------------------------

def render_all_cameras(cameras_json_raw, regions, tiles, tag: str):
    """tag='before' なら overrides無し(floorplan.json生値+DS共通デフォルト)、
    'after' ならMootrack_core.pyの現行CAMERA_PARAM_OVERRIDES/CAM_INTRINSICS適用後
    で全カメラを描画し、生成したオーバーレイ画像のリストとBEVマップを返す。"""
    if tag == "before":
        cameras_json = cameras_json_raw  # 元のまま(オーバーライド無し)
        cam_intrinsics_fn = lambda cid: (DS_FX, DS_FY, DS_CX, DS_CY, DS_XI, DS_ALPHA)
    else:
        # apply_camera_overrides()は引数を省略するとMootrack_core.py内の
        # 現行CAMERA_PARAM_OVERRIDESをそのまま使う
        cameras_json = apply_camera_overrides(cameras_json_raw)
        cam_intrinsics_fn = get_intrinsics

    poses = resolve_poses(cameras_json, root="Area1", fmt=ROT_FMT)

    results = []
    for cam_id, tile in zip(CAM_IDS, tiles):
        pose = poses[cam_id]
        fx, fy, cx, cy, xi, alpha = cam_intrinsics_fn(cam_id)
        overlay = overlay_regions_on_image(
            tile, regions, pose, fx, fy, cx, cy, xi, alpha,
            label=f"{cam_id} [{tag}]",
        )
        results.append(overlay)

    bev = draw_bev_map(regions, poses)
    return results, bev


def main():
    print("フロアプランJSON読み込み中...")
    regions = load_floorplan_regions(FLOORPLAN_JSON_PATH)
    cameras_json_raw = load_cameras_json(FLOORPLAN_JSON_PATH)

    print("画像読み込み・3分割中...")
    combined = cv2.imread(COMBINED_IMAGE_PATH)
    if combined is None:
        raise FileNotFoundError(COMBINED_IMAGE_PATH)
    h, w = combined.shape[:2]
    tw = w // 3
    tiles = [combined[:, i * tw:(i + 1) * tw].copy() for i in range(3)]

    print("\n[較正前] オーバーレイ生成中...")
    overlays_before, bev_before = render_all_cameras(cameras_json_raw, regions, tiles, tag="before")
    for i, img in enumerate(overlays_before):
        out_path = OUTPUT_DIR / f"overlay_cam{i + 1}_before.png"
        cv2.imwrite(str(out_path), img)
        print(f"  保存: {out_path}")
    cv2.imwrite(str(OUTPUT_DIR / "farm_bev_map_before.png"), bev_before)

    if SHOW_AFTER:
        print("\n[較正後](Mootrack_core.pyの現行値) オーバーレイ生成中...")
        overlays_after, bev_after = render_all_cameras(cameras_json_raw, regions, tiles, tag="after")
        for i, img in enumerate(overlays_after):
            out_path = OUTPUT_DIR / f"overlay_cam{i + 1}_after.png"
            cv2.imwrite(str(out_path), img)
            print(f"  保存: {out_path}")
        cv2.imwrite(str(OUTPUT_DIR / "farm_bev_map_after.png"), bev_after)

        print("\n[比較用] 較正前後を並べた画像を生成中...")
        for i, (b, a) in enumerate(zip(overlays_before, overlays_after)):
            combined_cmp = stack_side_by_side(b, a)
            out_path = OUTPUT_DIR / f"overlay_cam{i + 1}_before_after.png"
            cv2.imwrite(str(out_path), combined_cmp)
            print(f"  保存: {out_path}")

        bev_cmp = stack_side_by_side(bev_before, bev_after)
        cv2.imwrite(str(OUTPUT_DIR / "farm_bev_map_before_after.png"), bev_cmp)
        print(f"  保存: {OUTPUT_DIR / 'farm_bev_map_before_after.png'}")

    print("\n完了しました。")


if __name__ == "__main__":
    main()