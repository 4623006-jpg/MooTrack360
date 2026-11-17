"""
verify_camera_calibration.py

【vml36で実行】

3台の魚眼カメラを統合(ステッチ/床面射影)する前段階として、そもそもの
キャリブレーション(内部パラメータfx,fy,cx,cy,xi,alpha、およびカメラ間の
相対姿勢 CAMERA_PARAM_OVERRIDES)が実用に足る精度に達しているかを検証する。

floorplan.json の semantic_regions(区画)の頂点は、実世界での既知の床面座標
(x, y。単位メートル、floor_z=0を前提)を持っている。その頂点が写っている
各カメラの画像を目視し、対応するピクセル座標を読み取って下のLANDMARKSに
書き込む。このスクリプトはそのピクセル座標をDouble Sphereで床面へ逆投影し、
floorplan.json側の既知座標との誤差(ユークリッド距離、単位メートル)を計算する。

誤差が大きい場合、内部パラメータまたはカメラ間相対姿勢のキャリブレーションが
不十分であることを疑う根拠になる(=カメラ統合レベルにまだ達していない可能性)。

--- 使い方 ---
1. まず LANDMARKS を空のまま一度実行する。区画一覧と各頂点の既知世界座標が
   表示されるので、それを見ながら対応するカメラ画像を目視してピクセル座標を
   確認する。
2. LANDMARKS に {"camera": ..., "region": ..., "vertex_idx": ..., "pixel": (x, y)}
   の形で追記していく(1区画につき複数頂点、複数カメラにまたがってOK。
   同じ頂点が2台以上のカメラに写っていれば、カメラ間の相対姿勢のズレも
   別途チェックできる)。
3. ピクセル座標は「そのカメラ単体の画像」上の座標が基本(PIXELS_ARE_STITCHED
   を参照)。
4. python verify_camera_calibration.py
"""

import csv
from pathlib import Path

import numpy as np

from Mootrack_core import (
    NUM_CAMERAS, CAM_IDS, ROT_FMT,
    get_intrinsics,
    load_floorplan, apply_camera_overrides, resolve_poses,
    unproject_to_floor,
)

# ---------------------------------------------------------------------------
# 設定(パスはここを書き換える)
# ---------------------------------------------------------------------------
FLOORPLAN_JSON_PATH = "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/calibration/Farm2_Area_1_Artboard_config_.json"
OUTPUT_DIR = "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/calibration/calib_verify"

# ランドマークのピクセル座標が「ステッチ後の全体画像(8832px幅など)」で
# 読み取ったものならTrueにし、STITCHED_IMAGE_WIDTH_PXも実際の値に合わせる。
# その場合、内部でカメラごとのローカル座標に自動変換してから逆投影する。
# 各カメラ単体の画像(例: 000000_cam1.jpg のような切り出し済み画像)で
# 読み取った場合はFalseのままでよい。
PIXELS_ARE_STITCHED = False
STITCHED_IMAGE_WIDTH_PX = 8832

# ---------------------------------------------------------------------------
# ランドマーク: 各カメラ画像上で目視確認したピクセル座標
#
#   camera:     "Camera_1" / "Camera_2" / "Camera_3"
#   region:     floorplan.json の semantic_regions[].name と一致させる
#   vertex_idx: その区画のpolygon_world_meters内での頂点インデックス(0始まり)
#   pixel:      (x, y) ピクセル座標
#   label:      任意のメモ(結果表示用、省略可)
#
# region/vertex_idxの組み合わせで、floorplan.json側の既知世界座標
# (x_gt, y_gt)を自動的に引いてくる。
# ---------------------------------------------------------------------------
LANDMARKS = [
    {"camera": "Camera_1", "region" : "Wall_9" ,"vertex_idx": 1,
    "pixel": (1100,1090), "label":"wall_9上側"},
    {"camera": "Camera_1", "region" : "Wall_11" ,"vertex_idx": 1,
    "pixel": (887,2065), "label":"wall_11上側"},
    {"camera": "Camera_1", "region" : "Gate_1" ,"vertex_idx": 1,
    "pixel": (801,2044), "label":"Gate_1左側"},
    {"camera": "Camera_1", "region" : "Feeding_Bay_2" ,"vertex_idx": 0,
    "pixel": (777,839), "label":"Feeding_Bay_2の左上側"},
    {"camera": "Camera_1", "region" : "Wall_9" ,"vertex_idx": 0,
    "pixel": (1093,1859), "label":"Wall_9下側"},
    {"camera": "Camera_1", "region" : "Feeding_Bay_3" ,"vertex_idx": 1,
    "pixel": (2760,1231), "label":"Feeding_Bay_3の右上"},
    {"camera": "Camera_1", "region" : "Wall_12" ,"vertex_idx": 1,
    "pixel": (2665,1755), "label":"Wall_12の上"},
    {"camera": "Camera_1", "region" : "Wall_7" ,"vertex_idx": 0,
    "pixel": (2726,1768), "label":"Wall_7の右"},

    {"camera": "Camera_2", "region" : "Feeding_Bay_2" ,"vertex_idx": 0,
    "pixel": (370,948), "label":"Feeding_Bay_2の左上"},
    {"camera": "Camera_2", "region" : "Wall_9" ,"vertex_idx": 1,
    "pixel": (382,1130), "label":"Wall_9の上"},
    {"camera": "Camera_2", "region" : "Gate_1" ,"vertex_idx": 1,
    "pixel": (272,1637), "label":"Gate_1の左"},
    {"camera": "Camera_2", "region" : "Gate_1" ,"vertex_idx": 0,
    "pixel": (289,1646), "label":"Gate_1の右"},
    {"camera": "Camera_2", "region" : "Wall_11" ,"vertex_idx": 0,
    "pixel": (297,1779), "label":"Wall_11の下"},
    {"camera": "Camera_2", "region" : "Wall_10" ,"vertex_idx": 1,
    "pixel": (2148,1232), "label":"Wall_10の上"},
    {"camera": "Camera_2", "region" : "Wall_10" ,"vertex_idx": 0,
    "pixel": (2042,1842), "label":"Wall_10の下"},
    {"camera": "Camera_2", "region" : "Wall_12" ,"vertex_idx": 1,
    "pixel": (2234,2041), "label":"Wall_12の上"},
    {"camera": "Camera_2", "region" : "Wall_12" ,"vertex_idx": 0,
    "pixel": (2163,2213), "label":"Wall_12の下"},
    {"camera": "Camera_2", "region" : "Feeding_Bay_3" ,"vertex_idx": 1,
    "pixel": (2643,1282), "label":"Feeding_Bay_3の右上"},
    {"camera": "Camera_2", "region" : "Wall_3" ,"vertex_idx": 1,
    "pixel": (2467,1383), "label":"Wall_3左"},
    {"camera": "Camera_2", "region" : "Wall_4" ,"vertex_idx": 1,
    "pixel": (2307,2082), "label":"Wall_4左"},
    
    {"camera": "Camera_3", "region" : "Feeding_Bay_2" ,"vertex_idx": 0,
    "pixel": (185,1476), "label":"Feeding_Bay_2の左上"},
    {"camera": "Camera_3", "region" : "Gate_1" ,"vertex_idx": 1,
    "pixel": (269,1931), "label":"Gate_1の左"},
    {"camera": "Camera_3", "region" : "Wall_10" ,"vertex_idx": 1,
    "pixel": (682,1350), "label":"Wall_10の上"},
    {"camera": "Camera_3", "region" : "Wall_10" ,"vertex_idx": 0,
    "pixel": (791,1896), "label":"Wall_10の下"},
    {"camera": "Camera_3", "region" : "Wall_12" ,"vertex_idx": 1,
    "pixel": (1222,2128), "label":"Wall_12の上"},
    {"camera": "Camera_3", "region" : "Wall_12" ,"vertex_idx": 0,
    "pixel": (1302,2303), "label":"Wall_12の下"},
    {"camera": "Camera_3", "region" : "Wall_3" ,"vertex_idx": 1,
    "pixel": (1265,1133), "label":"Wall_3左"},
    {"camera": "Camera_3", "region" : "Wall_4" ,"vertex_idx": 1,
    "pixel": (1515,2217), "label":"Wall_4左"},
    {"camera": "Camera_3", "region" : "Feeding_Bay_3" ,"vertex_idx": 1,
    "pixel": (2109,779), "label":"Feeding_Bay_3の右上"},
    {"camera": "Camera_3", "region" : "Wall_14" ,"vertex_idx": 0,
    "pixel": (1692,731), "label":"Wall_14の上"},
    {"camera": "Camera_3", "region" : "Milking_Station_2" ,"vertex_idx": 0,
    "pixel": (2141,959), "label":"Milking_Station_2の左上"},
    {"camera": "Camera_3", "region" : "Milking_Station_2" ,"vertex_idx": 3,
    "pixel": (2293,1838), "label":"Milking_Station_2の左下"},
    # 例:
    # {"camera": "Camera_1", "region": "waiting_area", "vertex_idx": 0,
    #  "pixel": (1234.0, 567.0), "label": "待機エリア角(左手前)"},
]

def print_available_regions(regions) -> None:
    print("=== floorplan.json 内の区画(semantic_regions)一覧 ===")
    print("(LANDMARKSに書く region / vertex_idx の参考。既知世界座標もここに表示)\n")
    for r in regions:
        print(f"- region=\"{r.name}\" (type={r.type})")
        for i, (x, y) in enumerate(r.polygon_m):
            print(f"    vertex_idx={i}: world=({x:.3f}, {y:.3f})")
    print()


def find_region_vertex(regions, region_name: str, vertex_idx: int):
    for r in regions:
        if r.name == region_name:
            if vertex_idx < 0 or vertex_idx >= len(r.polygon_m):
                raise IndexError(
                    f"region=\"{region_name}\" にはvertex_idx={vertex_idx}が存在しません"
                    f"(頂点数={len(r.polygon_m)})"
                )
            x, y = r.polygon_m[vertex_idx]
            return float(x), float(y)
    raise KeyError(f"region=\"{region_name}\" がfloorplan.jsonに見つかりません")


def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("floorplan.json 読み込み中...")
    regions, cameras_json, real_w, real_h = load_floorplan(FLOORPLAN_JSON_PATH)
    cameras_json = apply_camera_overrides(cameras_json)
    poses = resolve_poses(cameras_json, root="Area1", fmt=ROT_FMT)
    print(f"床面サイズ: {real_w}m x {real_h}m / 区画数: {len(regions)}\n")

    if not LANDMARKS:
        print_available_regions(regions)
        print("LANDMARKSが空です。上の一覧を見ながら各カメラ画像で対応する頂点の")
        print("ピクセル座標を読み取り、スクリプト先頭のLANDMARKSに追記してください。")
        return

    tile_w = STITCHED_IMAGE_WIDTH_PX / NUM_CAMERAS

    csv_path = out_dir / "calib_verify_results.csv"
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "camera", "region", "vertex_idx", "label",
        "pixel_x", "pixel_y",
        "world_gt_x", "world_gt_y",
        "world_est_x", "world_est_y",
        "error_m", "valid",
    ])

    print("=== ランドマークごとの誤差 ===")
    print(f"{'camera':>9} | {'region/vertex':>22} | {'誤差[m]':>8} | 備考")

    errors_all = []
    errors_by_cam = {cid: [] for cid in CAM_IDS}

    for lm in LANDMARKS:
        cam_id = lm["camera"]
        region_name = lm["region"]
        vertex_idx = lm["vertex_idx"]
        px, py = lm["pixel"]
        label = lm.get("label", "")

        world_gt_x, world_gt_y = find_region_vertex(regions, region_name, vertex_idx)

        if PIXELS_ARE_STITCHED:
            cam_idx = CAM_IDS.index(cam_id)
            local_x = px - cam_idx * tile_w
        else:
            local_x = px
        local_y = py

        fx, fy, cx, cy, xi, alpha = get_intrinsics(cam_id)
        cam_pose = poses[cam_id]
        pts_px = np.array([[local_x, local_y]], dtype=np.float64)
        world_xy, valid = unproject_to_floor(
            pts_px, cam_pose.R, cam_pose.t, fx, fy, cx, cy, xi, alpha, floor_z=0.0,
        )
        world_est_x, world_est_y = world_xy[0]
        ok = bool(valid[0])

        if ok:
            err = float(np.hypot(world_est_x - world_gt_x, world_est_y - world_gt_y))
            errors_all.append(err)
            errors_by_cam[cam_id].append(err)
            err_str = f"{err:8.3f}"
        else:
            err = float("nan")
            err_str = "   invalid"

        tag = f"{region_name}#{vertex_idx}"
        print(f"{cam_id:>9} | {tag:>22} | {err_str} | {label}")

        csv_writer.writerow([
            cam_id, region_name, vertex_idx, label,
            round(local_x, 2), round(local_y, 2),
            round(world_gt_x, 4), round(world_gt_y, 4),
            round(float(world_est_x), 4), round(float(world_est_y), 4),
            round(err, 4) if ok else "invalid", ok,
        ])
        csv_file.flush()

    csv_file.close()
    print(f"\nCSV保存: {csv_path}")

    if errors_all:
        arr = np.array(errors_all)
        print("\n=== 全体サマリー ===")
        print(f"件数: {len(arr)} / 平均誤差: {arr.mean():.3f}m / "
              f"中央値: {np.median(arr):.3f}m / 最大: {arr.max():.3f}m / "
              f"RMSE: {np.sqrt((arr ** 2).mean()):.3f}m")

        print("\n=== カメラ別サマリー ===")
        for cam_id in CAM_IDS:
            errs = errors_by_cam[cam_id]
            if not errs:
                continue
            e = np.array(errs)
            print(f"{cam_id}: 件数={len(e)} 平均={e.mean():.3f}m "
                  f"最大={e.max():.3f}m RMSE={np.sqrt((e ** 2).mean()):.3f}m")
    else:
        print("\n有効な結果がありません(全ランドマークが逆投影不可でした)。")


if __name__ == "__main__":
    main()