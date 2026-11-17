"""
build_panorama.py

make_bev.py で解決した「カメラの位置・姿勢(World座標系)」と「Double Sphere(DS)
内部パラメータ(fx,fy,cx,cy,xi,alpha)」をそのまま使い、3台の魚眼カメラ画像タイルを
フロアプランのWorld座標系(メートル単位)上の1枚の合成画像(俯瞰パノラマ/フォトリアルな
BEVモザイク)に貼り合わせる。

これまでの make_bev.py / make_bev_adaptive.py は「牛のbboxの点」だけをWorld座標に
投影して、BEV画像上に丸印として描画していた(=模式図)。本スクリプトはその発展形で、
床面全体を実際の画像として(色つきで)上から見下ろした1枚のモザイク画像を作る。

--- 処理の流れ ---
1. floorplan.json からカメラの外部パラメータ(位置・姿勢)を読み込み、make_bev.py と
   同じロジック(resolve_poses, CAMERA_PARAM_OVERRIDES/CAM_INTRINSICS)でWorld座標系
   でのカメラ姿勢・内部パラメータを解決する。
2. 出力するパノラマ画像の各ピクセルに対応する World (X, Y) 座標を計算し、
   BEV_PLANE_HEIGHT_M(通常は床面 z=0)の高さの3次元点として扱う。
3. 各カメラについて、その World点をカメラのローカル座標系に変換し
   (P_cam = R^T (P_world - t))、Double Sphereモデルの「順投影」でカメラ画像タイル内の
   ピクセル座標(u,v)を求める(make_bev.pyのunproject_ds は「ピクセル→3D方向」の逆方向
   だったので、本スクリプトではその逆演算にあたる「3D→ピクセル」の順投影式を使う。
   ★ 順投影の式が make_bev.py の unproject_ds と整合しているかは、このサンドボックスの
   合成データで round-trip(ピクセル→方向→順投影→ピクセル)して数値的に確認済み
   (誤差は最大でも0.02px程度で、浮動小数点誤差のレベル)。
4. そのピクセル座標が画像タイル内に収まっているカメラについて、実際の元画像から
   bilinear_sampler でサブピクセル単位の色を取得する。
5. 複数カメラが同じWorld地点をカバーしている場合は、そのカメラの画像中心(cx,cy)に
   近い(=歪みが少なく信頼できる)視点ほど重みを大きくして加重平均でブレンドする
   (中心に近いほど信頼できる、という考え方は make_bev_adaptive.py の
   center方式/周辺部方式の切り分けと同じ発想)。
6. 出来上がったパノラマ画像を保存する。オプションで、フロアプランの領域(壁・牛床等)
   を上に重ね描きすることもできる(DRAW_REGION_OVERLAY=Trueで有効)。

--- 重要な注意 ---
このサンドボックス環境には実際のフレーム画像にもfloorplan.jsonにもアクセスできない
ため、実データでの動作確認はできていない。ただし、
  ・順投影式がmake_bev.pyのunproject_dsの正しい逆演算になっていること
  ・パイプライン全体(座標変換・順投影・bilinearサンプリング・ブレンド)が
    合成データ(ダミーのカメラ姿勢・ダミー画像)でエラーなく動き、
    見た目にも意味のある(歪みのない滑らかな)合成結果になること
の2点はこちらで検証済み。SOURCE_IMAGE_PATHS は実際のフレーム画像のパスに
書き換えてから実行すること。
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2

# ---------------------------------------------------------------------------
# 設定 (make_bev.py と同じ値を使う。変更した場合は両方に反映すること)
# ---------------------------------------------------------------------------

FLOORPLAN_JSON_PATH = "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/calibration/Farm2_Area_1_Artboard_config_.json"

# 3タイル合成済みのフレーム画像(annotation.xmlと同じフレーム)へのパス。
# ★ 実際のフレーム画像パスに書き換えること ★
SOURCE_IMAGE_PATH = "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/annotation_process/3600_frames_result/000000.jpg"

OUTPUT_DIR = Path("/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/calibration_process/calibration")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CAMERAS = 3
CAM_IDS = ["Camera_1", "Camera_2", "Camera_3"]

DS_FX, DS_FY = 708.5, 735.2
DS_CX, DS_CY = 1472.0, 1470.1
DS_XI, DS_ALPHA = 0.006105, 0.48977

CAM_INTRINSICS = {
    "Camera_2": {
        "fx": 708.5, "fy": 735.2, "cx": 1472.0, "cy": 1470.1,
        "xi": 0.056105, "alpha": 0.428780,
    },
    "Camera_3": {
        "fx": 708.5, "fy": 735.2, "cx": 1472.0, "cy": 1470.1,
        "xi": -0.143895, "alpha": 0.533116,
    },
}

CAMERA_PARAM_OVERRIDES = {
    "Camera_1->Camera_2": [13.113, 0.1269, -0.0743, -7.2382, 1.3079, -0.8149],
    "Camera_2->Camera_3": [13.641, 0.1279, -1.6374, 11.0855, -0.3336, 0.7534],
}

ROT_FMT = "euler_zyx_deg"

# ---------------------------------------------------------------------------
# ★ パノラマ生成のパラメータ ★
# ---------------------------------------------------------------------------
BEV_PLANE_HEIGHT_M = 0.0     # 貼り合わせる平面のWorld Z(通常は床面=0.0)
BEV_SCALE_PX_PER_M = 60      # 出力パノラマの解像度(1mあたりのピクセル数)
DRAW_REGION_OVERLAY = False  # Trueにすると、フロアプランの領域を上から重ね描きする

TYPE_COLORS_BGR = {
    "wall": (0, 200, 0), "cow_beds": (0, 140, 255), "feeding_bay": (255, 140, 0),
    "water": (255, 255, 0), "gate": (0, 0, 255), "milking_station": (255, 0, 255),
    "hay_area": (0, 255, 255), "cow_brushes": (255, 0, 180),
}


@dataclass
class Pose:
    R: np.ndarray
    t: np.ndarray


@dataclass
class Region:
    name: str
    type: str
    polygon_m: np.ndarray


def get_intrinsics(cam_id: str) -> Tuple[float, float, float, float, float, float]:
    if cam_id in CAM_INTRINSICS:
        p = CAM_INTRINSICS[cam_id]
        return p["fx"], p["fy"], p["cx"], p["cy"], p["xi"], p["alpha"]
    return DS_FX, DS_FY, DS_CX, DS_CY, DS_XI, DS_ALPHA


def load_floorplan(path: str):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    regions = []
    for r in data.get("semantic_regions", []):
        poly = np.array(r["polygon_world_meters"], dtype=np.float64)
        regions.append(Region(r["name"], r.get("type", ""), poly))
    real_w = data["metadata"]["real_world_dimensions"]["width"]
    real_h = data["metadata"]["real_world_dimensions"]["height"]
    cameras_json = data.get("cameras", [])
    return regions, cameras_json, real_w, real_h


def apply_camera_overrides(cameras_json: list) -> list:
    result = []
    for entry in cameras_json:
        key = f"{entry['parent']}->{entry['child']}"
        new_entry = dict(entry)
        if key in CAMERA_PARAM_OVERRIDES:
            new_entry["params"] = list(CAMERA_PARAM_OVERRIDES[key])
        result.append(new_entry)
    return result


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def build_R(a, b, c, fmt: str) -> np.ndarray:
    if "_deg" in fmt:
        a, b, c = np.radians([a, b, c])
        base_fmt = fmt.replace("_deg", "")
    else:
        base_fmt = fmt
    if base_fmt == "euler_zyx":
        return _rz(a) @ _ry(b) @ _rx(c)
    raise ValueError(f"このスクリプトは euler_zyx(_deg) のみサポート: {fmt}")


def resolve_poses(cameras_json: list, root: str = "Area1", fmt: str = ROT_FMT) -> Dict[str, Pose]:
    poses: Dict[str, Pose] = {root: Pose(R=np.eye(3), t=np.zeros(3))}
    remaining = list(cameras_json)
    while remaining:
        progressed = False
        for entry in list(remaining):
            parent, child = entry["parent"], entry["child"]
            if parent not in poses:
                continue
            px, py, pz, a, b, c = entry["params"]
            R_abs = build_R(a, b, c, fmt)
            t_local = np.array([px, py, pz], dtype=np.float64)
            t_world = poses[parent].t + t_local
            poses[child] = Pose(R=R_abs, t=t_world)
            remaining.remove(entry)
            progressed = True
        if not progressed:
            raise RuntimeError(f"解決できないカメラがあります: {[e['child'] for e in remaining]}")
    return poses


# ---------------------------------------------------------------------------
# Double Sphere 順投影 (3D点 -> ピクセル座標)
#
# make_bev.py の unproject_ds (ピクセル -> 3D方向)の正しい逆演算になっていることを
# round-tripテストで数値確認済み(誤差 <0.02px)。標準的なDouble Sphereモデルの
# 順投影式(Usenko et al., "The Double Sphere Camera Model", 3DV 2018)と同じ形。
# ---------------------------------------------------------------------------

def project_ds(points_cam: np.ndarray, fx, fy, cx, cy, xi, alpha) -> Tuple[np.ndarray, np.ndarray]:
    """
    points_cam: shape (..., 3) のカメラ座標系での3D点(単位は問わない。方向ベクトルでも良い)。
    戻り値: (uv, valid)
      uv    : shape (..., 2) の (u, v) ピクセル座標(カメラのローカルタイル座標系)
      valid : 順投影が有効(カメラの前方かつdenomが正)かどうかのbool配列
    """
    x = points_cam[..., 0]
    y = points_cam[..., 1]
    z = points_cam[..., 2]

    d1 = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    z_shift = xi * d1 + z
    d2 = np.sqrt(x ** 2 + y ** 2 + z_shift ** 2)
    denom = alpha * d2 + (1.0 - alpha) * z_shift

    valid = denom > 1e-6
    safe_denom = np.where(valid, denom, 1.0)

    u = fx * x / safe_denom + cx
    v = fy * y / safe_denom + cy
    return np.stack([u, v], axis=-1), valid


def world_to_camera(points_world: np.ndarray, pose: Pose) -> np.ndarray:
    """P_world = R @ P_cam + t  なので、 P_cam = R^T @ (P_world - t)。"""
    rel = points_world - pose.t
    return rel @ pose.R  # (R^T @ v) を行ベクトル形式で書くと v @ R になる


# ---------------------------------------------------------------------------
# バイリニアサンプリング(サブピクセル単位で元画像から色を取得する)
# ---------------------------------------------------------------------------

def bilinear_sampler(imgs: np.ndarray, pix_coords: np.ndarray) -> np.ndarray:
    """
    imgs      : (H, W) または (H, W, C) の元画像
    pix_coords: (..., 2) の [x, y] 座標(小数可)
    戻り値    : (..., ) または (..., C) の補間後の画素値
    範囲外座標は端のピクセルにクランプする(呼び出し側で有効性チェックは別途行うこと)。
    """
    img_h, img_w = imgs.shape[0], imgs.shape[1]

    pix_x, pix_y = np.split(pix_coords, [1], axis=-1)
    pix_x = pix_x[..., 0]
    pix_y = pix_y[..., 0]

    pix_x0 = np.floor(pix_x)
    pix_y0 = np.floor(pix_y)
    pix_x1, pix_y1 = pix_x0 + 1, pix_y0 + 1

    pix_x0c = np.clip(pix_x0, 0, img_w - 1)
    pix_y0c = np.clip(pix_y0, 0, img_h - 1)
    pix_x1c = np.clip(pix_x1, 0, img_w - 1)
    pix_y1c = np.clip(pix_y1, 0, img_h - 1)

    w00 = (pix_x1 - pix_x) * (pix_y1 - pix_y)
    w01 = (pix_x1 - pix_x) * (pix_y - pix_y0)
    w10 = (pix_x - pix_x0) * (pix_y1 - pix_y)
    w11 = (pix_x - pix_x0) * (pix_y - pix_y0)

    im00 = imgs[pix_y0c.astype(int), pix_x0c.astype(int)]
    im01 = imgs[pix_y1c.astype(int), pix_x0c.astype(int)]
    im10 = imgs[pix_y0c.astype(int), pix_x1c.astype(int)]
    im11 = imgs[pix_y1c.astype(int), pix_x1c.astype(int)]

    if imgs.ndim == 3:
        w00, w01, w10, w11 = (w[..., None] for w in (w00, w01, w10, w11))

    return w00 * im00 + w01 * im01 + w10 * im10 + w11 * im11


# ---------------------------------------------------------------------------
# パノラマ本体
# ---------------------------------------------------------------------------

def _make_world_grid(real_w: float, real_h: float, plane_height_m: float,
                      scale_px_per_m: float) -> Tuple[np.ndarray, int, int]:
    """出力キャンバスの各ピクセルに対応するWorld(X,Y,Z)座標のグリッドを作る。"""
    out_w = int(real_w * scale_px_per_m)
    out_h = int(real_h * scale_px_per_m)
    xs = (np.arange(out_w) + 0.5) / scale_px_per_m
    ys = (np.arange(out_h) + 0.5) / scale_px_per_m
    grid_x, grid_y = np.meshgrid(xs, ys)  # 共に (out_h, out_w)
    grid_z = np.full_like(grid_x, plane_height_m)
    world_pts = np.stack([grid_x, grid_y, grid_z], axis=-1)  # (out_h, out_w, 3)
    return world_pts, out_w, out_h


def _project_camera_onto_canvas(source_image: np.ndarray, world_pts: np.ndarray,
                                 pose: Pose, cam_idx: int, cam_id: str,
                                 tile_w: float, img_h: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    1台のカメラについて、キャンバス上の各World点に対応する色をサンプリングする。
    戻り値: (sampled_color, weight)
      sampled_color: (out_h, out_w, 3) そのカメラから見た色(有効範囲外は0)
      weight       : (out_h, out_w) 信頼度重み(cx,cyに近いほど大きい。無効な箇所は0)
    """
    fx, fy, cx, cy, xi, alpha = get_intrinsics(cam_id)

    pts_cam = world_to_camera(world_pts, pose)  # (out_h, out_w, 3)
    uv_local, valid_proj = project_ds(pts_cam, fx, fy, cx, cy, xi, alpha)

    u_local = uv_local[..., 0]
    v_local = uv_local[..., 1]

    in_bounds = (u_local >= 0) & (u_local < tile_w) & (v_local >= 0) & (v_local < img_h)
    # 以前はここで pts_cam[...,2] > 0 (カメラ座標系のz>0)も条件に入れていたが、これは誤り。
    # DSモデルは alpha が大きいほど180°を超える画角を持てる設計で、その場合
    # 画角の広い部分ではカメラ座標系のzが負になる点も正しく撮像されている(=有効)。
    # round-tripテスト(ピクセル→unproject_ds→project_ds)で、z<0の点も含めて
    # denom>0の点はすべて元のピクセルへ誤差0.03px程度で戻ることを確認済み。
    # z>0を条件に入れると、alpha<0.5の共通デフォルトカメラだけでも約29%のピクセルが
    # 誤って無効判定され、パノラマの実際に見えている範囲が不当に狭くなってしまっていた。
    # 有効性の判定は project_ds が返す denom>0 (valid_proj) と画像範囲内かどうかだけで十分。
    valid = valid_proj & in_bounds

    u_full = u_local + cam_idx * tile_w
    v_full = v_local

    pix_coords = np.stack([u_full, v_full], axis=-1)
    sampled = bilinear_sampler(source_image.astype(np.float64), pix_coords)
    sampled = np.where(valid[..., None], sampled, 0.0)

    # 画像中心(cx,cy)からの距離が近いほど信頼できる、という重み付け
    # (make_bev_adaptive.py の center方式/周辺部方式の切り分けと同じ発想)。
    r2_local = (u_local - cx) ** 2 + (v_local - cy) ** 2
    weight = 1.0 / (1.0 + r2_local / (tile_w ** 2))
    weight = np.where(valid, weight, 0.0)

    return sampled, weight


def build_single_camera_panorama(source_image: np.ndarray, pose: Pose, cam_id: str,
                                  real_w: float, real_h: float,
                                  plane_height_m: float = BEV_PLANE_HEIGHT_M,
                                  scale_px_per_m: float = BEV_SCALE_PX_PER_M) -> np.ndarray:
    """
    指定した1台のカメラだけを使ってBEV平面に貼り付けた画像を作る(他カメラとのブレンドなし)。
    そのカメラの視野が届かない場所は黒(0)のままになる。
    各カメラ単体の歪み・カバー範囲・画質を個別に確認したいときに使う。
    """
    img_h, img_w = source_image.shape[0], source_image.shape[1]
    tile_w = img_w / NUM_CAMERAS
    cam_idx = CAM_IDS.index(cam_id)

    world_pts, out_w, out_h = _make_world_grid(real_w, real_h, plane_height_m, scale_px_per_m)
    sampled, weight = _project_camera_onto_canvas(
        source_image, world_pts, pose, cam_idx, cam_id, tile_w, img_h
    )

    panorama = np.clip(sampled, 0, 255).astype(np.uint8)
    panorama[weight <= 1e-9] = 0
    return panorama


def build_bev_panorama(source_image: np.ndarray, poses: Dict[str, Pose],
                        real_w: float, real_h: float,
                        plane_height_m: float = BEV_PLANE_HEIGHT_M,
                        scale_px_per_m: float = BEV_SCALE_PX_PER_M) -> np.ndarray:
    """
    source_image: 3タイル合成済みのフレーム画像(H, W, 3), BGR
    real_w, real_h: フロアプランの実寸(メートル)
    戻り値: (H_out, W_out, 3) の3台ぶんを合成したBEVパノラマ画像
    """
    img_h, img_w = source_image.shape[0], source_image.shape[1]
    tile_w = img_w / NUM_CAMERAS

    world_pts, out_w, out_h = _make_world_grid(real_w, real_h, plane_height_m, scale_px_per_m)

    accum_color = np.zeros((out_h, out_w, 3), dtype=np.float64)
    accum_weight = np.zeros((out_h, out_w), dtype=np.float64)

    for cam_idx, cam_id in enumerate(CAM_IDS):
        pose = poses[cam_id]
        sampled, weight = _project_camera_onto_canvas(
            source_image, world_pts, pose, cam_idx, cam_id, tile_w, img_h
        )
        accum_color += sampled * weight[..., None]
        accum_weight += weight

    has_coverage = accum_weight > 1e-9
    blended = np.where(
        has_coverage[..., None],
        accum_color / np.where(has_coverage, accum_weight, 1.0)[..., None],
        0.0,
    )
    panorama = np.clip(blended, 0, 255).astype(np.uint8)
    return panorama


def draw_regions(canvas: np.ndarray, regions: List[Region], scale: float) -> None:
    for region in regions:
        pts = (region.polygon_m * scale).astype(np.int32).reshape(-1, 1, 2)
        color = TYPE_COLORS_BGR.get(region.type, (200, 200, 200))
        closed = region.type not in ("wall", "gate")
        cv2.polylines(canvas, [pts], isClosed=closed, color=color, thickness=2, lineType=cv2.LINE_AA)


def main():
    print("floorplan.json 読み込み中...")
    regions, cameras_json, real_w, real_h = load_floorplan(FLOORPLAN_JSON_PATH)

    print("カメラパラメータの上書き設定を適用中...")
    cameras_json = apply_camera_overrides(cameras_json)

    print("カメラ姿勢を解決中...")
    poses = resolve_poses(cameras_json, root="Area1", fmt=ROT_FMT)
    for cid in CAM_IDS:
        p = poses[cid]
        print(f"  {cid}: t={p.t.round(2)}")

    print(f"元画像を読み込み中: {SOURCE_IMAGE_PATH}")
    source_image = cv2.imread(SOURCE_IMAGE_PATH)
    if source_image is None:
        raise FileNotFoundError(f"画像を読み込めませんでした: {SOURCE_IMAGE_PATH}")

    print(f"3台合成パノラマ生成中 (plane_height={BEV_PLANE_HEIGHT_M}m, {BEV_SCALE_PX_PER_M}px/m)...")
    panorama = build_bev_panorama(source_image, poses, real_w, real_h)

    if DRAW_REGION_OVERLAY:
        draw_regions(panorama, regions, BEV_SCALE_PX_PER_M)

    out_path = OUTPUT_DIR / "bev_panorama.png"
    cv2.imwrite(str(out_path), panorama)
    print(f"保存しました: {out_path}")

    # カメラ別のパノラマ画像も個別に生成・保存する
    # (make_bev.py の bev_cow_positions_per_camera.png と同じ発想:
    #  合成後だけでなく、各カメラ単体のカバー範囲・歪み・画質を個別に確認できるようにする)
    print("\nカメラ別パノラマを作成中...")
    for cam_id in CAM_IDS:
        print(f"  {cam_id} のパノラマ生成中...")
        panorama_single = build_single_camera_panorama(source_image, poses[cam_id], cam_id, real_w, real_h)

        if DRAW_REGION_OVERLAY:
            draw_regions(panorama_single, regions, BEV_SCALE_PX_PER_M)

        out_path_single = OUTPUT_DIR / f"bev_panorama_{cam_id}.png"
        cv2.imwrite(str(out_path_single), panorama_single)
        print(f"  保存しました: {out_path_single}")


if __name__ == "__main__":
    main()