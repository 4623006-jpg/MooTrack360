"""
annotate_bev_gt_click_ver1.py

指定した1フレームについて、BEV(牛舎の壁・エリアつき)上をクリックして
「その地点に乳牛がいる」というBEVのGTを作るアノテーション用ツール。

phase3_bev_multicam_gt_zonepose_ver1.py と同じフォルダに置く。

画面:
  [BEV]       壁・エリア、各カメラの曲線境界、カメラ位置。
              参考として、既存GT(bboxの高さ仮定投影)を牛名つきの薄い点で重ねられる(gキー)。
  [CAMERAS]   同じフレームの3カメラ画像にGTのbbox+牛名を描いた画像(別ウィンドウ)。
              どの牛か見分ける用。SHOW_CAMERA_WINDOW=False で出さない。

操作(BEVウィンドウ上):
  左クリック           : その地点に牛を追加(番号は自動で 1,2,3...)
  左ドラッグ(点の近く) : 既存の点を動かす
  右クリック           : いちばん近い点を削除
  u                    : 直前の追加を取り消し
  g                    : 参考のGT点(既存GTの投影位置)の表示/非表示
  n                    : 番号ラベルの表示/非表示
  s                    : 保存
  q / ESC              : 終了(未保存なら警告を出す。もう一度押すと終了)

出力(OUTPUT_DIR / bev_gt_annotation /):
  {frame}_bev_gt_ver1.json   points: [{id, x_m, y_m}]  (BEV座標[m]。世界座標と同じ系)
  {frame}_bev_gt_ver1.png    保存時のBEV画像
同名のjsonがあれば起動時に読み込んで続きから編集できる。
(上書きを避けたいときは、ファイル名のverを上げるのではなく、保存先フォルダをコピーしておく)

使い方:
    python annotate_bev_gt_click_ver1.py 100      # フレーム番号を引数で指定
    python annotate_bev_gt_click_ver1.py          # 引数が無ければ FRAME_IDX を使う
"""

import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = "/mnt/vmlqnap02/dataset/MooTrack360/1-hour-farm1-area1-sample-sequence/1-hour-farm1-area1-frames"
ANALYZE_DIR = os.path.join(THIS_DIR, "..", "analyze")
for d in (os.path.join(THIS_DIR, "..", "calibration"), ANALYZE_DIR, THIS_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

import phase3_bev_multicam_gt_zonepose_ver1 as g
from Mootrack_core import load_floorplan

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
FRAME_IDX = 100
SCALE = 40                   # 1m = 40px (他のBEV描画と同じ)
SHOW_CAMERA_WINDOW = True    # 3カメラ画像+GTのbboxを別ウィンドウに出す
CAMERA_WINDOW_HEIGHT = 520   # そのウィンドウの高さ[px]
SHOW_REF_GT = True          # 起動時に参考のGT点を出すか(gキーで切替)
PICK_RADIUS_M = 0.4          # この距離以内ならドラッグ対象
DELETE_RADIUS_M = 0.6        # 右クリックで削除する最大距離

OUT_DIR = g.OUTPUT_DIR / "bev_gt_annotation"

TYPE_COLORS_BGR = {
    "wall": (0, 200, 0), "cow_beds": (0, 140, 255), "feeding_bay": (255, 140, 0),
    "water": (255, 255, 0), "gate": (0, 0, 255), "milking_station": (255, 0, 255),
    "hay_area": (0, 255, 255), "cow_brushes": (255, 0, 180),
}
CAM_COLORS_BGR = {"Camera_1": (255, 120, 60), "Camera_2": (60, 200, 60), "Camera_3": (60, 140, 255)}
MARK_COLOR = (0, 255, 255)   # アノテーション点(黄)
REF_COLOR = (150, 150, 150)  # 参考GT点(灰)


# ---------------------------------------------------------------------------
# 背景(壁・エリア・曲線境界・カメラ位置)
# ---------------------------------------------------------------------------
def build_base(regions, poses, cam_polygons, real_w, real_h):
    margin = 110
    w_px = int(real_w * SCALE) + margin
    h_px = int(real_h * SCALE) + 40
    w_px += w_px % 2
    h_px += h_px % 2
    canvas = np.full((h_px, w_px, 3), (30, 30, 30), dtype=np.uint8)
    for region in regions:
        pts = (region.polygon_m * SCALE).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], isClosed=region.type not in ("wall", "gate"),
                      color=TYPE_COLORS_BGR.get(region.type, (200, 200, 200)), thickness=2, lineType=cv2.LINE_AA)
    for cid, poly in cam_polygons.items():
        if poly is not None and len(poly) >= 3:
            cv2.polylines(canvas, [(poly * SCALE).astype(np.int32).reshape(-1, 1, 2)], True,
                          CAM_COLORS_BGR.get(cid, (255, 255, 255)), 1, cv2.LINE_AA)
    for cid in g.CAM_IDS:
        px, py = int(poses[cid].t[0] * SCALE), int(poses[cid].t[1] * SCALE)
        cv2.circle(canvas, (px, py), 9, (0, 0, 255), -1)
        cv2.circle(canvas, (px, py), 11, (255, 255, 255), 2)
        cv2.putText(canvas, cid, (px + 14, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    for k, cid in enumerate(g.CAM_IDS):
        y = 20 + k * 22
        cv2.rectangle(canvas, (w_px - margin + 5, y), (w_px - margin + 21, y + 16), CAM_COLORS_BGR[cid], -1)
        cv2.putText(canvas, cid, (w_px - margin + 26, y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


# ---------------------------------------------------------------------------
# 参考GT(既存GTの高さ仮定投影)とカメラ画像
# ---------------------------------------------------------------------------
def reference_points(frame_idx, poses):
    xml_path = Path(g.XML_DIR) / f"{g.frame_name(frame_idx)}.xml"
    if not xml_path.exists():
        return []
    raw = g.gt_detections_per_camera(g.parse_voc_annotation(xml_path), poses)
    out = []
    for cid, items in raw.items():
        for it in items:
            out.append((cid, it["name"], np.asarray(it["world_xy"], dtype=float)))
    return out


def build_camera_image(frame_idx):
    xml_path = Path(g.XML_DIR) / f"{g.frame_name(frame_idx)}.xml"
    image_path = Path(IMAGE_DIR) / f"{g.frame_name(frame_idx)}.jpg"
    if not xml_path.exists() or not image_path.exists():
        print(f"カメラ画像ウィンドウはスキップ (xml={xml_path.exists()}, image={image_path.exists()})")
        return None
    full = cv2.imread(str(image_path))
    if full is None:
        return None
    objs = []
    for obj in ET.parse(xml_path).getroot().findall("object"):
        bnd = obj.find("bndbox")
        objs.append((obj.find("name").text, float(bnd.find("xmin").text), float(bnd.find("ymin").text),
                     float(bnd.find("xmax").text), float(bnd.find("ymax").text)))
    n_cam = len(g.CAM_IDS)
    tile_w = g.IMAGE_WIDTH_PX / n_cam
    tiles = []
    for ci, cid in enumerate(g.CAM_IDS):
        crop = full[:, int(ci * tile_w):int((ci + 1) * tile_w)].copy()
        col = CAM_COLORS_BGR[cid]
        for name, x1, y1, x2, y2 in objs:
            cx = (x1 + x2) / 2.0
            if min(int(cx // tile_w), n_cam - 1) != ci:
                continue
            bx1, by1, bx2, by2 = [int(round(v)) for v in (x1 - ci * tile_w, y1, x2 - ci * tile_w, y2)]
            cv2.rectangle(crop, (bx1, by1), (bx2, by2), col, 3, cv2.LINE_AA)
            cv2.putText(crop, name.replace("cow", ""), (bx1, max(0, by1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, col, 3, cv2.LINE_AA)
        cv2.putText(crop, cid, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 255), 4, cv2.LINE_AA)
        s = CAMERA_WINDOW_HEIGHT / crop.shape[0]
        tiles.append(cv2.resize(crop, (int(crop.shape[1] * s), CAMERA_WINDOW_HEIGHT)))
    return np.hstack(tiles)


# ---------------------------------------------------------------------------
# アノテーション状態
# ---------------------------------------------------------------------------
class Annotator:
    def __init__(self, frame_idx, base, ref_pts, real_w, real_h):
        self.frame_idx = frame_idx
        self.base = base
        self.ref_pts = ref_pts
        self.real_w, self.real_h = real_w, real_h
        self.points = []            # [{"id": int, "xy": np.array([x_m, y_m])}]
        self.show_ref = SHOW_REF_GT
        self.show_ids = True
        self.drag_idx = None
        self.dirty = False
        self.quit_warned = False
        self.json_path = OUT_DIR / f"{g.frame_name(frame_idx)}_bev_gt_ver1.json"
        self.png_path = OUT_DIR / f"{g.frame_name(frame_idx)}_bev_gt_ver1.png"
        self.load()

    def load(self):
        if self.json_path.exists():
            data = json.loads(self.json_path.read_text(encoding="utf-8"))
            self.points = [{"id": int(p["id"]), "xy": np.array([p["x_m"], p["y_m"]], dtype=float)}
                           for p in data.get("points", [])]
            print(f"既存の保存を読み込みました: {self.json_path} ({len(self.points)}点)")

    def next_id(self):
        return max([p["id"] for p in self.points], default=0) + 1

    def px_to_m(self, x, y):
        return np.array([x / SCALE, y / SCALE], dtype=float)

    def inside(self, xy):
        return 0 <= xy[0] <= self.real_w and 0 <= xy[1] <= self.real_h

    def nearest(self, xy):
        if not self.points:
            return None, None
        d = [np.linalg.norm(p["xy"] - xy) for p in self.points]
        i = int(np.argmin(d))
        return i, d[i]

    def on_mouse(self, event, x, y, flags, param):
        xy = self.px_to_m(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.inside(xy):
                return
            i, d = self.nearest(xy)
            if i is not None and d <= PICK_RADIUS_M:
                self.drag_idx = i
            else:
                self.points.append({"id": self.next_id(), "xy": xy})
                self.dirty = True
                self.quit_warned = False
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_idx is not None:
            if self.inside(xy):
                self.points[self.drag_idx]["xy"] = xy
                self.dirty = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.drag_idx = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            i, d = self.nearest(xy)
            if i is not None and d <= DELETE_RADIUS_M:
                removed = self.points.pop(i)
                print(f"削除: #{removed['id']}")
                self.dirty = True
                self.quit_warned = False

    def undo(self):
        if self.points:
            removed = self.points.pop()
            print(f"取り消し: #{removed['id']}")
            self.dirty = True

    def render(self):
        canvas = self.base.copy()
        if self.show_ref:
            for cid, name, xy in self.ref_pts:
                p = (int(xy[0] * SCALE), int(xy[1] * SCALE))
                cv2.circle(canvas, p, 3, REF_COLOR, -1)
                cv2.putText(canvas, name.replace("cow", ""), (p[0] + 4, p[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, REF_COLOR, 1, cv2.LINE_AA)
        for p in self.points:
            c = (int(p["xy"][0] * SCALE), int(p["xy"][1] * SCALE))
            cv2.circle(canvas, c, 7, MARK_COLOR, -1)
            cv2.circle(canvas, c, 9, (255, 255, 255), 1)
            if self.show_ids:
                cv2.putText(canvas, str(p["id"]), (c[0] + 10, c[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, MARK_COLOR, 1, cv2.LINE_AA)
        state = "未保存" if self.dirty else "保存済み"
        header = (f"{g.frame_name(self.frame_idx)}  点={len(self.points)}  [{state}]  "
                  f"L:追加/ドラッグ移動 R:削除 u:取消 g:参考GT n:番号 s:保存 q:終了")
        cv2.putText(canvas, header, (8, canvas.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return canvas

    def save(self):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "frame": self.frame_idx,
            "scale_px_per_m": SCALE,
            "floor_size_m": [self.real_w, self.real_h],
            "points": [{"id": p["id"], "x_m": round(float(p["xy"][0]), 4), "y_m": round(float(p["xy"][1]), 4)}
                       for p in self.points],
        }
        self.json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        cv2.imwrite(str(self.png_path), self.render())
        self.dirty = False
        print(f"保存: {self.json_path} ({len(self.points)}点)\n      {self.png_path}")


def main():
    frame_idx = int(sys.argv[1]) if len(sys.argv) > 1 else FRAME_IDX
    regions, cameras_json, real_w, real_h = load_floorplan(g.FLOORPLAN_JSON_PATH)
    poses = g.resolve_poses(g.apply_camera_overrides(cameras_json), root="Area1", fmt=g.ROT_FMT)
    cam_polygons = g.build_camera_polygons(poses, g.CAM_IDS, g.get_intrinsics,
                                           height_m=g.ZONE_HEIGHT_M, n_angles=g.N_ANGLES)
    base = build_base(regions, poses, cam_polygons, real_w, real_h)
    ann = Annotator(frame_idx, base, reference_points(frame_idx, poses), real_w, real_h)

    win = "BEV GT annotation"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, ann.on_mouse)
    if SHOW_CAMERA_WINDOW:
        cam_img = build_camera_image(frame_idx)
        if cam_img is not None:
            cv2.namedWindow("CAMERAS (GT bbox)", cv2.WINDOW_NORMAL)
            cv2.imshow("CAMERAS (GT bbox)", cam_img)

    print(f"フレーム {g.frame_name(frame_idx)} のアノテーションを開始します。")
    while True:
        cv2.imshow(win, ann.render())
        key = cv2.waitKey(20) & 0xFF
        if key == ord("s"):
            ann.save()
        elif key == ord("u"):
            ann.undo()
        elif key == ord("g"):
            ann.show_ref = not ann.show_ref
        elif key == ord("n"):
            ann.show_ids = not ann.show_ids
        elif key in (ord("q"), 27):
            if ann.dirty and not ann.quit_warned:
                print("未保存の変更があります。保存するなら s、破棄して終了するならもう一度 q を押してください。")
                ann.quit_warned = True
                continue
            break
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()