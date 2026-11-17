"""
render_idf1_det_events_ver1.py  (配置: research/phase5/)

record_idf1_det_events_ver1.py が記録した events_<mode>.json を読み、IDFP / IDFN の該当部分を強調した画像を出す。
YOLO や統合は回さない(記録と元画像だけ使う)。

画像の内容(1 枚 = 1 フレーム × 1 カメラのタイル):
  灰色の細い枠   : そのカメラの検出(ラベル = 追跡 ID : 名前)
  緑の細い枠     : その牛の GT bbox(ラベル = 名前)
  赤の太い枠     : IDFP になった検出(ラベル = IDFP + 理由コード)
  青の太い枠     : IDFN になった GT の牛(ラベル = IDFN + 理由コード、他のカメラで検出できていれば "other cam OK")
  マゼンタの枠   : IDFP の検出と同じクラスタに入った、他のカメラの検出(MOSAIC=1 のとき、そのカメラのタイルも並べて出す)
  左上の文字     : フレーム名、カメラ、手法

使い方(research/ で):
    uv run python ./phase5/render_idf1_det_events_ver1.py baseline_out/idf1_det_events_ver1_px_var/events_euclid.json
絞り込み(環境変数、すべて任意):
    KIND=IDFP|IDFN           種別(既定: 両方)
    REASON=MISSED,NONAME_*   理由コード(カンマ区切り、* の前方一致が使える。既定: 全部)
    CAM=cam_id               カメラ(既定: 全部)
    NAME=牛の名前            牛(既定: 全部)
    FRAME_FROM / FRAME_TO    フレーム番号の範囲
    MAX_IMAGES=60            出す枚数の上限(既定 60。(フレーム, カメラ)ごとに 1 枚)
    MOSAIC=1                 IDFP の同じクラスタの、他のカメラのタイルも横に並べる
    SCALE=1.0                画像の拡大率
出力: events json と同じフォルダの images_<mode>/ 。ファイル名: <フレーム名>_<カメラ>_<種別と理由>.jpg
"""

import fnmatch
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

GRAY, GREEN, RED, BLUE, MAGENTA = (150, 150, 150), (0, 200, 0), (0, 0, 255), (255, 80, 0), (255, 0, 255)


def _match_reason(code, patterns):
    return not patterns or any(fnmatch.fnmatch(code, p) for p in patterns)


def _box(img, b, color, thick, label=None, label_pos="top"):
    x0, y0, x1, y1 = [int(round(v)) for v in b]
    cv2.rectangle(img, (x0, y0), (x1, y1), color, thick)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = y0 - 4 if label_pos == "top" and y0 - th - 6 > 0 else y1 + th + 4
        cv2.rectangle(img, (x0, ty - th - 3), (x0 + tw + 4, ty + 3), color, -1)
        cv2.putText(img, label, (x0 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def crop_tile(full, cam_idx, tile_w):
    x0 = int(cam_idx * tile_w)
    x1 = int((cam_idx + 1) * tile_w)
    return full[:, x0:x1].copy()


def main():
    if len(sys.argv) < 2:
        raise SystemExit("使い方: python render_idf1_det_events_ver1.py events_<mode>.json")
    path = Path(sys.argv[1])
    data = json.loads(path.read_text(encoding="utf-8"))
    meta, events = data["meta"], data["events"]
    tile_w, image_dir, cam_ids = meta["tile_w"], Path(meta["image_dir"]), meta["cam_ids"]

    kind = os.environ.get("KIND", "")
    patterns = [s.strip() for s in os.environ.get("REASON", "").split(",") if s.strip()]
    cam = os.environ.get("CAM", "")
    name = os.environ.get("NAME", "")
    f_from, f_to = os.environ.get("FRAME_FROM"), os.environ.get("FRAME_TO")
    max_images = int(os.environ.get("MAX_IMAGES", 60))
    mosaic = os.environ.get("MOSAIC", "0") == "1"
    scale = float(os.environ.get("SCALE", 1.0))

    def keep(e):
        return ((not kind or e["kind"] == kind) and _match_reason(e["reason"], patterns) and (not cam or e["cam"] == cam)
                and (not name or str(e["name"]) == name) and (f_from is None or e["frame"] >= int(f_from))
                and (f_to is None or e["frame"] <= int(f_to)))

    sel = [e for e in events if keep(e)]
    groups = defaultdict(list)          # (フレーム, カメラ) → イベント
    for e in sel:
        groups[(e["frame"], e["cam"])].append(e)
    print(f"{path.name}: 全イベント {len(events)} 件 → 条件に合うもの {len(sel)} 件 / (フレーム, カメラ) {len(groups)} 組 → 上限 {max_images} 枚")

    tiles_ctx = data.get("tiles", {})      # "フレーム|カメラ" → そのタイルの全検出と全 GT(TP を含む)。イベントのあったタイルだけ
    all_by_tile = defaultdict(list)
    for e in events:
        all_by_tile[(e["frame"], e["cam"])].append(e)

    out_dir = path.parent / f"images_{meta['mode'].replace('@', '_')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for (frame, cid), evs in sorted(groups.items()):
        if n >= max_images:
            break
        e0 = evs[0]
        img_path = image_dir / f"{e0['frame_name']}.jpg"
        full = cv2.imread(str(img_path))
        if full is None:
            print(f"  画像が読めない: {img_path}")
            continue
        tiles = [(cid, e0["cam_idx"], evs, [])]
        if mosaic:
            for e in evs:
                if e["kind"] == "IDFP":
                    for c2, j in e.get("cluster_members", []):
                        if c2 != cid:
                            ci2 = cam_ids.index(c2)
                            if all(t[0] != c2 for t in tiles):
                                tiles.append((c2, ci2, [], []))
                            for t in tiles:
                                if t[0] == c2:
                                    t[3].append(j)
        panels = []
        for c_, ci_, evs_, partner_idx in tiles:
            tile = crop_tile(full, ci_, tile_w)
            # 灰・緑: そのタイルの全検出と全 GT(TP も含む)
            ctx = tiles_ctx.get(f"{frame}|{c_}")
            if ctx:
                for d in ctx["det"]:
                    _box(tile, d["bbox"], GRAY, 1, f"id{d['hyp']}:{d['name']}")
                for g in ctx["gt"]:
                    _box(tile, g["bbox"], GREEN, 1, str(g["name"]) + ("" if g["in_zone"] else " (out)"), "bottom")
            for e in evs_:
                if e["kind"] == "IDFP":
                    _box(tile, e["bbox"], RED, 3, f"IDFP {e['reason']} id{e['hyp']}:{e['name']}")
                else:
                    extra = " other cam OK" if e.get("seen_other_cam") else ""
                    _box(tile, e["bbox"], BLUE, 3, f"IDFN {e['reason']} {e['name']}{extra}", "bottom")
            if ctx:      # 同じクラスタの相手(他のカメラ側)の検出を強調(そのタイルにイベントが無いと全体像が無いので、枠は出ない)
                for d in ctx["det"]:
                    if d["i"] in partner_idx:
                        _box(tile, d["bbox"], MAGENTA, 3, f"cluster id{d['hyp']}:{d['name']}")
            cv2.putText(tile, f"{e0['frame_name']} {c_} [{meta['mode']}]", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
            panels.append(tile)
        if len(panels) > 1:
            h = max(p.shape[0] for p in panels)
            panels = [cv2.copyMakeBorder(p, 0, h - p.shape[0], 0, 4, cv2.BORDER_CONSTANT, value=(40, 40, 40)) for p in panels]
        canvas = np.hstack(panels)
        if scale != 1.0:
            canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        tag = "_".join(sorted({f"{e['kind']}-{e['reason']}" for e in evs}))[:80]
        out = out_dir / f"{e0['frame_name']}_{cid}_{tag}.jpg"
        cv2.imwrite(str(out), canvas)
        n += 1
    print(f"保存: {n} 枚 → {out_dir}")


if __name__ == "__main__":
    main()