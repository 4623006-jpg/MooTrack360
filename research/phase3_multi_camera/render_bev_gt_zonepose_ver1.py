"""
render_bev_gt_zonepose_ver1.py

phase3_bev_multicam_gt_zonepose_ver1.py と同じフォルダに置く。
同じ設定(GTのbboxを入力、曲線境界+牛舎範囲の除外、隣接ペア+姿勢制約+距離閾値の統合、
ReIDトラッカー)で、指定フレームのBEV(牛舎の壁・エリアつき)を描く。

トラッカーは状態を持つので、FRAME_START から最大の FRAME_IDXS まで順に回し、
FRAME_IDXS に含まれるフレームだけ画像を保存する。

BEVの凡例:
  カメラ色の小さい丸 : 各カメラのbbox(GT)の高さ仮定投影位置。統合後の位置と細線で結ぶ
  灰の×             : 曲線境界/牛舎範囲の外で除外された検出
  大きい丸(青/灰)    : 統合後の位置(青=2カメラ以上、灰=1カメラ)。牛番号と #トラックID を併記
  マゼンタ輪 MERGE   : 異なる牛を誤って統合
  赤輪 SPLIT         : 同じ牛なのに別クラスタに分裂
  色つきの細い曲線    : 各カメラの曲線境界

使い方:
    python render_bev_gt_zonepose_ver1.py
"""

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import phase3_bev_multicam_gt_zonepose_ver1 as g
from Mootrack_core import load_floorplan, union_find_clusters  # gのimportでsys.pathは通っている

FRAME_IDXS = [0, 100, 200]
SCALE = 40
OUT_DIR = g.OUTPUT_DIR / "bev_gt_zonepose"

TYPE_COLORS_BGR = {
    "wall": (0, 200, 0), "cow_beds": (0, 140, 255), "feeding_bay": (255, 140, 0),
    "water": (255, 255, 0), "gate": (0, 0, 255), "milking_station": (255, 0, 255),
    "hay_area": (0, 255, 255), "cow_brushes": (255, 0, 180),
}
CAM_COLORS_BGR = {"Camera_1": (255, 120, 60), "Camera_2": (60, 200, 60), "Camera_3": (60, 140, 255)}
NAME_COLOR = (0, 255, 255)


def draw(frame_idx, regions, poses, cam_polygons, real_w, real_h, raw, kept, members, fused, track_id_of,
         tp, fn, fp):
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

    kept_uids = {it["uid"] for items in kept.values() for it in items}
    for items in raw.values():
        for it in items:
            if it["uid"] not in kept_uids:
                cv2.drawMarker(canvas, (int(it["world_xy"][0] * SCALE), int(it["world_xy"][1] * SCALE)),
                               (120, 120, 120), markerType=cv2.MARKER_TILTED_CROSS, markerSize=8, thickness=1)

    name_of = {(c, i): kept[c][i]["name"] for c, items in kept.items() for i in range(len(items))}
    for k, ms in enumerate(members):
        fx_, fy_ = int(fused[k]["world_xy"][0] * SCALE), int(fused[k]["world_xy"][1] * SCALE)
        for c, i in ms:
            q = kept[c][i]["world_xy"]
            col = CAM_COLORS_BGR.get(c, (255, 255, 255))
            cv2.line(canvas, (int(q[0] * SCALE), int(q[1] * SCALE)), (fx_, fy_), col, 1, cv2.LINE_AA)
            cv2.circle(canvas, (int(q[0] * SCALE), int(q[1] * SCALE)), 4, col, -1)

    split_names = defaultdict(set)
    for k, ms in enumerate(members):
        names = {name_of[m] for m in ms}
        px, py = int(fused[k]["world_xy"][0] * SCALE), int(fused[k]["world_xy"][1] * SCALE)
        multi = len(ms) >= 2
        cv2.circle(canvas, (px, py), 7, (255, 160, 60) if multi else (170, 170, 170), -1)
        cv2.circle(canvas, (px, py), 9, (255, 255, 255) if multi else (0, 0, 0), 2 if multi else 1)
        tid = track_id_of.get(k)
        label = "/".join(sorted(n.replace("cow", "") for n in names)) + (f" #{tid}" if tid is not None else "")
        cv2.putText(canvas, label, (px + 10, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, NAME_COLOR, 1, cv2.LINE_AA)
        if len(names) > 1:
            cv2.circle(canvas, (px, py), 15, (255, 0, 255), 2)
            cv2.putText(canvas, "MERGE", (px + 8, py + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1, cv2.LINE_AA)
        for n in names:
            split_names[n].add(k)
    for n, ks in split_names.items():
        if len(ks) > 1:
            p = np.mean([fused[k]["world_xy"] for k in ks], axis=0)
            cv2.circle(canvas, (int(p[0] * SCALE), int(p[1] * SCALE)), 18, (0, 0, 255), 2)
            cv2.putText(canvas, "SPLIT", (int(p[0] * SCALE) + 8, int(p[1] * SCALE) + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

    n_raw, n_kept = sum(len(v) for v in raw.values()), sum(len(v) for v in kept.values())
    header = (f"{frame_idx:06d} GT入力 距離<={g.DIST_THR_FUSION_M}m  bbox {n_raw}->範囲内{n_kept}  "
              f"クラスタ={len(members)}  ペア TP={tp} FN(統合漏れ)={fn} FP(誤統合)={fp}")
    cv2.putText(canvas, header, (8, h_px - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    for k, cid in enumerate(g.CAM_IDS):
        y = 20 + k * 22
        cv2.rectangle(canvas, (w_px - margin + 5, y), (w_px - margin + 21, y + 16), CAM_COLORS_BGR[cid], -1)
        cv2.putText(canvas, cid, (w_px - margin + 26, y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    regions, cameras_json, real_w, real_h = load_floorplan(g.FLOORPLAN_JSON_PATH)
    poses = g.resolve_poses(g.apply_camera_overrides(cameras_json), root="Area1", fmt=g.ROT_FMT)
    cam_polygons = g.build_camera_polygons(poses, g.CAM_IDS, g.get_intrinsics,
                                           height_m=g.ZONE_HEIGHT_M, n_angles=g.N_ANGLES)
    adjacent_pairs = g.get_adjacent_pairs(g.CAM_IDS)

    if g.USE_REID_TRACKER:
        tracker = g.ReIDBevTracker(
            dist_thr=g.DIST_THR_TRACK_M, max_missed=g.MAX_MISSED_FRAMES,
            real_w=real_w, real_h=real_h, edge_margin_m=g.EDGE_MARGIN_M,
            reid_dist_thr=g.REID_DIST_THR_M, lost_max_age=g.LOST_MAX_AGE)
    else:
        tracker = g.SimpleBevTracker(g.DIST_THR_TRACK_M, g.MAX_MISSED_FRAMES)

    for frame_idx in range(g.FRAME_START, max(FRAME_IDXS) + 1, g.FRAME_STEP):
        xml_path = Path(g.XML_DIR) / f"{g.frame_name(frame_idx)}.xml"
        if not xml_path.exists():
            continue
        raw = g.gt_detections_per_camera(g.parse_voc_annotation(xml_path), poses)
        uid = 0
        for items in raw.values():
            for it in items:
                uid += 1
                it["uid"] = uid
        kept = g.filter_by_curved_zone(raw, cam_polygons) if g.USE_ZONE_FILTER else raw
        if g.USE_FLOORPLAN_FILTER:
            kept = g.filter_outside_floorplan_bounds(kept, real_w, real_h, margin_m=0.0, verbose=False)

        pairs, _ = g.match_zone_pose_constrained(kept, adjacent_pairs, g.DIST_THR_FUSION_M, g.USE_POSE_CONSTRAINT)
        all_keys = [(c, i) for c, items in kept.items() for i in range(len(items))]
        roots = union_find_clusters(all_keys, pairs) if all_keys else {}
        groups = defaultdict(list)
        for k in all_keys:
            groups[roots[k]].append(k)
        members = list(groups.values())
        fused = [{"world_xy": np.mean([kept[c][i]["world_xy"] for c, i in ms], axis=0), "n_cams": len(ms)}
                 for ms in members]

        outputs = tracker.update(fused)      # [(track_id, xy), ...]
        track_id_of = {}
        for tid, xy in outputs:
            for k, f in enumerate(fused):
                if np.linalg.norm(np.asarray(xy) - f["world_xy"]) < 1e-6:
                    track_id_of[k] = tid
                    break

        if frame_idx not in FRAME_IDXS:
            continue
        name_of = {(c, i): kept[c][i]["name"] for c, items in kept.items() for i in range(len(items))}
        cluster_of = {m: k for k, ms in enumerate(members) for m in ms}
        keys = list(name_of)
        tp = fn = fp = 0
        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                if keys[a][0] == keys[b][0]:
                    continue
                same_name = name_of[keys[a]] == name_of[keys[b]]
                same_cl = cluster_of[keys[a]] == cluster_of[keys[b]]
                tp += same_name and same_cl
                fn += same_name and not same_cl
                fp += (not same_name) and same_cl

        canvas = draw(frame_idx, regions, poses, cam_polygons, real_w, real_h, raw, kept, members, fused,
                      track_id_of, tp, fn, fp)
        out = OUT_DIR / f"bev_gt_zonepose_ver1_{g.frame_name(frame_idx)}.png"
        cv2.imwrite(str(out), canvas)
        print(f"保存: {out}  クラスタ={len(members)} TP={tp} FN={fn} FP={fp}")


if __name__ == "__main__":
    main()