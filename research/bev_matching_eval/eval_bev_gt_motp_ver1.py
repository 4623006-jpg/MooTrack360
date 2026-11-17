"""
eval_bev_gt_motp_ver1.py

手動でクリックして作った1フレーム分のBEVのGT(annotate_bev_gt_web_*.py の保存jsonの点)を正解として、
「GTのbboxを高さ仮定でBEVに落とした位置」の位置誤差(MOTP)を出す。

MOTP = 対応づけできたペアの距離[m]の平均(距離版)。
  対応づけ : 予測点と正解点をハンガリアン法(距離最小)で1対1に対応づけ、距離が閾値以内のペアだけ採用。
             正解点には牛名が無いので、対応づけは位置だけで行う。
  出力     : モードごと・閾値ごとに MOTP / 中央値 / 90%点 / 最大 / 対応数 / 正解の取りこぼし / 余った予測。

比較する予測点のモード(MODES):
  name_mean      : 牛名ごとに、全bboxの高さ仮定投影位置を平均(YOLO評価で使うGT位置と同じ作り)
  fused          : 曲線境界+牛舎範囲で除外 -> 隣接カメラ+姿勢制約+距離閾値で統合した位置
                   (GT入力のzonepose評価と同じ。統合のミスも誤差に入る)
  cam:Camera_N   : そのカメラ1台のbboxの投影位置だけ(カメラごとの高さ仮定の誤差を見る。
                   そのカメラに写っていない牛は「取りこぼし」になる)

注意:
  - 正解は手動クリックなので、クリックのずれ(数十cm)が誤差に含まれる。
  - GTに姿勢が無いので、高さは全員 DEFAULT(0.8m)の仮定になる。
  - 閾値を小さくしすぎると、対応づけ自体が減ってMOTPが良く見える(生存者バイアス)。
    対応数・取りこぼしと一緒に見ること。

出力(OUTPUT_DIR / bev_gt_motp /):
  motp_{frame}_ver1.csv        : モード・閾値ごとの全ペアの距離
  motp_{frame}_{mode}_ver1.png : 対応づけの図(黄=手動GT、水色=予測、赤線=対応と距離)

phase3_bev_multicam_gt_zonepose_ver1.py / annotate_bev_gt_click_ver1.py と同じフォルダに置く。

使い方:
    python eval_bev_gt_motp_ver1.py 100
"""

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYZE_DIR = os.path.join(THIS_DIR, "..", "analyze")
ANNOTATE_DIR = os.path.join(THIS_DIR, "..", "annotate")
PHASE3_DIR = os.path.join(THIS_DIR, "..", "phase3_multi_camera")
for d in (os.path.join(THIS_DIR, "..", "calibration"), ANALYZE_DIR, ANNOTATE_DIR, PHASE3_DIR, THIS_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

import annotate_bev_gt_click_ver1 as a      # build_base と SCALE を借りる
import phase3_bev_multicam_gt_zonepose_ver1 as g
from Mootrack_core import load_floorplan, union_find_clusters

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
DEFAULT_FRAME = 100
ANNOTATION_JSON = None            # Noneなら OUT_DIR/{frame}_bev_gt_ver1.json (annotate_bev_gt_web_*.py の保存先)
ANNOTATION_DIR = a.OUT_DIR if hasattr(a, "OUT_DIR") else g.OUTPUT_DIR / "bev_gt_annotation"
THRESHOLDS_M = [1.0, 3.0]         # 1.0 = 論文のτd、3.0 = 現在のYOLO評価の閾値
MODES = ["name_mean", "fused", "cam:Camera_1", "cam:Camera_2", "cam:Camera_3"]
DRAW_MODES = ["name_mean", "fused"]
DRAW_THR_M = 1.0
DETAIL_MODE = "name_mean"         # 対応の明細を端末に出すモード
OUT_DIR = g.OUTPUT_DIR / "bev_gt_motp"


# ---------------------------------------------------------------------------
# 正解(手動BEVGT)と予測点
# ---------------------------------------------------------------------------
def load_annotation(frame_idx):
    path = Path(ANNOTATION_JSON) if ANNOTATION_JSON else ANNOTATION_DIR / f"{g.frame_name(frame_idx)}_bev_gt_ver1.json"
    if not path.exists():
        raise FileNotFoundError(f"手動BEVGTのjsonが見つかりません: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    pts = [(int(p["id"]), np.array([p["x_m"], p["y_m"]], dtype=float)) for p in data["points"]]
    print(f"手動BEVGT: {path} ({len(pts)}点)")
    return pts


def build_predictions(frame_idx, poses, cam_polygons, adjacent_pairs, real_w, real_h):
    xml_path = Path(g.XML_DIR) / f"{g.frame_name(frame_idx)}.xml"
    if not xml_path.exists():
        raise FileNotFoundError(f"XMLが見つかりません: {xml_path}")
    gt_objects = g.parse_voc_annotation(xml_path)
    preds = {}

    # name_mean: 牛名ごとの平均位置
    gt_world = g.gt_world_points_this_frame(gt_objects, poses, g.IMAGE_WIDTH_PX)
    preds["name_mean"] = [(name, np.mean(np.array(v), axis=0)) for name, v in gt_world.items()]

    raw = g.gt_detections_per_camera(gt_objects, poses)

    # cam:Camera_N: そのカメラ1台のbboxの投影位置
    for cid in g.CAM_IDS:
        preds[f"cam:{cid}"] = [(it["name"], np.asarray(it["world_xy"], dtype=float)) for it in raw.get(cid, [])]

    # fused: 除外 + 隣接/姿勢/距離で統合
    kept = g.filter_by_curved_zone(raw, cam_polygons) if g.USE_ZONE_FILTER else raw
    if g.USE_FLOORPLAN_FILTER:
        kept = g.filter_outside_floorplan_bounds(kept, real_w, real_h, margin_m=0.0, verbose=False)
    pairs, _ = g.match_zone_pose_constrained(kept, adjacent_pairs, g.DIST_THR_FUSION_M, g.USE_POSE_CONSTRAINT)
    all_keys = [(c, i) for c, items in kept.items() for i in range(len(items))]
    roots = union_find_clusters(all_keys, pairs) if all_keys else {}
    groups = defaultdict(list)
    for k in all_keys:
        groups[roots[k]].append(k)
    fused = []
    for ms in groups.values():
        names = sorted({kept[c][i]["name"].replace("cow", "") for c, i in ms})
        fused.append(("/".join(names), np.mean([kept[c][i]["world_xy"] for c, i in ms], axis=0)))
    preds["fused"] = fused
    return preds


# ---------------------------------------------------------------------------
# 対応づけとMOTP
# ---------------------------------------------------------------------------
def match_points(pred, ann, thr):
    """pred, ann: [(label, xy)]。戻り値: matches[(pi, ai, dist)], 余った予測のindex, 取りこぼした正解のindex"""
    if not pred or not ann:
        return [], list(range(len(pred))), list(range(len(ann)))
    P = np.array([p[1] for p in pred])
    A = np.array([q[1] for q in ann])
    D = np.linalg.norm(P[:, None, :] - A[None, :, :], axis=2)
    BIG = 1e6
    rows, cols = linear_sum_assignment(np.where(D <= thr, D, BIG))
    matches = [(int(r), int(c), float(D[r, c])) for r, c in zip(rows, cols) if D[r, c] <= thr]
    mp, ma = {m[0] for m in matches}, {m[1] for m in matches}
    return (matches,
            [i for i in range(len(pred)) if i not in mp],
            [j for j in range(len(ann)) if j not in ma])


def summarize(matches, n_pred, n_ann):
    d = np.array([m[2] for m in matches]) if matches else np.array([])
    return {
        "n_ann": n_ann, "n_pred": n_pred, "n_match": len(matches),
        "miss": n_ann - len(matches), "fp": n_pred - len(matches),
        "motp": float(d.mean()) if len(d) else float("nan"),
        "median": float(np.median(d)) if len(d) else float("nan"),
        "p90": float(np.percentile(d, 90)) if len(d) else float("nan"),
        "max": float(d.max()) if len(d) else float("nan"),
    }


def draw_matching(base, ann, pred, matches, unm_pred, unm_ann, title):
    S = a.SCALE
    canvas = base.copy()
    for pi, ai, dist in matches:
        p1 = (int(pred[pi][1][0] * S), int(pred[pi][1][1] * S))
        p2 = (int(ann[ai][1][0] * S), int(ann[ai][1][1] * S))
        cv2.line(canvas, p1, p2, (60, 60, 255), 1, cv2.LINE_AA)
        mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        cv2.putText(canvas, f"{dist:.2f}", (mid[0] + 3, mid[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (120, 120, 255), 1, cv2.LINE_AA)
    for label, xy in pred:
        p = (int(xy[0] * S), int(xy[1] * S))
        cv2.circle(canvas, p, 5, (255, 220, 0), -1)
        cv2.putText(canvas, str(label).replace("cow", ""), (p[0] + 6, p[1] + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (255, 220, 0), 1, cv2.LINE_AA)
    for aid, xy in ann:
        p = (int(xy[0] * S), int(xy[1] * S))
        cv2.circle(canvas, p, 7, (0, 255, 255), 2)
        cv2.putText(canvas, f"#{aid}", (p[0] + 8, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 255, 255), 1, cv2.LINE_AA)
    for pi in unm_pred:      # 余った予測(マゼンタ輪)
        cv2.circle(canvas, (int(pred[pi][1][0] * S), int(pred[pi][1][1] * S)), 11, (255, 0, 255), 2)
    for ai in unm_ann:       # 取りこぼした正解(赤輪)
        cv2.circle(canvas, (int(ann[ai][1][0] * S), int(ann[ai][1][1] * S)), 13, (0, 0, 255), 2)
    cv2.putText(canvas, title, (8, canvas.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def fmt(v):
    return f"{v:6.3f}" if v == v else "   -  "


def main():
    frame_idx = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FRAME
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    regions, cameras_json, real_w, real_h = load_floorplan(g.FLOORPLAN_JSON_PATH)
    poses = g.resolve_poses(g.apply_camera_overrides(cameras_json), root="Area1", fmt=g.ROT_FMT)
    cam_polygons = g.build_camera_polygons(poses, g.CAM_IDS, g.get_intrinsics,
                                           height_m=g.ZONE_HEIGHT_M, n_angles=g.N_ANGLES)
    adjacent_pairs = g.get_adjacent_pairs(g.CAM_IDS)

    ann = load_annotation(frame_idx)
    preds = build_predictions(frame_idx, poses, cam_polygons, adjacent_pairs, real_w, real_h)
    base = a.build_base(regions, poses, cam_polygons, real_w, real_h)

    print(f"\n=== フレーム {g.frame_name(frame_idx)}  手動BEVGT={len(ann)}点 ===")
    print(f"{'モード':14s} {'閾値':>5s} {'MOTP':>7s} {'中央値':>7s} {'90%点':>7s} {'最大':>7s} "
          f"{'予測数':>6s} {'対応':>4s} {'取りこぼし':>8s} {'余り':>4s}   (距離はm)")
    rows = []
    results = {}
    for mode in MODES:
        pred = preds.get(mode, [])
        for thr in THRESHOLDS_M:
            matches, unm_pred, unm_ann = match_points(pred, ann, thr)
            s = summarize(matches, len(pred), len(ann))
            results[(mode, thr)] = (matches, unm_pred, unm_ann)
            print(f"{mode:14s} {thr:5.1f} {fmt(s['motp']):>7s} {fmt(s['median']):>7s} {fmt(s['p90']):>7s} "
                  f"{fmt(s['max']):>7s} {s['n_pred']:6d} {s['n_match']:4d} {s['miss']:8d} {s['fp']:4d}")
            for pi, ai, dist in matches:
                rows.append([mode, thr, ann[ai][0], pred[pi][0], f"{dist:.4f}"])

    csv_path = OUT_DIR / f"motp_{g.frame_name(frame_idx)}_ver1.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mode", "thr_m", "ann_id", "pred_label", "dist_m"])
        w.writerows(rows)
    print(f"\nCSV: {csv_path}")

    # 明細(DETAIL_MODE, 最大の閾値)
    thr = max(THRESHOLDS_M)
    if DETAIL_MODE in preds:
        pred = preds[DETAIL_MODE]
        matches, unm_pred, unm_ann = results[(DETAIL_MODE, thr)]
        print(f"\n--- 明細: {DETAIL_MODE} (閾値{thr}m) 距離の大きい順 ---")
        for pi, ai, dist in sorted(matches, key=lambda m: -m[2]):
            print(f"  手動#{ann[ai][0]:<3d} <-> cow{str(pred[pi][0]).replace('cow', ''):<5s} {dist:.3f} m")
        if unm_ann:
            print("  取りこぼした手動GT: " + ", ".join(f"#{ann[j][0]}" for j in unm_ann))
        if unm_pred:
            print("  余った予測: " + ", ".join(str(pred[i][0]) for i in unm_pred))

    for mode in DRAW_MODES:
        if mode not in preds:
            continue
        matches, unm_pred, unm_ann = match_points(preds[mode], ann, DRAW_THR_M)
        s = summarize(matches, len(preds[mode]), len(ann))
        title = (f"{g.frame_name(frame_idx)} {mode} thr={DRAW_THR_M}m  MOTP={fmt(s['motp']).strip()}m "
                 f"対応={s['n_match']} 取りこぼし(赤輪)={s['miss']} 余り(マゼンタ輪)={s['fp']}")
        img = draw_matching(base, ann, preds[mode], matches, unm_pred, unm_ann, title)
        out = OUT_DIR / f"motp_{g.frame_name(frame_idx)}_{mode.replace(':', '_')}_ver1.png"
        cv2.imwrite(str(out), img)
        print(f"画像: {out}")


if __name__ == "__main__":
    main()