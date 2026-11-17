"""
gridsearch_height_yolo_ver1.py

手動で作ったBEVのGT(json)を正解として、YOLOの姿勢クラスごとの高さ仮定
(lying / standing)のうち、位置誤差が最小になる値をグリッドサーチする。

YOLOの推論は各フレーム1回だけ行い、検出(bboxと姿勢クラス)をキャッシュして、
高さを変えて何度も投影し直す。

2種類の目的関数で探索する:
  [A] 固定ペアの距離(1次元・姿勢クラスごとに独立):
      現在の高さ(lying=0.5 / standing=0.75)で、カメラごとに検出と手動GTを対応づけて
      ペアを固定し、高さを変えたときのペアの距離の平均を見る。
      lyingの検出の位置はlyingの高さだけ、standingはstandingの高さだけで決まるので、
      2つは独立に最適化できる。対応づけの変化(生存者バイアス)が入らないのが利点。
  [B] 融合まで通した誤差(2次元グリッド):
      (lying, standing) の組ごとに、重複除去後の検出を投影し直し、曲線境界/牛舎範囲での除外 ->
      隣接カメラ+姿勢制約+距離閾値の統合 -> 手動GTと対応づけ(閾値 FUSED_THR_M)をやり直す。
      スコア = (対応したペアの距離の合計 + FUSED_THR_M x 取りこぼし数) / 手動GTの点数。
      取りこぼし(閾値内に対応が無い)を最大誤差として数えるので、
      「対応が減ってMOTPだけ良く見える」ことを防ぐ。高さは統合の成否にも影響する。

注意:
  - 手動GTのクリックのずれ、bbox中心と実際の位置のずれも含む。1フレームだと牛が数十頭でも
    姿勢別のサンプルが少なく、最適値がぶれる。手動GTを複数フレーム作り、FRAMES に並べるのが望ましい。
  - 探索範囲の端が最適なら、範囲を広げて再実行すること。
  - 求めた高さは「手動GTに合う高さ」。実際の牛の体高ではない(カメラの校正誤差なども吸収する)。

使い方:
    python gridsearch_height_yolo_ver1.py            # 手動GTのjsonがある全フレームを自動で使う
    python gridsearch_height_yolo_ver1.py 100 200    # フレームを指定

出力(OUTPUT_DIR / bev_yolo_height_gridsearch /):
  gridsearch_1d_ver1.csv     [A] 姿勢ごとの高さ別の平均・中央値
  gridsearch_2d_ver1.csv     [B] (lying, standing)ごとの全スコア
  gridsearch_2d_ver1.png     [B] ヒートマップ(白枠=最良、黒枠=現在値)

eval_bev_yolo_motp_ver2.py / eval_bev_gt_motp_ver1.py と同じフォルダに置く。
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import eval_bev_yolo_motp_ver2 as ev



e, y = ev.e, ev.y
CAM_IDS = ev.CAM_IDS

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
FRAMES = None                      # None: 手動GTのjsonがある全フレーム。例: [100, 200]
PAIR_THR_M = 3.0                   # [A] 現在の高さで対応づけてペアを固定するときの閾値
FUSED_THR_M = 1.0                  # [B] 融合後の位置と手動GTを対応づける閾値(取りこぼしの罰もこの値)
H1D = [round(h, 2) for h in np.arange(0.20, 1.31, 0.02)]       # [A] の探索範囲
LYING_GRID = [round(h, 2) for h in np.arange(0.20, 1.01, 0.05)]    # [B] lyingの探索範囲
STANDING_GRID = [round(h, 2) for h in np.arange(0.40, 1.31, 0.05)] # [B] standingの探索範囲
OUT_DIR = y.OUTPUT_DIR / "bev_yolo_height_gridsearch"

BASE_L = float(y.LYING_HEIGHT_M)
BASE_S = float(y.STANDING_HEIGHT_M)
DEFAULT_H = float(getattr(y, "DEFAULT_HEIGHT_M", 0.8))   # 姿勢クラス不明のときの高さ


def discover_frames():
    out = []
    for p in sorted(Path(e.ANNOTATION_DIR).glob("*_bev_gt_ver1.json")):
        try:
            out.append(int(p.name[:6]))
        except ValueError:
            pass
    return out


def heights_for(poses_list, h_l, h_s):
    return np.array([h_l if p == "lying" else h_s if p == "standing" else DEFAULT_H for p in poses_list],
                    dtype=np.float64)


def project_cam(poses, cid, px, heights):
    if len(px) == 0:
        return np.empty((0, 2)), np.zeros(0, dtype=bool)
    fx, fy, cx, cy, xi, alpha = ev.get_intrinsics(cid)
    P = poses[cid]
    xy, valid = ev.unproject_to_floor(px.astype(np.float64), P.R, P.t, fx, fy, cx, cy, xi, alpha,
                                      floor_z=-heights)
    return np.asarray(xy), np.asarray(valid, dtype=bool)


def detect_frame(frame_idx, model, poses):
    """YOLO推論+重複除去を1回だけ行い、bbox中心・姿勢クラス・bboxをカメラごとに保持"""
    image_path = Path(y.IMAGE_DIR) / f"{y.frame_name(frame_idx)}.jpg"
    if not image_path.exists():
        raise FileNotFoundError(f"画像が見つかりません: {image_path}")
    raw = y.yolo_world_points_per_camera_with_pose(model, image_path, poses)
    deduped = y.deduplicate_per_camera(raw, y.DEDUP_IOU_THRESHOLD)
    cams = {}
    for cid in CAM_IDS:
        items = deduped[cid]
        cams[cid] = {
            "pose": [it["pose"] for it in items],
            "bbox": [it["bbox_local"] for it in items],
            "px": np.array([[(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0] for b in (it["bbox_local"] for it in items)],
                           dtype=np.float64).reshape(-1, 2),
        }
    return cams


# ---------------------------------------------------------------------------
# [A] 固定ペアの距離
# ---------------------------------------------------------------------------
def build_fixed_pairs(frames, dets, anns, poses):
    pairs = []        # (cam, px(2,), pose, ann_xy(2,))
    for fr in frames:
        for cid in CAM_IDS:
            d = dets[fr][cid]
            if len(d["px"]) == 0:
                continue
            xy, valid = project_cam(poses, cid, d["px"], heights_for(d["pose"], BASE_L, BASE_S))
            idx_map = [i for i in range(len(valid)) if valid[i]]
            pred = [(d["pose"][i], xy[i]) for i in idx_map]
            matches, _up, _ua = e.match_points(pred, anns[fr], PAIR_THR_M)
            for pi, ai, _dist in matches:
                i = idx_map[pi]
                pairs.append((cid, d["px"][i], d["pose"][i], anns[fr][ai][1]))
    return pairs


def curve_1d(pairs, poses, pose_name):
    items = [p for p in pairs if p[2] == pose_name]
    res = []
    for h in H1D:
        ds = []
        for cid in CAM_IDS:
            sub = [p for p in items if p[0] == cid]
            if not sub:
                continue
            xy, valid = project_cam(poses, cid, np.array([p[1] for p in sub]), np.full(len(sub), h))
            for k in range(len(sub)):
                if valid[k]:
                    ds.append(float(np.linalg.norm(xy[k] - sub[k][3])))
        res.append((h, float(np.mean(ds)) if ds else float("nan"),
                    float(np.median(ds)) if ds else float("nan"), len(ds)))
    return res, len(items)


# ---------------------------------------------------------------------------
# [B] 融合まで通したスコア
# ---------------------------------------------------------------------------
def fused_score(h_l, h_s, frames, dets, anns, poses, cam_polygons, adjacent_pairs, real_w, real_h):
    sum_d = 0.0
    n_match = n_ann = n_pred = 0
    for fr in frames:
        per_camera = {}
        for cid in CAM_IDS:
            d = dets[fr][cid]
            xy, valid = project_cam(poses, cid, d["px"], heights_for(d["pose"], h_l, h_s))
            per_camera[cid] = [{"name": None, "pose": d["pose"][i], "bbox_local": d["bbox"][i], "world_xy": xy[i]}
                               for i in range(len(valid)) if valid[i]]
        filtered = ev.filter_by_curved_zone(per_camera, cam_polygons)
        filtered = ev.filter_outside_floorplan_bounds(filtered, real_w, real_h, margin_m=0.0, verbose=False)
        fused = y.fuse_frame_zone_pose(filtered, adjacent_pairs, y.DIST_THR_FUSION_M, y.USE_POSE_CONSTRAINT)[0]
        pred = [("f", np.asarray(f["world_xy"], dtype=float)) for f in fused]
        matches, _up, _ua = e.match_points(pred, anns[fr], FUSED_THR_M)
        sum_d += sum(m[2] for m in matches)
        n_match += len(matches)
        n_ann += len(anns[fr])
        n_pred += len(pred)
    score = (sum_d + FUSED_THR_M * (n_ann - n_match)) / max(n_ann, 1)
    motp = sum_d / n_match if n_match else float("nan")
    return {"score": score, "motp": motp, "n_match": n_match, "miss": n_ann - n_match,
            "fp": n_pred - n_match, "n_ann": n_ann}


def draw_heatmap(score, best, base, path):
    cell = 30
    left, top = 70, 40
    H, W = score.shape
    img = np.full((top + H * cell + 50, left + W * cell + 20, 3), 255, dtype=np.uint8)
    lo, hi = float(np.nanmin(score)), float(np.nanmax(score))
    norm = ((score - lo) / (hi - lo + 1e-12) * 255).astype(np.uint8)
    cm = cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS)
    # 小さいスコア(=良い)を明るい色にする
    cm = cv2.applyColorMap(255 - norm, cv2.COLORMAP_VIRIDIS)
    for i in range(H):
        for j in range(W):
            x0, y0 = left + j * cell, top + i * cell
            img[y0:y0 + cell, x0:x0 + cell] = cm[i, j]
            cv2.putText(img, f"{score[i, j]:.2f}", (x0 + 2, y0 + cell // 2 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                        (0, 0, 0), 1, cv2.LINE_AA)
    for (i, j), col in ((best, (255, 255, 255)), (base, (0, 0, 0))):
        if i is not None:
            cv2.rectangle(img, (left + j * cell, top + i * cell), (left + (j + 1) * cell, top + (i + 1) * cell), col, 2)
    for j in range(W):
        if j % 2 == 0:
            cv2.putText(img, f"{STANDING_GRID[j]:.2f}", (left + j * cell, top - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        (0, 0, 0), 1, cv2.LINE_AA)
    for i in range(H):
        cv2.putText(img, f"{LYING_GRID[i]:.2f}", (8, top + i * cell + cell // 2 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "x: standing[m]  y: lying[m]  score(low=good, bright)  white=best black=current",
                (8, top + H * cell + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


def main():
    frames = [int(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else (FRAMES or discover_frames())
    if not frames:
        raise SystemExit(f"手動GTのjsonが見つかりません: {e.ANNOTATION_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO

    regions, cameras_json, real_w, real_h = e.load_floorplan(y.FLOORPLAN_JSON_PATH)
    poses = e.g.resolve_poses(e.g.apply_camera_overrides(cameras_json), root="Area1", fmt=e.g.ROT_FMT)
    cam_polygons = e.g.build_camera_polygons(poses, e.g.CAM_IDS, e.g.get_intrinsics,
                                             height_m=e.g.ZONE_HEIGHT_M, n_angles=e.g.N_ANGLES)
    adjacent_pairs = e.g.get_adjacent_pairs(e.g.CAM_IDS)

    print(f"対象フレーム: {frames}")
    anns = {fr: e.load_annotation(fr) for fr in frames}
    print(f"YOLOモデル読み込み中: {y.YOLO_WEIGHTS_PATH} (conf>={y.YOLO_CONF_THRESH})")
    model = YOLO(y.YOLO_WEIGHTS_PATH)
    dets = {}
    for fr in frames:
        dets[fr] = detect_frame(fr, model, poses)
        n = {c: len(dets[fr][c]["pose"]) for c in CAM_IDS}
        cnt = defaultdict(int)
        for c in CAM_IDS:
            for p in dets[fr][c]["pose"]:
                cnt[p] += 1
        print(f"  {y.frame_name(fr)}: 検出(重複除去後) {n} 姿勢 {dict(cnt)}  手動GT={len(anns[fr])}点")

    # ---------------- [A] 1次元 ----------------
    pairs = build_fixed_pairs(frames, dets, anns, poses)
    print(f"\n=== [A] 固定ペアの距離 (現在の高さ lying={BASE_L}, standing={BASE_S} で閾値{PAIR_THR_M}mで対応づけ) ===")
    rows_1d = []
    best_1d = {}
    for pose_name, base_h in (("lying", BASE_L), ("standing", BASE_S)):
        curve, n_pairs = curve_1d(pairs, poses, pose_name)
        if not curve or n_pairs == 0:
            print(f"  {pose_name}: ペアがありません")
            continue
        means = np.array([c[1] for c in curve])
        meds = np.array([c[2] for c in curve])
        hb_mean = curve[int(np.nanargmin(means))][0]
        hb_med = curve[int(np.nanargmin(meds))][0]
        cur = min(curve, key=lambda c: abs(c[0] - base_h))
        print(f"  {pose_name:9s} ペア数={n_pairs:3d}  平均が最小の高さ={hb_mean:.2f}m (誤差{np.nanmin(means):.3f}m)  "
              f"中央値が最小の高さ={hb_med:.2f}m (誤差{np.nanmin(meds):.3f}m)  "
              f"現在{cur[0]:.2f}m の誤差: 平均{cur[1]:.3f} 中央値{cur[2]:.3f}")
        best_1d[pose_name] = hb_mean
        for h, mean_d, med_d, n in curve:
            rows_1d.append([pose_name, h, n, f"{mean_d:.4f}", f"{med_d:.4f}"])
        edge = hb_mean in (H1D[0], H1D[-1])
        if edge:
            print(f"    ※最適が探索範囲の端です。H1D の範囲を広げてください。")
    with open(OUT_DIR / "gridsearch_1d_ver1.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pose", "height_m", "n", "mean_dist_m", "median_dist_m"])
        w.writerows(rows_1d)

    # ---------------- [B] 2次元 ----------------
    total = len(LYING_GRID) * len(STANDING_GRID)
    print(f"\n=== [B] 融合まで通した2次元グリッド ({len(LYING_GRID)} x {len(STANDING_GRID)} = {total}通り, "
          f"閾値{FUSED_THR_M}m) ===")
    score = np.full((len(LYING_GRID), len(STANDING_GRID)), np.nan)
    results = {}
    k = 0
    for i, hl in enumerate(LYING_GRID):
        for j, hs in enumerate(STANDING_GRID):
            r = fused_score(hl, hs, frames, dets, anns, poses, cam_polygons, adjacent_pairs, real_w, real_h)
            results[(i, j)] = r
            score[i, j] = r["score"]
            k += 1
            if k % 50 == 0:
                print(f"  {k}/{total}")
    bi, bj = np.unravel_index(np.nanargmin(score), score.shape)
    ci = int(np.argmin([abs(v - BASE_L) for v in LYING_GRID]))
    cj = int(np.argmin([abs(v - BASE_S) for v in STANDING_GRID]))

    def show(tag, i, j):
        r = results[(i, j)]
        print(f"  {tag}: lying={LYING_GRID[i]:.2f} standing={STANDING_GRID[j]:.2f}  スコア={r['score']:.3f}  "
              f"MOTP={r['motp']:.3f}  対応={r['n_match']}/{r['n_ann']} 取りこぼし={r['miss']} 余り={r['fp']}")

    show("最良      ", bi, bj)
    show("現在(近い)", ci, cj)
    print("  上位10:")
    order = np.dstack(np.unravel_index(np.argsort(np.where(np.isnan(score), np.inf, score), axis=None), score.shape))[0][:10]
    for i, j in order:
        show("   ", int(i), int(j))
    if bi in (0, len(LYING_GRID) - 1) or bj in (0, len(STANDING_GRID) - 1):
        print("  ※最良が探索範囲の端です。LYING_GRID / STANDING_GRID の範囲を広げてください。")

    with open(OUT_DIR / "gridsearch_2d_ver1.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lying_m", "standing_m", "score", "motp_m", "n_match", "miss", "fp", "n_ann"])
        for (i, j), r in results.items():
            w.writerow([LYING_GRID[i], STANDING_GRID[j], f"{r['score']:.4f}", f"{r['motp']:.4f}",
                        r["n_match"], r["miss"], r["fp"], r["n_ann"]])
    draw_heatmap(score, (int(bi), int(bj)), (ci, cj), OUT_DIR / "gridsearch_2d_ver1.png")
    print(f"\n出力: {OUT_DIR}")
    print("※姿勢ごとの最適値は、サンプル数が少ないとぶれる。ペア数が少ない姿勢クラスの結果は参考程度に。")


if __name__ == "__main__":
    main()