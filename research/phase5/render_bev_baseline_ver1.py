"""
render_bev_baseline_ver1.py  (配置: research/phase5/)

従来手法 B(IoU 統合)が悪い原因を、動画で見るためのスクリプト。
BEV(牛舎を真上から見た図)の動画に、次を重ねる:
  ・各カメラの検出 bbox を床面に投影した多角形(カメラごとの検出)と、その牛の名前(GT との IoU≥0.3 で対応づけ)
  ・カメラ間の候補ペアの IoU の値(線で結ぶ。採用したペアは実線、採用しなかったものは灰色)
  ・統合(クラスタ)の正しさで多角形の色を変える:
        緑   正しい統合   : 同じ牛の検出どうしが統合された
        赤   誤統合       : 別の牛の検出が同じクラスタに入った
        橙   統合し損ね   : 同じ牛の検出が別々のクラスタに分かれた(重複 -> FP の原因)
        白   単独         : 統合なし(問題なし)
        灰   名前なし     : GT に対応のない検出
  ・追跡の結果: track ID と軌跡、GT の十字、そのフレームの FN(赤い輪) / FP(マゼンタの輪) / SW(黄色の輪)と、その原因の推定
        FN の原因: 誤統合(その牛が誤統合クラスタにいる) / 未検出(その牛の検出が無い) / その他
        FP の原因: 重複(統合し損ねたクラスタの近く) / その他
        SW の原因: 誤統合・重複のクラスタの近く / その他
  ・左上に、フレームごとの 誤統合 / 重複 の数、FN/FP/SW の数。終了時に原因別の合計を表示。
描画・追跡・評価は render_bev_trajectory_he_kai_ver6.py と同じ(motmetrics、ReID トラッカー、対応づけのゲート 2×追跡閾値)。
B は baseline_iou_ver1.py と同じ設定(想定高さ phase3 の値、有効範囲は HE のみの円、単純平均で統合)。

使い方(research/ で):
    HE_POSE_CONSTRAINT=0 uv run python ./phase5/render_bev_baseline_ver1.py             # IoU 0.2(掃引の最良値)
    HE_POSE_CONSTRAINT=0 uv run python ./phase5/render_bev_baseline_ver1.py iou@0.1
    HE_FRAME_START=100 HE_FRAME_END=200 HE_FRAME_STEP=10 ...                            # フレームを絞る(cow036 の 150〜180 など)
環境変数:
    RB_ERR_ONLY=1   誤り(FN/FP/SW/誤統合/重複)のあったフレームとその前後だけを動画にする(PNG は全フレーム保存)
    RB_CONTEXT=1    RB_ERR_ONLY のとき、前後に何フレーム残すか (既定 1)
    HE_FRAME_START / HE_FRAME_END / HE_FRAME_STEP, HE_POSE_CONSTRAINT など: baseline_iou_ver1.py / render_bev_trajectory_he_kai_ver6.py と同じ
出力: research/phase5/baseline_out/bev_video_ver1/<iou_tX.XX>/frames/*.png と bev_baseline_<iou_tX.XX>.mp4
前提: research/phase4/he_kai/render_bev_trajectory_he_kai_ver6.py が import できること(描画・追跡の部品を借りる)
"""

import os
import re
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import baseline_iou_ver1 as bl          # パス設定(Mootrack_core / YOLO モジュールの探索)と B の多角形づくりを共有する

SCALE = 40                              # BEV の 1 m あたりの画素数(draw_frame_trajectory と同じ)
ERR_ONLY = os.environ.get("RB_ERR_ONLY", "0") == "1"
CONTEXT = int(os.environ.get("RB_CONTEXT", 1))
OUT_ROOT = Path(os.environ.get("RB_OUT_DIR", os.path.join(THIS_DIR, "baseline_out", "bev_video_ver1")))
NAME_IOU = 0.3                          # 検出と GT の bbox を対応づける IoU(σ の推定と同じ)

# BGR
C_OK, C_FALSE, C_SPLIT, C_SINGLE, C_NONAME = (0, 200, 0), (0, 0, 255), (0, 140, 255), (255, 255, 255), (140, 140, 140)
C_FN, C_FP, C_SW = (0, 0, 255), (255, 0, 255), (0, 255, 255)
STATE_COLOR = {"correct_merge": C_OK, "false_merge": C_FALSE, "split": C_SPLIT, "single": C_SINGLE, "unnamed": C_NONAME}
STATE_EN = {"correct_merge": "OK merge", "false_merge": "FALSE MERGE", "split": "SPLIT(dup)", "single": "single", "unnamed": "no GT name"}
CAUSE_EN = {"誤統合": "false-merge", "未検出": "no-det", "その他": "other", "重複": "dup", "誤統合近傍": "near false-merge", "重複近傍": "near dup"}
STATE_JA = {"correct_merge": "正しい統合", "false_merge": "誤統合", "split": "統合し損ね(重複)", "single": "単独", "unnamed": "名前なし"}


# ---------------------------------------------------------------------------
# 検出に牛の名前を付ける
# ---------------------------------------------------------------------------
def _iou_box(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def assign_names(gts, dets, thr=NAME_IOU):
    """GT bbox(タイル内座標)と YOLO 検出を、IoU の大きい順に 1 対 1 で対応づける。{gt番号: det番号}"""
    cand = sorted(((_iou_box(g, d["bbox_local"]), gi, di) for gi, g in enumerate(gts) for di, d in enumerate(dets)), reverse=True)
    used_g, used_d, out = set(), set(), {}
    for v, gi, di in cand:
        if v < thr:
            break
        if gi in used_g or di in used_d:
            continue
        used_g.add(gi)
        used_d.add(di)
        out[gi] = di
    return out


def detection_names(gt_objects, per_camera, cam_ids, image_width, exclude=()):
    """{(cam_id, 検出番号): 牛の名前}。GT の無い検出は含まれない"""
    tile_w = image_width / len(cam_ids)
    names = {}
    for ci, cid in enumerate(cam_ids):
        g_list = []
        for o in gt_objects:
            if o["name"] in exclude:
                continue
            cxm = (o["xmin"] + o["xmax"]) / 2.0
            if max(0, min(int(cxm // tile_w), len(cam_ids) - 1)) == ci:
                g_list.append((o["name"], [o["xmin"] - ci * tile_w, o["ymin"], o["xmax"] - ci * tile_w, o["ymax"]]))
        dets = per_camera.get(cid, [])
        m = assign_names([b for _, b in g_list], dets)
        for gi, di in m.items():
            names[(cid, di)] = g_list[gi][0]
    return names


# ---------------------------------------------------------------------------
# B(IoU 統合)。baseline_iou_ver1.fuse_iou と同じ処理で、途中経過(多角形・IoU・クラスタ)も返す
# ---------------------------------------------------------------------------
def fuse_iou_detail(per_camera, adjacent_pairs, thr, use_pose, height_for):
    from Mootrack_core import union_find_clusters
    polys = {(cid, i): bl.det_polygon(cid, it, height_for(it.get("pose"))) for cid, items in per_camera.items() for i, it in enumerate(items)}
    ious, accepted = {}, []
    for ci, cj in adjacent_pairs:
        A, B = per_camera.get(ci, []), per_camera.get(cj, [])
        if not A or not B:
            continue
        iou_mat = np.zeros((len(A), len(B)))
        allowed = np.zeros((len(A), len(B)), dtype=bool)
        for ia, a in enumerate(A):
            for ib, b in enumerate(B):
                pa, pb = polys[(ci, ia)], polys[(cj, ib)]
                if pa is None or pb is None:
                    continue
                inter = pa.intersection(pb).area
                if inter <= 0:
                    continue
                v = inter / (pa.area + pb.area - inter)
                ious[((ci, ia), (cj, ib))] = v
                if v < thr:
                    continue
                if use_pose and a.get("pose") is not None and b.get("pose") is not None and a["pose"] != b["pose"]:
                    continue
                iou_mat[ia, ib], allowed[ia, ib] = v, True
        for ia, ib in bl.assign_pairs(iou_mat, allowed, thr):
            accepted.append(((ci, ia), (cj, ib)))
    keys = [(cid, i) for cid, items in per_camera.items() for i in range(len(items))]
    clusters, fused = [], []
    if keys:
        roots = union_find_clusters(keys, accepted)
        groups = defaultdict(list)
        for k in keys:
            groups[roots[k]].append(k)
        for mem in groups.values():
            pts = np.array([per_camera[c][i]["world_xy"] for c, i in mem], dtype=np.float64)
            clusters.append(mem)
            fused.append({"world_xy": pts.mean(axis=0), "n_cams": len(mem)})
    return dict(polys=polys, ious=ious, accepted=accepted, clusters=clusters, fused=fused)


def classify_clusters(clusters, names):
    """クラスタごとの状態: correct_merge / false_merge / split / single / unnamed"""
    cow_clusters = defaultdict(set)
    for ci, mem in enumerate(clusters):
        for k in mem:
            if names.get(k):
                cow_clusters[names[k]].add(ci)
    states = []
    for ci, mem in enumerate(clusters):
        ns = {names[k] for k in mem if names.get(k)}
        if len(ns) >= 2:
            states.append("false_merge")
        elif len(ns) == 1:
            n = next(iter(ns))
            if len(cow_clusters[n]) >= 2:
                states.append("split")
            elif len(mem) >= 2:
                states.append("correct_merge")
            else:
                states.append("single")
        else:
            states.append("unnamed")
    return states


# ---------------------------------------------------------------------------
# 描画
# ---------------------------------------------------------------------------
def _poly_px(poly):
    return (np.array(poly.exterior.coords) * SCALE).astype(np.int32)


def draw_detections(canvas, det, states, names, thr):
    polys, ious, accepted = det["polys"], det["ious"], set(det["accepted"])
    key_state = {}
    for ci, mem in enumerate(det["clusters"]):
        for k in mem:
            key_state[k] = states[ci]
    for k, poly in polys.items():
        if poly is None:
            continue
        st = key_state.get(k, "single")
        cv2.polylines(canvas, [_poly_px(poly)], True, STATE_COLOR[st], 1 if st == "single" else 2, cv2.LINE_AA)
        cx, cy = poly.centroid.x * SCALE, poly.centroid.y * SCALE
        cv2.putText(canvas, f"{k[0][-1]}:{names.get(k, '?')[-3:] if names.get(k) else '?'}", (int(cx) - 14, int(cy) + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, STATE_COLOR[st], 1, cv2.LINE_AA)
    for (ka, kb), v in ious.items():
        if v < 0.05 or polys[ka] is None or polys[kb] is None:
            continue
        pa = np.array([polys[ka].centroid.x, polys[ka].centroid.y]) * SCALE
        pb = np.array([polys[kb].centroid.x, polys[kb].centroid.y]) * SCALE
        acc = (ka, kb) in accepted
        col = key_state.get(ka, "single")
        col = STATE_COLOR[col] if acc else (170, 170, 170)
        cv2.line(canvas, tuple(pa.astype(int)), tuple(pb.astype(int)), col, 2 if acc else 1, cv2.LINE_AA)
        mid = ((pa + pb) / 2).astype(int)
        cv2.putText(canvas, f"{v:.2f}", (int(mid[0]) + 3, int(mid[1]) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)


def draw_legend(canvas, y0):
    items = [(C_OK, "OK merge (same cow)"), (C_FALSE, "FALSE MERGE (different cows in one cluster)"),
             (C_SPLIT, "SPLIT (same cow in 2+ clusters = duplicate)"), (C_SINGLE, "single"), (C_NONAME, "no GT name"),
             (C_FN, "FN"), (C_FP, "FP"), (C_SW, "ID switch")]
    h = canvas.shape[0]
    y = h - 8 - 16 * len(items)
    for col, txt in items:
        cv2.rectangle(canvas, (8, y - 9), (20, y + 3), col, -1)
        cv2.putText(canvas, txt, (26, y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        y += 16


def _num(name):
    m = re.search(r"\d+", str(name))
    return int(m.group()) if m else -1


def overlay_events(acc, canvas, gt_raw, track_outputs, match_thr, info, causes):
    """motmetrics を 1 フレーム進め、FN/FP/SW を描く(render_bev_trajectory_he_kai_ver6 と同じ)。原因の推定を添える。
    info: dict(detected=GT番号の集合, false_merge=集合, split=集合, split_pts=[(x,y)], false_pts=[(x,y)])"""
    import motmetrics as mm
    by_cow = defaultdict(list)
    for it in gt_raw:
        by_cow[it["name"]].append(np.asarray(it["world_xy"], dtype=float))
    names = list(by_cow)
    gt_pts = np.array([np.mean(by_cow[n], axis=0) for n in names]) if names else np.empty((0, 2))
    ids = [_num(n) for n in names]
    tids = [int(t) for t, _ in track_outputs]
    tr_pts = np.array([np.asarray(xy, dtype=float) for _, xy in track_outputs]) if track_outputs else np.empty((0, 2))
    dists = mm.distances.norm2squared_matrix(gt_pts, tr_pts, max_d2=match_thr ** 2)
    fid = acc.update(ids, tids, dists)
    counts = dict(sw=0, fn=0, fp=0)
    try:
        ev = acc.mot_events.loc[fid]
    except KeyError:
        return counts
    name_xy, tid_xy = dict(zip(ids, gt_pts)), dict(zip(tids, tr_pts))

    def near(p, pts, r):
        return any(np.linalg.norm(np.asarray(p) - np.asarray(q)) <= r for q in pts)

    for _, row in ev.iterrows():
        t = row["Type"]
        if t == "MISS":
            o = int(row["OId"])
            cause = "誤統合" if o in info["false_merge"] else ("未検出" if o not in info["detected"] else "その他")
            causes["FN:" + cause] += 1
            counts["fn"] += 1
            px, py = (name_xy[o] * SCALE).astype(int)
            cv2.circle(canvas, (int(px), int(py)), 18, C_FN, 2, cv2.LINE_AA)
            cv2.putText(canvas, f"FN {o} ({CAUSE_EN[cause]})", (int(px) + 10, int(py) + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_FN, 1, cv2.LINE_AA)
        elif t == "FP":
            xy = tid_xy[int(row["HId"])]
            cause = "重複" if near(xy, info["split_pts"], 0.75) else "その他"
            causes["FP:" + cause] += 1
            counts["fp"] += 1
            px, py = (xy * SCALE).astype(int)
            cv2.circle(canvas, (int(px), int(py)), 14, C_FP, 2, cv2.LINE_AA)
            cv2.putText(canvas, f"FP ({CAUSE_EN[cause]})", (int(px) + 10, int(py) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_FP, 1, cv2.LINE_AA)
        elif t == "SWITCH":
            xy = tid_xy[int(row["HId"])]
            cause = "誤統合近傍" if near(xy, info["false_pts"], 1.0) else ("重複近傍" if near(xy, info["split_pts"], 1.0) else "その他")
            causes["SW:" + cause] += 1
            counts["sw"] += 1
            px, py = (xy * SCALE).astype(int)
            cv2.circle(canvas, (int(px), int(py)), 12, C_SW, 2, cv2.LINE_AA)
            cv2.putText(canvas, f"SW ({CAUSE_EN[cause]})", (int(px) + 10, int(py) + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_SW, 1, cv2.LINE_AA)
    return counts


def cluster_info(det, states, names):
    """FN/FP/SW の原因推定に使う集合・位置"""
    info = dict(detected={_num(n) for n in names.values()}, false_merge=set(), split=set(), split_pts=[], false_pts=[])
    for ci, mem in enumerate(det["clusters"]):
        if states[ci] == "false_merge":
            info["false_merge"] |= {_num(names[k]) for k in mem if names.get(k)}
            info["false_pts"].append(det["fused"][ci]["world_xy"])
        elif states[ci] == "split":
            info["split"] |= {_num(names[k]) for k in mem if names.get(k)}
            info["split_pts"].append(det["fused"][ci]["world_xy"])
    return info


# ---------------------------------------------------------------------------
def main():
    import motmetrics as mm
    from ultralytics import YOLO
    arg = (sys.argv[1:] or ["iou@0.2"])[0]
    thr = float(arg.split("@", 1)[1]) if arg.startswith("iou@") else (0.2 if arg == "iou" else None)
    if thr is None:
        raise SystemExit("引数は iou か iou@<しきい値> (例: iou@0.2)")
    name = f"iou_t{thr:.2f}"

    import render_bev_trajectory_he_kai_ver6 as rv
    rf, ym = rv.rf, rv.ym
    from Mootrack_core import CAM_IDS, ROT_FMT, apply_camera_overrides, filter_outside_floorplan_bounds, get_intrinsics, load_floorplan, resolve_poses
    from curved_zone_utils_ver1 import N_ANGLES, ZONE_HEIGHT_M, build_camera_polygons, filter_by_curved_zone, get_adjacent_pairs

    if "HE_POSE_CONSTRAINT" in os.environ:
        rf.USE_POSE_CONSTRAINT = os.environ["HE_POSE_CONSTRAINT"] == "1"
    print(f"姿勢別統合制約: {'ON' if rf.USE_POSE_CONSTRAINT else 'OFF'} / B: IoU しきい値 {thr} / フレーム {rv.FRAME_START}〜{rv.FRAME_END} 刻み {rv.FRAME_STEP}"
          f" / 追跡={'ReID' if rv.USE_REID else 'Simple'} 追跡閾値={rv.TRACK_THR_M:g}m 対応づけゲート={rv.MATCH_THR_M:g}m")

    import gt_ignore_ver1 as gi
    gt_ignore_state = gi.apply(ym)

    regions, cameras_json, real_w, real_h = load_floorplan(rf.FLOORPLAN_JSON_PATH)
    poses = resolve_poses(apply_camera_overrides(cameras_json), root="Area1", fmt=ROT_FMT)
    bl.hk._YM = ym
    bl.hk.setup_poses(poses)
    bl.MAP_WH = (real_w, real_h)                              # 論文方式の矩形を牛舎の範囲で切り取る
    print(f"B の方式: {bl.BASE_STYLE}")
    orig_heights = (rf.STANDING_HEIGHT_M, rf.LYING_HEIGHT_M)
    rv.set_heights("orig", orig_heights, None)               # B は phase3 の想定高さ

    polygons = build_camera_polygons(poses, CAM_IDS, get_intrinsics, height_m=ZONE_HEIGHT_M, n_angles=N_ANGLES)   # HE のみの円
    adjacent_pairs = get_adjacent_pairs(CAM_IDS)
    yolo_model = YOLO(rf.YOLO_WEIGHTS_PATH)
    xml_dir, image_dir = Path(rf.XML_DIR), Path(rf.IMAGE_DIR)
    frame_indices = list(range(rv.FRAME_START, rv.FRAME_END + 1, rv.FRAME_STEP))
    exclude = getattr(ym, "EXCLUDE_GT_NAMES", set())

    out_dir = OUT_ROOT / name / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    tracker = rv.make_tracker(real_w, real_h)
    history = defaultdict(lambda: deque(maxlen=rv.TRAIL_LENGTH))
    acc = mm.MOTAccumulator(auto_id=True)
    causes, state_counts = Counter(), Counter()
    saved, err_flags = [], []

    for frame_idx in frame_indices:
        xml_path = xml_dir / f"{rf.frame_name(frame_idx)}.xml"
        image_path = image_dir / f"{rf.frame_name(frame_idx)}.jpg"
        if not image_path.exists():
            continue
        has_gt = xml_path.exists()
        gt_objects = rf.parse_voc_annotation(xml_path) if has_gt else []

        raw = rf.yolo_world_points_per_camera_with_pose(yolo_model, image_path, poses)
        per_camera = {cid: [dict(it) for it in items] for cid, items in raw.items()}
        deduped = rf.deduplicate_per_camera(per_camera, rf.DEDUP_IOU_THRESHOLD)
        filtered = filter_by_curved_zone(deduped, polygons)
        filtered = filter_outside_floorplan_bounds(filtered, real_w, real_h, margin_m=0.0, verbose=False)

        det = fuse_iou_detail(filtered, adjacent_pairs, thr, rf.USE_POSE_CONSTRAINT, ym.height_for)
        names = detection_names(gt_objects, filtered, CAM_IDS, rf.IMAGE_WIDTH_PX, exclude) if has_gt else {}
        states = classify_clusters(det["clusters"], names)
        for s in states:
            state_counts[s] += 1
        info = cluster_info(det, states, names)

        gt_raw = []
        if has_gt:
            gt_per_cam = rf.gt_world_points_per_camera(gt_objects, poses, rf.IMAGE_WIDTH_PX)
            gt_raw = [{"name": it["name"], "cam_id": c, "world_xy": it["world_xy"]} for c, items in gt_per_cam.items() for it in items]

        track_outputs = tracker.update(det["fused"])
        for tid, xy in track_outputs:
            history[tid].append((float(xy[0]), float(xy[1])))
        canvas = rv.draw_frame_trajectory(frame_idx, regions, poses, polygons, real_w, real_h, gt_raw, track_outputs, history, has_gt)
        draw_detections(canvas, det, states, names, thr)
        counts = overlay_events(acc, canvas, gt_raw, track_outputs, rv.MATCH_THR_M, info, causes) if has_gt else dict(sw=0, fn=0, fp=0)
        n_fm, n_sp = states.count("false_merge"), states.count("split")
        cv2.putText(canvas, f"[B: IoU>={thr:g}]", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"false-merge={n_fm} split(dup)={n_sp} | FN={counts['fn']} FP={counts['fp']} SW={counts['sw']}", (8, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        draw_legend(canvas, 0)
        path = out_dir / f"{rf.frame_name(frame_idx)}.png"
        cv2.imwrite(str(path), canvas)
        saved.append(path)
        err_flags.append(bool(n_fm or n_sp or counts["fn"] or counts["fp"] or counts["sw"]))
        if len(saved) % 20 == 0:
            print(f"  {len(saved)}/{len(frame_indices)} 枚処理済み (直近 {rf.frame_name(frame_idx)})")

    print(f"\n{len(saved)}枚を保存: {out_dir}")
    print("\n■ クラスタの状態(全フレームの合計)")
    for s in ("correct_merge", "false_merge", "split", "single", "unnamed"):
        print(f"  {STATE_JA[s]:<14}{state_counts[s]:>7}")
    print("\n■ 誤りの原因の推定(全フレームの合計)")
    for k in sorted(causes):
        print(f"  {k:<14}{causes[k]:>6}")
    try:
        summary = mm.metrics.create().compute(acc, metrics=rv.METRIC_KEYS, name=name)
        print("\n■ 評価と同じ指標 (動画のフレームだけ)")
        print(summary.round(4).to_string())
    except Exception as e:
        print(f"(指標の表の計算に失敗: {e})")
    gi.report(gt_ignore_state)

    sel = saved
    if ERR_ONLY:
        keep = set()
        for i, f in enumerate(err_flags):
            if f:
                keep.update(range(max(0, i - CONTEXT), min(len(saved), i + CONTEXT + 1)))
        sel = [saved[i] for i in sorted(keep)]
        print(f"誤りのあったフレームと前後 {CONTEXT} フレーム: {len(sel)}/{len(saved)} 枚を動画にします")
    if sel:
        rf.make_video_from_frames(out_dir, OUT_ROOT / name / f"bev_baseline_{name}.mp4", rv.VIDEO_FPS, sel)


if __name__ == "__main__":
    main()