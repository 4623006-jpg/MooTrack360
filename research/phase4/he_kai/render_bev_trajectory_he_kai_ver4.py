"""
render_bev_trajectory_he_kai_ver4.py

ver4: 評価(phase3_he_kai_matching_ver2.py)と同じ設定で動画を作る。
  phase3 が使う YOLOモジュール(既定 phase3_bev_multicam_yolo_check_zonepose_ver6)を読み込み、
  想定高さ・融合/追跡の距離閾値・姿勢別統合制約・ReIDの有無・フレーム範囲・YOLO/校正の設定を、そこから取る。
  (ver3まで: render_bev_frames_yolo_zonepose_ver1 と render_bev_trajectory_yolo_zonepose_ver1 の値を使っていたため、
   評価と閾値や姿勢制約が違っていた)
  HE_POSE_CONSTRAINT / HE_USE_REID / HE_TRACK_THR_M / HE_MATCH_THR_M / HE_FRAME_* は指定すれば上書きできる。
  起動時に「評価と合わせた設定」を表示するので、phase3の実行時の表示と見比べること。

BEVの追跡動画(軌跡つき)に、そのフレームのFN/FPも重ねて表示する(ver1からの追加)。
  赤い輪 FN : 対応するtrackが無いGT(牛番号つき)
  マゼンタの輪 FP : 対応するGTが無いtrack
  左上 [モード] の下に TP/FN/FP の数。対応付けはフレームごとのハンガリアン法(距離 HE_MATCH_THR_M 以内、既定3.0m = phase3 ver6 の評価ゲート 2×DIST_THR_TRACK_M(1.5))。
  motmetrics は前フレームの対応も引き継ぐため、数字は評価と少しずれることがある(目安)。
  GT除外リスト(gt_ignore_ver1.py, HE_GT_IGNORE=0 で無効)も phase3 と同じく適用する。

BEVの追跡動画(軌跡つき)を、phase3_he_kai_matching_ver2.py のモードごとに作る。
複数モードを指定すると、同じフレームを左右に並べた比較動画になる(同じYOLO検出・同じ校正)。

  モード(phase3_he_kai_matching_ver2.py の MODES と同じ):
    euclid        既存(ユークリッド距離, 想定高さは phase3 の値, 曲線ゾーンは既存)
    euclid_h      euclid + 想定高さ h_eff
    maha / maha_h マハラノビス
    maha_h_valid  マハラノビス + h_eff + HE(改)の有効範囲(多角形の輪郭も描く)
  既定: maha_h_valid だけ。比較したいときは 引数に euclid maha_h_valid のように並べる。

描画・追跡は render_bev_trajectory_yolo_zonepose_ver1.py と同じ(軌跡の尾、IDの色、GTの十字)。
追跡は SimpleBevTracker(ReIDなし)。ReIDあり: HE_USE_REID=1

使い方:
    python render_bev_trajectory_he_kai_ver4.py                     # maha_h_valid だけ
    python render_bev_trajectory_he_kai_ver4.py euclid maha_h_valid # 左右に並べた比較動画
環境変数:
    HE_FRAME_START / HE_FRAME_END / HE_FRAME_STEP : 描画するフレーム (既定 0 / 290 / 10)
    HE_SIGMA_JSON / HE_SIGMA_MODEL / HE_EPS_M / HE_K_SIGMA / HE_GATE_CHI2 : phase3_he_kai_matching_ver2.py と同じ
出力: <OUTPUT_DIR>/bev_trajectory_frames_he_kai_<モード>/{frame}.png と bev_trajectory_he_kai_<モード>.mp4
配置: research/phase4/he_kai/ など (render_bev_*_ver1.py と curved_zone_utils_ver1.py が import できること)
"""

import os
import sys
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESEARCH_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
PHASE4_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
MOOTRACK_CODE_DIR = (os.path.abspath(os.path.join(RESEARCH_DIR, "calibration")))
for d in (RESEARCH_DIR, PHASE4_DIR, os.path.join(PHASE4_DIR, "calibration"),
          os.path.join(PHASE4_DIR, "calibration_try"), os.path.join(PHASE4_DIR, "he_kai"),
          MOOTRACK_CODE_DIR,
          os.path.join(RESEARCH_DIR, "analyze"), os.path.join(RESEARCH_DIR, "annotate"),
          os.path.join(RESEARCH_DIR, "phase3_multi_camera"), THIS_DIR):
    if os.path.isdir(d) and d not in sys.path:
        sys.path.insert(0, d)

import render_bev_frames_yolo_zonepose_ver1 as _rf_old
import phase3_he_kai_matching_ver2 as hk
from scipy.optimize import linear_sum_assignment
from curved_zone_utils_ver1 import N_ANGLES, ZONE_HEIGHT_M, build_camera_polygons, filter_by_curved_zone, get_adjacent_pairs
from he_kai_sigma_model_ver1 import build_valid_polygons, load_model
from Mootrack_core import (
    CAM_IDS, ROT_FMT, ReIDBevTracker, apply_camera_overrides, filter_outside_floorplan_bounds,
    get_intrinsics, load_floorplan, resolve_poses,
)
from render_bev_trajectory_yolo_zonepose_ver1 import (
    DIST_THR_TRACK_M, EDGE_MARGIN_M, LOST_MAX_AGE, MAX_MISSED_FRAMES, REID_DIST_THR_M,
    TRAIL_LENGTH, SimpleBevTracker, draw_frame_trajectory,
)

class _Proxy:
    """phase3 のYOLOモジュール(ym)を優先して属性を読む。無い属性は旧 rf から読む。代入は ym に入る"""

    def __init__(self, primary, fallback):
        object.__setattr__(self, "_p", primary)
        object.__setattr__(self, "_f", fallback)

    def __getattr__(self, name):
        try:
            return getattr(self._p, name)
        except AttributeError:
            return getattr(self._f, name)

    def __setattr__(self, name, value):
        setattr(self._p, name, value)


ym = hk.load_yolo_module()                 # phase3_he_kai_matching_ver2.py と同じモジュール(ver6)
rf = _Proxy(ym, _rf_old)


def _ym_attr(names, default):
    for n in names:
        if hasattr(ym, n):
            return getattr(ym, n), n
    return default, None


DEFAULT_MODES = ["maha_h_valid"]      # 比較したいときは引数で: ... euclid maha_h_valid
FRAME_START = int(os.environ.get("HE_FRAME_START", getattr(ym, "FRAME_START", 0)))
FRAME_END = int(os.environ.get("HE_FRAME_END", getattr(ym, "FRAME_END", 290)))
FRAME_STEP = int(os.environ.get("HE_FRAME_STEP", getattr(ym, "FRAME_STEP", 10)))
_reid_default, _reid_name = _ym_attr(("USE_REID", "USE_REID_TRACKER", "ENABLE_REID", "USE_REID_TRACKING"), None)
if "HE_USE_REID" in os.environ:
    USE_REID = os.environ["HE_USE_REID"] == "1"
elif _reid_default is not None:
    USE_REID = bool(_reid_default)
else:
    USE_REID = True            # 評価の出力名に reid が付いているので、ym側に設定が無いときは ReID あり
VIDEO_FPS = 2
TRACK_THR_M = float(os.environ.get("HE_TRACK_THR_M", getattr(ym, "DIST_THR_TRACK_M", DIST_THR_TRACK_M)))   # 追跡の距離閾値
MATCH_THR_M = float(os.environ.get("HE_MATCH_THR_M", 2.0 * TRACK_THR_M))   # 評価の対応付けゲート: max_d2=(2*DIST_THR_TRACK_M)^2
MAX_MISSED = getattr(ym, "MAX_MISSED_FRAMES", MAX_MISSED_FRAMES)
EDGE_MARGIN = getattr(ym, "EDGE_MARGIN_M", EDGE_MARGIN_M)
REID_THR = getattr(ym, "REID_DIST_THR_M", REID_DIST_THR_M)
LOST_AGE = getattr(ym, "LOST_MAX_AGE", LOST_MAX_AGE)
SCALE = 40            # draw_frame_trajectory と同じ
FN_COLOR, FP_COLOR = (0, 0, 255), (255, 0, 255)


def overlay_fn_fp(canvas, gt_raw, track_outputs):
    """GT(牛ごとにカメラ平均)とtrackをハンガリアン法で対応付け、FN/FPを輪で描く。(TP,FN,FP)を返す"""
    by_cow = defaultdict(list)
    for it in gt_raw:
        by_cow[it["name"]].append(np.asarray(it["world_xy"], dtype=float))
    names = list(by_cow)
    gt_pts = np.array([np.mean(by_cow[n], axis=0) for n in names]) if names else np.empty((0, 2))
    tr_pts = np.array([np.asarray(xy, dtype=float) for _, xy in track_outputs]) if track_outputs else np.empty((0, 2))
    mg, mt = set(), set()
    if len(gt_pts) and len(tr_pts):
        d = np.linalg.norm(gt_pts[:, None, :] - tr_pts[None, :, :], axis=-1)
        cost = np.where(d <= MATCH_THR_M, d, 1e6)
        for i, j in zip(*linear_sum_assignment(cost)):
            if d[i, j] <= MATCH_THR_M:
                mg.add(i)
                mt.add(j)
    for j, (tid, _) in enumerate(track_outputs):
        if j not in mt:
            px, py = int(tr_pts[j][0] * SCALE), int(tr_pts[j][1] * SCALE)
            cv2.circle(canvas, (px, py), 14, FP_COLOR, 2, cv2.LINE_AA)
            cv2.putText(canvas, "FP", (px + 10, py + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, FP_COLOR, 1, cv2.LINE_AA)
    for i, n in enumerate(names):
        if i not in mg:
            px, py = int(gt_pts[i][0] * SCALE), int(gt_pts[i][1] * SCALE)
            cv2.circle(canvas, (px, py), 18, FN_COLOR, 2, cv2.LINE_AA)
            cv2.putText(canvas, "FN " + n.replace("cow", ""), (px + 10, py + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, FN_COLOR, 1, cv2.LINE_AA)
    n_tp = len(mg)
    return n_tp, len(names) - n_tp, len(track_outputs) - len(mt)


def set_heights(kind, orig, model):
    if kind == "fit":
        rf.STANDING_HEIGHT_M, rf.LYING_HEIGHT_M = model.h_eff("standing"), model.h_eff("lying")
    else:
        rf.STANDING_HEIGHT_M, rf.LYING_HEIGHT_M = orig


def make_tracker(real_w, real_h):
    if USE_REID:
        return ReIDBevTracker(dist_thr=TRACK_THR_M, max_missed=MAX_MISSED, real_w=real_w, real_h=real_h,
                              edge_margin_m=EDGE_MARGIN, reid_dist_thr=REID_THR, lost_max_age=LOST_AGE)
    return SimpleBevTracker(dist_thr=TRACK_THR_M, max_missed=MAX_MISSED)


def main():
    from ultralytics import YOLO

    modes = sys.argv[1:] or DEFAULT_MODES
    bad = [m for m in modes if m not in hk.MODES]
    if bad:
        raise SystemExit(f"不明なモード: {bad} / 使えるモード: {list(hk.MODES)}")
    tag = "_".join(modes)
    if "HE_POSE_CONSTRAINT" in os.environ:                 # 姿勢別の統合制約 1=ON / 0=OFF
        rf.USE_POSE_CONSTRAINT = os.environ["HE_POSE_CONSTRAINT"] == "1"
    print(f"姿勢別統合制約: {'ON' if rf.USE_POSE_CONSTRAINT else 'OFF'}")
    print(f"評価と合わせた設定: YOLOモジュール={ym.__name__} / フレーム {FRAME_START}〜{FRAME_END} 刻み {FRAME_STEP} / "
          f"追跡={'ReID' if USE_REID else 'Simple'}" + (f"(ym.{_reid_name})" if _reid_name else "(ym側に設定名なし)")
          + f" / 追跡閾値={TRACK_THR_M:g}m 対応付けゲート={MATCH_THR_M:g}m / 融合閾値(euclid)={rf.DIST_THR_FUSION_M:g}m / "
          f"想定高さ(orig) standing={rf.STANDING_HEIGHT_M} lying={rf.LYING_HEIGHT_M} / "
          f"ε={hk.EPS_VALID_M} k={hk.K_SIGMA} gate={hk.GATE_CHI2}")

    import gt_ignore_ver1 as gi
    gt_ignore_state = gi.apply(ym)          # GT除外リスト(phase3_he_kai_matching_ver2.py と同じ)

    regions, cameras_json, real_w, real_h = load_floorplan(rf.FLOORPLAN_JSON_PATH)
    poses = resolve_poses(apply_camera_overrides(cameras_json), root="Area1", fmt=ROT_FMT)

    json_path = os.environ.get("HE_SIGMA_JSON") or str(Path(rf.OUTPUT_DIR) / "bev_calibration_refined" / "sigma_he_kai_ver1.json")
    model = load_model(json_path)
    print(model.describe())
    hk._MODEL, hk._YM = model, ym
    hk.setup_poses(poses)
    orig_heights = (rf.STANDING_HEIGHT_M, rf.LYING_HEIGHT_M)

    print(f"\n各モードの有効範囲(曲線ゾーン)を計算中... ε={hk.EPS_VALID_M}m k={hk.K_SIGMA}")
    polygons = {}
    for m in modes:
        if hk.MODES[m]["polygons"] == "he_kai":
            print(f"[{m}] HE(改)の有効範囲")
            polygons[m] = build_valid_polygons(poses, CAM_IDS, get_intrinsics, model, hk.EPS_VALID_M, hk.K_SIGMA,
                                               height_m=ZONE_HEIGHT_M, n_angles=N_ANGLES)
        else:
            print(f"[{m}] 既存の曲線ゾーン")
            polygons[m] = build_camera_polygons(poses, CAM_IDS, get_intrinsics, height_m=ZONE_HEIGHT_M, n_angles=N_ANGLES)
    adjacent_pairs = get_adjacent_pairs(CAM_IDS)

    print(f"\nYOLOモデル読み込み中: {rf.YOLO_WEIGHTS_PATH}")
    yolo_model = YOLO(rf.YOLO_WEIGHTS_PATH)
    xml_dir, image_dir = Path(rf.XML_DIR), Path(rf.IMAGE_DIR)
    frame_indices = list(range(FRAME_START, FRAME_END + 1, FRAME_STEP))
    print(f"対象フレーム: {FRAME_START}〜{FRAME_END} 刻み {FRAME_STEP} ({len(frame_indices)}枚) / モード {modes}")

    trackers = {m: make_tracker(real_w, real_h) for m in modes}
    histories = {m: defaultdict(lambda: deque(maxlen=TRAIL_LENGTH)) for m in modes}
    out_dir = Path(rf.OUTPUT_DIR) / f"bev_trajectory_frames_he_kai_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    tot = {m: [0, 0, 0] for m in modes}

    for frame_idx in frame_indices:
        xml_path = xml_dir / f"{rf.frame_name(frame_idx)}.xml"
        image_path = image_dir / f"{rf.frame_name(frame_idx)}.jpg"
        if not image_path.exists():
            continue
        has_gt = xml_path.exists()
        gt_objects = rf.parse_voc_annotation(xml_path) if has_gt else []
        raw_cache, panels = {}, []

        for m in modes:
            cfg = hk.MODES[m]
            set_heights(cfg["heights"], orig_heights, model)
            if cfg["heights"] not in raw_cache:
                raw_cache[cfg["heights"]] = rf.yolo_world_points_per_camera_with_pose(yolo_model, image_path, poses)
            per_camera_raw = {cid: [dict(it) for it in items] for cid, items in raw_cache[cfg["heights"]].items()}
            deduped = rf.deduplicate_per_camera(per_camera_raw, rf.DEDUP_IOU_THRESHOLD)
            filtered = filter_by_curved_zone(deduped, polygons[m])
            filtered = filter_outside_floorplan_bounds(filtered, real_w, real_h, margin_m=0.0, verbose=False)
            if cfg["match"] == "maha":
                fused = hk.fuse_mahalanobis(filtered, adjacent_pairs, rf.DIST_THR_FUSION_M, rf.USE_POSE_CONSTRAINT)[0]
            else:
                fused = rf.fuse_frame_zone_pose(filtered, adjacent_pairs, rf.DIST_THR_FUSION_M, rf.USE_POSE_CONSTRAINT)

            gt_raw = []
            if has_gt:
                gt_per_cam = filter_by_curved_zone(rf.gt_world_points_per_camera(gt_objects, poses, rf.IMAGE_WIDTH_PX), polygons[m])
                gt_raw = [{"name": it["name"], "cam_id": c, "world_xy": it["world_xy"]} for c, items in gt_per_cam.items() for it in items]

            track_outputs = trackers[m].update(fused)
            for tid, xy in track_outputs:
                histories[m][tid].append((float(xy[0]), float(xy[1])))
            canvas = draw_frame_trajectory(frame_idx, regions, poses, polygons[m], real_w, real_h,
                                           gt_raw, track_outputs, histories[m], has_gt)
            n_tp, n_fn, n_fp = overlay_fn_fp(canvas, gt_raw, track_outputs) if has_gt else (0, 0, 0)
            cv2.putText(canvas, f"[{m}]", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            if has_gt:
                cv2.putText(canvas, f"TP={n_tp} FN={n_fn} FP={n_fp} (<= {MATCH_THR_M:g}m)", (8, 46),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                tot[m][0] += n_tp
                tot[m][1] += n_fn
                tot[m][2] += n_fp
            panels.append(canvas)
        set_heights("orig", orig_heights, model)

        out = panels[0] if len(panels) == 1 else np.hstack(panels)
        if out.shape[1] % 2:
            out = np.pad(out, ((0, 0), (0, 1), (0, 0)))
        path = out_dir / f"{rf.frame_name(frame_idx)}.png"
        cv2.imwrite(str(path), out)
        saved.append(path)
        if len(saved) % 20 == 0:
            print(f"  {len(saved)}/{len(frame_indices)} 枚処理済み (直近 {rf.frame_name(frame_idx)})")

    for m in modes:
        print(f"[{m}] 描画フレーム合計 TP={tot[m][0]} FN={tot[m][1]} FP={tot[m][2]} (簡易対応付け, {MATCH_THR_M:g}m)")
    gi.report(gt_ignore_state)
    print(f"\n{len(saved)}枚を保存: {out_dir}")
    if saved:
        video_path = Path(rf.OUTPUT_DIR) / f"bev_trajectory_he_kai_{tag}.mp4"
        rf.make_video_from_frames(out_dir, video_path, VIDEO_FPS, saved)


if __name__ == "__main__":
    main()