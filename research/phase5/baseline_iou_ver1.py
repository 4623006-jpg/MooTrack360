"""
baseline_iou_ver1.py  (配置: research/phase5/)

従来手法(通常カメラの複数カメラ個体統合の代表)のベースライン B:
    各カメラの検出 bbox を、想定高さの水平面(床面)に変形(逆投影)して多角形にし、
    カメラ間の多角形の IoU >= しきい値 のペアを、ハンガリアン法で同一個体として統合する。
    カメラの有効範囲は HE(誤差のみ)の既存の円(polygons="orig", 高さ項なし)、想定高さは phase3 の値(heights="orig")。
    ※ 画像全体をパノラマにして IoU を取るのと同じ考え方(床面を共通の面にして、bbox の重なりで見る)。

提案手法(maha_h_valid)や、アブレーション(euclid など)と同じ土台(同じYOLO検出・同じ校正・同じ評価)で動かす。
phase4/he_kai/phase3_he_kai_matching_ver10.py の run_mode をそのまま使い、融合関数だけを IoU 版に差し替える。

YOLO の phase3 モジュール(phase3_bev_multicam_yolo_check_zonepose_ver*.py)が sys.path に無いとき、
      research/ 以下を検索して、見つかったフォルダを自動で追加する(ModuleNotFoundError 対策)。

使い方(research/ で):
    uv run python ./phase5/baseline_iou_ver1.py                 # IoU しきい値を IOU_THRS の全部で掃引
    uv run python ./phase5/baseline_iou_ver1.py iou@0.1         # しきい値 0.1 だけ
    uv run python ./phase5/baseline_iou_ver1.py iou euclid      # 掃引 + 既存モード(euclid 等)も同時に
環境変数:
    IOU_THRS          : しきい値のカンマ区切り (既定 "0.05,0.1,0.2,0.3")
    IOU_EDGE_SAMPLES  : bbox の各辺を何点で床に投影するか (既定 8)
    BASE_HEIGHTS      : orig(phase3の想定高さ, 既定) / fit(h_eff)
    BASE_POLYGONS     : orig(HEのみの円, 既定) / he_kai(HE(改)の有効範囲)
    HE_POSE_CONSTRAINT: 姿勢別統合制約 1/0 (未指定は phase3 モジュールの値。提案手法と揃えること)
    HE_FRAME_START / HE_FRAME_END / HE_FRAME_STEP, HE_SIGMA_JSON, HE_EPS_M なども phase3_he_kai_matching_ver10 と同じ
出力: research/phase5/baseline_out/iou_ver1/<モード名>/ (各モードのCSV。events_<モード>.csv も)
"""

import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESEARCH_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
PHASE4_DIR = os.path.join(RESEARCH_DIR, "phase4")
MOOTRACK_CODE_DIR = os.path.abspath(os.path.join(RESEARCH_DIR, "calibration"))
for d in (RESEARCH_DIR, PHASE4_DIR, os.path.join(PHASE4_DIR, "web_annotate"), os.path.join(PHASE4_DIR, "calibration"),
          os.path.join(PHASE4_DIR, "calibration_try"), os.path.join(PHASE4_DIR, "he_kai"),
          MOOTRACK_CODE_DIR,
          os.path.join(RESEARCH_DIR, "analyze"), os.path.join(RESEARCH_DIR, "annotate"),
          os.path.join(RESEARCH_DIR, "phase3_multi_camera"), THIS_DIR):
    if os.path.isdir(d) and d not in sys.path:
        sys.path.insert(0, d)


def _ensure_yolo_module_on_path():
    import glob
    for pat in ("phase3_bev_multicam_yolo_check_zonepose_ver*.py",):
        hits = glob.glob(os.path.join(RESEARCH_DIR, "**", pat), recursive=True)
        hits = [h for h in hits if "__pycache__" not in h]
        for h in sorted(hits):
            dd = os.path.dirname(h)
            if dd not in sys.path:
                sys.path.insert(0, dd)
                print(f"YOLOモジュールのフォルダを追加: {dd}")
        if hits:
            return
    print("!! phase3_bev_multicam_yolo_check_zonepose_ver*.py が research/ 以下に見つかりません")


def _ensure_core_on_path():
    """Mootrack_core.py が sys.path に無いとき、使うフォルダを探して追加する。
    優先順: 環境変数 MOOTRACK_CODE_DIR -> research/calibration -> calibration_process/trash/calibration
            -> calibration_process 以下を検索して、フォルダ名が calibration のもの(make_bev は避ける)。どれを使ったか表示する"""
    import glob
    try:
        import Mootrack_core  # noqa: F401
        print(f"Mootrack_core: {Mootrack_core.__file__}")
        return
    except ModuleNotFoundError:
        pass
    parent = os.path.abspath(os.path.join(RESEARCH_DIR, ".."))
    cands = [os.environ.get("MOOTRACK_CODE_DIR", ""), os.path.join(RESEARCH_DIR, "calibration"),
             os.path.join(parent, "trash", "calibration")]
    cands += [os.path.dirname(h) for h in glob.glob(os.path.join(parent, "**", "Mootrack_core.py"), recursive=True)
              if os.path.basename(os.path.dirname(h)) == "calibration"]
    for d in cands:
        if d and os.path.isfile(os.path.join(d, "Mootrack_core.py")):
            sys.path.insert(0, d)
            print(f"Mootrack_core を追加: {os.path.join(d, 'Mootrack_core.py')}")
            return
    print(f"!! Mootrack_core.py が見つかりません。環境変数 MOOTRACK_CODE_DIR にフォルダを指定してください (探した場所: {[c for c in cands if c]})")


_ensure_core_on_path()

_ensure_yolo_module_on_path()
import phase3_he_kai_matching_ver10 as hk
from shapely.geometry import MultiPoint, Polygon, box as shp_box

IOU_THRS = [float(x) for x in os.environ.get("IOU_THRS", "0.05,0.1,0.2,0.3").split(",") if x.strip()]
EDGE_SAMPLES = int(os.environ.get("IOU_EDGE_SAMPLES", 8))
BASE_HEIGHTS = os.environ.get("BASE_HEIGHTS", "orig")
BASE_POLYGONS = os.environ.get("BASE_POLYGONS", "orig")
OUT_ROOT = Path(os.environ.get("BASE_OUT_DIR", os.path.join(THIS_DIR, "baseline_out", "iou_ver1")))

# 山本先生の論文の方式(BASE_STYLE=paper, 既定)か、こちらで作った方式(ours)か
#   paper: bbox の4頂点を想定高さの水平面に射影し、x と y をそれぞれ小さい順に並べて x1'..x4', y1'..y4' とし、
#          内接矩形 (x2',y2',x3',y3') と外接矩形 (x1',y1',x4',y4') の平均 ((x1'+x2')/2,(y1'+y2')/2,(x3'+x4')/2,(y3'+y4')/2) を新しい矩形にする(論文の式8〜10)。
#          4頂点のどれかが射影できなければ除く。牛舎の範囲(MAP_WH)で切り取り、幅・高さが 0 なら除く。
#          BASE_WMAX_M / BASE_HMAX_M を指定すると、幅・高さがそれを超える矩形を除く(論文の W_max, H_max。既定は無効)。
#          割り当ては、重み IoU - τ を最大化する(重みが負のペアは使わない。論文の式12)。
#   ours : bbox の各辺を EDGE_SAMPLES 点ずつ射影した多角形。割り当ては 1-IoU の合計を最小化(使えるペアの数を優先)。
BASE_STYLE = os.environ.get("BASE_STYLE", "paper")
WMAX_M = float(os.environ["BASE_WMAX_M"]) if os.environ.get("BASE_WMAX_M") else None
HMAX_M = float(os.environ["BASE_HMAX_M"]) if os.environ.get("BASE_HMAX_M") else None
MAP_WH = None                           # (牛舎の幅, 高さ)[m]。main で設定する
_IOU_THR = 0.1
_IOU_STATS = {"overlap": 0, "pass": 0, "no_poly": 0, "dets": 0}
_IOU_SUMMARY = {}


def det_polygon(cam_id, item, height_m, n_side=EDGE_SAMPLES):
    """bbox の境界を n_side 点/辺でサンプリングし、想定高さ height_m の水平面に逆投影して多角形にする。
    全点が有効なら境界の順に並べた多角形、一部が無効(水平線より上など)なら有効な点の凸包。有効点が3未満なら None"""
    from Mootrack_core import get_intrinsics, unproject_to_floor
    x1, y1, x2, y2 = item["bbox_local"]
    if BASE_STYLE == "paper":
        cp = hk._POSES[cam_id]
        fx, fy, cx, cy, xi, alpha = get_intrinsics(cam_id)
        corners = np.array([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], dtype=np.float64)
        xy, ok = unproject_to_floor(corners, cp.R, cp.t, fx, fy, cx, cy, xi, alpha, floor_z=-height_m)
        if not ok.all():
            return None
        xs, ys = np.sort(xy[:, 0]), np.sort(xy[:, 1])
        xa, xb = (xs[0] + xs[1]) / 2.0, (xs[2] + xs[3]) / 2.0          # 内接と外接の平均(論文の式10)
        ya, yb = (ys[0] + ys[1]) / 2.0, (ys[2] + ys[3]) / 2.0
        if MAP_WH is not None:                                          # 牛舎の外にはみ出さないよう切り取る(式11)
            xa, ya, xb, yb = max(xa, 0.0), max(ya, 0.0), min(xb, MAP_WH[0]), min(yb, MAP_WH[1])
        w, h = xb - xa, yb - ya
        if w <= 1e-9 or h <= 1e-9:
            return None
        if (WMAX_M is not None and w > WMAX_M) or (HMAX_M is not None and h > HMAX_M):
            return None
        return shp_box(xa, ya, xb, yb)
    s = np.linspace(0.0, 1.0, n_side, endpoint=False)
    pts = np.array([(x1 + (x2 - x1) * t, y1) for t in s] + [(x2, y1 + (y2 - y1) * t) for t in s]
                   + [(x2 - (x2 - x1) * t, y2) for t in s] + [(x1, y2 - (y2 - y1) * t) for t in s], dtype=np.float64)
    cp = hk._POSES[cam_id]
    fx, fy, cx, cy, xi, alpha = get_intrinsics(cam_id)
    xy, ok = unproject_to_floor(pts, cp.R, cp.t, fx, fy, cx, cy, xi, alpha, floor_z=-height_m)
    if ok.sum() < 3:
        return None
    if ok.all():
        poly = Polygon(xy)
        if not poly.is_valid:
            poly = poly.buffer(0)
    else:
        poly = MultiPoint(xy[ok]).convex_hull
    if poly.is_empty or poly.geom_type != "Polygon" or poly.area < 1e-9:
        return None
    return poly


def assign_pairs(iou_mat, allowed, thr):
    """IoU 行列と、候補にできるペア(IoU>=しきい値・姿勢制約OK)から、1対1の割り当て [(行, 列)] を返す"""
    if BASE_STYLE == "paper":
        w = np.where(allowed, iou_mat - thr, 0.0)                       # 論文の式12: 重み = IoU - τ。候補でないものは 0(使わない)
        r, c = linear_sum_assignment(w, maximize=True)
        return [(a, b) for a, b in zip(r, c) if allowed[a, b] and w[a, b] > 0]
    cost = np.where(allowed, 1.0 - iou_mat, 1e6)
    r, c = linear_sum_assignment(cost)
    return [(a, b) for a, b in zip(r, c) if cost[a, b] < 1e5]


def fuse_iou(per_camera_filtered, adjacent_pairs, dist_thr=None, use_pose_constraint=True, *args, **kwargs):
    """ym.fuse_frame_zone_pose の置き換え(hk.fuse_mahalanobis と同じ戻り値 (fused, n_pose_rejected))"""
    from Mootrack_core import union_find_clusters
    ym = hk._YM
    polys = {}
    for cid, items in per_camera_filtered.items():
        for idx, it in enumerate(items):
            polys[(cid, idx)] = det_polygon(cid, it, ym.height_for(it.get("pose")))
            _IOU_STATS["dets"] += 1
            if polys[(cid, idx)] is None:
                _IOU_STATS["no_poly"] += 1
    accepted, n_rej = [], 0
    for ci, cj in adjacent_pairs:
        A, B = per_camera_filtered[ci], per_camera_filtered[cj]
        if not A or not B:
            continue
        iou_mat = np.zeros((len(A), len(B)))
        allowed = np.zeros((len(A), len(B)), dtype=bool)
        for ia, a_item in enumerate(A):
            for ib, b_item in enumerate(B):
                pa, pb = polys[(ci, ia)], polys[(cj, ib)]
                if pa is None or pb is None:
                    continue
                inter = pa.intersection(pb).area
                if inter <= 0:
                    continue
                _IOU_STATS["overlap"] += 1
                iou = inter / (pa.area + pb.area - inter)
                if iou < _IOU_THR:
                    continue
                if use_pose_constraint and a_item["pose"] is not None and b_item["pose"] is not None \
                        and a_item["pose"] != b_item["pose"]:
                    n_rej += 1
                    continue
                _IOU_STATS["pass"] += 1
                iou_mat[ia, ib], allowed[ia, ib] = iou, True
        for ia, ib in assign_pairs(iou_mat, allowed, _IOU_THR):
            accepted.append(((ci, ia), (cj, ib)))
    keys = [(cid, idx) for cid, items in per_camera_filtered.items() for idx in range(len(items))]
    if not keys:
        return [], n_rej
    roots = union_find_clusters(keys, accepted)
    clusters = defaultdict(list)
    for k in keys:
        clusters[roots[k]].append(k)
    fused = []
    for members in clusters.values():
        pts = np.array([per_camera_filtered[c][i]["world_xy"] for c, i in members], dtype=np.float64)
        fused.append({"world_xy": pts.mean(axis=0), "n_cams": len(members)})      # 従来手法なので単純平均
    hk._FUSE_STATS["frames"] += 1
    hk._FUSE_STATS["dets"] += len(keys)
    for f in fused:
        hk._FUSE_STATS["n%d" % min(f["n_cams"], 3)] += 1
    return fused, n_rej


def parse_args(argv):
    """['iou', 'iou@0.15', 'euclid'] -> [(表示名, しきい値 or None, hkのモード名 or None)]"""
    jobs = []
    for a in (argv or ["iou"]):
        if a == "iou":
            jobs += [(f"iou_t{t:.2f}", t, None) for t in IOU_THRS]
        elif a.startswith("iou@"):
            t = float(a.split("@", 1)[1])
            jobs.append((f"iou_t{t:.2f}", t, None))
        elif a in hk.MODES:
            jobs.append((a, None, a))
        else:
            raise SystemExit(f"不明な指定: {a} / 使えるもの: iou, iou@<しきい値>, {list(hk.MODES)}")
    return jobs


def load_ym():
    """hk.load_yolo_module と同じ候補順で読み込む。ただし、候補のファイルがあるのに中の import で
    ModuleNotFoundError になった場合は、本当のエラー(足りないモジュール名)をそのまま出す
    (hk 側は握りつぶして最後の候補の『見つかりません』だけを出すため、原因が分からなくなる)"""
    import importlib
    import traceback
    last = None
    for name in hk.YOLO_MODULE_CANDIDATES:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as e:
            if e.name == name:
                last = e
                continue
            print(f"\n!! {name} は見つかったが、その中で import に失敗しました: 足りないモジュール = {e.name}")
            traceback.print_exc()
            raise
    print(f"\n!! YOLOのphase3モジュールが見つかりません。RESEARCH_DIR={RESEARCH_DIR} / 候補 {hk.YOLO_MODULE_CANDIDATES}")
    raise last


def main():
    global _IOU_THR
    jobs = parse_args(sys.argv[1:])
    ym = hk._YM = load_ym()
    # σ の JSON: 既定は ver4(YOLO検出で推定したもの)、モデルは const。環境変数 HE_SIGMA_JSON / HE_SIGMA_MODEL で上書き可
    os.environ.setdefault("HE_SIGMA_MODEL", "const")
    json_path = os.environ.get("HE_SIGMA_JSON") or str(Path(ym.OUTPUT_DIR) / "bev_calibration_refined" / "sigma_he_kai_ver4.json")
    hk._MODEL = hk.load_model(json_path)
    print(hk._MODEL.describe())
    print(f"有効範囲(maha_h_valid): eps={hk.EPS_VALID_M}m k={hk.K_SIGMA} / GATE_CHI2={hk.GATE_CHI2}")
    ym.FRAME_START = int(os.environ.get("HE_FRAME_START", ym.FRAME_START))
    ym.FRAME_END = int(os.environ.get("HE_FRAME_END", ym.FRAME_END))
    ym.FRAME_STEP = int(os.environ.get("HE_FRAME_STEP", ym.FRAME_STEP))
    if "HE_POSE_CONSTRAINT" in os.environ:
        ym.USE_POSE_CONSTRAINT = os.environ["HE_POSE_CONSTRAINT"] == "1"
    print(f"姿勢別統合制約: {'ON' if ym.USE_POSE_CONSTRAINT else 'OFF'}")
    import gt_ignore_ver1 as gi
    gt_ignore_state = gi.apply(ym)
    print(f"phase3モジュール: {ym.__name__} / フレーム {ym.FRAME_START}〜{ym.FRAME_END} 刻み {ym.FRAME_STEP}")
    print(f"ベースライン B({BASE_STYLE}): IoU しきい値 {[j[1] for j in jobs if j[1] is not None]} / 想定高さ={BASE_HEIGHTS} / 有効範囲={BASE_POLYGONS}"
          f" / bbox 各辺 {EDGE_SAMPLES} 点を床へ投影")

    from Mootrack_core import ROT_FMT, apply_camera_overrides, load_floorplan_cameras, resolve_poses
    hk.setup_poses(resolve_poses(apply_camera_overrides(load_floorplan_cameras(ym.FLOORPLAN_JSON_PATH)), root="Area1", fmt=ROT_FMT))
    global MAP_WH
    from Mootrack_core import load_floorplan_dimensions
    MAP_WH = tuple(load_floorplan_dimensions(ym.FLOORPLAN_JSON_PATH))
    print(f"B の方式: {BASE_STYLE}" + (f" (牛舎 {MAP_WH[0]}x{MAP_WH[1]} m で切り取り / Wmax={WMAX_M} Hmax={HMAX_M})" if BASE_STYLE == "paper" else ""))
    orig_heights = (ym.STANDING_HEIGHT_M, ym.LYING_HEIGHT_M)
    cache, results = {}, {}
    orig_fuse_m = hk.fuse_mahalanobis
    for name, thr, hk_mode in jobs:
        if hk_mode is not None:                               # 既存モード(アブレーション等)はそのまま
            hk.fuse_mahalanobis = orig_fuse_m
            results[name] = hk.run_mode(ym, hk_mode, hk.MODES[hk_mode], OUT_ROOT, cache, orig_heights)
            continue
        _IOU_THR = thr
        for k in _IOU_STATS:
            _IOU_STATS[k] = 0
        hk.fuse_mahalanobis = fuse_iou                        # run_mode の match="maha" が差し替える関数を IoU 版にする
        cfg = dict(match="maha", heights=BASE_HEIGHTS, polygons=BASE_POLYGONS)
        results[name] = hk.run_mode(ym, name, cfg, OUT_ROOT, cache, orig_heights)
        hk.fuse_mahalanobis = orig_fuse_m
        _IOU_SUMMARY[name] = dict(_IOU_STATS)
        print(f"  [IoU] 検出 {_IOU_STATS['dets']} (床に変形できず {_IOU_STATS['no_poly']}) / 重なりのある候補ペア {_IOU_STATS['overlap']}"
              f" -> しきい値 {thr} を超えた {_IOU_STATS['pass']}")

    keys = ["idf1", "idp", "idr", "mota", "motp", "num_switches", "num_objects", "num_matches", "num_false_positives", "num_misses"]
    print(f"\n{'=' * 70}\n■ ベースライン B の比較 (同じYOLO検出・同じ校正)\n{'=' * 70}")
    print(f"{'指標':<22}" + "".join(f"{m:>16}" for m in results))
    for k in keys:
        row = f"{k:<22}"
        for m in results:
            v = results[m].get(k)
            row += f"{'-':>16}" if v is None else (f"{v:>16.4f}" if isinstance(v, float) and abs(v) < 10 else f"{v:>16.0f}")
        print(row)
    if hk._FUSE_SUMMARY:
        print("\n■ 個体統合の数")
        print(f"{'mode':<20}{'検出':>8}{'個体':>8}{'単独':>8}{'2台':>8}{'3台':>8}{'統合率%':>9}")
        for m, d in hk._FUSE_SUMMARY.items():
            nc = d["n1"] + d["n2"] + d["n3"]
            print(f"{m:<20}{d['dets']:>8}{nc:>8}{d['n1']:>8}{d['n2']:>8}{d['n3']:>8}{(d['n2'] + d['n3']) / max(nc, 1) * 100:>9.1f}")
    print("\n※ motp は小さいほど良い(距離の2乗)。num_false_positives / num_misses / num_switches は小さいほど良い。")
    gi.report(gt_ignore_state)


if __name__ == "__main__":
    main()