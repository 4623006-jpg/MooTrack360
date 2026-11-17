"""
eval_stratified_ver2.py  (配置: research/phase5/)  ※ver1 に「評価ゲートの掃引」を追加

層別評価: 動かない牛・単眼範囲の牛が大半で IDF1 などが飽和して差が出ないため、
GT の位置から先に決めた「層」ごとに FN / FP / ID switch を数え直す。1回の実行で全層・全モードを出す。

層(各 GT 個体 × フレームに GT の位置だけから付ける。手法には依存しない):
  重なり範囲   : GT 位置が 2台以上のカメラの有効範囲に入っている
  単眼範囲     : 1台以下
  動く牛       : 前後 STRAT_WIN サンプル(処理フレーム)の範囲で、現在位置から STRAT_MOVE_M [m] 以上動く
  静止牛       : 上以外
  境界またぎ   : 前後 STRAT_WIN サンプルの間で、最も近いカメラが入れ替わる
  重なり×動く / 単眼×静止 : 組み合わせ
  全体         : すべて(従来の数え方と同じ件数になる)
追加(層ごとの IDF1): 上の8つの層それぞれで、その層に入る「GT個体×フレーム」だけを GT に残して IDF1 を計算し直す
  (外した GT 個体に全体の評価で対応づけられた仮説は外す。牛の軌跡が層の出入りで途切れても、IDF1 は残ったフレームだけで計算される)。
  層が小さい(境界またぎ 84 延べなど)と値が不安定なので、GT延べ数を併記して読むこと。
追加(IDF1 の牛単位の絞り込み): IDF1 は牛ごとの軌跡の指標なので、フレームではなく「牛」単位で対象を絞って再計算する。
  動いた牛        : 全期間で GT の位置が STRAT_ACTIVE_M [m] 以上動いた牛だけを GT に残す(動かない牛は GT から外す)
  動いた牛(重なり通過): 上のうち、重なり範囲にいたフレームがある牛
  全牛            : 絞らない(従来の IDF1 と同じになる確認用)
  ストレス(動いた牛+誤り牛): 動いた牛に、実行した全モードのどれかで FN/FP/SW が出た牛(動かない牛でも)を足した集合。
                     全モードで同じ集合を使う。誤りで牛を選ぶので差が強調される。主指標ではなくストレステストとして併記する
  外した牛に(全体の評価で)対応づけられた仮説は、評価から外す(外した牛の仮説が FP に数えられないように)。
  対応づけの無い仮説(FP)は、いちばん近い GT(FP_NEAR_M 以内)が外した牛なら外す。
FP は GT を持たないので、そのフレームで仮説の位置に最も近い GT 個体(3 m 以内)の層に入れる。3 m 以内に GT が無い FP は「全体」にだけ数える。
指標: FN / FP / SW の件数と、err% = (FN+FP+SW) / GT延べ数 × 100、MOTA_s = 100 - err%。

追加(ver2: 評価ゲートの感度分析): 評価の対応づけゲート(既定 3 m = 2×DIST_THR_TRACK_M)を GATE_SWEEP_M の値に変えて、
  全体の FN / FP / SW / IDF1 / MOTA を同じ追跡結果から再計算して並べる。トラッカー(1.5 m)と統合は変えない。評価の距離だけを変える。
  ゲートが広いと誤統合(2頭の中間の点)や重複が誤りとして出ないので、狭めたときに手法ごとの誤りがどう増えるかを見る。
  3.0 m の行は、通常の評価(全体)と一致するはず(一致確認も出す)。

使い方(research/ で):
    HE_POSE_CONSTRAINT=0 uv run python ./phase5/eval_stratified_ver2.py iou@0.2 maha_h_valid euclid
    (指定の書き方は baseline_iou_ver1.py と同じ。iou は掃引、iou@0.2 は1つ、それ以外は phase3_he_kai_matching_ver10 のモード名)
環境変数:
    STRAT_RANGE   : 層分けに使う有効範囲 orig(HEのみの円, 既定) / he_kai(HE(改))   ※全モード共通で固定
    STRAT_WIN     : 窓の幅(片側のサンプル数, 既定 5)
    STRAT_MOVE_M  : 動くと見なす移動量 [m] (既定 0.5)
    STRAT_ACTIVE_M: IDF1 の層別で「動いた牛」と見なす全期間の移動量 [m] (既定 2.0)
    STRAT_FP_NEAR_M: FP を GT 個体に結びつける距離 [m] (既定 3.0)
    GATE_SWEEP_M  : 評価ゲートの掃引 [m] カンマ区切り (既定 1.0,1.5,2.0,3.0)
    その他は baseline_iou_ver1.py と同じ (IOU_THRS, BASE_HEIGHTS, HE_POSE_CONSTRAINT, HE_FRAME_* など)
出力: research/phase5/baseline_out/stratified_ver2/<モード名>/strat_<モード名>.csv と、全モードをまとめた strat_all.csv
"""

import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import baseline_iou_ver1 as bl          # パス設定・IoU 融合・引数の解釈を共有する
hk = bl.hk

STRAT_RANGE = os.environ.get("STRAT_RANGE", "orig")
WIN = int(os.environ.get("STRAT_WIN", 5))
MOVE_M = float(os.environ.get("STRAT_MOVE_M", 0.5))
FP_NEAR_M = float(os.environ.get("STRAT_FP_NEAR_M", 3.0))
ACTIVE_M = float(os.environ.get("STRAT_ACTIVE_M", 2.0))
GATE_SWEEP_M = [float(x) for x in os.environ.get("GATE_SWEEP_M", "1.0,1.5,2.0,3.0").split(",") if x.strip()]
OUT_ROOT = Path(os.environ.get("STRAT_OUT_DIR", os.path.join(THIS_DIR, "baseline_out", "stratified_ver2")))

STRATA = {
    "全体":         lambda f: True,
    "重なり範囲":   lambda f: f["overlap"],
    "単眼範囲":     lambda f: not f["overlap"],
    "動く牛":       lambda f: f["moving"],
    "静止牛":       lambda f: not f["moving"],
    "境界またぎ":   lambda f: f["crossing"],
    "重なり×動く":  lambda f: f["overlap"] and f["moving"],
    "単眼×静止":    lambda f: (not f["overlap"]) and (not f["moving"]),
}

HYP = []                                    # フレームごとのトラッカー出力 [(track_id, xy), ...]


def install_tracker_hook(ym):
    """トラッカーの出力(仮説の位置)を記録する。結果は変えない"""
    for name in ("ReIDBevTracker", "SimpleBevTracker"):
        base = getattr(ym, name, None)
        if base is None or getattr(base, "_strat_wrapped", False):
            continue

        class Wrapped(base):
            _strat_wrapped = True

            def update(self, *a, **k):
                out = super().update(*a, **k)
                HYP.append([(tid, np.asarray(xy, dtype=np.float64)) for tid, xy in out])
                return out
        Wrapped.__name__ = base.__name__
        setattr(ym, name, Wrapped)


def build_range_polygons(poses, cam_ids):
    from Mootrack_core import get_intrinsics
    if STRAT_RANGE == "he_kai":
        from he_kai_sigma_model_ver1 import build_valid_polygons
        return build_valid_polygons(poses, cam_ids, get_intrinsics, hk._MODEL, hk.EPS_VALID_M, hk.K_SIGMA, height_m=0.8, verbose=False)
    from curved_zone_utils_ver1 import build_camera_polygons
    return build_camera_polygons(poses, cam_ids, get_intrinsics, verbose=False)


def classify_instances(gt_pos, cam_xy, polys, win=WIN, move_m=MOVE_M):
    """gt_pos: フレームごとの {GT id: 位置 or None}。戻り値: {(フレーム番号, GT id): {overlap, moving, crossing}}"""
    from curved_zone_utils_ver1 import point_in_polygon
    series = defaultdict(list)
    for k, d in enumerate(gt_pos):
        for oid, pos in d.items():
            if pos is not None:
                series[int(oid)].append((k, np.asarray(pos, dtype=np.float64)))
    cids = list(cam_xy)
    flags = {}
    for oid, ser in series.items():
        pts = np.array([p for _, p in ser])
        near = [cids[int(np.argmin([np.linalg.norm(p - cam_xy[c]) for c in cids]))] for p in pts]
        for i, (k, p) in enumerate(ser):
            lo, hi = max(0, i - win), min(len(ser), i + win + 1)
            flags[(k, oid)] = dict(
                overlap=sum(point_in_polygon(p, polys[c]) for c in cids) >= 2,
                moving=float(np.max(np.linalg.norm(pts[lo:hi] - p, axis=1))) >= move_m,
                crossing=len(set(near[lo:hi])) >= 2)
    return flags


def tally(flags, events, gt_pos, hyp):
    """events: motmetrics の mot_events(reset_index 済み)。戻り値: {層: dict(N, FN, FP, SW)}"""
    import pandas as pd
    res = {s: dict(N=0, FN=0, FP=0, SW=0) for s in STRATA}
    for f in flags.values():
        for s, pred in STRATA.items():
            if pred(f):
                res[s]["N"] += 1
    n_unassigned_fp = 0

    def add(key, field):
        res["全体"][field] += 1
        f = flags.get(key)
        if f is None:
            return False
        for s, pred in STRATA.items():
            if s != "全体" and pred(f):
                res[s][field] += 1
        return True

    for _, e in events.iterrows():
        typ, fid = e["Type"], int(e["FrameId"])
        if typ in ("MISS", "SWITCH") and not pd.isna(e["OId"]):
            add((fid, int(e["OId"])), "FN" if typ == "MISS" else "SW")
        elif typ == "FP":
            pos = dict(hyp[fid]).get(int(e["HId"])) if fid < len(hyp) and not pd.isna(e["HId"]) else None
            best, bd = None, FP_NEAR_M
            if pos is not None and fid < len(gt_pos):
                for oid, gp in gt_pos[fid].items():
                    if gp is not None:
                        dd = float(np.linalg.norm(np.asarray(gp) - pos))
                        if dd <= bd:
                            best, bd = int(oid), dd
            if best is None or not add((fid, best), "FP"):
                res["全体"]["FP"] += 1
                n_unassigned_fp += 1
    res["_fp_unassigned"] = n_unassigned_fp
    return res


def cow_subsets(gt_pos, flags):
    """牛(GT id)単位の部分集合 {名前: set(GT id)}"""
    series = defaultdict(list)
    for d in gt_pos:
        for oid, pos in d.items():
            if pos is not None:
                series[int(oid)].append(np.asarray(pos, dtype=np.float64))
    active = set()
    for oid, pts in series.items():
        P = np.array(pts)
        if float(np.max(np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1))) >= ACTIVE_M:
            active.add(oid)
    ov = {oid for (k, oid), f in flags.items() if f["overlap"]}
    return {"全牛": set(series), "動いた牛": active, "動いた牛(重なり通過)": active & ov}


def matched_map(events):
    """全体の評価で、各フレームの仮説 id がどの GT 個体に対応づけられたか {フレーム: {仮説id: GT id}}"""
    import pandas as pd
    out = defaultdict(dict)
    for _, e in events.iterrows():
        if e["Type"] in ("MATCH", "SWITCH") and not pd.isna(e["HId"]) and not pd.isna(e["OId"]):
            out[int(e["FrameId"])][int(e["HId"])] = int(e["OId"])
    return out


def subset_metrics(gt_pos, hyp, keep, dist_thr, events, keep_inst=None):
    """keep の牛だけを GT に残して IDF1 などを計算し直す。
    外す仮説: 全体の評価で「外した牛」に対応づけられた仮説。対応づけの無かった仮説(FP)は、
    いちばん近い(FP_NEAR_M以内)GT が外した牛なら外す(近くに GT が無ければ残す)。
    ※ 最初の版は全部の仮説を『いちばん近い GT』で振り分けていたが、牛が密集した場所で誤って外す/残すため、
       SW・FP が実際より多く出ていた。今は全体の対応づけを使う。"""
    import motmetrics as mm
    matched = matched_map(events)
    # keep_inst(フレーム, GT id) が与えられたら、牛ではなく「GT個体×フレーム」単位で残す(層別 IDF1 用)
    is_kept = keep_inst if keep_inst is not None else (lambda k_, o_: o_ in keep)
    acc = mm.MOTAccumulator(auto_id=True)
    for k, d in enumerate(gt_pos):
        gids = [int(o) for o, p in d.items() if p is not None]
        kept_h = []
        for tid, xy in (hyp[k] if k < len(hyp) else []):
            if int(tid) in matched.get(k, {}):
                if is_kept(k, matched[k][int(tid)]):
                    kept_h.append((tid, xy))
                continue
            best, bd = None, FP_NEAR_M
            for o in gids:
                dd = float(np.linalg.norm(np.asarray(d[o]) - xy))
                if dd <= bd:
                    best, bd = o, dd
            if best is not None and not is_kept(k, best):
                continue
            kept_h.append((tid, xy))
        g_ids = [o for o in gids if is_kept(k, o)]
        g_pts = [np.asarray(d[o]) for o in g_ids]
        if g_pts and kept_h:
            dist = mm.distances.norm2squared_matrix(np.array(g_pts), np.array([xy for _, xy in kept_h]), max_d2=(2.0 * dist_thr) ** 2)
        else:
            dist = np.empty((len(g_ids), len(kept_h)))
        acc.update(g_ids, [t for t, _ in kept_h], dist)
    mh = mm.metrics.create()
    r = mh.compute(acc, metrics=["idf1", "idp", "idr", "mota", "num_switches", "num_false_positives", "num_misses", "num_objects"], name="x")
    return {k: float(v) for k, v in r.iloc[0].to_dict().items()}


def gate_metrics(gt_pos, hyp, gate_m):
    """評価の対応づけゲートを gate_m [m] にして、全体の指標を同じ追跡結果から計算し直す(絞り込みなし)。
    ver1 の max_d2=(2*DIST_THR_TRACK_M)**2 と同じ作り(gate_m=3.0 で一致する)"""
    import motmetrics as mm
    acc = mm.MOTAccumulator(auto_id=True)
    for k, d in enumerate(gt_pos):
        g_ids = [int(o) for o, p in d.items() if p is not None]
        g_pts = [np.asarray(d[o]) for o in g_ids]
        hh = list(hyp[k]) if k < len(hyp) else []
        if g_pts and hh:
            dist = mm.distances.norm2squared_matrix(np.array(g_pts), np.array([xy for _, xy in hh]), max_d2=gate_m ** 2)
        else:
            dist = np.empty((len(g_ids), len(hh)))
        acc.update(g_ids, [t for t, _ in hh], dist)
    r = mm.metrics.create().compute(acc, metrics=["idf1", "mota", "num_switches", "num_false_positives", "num_misses", "num_objects"], name="x")
    return {k: float(v) for k, v in r.iloc[0].to_dict().items()}


def error_cows(events, gt_pos, hyp):
    """FN / FP / SW が出た GT 個体(牛)の id の集合。FP は仮説に最も近い GT (FP_NEAR_M 以内) の牛とする"""
    import pandas as pd
    cows = set()
    for _, e in events.iterrows():
        typ, fid = e["Type"], int(e["FrameId"])
        if typ in ("MISS", "SWITCH") and not pd.isna(e["OId"]):
            cows.add(int(e["OId"]))
        elif typ == "FP" and fid < len(hyp) and not pd.isna(e["HId"]) and fid < len(gt_pos):
            pos = dict(hyp[fid]).get(int(e["HId"]))
            if pos is None:
                continue
            best, bd = None, FP_NEAR_M
            for oid, gp in gt_pos[fid].items():
                if gp is not None:
                    dd = float(np.linalg.norm(np.asarray(gp) - pos))
                    if dd <= bd:
                        best, bd = int(oid), dd
            if best is not None:
                cows.add(best)
    return cows


def table_rows(res):
    rows = []
    for s in STRATA:
        d = res[s]
        err = (d["FN"] + d["FP"] + d["SW"]) / max(d["N"], 1) * 100
        rows.append(dict(層=s, GT延べ数=d["N"], FN=d["FN"], FP=d["FP"], SW=d["SW"], **{"err%": err, "MOTA_s": 100 - err}))
    return rows


def main():
    import pandas as pd
    jobs = bl.parse_args(sys.argv[1:])
    ym = hk._YM = bl.load_ym()
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
    print(f"層分け: 有効範囲={STRAT_RANGE} / 窓=±{WIN}サンプル / 動く={MOVE_M}m以上 / FPは{FP_NEAR_M}m以内のGTに結びつける")

    from Mootrack_core import CAM_IDS, ROT_FMT, apply_camera_overrides, load_floorplan_cameras, resolve_poses
    poses = resolve_poses(apply_camera_overrides(load_floorplan_cameras(ym.FLOORPLAN_JSON_PATH)), root="Area1", fmt=ROT_FMT)
    hk.setup_poses(poses)
    polys = build_range_polygons(poses, CAM_IDS)
    cam_xy = {c: np.asarray(poses[c].t[:2], dtype=np.float64) for c in CAM_IDS}
    install_tracker_hook(ym)

    orig_heights = (ym.STANDING_HEIGHT_M, ym.LYING_HEIGHT_M)
    cache, all_rows, sub_all, store, strat_idf1_all = {}, [], [], [], []
    gate_rows = []
    orig_fuse_m = hk.fuse_mahalanobis
    for name, thr, hk_mode in jobs:
        HYP.clear()
        if hk_mode is not None:
            hk.fuse_mahalanobis = orig_fuse_m
            hk.run_mode(ym, hk_mode, hk.MODES[hk_mode], OUT_ROOT, cache, orig_heights)
        else:
            bl._IOU_THR = thr
            hk.fuse_mahalanobis = bl.fuse_iou
            hk.run_mode(ym, name, dict(match="maha", heights=bl.BASE_HEIGHTS, polygons=bl.BASE_POLYGONS), OUT_ROOT, cache, orig_heights)
            hk.fuse_mahalanobis = orig_fuse_m
        evt = hk._EVT
        if not evt.accs:
            print(f"!! {name}: イベントが記録されていません")
            continue
        events = evt.accs[-1].mot_events.reset_index()
        flags = classify_instances(evt.gt_pos, cam_xy, polys)
        res = tally(flags, events, evt.gt_pos, list(HYP))
        rows = table_rows(res)
        df = pd.DataFrame(rows)
        out_dir = OUT_ROOT / name
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / f"strat_{name}.csv", index=False)
        print(f"\n■ 層別 {name}  (GTに結びつかないFP {res['_fp_unassigned']} 件は「全体」にだけ計上)")
        print(df.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
        for r in rows:
            all_rows.append(dict(mode=name, **r))
        dist_thr = getattr(ym, "DIST_THR_TRACK_M", 1.5)
        store.append(dict(name=name, gt_pos=list(evt.gt_pos), hyp=list(HYP), events=events, flags=flags))
        sub_rows = []
        for sname, keep in cow_subsets(evt.gt_pos, flags).items():
            m = subset_metrics(evt.gt_pos, list(HYP), keep, dist_thr, events)
            n_inst = sum(1 for (k_, oid_) in flags if oid_ in keep)
            sub_rows.append(dict(部分集合=sname, 牛の数=len(keep), GT延べ数=n_inst, idf1=m["idf1"], idp=m["idp"], idr=m["idr"],
                                 mota=m["mota"], SW=int(m["num_switches"]), FP=int(m["num_false_positives"]), FN=int(m["num_misses"])))
            sub_all.append(dict(mode=name, **sub_rows[-1]))
        strat_rows = []
        for sname, pred in STRATA.items():
            keep_inst = (lambda k_, o_, p=pred: (k_, o_) in flags and p(flags[(k_, o_)]))
            m = subset_metrics(evt.gt_pos, list(HYP), None, dist_thr, events, keep_inst=keep_inst)
            strat_rows.append(dict(層=sname, GT延べ数=res[sname]["N"], idf1=m["idf1"], idp=m["idp"], idr=m["idr"], mota=m["mota"],
                                   SW=int(m["num_switches"]), FP=int(m["num_false_positives"]), FN=int(m["num_misses"])))
            strat_idf1_all.append(dict(mode=name, **strat_rows[-1]))
        print(f"\n■ 層別の IDF1 {name}  (層に入る GT 個体×フレームだけを GT に残して計算)")
        print(pd.DataFrame(strat_rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(f"\n■ IDF1 の層別(牛単位で絞る) {name}  (動いた牛 = 全期間で {ACTIVE_M} m 以上動いた牛)")
        print(pd.DataFrame(sub_rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        for g in GATE_SWEEP_M:
            gm = gate_metrics(evt.gt_pos, list(HYP), g)
            gate_rows.append(dict(mode=name, ゲートm=g, idf1=gm["idf1"], mota=gm["mota"], SW=int(gm["num_switches"]),
                                  FP=int(gm["num_false_positives"]), FN=int(gm["num_misses"]), GT延べ数=int(gm["num_objects"])))
        g3 = [r for r in gate_rows if r["mode"] == name and abs(r["ゲートm"] - 3.0) < 1e-9]
        if g3:
            o = res["全体"]
            ok = (g3[0]["FN"], g3[0]["FP"], g3[0]["SW"]) == (o["FN"], o["FP"], o["SW"])
            print(f"\n[一致確認] ゲート3.0mの再計算 FN/FP/SW = {g3[0]['FN']}/{g3[0]['FP']}/{g3[0]['SW']} / 通常の評価 {o['FN']}/{o['FP']}/{o['SW']} -> {'OK' if ok else '不一致(要確認)'}")

    if all_rows:
        al = pd.DataFrame(all_rows)
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        al.to_csv(OUT_ROOT / "strat_all.csv", index=False)
        print(f"\n{'=' * 70}\n■ 層ごとの err% = (FN+FP+SW)/GT延べ数 (小さいほど良い)\n{'=' * 70}")
        print(al.pivot(index="層", columns="mode", values="err%").reindex(list(STRATA)).to_string(float_format=lambda v: f"{v:.2f}"))
        print("\n■ 層ごとの (FN / FP / SW)")
        for s in STRATA:
            sub = al[al["層"] == s]
            print(f"{s:<12}" + "  ".join(f"{r['mode']}:{int(r['FN'])}/{int(r['FP'])}/{int(r['SW'])}" for _, r in sub.iterrows())
                  + f"   (GT延べ {int(sub['GT延べ数'].iloc[0])})")
        print(f"\n保存: {OUT_ROOT / 'strat_all.csv'}")
    if store:
        subsets0 = cow_subsets(store[0]["gt_pos"], store[0]["flags"])
        err_union = set()
        for it in store:
            err_union |= error_cows(it["events"], it["gt_pos"], it["hyp"])
        stress = subsets0["動いた牛"] | err_union
        n_static_err = len(err_union - subsets0["動いた牛"])
        print(f"\n■ ストレス集合: 動いた牛 {len(subsets0['動いた牛'])} + 動かないが誤りが出た牛 {n_static_err} = {len(stress)} 頭 "
              f"(全牛 {len(subsets0['全牛'])} 頭 / 全 {len(store)} モードの誤り牛の和集合)")
        for it in store:
            m = subset_metrics(it["gt_pos"], it["hyp"], stress, getattr(ym, "DIST_THR_TRACK_M", 1.5), it["events"])
            n_inst = sum(1 for (k_, oid_) in it["flags"] if oid_ in stress)
            row = dict(部分集合="ストレス(動いた牛+誤り牛)", 牛の数=len(stress), GT延べ数=n_inst, idf1=m["idf1"], idp=m["idp"], idr=m["idr"],
                       mota=m["mota"], SW=int(m["num_switches"]), FP=int(m["num_false_positives"]), FN=int(m["num_misses"]))
            sub_all.append(dict(mode=it["name"], **row))
    if strat_idf1_all:
        si = pd.DataFrame(strat_idf1_all)
        si.to_csv(OUT_ROOT / "strat_idf1_all.csv", index=False)
        print(f"\n{'=' * 70}\n■ 層別の IDF1 (大きいほど良い)\n{'=' * 70}")
        print(si.pivot(index="層", columns="mode", values="idf1").reindex(list(STRATA)).to_string(float_format=lambda v: f"{v:.4f}"))
        print("\n■ 層ごとの (SW / FP / FN)")
        for s_ in STRATA:
            sub = si[si["層"] == s_]
            print(f"{s_:<12}" + "  ".join(f"{r['mode']}:{int(r['SW'])}/{int(r['FP'])}/{int(r['FN'])}" for _, r in sub.iterrows())
                  + f"   (GT延べ {int(sub['GT延べ数'].iloc[0])})")
    if sub_all:
        sa = pd.DataFrame(sub_all)
        sa.to_csv(OUT_ROOT / "subset_idf1_all.csv", index=False)
        order = ["全牛", "動いた牛", "動いた牛(重なり通過)", "ストレス(動いた牛+誤り牛)"]
        print(f"\n{'=' * 70}\n■ 牛単位で絞った IDF1 (大きいほど良い)\n{'=' * 70}")
        print(sa.pivot(index="部分集合", columns="mode", values="idf1").reindex(order).to_string(float_format=lambda v: f"{v:.4f}"))
        print("\n■ 牛の数 / GT延べ数 / SW / FP / FN")
        for sname in order:
            sub = sa[sa["部分集合"] == sname]
            print(f"{sname:<22}牛{int(sub['牛の数'].iloc[0])} GT延べ{int(sub['GT延べ数'].iloc[0])}  " + "  ".join(
                f"{r['mode']}:{int(r['SW'])}/{int(r['FP'])}/{int(r['FN'])}" for _, r in sub.iterrows()))
    if gate_rows:
        gr = pd.DataFrame(gate_rows)
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        gr.to_csv(OUT_ROOT / "gate_sweep_all.csv", index=False)
        print(f"\n{'=' * 70}\n■ 評価ゲートの感度分析(トラッカーと統合は同じ。評価の距離だけを変える)\n{'=' * 70}")
        print("IDF1 (大きいほど良い)")
        print(gr.pivot(index="ゲートm", columns="mode", values="idf1").to_string(float_format=lambda v: f"{v:.4f}"))
        print("\nFN+FP+SW (小さいほど良い)")
        gr["誤り計"] = gr["FN"] + gr["FP"] + gr["SW"]
        print(gr.pivot(index="ゲートm", columns="mode", values="誤り計").to_string())
        print("\n(FN / FP / SW)")
        for g in sorted(gr["ゲートm"].unique()):
            sub = gr[gr["ゲートm"] == g]
            print(f"{g:>5.2f} m  " + "  ".join(f"{r['mode']}:{int(r['FN'])}/{int(r['FP'])}/{int(r['SW'])}" for _, r in sub.iterrows()))
        print(f"\n保存: {OUT_ROOT / 'gate_sweep_all.csv'}")
    gi.report(gt_ignore_state)


if __name__ == "__main__":
    main()