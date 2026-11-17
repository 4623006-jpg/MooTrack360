"""
calib_common_ver1.py

パラメータ推定(estimate_calib_*.py)と評価(evaluate_calib_params_ver1.py)で共通の部品。
  - データの読み込み (setup): 初期値(Mootrack_core.py の現在値) / ランドマーク43点 / 牛
  - 設定の候補 (VARIANTS, make_settings)
  - クロスバリデーション (run_cv): 設定ごとに、分割ごとのパラメータを保存する
  - 保存 (save_params): 最終パラメータ + 学習に使ったランドマークと牛 + CVの各分割のパラメータと外した点
  - 読み込み (load_params)
推定と評価を分けるため、パラメータのjsonに「使ったランドマーク/牛」と「CVで外した点」を一緒に保存する。

配置: research/phase4/calibration/ (ver1〜ver7 の各スクリプトと同じ場所)
"""

import datetime
import json
import os
import sys

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# 配置: research/phase4/calibration/(このファイル), research/phase4/web_annotate/, research/ 直下に他のフォルダ
RESEARCH_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
PHASE4_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
MOOTRACK_CODE_DIR = (os.path.abspath(os.path.join(RESEARCH_DIR, "calibration")))
for d in (RESEARCH_DIR, PHASE4_DIR, os.path.join(PHASE4_DIR, "web_annotate"),
          os.path.join(PHASE4_DIR, "calibration_try"),     # 過去のスクリプト(refine_calibration_ba_ver1〜7, evaluate_*)の置き場所
          MOOTRACK_CODE_DIR,
          os.path.join(RESEARCH_DIR, "analyze"), os.path.join(RESEARCH_DIR, "annotate"),
          os.path.join(RESEARCH_DIR, "phase3_multi_camera"), THIS_DIR):
    if os.path.isdir(d) and d not in sys.path:
        sys.path.insert(0, d)

import refine_calibration_ba_ver1 as b1
import refine_calibration_ba_ver2 as b2
import refine_calibration_ba_ver5 as b5
import refine_calibration_ba_ver7 as b7

N_EDGE_PER_CAM = 6
K_FOLDS = 5
REPEATS = 3
SEED = 0
# (名前, 動かす内部パラメータ, prior回転[deg], prior位置[m], prior xi, prior alpha)
VARIANTS = [
    ("外部のみ・強prior", (), 0.5, 0.05, 0.15, 0.10),
    ("外部のみ・標準prior", (), 1.0, 0.15, 0.15, 0.10),
    ("外部のみ・緩prior", (), 3.0, 0.6, 0.15, 0.10),
    ("外部+xi,alpha・緩prior", ("xi", "alpha"), 3.0, 0.6, 0.15, 0.10),
    ("内部込み・強prior", ("fx", "fy", "cx", "cy"), 0.5, 0.05, 0.15, 0.10),
]
LM_WEIGHTS_WITH_COWS = [0.5, 1.0, 2.0]       # 牛の重みを1としたときのLMの重み
LM_TOL_M = 0.10      # edge_cows: 出発点の校正より外したLMのRMSEが、これ以上悪化する設定は選ばない


def lm_id(lm):
    return f'{lm["cam"]}:{lm["name"]}'


def cow_id(c):
    return f'{c["frame"]}:{c["id"]}'


def make_settings(kind):
    """kind: edge_cows / lm_only / cows_only"""
    out = []
    for v in VARIANTS:
        if kind == "edge_cows":
            for w in LM_WEIGHTS_WITH_COWS:
                out.append(dict(name=f"{v[0]}/牛+LM x{w:g}", variant=v, w_cow=1.0, w_lm=w))
        elif kind == "lm_only":
            out.append(dict(name=f"{v[0]}/LMのみ", variant=v, w_cow=0.0, w_lm=1.0))
        elif kind == "cows_only":
            out.append(dict(name=f"{v[0]}/牛のみ", variant=v, w_cow=1.0, w_lm=0.0))
        else:
            raise ValueError(kind)
    return out


def setup(frames=None):
    """初期値(Mootrack_core.py の現在値)、ランドマーク43点、牛を読み込む"""
    import Mootrack_core as mc
    import phase3_bev_multicam_gt_zonepose_ver1 as g
    from Mootrack_core import ROT_FMT, apply_camera_overrides, get_intrinsics, load_floorplan, resolve_poses
    from evaluate_refined_calib_on_all_landmarks_ver3 import LANDMARKS, find_region_vertex

    in_dir = g.OUTPUT_DIR / "bev_gt_triangulated"
    out_dir = g.OUTPUT_DIR / "bev_calibration_refined"
    out_dir.mkdir(parents=True, exist_ok=True)
    regions, cameras_json, _w, _h = load_floorplan(g.FLOORPLAN_JSON_PATH)
    poses = resolve_poses(apply_camera_overrides(cameras_json), root="Area1", fmt=ROT_FMT)
    cam_ids = list(g.CAM_IDS)
    base = {c: (poses[c].R.copy(), poses[c].t.copy(), np.array(get_intrinsics(c), dtype=np.float64)) for c in cam_ids}
    landmarks = b2.build_landmarks(regions, LANDMARKS, find_region_vertex)
    for lm, d in zip(landmarks, LANDMARKS):
        lm["name"] = f'{d["region"]}#{d["vertex_idx"]}'
        lm["label"] = d.get("label", "")
    if not frames:
        frames = sorted(int(p.name[:6]) for p in in_dir.glob("*_tri_gt_ver1.json"))
    cows = []
    for fr in frames:
        p = in_dir / f"{fr:06d}_tri_gt_ver1.json"
        if not p.exists():
            print(f"見つかりません: {p}")
            continue
        for c in json.loads(p.read_text(encoding="utf-8"))["cows"]:
            clicks = {k: v for k, v in c.get("clicks", {}).items() if v is not None and k in cam_ids}
            if len(clicks) < 2:
                continue
            cow = {"frame": fr, "id": c["id"], "clicks": clicks}
            X, _r = b1.triangulate_cow(base, clicks)
            if X is None:
                continue
            e = b1.evaluate_cows(base, [cow])
            if (e["angle_deg"][0] >= b1.MIN_ANGLE_USE_DEG and e["reproj_px"].max() <= b1.MAX_INIT_REPROJ_PX
                    and b1.HEIGHT_USE_RANGE_M[0] <= e["height_m"][0] <= b1.HEIGHT_USE_RANGE_M[1]):
                cows.append(cow)
    return dict(g=g, mc=mc, cameras_json=cameras_json, cam_ids=cam_ids, base=base, landmarks=landmarks, cows=cows,
                frames=frames, out_dir=out_dir, init_set=getattr(mc, "CALIBRATION_SET", "不明(Mootrack_core.pyの現在値)"))


# ---------------------------------------------------------------------------
# クロスバリデーション
# ---------------------------------------------------------------------------
def adjust_idx(base, cows, landmarks, st, tr_l, tr_c):
    return b7.adjust(base, [cows[i] for i in tr_c], [landmarks[i] for i in tr_l], st["variant"], st["w_cow"], st["w_lm"])


def run_cv(base, cows, landmarks, settings, fold_specs):
    """fold_specs: [(tr_l_idx, tr_c_idx), ...]。外した点 = 学習に使っていないランドマーク/牛。
    戻り値: 設定ごとに dict(setting, folds=[{cams,tr_l,tr_c}], lm_err(外したときの誤差の平均, 外さなかった点はnan), cow_err(外した牛の再投影誤差))"""
    out = []
    for st in settings:
        folds, lm_sum, lm_cnt, cow_errs = [], np.zeros(len(landmarks)), np.zeros(len(landmarks)), []
        for tr_l, tr_c in fold_specs:
            cams = adjust_idx(base, cows, landmarks, st, tr_l, tr_c)
            s_l, s_c = set(tr_l), set(tr_c)
            te_l = [i for i in range(len(landmarks)) if i not in s_l]
            te_c = [i for i in range(len(cows)) if i not in s_c]
            if te_l:
                lm_sum[te_l] += b2.landmark_errors(cams, [landmarks[i] for i in te_l])
                lm_cnt[te_l] += 1
            if te_c:
                cow_errs.append(b1.evaluate_cows(cams, [cows[i] for i in te_c])["reproj_px"])
            folds.append(dict(cams=cams, tr_l=list(tr_l), tr_c=list(tr_c)))
        lm_err = np.where(lm_cnt > 0, lm_sum / np.maximum(lm_cnt, 1), np.nan)
        out.append(dict(setting=st, folds=folds, lm_err=lm_err,
                        cow_err=np.concatenate(cow_errs) if cow_errs else np.array([])))
    return out


def fold_specs_for(kind, landmarks, cows, base, rng, k=K_FOLDS, repeats=REPEATS):
    n_l, n_c = len(landmarks), len(cows)
    edge = b5.edge_pick_fixed(landmarks, base, N_EDGE_PER_CAM)
    specs = []
    for _ in range(repeats):
        if kind == "lm_only":
            fold = b2.make_folds(n_l, k, rng, strata=[lm["cam"] for lm in landmarks])
            specs += [([i for i in range(n_l) if fold[i] != f], []) for f in range(k)]
        else:
            fold = b2.make_folds(n_c, k, rng)
            tr_l = edge if kind == "edge_cows" else []
            specs += [(list(tr_l), [i for i in range(n_c) if fold[i] != f]) for f in range(k)]
    return specs, edge


def rmse(a):
    a = np.asarray(a, dtype=float)
    a = a[~np.isnan(a)]
    return float(np.sqrt(np.mean(a ** 2))) if len(a) else float("nan")


def med(a):
    return float(np.median(a)) if len(a) else float("nan")


# ---------------------------------------------------------------------------
# 保存 / 読み込み
# ---------------------------------------------------------------------------
def cams_to_json(cams):
    return {c: {"R": np.asarray(v[0]).tolist(), "t": np.asarray(v[1]).tolist(),
                "intrinsics": dict(zip(b1.INTR_NAMES, map(float, v[2])))} for c, v in cams.items()}


def json_to_cams(d):
    return {c: (np.array(v["R"], dtype=np.float64), np.array(v["t"], dtype=np.float64),
                np.array([v["intrinsics"][nm] for nm in b1.INTR_NAMES], dtype=np.float64)) for c, v in d.items()}


def save_params(path, kind, S, setting, cams, tr_l, tr_c, cv_result, cv_table, selection_metric):
    landmarks, cows = S["landmarks"], S["cows"]
    s_l, s_c = set(tr_l), set(tr_c)
    ov = b1.overrides_from_cams(S["cameras_json"], cams)
    doc = {
        "kind": kind, "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "init_calibration_set": S["init_set"], "init": cams_to_json(S["base"]),
        "frames": S["frames"],
        "setting": {"name": setting["name"], "variant": setting["variant"][0], "w_cow": setting["w_cow"], "w_lm": setting["w_lm"]},
        "selection_metric": selection_metric,
        "cameras": cams_to_json(cams), "overrides": ov,
        "train_landmarks": [lm_id(landmarks[i]) for i in sorted(s_l)],
        "unused_landmarks": [lm_id(landmarks[i]) for i in range(len(landmarks)) if i not in s_l],
        "train_cows": [cow_id(cows[i]) for i in sorted(s_c)],
        "unused_cows": [cow_id(cows[i]) for i in range(len(cows)) if i not in s_c],
        "cv_table": cv_table,
        "cv_folds": [{"cameras": cams_to_json(f["cams"]),
                      "heldout_landmarks": [lm_id(landmarks[i]) for i in range(len(landmarks)) if i not in set(f["tr_l"])],
                      "heldout_cows": [cow_id(cows[i]) for i in range(len(cows)) if i not in set(f["tr_c"])]}
                     for f in cv_result["folds"]],
    }
    Path_ = __import__("pathlib").Path
    Path_(path).write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    intr_txt = {c: {nm: round(float(cams[c][2][i]), 6) for i, nm in enumerate(b1.INTR_NAMES)} for c in cams}
    snippet = [f"# Mootrack_core.py に貼る内容 ({kind}: {setting['name']})", "CAMERA_PARAM_OVERRIDES = {"] + \
              [f'    "{a}": {b},' for a, b in ov.items()] + ["}", "", "CAM_INTRINSICS = {"] + \
              [f'    "{c}": {v},' for c, v in intr_txt.items()] + ["}"]
    Path_(str(path).replace(".json", ".txt")).write_text("\n".join(snippet), encoding="utf-8")
    return doc


def estimate_main(kind):
    """kind: edge_cows / lm_only / cows_only。CVで設定を選び、最終パラメータを保存する"""
    frames = [int(a) for a in sys.argv[1:]] or None
    S = setup(frames)
    base, landmarks, cows = S["base"], S["landmarks"], S["cows"]
    print(f"フレーム {S['frames']}: 牛 {len(cows)}頭 / ランドマーク {len(landmarks)}点 / 出発点の校正: {S['init_set']}")
    if len(cows) < 2 * K_FOLDS and kind != "lm_only":
        raise SystemExit("牛の数が少なすぎます")
    rng = np.random.default_rng(SEED)
    specs, edge = fold_specs_for(kind, landmarks, cows, base, rng)
    settings = make_settings(kind)
    title = {"edge_cows": "各カメラ端のLM + 牛", "lm_only": "LMのみ", "cows_only": "牛のみ"}[kind]
    metric = "lm_rmse" if kind == "lm_only" else "cow_median"
    print(f"\n=== {title}: クロスバリデーション(設定の選択)。選択基準: 外した{'ランドマーク' if kind == 'lm_only' else '牛'}の誤差 ===")
    if kind == "edge_cows":
        print("学習に使う端のLM: " + ", ".join(lm_id(landmarks[i]) for i in edge))
    res = run_cv(base, cows, landmarks, settings, specs)
    table = []
    print(f"{'設定':<36}{'外したLM RMSE[m]':>16}{'外した牛 中央値[px]':>18}{'平均':>8}{'90%点':>8}")
    for r in res:
        row = dict(setting=r["setting"]["name"], heldout_lm_rmse=rmse(r["lm_err"]), heldout_cow_median=med(r["cow_err"]),
                   heldout_cow_mean=float(np.mean(r["cow_err"])) if len(r["cow_err"]) else float("nan"),
                   heldout_cow_p90=float(np.percentile(r["cow_err"], 90)) if len(r["cow_err"]) else float("nan"))
        table.append(row)
        print(f"{row['setting']:<36}{row['heldout_lm_rmse']:>16.3f}{row['heldout_cow_median']:>18.2f}"
              f"{row['heldout_cow_mean']:>8.2f}{row['heldout_cow_p90']:>8.2f}")
    key = (lambda t: t["heldout_lm_rmse"]) if kind == "lm_only" else (lambda t: t["heldout_cow_median"])
    cand = list(range(len(table)))
    if kind == "edge_cows":
        # 牛の誤差だけで選ぶと、床の誤差を犠牲にする設定(LMの重みが最小のもの)が選ばれやすい。
        # 出発点の校正の「外したLM」での誤差 + LM_TOL_M 以内の設定だけを候補にして、その中で牛の誤差が最小のものを選ぶ。
        inner = [i for i in range(len(landmarks)) if i not in set(edge)]
        base_lm = rmse(b2.landmark_errors(base, [landmarks[i] for i in inner]))
        cand = [i for i, t in enumerate(table) if t["heldout_lm_rmse"] <= base_lm + LM_TOL_M]
        print(f"出発点の校正の、外したLM(内側{len(inner)}点)のRMSE: {base_lm:.3f} m -> 候補: 外したLM RMSE <= {base_lm + LM_TOL_M:.3f} m の設定のみ")
        if not cand:
            print("候補がありません。外したLM RMSEが最小の設定を選びます。")
            cand = [int(np.argmin([t["heldout_lm_rmse"] for t in table]))]
    best_i = min(cand, key=lambda i: key(table[i]))
    best, st = res[best_i], res[best_i]["setting"]
    print(f"\n選択された設定: {st['name']}  (選択基準 {metric} = {key(table[best_i]):.3f})")

    if kind == "edge_cows":
        tr_l, tr_c = list(edge), list(range(len(cows)))
    elif kind == "lm_only":
        tr_l, tr_c = list(range(len(landmarks))), []
    else:
        tr_l, tr_c = [], list(range(len(cows)))
    cams = adjust_idx(base, cows, landmarks, st, tr_l, tr_c)
    path = S["out_dir"] / f"calib_{kind}_ver1.json"
    save_params(path, kind, S, st, cams, tr_l, tr_c, best, table, metric)
    print(f"最終パラメータ: 学習に使ったLM {len(tr_l)}点 / 牛 {len(tr_c)}頭 -> {path}")
    print("評価は: python evaluate_calib_params_ver1.py " + str(path))