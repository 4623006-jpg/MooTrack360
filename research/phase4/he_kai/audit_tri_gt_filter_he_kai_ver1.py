"""
audit_tri_gt_filter_he_kai_ver1.py

三角測量GT の品質フィルタ (level 1 / 交差角 >= MIN_ANGLE_DEG / 残差 <= MAX_RESID_M) が
σ の推定にどう効いていたかを調べる。

  1. 内訳: 牛が何頭あり、どの基準で何頭除かれたか
  2. 食い違い |d - m| (d = P_i - P_j, m = 偏りの予測) の分布を、通過した牛 / 除かれた牛で比べる
  3. σ を 「通過した牛だけ」「全部」「除かれた牛だけ」で推定し直して比べる
     (全部で σ が大きく増えれば、フィルタで σ が小さめに出ていたことになる)
     ※ 全部には ID の取り違え(アノテーションミス)も混ざる。「検出そのものの誤差」ではなく「アノテーション込みの食い違い」

使い方: uv run python ./phase4/he_kai/audit_tri_gt_filter_he_kai_ver1.py [フレーム ...]   (省略時は三角測量GT のある全フレーム)
環境変数: N_BOOT (既定 50), H_A_STANDING / H_A_LYING (estimate_sigma と同じ)
配置: research/phase4/he_kai/  (estimate_sigma_he_kai_ver3.py と同じ場所)
注意: 三角測量GT のJSON があるフレームしか調べられない (今は frame 100 のみ)。
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESEARCH_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
PHASE4_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
MOOTRACK_CODE_DIR = (os.path.abspath(os.path.join(RESEARCH_DIR, "calibration")))
for d in (RESEARCH_DIR, PHASE4_DIR, os.path.join(PHASE4_DIR, "web_annotate"),
          os.path.join(PHASE4_DIR, "calibration"), os.path.join(PHASE4_DIR, "calibration_try"),
          os.path.join(PHASE4_DIR, "he_kai"),
          MOOTRACK_CODE_DIR,
          os.path.join(RESEARCH_DIR, "analyze"), os.path.join(RESEARCH_DIR, "annotate"),
          os.path.join(RESEARCH_DIR, "phase3_multi_camera"), THIS_DIR):
    if os.path.isdir(d) and d not in sys.path:
        sys.path.insert(0, d)

N_BOOT = int(os.environ.get("N_BOOT", 50))


def classify(cow, an):
    """牛が品質フィルタを通るか。戻り値: (通過したか, 理由)"""
    r = cow.get("result", {})
    if r.get("level") != 1 or r.get("X") is None:
        return False, "level!=1 / 位置なし"
    if r.get("max_angle_deg", 0) < an.MIN_ANGLE_DEG:
        return False, f"交差角 < {an.MIN_ANGLE_DEG}°"
    if r.get("resid_m", 9) > an.MAX_RESID_M:
        return False, f"残差 > {an.MAX_RESID_M} m"
    return True, "通過"


def build_raw_all(tri_cows, xml_objs, cam_ids, an, frame):
    """fit_assumed_height の build_raw から品質フィルタを外したもの。サンプルに passed / reason を付ける"""
    tile_w = an.IMAGE_WIDTH_PX / an.N_CAMS
    raw = []
    for cow in tri_cows:
        passed, reason = classify(cow, an)
        r = cow.get("result", {})
        for k, cid in enumerate(cam_ids):
            uv = cow.get("clicks", {}).get(cid)
            if uv is None:
                continue
            x_full = uv[0] + k * tile_w
            hit = [o for o in xml_objs if o["box"][0] <= x_full <= o["box"][2] and o["box"][1] <= uv[1] <= o["box"][3]]
            if not hit:
                continue
            o = min(hit, key=lambda o: (o["box"][2] - o["box"][0]) * (o["box"][3] - o["box"][1]))
            pose = o["pose"] if o["pose"] in ("standing", "lying") else (cow.get("pose") or None)
            if pose not in ("standing", "lying"):
                hb = r.get("height_m")
                pose = None if hb is None else ("lying" if hb < an.POSE_HEIGHT_THR_M else "standing")
            if pose is None:
                continue
            raw.append(dict(frame=frame, cow=cow["id"], cam=cid, pose=pose, passed=passed, reason=reason,
                            P_gt=np.zeros(2),
                            uv=((o["box"][0] + o["box"][2]) / 2.0 - k * tile_w, (o["box"][1] + o["box"][3]) / 2.0)))
    return raw


def sub_data(data, sel):
    """ペアを選んだ部分データ (サンプル側の配列はそのまま、ペア側だけ絞る)"""
    d = dict(data)
    d["i"], d["j"], d["cow"] = data["i"][sel], data["j"][sel], data["cow"][sel]
    return d


def fit_report(name, es, data, sel, rng):
    if len(sel) < 12:
        print(f"  {name:<22}{len(sel):>6}組  (少なすぎるので省略)")
        return None
    spec = es.Spec("const")
    dsub = sub_data(data, sel)
    th, nll = es.fit(spec, dsub, np.arange(len(sel)))
    boots = es.bootstrap(spec, dsub, th, rng, n_boot=N_BOOT)
    lo, hi = np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)
    print(f"  {name:<22}{len(sel):>6}組  σ_px {th[0]:5.1f}px [{lo[0]:.1f}, {hi[0]:.1f}]"
          f" / σ_h 立位 {th[1]:.2f} 横臥 {th[2]:.2f} / μ_h 立位 {th[3]:+.2f} 横臥 {th[4]:+.2f}")
    return th


def main():
    import estimate_sigma_he_kai_ver2 as es
    import phase3_bev_multicam_gt_zonepose_ver1 as g
    import analyze_projection_error_vs_tri_gt_ver3 as an
    import fit_assumed_height_vs_tri_gt_ver1 as fh
    from Mootrack_core import ROT_FMT, apply_camera_overrides, get_intrinsics, load_floorplan_cameras, resolve_poses

    in_dir = g.OUTPUT_DIR / "bev_gt_triangulated"
    frames = [int(a) for a in sys.argv[1:]] or sorted(int(p.name[:6]) for p in in_dir.glob("*_tri_gt_ver1.json"))
    cam_ids = list(g.CAM_IDS)
    poses = resolve_poses(apply_camera_overrides(load_floorplan_cameras(g.FLOORPLAN_JSON_PATH)), root="Area1", fmt=ROT_FMT)
    print(f"フレーム {frames} / 基準: 交差角 >= {an.MIN_ANGLE_DEG}°, 残差 <= {an.MAX_RESID_M} m")

    raw, reasons = [], Counter()
    for fr in frames:
        xp, tp = Path(an.XML_DIR) / f"{fr:06d}.xml", in_dir / f"{fr:06d}_tri_gt_ver1.json"
        if not (xp.exists() and tp.exists()):
            print(f"見つかりません: {xp if not xp.exists() else tp}")
            continue
        cows = json.loads(tp.read_text(encoding="utf-8"))["cows"]
        for c in cows:
            reasons[classify(c, an)[1]] += 1
        raw += build_raw_all(cows, an.parse_xml(xp), cam_ids, an, fr)

    print(f"\n■ 1. 牛の内訳 (三角測量GT の牛 {sum(reasons.values())}頭)")
    for k, v in reasons.most_common():
        print(f"  {k:<28}{v:>5}頭")
    n_used = len({(s['frame'], s['cow']) for s in raw})
    print(f"  bbox と姿勢が取れて、サンプルになった牛: {n_used}頭 ({len(raw)}サンプル)")

    es._N_CAM = len(cam_ids)
    data = es.prepare(raw, poses, get_intrinsics, fh, cam_ids)
    pass_pair = np.array([raw[a]["passed"] for a in data["i"]], dtype=bool)
    n_all = len(data["i"])
    print(f"  2カメラの組: 全部 {n_all} / 通過 {int(pass_pair.sum())} / 除かれた {int((~pass_pair).sum())}")
    if n_all < 20:
        print("組が少なすぎます")
        return

    rng = np.random.default_rng(0)
    spec = es.Spec("const")
    th_pass, _ = es.fit(spec, data, np.where(pass_pair)[0]) if pass_pair.sum() >= 12 else (None, None)

    print("\n■ 2. 食い違いの大きさ (通過した牛の推定値で偏り m を引いた |d - m| [m])")
    if th_pass is not None:
        d, _, _, _ = es.pair_stats(th_pass, spec, data, np.arange(n_all))
        mag = np.hypot(d[:, 0], d[:, 1])
        for name, sel in (("通過", pass_pair), ("除かれた", ~pass_pair)):
            if sel.sum() == 0:
                continue
            v = mag[sel]
            print(f"  {name:<10}{int(sel.sum()):>5}組  中央値 {np.median(v):.2f} / 90%点 {np.percentile(v, 90):.2f} / 最大 {v.max():.2f} m")

    print("\n■ 3. σ の推定 (const モデル。区間は牛単位ブートストラップ)")
    fit_report("通過した牛だけ (今まで)", es, data, np.where(pass_pair)[0], rng)
    fit_report("全部 (フィルタなし)", es, data, np.arange(n_all), rng)
    fit_report("除かれた牛だけ", es, data, np.where(~pass_pair)[0], rng)
    print("\n読み方: 『全部』の σ_px が『通過』より大きく増えれば、フィルタで σ が小さめに出ていた。")
    print("        ただし『全部』には ID の取り違え(アノテーションミス)も混ざっているので、増えた分がすべて検出の誤差とは限らない。")


if __name__ == "__main__":
    main()