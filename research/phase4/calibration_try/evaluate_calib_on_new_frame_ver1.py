"""
evaluate_calib_on_new_frame_ver1.py

調整に使っていない別フレームの牛(三角測量GTツールで付けたクリック)で、
現在の校正と新しい校正(calibration_refined_cv_ver6.json)を比べる。

  [1] 再投影誤差(牛ごと, 擬似px) : 複数カメラのクリックが三角測量とどれだけ整合するか
  [2] 転送誤差(牛ごと, 擬似px)   : 3カメラに写る牛で、2カメラから三角測量 -> 残り1カメラのクリット方向と比べる
                                  (使っていないカメラのクリックを予測できるか。いちばん厳しい指標)
  [3] 三角測量位置の差[m]       : 新旧の校正で、同じ牛の位置がどれだけ動くか
  牛は新旧とも同じ集合で比べる(対応のある比較)。P(新 < 現在) は牛の復元抽出(ブートストラップ)で出す。

使い方:
    python evaluate_calib_on_new_frame_ver1.py 600 700     # フレーム番号(調整に使ったフレーム100は不可)
  事前に annotate_triangulate_gt_web_ver5.py で、そのフレームの牛をクリットして保存しておく
  (bev_gt_triangulated/{frame:06d}_tri_gt_ver1.json)。

配置: research/phase4/calibration/
"""

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
          MOOTRACK_CODE_DIR,
          os.path.join(RESEARCH_DIR, "analyze"), os.path.join(RESEARCH_DIR, "annotate"),
          os.path.join(RESEARCH_DIR, "phase3_multi_camera"), THIS_DIR):
    if os.path.isdir(d) and d not in sys.path:
        sys.path.insert(0, d)

import refine_calibration_ba_ver1 as b1

REFINED_JSON_NAME = "calibration_refined_cv_ver6.json"
MIN_ANGLE_DEG = 30.0
N_BOOT = 5000
SEED = 0


def cow_metrics(cams, cow):
    """1頭の (平均再投影誤差px, 転送誤差px(3カメラのとき), 三角測量位置, 交差角) 。失敗は None"""
    X, rays = b1.triangulate_cow(cams, cow["clicks"])
    if X is None:
        return None
    F = {cid: 0.5 * (cams[cid][2][0] + cams[cid][2][1]) for cid, _c, _d in rays}
    rp = float(np.mean([F[cid] * np.linalg.norm(d - (X - c) / np.linalg.norm(X - c)) for cid, c, d in rays]))
    ang = max(float(np.degrees(np.arccos(np.clip(np.dot(rays[i][2], rays[j][2]), -1, 1))))
              for i in range(len(rays)) for j in range(i + 1, len(rays)))
    tr = None
    if len(rays) >= 3:
        errs = {}
        for h in range(len(rays)):
            others = {rays[k][0]: cow["clicks"][rays[k][0]] for k in range(len(rays)) if k != h}
            Xh, _r = b1.triangulate_cow(cams, others)
            if Xh is None:
                return None
            cid, c, d = rays[h]
            errs[cid] = F[cid] * float(np.linalg.norm(d - (Xh - c) / np.linalg.norm(Xh - c)))
        tr = errs
    return {"reproj": rp, "transfer": tr, "X": X, "angle": ang}


def boot_p_lower(a, b, rng, stat=np.median, n=N_BOOT):
    """牛を復元抽出して、stat(a) < stat(b) となる割合 (a=新, b=現在)"""
    a, b = np.asarray(a), np.asarray(b)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    return float(np.mean(stat(a[idx], axis=1) < stat(b[idx], axis=1)))


def compare(cams_cur, cams_new, cows, rng):
    rows = []
    for cow in cows:
        mc, mn = cow_metrics(cams_cur, cow), cow_metrics(cams_new, cow)
        if mc is None or mn is None or mc["angle"] < MIN_ANGLE_DEG:
            continue
        rows.append((cow, mc, mn))
    out = {"n": len(rows)}
    if not rows:
        return out
    rc = np.array([m["reproj"] for _c, m, _n in rows])
    rn = np.array([n["reproj"] for _c, _m, n in rows])
    out["reproj"] = dict(cur=rc, new=rn, p=boot_p_lower(rn, rc, rng), win=float(np.mean(rn < rc)))
    three = [(c, m, n) for c, m, n in rows if m["transfer"] is not None and n["transfer"] is not None]
    if three:
        tc = np.array([np.mean(list(m["transfer"].values())) for _c, m, _n in three])
        tn = np.array([np.mean(list(n["transfer"].values())) for _c, _m, n in three])
        by_cam = {}
        for _c, m, n in three:
            for cid in m["transfer"]:
                by_cam.setdefault(cid, ([], []))
                by_cam[cid][0].append(m["transfer"][cid])
                by_cam[cid][1].append(n["transfer"][cid])
        out["transfer"] = dict(cur=tc, new=tn, p=boot_p_lower(tn, tc, rng), win=float(np.mean(tn < tc)), by_cam=by_cam)
    out["shift_m"] = np.array([np.linalg.norm(n["X"][:2] - m["X"][:2]) for _c, m, n in rows])
    return out


def show(title, cur, new, p, win):
    print(f"  {title:<26} 現在 中央値 {np.median(cur):6.2f} 平均 {np.mean(cur):6.2f} / 新 中央値 {np.median(new):6.2f} "
          f"平均 {np.mean(new):6.2f} / P(新<現在) {p:4.0%} / 牛ごとの勝率 {win:4.0%}")


def main():
    import phase3_bev_multicam_gt_zonepose_ver1 as g
    from Mootrack_core import apply_camera_overrides, get_intrinsics, load_floorplan, resolve_poses

    out_dir = g.OUTPUT_DIR / "bev_calibration_refined"
    in_dir = g.OUTPUT_DIR / "bev_gt_triangulated"
    if len(sys.argv) < 2:
        raise SystemExit("使い方: python evaluate_calib_on_new_frame_ver1.py 600 700  (調整に使っていないフレーム)")
    frames = [int(a) for a in sys.argv[1:]]
    cal = json.loads((out_dir / REFINED_JSON_NAME).read_text(encoding="utf-8"))
    if set(frames) & set(cal.get("frames", [])):
        print(f"警告: 調整に使ったフレーム {sorted(set(frames) & set(cal['frames']))} が含まれています。"
              "これらの牛は学習済みなので、公平な検証になりません。")

    regions, cameras_json, _w, _h = load_floorplan(g.FLOORPLAN_JSON_PATH)
    poses = resolve_poses(apply_camera_overrides(cameras_json), root="Area1", fmt=g.ROT_FMT)
    cam_ids = list(g.CAM_IDS)
    cur = {c: (poses[c].R.copy(), poses[c].t.copy(), np.array(get_intrinsics(c), dtype=np.float64)) for c in cam_ids}
    new = {c: (np.array(cal["cameras"][c]["R"]), np.array(cal["cameras"][c]["t"]),
               np.array([cal["cameras"][c]["intrinsics"][nm] for nm in b1.INTR_NAMES], dtype=np.float64)) for c in cam_ids}

    cows = []
    for fr in frames:
        p = in_dir / f"{fr:06d}_tri_gt_ver1.json"
        if not p.exists():
            print(f"見つかりません(先にクリックして保存): {p}")
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        for c in data["cows"]:
            clicks = {k: v for k, v in c.get("clicks", {}).items() if v is not None and k in cam_ids}
            if len(clicks) >= 2:
                cows.append({"frame": fr, "id": c["id"], "clicks": clicks})
    print(f"フレーム {frames}: 牛 {len(cows)}頭 (2カメラ以上でクリット済み)")
    if not cows:
        raise SystemExit("牛がありません")

    res = compare(cur, new, cows, np.random.default_rng(SEED))
    print(f"比較に使えた牛(両方で三角測量成功, 交差角>={MIN_ANGLE_DEG:.0f}°): {res['n']}頭")
    if res["n"] < 5:
        raise SystemExit("牛が少なすぎます。もっとクリックしてください")
    print("\n=== 調整に使っていないフレームでの比較(擬似px, 小さいほど良い) ===")
    r = res["reproj"]
    show("[1] 再投影誤差", r["cur"], r["new"], r["p"], r["win"])
    if "transfer" in res:
        t = res["transfer"]
        print(f"      (3カメラに写る牛 {len(t['cur'])}頭)")
        show("[2] 転送誤差", t["cur"], t["new"], t["p"], t["win"])
        for cid, (a, b) in sorted(t["by_cam"].items()):
            print(f"        予測先 {cid}: 現在 中央値 {np.median(a):6.2f} -> 新 {np.median(b):6.2f} (n={len(a)})")
    else:
        print("  [2] 転送誤差: 3カメラに写る牛がありません")
    s = res["shift_m"]
    print(f"  [3] 三角測量位置の差(新旧): 中央値 {np.median(s):.3f} m / 90%点 {np.percentile(s, 90):.3f} m / 最大 {s.max():.3f} m")

    print("\n--- 判定 ---")
    ok1 = r["p"] >= 0.8 and np.median(r["new"]) <= np.median(r["cur"])
    ok2 = ("transfer" in res) and res["transfer"]["p"] >= 0.8 and \
        np.median(res["transfer"]["new"]) <= np.median(res["transfer"]["cur"])
    if ok1 and (ok2 or "transfer" not in res):
        print("新しい校正が、調整に使っていないフレームでも整合の面で良くなっています。採用してよいと考えられます。")
    elif np.median(r["new"]) > np.median(r["cur"]) * 1.05 or ("transfer" in res and
                                                              np.median(res["transfer"]["new"]) > np.median(res["transfer"]["cur"]) * 1.05):
        print("新しい校正は、別フレームで悪化しています。採用しないでください。")
    else:
        print("差が小さいか、不確かです(P < 80%)。牛を増やして再評価するか、現在の校正のままにしてください。")


if __name__ == "__main__":
    main()