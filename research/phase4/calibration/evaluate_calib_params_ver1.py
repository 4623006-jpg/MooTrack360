"""
evaluate_calib_params_ver1.py

パラメータ(estimate_calib_*.py が出力したjson)を入れて、LM誤差と牛の誤差を出力する。
学習に使ったランドマーク/牛と、使っていないものを区別して表示する。

  [A] 最終パラメータ : LM誤差[m] = 学習に使った点 / 使っていない点
                       牛の投影誤差[px] = 学習に使った牛 / 使っていない牛
  [B] CVの各分割     : 各分割のパラメータで、その分割が外した点(ランドマーク/牛)だけを測り、まとめた値
                       (全点が学習に使われた最終パラメータでも、未使用データでの精度が分かる)
  "core" を渡すと、Mootrack_core.py の現在値を評価する(学習なし: 全点が未使用)。

使い方:
    python evaluate_calib_params_ver1.py core calib_edge_cows_ver1.json calib_lm_only_ver1.json calib_cows_only_ver1.json
    (jsonは bev_calibration_refined/ からの相対名、またはフルパス。数字は牛のフレーム番号 = 評価する牛。省略時は保存済みの全フレーム)
出力: bev_calibration_refined/evaluate_calib_params_ver1.csv (ランドマークごとの誤差)

配置: research/phase4/calibration/ (calib_common_ver1.py と同じ場所)
"""
import csv
import json
import os
import sys

import numpy as np

import calib_common_ver1 as cc


def stats_lm(e):
    e = np.asarray(e, dtype=float)
    e = e[~np.isnan(e)]
    if not len(e):
        return "n=0"
    return f"n={len(e)} RMSE {np.sqrt(np.mean(e ** 2)):.3f} / 平均 {np.mean(e):.3f} / 中央値 {np.median(e):.3f} / 最大 {np.max(e):.3f}"


def stats_cow(a):
    a = np.asarray(a, dtype=float)
    if not len(a):
        return "n=0"
    return f"n={len(a)} 中央値 {np.median(a):.2f} / 平均 {np.mean(a):.2f} / 90%点 {np.percentile(a, 90):.2f} / 最大 {np.max(a):.2f}"


def evaluate_params(doc, landmarks, cows):
    """doc: json(dict)。戻り値: 評価結果のdict"""
    cams = cc.json_to_cams(doc["cameras"])
    used_l, used_c = set(doc["train_landmarks"]), set(doc["train_cows"])
    ids_l = [cc.lm_id(lm) for lm in landmarks]
    ids_c = [cc.cow_id(c) for c in cows]
    e_lm = cc.b2.landmark_errors(cams, landmarks)
    m_used = np.array([i in used_l for i in ids_l])
    cow_used = [c for c, i in zip(cows, ids_c) if i in used_c]
    cow_unused = [c for c, i in zip(cows, ids_c) if i not in used_c]
    res = dict(e_lm=e_lm, lm_used=m_used,
               cow_used=cc.b1.evaluate_cows(cams, cow_used)["reproj_px"] if cow_used else np.array([]),
               cow_unused=cc.b1.evaluate_cows(cams, cow_unused)["reproj_px"] if cow_unused else np.array([]))
    # CV: 各分割のパラメータで、その分割が外した点だけを測る
    idx_l = {i: k for k, i in enumerate(ids_l)}
    lm_sum, lm_cnt, cow_h = np.zeros(len(landmarks)), np.zeros(len(landmarks)), []
    for f in doc.get("cv_folds", []):
        cams_f = cc.json_to_cams(f["cameras"])
        ks = [idx_l[i] for i in f["heldout_landmarks"] if i in idx_l]
        if ks:
            lm_sum[ks] += cc.b2.landmark_errors(cams_f, [landmarks[k] for k in ks])
            lm_cnt[ks] += 1
        hc = set(f["heldout_cows"])
        cs = [c for c, i in zip(cows, ids_c) if i in hc]
        if cs:
            cow_h.append(cc.b1.evaluate_cows(cams_f, cs)["reproj_px"])
    res["cv_lm"] = np.where(lm_cnt > 0, lm_sum / np.maximum(lm_cnt, 1), np.nan)
    res["cv_cow"] = np.concatenate(cow_h) if cow_h else np.array([])
    return res


def doc_for_core(S):
    return dict(kind="core", setting={"name": f"Mootrack_core現在値 ({S['init_set']})"}, cameras=cc.cams_to_json(S["base"]),
                train_landmarks=[], train_cows=[], cv_folds=[], init_calibration_set=S["init_set"])


def main():
    args = sys.argv[1:]
    frames = [int(a) for a in args if a.isdigit()] or None
    targets = [a for a in args if not a.isdigit()] or ["core"]
    S = cc.setup(frames)
    landmarks, cows, out_dir = S["landmarks"], S["cows"], S["out_dir"]
    cam_of = np.array([lm["cam"] for lm in landmarks])
    print(f"評価データ: ランドマーク {len(landmarks)}点 / 牛 {len(cows)}頭 (フレーム {S['frames']})")

    results = []
    for t in targets:
        if t == "core":
            doc = doc_for_core(S)
        else:
            p = t if os.path.isabs(t) or os.path.exists(t) else str(out_dir / t)
            doc = json.loads(open(p, encoding="utf-8").read())
        r = evaluate_params(doc, landmarks, cows)
        results.append((t, doc, r))
        m = r["lm_used"]
        print(f"\n{'=' * 78}\n■ {t}   設定: {doc['setting']['name']}   出発点: {doc.get('init_calibration_set')}")
        print(f"  学習に使った: LM {int(m.sum())}点 / 牛 {len(doc['train_cows'])}頭")
        print("  [A] 最終パラメータ")
        print(f"    LM誤差[m]  学習に使った点     : {stats_lm(r['e_lm'][m])}")
        print(f"               使っていない点     : {stats_lm(r['e_lm'][~m])}")
        print(f"    牛[px]     学習に使った牛     : {stats_cow(r['cow_used'])}")
        print(f"               使っていない牛     : {stats_cow(r['cow_unused'])}")
        if doc.get("cv_folds"):
            print("  [B] CVの各分割のパラメータで、外した点だけを測った値 (未使用データでの精度)")
            print(f"    LM誤差[m]  外したLM            : {stats_lm(r['cv_lm'])}")
            print(f"    牛[px]     外した牛            : {stats_cow(r['cv_cow'])}")
        print("    カメラ別LM RMSE[m] (最終 / CVで外したとき): " + "  ".join(
            f"{c}: {np.sqrt(np.mean(r['e_lm'][cam_of == c] ** 2)):.3f} / "
            + (f"{np.sqrt(np.nanmean(r['cv_lm'][cam_of == c] ** 2)):.3f}" if np.any(~np.isnan(r['cv_lm'][cam_of == c])) else "-")
            for c in S["cam_ids"]))

    if len(results) > 1:
        print(f"\n{'=' * 78}\n■ 比較表 (未使用データでの値を中心に)")
        print(f"{'':<14}{'LM(使っていない点)':>20}{'LM(CVで外した)':>16}{'牛(使っていない)':>18}{'牛(CVで外した)':>16}{'LM(学習)':>10}")
        for t, doc, r in results:
            m = r["lm_used"]
            def f_l(a):
                a = np.asarray(a, dtype=float); a = a[~np.isnan(a)]
                return f"{np.sqrt(np.mean(a ** 2)):.3f}" if len(a) else "-"
            def f_c(a):
                return f"{np.median(a):.2f}" if len(a) else "-"
            print(f"{os.path.basename(t)[:13]:<14}{f_l(r['e_lm'][~m]):>20}{f_l(r['cv_lm']):>16}{f_c(r['cow_unused']):>18}"
                  f"{f_c(r['cv_cow']):>16}{f_l(r['e_lm'][m]):>10}")
        print("  (LMはm, 牛はpxの中央値)")

    path = out_dir / "evaluate_calib_params_ver1.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["landmark", "label"] + [f"{os.path.basename(t)}:{k}" for t, _d, _r in results for k in ("used", "err_m", "cv_err_m")])
        for i, lm in enumerate(landmarks):
            row = [cc.lm_id(lm), lm["label"]]
            for _t, _d, r in results:
                row += [int(r["lm_used"][i]), f"{r['e_lm'][i]:.4f}",
                        "" if np.isnan(r["cv_lm"][i]) else f"{r['cv_lm'][i]:.4f}"]
            w.writerow(row)
    print(f"\nCSV: {path}")


if __name__ == "__main__":
    main()