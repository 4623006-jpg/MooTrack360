"""
gt_ignore_ver1.py

『未検出の原因』に分類したGTのミス(アノテーションミス確認リスト)を、評価のGTから除く。
FNとして数えない(GTにも含めない)ようにするための共通部品。

除外リスト: gt_ignore_fn_ver1.csv (列: frame, cow, camera[, note])
   frame  : フレーム番号(例 1490)   cow : 牛の番号(例 50 = cow050)   camera : カメラ番号(1〜3)
   → 除外の範囲は HE_GT_IGNORE_SCOPE で選ぶ(既定 label):
        label     : 牛ごとに、リストにある最初のフレーム以降の、その牛のBBOXをすべて(全カメラ)除く。
                    (そのラベルは、最初にミスが見つかった時点から信用できないものとして扱う)
        label_cam : (牛, カメラ)ごとに、リストにある最初のフレーム以降の、そのカメラのその牛のBBOXを除く
        exact     : リストの行(フレーム・牛・カメラが一致するBBOX)だけを除く
   リストの牛より前のフレームのBBOXや、リストに無い牛のBBOXは、どの設定でも残る。
   新しくラベルのミスが見つかったら、CSVに1行(最初に見つかったフレームの行)を足す。

使い方(評価・診断スクリプトの中で):
    import gt_ignore_ver1 as gi
    st = gi.apply(ym)        # ym = phase3モジュール。ym.parse_voc_annotation を差し替える
    ...
    gi.report(st)            # 最後に、適用した件数と、該当が無かった行を表示
除外リストの場所: 環境変数 HE_GT_IGNORE(パス。0 なら無効) > 次のフォルダの gt_ignore_fn_ver1.csv
    このファイルのフォルダ / カレント / phase4/diagnose / phase4/he_kai / phase4/calibration
"""

import csv
import os
import re
from collections import Counter
from pathlib import Path

CSV_NAME = "gt_ignore_fn_ver1.csv"
SCOPE = os.environ.get("HE_GT_IGNORE_SCOPE", "label")      # label / label_cam / exact


def find_csv():
    env = os.environ.get("HE_GT_IGNORE")
    if env == "0":
        return None
    if env:
        return env if os.path.exists(env) else None
    here = os.path.dirname(os.path.abspath(__file__))
    phase4 = os.path.abspath(os.path.join(here, ".."))
    for d in (here, os.getcwd(), os.path.join(phase4, "diagnose"), os.path.join(phase4, "he_kai"), os.path.join(phase4, "calibration")):
        p = os.path.join(d, CSV_NAME)
        if os.path.exists(p):
            return p
    return None


def load(path):
    ignore = set()
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ignore.add((int(row["frame"]), int(row["cow"]), int(row["camera"])))
    return ignore


def apply(ym, num_cams=3):
    """ym.parse_voc_annotation を、除外リストのBBOXを取り除く版に差し替える。戻り値は集計用の辞書(無ければ None)"""
    path = find_csv()
    if path is None:
        print("GT除外リスト: なし(評価のGTはそのまま)")
        return None
    if getattr(ym.parse_voc_annotation, "_gt_ignore", None) is not None:
        return ym.parse_voc_annotation._gt_ignore
    if SCOPE not in ("label", "label_cam", "exact"):
        raise ValueError(f"HE_GT_IGNORE_SCOPE は label / label_cam / exact のどれか: {SCOPE}")
    ignore = load(path)
    cutoff, cutoff_cam = {}, {}                       # 牛 -> 最初のフレーム / (牛, カメラ) -> 最初のフレーム
    for f, c, k in ignore:
        cutoff[c] = min(f, cutoff.get(c, f))
        cutoff_cam[(c, k)] = min(f, cutoff_cam.get((c, k), f))
    orig = ym.parse_voc_annotation
    tile_w = ym.IMAGE_WIDTH_PX / num_cams
    st = {"path": path, "ignore": ignore, "hit": set(), "removed": 0, "by_cow": Counter(),
          "cutoff": cutoff, "cutoff_cam": cutoff_cam, "scope": SCOPE}

    def wrapped(xml_path):
        objs = orig(xml_path)
        try:
            frame = int(Path(xml_path).stem)
        except ValueError:
            return objs
        out = []
        for o in objs:
            cam = max(0, min(int((o["xmin"] + o["xmax"]) / 2.0 // tile_w), num_cams - 1)) + 1
            m = re.search(r"\d+", str(o["name"]))
            cow = int(m.group()) if m else -1
            key = (frame, cow, cam)
            drop = False
            if key in ignore:
                st["hit"].add(key)
                drop = True
            elif SCOPE == "label" and cow in cutoff and frame >= cutoff[cow]:
                drop = True
            elif SCOPE == "label_cam" and (cow, cam) in cutoff_cam and frame >= cutoff_cam[(cow, cam)]:
                drop = True
            if drop:
                st["removed"] += 1
                st["by_cow"][cow] += 1
                continue
            out.append(o)
        return out

    wrapped._gt_ignore = st
    ym.parse_voc_annotation = wrapped
    print(f"GT除外リスト: {path} ({len(ignore)}行) / 範囲={SCOPE}")
    if SCOPE == "label":
        print("  牛ごとの除外開始フレーム(以降、その牛のBBOXを全カメラで評価から除く): "
              + ", ".join(f"cow{c:03d}:{f}~" for c, f in sorted(cutoff.items())))
    elif SCOPE == "label_cam":
        print("  (牛,カメラ)ごとの除外開始フレーム: "
              + ", ".join(f"cow{c:03d}/cam{k}:{f}~" for (c, k), f in sorted(cutoff_cam.items())))
    return st


def report(st):
    if not st:
        return
    unmatched = sorted(st["ignore"] - st["hit"])
    print(f"\nGT除外(範囲={st['scope']}): 延べ{st['removed']}個のBBOXを評価から除きました。"
          f"リストの行のうち実際にBBOXがあったのは {len(st['hit'])}/{len(st['ignore'])}行")
    if st["by_cow"]:
        print("  牛ごとの除外数: " + ", ".join(f"cow{c:03d}:{n}" for c, n in sorted(st["by_cow"].items())))
    if unmatched:
        print(f"  該当が無かった行({len(unmatched)}行。処理したフレームの範囲外か、そのフレームにそのBBOXが無い):")
        print("   " + ", ".join(f"({f},cow{c:03d},cam{k})" for f, c, k in unmatched))