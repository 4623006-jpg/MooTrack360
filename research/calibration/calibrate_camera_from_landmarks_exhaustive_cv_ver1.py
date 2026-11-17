"""
calibrate_camera_from_landmarks_exhaustive_cv_ver1.py

leave-one-out(1点ずつ除く)ではなく、ランドマークの「部分集合の全組み合わせ」で
較正→残りの点で検証、を総当たりするクロスバリデーション。

各カメラのランドマークn点に対して、学習用の部分集合Sを
(MIN_TRAIN_POINTS 〜 n - MIN_TEST_POINTS)点の全組み合わせで列挙し、
  1. Sだけで較正(モード3: 回転+位置+内部パラメータ。LOOCV版と同じfit_full)
  2. Sに含まれない点(テスト点)での世界座標誤差を記録
を繰り返す。LOOCVは「1点を除く」n通りだけだったが、これは較正に使う点の
組み合わせを網羅するので、特定の1点・1通りの除外に結果が左右されにくい。

出力(カメラごと):
  1. 学習点数ごとのテスト誤差の推移(点を増やすと誤差がどこまで下がるか)
  2. テスト誤差が小さかった部分集合の上位(=良い較正になった点の組み合わせ)
  3. ランドマーク別の指標
       - held-out平均: その点がテストに回ったときの平均誤差(多数の組み合わせの平均)
       - 含めた効果: 学習点数を揃えた上で、その点を学習に「含めた」場合と
         「含めない」場合の、他のテスト点の平均誤差の差。
         マイナス=含めると他の点の誤差が減る(有用)、
         プラス=含めると他の点の誤差が増える(外す候補)
  4. CSV(ランドマーク別指標、上位部分集合)

★計算量★ 組み合わせ数が多い(例: 17点だと約13万通り)ので時間がかかる。
multiprocessingで並列化している。先頭で数十回の較正を試して所要時間を見積もり、
表示する。MAX_SUBSETS_PER_CAMERAに数値を入れると、その数に間引く(ランダム抽出)。

★注意★ 「上位の部分集合」は、テスト誤差が小さい組み合わせを選んでいるので、
その誤差値は楽観的(選択バイアスあり)。絶対値としてではなく、どの点が
入っている/いない組み合わせが良かったかの傾向を見るのに使うこと。

--- 使い方 ---
    python calibrate_camera_from_landmarks_exhaustive_cv_ver1.py
(LANDMARKS等はcalibrate_camera_from_landmarks_loocv_ver1.pyからimportするので、
同じフォルダに置くこと。ランドマークを更新するときはそちらを更新する)
"""

import csv
import itertools
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

MOOTRACK_CODE_DIR = os.path.join(THIS_DIR, "..", "calibration")
if MOOTRACK_CODE_DIR not in sys.path:
    sys.path.insert(0, MOOTRACK_CODE_DIR)

from Mootrack_core import ROT_FMT, get_intrinsics, build_R, load_floorplan

from calibrate_camera_from_landmarks_loocv_ver1 import (  # noqa: E402
    LANDMARKS, CAMERA_CHAIN, FLOORPLAN_JSON_PATH,
    find_region_vertex, group_landmarks_by_camera, fit_full, point_error,
)

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------

MIN_TRAIN_POINTS = 6      # 較正に使う最小点数(8パラメータに対して少なすぎると不安定)
MIN_TEST_POINTS = 2       # テストに残す最小点数
MAX_SUBSETS_PER_CAMERA: Optional[int] = None  # Noneなら全組み合わせ。数値ならランダムに間引く
RANDOM_SEED = 0
N_PROCESSES = max(1, (os.cpu_count() or 2) - 1)
TOP_K_SUBSETS = 10

OUTPUT_DIR = Path(THIS_DIR)

# ---------------------------------------------------------------------------
# ワーカー用のグローバル(Pool initializerで設定)
# ---------------------------------------------------------------------------
_G: Dict[str, object] = {}


def _init_worker(parent_t, pts_px, world_gt, init_params6, init_xi, init_alpha):
    _G["parent_t"] = parent_t
    _G["pts_px"] = pts_px
    _G["world_gt"] = world_gt
    _G["init_params6"] = init_params6
    _G["init_xi"] = init_xi
    _G["init_alpha"] = init_alpha


def _eval_subset(train_idx: Tuple[int, ...]):
    """train_idxで較正し、全点の世界座標誤差(長さn)を返す。失敗時はNone。"""
    pts_px = _G["pts_px"]
    world_gt = _G["world_gt"]
    parent_t = _G["parent_t"]
    n = len(pts_px)
    idx = np.array(train_idx, dtype=int)
    try:
        params6, xi, alpha = fit_full(
            parent_t, pts_px[idx], world_gt[idx],
            _G["init_params6"], _G["init_xi"], _G["init_alpha"],
        )
    except Exception:
        return train_idx, None
    errs = np.array([
        point_error(params6, parent_t, pts_px[i], world_gt[i], xi, alpha) for i in range(n)
    ])
    return train_idx, errs


def enumerate_subsets(n: int) -> List[Tuple[int, ...]]:
    subsets: List[Tuple[int, ...]] = []
    for k in range(MIN_TRAIN_POINTS, n - MIN_TEST_POINTS + 1):
        subsets.extend(itertools.combinations(range(n), k))
    return subsets


def count_subsets(n: int) -> int:
    from math import comb
    return sum(comb(n, k) for k in range(MIN_TRAIN_POINTS, n - MIN_TEST_POINTS + 1))


def main():
    print("floorplan.json 読み込み中...")
    regions, cameras_json_raw, real_w, real_h = load_floorplan(FLOORPLAN_JSON_PATH)
    raw_params_by_key = {
        f"{e['parent']}->{e['child']}": np.array(e["params"], dtype=np.float64)
        for e in cameras_json_raw
    }
    landmarks_by_cam = group_landmarks_by_camera(LANDMARKS)

    # --- 1. チェーンに沿ってフル較正し、各カメラの親位置parent_tを確定(LOOCV版と同じ) ---
    poses_fitted: Dict[str, Tuple[np.ndarray, np.ndarray]] = {"Area1": (np.eye(3), np.zeros(3))}
    cam_info: Dict[str, dict] = {}

    for parent_id, cam_id in CAMERA_CHAIN:
        parent_R, parent_t = poses_fitted[parent_id]
        key = f"{parent_id}->{cam_id}"
        init_params6 = raw_params_by_key[key]
        _, _, _, _, init_xi, init_alpha = get_intrinsics(cam_id)

        if cam_id not in landmarks_by_cam:
            R = build_R(*init_params6[3:], ROT_FMT)
            poses_fitted[cam_id] = (R, parent_t + init_params6[:3])
            continue

        lms = landmarks_by_cam[cam_id]
        pts_px = np.array([lm["pixel"] for lm in lms], dtype=np.float64)
        world_gt = np.array(
            [find_region_vertex(regions, lm["region"], lm["vertex_idx"]) for lm in lms], dtype=np.float64,
        )
        params6, xi, alpha = fit_full(parent_t, pts_px, world_gt, init_params6, init_xi, init_alpha)
        R = build_R(*params6[3:], ROT_FMT)
        poses_fitted[cam_id] = (R, parent_t + params6[:3])
        cam_info[cam_id] = {
            "parent_t": parent_t, "landmarks": lms, "pts_px": pts_px, "world_gt": world_gt,
            "init_params6": init_params6, "init_xi": init_xi, "init_alpha": init_alpha,
        }

    # --- 2. カメラごとに全部分集合を総当たり ---
    for cam_id, info in cam_info.items():
        lms = info["landmarks"]
        n = len(lms)
        labels = [lm["label"] for lm in lms]

        print(f"\n{'=' * 70}")
        print(f"■ {cam_id}  ランドマーク{n}点  部分集合(学習{MIN_TRAIN_POINTS}〜{n - MIN_TEST_POINTS}点): "
              f"全{count_subsets(n):,}通り")
        print(f"{'=' * 70}")

        if n < MIN_TRAIN_POINTS + MIN_TEST_POINTS:
            print("  ランドマーク数が少なすぎるためスキップ")
            continue

        subsets = enumerate_subsets(n)
        if MAX_SUBSETS_PER_CAMERA is not None and len(subsets) > MAX_SUBSETS_PER_CAMERA:
            rng = np.random.default_rng(RANDOM_SEED)
            pick = rng.choice(len(subsets), size=MAX_SUBSETS_PER_CAMERA, replace=False)
            subsets = [subsets[i] for i in sorted(pick)]
            print(f"  → {MAX_SUBSETS_PER_CAMERA:,}通りにランダム間引き")

        init_args = (info["parent_t"], info["pts_px"], info["world_gt"],
                     info["init_params6"], info["init_xi"], info["init_alpha"])

        # 所要時間の見積もり(先頭30回を1プロセスで実測)
        _init_worker(*init_args)
        t0 = time.time()
        pilot = min(30, len(subsets))
        for s in subsets[:pilot]:
            _eval_subset(s)
        per_fit = (time.time() - t0) / max(1, pilot)
        est_sec = per_fit * len(subsets) / N_PROCESSES
        print(f"  1回の較正: 約{per_fit * 1000:.0f}ms → 全{len(subsets):,}通りを{N_PROCESSES}並列で"
              f"約{est_sec / 60:.1f}分の見込み")

        # 並列実行
        rows_mask = []
        rows_err = []
        t_start = time.time()
        with mp.Pool(N_PROCESSES, initializer=_init_worker, initargs=init_args) as pool:
            done = 0
            for train_idx, errs in pool.imap_unordered(_eval_subset, subsets, chunksize=64):
                done += 1
                if errs is not None:
                    m = np.zeros(n, dtype=bool)
                    m[list(train_idx)] = True
                    rows_mask.append(m)
                    rows_err.append(errs)
                if done % 5000 == 0:
                    print(f"  {done:,}/{len(subsets):,} 完了 ({time.time() - t_start:.0f}秒経過)")
        print(f"  完了: 成功{len(rows_mask):,}/{len(subsets):,}通り ({time.time() - t_start:.0f}秒)")

        M = np.array(rows_mask)                      # (S, n) True=学習に使用
        E = np.array(rows_err)                       # (S, n) 全点の誤差
        E_test = np.where(M, np.nan, E)              # テスト点のみ(学習点はnan)
        E_train = np.where(M, E, np.nan)
        sizes = M.sum(axis=1)

        with np.errstate(all="ignore"):
            test_rmse = np.sqrt(np.nanmean(E_test ** 2, axis=1))
            train_rmse = np.sqrt(np.nanmean(E_train ** 2, axis=1))
            test_mean = np.nanmean(E_test, axis=1)

        # (1) 学習点数ごとのテスト誤差
        print("\n  [学習点数ごとのテスト誤差(テスト点の平均誤差[m])]")
        print(f"  {'学習点数':>8}{'組合せ数':>10}{'平均':>10}{'中央値':>10}{'最小':>10}{'in-sample RMSE平均':>20}")
        for k in sorted(set(sizes.tolist())):
            sel = sizes == k
            print(f"  {k:>8}{int(sel.sum()):>10}{np.nanmean(test_mean[sel]):>10.3f}"
                  f"{np.nanmedian(test_mean[sel]):>10.3f}{np.nanmin(test_mean[sel]):>10.3f}"
                  f"{np.nanmean(train_rmse[sel]):>20.3f}")

        # (2) テスト誤差(RMSE)が小さかった部分集合
        order = np.argsort(np.nan_to_num(test_rmse, nan=1e9))
        print(f"\n  [テストRMSEが小さかった学習用部分集合 上位{TOP_K_SUBSETS}]"
              f"(選択バイアスあり: 傾向を見る用)")
        top_rows = []
        for rank, si in enumerate(order[:TOP_K_SUBSETS], start=1):
            used = [labels[i] for i in range(n) if M[si, i]]
            print(f"  {rank:>2}. 学習{int(sizes[si])}点  test RMSE={test_rmse[si]:.3f}m  "
                  f"train RMSE={train_rmse[si]:.3f}m")
            print(f"      使用: {', '.join(used)}")
            top_rows.append({"rank": rank, "n_train": int(sizes[si]), "test_rmse_m": float(test_rmse[si]),
                             "train_rmse_m": float(train_rmse[si]), "train_landmarks": " | ".join(used)})

        # (3) ランドマーク別指標
        stats = []
        for i in range(n):
            col = E_test[:, i]
            held_out_mean = float(np.nanmean(col)) if np.any(~np.isnan(col)) else float("nan")
            held_out_med = float(np.nanmedian(col)) if np.any(~np.isnan(col)) else float("nan")

            diffs = []
            for k in sorted(set(sizes.tolist())):
                with_rows = (sizes == k) & M[:, i]
                wo_rows = (sizes == k) & ~M[:, i]
                if not with_rows.any() or not wo_rows.any():
                    continue
                with np.errstate(all="ignore"):
                    # iを学習に含む: テスト点は全てi以外 → テスト点の平均誤差
                    a = np.nanmean(E_test[with_rows], axis=1)
                    # iを含まない: iはテストに回るので、i以外のテスト点の平均誤差で比べる
                    E_wo = E_test[wo_rows].copy()
                    E_wo[:, i] = np.nan
                    b = np.nanmean(E_wo, axis=1)
                a = a[~np.isnan(a)]
                b = b[~np.isnan(b)]
                if len(a) and len(b):
                    diffs.append(a.mean() - b.mean())
            include_effect = float(np.mean(diffs)) if diffs else float("nan")
            stats.append({"label": labels[i], "held_out_mean_m": held_out_mean,
                          "held_out_median_m": held_out_med, "include_effect_m": include_effect})

        print("\n  [ランドマーク別指標]")
        print("   held-out平均: テストに回ったときの平均誤差(大きい=予測しにくい点)")
        print("   含めた効果  : 学習点数を揃えて、その点を含めた時-含めない時の他点の平均誤差の差"
              "(マイナス=有用、プラス=外す候補)")
        print(f"  {'label':<28}{'held-out平均[m]':>16}{'held-out中央値[m]':>18}{'含めた効果[m]':>16}")
        for s in sorted(stats, key=lambda d: (np.isnan(d["include_effect_m"]), -d["include_effect_m"]
                                              if not np.isnan(d["include_effect_m"]) else 0)):
            print(f"  {s['label']:<28}{s['held_out_mean_m']:>16.3f}{s['held_out_median_m']:>18.3f}"
                  f"{s['include_effect_m']:>16.3f}")

        # CSV
        stats_csv = OUTPUT_DIR / f"exhaustive_cv_landmark_stats_{cam_id}.csv"
        with open(stats_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["label", "held_out_mean_m", "held_out_median_m", "include_effect_m"])
            w.writeheader()
            w.writerows(stats)
        top_csv = OUTPUT_DIR / f"exhaustive_cv_top_subsets_{cam_id}.csv"
        with open(top_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["rank", "n_train", "test_rmse_m", "train_rmse_m", "train_landmarks"])
            w.writeheader()
            w.writerows(top_rows)
        print(f"\n  CSV保存: {stats_csv.name}, {top_csv.name}")


if __name__ == "__main__":
    main()