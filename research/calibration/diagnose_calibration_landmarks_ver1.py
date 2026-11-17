"""
diagnose_calibration_landmarks_ver1.py

目的: キャリブレーションのランドマークを「有効なもの」「修正するべきもの」
「中立」に仕分ける。calibrate_camera_from_landmarks_exhaustive_cv_ver1.py
(部分集合の総当たり)の結果を、外れ値に引っ張られにくい形(中央値・誤差の
打ち切り・破綻率)で集計し直し、さらに修正候補には「どちらにどれだけずれて
いるか」「ピクセルをどこに取り直すべきか」の目安まで出す。

exhaustive_cv_ver1の問題点(平均が、外挿が破綻した少数の組み合わせに引っ張られる)
への対処として、以下の指標はすべて中央値ベース、または誤差をCLIP_Mで打ち切る。

ランドマークごとの指標(学習点数がGOOD_FRACTION以上の組み合わせのみで算出):
  insample_med   そのランドマークを較正に使ったときの、その点自身の誤差の中央値。
                 大きい=他の点と矛盾している(どう使っても合わない)。
  heldout_med    そのランドマークを較正に使わなかったときの予測誤差の中央値。
  leverage       heldout_med - insample_med。大きい=この点が較正を強く拘束している。
  effect         対応のある比較: 部分集合S∪{この点} と S(この点だけが違う)のペアで、
                 共通のテスト点での平均誤差(CLIP_Mで打ち切り)の差の中央値。
                 マイナス=この点を足すと他の点が良くなる。
                 (他の点が入っているかによる交絡を除くため、単純な「含む/含まない」の
                  平均比較ではなく、ペア比較にしている)
  protect        上と同じペアで、Sでは他の点の誤差がBLOWUP_Mを超える(外挿が破綻する)が、
                 S∪{この点}では破綻しない、という割合の差。
                 大きい=この点が無いと較正が破綻しやすい=必須の点。
  discrepancy    この点を使わずに予測した世界座標が、登録した世界座標(GT頂点)から
                 (dx,dy)[m]ずれている。consistencyが大きい=組み合わせによらず同じ方向に
                 ずれる(系統的なずれ=クリック位置かGT頂点のどちらかが誤っている疑い)。

判定(しきい値はすべて設定で変更可能。経験則なので数字そのものより順位を重視):
  修正候補: insample_zがZ_FIX以上(他の点と矛盾)、または effect がEFFECT_FIX_M以上(含めると悪化)
  有効     : effect が-EFFECT_OK_M以下(含めると良くなる)、または protect がPROTECT_MIN以上
             (protectが大きければ「★必須」)
  中立     : 上のどちらでもない

修正候補を除いて再較正(クリーン較正)し、残りの点の誤差が下がるかを確認する。
下がらないなら、その判定は支持されない(修正候補がむしろ較正を支えている)。
また、クリーン較正を基準に、修正候補ごとに「このピクセルにすれば登録した世界座標に
一致する」という目安のピクセル座標を出す。目安であって正解ではない
(基準の較正自体に誤差があるため)。

出力:
  - 標準出力: 学習点数ごとの推移、ランドマーク別の判定表、修正候補の詳細
  - CSV: landmark_diagnosis_{cam_id}.csv
  - TXT: landmark_diagnosis_report.txt(全カメラ分の判定と修正の目安。Notion/Wordに貼る用)

--- 使い方 ---
    python diagnose_calibration_landmarks_ver1.py
(LANDMARKS等は calibrate_camera_from_landmarks_loocv_ver1.py からimportする)
Camera_2(17点)は約12万通りで10分弱。長い場合は MAX_SUBSETS_PER_CAMERA で間引く。
"""

import csv
import itertools
import multiprocessing as mp
import os
import sys
import time
import warnings
from math import comb
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

MOOTRACK_CODE_DIR = os.path.join(THIS_DIR, "..", "calibration")
if MOOTRACK_CODE_DIR not in sys.path:
    sys.path.insert(0, MOOTRACK_CODE_DIR)

from Mootrack_core import ROT_FMT, get_intrinsics, build_R, load_floorplan

from calibrate_camera_from_landmarks_loocv_ver1 import (  # noqa: E402
    LANDMARKS, CAMERA_CHAIN, FLOORPLAN_JSON_PATH,
    find_region_vertex, group_landmarks_by_camera, fit_full, predict_world_xy,
)

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------

MIN_TRAIN_POINTS = 6
MIN_TEST_POINTS = 2
MAX_SUBSETS_PER_CAMERA: Optional[int] = None
RANDOM_SEED = 0
N_PROCESSES = max(1, (os.cpu_count() or 2) - 1)

GOOD_FRACTION = 0.6      # 学習点数がn×この割合以上の組み合わせだけを、insample/heldout/discrepancyの算出に使う
CLIP_M = 3.0             # effect算出時、1点の誤差をこの値で打ち切る(破綻した組み合わせの影響を抑える)
BLOWUP_M = 3.0           # 他の点の誤差の最大値がこれを超えた組み合わせを「破綻」とみなす

# 判定しきい値(経験則)
Z_FIX = 1.0              # insample_med のロバストz値がこれ以上なら修正候補
EFFECT_FIX_M = 0.03      # 含めると他の点の誤差がこれ以上悪化するなら修正候補
EFFECT_OK_M = 0.03       # 含めると他の点の誤差がこれ以上改善するなら有効
PROTECT_MIN = 0.05       # 含めない場合の破綻率の増加がこれ以上なら「必須」

OUTPUT_DIR = Path(THIS_DIR)

_G: Dict[str, object] = {}


def _init_worker(parent_t, pts_px, world_gt, init_params6, init_xi, init_alpha):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    _G["parent_t"] = parent_t
    _G["pts_px"] = pts_px
    _G["init_params6"] = init_params6
    _G["init_xi"] = init_xi
    _G["init_alpha"] = init_alpha
    _G["world_gt"] = world_gt


def _eval_subset(train_idx: Tuple[int, ...]):
    """train_idxで較正し、全点の予測世界座標(n,2)を返す(無効はnan)。失敗時はNone。"""
    pts_px = _G["pts_px"]
    world_gt = _G["world_gt"]
    idx = np.array(train_idx, dtype=int)
    try:
        params6, xi, alpha = fit_full(
            _G["parent_t"], pts_px[idx], world_gt[idx],
            _G["init_params6"], _G["init_xi"], _G["init_alpha"],
        )
        pred, valid = predict_world_xy(params6, _G["parent_t"], pts_px, xi, alpha)
    except Exception:
        return train_idx, None
    pred = np.where(np.asarray(valid)[:, None], pred, np.nan)
    return train_idx, pred


def enumerate_subsets(n: int) -> List[Tuple[int, ...]]:
    subsets: List[Tuple[int, ...]] = []
    for k in range(MIN_TRAIN_POINTS, n - MIN_TEST_POINTS + 1):
        subsets.extend(itertools.combinations(range(n), k))
    return subsets


def count_subsets(n: int) -> int:
    return sum(comb(n, k) for k in range(MIN_TRAIN_POINTS, n - MIN_TEST_POINTS + 1))


def robust_z(values) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    med = np.nanmedian(v)
    mad = np.nanmedian(np.abs(v - med)) * 1.4826
    scale = mad if mad > 1e-9 else (np.nanstd(v) if np.nanstd(v) > 1e-9 else 1.0)
    return (v - med) / scale


# ---------------------------------------------------------------------------
# 集計(numpyだけで完結。テストしやすいよう独立関数にしてある)
# ---------------------------------------------------------------------------

def analyze_camera(M: np.ndarray, P: np.ndarray, world_gt: np.ndarray) -> List[dict]:
    """M: (S,n) bool 学習に使ったか。P: (S,n,2) 各組み合わせでの全点の予測世界座標。
    world_gt: (n,2) 登録した世界座標。ランドマークごとの指標dictのリストを返す。"""
    S, n = M.shape
    D = P - world_gt[None, :, :]                 # (S,n,2) 予測-登録
    E = np.linalg.norm(D, axis=2)                # (S,n) 誤差(無効はnan)
    sizes = M.sum(axis=1)

    E_fill = np.where(np.isnan(E), 1e3, E)       # 無効=破綻扱い
    E_clip = np.minimum(E_fill, CLIP_M)
    T_clip = np.where(M, np.nan, E_clip)         # テスト点のみ(打ち切り)
    T_fill = np.where(M, np.nan, E_fill)         # テスト点のみ(打ち切りなし)

    good_min = int(np.ceil(GOOD_FRACTION * n))
    good = sizes >= good_min

    # 部分集合 -> 行番号 の対応(ビットマスクをキーにする)
    keys = (M.astype(np.int64) * (np.int64(1) << np.arange(n, dtype=np.int64))[None, :]).sum(axis=1)
    key_to_row = {int(k): r for r, k in enumerate(keys)}

    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for i in range(n):
            rows_in = good & M[:, i]
            rows_out = good & ~M[:, i]

            insample_med = float(np.nanmedian(E[rows_in, i])) if rows_in.any() else float("nan")
            heldout_med = float(np.nanmedian(E[rows_out, i])) if rows_out.any() else float("nan")

            if rows_out.any():
                dv = D[rows_out, i, :]
                vec = np.nanmedian(dv, axis=0)
                spread = float(np.nanmedian(np.linalg.norm(dv - vec, axis=1)))
                consistency = float(np.linalg.norm(vec) / max(spread, 1e-3))
            else:
                vec = np.array([np.nan, np.nan])
                consistency = float("nan")

            # --- 対応のある比較: 部分集合S∪{i} と S(iだけが違う)のペアで比べる ---
            # 他のランドマーク(特に必須の点)が入っているかによる交絡を取り除くため、
            # 「iを足したときに、同じSに対して何が変わるか」だけを見る。
            # 評価に使うテスト点は両者に共通する S∪{i} の補集合(iを含まない)。
            rows_with = np.where(M[:, i] & (sizes - 1 >= MIN_TRAIN_POINTS))[0]
            pair_with, pair_wo = [], []
            bit = np.int64(1) << np.int64(i)
            for r in rows_with:
                r0 = key_to_row.get(int(keys[r] - bit))
                if r0 is not None:
                    pair_with.append(r)
                    pair_wo.append(r0)
            if pair_with:
                pair_with = np.array(pair_with)
                pair_wo = np.array(pair_wo)
                Tm = ~M[pair_with]                                  # (P,n) 共通テスト点
                a = np.where(Tm, E_clip[pair_with], np.nan)         # S∪{i}で較正
                b = np.where(Tm, E_clip[pair_wo], np.nan)           # Sで較正
                d_pair = np.nanmean(a, axis=1) - np.nanmean(b, axis=1)
                effect_i = float(np.nanmedian(d_pair))
                ca = np.nanmax(np.where(Tm, E_fill[pair_with], np.nan), axis=1) > BLOWUP_M
                cb = np.nanmax(np.where(Tm, E_fill[pair_wo], np.nan), axis=1) > BLOWUP_M
                protect_i = float(cb.mean() - ca.mean())
            else:
                effect_i, protect_i = float("nan"), float("nan")

            out.append({
                "insample_med": insample_med,
                "heldout_med": heldout_med,
                "leverage": heldout_med - insample_med,
                "effect": effect_i,
                "protect": protect_i,
                "dx": float(vec[0]), "dy": float(vec[1]),
                "consistency": consistency,
            })

    z = robust_z([d["insample_med"] for d in out])
    for d, zi in zip(out, z):
        d["insample_z"] = float(zi)
        fix = (zi >= Z_FIX) or (d["effect"] >= EFFECT_FIX_M)
        ok = (d["effect"] <= -EFFECT_OK_M) or (d["protect"] >= PROTECT_MIN)
        essential = d["protect"] >= PROTECT_MIN
        if fix:
            d["verdict"] = "修正候補(必須級)" if essential else "修正候補"
        elif ok:
            d["verdict"] = "有効★必須" if essential else "有効"
        else:
            d["verdict"] = "中立"
    return out


def suggest_pixel(params6, parent_t, xi, alpha, px0, gt) -> Tuple[float, float]:
    def res(p):
        w, v = predict_world_xy(params6, parent_t, np.array([p], dtype=np.float64), xi, alpha)
        return (w[0] - gt) if v[0] else np.array([10.0, 10.0])
    r = least_squares(res, np.array(px0, dtype=np.float64))
    return float(r.x[0]), float(r.x[1])


def main():
    print("floorplan.json 読み込み中...")
    regions, cameras_json_raw, real_w, real_h = load_floorplan(FLOORPLAN_JSON_PATH)
    raw_params_by_key = {
        f"{e['parent']}->{e['child']}": np.array(e["params"], dtype=np.float64)
        for e in cameras_json_raw
    }
    landmarks_by_cam = group_landmarks_by_camera(LANDMARKS)

    # --- チェーンに沿ったフル較正(親位置parent_tの確定。LOOCV版と同じ) ---
    poses_fitted: Dict[str, Tuple[np.ndarray, np.ndarray]] = {"Area1": (np.eye(3), np.zeros(3))}
    cam_info: Dict[str, dict] = {}
    for parent_id, cam_id in CAMERA_CHAIN:
        parent_R, parent_t = poses_fitted[parent_id]
        init_params6 = raw_params_by_key[f"{parent_id}->{cam_id}"]
        _, _, _, _, init_xi, init_alpha = get_intrinsics(cam_id)
        if cam_id not in landmarks_by_cam:
            poses_fitted[cam_id] = (build_R(*init_params6[3:], ROT_FMT), parent_t + init_params6[:3])
            continue
        lms = landmarks_by_cam[cam_id]
        pts_px = np.array([lm["pixel"] for lm in lms], dtype=np.float64)
        world_gt = np.array(
            [find_region_vertex(regions, lm["region"], lm["vertex_idx"]) for lm in lms], dtype=np.float64,
        )
        params6, xi, alpha = fit_full(parent_t, pts_px, world_gt, init_params6, init_xi, init_alpha)
        poses_fitted[cam_id] = (build_R(*params6[3:], ROT_FMT), parent_t + params6[:3])
        cam_info[cam_id] = {
            "parent_t": parent_t, "landmarks": lms, "pts_px": pts_px, "world_gt": world_gt,
            "init_params6": init_params6, "init_xi": init_xi, "init_alpha": init_alpha,
            "full": (params6, xi, alpha),
        }

    report_lines: List[str] = ["キャリブレーションランドマーク診断レポート", ""]

    for cam_id, info in cam_info.items():
        lms = info["landmarks"]
        n = len(lms)
        labels = [lm["label"] for lm in lms]
        pts_px, world_gt, parent_t = info["pts_px"], info["world_gt"], info["parent_t"]

        print(f"\n{'=' * 78}")
        print(f"■ {cam_id}  ランドマーク{n}点  部分集合(学習{MIN_TRAIN_POINTS}〜{n - MIN_TEST_POINTS}点): "
              f"全{count_subsets(n):,}通り")
        print(f"{'=' * 78}")
        if n < MIN_TRAIN_POINTS + MIN_TEST_POINTS:
            print("  ランドマーク数が少なすぎるためスキップ")
            continue

        subsets = enumerate_subsets(n)
        if MAX_SUBSETS_PER_CAMERA is not None and len(subsets) > MAX_SUBSETS_PER_CAMERA:
            rng = np.random.default_rng(RANDOM_SEED)
            pick = rng.choice(len(subsets), size=MAX_SUBSETS_PER_CAMERA, replace=False)
            subsets = [subsets[i] for i in sorted(pick)]
            print(f"  → {MAX_SUBSETS_PER_CAMERA:,}通りにランダム間引き")

        init_args = (parent_t, pts_px, world_gt, info["init_params6"], info["init_xi"], info["init_alpha"])
        _init_worker(*init_args)
        t0 = time.time()
        pilot = min(30, len(subsets))
        for s in subsets[:pilot]:
            _eval_subset(s)
        per_fit = (time.time() - t0) / max(1, pilot)
        print(f"  1回の較正: 約{per_fit * 1000:.0f}ms → 約{per_fit * len(subsets) / N_PROCESSES / 60:.1f}分の見込み"
              f"({N_PROCESSES}並列)")

        rows_mask, rows_pred = [], []
        t_start = time.time()
        with mp.Pool(N_PROCESSES, initializer=_init_worker, initargs=init_args) as pool:
            done = 0
            for train_idx, pred in pool.imap_unordered(_eval_subset, subsets, chunksize=64):
                done += 1
                if pred is not None:
                    m = np.zeros(n, dtype=bool)
                    m[list(train_idx)] = True
                    rows_mask.append(m)
                    rows_pred.append(pred)
                if done % 20000 == 0:
                    print(f"  {done:,}/{len(subsets):,} 完了 ({time.time() - t_start:.0f}秒)")
        print(f"  完了: 成功{len(rows_mask):,}/{len(subsets):,}通り ({time.time() - t_start:.0f}秒)")

        M = np.array(rows_mask)
        P = np.array(rows_pred)
        stats = analyze_camera(M, P, world_gt)

        # 学習点数ごとのテスト誤差(中央値): 点を増やす価値があるかの目安
        E = np.linalg.norm(P - world_gt[None], axis=2)
        sizes = M.sum(axis=1)
        T = np.where(M, np.nan, E)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            test_med = np.nanmedian(T, axis=1)
        print("\n  [学習点数ごとのテスト誤差(各組み合わせのテスト点の中央値 → その中央値)]")
        print("   " + "  ".join(
            f"{int(k)}点:{np.nanmedian(test_med[sizes == k]):.2f}m" for k in sorted(set(sizes.tolist()))))

        # --- ランドマーク別判定表 ---
        order = sorted(range(n), key=lambda i: (
            0 if stats[i]["verdict"].startswith("修正") else (1 if stats[i]["verdict"].startswith("有効") else 2),
            -stats[i]["insample_z"]))
        print(f"\n  {'label':<26}{'判定':<14}{'in-sample':>10}{'held-out':>10}{'effect':>9}{'protect':>9}"
              f"{'ずれdx,dy[m]':>16}{'一貫性':>8}")
        for i in order:
            s = stats[i]
            print(f"  {labels[i]:<26}{s['verdict']:<14}{s['insample_med']:>10.3f}{s['heldout_med']:>10.3f}"
                  f"{s['effect']:>9.3f}{s['protect']:>9.2f}"
                  f"{s['dx']:>9.2f},{s['dy']:>5.2f}{s['consistency']:>8.1f}")
        print("   (in-sample/held-outは誤差の中央値[m]、effect[m]はマイナスほど有用、"
              "protectは大きいほど必須、一貫性は大きいほど系統的なずれ)")

        # --- クリーン較正(修正候補を除いて再較正)と、修正の目安ピクセル ---
        suspects = [i for i in range(n) if stats[i]["verdict"].startswith("修正")]
        keep = [i for i in range(n) if i not in suspects]
        suggestions: Dict[int, Tuple[float, float]] = {}
        report_lines.append(f"■ {cam_id}")
        report_lines.append("有効: " + (", ".join(labels[i] for i in range(n) if stats[i]["verdict"].startswith("有効")) or "なし"))
        report_lines.append("修正候補: " + (", ".join(labels[i] for i in suspects) or "なし"))

        if suspects and len(keep) >= MIN_TRAIN_POINTS:
            kidx = np.array(keep, dtype=int)
            c_params6, c_xi, c_alpha = fit_full(
                parent_t, pts_px[kidx], world_gt[kidx],
                info["init_params6"], info["init_xi"], info["init_alpha"])
            f_params6, f_xi, f_alpha = info["full"]

            def rmse_on(idx, p6, xi_, al_):
                pr, v = predict_world_xy(p6, parent_t, pts_px[idx], xi_, al_)
                e = np.linalg.norm(pr - world_gt[idx], axis=1)
                return float(np.sqrt(np.nanmean(e ** 2)))

            r_full = rmse_on(kidx, f_params6, f_xi, f_alpha)
            r_clean = rmse_on(kidx, c_params6, c_xi, c_alpha)
            verdict_txt = "改善 → 修正候補の判定を支持" if r_clean < r_full else "改善せず → 修正候補の判定は弱い"
            print(f"\n  [クリーン較正(修正候補{len(suspects)}点を除外)] 残り{len(keep)}点でのin-sample RMSE: "
                  f"全点較正={r_full:.3f}m → クリーン較正={r_clean:.3f}m ({verdict_txt})")
            report_lines.append(f"クリーン較正の効果(残り{len(keep)}点のRMSE): {r_full:.3f}m → {r_clean:.3f}m ({verdict_txt})")

            print("\n  [修正候補の詳細(目安ピクセル=クリーン較正で登録世界座標に一致するピクセル。正解ではなく目安)]")
            for i in suspects:
                u0, v0 = pts_px[i]
                u1, v1 = suggest_pixel(c_params6, parent_t, c_xi, c_alpha, (u0, v0), world_gt[i])
                suggestions[i] = (u1, v1)
                line = (f"{labels[i]}: 現在ピクセル=({u0:.0f},{v0:.0f}) → 目安=({u1:.0f},{v1:.0f}) "
                        f"[Δ=({u1 - u0:+.0f},{v1 - v0:+.0f})px]  / 世界座標のずれ(dx,dy)=({stats[i]['dx']:+.2f},{stats[i]['dy']:+.2f})m "
                        f"一貫性={stats[i]['consistency']:.1f}")
                print("   " + line)
                report_lines.append("  - " + line)
        elif suspects:
            print("\n  修正候補を除くと残りが少なすぎるため、クリーン較正はスキップ")
        else:
            print("\n  修正候補なし")
        report_lines.append("")

        # CSV
        csv_path = OUTPUT_DIR / f"landmark_diagnosis_{cam_id}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["label", "region", "vertex_idx", "verdict", "insample_med_m", "heldout_med_m",
                        "leverage_m", "effect_m", "protect", "dx_m", "dy_m", "consistency",
                        "pixel_u", "pixel_v", "suggested_u", "suggested_v"])
            for i in range(n):
                s = stats[i]
                su = suggestions.get(i, ("", ""))
                w.writerow([labels[i], lms[i]["region"], lms[i]["vertex_idx"], s["verdict"],
                            round(s["insample_med"], 4), round(s["heldout_med"], 4), round(s["leverage"], 4),
                            round(s["effect"], 4), round(s["protect"], 4), round(s["dx"], 4), round(s["dy"], 4),
                            round(s["consistency"], 2), pts_px[i][0], pts_px[i][1], su[0], su[1]])
        print(f"\n  CSV保存: {csv_path.name}")

    report_path = OUTPUT_DIR / "landmark_diagnosis_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\nレポート保存: {report_path}")


if __name__ == "__main__":
    main()