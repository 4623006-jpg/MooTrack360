"""
range_px_var_ver1.py  (配置: research/phase5/)

従来の有効範囲(orig: 許容ピクセル誤差の表、半径 300 px ごとの誤差の平均から決める)を、
検出誤差だけの分散(px_var)から作る範囲に置き換えるための共通部品。

px_var = HE(改) の有効範囲(he_kai)と同じ基準 k·e <= eps(e = 位置の分散の最大固有値の平方根)で、高さ誤差 σ_h を 0 にしたもの。
  検出誤差 σ_px(r)² × J Jᵀ だけで決まる。σ_px は sigma_he_kai_ver4.json の値をそのまま使う(新しく推定しない)。
  eps と k は maha_h_valid と同じ値(phase3_he_kai_matching の EPS_VALID_M / K_SIGMA)。

使い方:
  install(hk): phase3_he_kai_matching の run_mode を包み、有効範囲が "orig" のモード(B の iou / euclid など)を px_var に置き換える。
  環境変数 ORIG_RANGE: px_var(既定) / orig(従来の表の範囲に戻す。以前の結果の再現用)
  maha_h_valid(he_kai)や tri_gt のモードは変えない。
"""

import os

ORIG_RANGE = os.environ.get("ORIG_RANGE", "px_var")


class PxOnlyModel:
    """σ_h だけを 0 にしたラッパー(他の属性は元のモデルに任せる)。build_valid_polygons が使うのは sigma_px(r) と sigma_h_max()"""

    def __init__(self, model):
        self._m = model

    def sigma_h_max(self):
        return 0.0

    def __getattr__(self, name):
        return getattr(self._m, name)


def build_px_var_polygons(poses, cam_ids, get_intrinsics_fn, model, eps_m, k_sigma, height_m=0.8, n_angles=144, verbose=True):
    from he_kai_sigma_model_ver1 import build_valid_polygons
    return build_valid_polygons(poses, cam_ids, get_intrinsics_fn, PxOnlyModel(model), eps_m, k_sigma,
                                height_m=height_m, n_angles=n_angles, verbose=verbose)


def install(hk):
    """hk.run_mode を包む。cfg["polygons"] == "orig" のとき、he_kai の作り方で、σ_h = 0 のモデルを使う"""
    if ORIG_RANGE != "px_var" or getattr(hk, "_px_var_installed", False):
        return
    orig_run = hk.run_mode

    def run_mode(ym, mode, cfg, out_root, cache, orig_heights):
        if cfg.get("polygons") == "orig":
            saved = hk._MODEL
            hk._MODEL = PxOnlyModel(saved)
            try:
                return orig_run(ym, mode, dict(cfg, polygons="he_kai"), out_root, cache, orig_heights)
            finally:
                hk._MODEL = saved
        return orig_run(ym, mode, cfg, out_root, cache, orig_heights)

    hk.run_mode = run_mode
    hk._px_var_installed = True
    print("有効範囲: orig を px_var(検出誤差だけの分散)に置き換え(ORIG_RANGE=orig で従来の表の範囲)")