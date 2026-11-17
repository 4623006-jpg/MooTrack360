"""
estimate_calib_edge_lm_cows_ver1.py

パラメータ推定 (1/3): 各カメラの画面の端のランドマーク6点(固定) + 牛。
  - 設定(動かすパラメータ・prior・LMの重み)は、牛のクロスバリデーション(5分割×3回)の
    「外した牛の投影誤差(中央値)」が最小のものを選ぶ。
  - 最終パラメータは、端のLM + 全牛で作る。
  - 保存: bev_calibration_refined/calib_edge_cows_ver1.json (+ .txt)
          学習に使ったLM/牛、使っていないLM/牛、CVの各分割のパラメータと外した点を含む。
評価は evaluate_calib_params_ver1.py に jsonを渡す。
出発点は Mootrack_core.py の現在値 (jsonの init_calibration_set に記録)。

配置: research/phase4/calibration/ (calib_common_ver1.py と同じ場所)
使い方: python estimate_calib_edge_lm_cows_ver1.py [フレーム ...]
"""
import calib_common_ver1 as cc

if __name__ == "__main__":
    cc.estimate_main("edge_cows")