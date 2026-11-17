"""
estimate_calib_lm_only_ver1.py

パラメータ推定 (2/3): ランドマークのみ(43点)。
  - 設定(動かすパラメータ・prior)は、ランドマークのクロスバリデーション(5分割×3回、カメラごとに均等に分割)の
    「外したランドマークの床誤差(RMSE)」が最小のものを選ぶ。
  - 最終パラメータは、全ランドマーク43点で作る。
  - 保存: bev_calibration_refined/calib_lm_only_ver1.json (+ .txt)
  - 最終パラメータは全LMを学習に使うので、未使用の点での値はjsonに保存したCVの各分割(外した点)で評価する。
評価は evaluate_calib_params_ver1.py に jsonを渡す。

配置: research/phase4/calibration/ (calib_common_ver1.py と同じ場所)
使い方: python estimate_calib_lm_only_ver1.py [フレーム ...]   (フレームは牛を読み込むためだけに使う)
"""
import calib_common_ver1 as cc

if __name__ == "__main__":
    cc.estimate_main("lm_only")