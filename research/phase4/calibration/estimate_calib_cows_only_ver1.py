"""
estimate_calib_cows_only_ver1.py

パラメータ推定 (3/3): 牛のみ(ランドマークは学習に使わない)。
  - 設定は、牛のクロスバリデーション(5分割×3回)の「外した牛の投影誤差(中央値)」が最小のものを選ぶ。
  - 最終パラメータは、全牛で作る。
  - 牛のクリックだけでは、全体の位置・向き・大きさは決まらない(priorで出発点の近くに留めている)。
    床の絶対位置の精度は、使っていないランドマークの誤差で見る。
  - 保存: bev_calibration_refined/calib_cows_only_ver1.json (+ .txt)
評価は evaluate_calib_params_ver1.py に jsonを渡す。

配置: research/phase4/calibration/ (calib_common_ver1.py と同じ場所)
使い方: python estimate_calib_cows_only_ver1.py [フレーム ...]
"""
import calib_common_ver1 as cc

if __name__ == "__main__":
    cc.estimate_main("cows_only")