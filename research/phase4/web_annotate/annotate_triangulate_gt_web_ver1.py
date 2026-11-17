"""
annotate_triangulate_gt_web_ver5.py   (ver4からの変更: マウスのホイールを前に回すと拡大、後ろに回すと縮小。トラックパッドの2本指は今まで通り移動)
  (ver4: ver3からの変更: 単眼推定の高さを固定値にできる設定 HEIGHT_OVERRIDE_M を追加。lyingは0.5m)
  (ver3: 1台にしか写らない牛を「単眼推定」で補う機能を追加。牛ごとに姿勢を選ぶ)

2台以上のカメラ画像の上で、牛の背中の「同じ点」をクリックし、三角測量で世界座標(x, y, z)を求めて
BEVのGT(牛の位置)を作るブラウザ版ツール。高さの仮定は使わない。

  画面: 3カメラの画像が横に並ぶ(各カメラ1枚。拡大・移動できる)。下に BEV と、牛の一覧表。
  1頭ずつ「牛」を作り、その牛の背中の同じ点を、見えている2台以上のカメラでクリックする。
    - 1台目をクリックすると、他のカメラの画像に「エピポーラ曲線」(その点が3D空間で取りうる
      高さ 0〜1.8m の範囲を、他のカメラに写した線。目盛りは高さ[m])が出る。
      背中の同じ点は、この線の上にあるはずなので、線に沿って探してクリックする。
    - 2点そろうと三角測量し、3台目のカメラには予測位置(橙の×)が出る。
  一覧表に、交差角・光線の残差・高さ・再投影誤差・BEV手動GTとの最近傍距離・状態(警告)が出る。
  画像に重ねた白い細い枠はGTのbbox(牛名つき)。クリックした位置が枠の中なら、牛名を自動で入れる。

操作:
  左クリック         : その牛の、そのカメラでの点を置く(置き直すと上書き)
  Shift + 左クリック : その牛の、そのカメラの点を消す
  マウスのホイール: 前(奥)に回すと拡大、後ろ(手前)に回すと縮小(1ノッチ WHEEL_STEP倍)
  2本指スライド(トラックパッド): 画像を上下左右に移動
  ホイールの種類は自動判定する。うまく判定されないときは、w キー(またはボタン)で
  「自動 → ホイール=拡大縮小 → ホイール=移動」を切り替える
  ピンチ、または Ctrl(⌘)+ホイール: 拡大・縮小(感度は ZOOM_SENS で調整)   /   右ドラッグ: 画像を移動
  n 新しい牛 / u 直前の取り消し / b bbox表示切替 / s 保存  (名前の入力欄にいるときは無効)
  一覧表の行をクリック: その牛を選択して編集

単眼推定(レベル2):
  1台にしか写らない牛は、その牛の「姿勢」(standing / lying)を表で選ぶと、背中の点の光線を、
  2台以上で三角測量できた牛(レベル1)の高さの中央値(姿勢ごと)の平面に当てて位置を出す。
  レベル1の牛が姿勢ごとに MIN_SAMPLES_HEIGHT 頭未満のうちは、仮の高さ(DEFAULT_BACK_HEIGHT_M)を使う
  (表の状態欄に「仮の値」と出る)。位置の不確かさσ ≈ (高さのばらつき) x (水平方向/鉛直方向の光線の傾き)
  も出す。レベル1の牛にも姿勢を選んでおくと、高さの実測値が貯まる。
  二台目をクリックすれば、その牛は自動でレベル1(三角測量)に上がる。

出力(OUTPUT_DIR / bev_gt_triangulated /):
  {frame}_tri_gt_ver1.json      : 牛ごとのクリック(各カメラの画素)と三角測量の結果(続きから編集できる)
  {frame}_bev_gt_tri_ver1.json  : BEVのGT(id, x_m, y_m, name, height_m, level, 状態)。レベル1と2の両方
  {frame}_bev_gt_tri_level1_ver1.json : レベル1(三角測量)だけ。eval_bev_gt_motp_ver1.py の
                                  ANNOTATION_JSON に指定すると評価に使える

注意:
  - 交差角が小さい(目安60°未満)と、奥行き(高さ)の誤差が大きい。警告が出る。
  - 1台にしか写らない牛、カメラ間で同じ点を特定できない牛は、三角測量できない。
  - 校正が合っていないと、再投影誤差(px)に出る。大きいときは、点の置き方か校正を疑う。
  - 光線とエピポーラ曲線は、魚眼モデルの逆投影の結果を格子で探索して作る(誤差は数px程度)。

起動:
    uv run python ../annotate/annotate_triangulate_gt_web_ver1.py 100          # フレーム100
    uv run python ../annotate/annotate_triangulate_gt_web_ver1.py 100 8123     # ポート指定(省略時8000)
  手元のPCで ssh -L 8123:localhost:8123 <ユーザー>@<サーバー> を張り、http://localhost:8123 を開く。
  終了はサーバー側で Ctrl+C。保存し忘れに注意(「手元にDL」ボタンで、接続が切れても控えが取れる)。

annotate_bev_gt_click_ver1.py / phase3_bev_multicam_gt_zonepose_ver1.py と同じ環境で動かす。
"""

import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYZE_DIR = os.path.join(THIS_DIR, "..", "analyze")
ANNOTATE_DIR = os.path.join(THIS_DIR, "..", "annotate")
PHASE3_DIR = os.path.join(THIS_DIR, "..", "phase3_multi_camera")
for d in (os.path.join(THIS_DIR, "..", "calibration"), ANALYZE_DIR, ANNOTATE_DIR, PHASE3_DIR, THIS_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

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

from Mootrack_core import get_intrinsics, unproject_ds

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
DEFAULT_PORT = 8000
DEFAULT_FRAME = 100
WHEEL_STEP = 1.12             # マウスのホイール1ノッチあたりの拡大率(縮小は逆数)
ZOOM_SENS = 0.006            # 拡大縮小の感度(大きいほど敏感。ブラウザ側のJSでも使う)
JPEG_SCALE = 0.75              # ブラウザに送るカメラ画像の縮小率(1.0で原寸。重ければ下げる)
LOOKUP_STEP_PX = 4             # 世界座標 -> 画素 の探索格子の間隔
EPI_HEIGHT_BAND_M = (0.0, 1.8) # エピポーラ曲線として描く高さの範囲
EPI_TICKS_M = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
MIN_ANGLE_WARN_DEG = 60.0      # これ未満は「交差角小」の警告
MAX_RESID_WARN_M = 0.30        # 光線の残差(光線から点までの距離の最大)がこれ超なら警告
HEIGHT_OK_RANGE_M = (0.2, 1.8) # 三角測量した高さがこの範囲外なら警告
MAX_REPROJ_WARN_PX = 15.0      # 再投影誤差がこれ超なら警告
DEFAULT_BACK_HEIGHT_M = {"standing": 1.35, "lying": 0.5}   # 仮の値(実測が貯まるまで使う。要見直し)
# 単眼推定の高さを固定する。Noneなら、レベル1の実測中央値(足りなければ上の仮の値)を使う。
# 胴体の中央付近(背中の頂点より低い所)をクリックするなら、実測より低めの値が合うことがある。
HEIGHT_OVERRIDE_M = {"standing": None, "lying": 0.5}
OVERRIDE_SIGMA_M = 0.10        # 固定値のときの高さのばらつき(位置σの計算に使う)
DEFAULT_SIGMA_H_M = 0.15       # 仮の高さのばらつき
MIN_SAMPLES_HEIGHT = 3         # 姿勢ごとに、この頭数以上で実測の中央値に切り替える
FALLBACK_IMAGE_DIR = "/mnt/vmlqnap02/dataset/MooTrack360/1-hour-farm1-area1-sample-sequence/1-hour-farm1-area1-frames"

S = {}   # サーバーの状態


# ---------------------------------------------------------------------------
# カメラモデル(光線と、世界座標 -> 画素)
# ---------------------------------------------------------------------------
class CamModel:
    def __init__(self, cam_id, R, t, width, height, step=LOOKUP_STEP_PX):
        self.id = cam_id
        self.R = np.asarray(R, dtype=np.float64)
        self.t = np.asarray(t, dtype=np.float64).reshape(3)
        self.width, self.height = int(width), int(height)
        self.intr = get_intrinsics(cam_id)
        self.step = step
        xs = np.arange(0, self.width, step, dtype=np.float64)
        ys = np.arange(0, self.height, step, dtype=np.float64)
        gx, gy = np.meshgrid(xs, ys)
        px = np.stack([gx.ravel(), gy.ravel()], axis=1)
        dirs, ok = unproject_ds(px, *self.intr)
        self.grid_px = px[ok]
        self.tree = cKDTree(dirs[ok])
        offs = np.arange(-step, step + 1, 1, dtype=np.float64)
        ox, oy = np.meshgrid(offs, offs)
        self.offsets = np.stack([ox.ravel(), oy.ravel()], axis=1)

    def ray(self, u, v):
        """画素(u, v) -> (カメラ中心, 世界座標系の単位方向)。無効な画素はNone"""
        d_cam, ok = unproject_ds(np.array([[u, v]], dtype=np.float64), *self.intr)
        if not ok[0]:
            return None
        d = self.R @ d_cam[0]
        n = np.linalg.norm(d)
        if n < 1e-9:
            return None
        return self.t.copy(), d / n

    def pixel_of_world(self, X, tol_deg=0.5):
        """世界座標の点 -> 画素(u, v)。画像の外・視野の外ならNone"""
        d_cam = self.R.T @ (np.asarray(X, dtype=np.float64) - self.t)
        n = np.linalg.norm(d_cam)
        if n < 1e-9:
            return None
        d_cam = d_cam / n
        _dist, i = self.tree.query(d_cam)
        coarse = self.grid_px[i]
        cand = coarse + self.offsets
        dirs, ok = unproject_ds(cand, *self.intr)
        if not ok.any():
            return None
        dots = np.where(ok, dirs @ d_cam, -1.0)
        j = int(np.argmax(dots))
        if np.degrees(np.arccos(np.clip(dots[j], -1.0, 1.0))) > tol_deg:
            return None
        u, v = cand[j]
        if not (0 <= u < self.width and 0 <= v < self.height):
            return None
        return float(u), float(v)


def triangulate(rays):
    """rays: [(c, d)]。光線への距離の二乗和を最小にする点 (sum(I-ddT)) X = sum((I-ddT) c)。失敗はNone"""
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for c, d in rays:
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ c
    if np.linalg.cond(A) > 1e8:       # ほぼ平行
        return None
    X = np.linalg.solve(A, b)
    if any(np.dot(X - c, d) <= 0 for c, d in rays):   # カメラの後ろ
        return None
    return X


def angle_deg(d1, d2):
    return float(np.degrees(np.arccos(np.clip(np.dot(d1, d2), -1.0, 1.0))))


def _plane_point(c, d, h):
    """光線(c, d)が、高さhの水平面(z=-h)と交わる点。交わらなければNone"""
    if d[2] <= 1e-6:
        return None
    t = (-c[2] - h) / d[2]
    if t <= 0:
        return None
    return c + t * d


def _collect_rays(models, clicks):
    rays = []
    for cid, uv in clicks.items():
        m = models.get(cid)
        if m is None or uv is None:
            continue
        r = m.ray(uv[0], uv[1])
        if r is not None:
            rays.append((cid, r[0], r[1]))
    return rays


def _add_nn(out, X, hand_xy):
    if hand_xy is not None and len(hand_xy):
        out["nn_hand_m"] = round(float(np.min(np.linalg.norm(hand_xy - X[:2], axis=1))), 3)


def solve_cow(models, cow, hand_xy=None, hstats=None):
    """レベル1: 2台以上で三角測量 / レベル2: 1台+姿勢+高さの中央値で単眼推定 / 0: 位置なし"""
    clicks = cow.get("clicks", {})
    rays = _collect_rays(models, clicks)
    out = {"id": cow["id"], "pose": cow.get("pose", ""), "n_clicks": len(rays), "cams": [r[0] for r in rays],
           "X": None, "flags": [], "info": [], "level": 0}
    if len(rays) == 0:
        out["flags"].append("点が無い")
        return out
    if len(rays) == 1:
        pose = cow.get("pose", "")
        if pose not in ("standing", "lying") or not hstats:
            out["flags"].append("1台のみ: 姿勢を選ぶと単眼推定できる、または2台目をクリック")
            return out
        cid, c, d = rays[0]
        stt = hstats[pose]
        X = _plane_point(c, d, stt["median"])
        if X is None:
            out["flags"].append("単眼推定不可(光線が平面に届かない)")
            return out
        sigma = stt["std"] * float(np.hypot(d[0], d[1])) / float(d[2])
        pred = {}
        for oid, m in models.items():
            if oid != cid:
                pix = m.pixel_of_world(X)
                if pix is not None:
                    pred[oid] = [round(pix[0], 1), round(pix[1], 1)]
        out.update({"X": [round(float(x), 4) for x in X], "height_m": round(float(stt["median"]), 3),
                    "level": 2, "pos_sigma_m": round(sigma, 3), "pred_px": pred, "reproj_px": {}})
        out["info"].append(f"単眼推定(高さ{stt['median']:.2f}m {stt['source']}, 位置σ≈{sigma:.2f}m)")
        _add_nn(out, X, hand_xy)
        return out

    X = triangulate([(c, d) for _cid, c, d in rays])
    if X is None:
        out["flags"].append("三角測量不可(ほぼ平行/後方)")
        return out
    max_angle = max(angle_deg(rays[i][2], rays[j][2])
                    for i in range(len(rays)) for j in range(i + 1, len(rays)))
    resid = max(float(np.linalg.norm((np.eye(3) - np.outer(d, d)) @ (X - c))) for _cid, c, d in rays)
    height = float(-X[2])
    reproj, pred = {}, {}
    for cid, m in models.items():
        pix = m.pixel_of_world(X)
        if cid in clicks and clicks[cid] is not None:
            reproj[cid] = None if pix is None else round(float(np.hypot(pix[0] - clicks[cid][0],
                                                                         pix[1] - clicks[cid][1])), 1)
        elif pix is not None:
            pred[cid] = [round(pix[0], 1), round(pix[1], 1)]
    out.update({"X": [round(float(x), 4) for x in X], "height_m": round(height, 3), "level": 1,
                "max_angle_deg": round(max_angle, 1), "resid_m": round(resid, 3),
                "reproj_px": reproj, "pred_px": pred})
    if max_angle < MIN_ANGLE_WARN_DEG:
        out["flags"].append(f"交差角小({max_angle:.0f}°)")
    if resid > MAX_RESID_WARN_M:
        out["flags"].append(f"残差大({resid:.2f}m)")
    if not (HEIGHT_OK_RANGE_M[0] <= height <= HEIGHT_OK_RANGE_M[1]):
        out["flags"].append(f"高さ範囲外({height:.2f}m)")
    big = [cid for cid, e in reproj.items() if e is not None and e > MAX_REPROJ_WARN_PX]
    if big:
        out["flags"].append("再投影誤差大(" + ",".join(c.replace("Camera_", "C") for c in big) + ")")
    _add_nn(out, X, hand_xy)
    return out


def compute_height_stats(models, cows):
    """レベル1(警告なし)の牛の高さから、姿勢ごとの中央値とばらつきを出す。足りなければ仮の値"""
    hs = {"standing": [], "lying": []}
    for cow in cows:
        pose = cow.get("pose", "")
        if pose not in hs:
            continue
        r = solve_cow(models, cow, None, None)
        if r["level"] == 1 and not r["flags"]:
            hs[pose].append(r["height_m"])
    stats = {}
    for pose, arr in hs.items():
        fixed = HEIGHT_OVERRIDE_M.get(pose)
        if fixed is not None:
            ref = f", 実測中央値{np.median(arr):.2f}m n={len(arr)}" if len(arr) else ", 実測なし"
            stats[pose] = {"median": float(fixed), "std": OVERRIDE_SIGMA_M, "n": len(arr),
                           "source": "固定値" + ref}
        elif len(arr) >= MIN_SAMPLES_HEIGHT:
            stats[pose] = {"median": float(np.median(arr)), "std": float(max(np.std(arr, ddof=1), 0.03)),
                           "n": len(arr), "source": "実測中央値"}
        else:
            stats[pose] = {"median": DEFAULT_BACK_HEIGHT_M[pose], "std": DEFAULT_SIGMA_H_M,
                           "n": len(arr), "source": "仮の値"}
    return stats


def solve_all(models, cows, hand_xy=None):
    stats = compute_height_stats(models, cows)
    return [solve_cow(models, c, hand_xy, stats) for c in cows], stats


def epipolar_curves(models, sources):
    """sources: [{cam,u,v}] -> 他のカメラ画像に写した曲線(高さ EPI_HEIGHT_BAND_M の範囲)"""
    curves = []
    for s in sources:
        src = models.get(s["cam"])
        if src is None:
            continue
        r = src.ray(s["u"], s["v"])
        if r is None:
            continue
        c, d = r
        if d[2] <= 1e-6:       # 光線が上を向いている
            continue

        def t_of_h(h):
            return (-c[2] - h) / d[2]

        h_lo, h_hi = EPI_HEIGHT_BAND_M
        ts = np.linspace(t_of_h(h_hi), t_of_h(h_lo), 90)
        ts = ts[ts > 0]
        for tid, tgt in models.items():
            if tid == s["cam"]:
                continue
            poly = []
            for t in ts:
                pix = tgt.pixel_of_world(c + t * d)
                if pix is not None:
                    poly.append([round(pix[0], 1), round(pix[1], 1)])
            ticks = []
            for h in EPI_TICKS_M:
                t = t_of_h(h)
                if t <= 0:
                    continue
                pix = tgt.pixel_of_world(c + t * d)
                if pix is not None:
                    ticks.append({"h": h, "u": round(pix[0], 1), "v": round(pix[1], 1)})
            if poly:
                curves.append({"src": s["cam"], "tgt": tid, "poly": poly, "ticks": ticks})
    return curves


# ---------------------------------------------------------------------------
# ブラウザ画面
# ---------------------------------------------------------------------------
HTML = r"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><title>Triangulate GT</title>
<style>
 body{margin:6px;background:#222;color:#eee;font-family:Arial,sans-serif;font-size:13px}
 button{margin-right:4px;padding:3px 9px}
 #bar{margin-bottom:4px} #msg{color:#fc6;margin-left:8px}
 #cams{display:flex;gap:4px}
 .panel{flex:1;min-width:0;background:#111}
 .panel .t{font-weight:bold;padding:1px 4px}
 .panel canvas{display:block;width:100%;height:56vh;cursor:crosshair}
 #low{display:flex;gap:8px;margin-top:6px;align-items:flex-start}
 #bevwrap{flex:0 0 40%} #bevwrap canvas{width:100%;display:block}
 #tblwrap{flex:1;overflow:auto;max-height:40vh}
 table{border-collapse:collapse;width:100%} td,th{border:1px solid #555;padding:2px 5px;white-space:nowrap}
 th{background:#334;position:sticky;top:0} tr.sel{background:#445} tr{cursor:pointer}
 .warn{color:#f88} .ok{color:#8d8} .info{color:#7cf}
 select{background:#333;color:#eee;border:1px solid #666}
 input.nm{width:70px;background:#333;color:#eee;border:1px solid #666}
 .dirty{color:#f66}.clean{color:#6c6}
</style></head><body>
<div id="bar">
 <b id="title"></b>
 <button onclick="newCow()">新しい牛(n)</button>
 <button onclick="undo()">取消(u)</button>
 <button onclick="toggleBox()">bbox(b)</button>
 <button id="wbtn" onclick="cycleWheel()">ホイール(w): 自動</button>
 <button onclick="save()">保存(s)</button>
 <button onclick="dl()">手元にDL</button>
 <span id="status" class="clean"></span><span id="msg"></span>
 <br><small>1台のみの牛は、表で姿勢を選ぶと単眼推定(Lv2)。2台目をクリックするとLv1に上がる。<br>左クリック:点を置く / Shift+左クリック:その点を消す / 2本指スライド:移動 / ピンチ:拡大縮小 / マウスのホイール(前=拡大,後=縮小) /
 曲線の目盛り=高さ[m] / 橙の×=予測位置</small>
 <div id="sum"></div>
</div>
<div id="cams"></div>
<div id="low">
 <div id="bevwrap"><canvas id="bev"></canvas></div>
 <div id="tblwrap"><table id="tbl"></table></div>
</div>
<script>
const ZOOM_SENS=__ZOOM_SENS__, WHEEL_STEP=__WHEEL_STEP__;
let wheelMode='auto', wLast=0, wDev='pad';
const MODE_TXT={auto:'自動',zoom:'拡大縮小',pan:'移動'};
function cycleWheel(){wheelMode=wheelMode==='auto'?'zoom':(wheelMode==='zoom'?'pan':'auto');
  document.getElementById('wbtn').textContent='ホイール(w): '+MODE_TXT[wheelMode];}
// マウスのホイールか、トラックパッドかを判定する(操作の始まりで1回判定し、続く操作はそれを引き継ぐ)
function wheelDevice(ev){
  if(wheelMode==='zoom')return 'mouse'; if(wheelMode==='pan')return 'pad';
  const now=performance.now(), gap=now-wLast; wLast=now;
  if(gap>300){
    const lines=ev.deltaMode===1;
    const notch=ev.deltaX===0&&Number.isInteger(ev.deltaY)&&ev.wheelDeltaY!==undefined&&ev.wheelDeltaY!==0&&Math.abs(ev.wheelDeltaY)%120===0;
    wDev=(lines||notch)?'mouse':'pad';
  }
  return wDev;}
let stats={};
let st, cows=[], cur=-1, dirty=false, results={}, curves=[], showBox=true, hist=[];
const P={}, ORDER=[];
const CAMCOL={Camera_1:'#3c78ff',Camera_2:'#3cc83c',Camera_3:'#ff8c3c'};
const bevImg=new Image(); const bev=document.getElementById('bev'), bctx=bev.getContext('2d');

function setMsg(t){document.getElementById('msg').textContent=t||'';}
function setDirty(v){dirty=v;const s=document.getElementById('status');
  s.textContent=(v?'未保存':'保存済み')+'  牛='+cows.length; s.className=v?'dirty':'clean';}
function nextId(){return cows.reduce((a,c)=>Math.max(a,c.id),0)+1;}
function newCow(){cows.push({id:nextId(),name:'',pose:'',clicks:{}});cur=cows.length-1;setDirty(true);refresh();}
function curCow(){return cur>=0?cows[cur]:null;}

function makePanel(id){
  const div=document.createElement('div'); div.className='panel';
  div.innerHTML='<div class="t" style="color:'+CAMCOL[id]+'">'+id+'</div><canvas></canvas>';
  document.getElementById('cams').appendChild(div);
  const cv=div.querySelector('canvas');
  const p={id,cv,ctx:cv.getContext('2d'),img:new Image(),zoom:1,ox:0,oy:0,
           tileW:st.cams[id].w,tileH:st.cams[id].h,pan:null,loaded:false};
  P[id]=p; ORDER.push(id);
  p.img.onload=()=>{p.loaded=true;fit(p);drawPanel(p);};
  p.img.src='/cam/'+id+'.jpg';
  cv.addEventListener('mousedown',ev=>onDown(p,ev));
  cv.addEventListener('contextmenu',ev=>ev.preventDefault());
  cv.addEventListener('wheel',ev=>{ev.preventDefault();
    const k=ev.deltaMode===1?16:1; const dx=ev.deltaX*k, dy=ev.deltaY*k;
    const r=cv.getBoundingClientRect(), x=ev.clientX-r.left, y=ev.clientY-r.top;
    const zoomAt=f=>{const nz=Math.min(8,Math.max(0.03,p.zoom*f));
      p.ox=x-(x-p.ox)*nz/p.zoom; p.oy=y-(y-p.oy)*nz/p.zoom; p.zoom=nz;};
    if(ev.ctrlKey||ev.metaKey){          // ピンチ(Macはctrl付きで届く)、またはCtrl/⌘+ホイール
      const d=Math.max(-30,Math.min(30,dy)); zoomAt(Math.exp(-d*ZOOM_SENS));
    } else if(wheelDevice(ev)==='mouse'){ // マウスのホイール: 前に回す(dy<0)で拡大、後ろ(dy>0)で縮小
      if(dy!==0)zoomAt(dy<0?WHEEL_STEP:1/WHEEL_STEP);
    } else {                              // 2本指スライド: 移動
      p.ox-=dx; p.oy-=dy;
    }
    drawPanel(p);},{passive:false});
}
function sizeCanvas(p){const r=p.cv.getBoundingClientRect(); p.cv.width=Math.max(50,Math.round(r.width)); p.cv.height=Math.max(50,Math.round(r.height));}
function fit(p){sizeCanvas(p); p.zoom=Math.min(p.cv.width/p.tileW,p.cv.height/p.tileH); p.ox=0; p.oy=0;}
function toTile(p,ev){const r=p.cv.getBoundingClientRect();
  return [((ev.clientX-r.left)-p.ox)/p.zoom,((ev.clientY-r.top)-p.oy)/p.zoom];}

function bboxAt(p,u,v){let best=null,ba=1e18;
  (st.bboxes[p.id]||[]).forEach(b=>{if(u>=b.x1&&u<=b.x2&&v>=b.y1&&v<=b.y2){const a=(b.x2-b.x1)*(b.y2-b.y1); if(a<ba){ba=a;best=b;}}});
  return best;}

function onDown(p,ev){
  if(ev.button===2){p.pan={x:ev.clientX,y:ev.clientY,ox:p.ox,oy:p.oy};return;}
  if(ev.button!==0)return;
  if(cur<0)newCow();
  const [u,v]=toTile(p,ev); if(u<0||v<0||u>p.tileW||v>p.tileH)return;
  const c=cows[cur];
  hist.push({id:c.id,cam:p.id,prev:c.clicks[p.id]||null,prevName:c.name});
  if(ev.shiftKey){delete c.clicks[p.id];}
  else{
    c.clicks[p.id]=[u,v];
    const b=bboxAt(p,u,v);
    if(b){ if(!c.name){c.name=b.name;setMsg('');}
           else if(c.name!==b.name){setMsg('注意: クリック位置のbboxは '+b.name+'、この牛の名前は '+c.name);} }
  }
  setDirty(true); refresh();
}
window.addEventListener('mousemove',ev=>{ORDER.forEach(id=>{const p=P[id]; if(p.pan){
  p.ox=p.pan.ox+(ev.clientX-p.pan.x); p.oy=p.pan.oy+(ev.clientY-p.pan.y); drawPanel(p);}});});
window.addEventListener('mouseup',()=>{ORDER.forEach(id=>{P[id].pan=null;});});

function undo(){const h=hist.pop(); if(!h)return; const c=cows.find(x=>x.id===h.id); if(!c)return;
  if(h.prev)c.clicks[h.cam]=h.prev; else delete c.clicks[h.cam]; c.name=h.prevName; setDirty(true); refresh();}
function toggleBox(){showBox=!showBox; drawAll();}
function delCow(i){ev_stop(); const id=cows[i].id; cows.splice(i,1); if(cur>=cows.length)cur=cows.length-1; delete results[id]; setDirty(true); refresh();}
function ev_stop(){ if(window.event)window.event.stopPropagation(); }
function selCow(i){cur=i; refresh();}

async function post(url,body,ms){
  const ctl=new AbortController(); const t=setTimeout(()=>ctl.abort(),ms||20000);
  try{const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body),signal:ctl.signal});
      if(!r.ok)throw new Error('HTTP '+r.status); return await r.json();} finally{clearTimeout(t);}}
function plain(){return cows.map(c=>({id:c.id,name:c.name,pose:c.pose||'',clicks:c.clicks}));}

async function refresh(){
  try{
    const sol=await post('/solve',{cows:plain()});
    results={}; sol.results.forEach(r=>{results[r.id]=r;}); stats=sol.stats||{};
    const c=curCow(); const src=[];
    if(c){Object.keys(c.clicks).forEach(cid=>{src.push({cam:cid,u:c.clicks[cid][0],v:c.clicks[cid][1]});});}
    curves=src.length?(await post('/epi',{sources:src})).curves:[];
  }catch(e){setMsg('サーバーと通信できません(ssh転送が切れた可能性)。画面は閉じず「手元にDL」で控えを取ってください。');}
  drawAll(); renderTable(); drawBev(); setDirty(dirty);
}

function drawAll(){ORDER.forEach(id=>drawPanel(P[id]));}
function drawPanel(p){
  if(!p.loaded)return; const ctx=p.ctx;
  ctx.setTransform(1,0,0,1,0,0); ctx.clearRect(0,0,p.cv.width,p.cv.height);
  ctx.setTransform(p.zoom,0,0,p.zoom,p.ox,p.oy);
  ctx.drawImage(p.img,0,0,p.tileW,p.tileH);
  const z=p.zoom, fs=12/z; ctx.font=fs+'px Arial'; ctx.lineWidth=1/z;
  if(showBox){ctx.strokeStyle='rgba(255,255,255,0.45)'; ctx.fillStyle='rgba(255,255,255,0.85)';
    (st.bboxes[p.id]||[]).forEach(b=>{ctx.strokeRect(b.x1,b.y1,b.x2-b.x1,b.y2-b.y1); ctx.fillText(b.name.replace('cow',''),b.x1,b.y1-2/z);});}
  curves.filter(k=>k.tgt===p.id).forEach(k=>{
    ctx.strokeStyle=CAMCOL[k.src]; ctx.lineWidth=2/z; ctx.beginPath();
    k.poly.forEach((q,i)=>{if(i===0)ctx.moveTo(q[0],q[1]); else ctx.lineTo(q[0],q[1]);}); ctx.stroke();
    ctx.fillStyle=CAMCOL[k.src];
    k.ticks.forEach(t=>{ctx.beginPath();ctx.arc(t.u,t.v,3/z,0,7);ctx.fill(); ctx.fillText(t.h+'m',t.u+5/z,t.v-4/z);});});
  cows.forEach((c,i)=>{const q=c.clicks[p.id]; if(!q)return; const isCur=i===cur;
    ctx.strokeStyle=isCur?'#ffff00':'#ff66ff'; ctx.lineWidth=(isCur?2.5:1.5)/z;
    const r=(isCur?9:6)/z; ctx.beginPath(); ctx.arc(q[0],q[1],r,0,7); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(q[0]-r*1.6,q[1]); ctx.lineTo(q[0]+r*1.6,q[1]); ctx.moveTo(q[0],q[1]-r*1.6); ctx.lineTo(q[0],q[1]+r*1.6); ctx.stroke();
    ctx.fillStyle=ctx.strokeStyle; ctx.font=(14/z)+'px Arial'; ctx.fillText(c.id,q[0]+r*1.7,q[1]-r);});
  const c=curCow(); const r=c?results[c.id]:null;
  if(r&&r.pred_px&&r.pred_px[p.id]&&!c.clicks[p.id]){const q=r.pred_px[p.id]; const s=12/z;
    ctx.strokeStyle='#ff9900'; ctx.lineWidth=3/z; ctx.beginPath();
    ctx.moveTo(q[0]-s,q[1]-s);ctx.lineTo(q[0]+s,q[1]+s);ctx.moveTo(q[0]+s,q[1]-s);ctx.lineTo(q[0]-s,q[1]+s);ctx.stroke();}
}

function renderTable(){
  let h='<tr><th>id</th><th>名前</th><th>姿勢</th><th>Lv</th><th>カメラ</th><th>交差角</th><th>残差m</th><th>高さm</th><th>x</th><th>y</th><th>手動GTとの距離m</th><th>再投影px</th><th>位置σm</th><th>状態</th><th></th></tr>';
  cows.forEach((c,i)=>{const r=results[c.id]||{}; const fl=(r.flags||[]), inf=(r.info||[]);
    const cams=(r.cams||Object.keys(c.clicks)).map(x=>x.replace('Camera_','C')).join(',');
    const rp=r.reproj_px?Object.keys(r.reproj_px).map(k=>k.replace('Camera_','C')+':'+(r.reproj_px[k]??'-')).join(' '):'';
    const opt=(v,t)=>'<option value="'+v+'"'+((c.pose||'')===v?' selected':'')+'>'+t+'</option>';
    let stc='', stt='';
    if(fl.length){stc='warn';stt=fl.join(' / ');} else if(r.level===2){stc='info';stt=inf.join(' / ');} else if(r.X){stc='ok';stt='OK';}
    h+='<tr class="'+(i===cur?'sel':'')+'" onclick="selCow('+i+')"><td>'+c.id+'</td>'+
      '<td><input class="nm" value="'+(c.name||'')+'" onclick="event.stopPropagation()" onchange="cows['+i+'].name=this.value;setDirty(true);"></td>'+
      '<td><select onclick="event.stopPropagation()" onchange="cows['+i+'].pose=this.value;setDirty(true);refresh();">'+
        opt('','-')+opt('standing','standing')+opt('lying','lying')+'</select></td>'+
      '<td>'+(r.level||'')+'</td><td>'+cams+'</td><td>'+(r.max_angle_deg??'')+'</td><td>'+(r.resid_m??'')+'</td><td>'+(r.height_m??'')+'</td>'+
      '<td>'+(r.X?r.X[0].toFixed(2):'')+'</td><td>'+(r.X?r.X[1].toFixed(2):'')+'</td><td>'+(r.nn_hand_m??'')+'</td><td>'+rp+'</td>'+
      '<td>'+(r.pos_sigma_m??'')+'</td><td class="'+stc+'">'+stt+'</td>'+
      '<td><button onclick="event.stopPropagation();delCow('+i+')">削除</button></td></tr>';});
  document.getElementById('tbl').innerHTML=h;
  const sv=cows.map(c=>results[c.id]).filter(r=>r&&r.X); const l1=sv.filter(r=>r.level===1), l2=sv.filter(r=>r.level===2);
  const nn=l1.map(r=>r.nn_hand_m).filter(x=>x!=null).sort((a,b)=>a-b);
  const med=a=>a.length?a[Math.floor(a.length/2)]:null;
  const good=l1.filter(r=>!(r.flags||[]).length).length;
  const stx=['standing','lying'].filter(k=>stats[k]).map(k=>k+' '+stats[k].median.toFixed(2)+'m (±'+stats[k].std.toFixed(2)+', n='+stats[k].n+', '+stats[k].source+')').join(' / ');
  document.getElementById('sum').textContent='レベル1(三角測量)='+l1.length+' (警告なし '+good+')  レベル2(単眼)='+l2.length+
    '  手動GTとの最近傍距離[Lv1]: 平均='+(nn.length?(nn.reduce((a,b)=>a+b,0)/nn.length).toFixed(3):'-')+'m 中央値='+(med(nn)??'-')+'m (n='+nn.length+')   背中の高さ: '+stx;
}

function drawBev(){
  if(!bevImg.complete||!bevImg.naturalWidth)return;
  if(bev.width!==bevImg.naturalWidth){bev.width=bevImg.naturalWidth;bev.height=bevImg.naturalHeight;}
  bctx.drawImage(bevImg,0,0); const S=st.bev.scale;
  bctx.lineWidth=1.5; bctx.font='12px Arial';
  st.bev.hand.forEach(h=>{bctx.strokeStyle='#999';bctx.beginPath();bctx.arc(h[0]*S,h[1]*S,6,0,7);bctx.stroke();});
  cows.forEach((c,i)=>{const r=results[c.id]; if(!r||!r.X)return; const x=r.X[0]*S,y=r.X[1]*S;
    if(r.nn_hand_m!=null&&st.bev.hand.length){let bd=1e9,bh=null; st.bev.hand.forEach(h=>{const d=Math.hypot(h[0]-r.X[0],h[1]-r.X[1]); if(d<bd){bd=d;bh=h;}});
      bctx.strokeStyle='rgba(255,255,255,0.5)';bctx.beginPath();bctx.moveTo(x,y);bctx.lineTo(bh[0]*S,bh[1]*S);bctx.stroke();}
    if(r.level===2){
      if(r.pos_sigma_m){bctx.strokeStyle='rgba(80,200,255,0.5)';bctx.lineWidth=1;bctx.beginPath();bctx.arc(x,y,Math.max(8,r.pos_sigma_m*S),0,7);bctx.stroke();}
      bctx.fillStyle='#4cc8ff';bctx.beginPath();bctx.moveTo(x,y-8);bctx.lineTo(x+8,y);bctx.lineTo(x,y+8);bctx.lineTo(x-8,y);bctx.closePath();bctx.fill();
      bctx.strokeStyle=i===cur?'#fff':'#000';bctx.lineWidth=i===cur?2.5:1;bctx.stroke();
    } else {
      bctx.fillStyle=(r.flags||[]).length?'#ffa500':'#44dd66'; bctx.beginPath();bctx.arc(x,y,6,0,7);bctx.fill();
      bctx.strokeStyle=i===cur?'#fff':'#000';bctx.lineWidth=i===cur?2.5:1;bctx.beginPath();bctx.arc(x,y,8,0,7);bctx.stroke();
    }
    bctx.fillStyle='#ff0';bctx.fillText(c.id,x+9,y-7);});
  bctx.fillStyle='#ccc'; bctx.fillText('灰の輪=BEV手動GT  緑/橙=三角測量Lv1(橙は警告あり)  水色のひし形=単眼推定Lv2(輪=位置σ)',8,bev.height-8);
}

async function save(){
  try{const r=await post('/save',{cows:plain()},20000); setDirty(false); setMsg('保存しました: 牛'+r.n_cows+'頭、BEVGT '+r.n_solved+'点');}
  catch(e){alert('保存に失敗しました(サーバーに接続できない可能性)。画面は閉じず「手元にDL」で控えを取ってください。');}}
function dl(){const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([JSON.stringify({frame:st.frame,cows:plain()},null,2)],{type:'application/json'}));
  a.download=st.frame_name+'_tri_gt_backup.json';a.click();URL.revokeObjectURL(a.href);}

window.addEventListener('keydown',ev=>{ if(ev.target.tagName==='INPUT')return;
  if(ev.key==='n')newCow(); else if(ev.key==='u')undo(); else if(ev.key==='b')toggleBox(); else if(ev.key==='s')save(); else if(ev.key==='w')cycleWheel();});
window.addEventListener('beforeunload',ev=>{if(dirty){ev.preventDefault();ev.returnValue='';}});
window.addEventListener('resize',()=>{ORDER.forEach(id=>{sizeCanvas(P[id]);drawPanel(P[id]);});});

(async()=>{
  st=await (await fetch('/state')).json();
  document.getElementById('title').textContent='frame '+st.frame_name;
  Object.keys(st.cams).forEach(makePanel);
  cows=st.cows; cur=cows.length?0:-1;
  bevImg.onload=()=>{drawBev();}; bevImg.src='/base.png';
  setDirty(false); refresh();
})();
</script></body></html>
"""


# ---------------------------------------------------------------------------
# サーバー
# ---------------------------------------------------------------------------
def load_cows():
    p = S["json_path"]
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        print(f"既存の保存を読み込みました: {p} ({len(data.get('cows', []))}頭)")
        return [{"id": int(c["id"]), "name": c.get("name", ""), "pose": c.get("pose", ""), "clicks": c.get("clicks", {})}
                for c in data.get("cows", [])]
    return []


def save_all(cows):
    S["out_dir"].mkdir(parents=True, exist_ok=True)
    results, stats = solve_all(S["models"], cows, S["hand_xy"])
    solved, cow_out = [], []
    for c, r in zip(cows, results):
        cow_out.append({"id": int(c["id"]), "name": c.get("name", ""), "pose": c.get("pose", ""),
                        "clicks": c.get("clicks", {}), "result": r})
        if r.get("X") is not None:
            pt = {"id": int(c["id"]), "x_m": r["X"][0], "y_m": r["X"][1], "name": c.get("name", ""),
                  "pose": c.get("pose", ""), "height_m": r["height_m"], "level": r["level"],
                  "status": " / ".join(r["flags"] + r["info"]) if (r["flags"] or r["info"]) else "OK"}
            if r["level"] == 1:
                pt["max_angle_deg"] = r["max_angle_deg"]
            else:
                pt["pos_sigma_m"] = r["pos_sigma_m"]
            solved.append(pt)
    S["json_path"].write_text(json.dumps({"frame": S["frame"], "height_stats": stats, "cows": cow_out},
                                         ensure_ascii=False, indent=2), encoding="utf-8")
    meta = {"frame": S["frame"], "floor_size_m": S["floor"]}
    S["export_path"].write_text(json.dumps({**meta, "points": solved}, ensure_ascii=False, indent=2), encoding="utf-8")
    lv1 = [p for p in solved if p["level"] == 1]
    S["export_lv1_path"].write_text(json.dumps({**meta, "points": lv1}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"保存: {S['json_path']} ({len(cows)}頭)\n      {S['export_path']} (BEVGT {len(solved)}点: "
          f"レベル1={len(lv1)}, レベル2={len(solved) - len(lv1)})\n      {S['export_lv1_path']}")
    return len(cows), len(solved)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, ctype, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, "application/json", json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", HTML.replace("__ZOOM_SENS__", str(ZOOM_SENS)).replace("__WHEEL_STEP__", str(WHEEL_STEP)).encode("utf-8"))
        elif path == "/base.png":
            self._send(200, "image/png", S["base_png"])
        elif path.startswith("/cam/") and path.endswith(".jpg"):
            cid = path[len("/cam/"):-len(".jpg")]
            if cid in S["jpgs"]:
                self._send(200, "image/jpeg", S["jpgs"][cid])
            else:
                self._send(404, "text/plain", b"no such camera")
        elif path == "/state":
            self._json({
                "frame": S["frame"], "frame_name": S["frame_name"],
                "cams": {cid: {"w": m.width, "h": m.height} for cid, m in S["models"].items()},
                "bboxes": S["bboxes"],
                "bev": {"scale": S["bev_scale"], "floor": S["floor"], "hand": S["hand"]},
                "cows": load_cows(),
            })
        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n).decode("utf-8"))
            if self.path == "/solve":
                results, stats = solve_all(S["models"], body["cows"], S["hand_xy"])
                self._json({"results": results, "stats": stats})
            elif self.path == "/epi":
                self._json({"curves": epipolar_curves(S["models"], body["sources"])})
            elif self.path == "/save":
                n_cows, n_solved = save_all(body["cows"])
                self._json({"ok": True, "n_cows": n_cows, "n_solved": n_solved})
            else:
                self._send(404, "text/plain", b"not found")
        except Exception as e:
            print(f"エラー({self.path}): {e}")
            self._json({"ok": False, "error": str(e)}, 500)

    def log_message(self, fmt, *args):
        pass


def parse_bboxes(xml_path, tile_w, n_cam, cam_ids):
    out = {cid: [] for cid in cam_ids}
    for obj in ET.parse(xml_path).getroot().findall("object"):
        b = obj.find("bndbox")
        x1, y1, x2, y2 = (float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        ci = max(0, min(int(((x1 + x2) / 2.0) // tile_w), n_cam - 1))
        off = ci * tile_w
        out[cam_ids[ci]].append({"name": obj.find("name").text, "x1": x1 - off, "y1": y1,
                                 "x2": x2 - off, "y2": y2})
    return out


def main():
    import annotate_bev_gt_click_ver1 as a
    import phase3_bev_multicam_gt_zonepose_ver1 as g
    from Mootrack_core import load_floorplan

    frame_idx = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FRAME
    port = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_PORT
    cam_ids = list(g.CAM_IDS)
    n_cam = len(cam_ids)

    image_dir = getattr(a, "IMAGE_DIR", None) or getattr(g, "IMAGE_DIR", None) or FALLBACK_IMAGE_DIR
    frame_name = g.frame_name(frame_idx)
    image_path = Path(image_dir) / f"{frame_name}.jpg"
    xml_path = Path(g.XML_DIR) / f"{frame_name}.xml"
    full = cv2.imread(str(image_path))
    if full is None:
        raise SystemExit(f"画像を読み込めません: {image_path}")
    tile_w = g.IMAGE_WIDTH_PX / n_cam
    tile_h = full.shape[0]
    print(f"画像: {image_path}  タイル {int(tile_w)} x {tile_h}")

    regions, cameras_json, real_w, real_h = load_floorplan(g.FLOORPLAN_JSON_PATH)
    poses = g.resolve_poses(g.apply_camera_overrides(cameras_json), root="Area1", fmt=g.ROT_FMT)
    cam_polygons = g.build_camera_polygons(poses, g.CAM_IDS, g.get_intrinsics,
                                           height_m=g.ZONE_HEIGHT_M, n_angles=g.N_ANGLES)

    models, jpgs = {}, {}
    for ci, cid in enumerate(cam_ids):
        t0 = time.time()
        models[cid] = CamModel(cid, poses[cid].R, poses[cid].t, int(tile_w), tile_h)
        crop = full[:, int(ci * tile_w):int((ci + 1) * tile_w)]
        if JPEG_SCALE != 1.0:
            crop = cv2.resize(crop, None, fx=JPEG_SCALE, fy=JPEG_SCALE, interpolation=cv2.INTER_AREA)
        jpgs[cid] = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()
        print(f"  {cid}: 探索格子を作成 ({time.time() - t0:.1f}秒)")

    base = a.build_base(regions, poses, cam_polygons, real_w, real_h)
    hand_path = a.OUT_DIR / f"{frame_name}_bev_gt_ver1.json"
    hand = []
    if hand_path.exists():
        hand = [[p["x_m"], p["y_m"], p["id"]] for p in json.loads(hand_path.read_text(encoding="utf-8"))["points"]]
        print(f"BEV手動GTを読み込みました: {hand_path} ({len(hand)}点)")
    else:
        print(f"BEV手動GTは見つかりません(比較なしで動きます): {hand_path}")

    out_dir = g.OUTPUT_DIR / "bev_gt_triangulated"
    S.update({
        "frame": frame_idx, "frame_name": frame_name, "models": models, "jpgs": jpgs,
        "base_png": cv2.imencode(".png", base)[1].tobytes(), "bev_scale": a.SCALE, "floor": [real_w, real_h],
        "bboxes": parse_bboxes(xml_path, tile_w, n_cam, cam_ids) if xml_path.exists() else {c: [] for c in cam_ids},
        "hand": hand, "hand_xy": np.array([[h[0], h[1]] for h in hand]) if hand else None,
        "out_dir": out_dir,
        "json_path": out_dir / f"{frame_name}_tri_gt_ver1.json",
        "export_path": out_dir / f"{frame_name}_bev_gt_tri_ver1.json",
        "export_lv1_path": out_dir / f"{frame_name}_bev_gt_tri_level1_ver1.json",
    })

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"\nフレーム {frame_name} の三角測量GTツールを起動しました (127.0.0.1:{port})")
    print(f"  手元のPCで:  ssh -L {port}:localhost:{port} <ユーザー>@<サーバー>")
    print(f"  ブラウザで:  http://localhost:{port}")
    print("  終了: Ctrl+C")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n終了します")
        server.server_close()


if __name__ == "__main__":
    main()