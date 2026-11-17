"""
refine_calibration_ba_ver1.py

三角測量GTツール(annotate_triangulate_gt_web_*.py)で作った「牛の背中の同じ点を複数カメラでクリックした」
データを使い、カメラの外部パラメータ(位置・向き)と内部パラメータ(fx, fy, cx, cy など)を調整する
(バンドル調整)。同じ物理点を指す光線が一点で交わるように、カメラのパラメータと3D点を同時に動かす。

  観測の誤差 : クリックした画素の光線(逆投影、現在の内部・外部パラメータ)と、3D点の方向のずれ。
               焦点距離をかけて「擬似ピクセル」にそろえ、SIGMA_CLICK_PX で割って標準化する。
               ※順投影(世界座標→画素)を使わず、逆投影の光線で誤差を測るので、魚眼モデルの
                 順投影の式を別に用意する必要がない。
  事前分布   : 各パラメータの初期値からのずれに、ガウスの事前分布(PRIOR)をかける。
               (1) 画像の対応だけでは決まらない自由度(全体の位置・向き・大きさ)を、初期値の近くに留める。
               (2) クリックのずれを吸収して、過剰に合わせこむのを防ぐ。
  外れ値     : robust損失(soft_l1)で、クリックの大きな間違いの影響を抑える。
  検証       : 牛の一部(TEST_FRAC)を調整に使わず取っておき、調整前後で、その牛の
               再投影誤差(擬似ピクセル)と光線の残差[m]がどう変わるかを比べる。
               使った点の誤差は下がって当然なので、評価はこの「使っていない牛」で見る。

出力(OUTPUT_DIR / bev_calibration_refined /):
  calibration_refined_ver1.json : 調整後の各カメラの R, t, 内部パラメータ、検証の統計
  calibration_refined_ver1.txt  : Mootrack_core.py に貼れる形式
                                  (CAMERA_PARAM_OVERRIDES と CAM_INTRINSICS)
※ 元のファイルは変更しない。反映するかは自分で決める(反映すると、これまでのIDF1・MOTPなどの
  数値が変わる。結果は別の版として保存しておくこと)。

注意:
  - 外部パラメータの全体の大きさ(カメラの高さ・基線長)は、画像の対応だけでは決まらない。
    初期値の近くに留めるのは事前分布の働きで、床の目印の実測などで拘束するともっとよい。
  - fx と xi/alpha は互いに補い合うので、既定では xi, alpha は固定(FREE_INTRINSICS)。
  - 点の数が少ない、画像の一部にしか無い、交差角が小さい点ばかり、だと過剰に合わせやすい。
    数十頭以上、画像のあちこちに点があるほうがよい。検証(使っていない牛)の誤差が下がらなければ、
    調整後の値を採用しないこと。

使い方:
    python refine_calibration_ba_ver1.py            # 三角測量GTのjsonがある全フレームを使う
    python refine_calibration_ba_ver1.py 100 200    # フレームを指定
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation as Rot

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYZE_DIR = os.path.join(THIS_DIR, "..", "analyze")
ANNOTATE_DIR = os.path.join(THIS_DIR, "..", "annotate")
PHASE3_DIR = os.path.join(THIS_DIR, "..", "phase3_multi_camera")
for d in (os.path.join(THIS_DIR, "..", "calibration"), ANALYZE_DIR, ANNOTATE_DIR, PHASE3_DIR, THIS_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

from Mootrack_core import unproject_ds, get_intrinsics

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
INTR_NAMES = ("fx", "fy", "cx", "cy", "xi", "alpha")
FREE_INTRINSICS = ("fx", "fy", "cx", "cy")     # 動かす内部パラメータ。xi, alpha も動かすなら足す
FIX_EXTRINSICS_CAMERA = None                    # 外部パラメータを固定するカメラ。例: "Camera_2"。Noneは事前分布だけで留める
PRIOR = {                                       # 初期値からのずれの標準偏差(事前分布)。小さいほど動きにくい
    "rot_deg": 1.0,        # 向き
    "trans_m": 0.15,       # 位置
    "fx_ratio": 0.02,      # fx, fy(初期値に対する割合)
    "cxy_px": 20.0,        # cx, cy
    "xi": 0.05,
    "alpha": 0.05,
}
SIGMA_CLICK_PX = 4.0       # クリックのずれの大きさの見積もり(擬似ピクセル)
LOSS_F_SCALE = 3.0         # robust損失のしきい値(標準化後の単位)
TEST_FRAC = 0.25           # 検証に取っておく牛の割合
SEED = 0
MIN_ANGLE_USE_DEG = 30.0   # 初期値で交差角がこれ未満の牛は使わない
MAX_INIT_REPROJ_PX = 60.0  # 初期値で再投影誤差(擬似ピクセル)がこれを超える牛は使わない(大きなクリックミス)
HEIGHT_USE_RANGE_M = (0.1, 2.2)
MAX_NFEV = 200


# ---------------------------------------------------------------------------
# 幾何
# ---------------------------------------------------------------------------
def rays_world(R, t, intr, uvs):
    """画素の配列 -> 世界座標系の単位方向 (n,3)、有効フラグ"""
    d_cam, ok = unproject_ds(np.asarray(uvs, dtype=np.float64).reshape(-1, 2), *intr)
    return (R @ d_cam.T).T, ok


def triangulate(rays):
    """rays: [(c, d)]。光線への距離の二乗和を最小にする点。失敗はNone"""
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for c, d in rays:
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ c
    if np.linalg.cond(A) > 1e8:
        return None
    X = np.linalg.solve(A, b)
    if any(np.dot(X - c, d) <= 0 for c, d in rays):
        return None
    return X


def triangulate_cow(cams, clicks):
    """clicks: {cam: [u,v]}。戻り値 (X, [(cam, c, d)...])。失敗は (None, None)"""
    rays = []
    for cid, uv in clicks.items():
        if cid not in cams or uv is None:
            continue
        R, t, intr = cams[cid]
        d, ok = rays_world(R, t, intr, [uv])
        if ok[0]:
            rays.append((cid, t, d[0] / np.linalg.norm(d[0])))
    if len(rays) < 2:
        return None, None
    X = triangulate([(c, d) for _cid, c, d in rays])
    return X, rays


def evaluate_cows(cams, cows):
    """カメラ設定camsのもとで、各牛を三角測量し直して、再投影誤差(擬似px)と光線の残差[m]を集める"""
    reproj, resid, heights, angles = [], [], [], []
    for cow in cows:
        X, rays = triangulate_cow(cams, cow["clicks"])
        if X is None:
            continue
        for cid, c, d in rays:
            R, t, intr = cams[cid]
            F = 0.5 * (intr[0] + intr[1])
            v = X - c
            d_pred = v / np.linalg.norm(v)
            reproj.append(F * float(np.linalg.norm(d - d_pred)))
            resid.append(float(np.linalg.norm((np.eye(3) - np.outer(d, d)) @ v)))
        heights.append(float(-X[2]))
        angles.append(max(float(np.degrees(np.arccos(np.clip(np.dot(rays[i][2], rays[j][2]), -1, 1))))
                          for i in range(len(rays)) for j in range(i + 1, len(rays))))
    return {"reproj_px": np.array(reproj), "resid_m": np.array(resid),
            "height_m": np.array(heights), "angle_deg": np.array(angles)}


def stat_line(a):
    if len(a) == 0:
        return "n=0"
    return f"n={len(a)} 平均={a.mean():.3f} 中央値={np.median(a):.3f} 90%点={np.percentile(a, 90):.3f} 最大={a.max():.3f}"


# ---------------------------------------------------------------------------
# バンドル調整
# ---------------------------------------------------------------------------
class Layout:
    """パラメータベクトル x = [各カメラの(回転ベクトル3, 平行移動3, 内部パラメータの一部)のずれ..., 3D点(3N)]"""

    def __init__(self, cam_ids, free_intr, fix_cam):
        self.cam_ids = list(cam_ids)
        self.free_intr = tuple(free_intr)
        self.info = {}
        n = 0
        for c in self.cam_ids:
            d = {"rot": None, "trans": None, "intr": {}}
            if c != fix_cam:
                d["rot"] = np.arange(n, n + 3)
                n += 3
                d["trans"] = np.arange(n, n + 3)
                n += 3
            for nm in self.free_intr:
                d["intr"][nm] = n
                n += 1
            self.info[c] = d
        self.n_cam_params = n

    def cam_columns(self, c):
        d = self.info[c]
        cols = []
        if d["rot"] is not None:
            cols += list(d["rot"]) + list(d["trans"])
        cols += list(d["intr"].values())
        return cols


def camera_from_x(layout, x, base, c):
    R0, t0, i0 = base[c]
    d = layout.info[c]
    R, t = R0, t0
    if d["rot"] is not None:
        R = R0 @ Rot.from_rotvec(x[d["rot"]]).as_matrix()
        t = t0 + x[d["trans"]]
    intr = np.array(i0, dtype=np.float64)
    for nm, idx in d["intr"].items():
        intr[INTR_NAMES.index(nm)] += x[idx]
    return R, t, intr


def prior_sigmas(layout, base):
    """x_cam の各成分の事前標準偏差(layoutの並びと同じ)"""
    sig = np.ones(layout.n_cam_params)
    for c in layout.cam_ids:
        d = layout.info[c]
        if d["rot"] is not None:
            sig[d["rot"]] = np.radians(PRIOR["rot_deg"])
            sig[d["trans"]] = PRIOR["trans_m"]
        i0 = base[c][2]
        for nm, idx in d["intr"].items():
            sig[idx] = {"fx": i0[0] * PRIOR["fx_ratio"], "fy": i0[1] * PRIOR["fx_ratio"],
                        "cx": PRIOR["cxy_px"], "cy": PRIOR["cxy_px"],
                        "xi": PRIOR["xi"], "alpha": PRIOR["alpha"]}[nm]
    return sig


def bundle_adjust(base, cows, free_intr=FREE_INTRINSICS, fix_cam=FIX_EXTRINSICS_CAMERA, verbose=1):
    """base: {cam: (R, t, intr)}(初期値)。cows: [{clicks}]。戻り値: (調整後のcams, 使った牛のindex, 結果情報)"""
    cam_ids = list(base.keys())
    layout = Layout(cam_ids, free_intr, fix_cam)

    # 初期の3D点(初期値で三角測量)
    X0, used = [], []
    for k, cow in enumerate(cows):
        X, rays = triangulate_cow(base, cow["clicks"])
        if X is not None:
            X0.append(X)
            used.append(k)
    X0 = np.array(X0)
    n_pts = len(used)
    # 観測(カメラごとにまとめる)
    obs = {c: {"pt": [], "uv": []} for c in cam_ids}
    for pi, k in enumerate(used):
        for cid, uv in cows[k]["clicks"].items():
            if cid in obs and uv is not None:
                obs[cid]["pt"].append(pi)
                obs[cid]["uv"].append(uv)
    for c in cam_ids:
        obs[c]["pt"] = np.array(obs[c]["pt"], dtype=int)
        obs[c]["uv"] = np.array(obs[c]["uv"], dtype=np.float64).reshape(-1, 2)
    F = {c: 0.5 * (base[c][2][0] + base[c][2][1]) for c in cam_ids}
    sig = prior_sigmas(layout, base)
    nc = layout.n_cam_params

    def fun(x):
        xc, P = x[:nc], x[nc:].reshape(-1, 3)
        out = []
        for c in cam_ids:
            if len(obs[c]["pt"]) == 0:
                continue
            R, t, intr = camera_from_x(layout, xc, base, c)
            d_obs, ok = rays_world(R, t, intr, obs[c]["uv"])
            v = P[obs[c]["pt"]] - t
            d_pred = v / np.linalg.norm(v, axis=1, keepdims=True)
            r = F[c] * (d_obs - d_pred) / SIGMA_CLICK_PX
            r[~ok] = 0.0
            out.append(r.ravel())
        out.append(xc / sig)
        return np.concatenate(out)

    # ヤコビアンの疎構造
    n_obs_rows = 3 * sum(len(obs[c]["pt"]) for c in cam_ids)
    n_rows = n_obs_rows + nc
    S = lil_matrix((n_rows, nc + 3 * n_pts), dtype=int)
    row = 0
    for c in cam_ids:
        cols = layout.cam_columns(c)
        for pi in obs[c]["pt"]:
            pcols = [nc + 3 * pi, nc + 3 * pi + 1, nc + 3 * pi + 2]
            for r in range(3):
                for col in cols + pcols:
                    S[row + r, col] = 1
            row += 3
    for j in range(nc):
        S[n_obs_rows + j, j] = 1

    x0 = np.concatenate([np.zeros(nc), X0.ravel()])
    r0 = fun(x0)
    res = least_squares(fun, x0, jac_sparsity=S, loss="soft_l1", f_scale=LOSS_F_SCALE,
                        x_scale="jac", max_nfev=MAX_NFEV, verbose=verbose)
    xc = res.x[:nc]
    cams = {c: camera_from_x(layout, xc, base, c) for c in cam_ids}
    info = {"n_points": n_pts, "n_obs": n_obs_rows // 3, "cost0": float(0.5 * np.sum(r0 ** 2)),
            "cost": float(res.cost), "nfev": int(res.nfev), "status": int(res.status)}
    return cams, used, info


# ---------------------------------------------------------------------------
# 出力用の変換
# ---------------------------------------------------------------------------
def euler_zyx_deg(R):
    """build_R(a,b,c) = Rz(a) Ry(b) Rx(c) の (a,b,c)[deg]"""
    return Rot.from_matrix(R).as_euler("ZYX", degrees=True)


def overrides_from_cams(cameras_json, cams, root="Area1"):
    """調整後の絶対位置・向きから、floorplanのparent->child形式のparamsを作る。
    resolve_posesは child.t = parent.t + t_local、R=build_R(角度) (絶対角) なので、それに合わせる"""
    abs_t = {root: np.zeros(3)}
    for c, (R, t, _i) in cams.items():
        abs_t[c] = t
    out = {}
    for entry in cameras_json:
        parent, child = entry["parent"], entry["child"]
        if child not in cams or parent not in abs_t:
            continue
        R, t, _i = cams[child]
        t_local = t - abs_t[parent]
        a, b, c = euler_zyx_deg(R)
        out[f"{parent}->{child}"] = [round(float(v), 4) for v in (*t_local, a, b, c)]
    return out


def main():
    import phase3_bev_multicam_gt_zonepose_ver1 as g
    from Mootrack_core import load_floorplan_cameras, apply_camera_overrides, resolve_poses

    in_dir = g.OUTPUT_DIR / "bev_gt_triangulated"
    out_dir = g.OUTPUT_DIR / "bev_calibration_refined"
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 1:
        frames = [int(a) for a in sys.argv[1:]]
    else:
        frames = sorted(int(p.name[:6]) for p in in_dir.glob("*_tri_gt_ver1.json"))
    if not frames:
        raise SystemExit(f"三角測量GTのjsonが見つかりません: {in_dir}")

    cameras_json = load_floorplan_cameras(g.FLOORPLAN_JSON_PATH)
    poses = resolve_poses(apply_camera_overrides(cameras_json), root="Area1", fmt=g.ROT_FMT)
    cam_ids = list(g.CAM_IDS)
    base = {c: (poses[c].R.copy(), poses[c].t.copy(), np.array(get_intrinsics(c), dtype=np.float64))
            for c in cam_ids}
    print("floorplanのカメラ定義(parent->child):", [f"{e['parent']}->{e['child']}" for e in cameras_json])

    # 牛(クリック)を集める
    cows = []
    for fr in frames:
        data = json.loads((in_dir / f"{fr:06d}_tri_gt_ver1.json").read_text(encoding="utf-8"))
        for c in data["cows"]:
            clicks = {k: v for k, v in c.get("clicks", {}).items() if v is not None and k in cam_ids}
            if len(clicks) >= 2:
                cows.append({"frame": fr, "id": c["id"], "clicks": clicks})
    print(f"フレーム {frames}: 2台以上でクリックされた牛 {len(cows)}頭")

    # 初期値で品質フィルタ
    ev0 = None
    good = []
    for cow in cows:
        X, rays = triangulate_cow(base, cow["clicks"])
        if X is None:
            continue
        e = evaluate_cows(base, [cow])
        if (e["angle_deg"][0] >= MIN_ANGLE_USE_DEG and e["reproj_px"].max() <= MAX_INIT_REPROJ_PX
                and HEIGHT_USE_RANGE_M[0] <= e["height_m"][0] <= HEIGHT_USE_RANGE_M[1]):
            good.append(cow)
    print(f"品質フィルタ後(交差角>={MIN_ANGLE_USE_DEG}°, 初期の再投影誤差<={MAX_INIT_REPROJ_PX}px, 高さ範囲内): {len(good)}頭")
    if len(good) < 12:
        raise SystemExit("牛の数が少なすぎます(12頭以上、できれば数十頭)。もっとクリックしてから実行してください。")

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(good))
    n_test = max(3, int(round(TEST_FRAC * len(good))))
    test = [good[i] for i in idx[:n_test]]
    train = [good[i] for i in idx[n_test:]]
    print(f"調整に使う牛={len(train)}頭、検証に取っておく牛={len(test)}頭")

    cams, used, info = bundle_adjust(base, train)
    print(f"\n最適化: 点={info['n_points']} 観測={info['n_obs']} 目的関数 {info['cost0']:.1f} -> {info['cost']:.1f} (nfev={info['nfev']})")

    def report(title, cws):
        print(f"\n--- {title} ---")
        for nm, key in (("再投影誤差[擬似px]", "reproj_px"), ("光線の残差[m]", "resid_m")):
            b, a = evaluate_cows(base, cws)[key], evaluate_cows(cams, cws)[key]
            print(f"  {nm}\n    調整前: {stat_line(b)}\n    調整後: {stat_line(a)}")
        hb, ha = evaluate_cows(base, cws)["height_m"], evaluate_cows(cams, cws)["height_m"]
        print(f"  三角測量した高さ[m] 中央値: 調整前 {np.median(hb):.3f} -> 調整後 {np.median(ha):.3f}")

    report("検証用(調整に使っていない牛)", test)
    report("調整に使った牛(参考。下がって当然)", train)

    print("\n--- パラメータの変化 ---")
    for c in cam_ids:
        R0, t0, i0 = base[c]
        R1, t1, i1 = cams[c]
        rot_deg = float(np.degrees(np.linalg.norm(Rot.from_matrix(R0.T @ R1).as_rotvec())))
        print(f"  {c}: 向きの変化 {rot_deg:.3f}°  位置の変化 {np.round(t1 - t0, 4).tolist()} m")
        print("      内部: " + "  ".join(f"{nm} {i0[k]:.4f}->{i1[k]:.4f}" for k, nm in enumerate(INTR_NAMES)
                                         if nm in FREE_INTRINSICS))

    ov = overrides_from_cams(cameras_json, cams)
    intr_txt = {c: {nm: round(float(cams[c][2][k]), 6) for k, nm in enumerate(INTR_NAMES)} for c in cam_ids}
    snippet = ["# Mootrack_core.py に貼る内容(調整後)", "CAMERA_PARAM_OVERRIDES = {"]
    snippet += [f'    "{k}": {v},' for k, v in ov.items()]
    snippet += ["}", "", "CAM_INTRINSICS = {"]
    snippet += [f'    "{c}": {v},' for c, v in intr_txt.items()]
    snippet += ["}"]
    (out_dir / "calibration_refined_ver1.txt").write_text("\n".join(snippet), encoding="utf-8")

    e_test0, e_test1 = evaluate_cows(base, test), evaluate_cows(cams, test)
    (out_dir / "calibration_refined_ver1.json").write_text(json.dumps({
        "frames": frames, "n_train": len(train), "n_test": len(test), "info": info,
        "cameras": {c: {"R": cams[c][0].tolist(), "t": cams[c][1].tolist(),
                        "intrinsics": dict(zip(INTR_NAMES, map(float, cams[c][2])))} for c in cam_ids},
        "overrides": ov,
        "test_reproj_px": {"before_median": float(np.median(e_test0["reproj_px"])),
                           "after_median": float(np.median(e_test1["reproj_px"]))},
        "test_resid_m": {"before_median": float(np.median(e_test0["resid_m"])),
                         "after_median": float(np.median(e_test1["resid_m"]))},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n出力: {out_dir}")
    if np.median(e_test1["reproj_px"]) >= np.median(e_test0["reproj_px"]):
        print("※検証用の牛の誤差が下がっていません。調整後の値は採用しないでください"
              "(点の数・分布・事前分布の強さを見直す)。")


if __name__ == "__main__":
    main()