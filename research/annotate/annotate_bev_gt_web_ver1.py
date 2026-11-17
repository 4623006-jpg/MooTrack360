"""
annotate_bev_gt_web_ver1.py

annotate_bev_gt_click_ver1.py のブラウザ版。ディスプレイの無いサーバーで動かし、
手元のPCのブラウザでBEVをクリックして、指定フレームのBEVのGTを作る。
GUI版のOpenCV(imshow)は不要。追加ライブラリも不要(標準ライブラリのhttp.serverだけ使う)。

annotate_bev_gt_click_ver1.py と同じフォルダに置く(背景描画・参考GT・カメラ画像の関数を再利用する)。
保存形式・保存先もGUI版と同じ:
    OUTPUT_DIR/bev_gt_annotation/{frame}_bev_gt_ver1.json / .png
同名のjsonがあれば起動時に読み込んで続きから編集できる。

使い方:
  1. サーバーで起動
        uv run python ../annotate/annotate_bev_gt_web_ver1.py 100        # フレーム100
        uv run python ../annotate/annotate_bev_gt_web_ver1.py 100 8123   # ポート指定(省略時8000)
  2. 手元のPCで、別のターミナルからポート転送
        ssh -L 8000:localhost:8000 ara@vml36
     (ポートを変えたら両方同じ番号にする)
  3. 手元のブラウザで http://localhost:8000 を開く

操作(ブラウザ上):
  左クリック           : その地点に牛を追加(番号は自動で 1,2,3...)
  左ドラッグ(点の近く) : 既存の点を動かす
  右クリック           : いちばん近い点を削除
  u : 直前の追加を取り消し / g : 参考GT点の表示切替 / n : 番号の表示切替 / s : 保存
  (画面のボタンでも同じ操作ができる)
  終了はサーバー側で Ctrl+C。保存し忘れに注意(未保存のままタブを閉じると警告が出る)。

サーバーは 127.0.0.1 にだけ待ち受けるので、ssh転送した自分以外からはアクセスできない。
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYZE_DIR = os.path.join(THIS_DIR, "..", "analyze")
PHASE3_DIR = os.path.join(THIS_DIR, "..", "phase3_multi_camera")
for d in (os.path.join(THIS_DIR, "..", "calibration"), ANALYZE_DIR, THIS_DIR,PHASE3_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

import annotate_bev_gt_click_ver1 as a
import phase3_bev_multicam_gt_zonepose_ver1 as g
from Mootrack_core import load_floorplan

DEFAULT_PORT = 8123
S = {}   # サーバーの状態(起動時に作る)


HTML = r"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><title>BEV GT annotation</title>
<style>
 body{margin:8px;background:#222;color:#eee;font-family:sans-serif}
 #bar{margin-bottom:6px} button{margin-right:4px;padding:4px 10px}
 #wrap{display:inline-block;background:#1e1e1e}
 canvas{display:block;max-width:100%;cursor:crosshair}
 #cams{width:100%;margin-top:8px}
 .dirty{color:#f66}.clean{color:#6c6}
</style></head><body>
<div id="bar">
 <b id="title"></b>
 <button onclick="undo()">取消(u)</button>
 <button onclick="toggleRef()">参考GT(g)</button>
 <button onclick="toggleIds()">番号(n)</button>
 <button onclick="save()">保存(s)</button>
 <span id="status" class="clean"></span> <span id="pos"></span>
 <br><small>左クリック:追加 / 点の近くをドラッグ:移動 / 右クリック:削除</small>
</div>
<div id="wrap"><canvas id="cv"></canvas></div>
<img id="cams" src="/cameras.jpg" onerror="this.style.display='none'">
<script>
let st, pts=[], ref=[], showRef=true, showIds=true, dirty=false, drag=-1;
const cv=document.getElementById('cv'), ctx=cv.getContext('2d');
const img=new Image();
const CAM={Camera_1:'#3c78ff',Camera_2:'#3cc83c',Camera_3:'#ff8c3c'};
const PICK=0.4, DEL=0.6;

function setDirty(v){dirty=v;const s=document.getElementById('status');
  s.textContent=(v?'未保存':'保存済み')+'  点='+pts.length; s.className=v?'dirty':'clean';}
function toM(ev){const r=cv.getBoundingClientRect();
  const x=(ev.clientX-r.left)*cv.width/r.width, y=(ev.clientY-r.top)*cv.height/r.height;
  return [x/st.scale,y/st.scale];}
function inside(m){return m[0]>=0&&m[0]<=st.floor[0]&&m[1]>=0&&m[1]<=st.floor[1];}
function nearest(m){let bi=-1,bd=1e9;
  pts.forEach((p,i)=>{const d=Math.hypot(p.x-m[0],p.y-m[1]); if(d<bd){bd=d;bi=i;}});
  return [bi,bd];}
function nextId(){return pts.reduce((a,p)=>Math.max(a,p.id),0)+1;}

function draw(){
  ctx.drawImage(img,0,0);
  if(showRef){ctx.font='11px sans-serif';
    ref.forEach(r=>{const x=r.x*st.scale,y=r.y*st.scale;
      ctx.fillStyle='#999';ctx.beginPath();ctx.arc(x,y,3,0,7);ctx.fill();
      ctx.fillText(r.name.replace('cow',''),x+4,y-4);});}
  ctx.font='14px sans-serif';
  pts.forEach(p=>{const x=p.x*st.scale,y=p.y*st.scale;
    ctx.fillStyle='#ffff00';ctx.beginPath();ctx.arc(x,y,7,0,7);ctx.fill();
    ctx.strokeStyle='#fff';ctx.lineWidth=1;ctx.beginPath();ctx.arc(x,y,9,0,7);ctx.stroke();
    if(showIds){ctx.fillStyle='#ffff00';ctx.fillText(p.id,x+10,y-8);}});
}

cv.addEventListener('mousedown',ev=>{
  if(ev.button!==0)return; const m=toM(ev); if(!inside(m))return;
  const [i,d]=nearest(m);
  if(i>=0&&d<=PICK){drag=i;} else {pts.push({id:nextId(),x:m[0],y:m[1]});setDirty(true);}
  draw();});
cv.addEventListener('mousemove',ev=>{const m=toM(ev);
  document.getElementById('pos').textContent='  ('+m[0].toFixed(2)+', '+m[1].toFixed(2)+') m';
  if(drag>=0&&inside(m)){pts[drag].x=m[0];pts[drag].y=m[1];setDirty(true);draw();}});
window.addEventListener('mouseup',()=>{drag=-1;});
cv.addEventListener('contextmenu',ev=>{ev.preventDefault();
  const m=toM(ev); const [i,d]=nearest(m);
  if(i>=0&&d<=DEL){pts.splice(i,1);setDirty(true);draw();}});

function undo(){if(pts.length){pts.pop();setDirty(true);draw();}}
function toggleRef(){showRef=!showRef;draw();}
function toggleIds(){showIds=!showIds;draw();}
async function save(){
  const body={points:pts.map(p=>({id:p.id,x_m:+p.x.toFixed(4),y_m:+p.y.toFixed(4)}))};
  const r=await fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  if(r.ok){setDirty(false);} else {alert('保存に失敗しました');}}
window.addEventListener('keydown',ev=>{
  if(ev.key==='u')undo(); else if(ev.key==='g')toggleRef();
  else if(ev.key==='n')toggleIds(); else if(ev.key==='s')save();});
window.addEventListener('beforeunload',ev=>{if(dirty){ev.preventDefault();ev.returnValue='';}});

(async()=>{
  st=await (await fetch('/state')).json();
  pts=st.points.map(p=>({id:p.id,x:p.x_m,y:p.y_m})); ref=st.ref;
  document.getElementById('title').textContent='frame '+st.frame_name;
  img.onload=()=>{cv.width=img.naturalWidth;cv.height=img.naturalHeight;setDirty(false);draw();};
  img.src='/base.png';
})();
</script></body></html>
"""


def load_points():
    if S["json_path"].exists():
        data = json.loads(S["json_path"].read_text(encoding="utf-8"))
        print(f"既存の保存を読み込みました: {S['json_path']} ({len(data.get('points', []))}点)")
        return data.get("points", [])
    return []


def save_points(points):
    a.OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "frame": S["frame"],
        "scale_px_per_m": a.SCALE,
        "floor_size_m": [S["real_w"], S["real_h"]],
        "points": [{"id": int(p["id"]), "x_m": round(float(p["x_m"]), 4), "y_m": round(float(p["y_m"]), 4)}
                   for p in points],
    }
    S["json_path"].write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    canvas = S["base"].copy()
    for p in data["points"]:
        c = (int(p["x_m"] * a.SCALE), int(p["y_m"] * a.SCALE))
        cv2.circle(canvas, c, 7, a.MARK_COLOR, -1)
        cv2.circle(canvas, c, 9, (255, 255, 255), 1)
        cv2.putText(canvas, str(p["id"]), (c[0] + 10, c[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    a.MARK_COLOR, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{g.frame_name(S['frame'])}  points={len(points)}", (8, canvas.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(S["png_path"]), canvas)
    print(f"保存: {S['json_path']} ({len(points)}点)\n      {S['png_path']}")


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, ctype, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", HTML.encode("utf-8"))
        elif self.path.startswith("/base.png"):
            self._send(200, "image/png", S["base_png"])
        elif self.path.startswith("/cameras.jpg"):
            if S["cam_jpg"] is None:
                self._send(404, "text/plain", b"no camera image")
            else:
                self._send(200, "image/jpeg", S["cam_jpg"])
        elif self.path.startswith("/state"):
            state = {
                "frame": S["frame"], "frame_name": g.frame_name(S["frame"]),
                "scale": a.SCALE, "floor": [S["real_w"], S["real_h"]],
                "points": load_points(), "ref": S["ref"],
            }
            self._send(200, "application/json", json.dumps(state, ensure_ascii=False).encode("utf-8"))
        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self):
        if self.path != "/save":
            self._send(404, "text/plain", b"not found")
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n).decode("utf-8"))
            save_points(body["points"])
            self._send(200, "application/json", b'{"ok": true}')
        except Exception as e:      # 保存失敗はブラウザ側に伝える
            print(f"保存エラー: {e}")
            self._send(500, "application/json", json.dumps({"ok": False, "error": str(e)}).encode("utf-8"))

    def log_message(self, fmt, *args):
        pass    # アクセスログは出さない


def main():
    frame_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    port = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_PORT

    regions, cameras_json, real_w, real_h = load_floorplan(g.FLOORPLAN_JSON_PATH)
    poses = g.resolve_poses(g.apply_camera_overrides(cameras_json), root="Area1", fmt=g.ROT_FMT)
    cam_polygons = g.build_camera_polygons(poses, g.CAM_IDS, g.get_intrinsics,
                                           height_m=g.ZONE_HEIGHT_M, n_angles=g.N_ANGLES)
    base = a.build_base(regions, poses, cam_polygons, real_w, real_h)

    cam_img = a.build_camera_image(frame_idx)
    S.update({
        "frame": frame_idx, "real_w": real_w, "real_h": real_h, "base": base,
        "base_png": cv2.imencode(".png", base)[1].tobytes(),
        "cam_jpg": cv2.imencode(".jpg", cam_img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
        if cam_img is not None else None,
        "ref": [{"cam": cid, "name": name, "x": float(xy[0]), "y": float(xy[1])}
                for cid, name, xy in a.reference_points(frame_idx, poses)],
        "json_path": a.OUT_DIR / f"{g.frame_name(frame_idx)}_bev_gt_ver1.json",
        "png_path": a.OUT_DIR / f"{g.frame_name(frame_idx)}_bev_gt_ver1.png",
    })

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"フレーム {g.frame_name(frame_idx)} のアノテーションサーバーを起動しました (127.0.0.1:{port})")
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