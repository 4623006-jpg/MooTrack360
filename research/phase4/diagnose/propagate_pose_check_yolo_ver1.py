"""
propagate_pose_check_yolo_ver2.py   (配置: research/phase4/diagnose/)

1. アノテーションXML(全フレーム)で、牛(<name>)ごとに standing / lying の <pose> が付いているフレームから、
   付いていないフレームの同じ牛へ姿勢をコピーする (時間的にいちばん近い『姿勢ありフレーム』の値。同距離なら前のフレーム)。
   結果は別フォルダ(xml_pose_filled/)に書く。ANN_OVERWRITE=1 なら元のXMLを上書きする (コピーしたデータ用)。
2. 補完後のGT姿勢と、YOLOが検出した姿勢が違うbboxを、画像から切り取って保存する。
     - GTのbboxと、同じカメラのYOLO検出を IoU >= IOU_THR で対応づける (最も重なる検出)
     - GT姿勢 != YOLO姿勢 なら crops/ に保存。ファイル名にフレーム・牛・カメラ・GT姿勢・YOLO姿勢を入れる
     - YOLOに対応する検出がないbboxは、別に数える (保存しない。HE_SAVE_MISSED=1 なら保存)
3. 一覧CSVと集計 (GT姿勢 x YOLO姿勢の表。『元から付いていた姿勢』と『補完した姿勢』を分ける) を出す。
   途中で姿勢が変わる牛 (standing -> lying など) も一覧にする: 本当に姿勢が変わった牛か、アノテーションミスかの確認用。

ver2 (ver1 からの変更): ①YOLO は ultralytics から直接読む (phase3 モジュールに YOLO 属性が無いため)
                      ②姿勢のラベルが1つも見つからないときは、XMLの中身(objectの子タグと値)を表示して止まる

使い方: uv run python ./phase4/diagnose/propagate_pose_check_yolo_ver2.py
環境変数:
  ANN_XML_DIR    : アノテーションXMLのフォルダ (既定 .../annotation_process/xml-1fps)
  ANN_OVERWRITE  : 1 でXMLを上書き (既定 0: <出力>/xml_pose_filled/ に書く)
  HE_FRAME_START / HE_FRAME_END / HE_FRAME_STEP : YOLOを回すフレーム (既定 0 / 2600 / 10。補完は全XMLで行う)
  IOU_THR        : GTとYOLOの対応づけのIoU (既定 0.3)
  MAX_CROPS      : 保存する切り取りの最大数 (既定 500)
  CROP_MARGIN    : 切り取りの余白 (bboxの幅・高さに対する割合, 既定 0.2)
  HE_SAVE_MISSED : 1 でYOLOに検出されなかったbboxも保存 (既定 0)
  POSE_OUT_DIR   : 出力先 (既定 phase4/diagnose/pose_check)
出力: <POSE_OUT_DIR>/{xml_pose_filled/, crops/, pose_check_ver2.csv, pose_changes_ver2.csv}
"""

import csv
import importlib
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESEARCH_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
PHASE4_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
MOOTRACK_CODE_DIR = (os.path.abspath(os.path.join(RESEARCH_DIR, "calibration")))
for d in (RESEARCH_DIR, PHASE4_DIR, os.path.join(PHASE4_DIR, "web_annotate"),
          os.path.join(PHASE4_DIR, "calibration"), os.path.join(PHASE4_DIR, "calibration_try"),
          os.path.join(PHASE4_DIR, "he_kai"),
          MOOTRACK_CODE_DIR,
          os.path.join(RESEARCH_DIR, "analyze"), os.path.join(RESEARCH_DIR, "annotate"),
          os.path.join(RESEARCH_DIR, "phase3_multi_camera"), THIS_DIR):
    if os.path.isdir(d) and d not in sys.path:
        sys.path.insert(0, d)

ANN_XML_DIR = os.environ.get("ANN_XML_DIR", "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/annotation_process/xml-1fps")
OVERWRITE = os.environ.get("ANN_OVERWRITE", "0") == "1"
FRAME_START = int(os.environ.get("HE_FRAME_START", 0))
FRAME_END = int(os.environ.get("HE_FRAME_END", 2600))
FRAME_STEP = int(os.environ.get("HE_FRAME_STEP", 10))
IOU_THR = float(os.environ.get("IOU_THR", 0.3))
MAX_CROPS = int(os.environ.get("MAX_CROPS", 500))
CROP_MARGIN = float(os.environ.get("CROP_MARGIN", 0.2))
SAVE_MISSED = os.environ.get("HE_SAVE_MISSED", "0") == "1"
OUT_DIR = Path(os.environ.get("POSE_OUT_DIR") or Path(PHASE4_DIR) / "diagnose" / "pose_check")
VALID = ("standing", "lying")
YOLO_MODULE_CANDIDATES = ["phase3_bev_multicam_yolo_check_zonepose_ver6", "phase3_bev_multicam_yolo_check_zonepose_ver3",
                          "phase3_bev_multicam_yolo_check_zonepose_ver1"]


# ---------------------------------------------------------------------------
# XML 読み込み / 姿勢の補完
# ---------------------------------------------------------------------------
def frame_of(path):
    m = re.findall(r"\d+", Path(path).stem)
    return int(m[-1]) if m else None


def read_all(xml_dir):
    """{frame: (path, tree, [object要素])}"""
    out = {}
    for p in sorted(Path(xml_dir).glob("*.xml")):
        fr = frame_of(p)
        if fr is None:
            continue
        tree = ET.parse(p)
        out[fr] = (p, tree, tree.getroot().findall("object"))
    return out


def get_pose(obj):
    pe = obj.find("pose")
    t = pe.text.strip().lower() if pe is not None and pe.text else None
    return t if t in VALID else None


def get_name(obj):
    ne = obj.find("name")
    return ne.text.strip() if ne is not None and ne.text else None


def set_pose(obj, pose):
    pe = obj.find("pose")
    if pe is None:
        pe = ET.Element("pose")
        ne = obj.find("name")
        obj.insert(list(obj).index(ne) + 1 if ne is not None else 0, pe)
    pe.text = pose


def propagate(frames):
    """牛ごとに、姿勢ありフレームの値を、姿勢なしフレームへコピー(時間的にいちばん近いもの)。
    戻り値: (補完した数, 補完できなかった(姿勢が一度も無い)牛, 姿勢が変わる牛 {name: [(frame, pose)...]}, source {(frame, idx): 'original'|'propagated'})"""
    by_cow = defaultdict(list)                       # name -> [(frame, idx, pose or None)]
    for fr, (_, _, objs) in frames.items():
        for k, o in enumerate(objs):
            nm = get_name(o)
            if nm:
                by_cow[nm].append((fr, k, get_pose(o)))
    source, n_fill, unresolved, changes = {}, 0, [], {}
    for nm, lst in by_cow.items():
        lst.sort()
        labeled = [(fr, p) for fr, _, p in lst if p]
        if not labeled:
            unresolved.append(nm)
            continue
        seq = []
        for fr, p in labeled:                        # 姿勢が変わる所を記録
            if not seq or seq[-1][1] != p:
                seq.append((fr, p))
        if len(seq) > 1:
            changes[nm] = seq
        lf = np.array([fr for fr, _ in labeled])
        for fr, k, p in lst:
            if p:
                source[(fr, k)] = "original"
                continue
            j = int(np.argmin(np.abs(lf - fr) + 1e-3 * (lf > fr)))      # 同距離なら前のフレーム
            set_pose(frames[fr][2][k], labeled[j][1])
            source[(fr, k)] = "propagated"
            n_fill += 1
    return n_fill, unresolved, changes, source


def write_xmls(frames, out_dir, overwrite):
    out_dir = Path(out_dir)
    if not overwrite:
        out_dir.mkdir(parents=True, exist_ok=True)
    for fr, (p, tree, _) in frames.items():
        dst = p if overwrite else out_dir / p.name
        tree.write(dst, encoding="utf-8", xml_declaration=True)


# ---------------------------------------------------------------------------
# YOLO との照合
# ---------------------------------------------------------------------------
def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match_best(gt_local, dets):
    """GT(タイル内座標)に最も重なるYOLO検出。戻り値: (item or None, iou)"""
    best, best_iou = None, 0.0
    for it in dets:
        v = iou(gt_local, it["bbox_local"])
        if v > best_iou:
            best, best_iou = it, v
    return (best, best_iou) if best_iou >= IOU_THR else (None, best_iou)


def load_yolo_module():
    last = None
    for name in YOLO_MODULE_CANDIDATES:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as e:
            last = e
    raise last


def save_crop(img, box, tag, path, margin):
    import cv2
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    x1, y1, x2, y2 = int(max(0, x1 - mx)), int(max(0, y1 - my)), int(min(w, x2 + mx)), int(min(h, y2 + my))
    crop = img[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return False
    cv2.putText(crop, tag, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), crop)
    return True


def describe_tags(frames, n_show=8):
    """objectの子タグ(入れ子も含む)ごとに、値の種類と件数を表示 (姿勢がどのタグに入っているかを探す用)"""
    vals = defaultdict(Counter)

    def walk(el, path):
        for ch in el:
            if ch.tag == "bndbox":
                continue
            key = f"{path}/{ch.tag}" if path else ch.tag
            txt = (ch.text or "").strip()
            if len(ch) == 0:
                vals[key]["(牛の名前)" if key == "name" else txt] += 1
            walk(ch, key)

    for fr, (_, _, objs) in list(frames.items())[:200]:
        for o in objs:
            walk(o, "")
    for tag, c in vals.items():
        top = ", ".join(f"{k!r}:{v}" for k, v in c.most_common(n_show) if v)
        print(f"    <{tag}>  {top}")


def main():
    print(f"アノテーションXML: {ANN_XML_DIR}")
    frames = read_all(ANN_XML_DIR)
    if not frames:
        raise SystemExit("XMLが見つかりません。ANN_XML_DIR を確認してください")
    n_obj = sum(len(v[2]) for v in frames.values())
    n_pose = sum(1 for v in frames.values() for o in v[2] if get_pose(o))
    print(f"XML {len(frames)}ファイル / bbox {n_obj} / 姿勢あり {n_pose} ({n_pose / max(n_obj, 1) * 100:.1f}%)")

    if n_pose == 0:
        print("\n姿勢(standing / lying)の付いたbboxが1つもありません。objectの子タグと値(最初の200ファイル):")
        describe_tags(frames)
        print("\n姿勢のラベルが入っているXMLのフォルダが別にあるかもしれません。ANN_XML_DIR でそのフォルダを指定してください。")
        print("確認コマンド例: grep -rhoE \"<pose>[^<]*</pose>\" <XMLフォルダ> | sort | uniq -c")
        return

    n_fill, unresolved, changes, source = propagate(frames)
    print(f"\n■ 姿勢の補完: {n_fill}個のbboxにコピー / 姿勢が一度も無い牛 {len(unresolved)}頭 {unresolved[:10]}{' ...' if len(unresolved) > 10 else ''}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "pose_changes_ver2.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["cow", "pose_sequence(frame:pose)"])
        for nm, seq in sorted(changes.items()):
            w.writerow([nm, " -> ".join(f"{fr}:{p}" for fr, p in seq)])
    print(f"  途中で姿勢が変わる牛: {len(changes)}頭 (一覧: pose_changes_ver2.csv)  ※ 本当の変化か、ラベルミスかの確認用")
    for nm, seq in list(sorted(changes.items()))[:10]:
        print(f"    {nm}: " + " -> ".join(f"{fr}:{p}" for fr, p in seq))
    write_xmls(frames, OUT_DIR / "xml_pose_filled", OVERWRITE)
    print(f"  補完したXML: {'元のXMLを上書き' if OVERWRITE else OUT_DIR / 'xml_pose_filled'}")

    # --- YOLO と照合 ---
    import cv2
    from ultralytics import YOLO
    ym = load_yolo_module()
    from Mootrack_core import ROT_FMT, apply_camera_overrides, load_floorplan_cameras, resolve_poses
    poses = resolve_poses(apply_camera_overrides(load_floorplan_cameras(ym.FLOORPLAN_JSON_PATH)), root="Area1", fmt=ROT_FMT)
    cam_ids = list(ym.CAM_IDS) if hasattr(ym, "CAM_IDS") else sorted(poses)
    tile_w = ym.IMAGE_WIDTH_PX / len(cam_ids)
    model = YOLO(ym.YOLO_WEIGHTS_PATH)
    crops_dir = OUT_DIR / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    rows, conf = [], Counter()
    n_saved = 0
    target = [fr for fr in range(FRAME_START, FRAME_END + 1, FRAME_STEP) if fr in frames]
    print(f"\n■ YOLO と照合: {len(target)}フレーム")
    for fr in target:
        img_path = Path(ym.IMAGE_DIR) / f"{ym.frame_name(fr)}.jpg"
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        per_cam = ym.yolo_world_points_per_camera_with_pose(model, img_path, poses)
        for k, o in enumerate(frames[fr][2]):
            nm, gp = get_name(o), get_pose(o)
            bnd = o.find("bndbox")
            if not nm or gp is None or bnd is None:
                continue
            box = [float(bnd.find(t).text) for t in ("xmin", "ymin", "xmax", "ymax")]
            ci = max(0, min(int((box[0] + box[2]) / 2.0 // tile_w), len(cam_ids) - 1))
            cam = cam_ids[ci]
            gt_local = [box[0] - ci * tile_w, box[1], box[2] - ci * tile_w, box[3]]
            it, v = match_best(gt_local, per_cam.get(cam, []))
            src = source.get((fr, k), "original")
            yp = it["pose"] if it is not None else None
            conf[(src, gp, yp or "no_det")] += 1
            crop_path = ""
            differ = it is not None and yp in VALID and yp != gp
            if (differ or (SAVE_MISSED and it is None)) and n_saved < MAX_CROPS:
                crop_path = crops_dir / f"{fr:06d}_{nm}_{cam}_GT-{gp}_YOLO-{yp or 'none'}_{src}.jpg"
                tag = f"{nm} GT:{gp}({src[:4]}) YOLO:{yp or 'none'}"
                if save_crop(img, box, tag, crop_path, CROP_MARGIN):
                    n_saved += 1
                else:
                    crop_path = ""
            rows.append([fr, nm, cam, gp, src, yp or "", round(v, 3), str(crop_path)])
        if len(rows) and fr % 100 == 0:
            print(f"  frame {fr}: 累計 {len(rows)} bbox / 切り取り {n_saved}")

    with open(OUT_DIR / "pose_check_ver2.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["frame", "cow", "camera", "gt_pose", "gt_source", "yolo_pose", "iou", "crop"])
        w.writerows(rows)

    print(f"\n■ 集計 (bbox {len(rows)}個)")
    for src in ("original", "propagated"):
        sub = {k: v for k, v in conf.items() if k[0] == src}
        tot = sum(sub.values())
        if tot == 0:
            continue
        det = sum(v for k, v in sub.items() if k[2] != "no_det")
        agree = sum(v for k, v in sub.items() if k[2] == k[1])
        print(f"  GT姿勢 = {src}: {tot}個 / YOLOに対応あり {det}個 / 姿勢が一致 {agree}個 ({agree / max(det, 1) * 100:.1f}% of 対応あり)")
        for gp in VALID:
            line = " ".join(f"YOLO={yp}: {sub.get((src, gp, yp), 0)}" for yp in (*VALID, "no_det"))
            print(f"     GT {gp:<9}-> {line}")
    print(f"\n切り取りを保存: {n_saved}枚 -> {crops_dir}")
    print(f"一覧: {OUT_DIR / 'pose_check_ver2.csv'}")
    print("※ 『propagated』で不一致が多い牛は、補完が間違っている(途中で姿勢が変わった)可能性。pose_changes_ver2.csv と見比べる。")


if __name__ == "__main__":
    main()