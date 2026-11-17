"""
yolo_pose_label_ver1.py   (配置: research/phase4/diagnose/ , propagate_pose_check_yolo_ver2.py と同じ場所)

YOLO の姿勢(standing / lying)を、アノテーションXMLの <pose> に書き込む。
  対象フレーム: HE_FRAME_START..HE_FRAME_END を HE_FRAME_STEP 刻み (既定 0..2000 を 100刻み)
  手順: 各フレームでYOLOを回す -> 各GT bbox を同じカメラのYOLO検出とIoUで対応づける
        -> 牛(<name>)ごとに、カメラをまたいだYOLOの姿勢を集める
     全カメラで一致         : その姿勢を <pose> に書く                 (status = agree)
     1台しか対応がない       : その姿勢を書く。ただし『単独』と記録     (status = single)
     カメラ間で食い違う      : 書かない。そのフレームの各カメラのbboxを切り取って保存 (status = disagree)
     YOLOの対応がない        : 書かない                               (status = no_det)
     XMLに元から姿勢があり、YOLOの結果と違う : 書かない(元を残す)。切り取りを保存 (status = conflict)
  出力: <POSE_OUT_DIR>/xml_pose_yolo/ (対象フレームのXML。姿勢を書いたもの), crops_review/, pose_label_ver1.csv

使い方: ANN_XML_DIR=<XMLフォルダ> uv run python ./phase4/diagnose/yolo_pose_label_ver1.py
環境変数: ANN_XML_DIR (既定 .../annotation_process/3600_frames_result/annotations) / HE_FRAME_START,END,STEP /
          IOU_THR (0.3) / CROP_MARGIN (0.2) / MAX_CROPS (1000) / POSE_OUT_DIR
注意: YOLOの姿勢を正解として書くので、『YOLOの姿勢の精度』はこのXMLでは評価できない (σ の推定用)。
      同じ牛が複数フレームに出るので、σ の推定の交差検証・ブートストラップは牛の名前ごとに分けること。
"""

import csv
import os
import sys
from collections import Counter
from pathlib import Path

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
import propagate_pose_check_yolo_ver1 as pp   # noqa: E402  (sys.path の設定・XML/IoU/切り取りの関数を使う)

FRAME_START = int(os.environ.get("HE_FRAME_START", 0))
FRAME_END = int(os.environ.get("HE_FRAME_END", 2000))
FRAME_STEP = int(os.environ.get("HE_FRAME_STEP", 100))
MAX_CROPS = int(os.environ.get("MAX_CROPS", 1000))
ANN_DIR = os.environ.get("ANN_XML_DIR", "/home/ara/myproject/project2/MooTrack360_source_code/1-hour-farm1-area1-frames/annotation_process/3600_frames_result/annotations")
OUT_DIR = Path(os.environ.get("POSE_OUT_DIR") or Path(pp.PHASE4_DIR) / "diagnose" / "pose_label")
VALID = pp.VALID


def decide(yolo_poses, existing):
    """牛1頭 (1フレーム) の判定。yolo_poses: 対応づいたYOLOの姿勢のリスト(カメラごと, 無効は除く)、existing: XMLの元の姿勢 or None
    戻り値: (status, 書く姿勢 or None)"""
    if not yolo_poses:
        return "no_det", None
    kinds = set(yolo_poses)
    if len(kinds) > 1:
        return "disagree", None
    pose = yolo_poses[0]
    if existing is not None and existing != pose:
        return "conflict", None
    return ("agree" if len(yolo_poses) >= 2 else "single"), pose


def main():
    import cv2
    from ultralytics import YOLO
    print(f"アノテーションXML: {ANN_DIR}")
    frames = pp.read_all(ANN_DIR)
    if not frames:
        raise SystemExit("XMLが見つかりません。ANN_XML_DIR を確認してください")
    ym = pp.load_yolo_module()
    from Mootrack_core import ROT_FMT, apply_camera_overrides, load_floorplan_cameras, resolve_poses
    poses = resolve_poses(apply_camera_overrides(load_floorplan_cameras(ym.FLOORPLAN_JSON_PATH)), root="Area1", fmt=ROT_FMT)
    cam_ids = list(ym.CAM_IDS) if hasattr(ym, "CAM_IDS") else sorted(poses)
    tile_w = ym.IMAGE_WIDTH_PX / len(cam_ids)
    model = YOLO(ym.YOLO_WEIGHTS_PATH)

    xml_out = OUT_DIR / "xml_pose_yolo"
    crops = OUT_DIR / "crops_review"
    xml_out.mkdir(parents=True, exist_ok=True)
    crops.mkdir(parents=True, exist_ok=True)
    target = [fr for fr in range(FRAME_START, FRAME_END + 1, FRAME_STEP) if fr in frames]
    print(f"対象 {len(target)}フレーム: {target[:3]} ... {target[-2:]}")

    rows, stat, n_saved = [], Counter(), 0
    for fr in target:
        img_path = Path(ym.IMAGE_DIR) / f"{ym.frame_name(fr)}.jpg"
        if not img_path.exists():
            print(f"  画像なし: {img_path}")
            continue
        img = cv2.imread(str(img_path))
        per_cam = ym.yolo_world_points_per_camera_with_pose(model, img_path, poses)
        path, tree, objs = frames[fr]
        by_cow = {}                                            # name -> [(obj, cam, box, yolo_pose or None)]
        for o in objs:
            nm, bnd = pp.get_name(o), o.find("bndbox")
            if not nm or bnd is None:
                continue
            box = [float(bnd.find(t).text) for t in ("xmin", "ymin", "xmax", "ymax")]
            ci = max(0, min(int((box[0] + box[2]) / 2.0 // tile_w), len(cam_ids) - 1))
            gt_local = [box[0] - ci * tile_w, box[1], box[2] - ci * tile_w, box[3]]
            it, _ = pp.match_best(gt_local, per_cam.get(cam_ids[ci], []))
            yp = it["pose"] if (it is not None and it.get("pose") in VALID) else None
            by_cow.setdefault(nm, []).append((o, cam_ids[ci], box, yp))
        for nm, items in by_cow.items():
            ex = {pp.get_pose(o) for o, _, _, _ in items} - {None}
            existing = next(iter(ex)) if len(ex) == 1 else None
            yposes = [yp for _, _, _, yp in items if yp]
            status, pose = decide(yposes, existing)
            if len(ex) > 1:
                status, pose = "conflict", None
            stat[status] += 1
            saved = []
            if pose is not None:
                for o, _, _, _ in items:
                    if pp.get_pose(o) is None:
                        pp.set_pose(o, pose)
            if status in ("disagree", "conflict") and n_saved < MAX_CROPS:
                for o, cam, box, yp in items:
                    cp = crops / f"{fr:06d}_{nm}_{cam}_YOLO-{yp or 'none'}_GT-{pp.get_pose(o) or 'none'}_{status}.jpg"
                    if pp.save_crop(img, box, f"{nm} {cam} YOLO:{yp or 'none'} GT:{pp.get_pose(o) or 'none'}", cp, pp.CROP_MARGIN):
                        saved.append(str(cp))
                        n_saved += 1
            rows.append([fr, nm, len(items), "/".join(yp or "-" for _, _, _, yp in items), status,
                         existing or "", pose or "", len(saved)])
        tree.write(xml_out / path.name, encoding="utf-8", xml_declaration=True)
        print(f"  frame {fr}: 牛 {len(by_cow)}頭 / 累計 " + " ".join(f"{k}={v}" for k, v in sorted(stat.items())))

    with open(OUT_DIR / "pose_label_ver1.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["frame", "cow", "n_cameras", "yolo_poses(cam順)", "status", "xml_existing_pose", "written_pose", "n_crops"])
        w.writerows(rows)

    tot = sum(stat.values())
    print(f"\n■ 集計 ({tot}頭・フレーム)")
    labels = {"agree": "全カメラで一致(2台以上) -> 書いた", "single": "1台のみ -> 書いた(単独)", "disagree": "カメラ間で食い違い -> 書かない",
              "conflict": "XMLの元の姿勢と食い違い -> 書かない", "no_det": "YOLOの対応なし -> 書かない"}
    for k in ("agree", "single", "disagree", "conflict", "no_det"):
        print(f"  {labels[k]:<34}{stat.get(k, 0):>6}  ({stat.get(k, 0) / max(tot, 1) * 100:.1f}%)")
    two = stat.get("agree", 0) + stat.get("disagree", 0)
    print(f"  2台以上の牛 {two}頭のうち、一致した割合: {stat.get('agree', 0) / max(two, 1) * 100:.1f}%")
    print(f"切り取り {n_saved}枚 -> {crops}")
    print(f"姿勢を書いたXML -> {xml_out}\n一覧 -> {OUT_DIR / 'pose_label_ver1.csv'}")


if __name__ == "__main__":
    main()