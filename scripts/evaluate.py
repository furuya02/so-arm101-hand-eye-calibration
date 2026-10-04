"""求めたカメラ位置で、ロボット本体の関節位置を実写画像に重ねて評価する

マーカーの再投影誤差は、カメラ位置の誤差をマーカーの取り付け位置の推定が吸収してしまい、評価に使えない（simulate.py 参照）。
そこでマーカーを使わず、URDF の関節位置を投影して、画像上で手で打った関節位置とのズレ [px] を測る。

  $ python scripts/evaluate.py --click     # 1 枚ごとに 6 点（shoulder_pan -> gripper_frame の順）をクリック
  $ python scripts/evaluate.py --before data/before.json   # 簡易的な方法で求めた T_base_cam と比べる
"""
import argparse
import json

import cv2
import numpy as np

from common import DATA, load_intrinsics, to_urdf_rad
from fk import JOINTS, keypoints

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", default=str(DATA / "pairs"))
ap.add_argument("--click", action="store_true")
ap.add_argument("--before", help="比較用の T_base_cam（{'T_base_cam': 4x4}）")
a = ap.parse_args()

K, dist, _ = load_intrinsics()
rows = [json.loads(line) for line in open(f"{a.pairs}/pairs.jsonl")]
clicks_path = DATA / "clicks.json"


def project(T_base_cam, P):
    T = np.linalg.inv(T_base_cam)
    img, _ = cv2.projectPoints(P, cv2.Rodrigues(T[:3, :3])[0], T[:3, 3], K, dist)
    return img.reshape(-1, 2)


if a.click:
    clicks = {}
    for r in rows:
        img, pts = cv2.imread(f"{a.pairs}/{r['image']}"), []
        cv2.imshow("click", img)
        cv2.setMouseCallback("click", lambda e, x, y, *_: e == cv2.EVENT_LBUTTONDOWN and pts.append([x, y]))
        while len(pts) < 6 and cv2.waitKey(20) != 27:
            pass
        if len(pts) == 6:
            clicks[r["image"]] = pts
    clicks_path.write_text(json.dumps(clicks))

clicks = json.loads(clicks_path.read_text())
cands = {k: np.array(v) for k, v in json.loads((DATA / "cam2base.json").read_text())["all"].items()}
if a.before:
    cands["before"] = np.array(json.loads(open(a.before).read())["T_base_cam"])

for name, T in cands.items():
    d = [np.linalg.norm(project(T, keypoints(to_urdf_rad([r["q_deg"][j] for j in JOINTS]))) - np.array(clicks[r["image"]]), axis=1).mean()
         for r in rows if r["image"] in clicks]
    print(f"{name:11s} 関節点のズレ 平均 {np.mean(d):5.1f} px / 最大 {np.max(d):5.1f} px（{len(d)} 枚）")

# 1 枚目に骨格を重ねた画像を保存（緑: 採用した手法 / 赤: before）
r = rows[0]
img = cv2.imread(f"{a.pairs}/{r['image']}")
q = to_urdf_rad([r["q_deg"][j] for j in JOINTS])
for name, color in [(json.loads((DATA / "cam2base.json").read_text())["method"], (0, 255, 0)), ("before", (0, 0, 255))]:
    if name in cands:
        p = project(cands[name], keypoints(q)).astype(int)
        cv2.polylines(img, [p], False, color, 2)
        for x, y in p:
            cv2.circle(img, (x, y), 4, color, -1)
cv2.imwrite(str(DATA / "overlay.png"), img)
