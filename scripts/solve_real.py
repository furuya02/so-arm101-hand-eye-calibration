"""実機で集めた姿勢ペアから、カメラの位置（base_link 基準）を 5 手法で解く

結果は data/cam2base.json に保存する。Isaac Lab の CameraCfg.OffsetCfg（convention="ros"）にそのまま入る形も出力する。
"""
import argparse
import json

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from common import DATA, load_intrinsics, make_T, marker_pose, to_urdf_rad
from fk import JOINTS, fk
from simulate import METHODS

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", default=str(DATA / "pairs"))
ap.add_argument("--marker-size", type=float, default=0.04)
ap.add_argument("--marker-id", type=int, default=0)
ap.add_argument("--min-ratio", type=float, default=3.0)
ap.add_argument("--method", default="PARK", help="保存に使う手法")
a = ap.parse_args()

K, dist, _ = load_intrinsics()
pairs = []
for line in open(f"{a.pairs}/pairs.jsonl"):
    r = json.loads(line)
    m = marker_pose(cv2.imread(f"{a.pairs}/{r['image']}"), K, dist, a.marker_size, a.marker_id)
    if m is None or m[1] < a.min_ratio:  # マーカーが見えない / 向きが曖昧な観測は捨てる
        print("skip", r["image"], None if m is None else round(m[1], 2))
        continue
    pairs.append(dict(T_bg=fk(to_urdf_rad([r["q_deg"][j] for j in JOINTS])), T_cm=m[0]))
print(f"{len(pairs)} pairs")

# eye-to-hand: gripper2base を反転して base2gripper として渡す -> 戻り値は cam2base
T_gb = [np.linalg.inv(p["T_bg"]) for p in pairs]
res = {}
for name, flag in METHODS.items():
    R, t = cv2.calibrateHandEye([T[:3, :3] for T in T_gb], [T[:3, 3] for T in T_gb],
                                [p["T_cm"][:3, :3] for p in pairs], [p["T_cm"][:3, 3] for p in pairs], method=flag)
    res[name] = make_T(R, t)
    print(f"{name:11s} t = {np.round(t.ravel() * 1000, 1)} mm")

# 手法間のばらつき（真値が無いので、手法どうしの一致度を見る）
med = np.median([T[:3, 3] for T in res.values()], axis=0)
for name, T in res.items():
    print(f"{name:11s} 中央値からの距離 {np.linalg.norm(T[:3, 3] - med) * 1000:6.1f} mm")

T = res[a.method]
qx, qy, qz, qw = Rotation.from_matrix(T[:3, :3]).as_quat()
out = {
    "method": a.method,
    "T_base_cam": T.tolist(),
    "all": {k: v.tolist() for k, v in res.items()},
    # Isaac Lab: CameraCfg.OffsetCfg(pos=..., rot=..., convention="ros")。base_link を親にした場合の値
    # rot の並びは Isaac Lab 2.x が (w, x, y, z)、3.0 以降が (x, y, z, w)
    "isaaclab_offset": {"pos": T[:3, 3].tolist(), "rot_wxyz_2x": [qw, qx, qy, qz], "rot_xyzw_3x": [qx, qy, qz, qw], "convention": "ros"},
}
(DATA / "cam2base.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out["isaaclab_offset"]))
