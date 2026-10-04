"""SO-ARM101 の順運動学（URDF を読んで base_link -> gripper_frame_link の 4x4 行列を返す）"""
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

URDF = Path(__file__).resolve().parent.parent / "urdf" / "so_arm101.urdf"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
BASE, TIP = "base_link", "gripper_frame_link"


def rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def axis_angle_to_R(axis, a):
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def load_chain(urdf=URDF):
    joints = {}
    for j in ET.parse(urdf).getroot().findall("joint"):
        o = j.find("origin")
        T = np.eye(4)
        T[:3, :3] = rpy_to_R(*map(float, o.get("rpy", "0 0 0").split()))
        T[:3, 3] = list(map(float, o.get("xyz", "0 0 0").split()))
        ax = j.find("axis")
        joints[j.find("child").get("link")] = dict(
            name=j.get("name"), parent=j.find("parent").get("link"), origin=T,
            axis=list(map(float, ax.get("xyz").split())) if ax is not None else [0, 0, 1],
            type=j.get("type"),
        )
    chain, link = [], TIP
    while link != BASE:
        chain.append(joints[link])
        link = joints[link]["parent"]
    return chain[::-1]


CHAIN = load_chain()


def fk_all(q_rad):
    """各関節フレームの姿勢を base_link 基準で全部返す（最後が gripper_frame_link）"""
    q = dict(zip(JOINTS, q_rad))
    T, out = np.eye(4), []
    for j in CHAIN:
        T = T @ j["origin"]
        if j["type"] == "revolute":
            R = np.eye(4)
            R[:3, :3] = axis_angle_to_R(j["axis"], q[j["name"]])
            T = T @ R
        out.append(T)
    return out


def fk(q_rad):
    """q_rad: JOINTS 順の関節角 [rad]（URDF 基準）。base_link から見た gripper_frame_link の姿勢を返す"""
    return fk_all(q_rad)[-1]


def keypoints(q_rad):
    """重ね合わせ評価に使うロボット本体の点（各関節の位置）[m]"""
    return np.array([T[:3, 3] for T in fk_all(q_rad)])


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    print([j["name"] for j in CHAIN])
    print(fk(np.zeros(5)))
