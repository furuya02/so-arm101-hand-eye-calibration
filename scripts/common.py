"""実機スクリプトで共通に使う処理（内部パラメータ・関節角の変換・ArUco 検出）"""
import json
from pathlib import Path

import cv2
import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"
_params = cv2.aruco.DetectorParameters()
_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX  # 角点をサブピクセルで取る
ARUCO = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), _params)


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, np.ravel(t)
    return T


def load_intrinsics(path=DATA / "intrinsics.json"):
    """内部パラメータ（camera_matrix / dist_coeffs / image_size）。カメラの内部パラメータを求めたときの値をそのまま使う"""
    d = json.loads(Path(path).read_text())
    return np.array(d["camera_matrix"], float), np.array(d["dist_coeffs"], float), tuple(d["image_size"])


def to_urdf_rad(q_deg, path=DATA / "joint_map.json"):
    """LeRobot の関節角 [deg] -> URDF の関節角 [rad]。ゼロ点と符号は実機で確認した値を joint_map.json に書いておく"""
    m = json.loads(Path(path).read_text())
    return np.radians(np.array(m["sign"]) * (np.array(q_deg) - np.array(m["offset_deg"])))


def marker_pose(img, K, dist, size, marker_id):
    """ArUco マーカーの姿勢（camera 基準）と、2 つの解の再投影誤差の比を返す。見つからなければ None"""
    corners, ids, _ = ARUCO.detectMarkers(img)
    if ids is None or marker_id not in ids.ravel():
        return None
    c = corners[list(ids.ravel()).index(marker_id)].reshape(-1, 2)
    s = size / 2
    obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]])
    _, rvecs, tvecs, err = cv2.solvePnPGeneric(obj, c, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    ratio = float(err[1][0] / max(err[0][0], 1e-9))
    return make_T(cv2.Rodrigues(rvecs[0])[0], tvecs[0]), ratio, c
