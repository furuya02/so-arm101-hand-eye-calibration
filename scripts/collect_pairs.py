"""実機で姿勢ペア（関節角 + 画像）を集める

トルクを切った SO-ARM101 を手で動かし、s キーで保存する。q で終了。
画面には検出したマーカーの軸と、向きの曖昧さ（ratio）を表示する。ratio が 3 未満のときは向きを変えて撮り直す。
"""
import argparse
import json
import time
from pathlib import Path

import cv2
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from common import DATA, load_intrinsics, marker_pose

ap = argparse.ArgumentParser()
ap.add_argument("--port", default="/dev/tty.usbmodem5A460829821")
ap.add_argument("--robot-id", default="follower")
ap.add_argument("--camera", type=int, default=0)
ap.add_argument("--marker-size", type=float, default=0.04)
ap.add_argument("--marker-id", type=int, default=0)
ap.add_argument("--out", default=str(DATA / "pairs"))
a = ap.parse_args()

K, dist, (w, h) = load_intrinsics()
robot = SO101Follower(SO101FollowerConfig(port=a.port, id=a.robot_id, use_degrees=True))
robot.connect()
robot.bus.disable_torque()  # 手で動かす

cap = cv2.VideoCapture(a.camera)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
log = (out / "pairs.jsonl").open("a")

while True:
    ok, img = cap.read()
    view = img.copy()
    m = marker_pose(img, K, dist, a.marker_size, a.marker_id)
    if m:
        T, ratio, _ = m
        cv2.drawFrameAxes(view, K, dist, cv2.Rodrigues(T[:3, :3])[0], T[:3, 3], a.marker_size)
        cv2.putText(view, f"ratio {ratio:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if ratio >= 3 else (0, 0, 255), 2)
    cv2.imshow("collect", view)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        time.sleep(0.5)  # 手を離して止まるのを待つ
        q = robot.bus.sync_read("Present_Position")
        ok, img = cap.read()
        name = f"{int(time.time() * 1000)}.png"
        cv2.imwrite(str(out / name), img)
        log.write(json.dumps({"image": name, "q_deg": q}) + "\n")
        log.flush()
        print("saved", name, {k: round(v, 1) for k, v in q.items()})
    elif key == ord("q"):
        break

robot.disconnect()
