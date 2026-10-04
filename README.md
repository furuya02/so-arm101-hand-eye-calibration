# so-arm101-hand-eye-calibration

Scripts that find the pose of a camera fixed next to an SO-ARM101 (eye-to-hand)
with `cv2.calibrateHandEye`, using pairs collected by moving the arm.

- `scripts/simulate.py` … compares the 5 methods (TSAI / PARK / HORAUD / ANDREFF / DANIILIDIS) on synthetic data with known ground truth. No hardware needed
- `scripts/collect_pairs.py` → `solve_real.py` → `evaluate.py` … collect pose pairs on the real robot, solve, and evaluate by overlaying the robot body

> The real-robot scripts (collect_pairs / solve_real / evaluate) have only been run end-to-end on synthetic images. They have not been verified on the real robot yet.

## Tested environment

| | |
|---|---|
| OS | macOS 15.7.3 (Darwin 24.6.0) |
| Python | 3.10.13 |
| OpenCV | 4.10.0 (opencv-python 4.10.0.84) |
| numpy / scipy / matplotlib | 2.2.6 / 1.15.3 / 3.10.5 |
| URDF | so_arm101.urdf from [MuammerBay/isaac_so_arm101](https://github.com/MuammerBay/isaac_so_arm101) (commit e4624de) |
| LeRobot (real robot only) | 0.6.1 |

## Setup

```bash
git clone https://github.com/furuya02/so-arm101-hand-eye-calibration.git
cd so-arm101-hand-eye-calibration

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Try it on synthetic data (no hardware)

```bash
python scripts/simulate.py              # about 6 min. Writes results.json and figures to results/
python scripts/simulate.py --plot-only  # regenerate figures only
```

| Argument | Default | Meaning |
|---|---|---|
| --trials | 100 | trials per condition |
| --size | 0.04 | marker side length [m] |
| --px | 0.3 | marker corner detection noise [px] |
| --joint | 0.3 | joint angle reading noise [deg] |
| --min-ratio | 3.0 | threshold for dropping observations with ambiguous orientation (0 disables) |

### Results (20 poses, median of 100 trials, ambiguous observations dropped)

| Method | Position error [mm] | Position error p90 [mm] | Rotation error [deg] |
|---|---|---|---|
| TSAI | 4.66 | 7.6 | 0.45 |
| PARK | 3.70 | 6.8 | 0.40 |
| HORAUD | 3.65 | 5.8 | 0.41 |
| ANDREFF | 38.10 | 73.7 | 0.37 |
| DANIILIDIS | 4.35 | 16.2 | 0.43 |

Findings:

- If you forget to invert gripper2base, you get a result that is off by 840 mm, with no exception
- With too little rotation, TSAI only logs a message and returns identity / zero (no exception)
- Markers of 30 mm or smaller mix in pose-flipped solutions. Dropping observations whose ratio of the two solutions' reprojection errors is below 3 brings 30 mm from 75 to 5.4 mm
- The marker reprojection error on held-out poses barely correlates with the true error (Spearman 0.29). Overlaying the robot's joint positions does (0.91)

## On the real robot

### (1) Preparation

- Copy `data/intrinsics.example.json` to `data/intrinsics.json` and fill in the camera intrinsics (output of `cv2.calibrateCamera`)
- Write the zero offset and sign between LeRobot joint angles [deg] and URDF joint angles in `data/joint_map.json`
- Attach an ArUco marker (DICT_4X4_50 / ID 0 / side 40 mm or larger) to the gripper

### (2) Collect pose pairs

```bash
pip install "lerobot[feetech]==0.6.1"
python scripts/collect_pairs.py --port /dev/tty.usbmodemXXXX --robot-id follower
```

Torque is disabled, so move the arm by hand. Press `s` to save, `q` to quit.
If the ratio on screen is red (below 3), change the orientation. Collect 20–30 poses and move each joint widely (±30 degrees or more).

### (3) Solve

```bash
python scripts/solve_real.py --method PARK
```

`data/cam2base.json` contains the camera pose seen from base_link, plus pos / rot for Isaac Lab `CameraCfg.OffsetCfg(convention="ros")`. The quaternion order is (w, x, y, z) in Isaac Lab 2.x and (x, y, z, w) from 3.0, so pick rot_wxyz_2x or rot_xyzw_3x to match your version.

### (4) Evaluate

```bash
python scripts/evaluate.py --click        # click 6 joints per image (shoulder_pan → gripper_frame)
python scripts/evaluate.py --before data/before.json   # compare with a value from a simpler method
```

## License

MIT (`urdf/so_arm101.urdf` is copied from [MuammerBay/isaac_so_arm101](https://github.com/MuammerBay/isaac_so_arm101) and is under BSD 3-Clause; see `urdf/LICENSE`)
