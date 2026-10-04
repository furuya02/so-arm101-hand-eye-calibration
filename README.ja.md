# so-arm101-hand-eye-calibration

SO-ARM101 の脇に固定したカメラ（eye-to-hand）の位置を、アームを動かして集めたデータから
`cv2.calibrateHandEye` で求めるスクリプトです。

- `scripts/simulate.py` … 真値が分かっている合成データで、5 手法（TSAI / PARK / HORAUD / ANDREFF / DANIILIDIS）を比べる。実機は不要
- `scripts/collect_pairs.py` → `solve_real.py` → `evaluate.py` … 実機で姿勢ペアを集めて解き、ロボット本体の重ね合わせで評価する

> 実機用スクリプト（collect_pairs / solve_real / evaluate）は、合成画像での通し確認のみ行っています。実機での検証は未実施です。

## 動作確認環境

| | |
|---|---|
| OS | macOS 15.7.3（Darwin 24.6.0） |
| Python | 3.10.13 |
| OpenCV | 4.10.0（opencv-python 4.10.0.84） |
| numpy / scipy / matplotlib | 2.2.6 / 1.15.3 / 3.10.5 |
| URDF | [MuammerBay/isaac_so_arm101](https://github.com/MuammerBay/isaac_so_arm101) の so_arm101.urdf（commit e4624de） |
| LeRobot（実機のみ） | 0.6.1 |

## セットアップ

```bash
git clone https://github.com/furuya02/so-arm101-hand-eye-calibration.git
cd so-arm101-hand-eye-calibration

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 合成データで試す（実機不要）

```bash
python scripts/simulate.py            # 約 6 分。results/ に results.json と図が出る
python scripts/simulate.py --plot-only  # 図だけ作り直す
```

| 引数 | 既定値 | 内容 |
|---|---|---|
| --trials | 100 | 各条件の試行回数 |
| --size | 0.04 | マーカーの一辺 [m] |
| --px | 0.3 | マーカー角点の検出ノイズ [px] |
| --joint | 0.3 | 関節角の読み取りノイズ [deg] |
| --min-ratio | 3.0 | 向きが曖昧な観測を捨てるしきい値（0 で無効） |

### 結果（20 姿勢・100 試行の中央値、曖昧な観測は捨てる）

| 手法 | 位置の誤差 [mm] | 位置の誤差 p90 [mm] | 向きの誤差 [deg] |
|---|---|---|---|
| TSAI | 4.66 | 7.6 | 0.45 |
| PARK | 3.70 | 6.8 | 0.40 |
| HORAUD | 3.65 | 5.8 | 0.41 |
| ANDREFF | 38.10 | 73.7 | 0.37 |
| DANIILIDIS | 4.35 | 16.2 | 0.43 |

分かったこと:

- gripper2base を反転し忘れると、例外もなく 840 mm ずれた値が返る
- 回転が足りないと、TSAI はログを出すだけで単位行列とゼロを返す（例外にならない）
- 一辺 30 mm 以下のマーカーでは向きが反転した解が混ざる。2 つの解の再投影誤差の比が 3 未満の観測を捨てると、30 mm で 75 → 5.4 mm
- マーカーの再投影誤差（holdout）は真の誤差とほとんど相関しない（順位相関 0.29）。ロボット本体の関節位置の重ね合わせは 0.91

## 実機で求める

### (1) 準備

- `data/intrinsics.example.json` を `data/intrinsics.json` にコピーし、カメラの内部パラメータ（`cv2.calibrateCamera` の結果）を書く
- `data/joint_map.json` に、LeRobot の関節角 [deg] と URDF の関節角のゼロ点・符号の対応を書く
- ArUco マーカー（DICT_4X4_50 / ID 0 / 一辺 40 mm 以上）をグリッパーに固定する

### (2) 姿勢ペアを集める

```bash
pip install "lerobot[feetech]==0.6.1"
python scripts/collect_pairs.py --port /dev/tty.usbmodemXXXX --robot-id follower
```

トルクが切れるので手でアームを動かし、`s` で保存、`q` で終了します。
画面の ratio が赤（3 未満）のときは向きを変えてください。20〜30 姿勢、各関節を大きく（±30 度以上）動かします。

### (3) 解く

```bash
python scripts/solve_real.py --method PARK
```

`data/cam2base.json` に、base_link から見たカメラの姿勢と、Isaac Lab の `CameraCfg.OffsetCfg(convention="ros")` に入れる pos / rot が出ます。rot の並びは Isaac Lab 2.x が (w, x, y, z)、3.0 以降が (x, y, z, w) なので、使う版に合わせて rot_wxyz_2x / rot_xyzw_3x を選んでください。

### (4) 評価する

```bash
python scripts/evaluate.py --click        # 画像ごとに 6 関節（shoulder_pan → gripper_frame の順）をクリック
python scripts/evaluate.py --before data/before.json   # 簡易的な方法で求めた値と比べる
```

## ライセンス

MIT（`urdf/so_arm101.urdf` は [MuammerBay/isaac_so_arm101](https://github.com/MuammerBay/isaac_so_arm101) からのコピーで、BSD 3-Clause。`urdf/LICENSE` を参照）
