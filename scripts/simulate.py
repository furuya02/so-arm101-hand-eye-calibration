"""真値が分かっている合成データで cv2.calibrateHandEye（eye-to-hand）を検証する

カメラ位置（真値）を決めておき、SO-ARM101 の順運動学とマーカーの投影から姿勢ペアを作って解かせる。
ノイズは 2 種類だけ入れる。
  - マーカー角点の検出ノイズ [px]
  - 関節角の読み取りノイズ [deg]（順運動学の誤差源）
"""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fk import fk, keypoints  # noqa: E402

plt.rcParams["font.family"] = "Hiragino Sans"

METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}

# 仮のカメラ（640x480）。実機では idea028 で求めた値に置き換える
K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
W, H = 640, 480
Q_LO = np.radians([-110, -100, -97, -95, -157])
Q_HI = -Q_LO
HOME = np.radians([-48, -40, 44, 45, 94])  # マーカーがカメラの正面を向く姿勢


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, np.ravel(t)
    return T


def look_at(eye, target):
    """target を向くカメラ（OpenCV 規約: z 前方 / x 右 / y 下）の base 基準の姿勢"""
    z = np.subtract(target, eye) / np.linalg.norm(np.subtract(target, eye))
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    return make_T(np.c_[x, np.cross(z, x), z], eye)


T_BASE_CAM = look_at([0.45, -0.30, 0.35], [0.20, 0.0, 0.12])  # 真値（求めたい答え）
T_GRIP_MARKER = make_T(cv2.Rodrigues(np.array([0, np.pi / 2, 0]))[0], [0.0, -0.02, 0.03])  # 治具の取り付け位置（未知）


def corners(size):
    s = size / 2
    return np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]])


def observe(T_cam_marker, size, px_noise, rng):
    """マーカー角点を投影してノイズを足し、solvePnP で姿勢を復元する（実機の ArUco 検出の代わり）"""
    rvec, _ = cv2.Rodrigues(T_cam_marker[:3, :3])
    img, _ = cv2.projectPoints(corners(size), rvec, T_cam_marker[:3, 3], K, None)
    img = img.reshape(-1, 2) + rng.normal(0, px_noise, (4, 2))
    # IPPE_SQUARE は 2 つの解を返す。2 番目の解の再投影誤差が 1 番目と近いときは向きが反転しやすい
    _, rvecs, tvecs, err = cv2.solvePnPGeneric(corners(size), img, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    ratio = float(err[1][0] / max(err[0][0], 1e-9))
    return make_T(cv2.Rodrigues(rvecs[0])[0], tvecs[0]), img, ratio


def visible(T_cam_marker, size):
    if T_cam_marker[2, 3] < 0.15 or T_cam_marker[2, 2] > -np.cos(np.radians(60)):  # 正面から 60 度以内
        return False
    rvec, _ = cv2.Rodrigues(T_cam_marker[:3, :3])
    img, _ = cv2.projectPoints(corners(size), rvec, T_cam_marker[:3, 3], K, None)
    img = img.reshape(-1, 2)
    return bool((img[:, 0] > 20).all() and (img[:, 0] < W - 20).all() and (img[:, 1] > 20).all() and (img[:, 1] < H - 20).all())


def sample_poses(n, rng, size, spread=None):
    """見えるマーカー姿勢を n 個作る。spread[deg] を指定すると HOME まわりの小さな動きに限定する"""
    qs = []
    for _ in range(100000):
        if len(qs) == n:
            return qs
        q = rng.uniform(Q_LO, Q_HI) if spread is None else HOME + np.radians(rng.uniform(-spread, spread, 5))
        T_bg = fk(q)
        if T_bg[2, 3] < 0.05:
            continue
        if visible(np.linalg.inv(T_BASE_CAM) @ T_bg @ T_GRIP_MARKER, size):
            qs.append(q)
    raise RuntimeError("見える姿勢が見つからない")


def make_pairs(n, rng, size, px_noise, joint_noise, spread=None, min_ratio=None):
    """n 組の姿勢ペアを作る。min_ratio を指定すると向きが曖昧な観測を捨てて撮り直す"""
    pairs = []
    while len(pairs) < n:
        q = sample_poses(1, rng, size, spread)[0]
        T_cm_true = np.linalg.inv(T_BASE_CAM) @ fk(q) @ T_GRIP_MARKER
        T_cm, img, ratio = observe(T_cm_true, size, px_noise, rng)
        if min_ratio and ratio < min_ratio:
            continue
        q_read = q + np.radians(rng.normal(0, joint_noise, 5))  # サーボから読んだ値（誤差入り）
        pairs.append(dict(T_bg=fk(q_read), T_cm=T_cm, img=img))
    return pairs


def solve(pairs, method, invert=True):
    """eye-to-hand では gripper2base を反転して base2gripper として渡す。戻り値が cam2base になる"""
    Tg = [np.linalg.inv(p["T_bg"]) if invert else p["T_bg"] for p in pairs]
    R, t = cv2.calibrateHandEye(
        [T[:3, :3] for T in Tg], [T[:3, 3] for T in Tg],
        [p["T_cm"][:3, :3] for p in pairs], [p["T_cm"][:3, 3] for p in pairs],
        method=METHODS[method],
    )
    if np.allclose(t, 0):  # TSAI は回転が足りないとログを出すだけで、単位行列とゼロを返す
        raise cv2.error("degenerate")
    return make_T(R, t)


def errors(T_est, T_true=T_BASE_CAM):
    dR = T_est[:3, :3].T @ T_true[:3, :3]
    rot = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
    return np.linalg.norm(T_est[:3, 3] - T_true[:3, 3]) * 1000, rot  # [mm], [deg]


def holdout_px(T_bc, train, test, size):
    """真値が無い実機でも計算できる指標。学習に使っていない姿勢でマーカー角点の位置を予測し、観測とのズレを測る"""
    T_gms = [np.linalg.inv(p["T_bg"]) @ T_bc @ p["T_cm"] for p in train]
    T_gm = make_T(cv2.Rodrigues(np.mean([cv2.Rodrigues(T[:3, :3])[0] for T in T_gms], axis=0))[0],
                  np.mean([T[:3, 3] for T in T_gms], axis=0))
    d = []
    for p in test:
        T = np.linalg.inv(T_bc) @ p["T_bg"] @ T_gm
        img, _ = cv2.projectPoints(corners(size), cv2.Rodrigues(T[:3, :3])[0], T[:3, 3], K, None)
        d.append(np.linalg.norm(img.reshape(-1, 2) - p["img"], axis=1).mean())
    return float(np.mean(d))


def project(T_base_cam, P):
    T = np.linalg.inv(T_base_cam)
    img, _ = cv2.projectPoints((T[:3, :3] @ P.T).T + T[:3, 3], np.zeros(3), np.zeros(3), K, None)
    return img.reshape(-1, 2)


def keypoint_px(T_bc, qs, rng, click_noise=1.0):
    """マーカーを使わない指標。ロボット本体の関節位置を投影し、画像上の位置（手で打った点を想定）とのズレを測る"""
    return float(np.mean([np.linalg.norm(project(T_bc, keypoints(q)) - (project(T_BASE_CAM, keypoints(q)) + rng.normal(0, click_noise, (6, 2))), axis=1).mean() for q in qs]))


def run(trials, n, rng, size, px, jn, spread=None, method="TSAI", invert=True, min_ratio=None):
    out = []
    for _ in range(trials):
        pairs = make_pairs(n, rng, size, px, jn, spread, min_ratio)
        try:
            out.append(errors(solve(pairs, method, invert)))
        except cv2.error:  # ANDREFF の "determinant(R) is null" と TSAI の縮退。失敗回数として数える
            out.append((np.nan, np.nan))
    out = np.array(out)
    return np.r_[np.nanmedian(out, axis=0), np.isnan(out[:, 0]).sum(), np.nanpercentile(out[:, 0], 90)]  # [mm, deg, 失敗回数, p90 mm]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--size", type=float, default=0.04, help="マーカーの一辺 [m]")
    ap.add_argument("--px", type=float, default=0.3, help="角点検出ノイズ [px]")
    ap.add_argument("--joint", type=float, default=0.3, help="関節角ノイズ [deg]")
    ap.add_argument("--min-ratio", type=float, default=3.0, help="向きが曖昧な観測を捨てるしきい値（0 で無効）")
    ap.add_argument("--out", default="results")
    ap.add_argument("--plot-only", action="store_true", help="results.json から図だけ作り直す")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    if a.plot_only:
        r = json.loads((out / "results.json").read_text())
        return plot(r, r["ns"], r["spreads"], out)
    rng = np.random.default_rng(0)
    res = {}
    mr = a.min_ratio or None

    # (1) 5 手法比較（20 姿勢）
    res["methods"] = {m: run(a.trials, 20, rng, a.size, a.px, a.joint, method=m, min_ratio=mr).tolist() for m in METHODS}

    # (2) 姿勢数を変える
    ns = [3, 5, 7, 10, 15, 20, 30]
    res["n_sweep"] = {m: [run(a.trials, n, rng, a.size, a.px, a.joint, method=m, min_ratio=mr).tolist() for n in ns] for m in METHODS}

    # (3) 動かす範囲（回転の多様さ）を変える
    spreads = [2, 5, 10, 20, 30, 45]
    res["spread_sweep"] = [run(a.trials, 20, rng, a.size, a.px, a.joint, spread=s, min_ratio=mr).tolist() for s in spreads]

    # (4) gripper2base を反転し忘れた場合
    res["no_invert"] = run(a.trials, 20, rng, a.size, a.px, a.joint, invert=False, min_ratio=mr).tolist()

    # (5) ノイズ要因の切り分け / マーカーサイズ
    res["noise"] = {
        "ノイズなし": run(a.trials, 20, rng, a.size, 0, 0, min_ratio=mr).tolist(),
        "角点のみ": run(a.trials, 20, rng, a.size, a.px, 0, min_ratio=mr).tolist(),
        "関節角のみ": run(a.trials, 20, rng, a.size, 0, a.joint, min_ratio=mr).tolist(),
        "両方": run(a.trials, 20, rng, a.size, a.px, a.joint, min_ratio=mr).tolist(),
    }
    res["marker_size"] = {f"{s * 1000:.0f}mm": run(a.trials, 20, rng, s, a.px, a.joint, min_ratio=mr).tolist() for s in [0.02, 0.03, 0.04, 0.06]}

    # (6) 向きが曖昧な観測を捨てる（2 解の再投影誤差の比が 3 未満なら捨てる）
    res["ambiguity"] = {f"{s * 1000:.0f}mm": [run(a.trials, 20, rng, s, a.px, a.joint, method="PARK", min_ratio=r).tolist() for r in (None, 3)]
                        for s in [0.02, 0.03, 0.04, 0.06]}

    # (7) 真値誤差と holdout ピクセル誤差の関係（実機の評価指標として使えるか）
    sc = []
    for spread in [3, 5, 10, 20, None]:
        for _ in range(a.trials // 2):
            pairs = make_pairs(30, rng, a.size, a.px, a.joint, spread, mr)
            try:
                T = solve(pairs[:20], "PARK")
            except cv2.error:
                continue
            sc.append([*errors(T), holdout_px(T, pairs[:20], pairs[20:], a.size), keypoint_px(T, sample_poses(10, rng, a.size), rng)])
    res["holdout"] = sc

    (out / "results.json").write_text(json.dumps(dict(args=vars(a), ns=ns, spreads=spreads, **res), ensure_ascii=False, indent=1))
    plot(res, ns, spreads, out)

    print("== 5手法（20姿勢, 中央値） [mm, deg]")
    for m, v in res["methods"].items():
        print(f"{m:11s} {v[0]:7.2f} {v[1]:6.3f}  失敗 {v[2]:.0f}/{a.trials}  p90 {v[3]:.1f}mm")
    print("== 反転し忘れ", np.round(res["no_invert"], 2))
    print("== ノイズ要因", {k: np.round(v, 2).tolist() for k, v in res["noise"].items()})
    print("== マーカーサイズ", {k: np.round(v, 2).tolist() for k, v in res["marker_size"].items()})
    print("== 曖昧な観測を捨てる [なし, あり]", {k: np.round(v, 2).tolist() for k, v in res["ambiguity"].items()})
    print("== 姿勢数", {m: np.round(np.array(v)[:, [0, 3]], 1).tolist() for m, v in res["n_sweep"].items()})
    from scipy.stats import spearmanr
    h = np.array(res["holdout"])
    print("== 評価指標と真の誤差の順位相関", {"マーカー holdout": round(spearmanr(h[:, 0], h[:, 2])[0], 2), "関節点": round(spearmanr(h[:, 0], h[:, 3])[0], 2)})
    print("== 動かす範囲", dict(zip(spreads, np.round(res["spread_sweep"], 2).tolist())))


def plot(res, ns, spreads, out):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for m, v in res["n_sweep"].items():
        v = np.array(v)
        ax[0].plot(ns, v[:, 0], marker="o", label=m)
        ax[1].plot(ns, v[:, 1], marker="o", label=m)
    for a_, y in zip(ax, ["位置の誤差 [mm]", "向きの誤差 [deg]"]):
        a_.set_xlabel("姿勢数")
        a_.set_ylabel(y)
        a_.set_yscale("log")
        a_.grid(alpha=0.3)
    ax[0].legend()
    fig.suptitle("姿勢数と誤差（中央値）")
    fig.tight_layout()
    fig.savefig(out / "n_sweep.png", dpi=120)

    v = np.array(res["spread_sweep"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(spreads, v[:, 0], marker="o", label="位置 [mm]")
    ax.plot(spreads, v[:, 1], marker="s", label="向き [deg]")
    ax.set_xlabel("各関節を動かす範囲 ±[deg]（20 姿勢 / TSAI）")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("姿勢のバリエーションと誤差（中央値）")
    fig.tight_layout()
    fig.savefig(out / "spread_sweep.png", dpi=120)

    labels = list(res["ambiguity"])
    v = np.array([res["ambiguity"][k] for k in labels])  # [サイズ, なし/あり, 指標]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(labels))
    for a_, col, title in [(ax[0], 0, "中央値"), (ax[1], 3, "p90（悪い方から 10%）")]:
        a_.bar(x - 0.2, v[:, 0, col], 0.4, label="そのまま使う")
        a_.bar(x + 0.2, v[:, 1, col], 0.4, label="向きが曖昧な観測を捨てる")
        a_.set_xticks(x, labels)
        a_.set_xlabel("マーカーの一辺")
        a_.set_ylabel("カメラ位置の誤差 [mm]")
        a_.set_yscale("log")
        a_.set_title(title)
        a_.grid(alpha=0.3, axis="y")
    ax[0].legend()
    fig.suptitle("マーカーの大きさと、曖昧な観測を捨てる効果（20 姿勢 / PARK）")
    fig.tight_layout()
    fig.savefig(out / "ambiguity.png", dpi=120)

    s = np.array(res["holdout"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for a_, col, lab in [(ax[0], 2, "マーカー角点のズレ（holdout 姿勢）[px]"), (ax[1], 3, "ロボット関節点のズレ（重ね合わせ）[px]")]:
        a_.scatter(s[:, col], s[:, 0], s=10, alpha=0.6)
        a_.set_xlabel(lab)
        a_.set_xscale("log")
        a_.set_yscale("log")
        a_.grid(alpha=0.3)
    ax[0].set_ylabel("カメラ位置の真の誤差 [mm]")
    fig.suptitle("真値が無くても測れる指標は、真の誤差を反映するか")
    fig.tight_layout()
    fig.savefig(out / "holdout.png", dpi=120)


if __name__ == "__main__":
    main()
