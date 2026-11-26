"""Rule-based stroke detector: bouts -> features -> evidence accumulator -> alarms.

  load_and_sync -> preprocess (+ gravity) -> find_bouts -> window_features -> evidence_accumulator.run

Usage (repo root):
  python -m src.models.rule_detector data/raw/LEFT.csv data/raw/RIGHT.csv [--floor G] [--k K] [--h H]
"""

import numpy as np
import pandas as pd

from src.features.bouts import find_bouts, gravity
from src.features.noise_floor import load_floor
from src.features.window_features import window_features
from src.models.arm_correlation import FS, load_and_sync, preprocess
from src.models.evidence_accumulator import Params, run


def extract(acc_l, acc_r, floor=None, fs=FS, fc=0.3):
    """Raw (n, 3) arrays of both wrists -> (windows, bouts) DataFrames. The slow part: do it once
    per night and call evidence_accumulator.run() as many times as you want with other parameters."""
    floor = load_floor() if floor is None else floor
    mag_l, mag_r = preprocess(acc_l, fs, fc), preprocess(acc_r, fs, fc)
    grav_l, grav_r = gravity(acc_l, fs, fc), gravity(acc_r, fs, fc)
    bouts = find_bouts(mag_l, mag_r, grav_l, grav_r, floor, fs)
    return window_features(mag_l, mag_r, grav_l, grav_r, bouts, floor, fs), bouts


def detect(acc_l, acc_r, floor=None, params=None, fs=FS):
    """Whole detector. Returns dict(windows, bouts, curve, alarms); alarms = t_s, side
    (side "left" = suspected left-arm weakness)."""
    windows_df, bouts = extract(acc_l, acc_r, floor, fs)
    curve, alarms = run(bouts, params)
    return {"windows": windows_df, "bouts": bouts, "curve": curve, "alarms": alarms}


def mark_chest_turns(bouts, intervals):
    """Adds a `chest_turn` column (bout overlaps a whole-body turn seen by the chest sensor).
    intervals: list of (start_s, end_s). The accumulator then uses it instead of the wrist orientation."""
    bouts = bouts.copy()
    bouts["chest_turn"] = [any(a < st + d and b > st for a, b in intervals) for st, d in zip(bouts["start_s"], bouts["dur_s"])]
    return bouts


def align_by_clap(acc_l, acc_r, fs=FS, search_s=30):
    """Not used yet (load_and_sync only trims). Aligns the two boards with a clap: the wearer claps
    at the start, so the largest acceleration peak of each wrist in the first `search_s` seconds is
    the same instant. Returns both arrays shifted to that peak and trimmed to the same length."""
    n = int(search_s * fs)
    peaks = [int(np.argmax(np.linalg.norm(np.diff(a[:n], axis=0), axis=1))) for a in (acc_l, acc_r)]
    acc_l, acc_r = acc_l[peaks[0]:], acc_r[peaks[1]:]
    m = min(len(acc_l), len(acc_r))
    return acc_l[:m], acc_r[:m]


def plot(res, path, h=Params.h, title="", onset_s=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    w, b, c, al = res["windows"], res["bouts"], res["curve"], res["alarms"]
    colors = {"left_only": "tab:blue", "right_only": "tab:orange", "both": "tab:green"}
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(11, 8), sharex=True,
                                     gridspec_kw={"height_ratios": [2, 1, 2]})
    a1.plot(w["start_s"] / 60, 1000 * w["act_l"], color=colors["left_only"], label="left")
    a1.plot(w["start_s"] / 60, 1000 * w["act_r"], color=colors["right_only"], label="right")
    a1.set_ylabel("activity above floor (mg)")
    a1.legend(loc="upper right")
    y = {"left_only": 1, "both": 0, "right_only": -1}
    a2.scatter(b["start_s"] / 60, b["kind"].map(y), c=b["kind"].map(colors), marker="|", s=200)
    a2.set_yticks(list(y.values()), list(y))
    a2.set_ylim(-1.8, 1.8)
    a2.set_ylabel("bouts")
    a3.step(c["t_s"] / 60, c["S_left"], where="post", color=colors["left_only"], label="S: left weak")
    a3.step(c["t_s"] / 60, c["S_right"], where="post", color=colors["right_only"], label="S: right weak")
    a3.axhline(h, ls="--", color="tab:red", lw=0.8, label="threshold h")
    for t, side in al.itertuples(index=False):
        a3.axvline(t / 60, color="tab:red", lw=1.5)
        a3.text(t / 60, h, f" alarm {side}", color="tab:red", rotation=90, va="bottom")
    if onset_s is not None:
        for ax in (a1, a2, a3):
            ax.axvline(onset_s / 60, color="k", ls=":", lw=1.5)
        a1.text(onset_s / 60, a1.get_ylim()[1], " stroke onset", va="top")
    a3.set_ylabel("accumulated evidence")
    a3.set_xlabel("time (min)")
    a3.legend(loc="upper right")
    if title:
        a1.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    ap = argparse.ArgumentParser(description="Rule-based stroke detector on two wrist recordings.")
    ap.add_argument("left", help="CSV of the left wrist")
    ap.add_argument("right", help="CSV of the right wrist")
    ap.add_argument("--floor", type=float, help="noise floor (g); default: results/noise_floor.json")
    ap.add_argument("--k", type=float, default=Params.k)
    ap.add_argument("--h", type=float, default=Params.h)
    ap.add_argument("--pause", type=float, default=30, help="minutes of silence after an alarm")
    ap.add_argument("--title", default="", help="figure title")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[2] / "results")
    a = ap.parse_args()
    res = detect(*load_and_sync(a.left, a.right), floor=a.floor,
                 params=Params(k=a.k, h=a.h, pause_s=a.pause * 60))
    a.out.mkdir(parents=True, exist_ok=True)
    for name in ("windows", "bouts", "curve", "alarms"):
        res[name].to_csv(a.out / f"rule_detector_{name}.csv", index=False)
    plot(res, a.out / "rule_detector.png", a.h, a.title)
    print(res["bouts"]["kind"].value_counts().to_string())
    print(f"\n{len(res['alarms'])} alarm(s)")
    if len(res["alarms"]):
        print(res["alarms"].assign(min=lambda d: d.t_s / 60).round(1).to_string(index=False))
    print(f"\nSaved to {a.out}")
