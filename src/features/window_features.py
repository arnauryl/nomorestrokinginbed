"""Per-window features (60 s windows, 30 s step) for the rule detector.

One row per window: activity of each arm above the noise floor, the asymmetry of that
activity (the main signal; Iacovelli 2019, Reale 2023, Messe 2023), the correlation features
of arm_correlation.py (support only), orientation change of each wrist and bout counts.
"""

import numpy as np
import pandas as pd

from src.features.bouts import orient_change
from src.features.noise_floor import block_means, load_floor
from src.models.arm_correlation import FS, correlation, windows

COLUMNS = ["start_s", "act_l", "act_r", "asym", "pearson", "best_cc", "best_lag_s",
           "orient_change_l", "orient_change_r", "n_bouts_l_only", "n_bouts_r_only", "n_bouts_both"]


def activity(segment, floor, fs=FS):
    """Mean activity above the floor: 1 s block means, minus the floor, negatives to 0."""
    return float(np.maximum(block_means(segment, fs) - floor, 0).mean())


def asymmetry(act_l, act_r):
    """(L - R) / (L + R) in [-1, 1]; NaN if neither arm is above the noise floor.
    Unlike Pearson it changes when one signal is scaled (a weak arm gives |asym| -> 1)."""
    tot = act_l + act_r
    return (act_l - act_r) / tot if tot > 0 else np.nan


def window_features(mag_l, mag_r, grav_l, grav_r, bouts, floor=None, fs=FS,
                    win_s=60, step_s=30, max_lag_s=1.0):
    """mag_*: preprocess() output; grav_*: gravity() output; bouts: find_bouts() output.
    Correlation is only computed for windows where some arm is active (the rest is
    NaN, as asym): a still window has nothing to correlate and it is the slow part."""
    floor = load_floor() if floor is None else floor
    start, kind = bouts["start_s"].to_numpy(), bouts["kind"].to_numpy()
    rows = []
    for start_s, l, r in windows(mag_l, mag_r, fs, win_s, step_s):
        act_l, act_r = activity(l, floor, fs), activity(r, floor, fs)
        a = asymmetry(act_l, act_r)
        pearson = best_cc = lag = np.nan
        if not np.isnan(a):
            pearson, best_cc, lag = correlation(l, r, fs, max_lag_s)
        i0, i1 = int(start_s * fs), int((start_s + win_s) * fs)
        inside = (start >= start_s) & (start < start_s + win_s)
        rows.append([start_s, act_l, act_r, a, pearson, best_cc, lag,
                     orient_change(grav_l, i0, i1, fs), orient_change(grav_r, i0, i1, fs),
                     *(int((inside & (kind == k)).sum()) for k in ("left_only", "right_only", "both"))])
    return pd.DataFrame(rows, columns=COLUMNS)
