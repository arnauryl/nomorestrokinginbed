"""Movement bouts: stretches where an arm's activity (1 s mean of the gravity-free magnitude)
is above the noise floor.

Per arm: blocks above the floor, gaps shorter than `merge_gap_s` filled, runs shorter than
`min_dur_s` dropped (1 s blips of sensor noise; ponytail: not in the brief, set to 0 to disable).
Across arms: a left run and a right run that overlap for at least `min_overlap` of the shorter
one are ONE bout, "both"; otherwise each run is "left_only" / "right_only".
"""

import numpy as np
import pandas as pd
from scipy.ndimage import find_objects, label
from scipy.signal import butter, sosfiltfilt

from src.features.noise_floor import block_means, load_floor
from src.models.arm_correlation import FS

COLUMNS = ["start_s", "dur_s", "kind", "amp_l", "amp_r", "orient_l", "orient_r"]


def gravity(acc, fs=FS, fc=0.3):
    """Slow part of the acceleration (what preprocess() throws away): the gravity vector, (n, 3)."""
    return sosfiltfilt(butter(2, fc, fs=fs, btype="lowpass", output="sos"), acc, axis=0)


def orient_change(grav, i0, i1, fs=FS, edge_s=2):
    """Angle (degrees) between the mean gravity vector in the first and in the last `edge_s`
    seconds of grav[i0:i1]. Indices are clipped to the array."""
    i0, i1, e = max(i0, 0), min(i1, len(grav)), int(edge_s * fs)
    a, b = grav[i0:i0 + e].mean(axis=0), grav[max(i1 - e, i0):i1].mean(axis=0)
    cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def _active(blocks, floor, merge_gap_s, min_dur_s):
    on = blocks > floor
    lab, _ = label(~on)
    for sl in find_objects(lab):                       # fill short gaps between two active runs
        s = sl[0]
        if s.stop - s.start < merge_gap_s and s.start > 0 and s.stop < len(on):
            on[s] = True
    lab, _ = label(on)
    for sl in find_objects(lab):                       # drop runs that are too short
        if sl[0].stop - sl[0].start < min_dur_s:
            on[sl[0]] = False
    return on


def find_bouts(mag_l, mag_r, grav_l, grav_r, floor=None, fs=FS,
               merge_gap_s=2, min_dur_s=2, min_overlap=0.5):
    """mag_*: preprocess() output; grav_*: gravity() output. Returns a DataFrame (COLUMNS),
    one row per bout: start and duration (s), kind, amplitude of each arm (mean activity above
    the floor over its own active blocks, g) and orientation change of each wrist (degrees)."""
    floor = load_floor() if floor is None else floor
    bl, br = block_means(mag_l, fs), block_means(mag_r, fs)
    al, ar = (_active(b, floor, merge_gap_s, min_dur_s) for b in (bl, br))
    rows = []

    def add(s, e, kind):
        amp = [np.maximum(b[s:e] - floor, 0)[a[s:e]].mean() if a[s:e].any() else 0.0
               for b, a in ((bl, al), (br, ar))]
        ori = [orient_change(g, (s - 2) * fs, (e + 2) * fs, fs) for g in (grav_l, grav_r)]
        rows.append([float(s), float(e - s), kind, *amp, *ori])

    lab, _ = label(al | ar)
    for sl in find_objects(lab):
        s, e = sl[0].start, sl[0].stop
        l, r = al[s:e], ar[s:e]
        if l.any() and r.any() and (l & r).sum() >= min_overlap * min(l.sum(), r.sum()):
            add(s, e, "both")
            continue
        for mask, kind in ((l, "left_only"), (r, "right_only")):
            for sub in find_objects(label(mask)[0]):
                add(s + sub[0].start, s + sub[0].stop, kind)
    return pd.DataFrame(rows, columns=COLUMNS).sort_values("start_s", ignore_index=True)
