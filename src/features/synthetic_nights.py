"""Synthetic 8 h nights of two wrists, built from FRAGMENTS of the single real recording.

DATA ARE SYNTHETIC: this only checks that the detector logic works, it says nothing clinical.
Less circular than synthetic_two_wrists.py (the right arm is not derived from the left), but
still circular: every movement comes from 13 min of one awake healthy person, and the
movement rate in sleep is an assumption (`bouts_per_hour`).

Per wrist: acceleration = gravity track + dynamic part.
  dynamic = quiet fragments (random 20 s chunks of the "quiet"/"final" phases, which carry the
            real sensor noise) + movement fragments (5-30 s of "lineal", "lliure" or "girs")
  gravity = 1 g vector that only changes at postural changes (bilateral events).
Events are placed as a Poisson process of rate `bouts_per_hour`:
  bilateral  (posture change): a DIFFERENT fragment in each arm, overlapping in time, plus a
             gravity rotation of 30-120 deg in both wrists
  left_only / right_only (scratching...): a fragment in one arm, the other stays still
Healthy dominance: one random arm is scaled by a factor in [0.8, 1.2] for the whole night.
Stroke (stroke=True): onset minute, side and severity (amplitude of the affected arm left,
5-60 %) drawn at random, abrupt or ramped (10-30 min). From onset on, bilateral events keep the
gravity rotation but the affected arm's own acceleration is scaled by the current level f;
affected-side-only events are dropped with probability 1-f and scaled by f otherwise.
Same seed -> same night (the stroke draws use their own random stream, so a healthy night and
the stroke night of the same seed share the same events).

Usage (repo root): python -m src.features.synthetic_nights --seeds 0 1 --rate 5 [--stroke]
  -> data/processed/synthetic_nights/night_<seed>_<rate>_<healthy|stroke>_{L,R}.csv + .json
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.bouts import gravity
from src.features.noise_floor import SRC, phase_bounds
from src.models.arm_correlation import FS
from src.models.model_asimetria import carregar_csv

MOVES = ("lineal", "lliure", "girs")
MIX = {"bilateral": 0.3, "left_only": 0.35, "right_only": 0.35}
OUT = Path(__file__).resolve().parents[2] / "data/processed/synthetic_nights"


@lru_cache(maxsize=1)
def _real(fs=FS):
    acc = carregar_csv(SRC)[["ax", "ay", "az"]].to_numpy()
    grav = gravity(acc, fs)
    a, b = phase_bounds("quiet")
    quiet = np.concatenate([(acc - grav)[int(a * fs):int(b * fs)],
                            (acc - grav)[int(phase_bounds("final")[0] * fs):int(phase_bounds("final")[1] * fs)]])
    g0 = grav[int(a * fs):int(b * fs)].mean(axis=0)
    return acc - grav, quiet, g0 / np.linalg.norm(g0)


def _fragment(rng, fs=FS, edge_s=0.5):
    """(type, dynamic fragment (L, 3)) from a random movement phase, 5-30 s, faded at the edges."""
    dyn, _, _ = _real(fs)
    typ = MOVES[rng.integers(len(MOVES))]
    a, b = phase_bounds(typ)
    length = rng.uniform(5, 30)
    s = rng.uniform(a, b - length)
    frag = dyn[int(s * fs):int((s + length) * fs)].copy()
    w, f = np.ones(len(frag)), int(edge_s * fs)
    w[:f], w[-f:] = np.linspace(0, 1, f), np.linspace(1, 0, f)
    return typ, frag * w[:, None]


def _rotate(v, rng, lo=30, hi=120):
    """v rotated by a random angle (deg) around a random axis (Rodrigues)."""
    k = rng.normal(size=3)
    k /= np.linalg.norm(k)
    th = np.radians(rng.uniform(lo, hi))
    return v * np.cos(th) + np.cross(k, v) * np.sin(th) + k * (k @ v) * (1 - np.cos(th))


def _level(t, st):
    """Strength of the affected arm (1 = healthy) at time t (s)."""
    if t < st["onset_min"] * 60:
        return 1.0
    if not st["ramp_min"]:
        return st["severity"]
    return 1 - (1 - st["severity"]) * min((t - st["onset_min"] * 60) / (st["ramp_min"] * 60), 1)


def make_night(seed, bouts_per_hour=5.0, hours=8.0, stroke=False, mix=None, ramp_prob=0.5, fs=FS):
    """Returns (acc_l, acc_r, params): two (n, 3) arrays in g and a JSON-able dict."""
    mix = mix or MIX
    _, quiet, g0 = _real(fs)
    rng = np.random.default_rng(seed)
    n, T = int(hours * 3600 * fs), hours * 3600
    chunk = 20 * fs
    base = [np.concatenate([quiet[s:s + chunk] for s in rng.integers(0, len(quiet) - chunk, n // chunk + 1)])[:n]
            for _ in range(2)]                                              # L, R quiet dynamic part
    dom_arm, dom = int(rng.integers(2)), float(rng.uniform(0.8, 1.2))
    scale = [dom if i == dom_arm else 1.0 for i in range(2)]

    st = None
    if stroke:
        rs = np.random.default_rng([seed, 1])
        st = {"onset_min": float(rs.uniform(60, hours * 60 - 120)), "side": ["left", "right"][int(rs.integers(2))],
              "severity": float(rs.uniform(0.05, 0.6)),
              "ramp_min": float(rs.uniform(10, 30)) if rs.random() < ramp_prob else 0.0}
    aff = None if st is None else int(st["side"] == "right")

    kinds, p = list(mix), np.array(list(mix.values()), float)
    changes, counts, events, prev_end = [], dict.fromkeys(mix, 0), [], 0.0
    for t in np.sort(rng.uniform(0, T, rng.poisson(bouts_per_hour * hours))):
        kind = kinds[rng.choice(len(kinds), p=p / p.sum())]
        frags = {0: None, 1: None}                                           # arm -> (type, fragment)
        if kind in ("bilateral", "left_only"):
            frags[0] = _fragment(rng, fs)
        if kind in ("bilateral", "right_only"):
            frags[1] = _fragment(rng, fs)
        dur = max(len(f[1]) for f in frags.values() if f) / fs
        if t < prev_end + 2 or t + dur > T:                                  # no overlap, inside the night
            continue
        prev_end, f_t = t + dur, 1.0 if st is None else _level(t, st)
        counts[kind] += 1
        events.append([round(float(t), 1), kind, round(dur, 1)])
        if kind == "bilateral":
            changes.append((int(t * fs), int((t + dur) * fs), [_rotate(g0, rng), _rotate(g0, rng)]))
        for arm, fr in frags.items():
            if fr is None:
                continue
            amp = scale[arm]
            if arm == aff:
                if kind != "bilateral" and f_t < 1 and rs.random() > f_t:
                    continue                                                 # affected-only event vanishes
                amp *= f_t
            off = int((rng.uniform(0, dur - len(fr[1]) / fs) if kind == "bilateral" else 0) * fs)
            i = int(t * fs) + off
            base[arm][i:i + len(fr[1])] += amp * fr[1]

    acc = []
    for arm in range(2):                                                     # gravity track + dynamic part
        g, cur = np.tile(g0, (n, 1)), g0
        for i0, i1, new in changes:
            new = new[arm]
            g[i0:i1] = cur + (new - cur) * np.linspace(0, 1, i1 - i0)[:, None]
            g[i1:], cur = new, new
        acc.append(g + base[arm])
    params = {"seed": seed, "bouts_per_hour": bouts_per_hour, "hours": hours, "mix": mix,
              "dominance": {"arm": ["left", "right"][dom_arm], "factor": dom},
              "stroke": st, "n_events": counts, "events": events}
    return acc[0], acc[1], params


def save_night(prefix, acc_l, acc_r, params):
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for side, a in (("L", acc_l), ("R", acc_r)):
        df = pd.DataFrame(a, columns=["ax", "ay", "az"])
        df.insert(0, "t_s", np.arange(len(a)) / FS)
        df.to_csv(f"{prefix}_{side}.csv", index=False, float_format="%.4f")
    Path(f"{prefix}.json").write_text(json.dumps(params, indent=1))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Write synthetic two-wrist nights (CSV per wrist + JSON).")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--rate", type=float, default=5, help="bouts per hour")
    ap.add_argument("--hours", type=float, default=8)
    ap.add_argument("--stroke", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    for s in a.seeds:
        night = make_night(s, a.rate, a.hours, a.stroke)
        save_night(a.out / f"night_{s}_{a.rate:g}_{'stroke' if a.stroke else 'healthy'}", *night)
        print(s, night[2]["n_events"], night[2]["stroke"])
