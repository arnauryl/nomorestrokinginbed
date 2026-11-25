"""Evidence accumulator: one one-sided CUSUM per side over the bouts of the night.

    S_t = max(0, S_{t-1} + e_t - k)      alarm when S_t > h

There is one CUSUM for "left arm weak" and one for "right arm weak". Each bout adds evidence
e_t to both (positive: this arm looks weaker; negative: it looks fine). Time without movement
adds nothing: S only changes at bouts, so still arms (the normal state in sleep) neither
accumulate nor erase evidence. After an alarm both CUSUMs reset and bouts are ignored for
`pause_s` (the patient has just been woken up).

Evidence for the watched arm ("self") vs the other arm ("contra"):
  contra moves alone ........... + w_contra_only   (the watched arm could have moved and did not)
  self moves alone ............. + w_self_only     (negative: it works, so less suspicion)
  both move, self < weak_ratio of contra ... + w_both_weak   (even if its orientation changes:
                                                  the body moved the arm, the arm did not)
  both move, similar amplitude . + w_both_ok      (negative; similar = ratio within ok_ratio..1/ok_ratio)
  anything else ................ 0
All weights, k, h and the ratios are placeholders chosen by eye (no real nights exist).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Params:
    k: float = 0.25               # CUSUM drift: evidence per bout needed just to stay level
    h: float = 4.0                # alarm threshold, in units of w_* (about 4 net suspicious bouts)
    w_contra_only: float = 1.0
    w_self_only: float = -1.0
    w_both_weak: float = 1.0
    w_both_ok: float = -0.5
    weak_ratio: float = 0.3       # both-bout: self amplitude / contra amplitude below this = weak
    ok_ratio: float = 0.5         # ratio within [ok_ratio, 1/ok_ratio] = similar
    pause_s: float = 30 * 60      # silence after an alarm
    ref_ratio: float = 1.0        # optional per-patient baseline: usual amp_l / amp_r (e.g. from 2 weeks of nights)


def evidence(kind, amp_self, amp_contra, p, ref=1.0):
    """Evidence e_t of one bout for the watched arm. `ref` = usual amp_self / amp_contra of this patient."""
    if kind == "self_only":
        return p.w_self_only
    if kind == "contra_only":
        return p.w_contra_only
    if amp_contra <= 0 or amp_self <= 0:
        return 0.0
    ratio = amp_self / amp_contra / ref
    if ratio < p.weak_ratio:
        return p.w_both_weak
    return p.w_both_ok if p.ok_ratio <= ratio <= 1 / p.ok_ratio else 0.0


def run(bouts, p=None):
    """bouts: find_bouts() output. Returns (curve, alarms):
    curve: one row per bout used (t_s = end of the bout, S_left, S_right; S before the reset on an alarm)
    alarms: DataFrame t_s, side ("left" = suspected left weakness)."""
    p = p or Params()
    S = {"left": 0.0, "right": 0.0}
    pause_until = -np.inf
    curve, alarms = [], []
    for b in bouts.itertuples():
        t = b.start_s + b.dur_s
        if t < pause_until:
            continue
        for side in S:
            mine, other = ("l", "r") if side == "left" else ("r", "l")
            kind = "both" if b.kind == "both" else "self_only" if b.kind == f"{side}_only" else "contra_only"
            amp_s, amp_c = getattr(b, f"amp_{mine}"), getattr(b, f"amp_{other}")
            ref = p.ref_ratio if side == "left" else 1 / p.ref_ratio
            S[side] = max(0.0, S[side] + evidence(kind, amp_s, amp_c, p, ref) - p.k)
        curve.append((t, S["left"], S["right"]))
        side = max(S, key=S.get)
        if S[side] > p.h:
            alarms.append((t, side))
            S = {"left": 0.0, "right": 0.0}
            pause_until = t + p.pause_s
    return (pd.DataFrame(curve, columns=["t_s", "S_left", "S_right"]),
            pd.DataFrame(alarms, columns=["t_s", "side"]))
