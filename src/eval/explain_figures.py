"""Explanatory figures on SYNTHETIC data (datos sintéticos) -> results/explain/.
Usage (repo root): python -m src.eval.explain_figures"""

from pathlib import Path

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eval.evaluate_detector import OUT, TITLE, corr_alarms
from src.features.noise_floor import load_floor
from src.features.synthetic_nights import make_night
from src.models.evidence_accumulator import Params, run
from src.models.rule_detector import extract, plot

DEST = Path(__file__).resolve().parents[2] / "results/explain"
SEED, RATE = 7, 10
COL = {"left_only": "tab:blue", "right_only": "tab:orange", "both": "tab:green"}


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    nights = {}
    for stroke in (False, True):
        l, r, p = make_night(SEED, RATE, 8, stroke)
        nights[stroke] = (*extract(l, r), p)
    ps = nights[True][2]["stroke"]
    onset = ps["onset_min"] * 60
    for stroke, name in ((False, "healthy"), (True, "stroke")):
        w, b, p = nights[stroke]
        c, a = run(b)
        st = f", {ps['side']} arm weak ({ps['severity']:.0%} strength left)" if stroke else ""
        plot({"windows": w, "bouts": b, "curve": c, "alarms": a}, DEST / f"1_night_{name}.png",
             title=f"{TITLE}: {name} night, {RATE} bouts/h{st}", onset_s=onset if stroke else None)

    # 2. why k: same healthy night, three values of k
    b = nights[False][1]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), sharey=True)
    for ax, k in zip(axes, (0.0, 0.25, 0.5)):
        c, a = run(b, Params(k=k))
        ax.step(c["t_s"] / 60, c["S_left"], where="post", label="S left weak")
        ax.step(c["t_s"] / 60, c["S_right"], where="post", label="S right weak")
        ax.axhline(Params.h, color="r", ls="--", lw=.8)
        for t in a["t_s"]:
            ax.axvline(t / 60, color="r", lw=.8, alpha=.6)
        ax.set_title(f"k = {k}: {len(a)} false alarms")
        ax.set_xlabel("time (min)")
    axes[0].set_ylabel("accumulated evidence (h = 4, red)")
    axes[0].legend()
    fig.suptitle(f"{TITLE}: the same HEALTHY night with different k (without k a random walk always reaches h)")
    fig.tight_layout(); fig.savefig(DEST / "2_why_k.png", dpi=130); plt.close(fig)

    # 3. correlation baseline on the healthy night
    w = nights[False][0]
    al = corr_alarms(w, 5)
    fig, ax = plt.subplots(figsize=(12, 4))
    ok = w["best_cc"].notna()
    ax.scatter(w.loc[ok, "start_s"] / 60, w.loc[ok, "best_cc"], s=10,
               c=np.where(w.loc[ok, "best_cc"] < 0.5, "tab:red", "tab:green"))
    ax.axhline(0.5, ls="--", color="k", lw=.8)
    for t in al:
        ax.axvline(t / 60, color="tab:red", alpha=.5)
    ax.set_xlabel("time (min)"); ax.set_ylabel("best_cc (red = suspect)")
    ax.set_title(f"{TITLE}: HEALTHY night, correlation classifier: {len(al)} alarms (red lines; max possible = 16 with a 30 min pause)")
    fig.tight_layout(); fig.savefig(DEST / "3_correlation_false_alarms.png", dpi=130); plt.close(fig)

    # 4. bout amplitude: what min_amp removes
    fl = load_floor()
    fig, ax = plt.subplots(figsize=(10, 4))
    for kind in COL:
        d = b[b.kind == kind]
        ax.hist(1000 * np.maximum(d.amp_l, d.amp_r).clip(lower=1e-3), bins=np.geomspace(1, 600, 40), alpha=.6, color=COL[kind], label=kind)
    ax.axvline(1000 * Params.min_amp, color="r", ls="--"); ax.text(1000 * Params.min_amp, ax.get_ylim()[1] * .9, " min_amp: ignored below", color="r")
    ax.set_xscale("log"); ax.set_xlabel("larger arm amplitude of the bout above the floor (mg)"); ax.set_ylabel("bouts")
    ax.set_title(f"{TITLE}: healthy night, bouts by size (noise floor = {1000 * fl:.0f} mg)"); ax.legend()
    fig.tight_layout(); fig.savefig(DEST / "4_bout_sizes.png", dpi=130); plt.close(fig)

    # 5. trade-off: before filters, now, correlation
    chosen = json.loads((OUT / "chosen.json").read_text())
    now = pd.read_csv(OUT / "test/sweep.csv")
    now = now[(now.model != "rule") | (now.k == chosen["k"])]          # one curve: the k chosen on the tune nights
    old = pd.read_csv(OUT / "sweep_v1_before_filters.csv")             # earlier version (k=0.5, no filters), tune seeds
    rates = sorted(now.rate.unique())
    fig, axes = plt.subplots(1, len(rates), figsize=(4 * len(rates), 4), sharey=True)
    for ax, rate in zip(axes, rates):
        for df, m, lab, c in ((old, "rule", "before filters (tune nights)", "tab:gray"), (now, "rule", f"now, k={chosen['k']:g} (test nights)", "tab:blue"), (now, "correlation", "correlation", "tab:orange")):
            d = df[(df.model == m) & (df.rate == rate)].sort_values("param")
            ax.plot(d.false_alarms_per_night, d.detected_frac, "o-", color=c, label=lab, ms=4)
        ax.set_xlim(-0.2, 4); ax.set_title(f"{rate:g} bouts/h"); ax.set_xlabel("false alarms per healthy night")
    axes[0].set_ylabel("fraction of strokes detected"); axes[0].legend(fontsize=8)
    fig.suptitle(f"{TITLE}: detection vs false alarms (correlation sits at 11-16 false alarms, off the chart); each point = one h or N")
    fig.tight_layout(); fig.savefig(DEST / "5_tradeoff.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
