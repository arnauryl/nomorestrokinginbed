"""Evaluation of the rule detector on SYNTHETIC nights (datos sintéticos), vs the correlation classifier.

For each bout rate: N healthy and N stroke nights (src.features.synthetic_nights). Per night the
slow part (features + bouts) is computed once; the accumulator threshold h is then swept.
Baseline: arm_correlation's rule (window "suspect" = best_cc < 0.5, "not informative" = no arm above
the noise floor) with an alarm after `n` consecutive suspect windows; not-informative windows are
skipped (they neither extend nor break the run), and the same 30 min pause follows an alarm.

Metrics: false alarms per healthy night; fraction of stroke nights with an alarm after onset;
minutes from onset to that first alarm (median, IQR). Alarms before onset in stroke nights are
counted per night in nights.csv but not in the false-alarm rate.

Usage (repo root): python -m src.eval.evaluate_detector [--nights 50] [--rates 2 5 10 20]
Writes results/eval_synthetic/{nights,sweep,summary}.csv and figures.
"""

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.synthetic_nights import make_night
from src.models.evidence_accumulator import Params, run
from src.models.rule_detector import extract, mark_chest_turns

OUT = Path(__file__).resolve().parents[2] / "results/eval_synthetic"
H_SWEEP = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16]
N_SWEEP = [1, 2, 3, 5, 8, 12, 20]
H_DEFAULT, N_DEFAULT = Params.h, 5
TITLE = "Datos sintéticos (synthetic data)"


def corr_alarms(win, n_suspect, threshold=0.5, pause_s=1800):
    """Alarm times (s) of the correlation baseline: n_suspect consecutive suspect windows."""
    out, streak, pause_until = [], 0, -np.inf
    for start, cc in zip(win["start_s"], win["best_cc"]):
        t = start + 60
        if t < pause_until or np.isnan(cc):
            continue
        streak = streak + 1 if cc < threshold else 0
        if streak >= n_suspect:
            out.append(t)
            streak, pause_until = 0, t + pause_s
    return out


def one_night(job):
    rate, stroke, seed, hours, w_turn = job
    acc_l, acc_r, p = make_night(seed, rate, hours, stroke)
    win, bouts = extract(acc_l, acc_r)
    turns = [(t, t + d) for t, k, d in p.pop("events") if k == "bilateral"]   # IDEAL chest sensor: every posture change seen
    chest = mark_chest_turns(bouts, turns)
    return {"rate": rate, "stroke": stroke, "seed": seed, "params": p, "n_bouts": len(bouts),
            "rule": {h: run(bouts, Params(h=h, w_both_weak_turn=w_turn))[1] for h in H_SWEEP},
            "rule_chest": {h: run(chest, Params(h=h, w_both_weak_turn=w_turn))[1] for h in H_SWEEP},
            "corr": {n: corr_alarms(win, n) for n in N_SWEEP}}


def score(times, sides, params):
    """One night, one model -> dict(n_alarms, n_before_onset, detected, delay_min, side_ok)."""
    times = np.asarray(times, float)
    onset = params["stroke"]["onset_min"] * 60 if params["stroke"] else None
    if onset is None:
        return {"n_alarms": len(times), "n_before_onset": np.nan, "detected": np.nan, "delay_min": np.nan, "side_ok": np.nan}
    after = np.flatnonzero(times >= onset)
    d = {"n_alarms": len(times), "n_before_onset": int((times < onset).sum()), "detected": len(after) > 0,
         "delay_min": (times[after[0]] - onset) / 60 if len(after) else np.nan, "side_ok": np.nan}
    if len(after) and sides is not None:
        d["side_ok"] = sides[after[0]] == params["stroke"]["side"]
    return d


def summarize(df):
    """df: rows from score() + rate/stroke columns -> one row per rate."""
    rows = []
    for rate, g in df.groupby("rate"):
        h, s = g[~g["stroke"]], g[g["stroke"]]
        delay = s["delay_min"].dropna()
        rows.append({"rate": rate, "n_healthy": len(h), "n_stroke": len(s),
                     "false_alarms_per_night": h["n_alarms"].mean(),
                     "detected_frac": s["detected"].astype(float).mean(),
                     "delay_median_min": delay.median(), "delay_q25_min": delay.quantile(.25),
                     "delay_q75_min": delay.quantile(.75)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nights", type=int, default=50, help="healthy and stroke nights per rate")
    ap.add_argument("--rates", type=float, nargs="+", default=[2, 5, 10, 20], help="bouts per hour")
    ap.add_argument("--hours", type=float, default=8)
    ap.add_argument("--w-turn", type=float, default=0.0, help="evidence for a weak arm during a whole-body turn")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    jobs = [(r, s, (5000 if s else 1000) + i, a.hours, a.w_turn) for r in a.rates for s in (False, True) for i in range(a.nights)]
    with Pool(a.workers) as pool:
        res = pool.map(one_night, jobs, chunksize=1)

    nights, long = [], []
    for r in res:
        p = r["params"]
        base = {"rate": r["rate"], "stroke": r["stroke"], "seed": r["seed"], "n_bouts": r["n_bouts"],
                "dominance_arm": p["dominance"]["arm"], "dominance_factor": p["dominance"]["factor"],
                **{f"n_{k}": v for k, v in p["n_events"].items()}, **(p["stroke"] or {})}
        nights.append({**base,
                       **{f"rule_{k}": v for k, v in score(r["rule"][H_DEFAULT]["t_s"], r["rule"][H_DEFAULT]["side"].tolist(), p).items()},
                       **{f"chest_{k}": v for k, v in score(r["rule_chest"][H_DEFAULT]["t_s"], r["rule_chest"][H_DEFAULT]["side"].tolist(), p).items()},
                       **{f"corr_{k}": v for k, v in score(r["corr"][N_DEFAULT], None, p).items()}})
        for m in ("rule", "rule_chest"):
            for h, al in r[m].items():
                long.append({"model": m, "param": h, "rate": r["rate"], "stroke": r["stroke"], **score(al["t_s"], al["side"].tolist(), p)})
        for n, al in r["corr"].items():
            long.append({"model": "correlation", "param": n, "rate": r["rate"], "stroke": r["stroke"], **score(al, None, p)})
    pd.DataFrame(nights).to_csv(a.out / "nights.csv", index=False)
    long = pd.DataFrame(long)
    sweep = pd.concat([summarize(g).assign(model=m, param=x) for (m, x), g in long.groupby(["model", "param"])],
                      ignore_index=True)[["model", "param", "rate", "n_healthy", "n_stroke", "false_alarms_per_night",
                                          "detected_frac", "delay_median_min", "delay_q25_min", "delay_q75_min"]]
    sweep.to_csv(a.out / "sweep.csv", index=False)
    defaults = sweep[((sweep.model.str.startswith("rule")) & (sweep.param == H_DEFAULT)) | ((sweep.model == "correlation") & (sweep.param == N_DEFAULT))]
    defaults.to_csv(a.out / "summary.csv", index=False)
    figures(sweep, a.out)
    example(a)
    pd.set_option("display.width", 200)
    print("\n== sweep (rule: param = h; correlation: param = consecutive suspect windows) ==")
    print(sweep.round(2).to_string(index=False))
    print(f"\n== defaults: rule h={H_DEFAULT:g}, correlation N={N_DEFAULT} ==")
    print(defaults.round(2).to_string(index=False))
    print(f"\nSaved to {a.out}")


def figures(sweep, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rates = sorted(sweep["rate"].unique())
    for ycol, ylabel, name in (("delay_median_min", "median minutes from onset to alarm (detected only)", "fa_vs_delay.png"),
                               ("detected_frac", "fraction of stroke nights detected", "fa_vs_detection.png")):
        fig, axes = plt.subplots(1, len(rates), figsize=(4 * len(rates), 4), sharey=True, squeeze=False)
        for ax, rate in zip(axes[0], rates):
            for model, color, label in (("rule", "tab:blue", "rule detector (h)"), ("rule_chest", "tab:green", "rule + ideal chest sensor (h)"), ("correlation", "tab:orange", "correlation (N windows)")):
                d = sweep[(sweep.model == model) & (sweep.rate == rate)].sort_values("param")
                ax.plot(d["false_alarms_per_night"], d[ycol], "o-", color=color, label=label)
                for _, r in d.iterrows():
                    ax.annotate(f"{r['param']:g}", (r["false_alarms_per_night"], r[ycol]), fontsize=7, xytext=(2, 3), textcoords="offset points")
            ax.set_title(f"{rate:g} bouts/h")
            ax.set_xlabel("false alarms per healthy night")
        axes[0][0].set_ylabel(ylabel)
        axes[0][0].legend(fontsize=8)
        fig.suptitle(f"{TITLE}: false alarms vs " + ("detection delay" if "delay" in ycol else "detection rate"))
        fig.tight_layout()
        fig.savefig(out / name, dpi=130)
        plt.close(fig)


def example(a):
    from src.models.rule_detector import plot
    acc_l, acc_r, p = make_night(5000, 10, a.hours, True)
    win, bouts = extract(acc_l, acc_r)
    curve, alarms = run(bouts)
    st = p["stroke"]
    plot({"windows": win, "bouts": bouts, "curve": curve, "alarms": alarms}, a.out / "example_night.png",
         title=f"{TITLE}: stroke night, 10 bouts/h, {st['side']} weak {st['severity']:.0%} from min {st['onset_min']:.0f}")


if __name__ == "__main__":
    main()
