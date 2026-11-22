"""
Inter-arm correlation model: if one wrist stops moving in sync with the other,
the correlation between their acceleration magnitudes drops.

Pipeline (one function per step):
  1. preprocess   raw (ax, ay, az) of one wrist -> dynamic magnitude (gravity removed)
  2. windowing    two aligned magnitude series -> list of windows (1-5 min)
  3. correlation  Pearson + cross-correlation with a small max lag, per window
  4. threshold    correlation per window -> normal / suspect
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

FS = 50  # Hz, same as model_asimetria.py


def preprocess(acc, fs=FS, fc=0.3):
    """acc: array (n, 3) amb ax, ay, az en g. Retorna array (n,) amb el mòdul de
    l'acceleració dinàmica (sense gravetat).
    """

    # Filtre passa-alts (HPF) per treure la gravetat, per dos motius:
    # 1. Un sensor sempre nota la gravetat (~1 g), i el moviment a la nit és un parell
    #    d'ordres de magnitud més petit. La correlació mesura com varien alhora dos
    #    senyals, així que sense filtrar dependria de la gravetat, no del moviment.
    # 2. La gravetat no és constant: quan el canell gira, es reparteix de manera diferent
    #    entre ax, ay i az. Restar la mitjana només treuria un valor fix; el filtre
    #    segueix els canvis lents (< fc) i deixa passar el moviment, que és més ràpid.
    # fc = 0.3 Hz és el paràmetre a ajustar.
    #
    # Format SOS (second-order sections): per a ordre 2 és equivalent a la forma ba normal.
    # Es fa servir perquè és la pràctica recomanada per SciPy: en ordres més alts, els errors
    # d'arrodoniment s'amplifiquen amb la forma ba, i amb SOS (cadena de filtres d'ordre 2) no.
    sos = butter(N=2, Wn=fc, fs=fs, btype="highpass", output="sos")

    # axis=0: filtra al llarg del temps (files), cada columna (ax, ay, az) per separat.
    # sosfiltfilt aplica el filtre endavant i endarrere: no introdueix retard de fase,
    # així un retard artificial no es confondria amb un lag real entre canells (pas 3).
    # Contrapartida: no és causal, no es pot fer servir en temps real tal com està.
    filtered_signal = sosfiltfilt(sos, acc, axis=0)

    # axis=1: combina les tres columnes de cada mostra en un sol nombre, sqrt(ax²+ay²+az²).
    acc_norm = np.linalg.norm(filtered_signal, axis=1)

    return acc_norm


def windows(mag_l, mag_r, fs=FS, win_s=60, step_s=30):
    """mag_l, mag_r: arrays (n,) from preprocess(), same length, same time axis.
    Returns a list of (start_s, left_segment, right_segment), one per full window.
    The last incomplete window is dropped.
    """
    # Els dos senyals s'han de poder tallar amb els mateixos índexs: la mostra i de l'esquerre
    # i la mostra i del dret han de ser del mateix instant. La sincronització en el temps es
    # fa abans (pas a); aquí només es comprova que tinguin la mateixa longitud.
    if len(mag_l) != len(mag_r):
        raise ValueError(f"mag_l and mag_r must have the same length ({len(mag_l)} != {len(mag_r)})")

    # Els arrays s'indexen en mostres, no en segons.
    win = int(win_s * fs)
    step = int(step_s * fs)
    n = len(mag_l)

    # L'última finestra completa comença a n - win; el +1 és perquè range no inclou el límit.
    out = []
    for start in range(0, n - win + 1, step):
        out.append((start / fs, mag_l[start:start + win], mag_r[start:start + win]))
    return out


def correlation(left, right, fs=FS, max_lag_s=1.0):
    """left, right: two segments (n,) of the same window (from windows()).
    Returns (pearson, best_cc, best_lag_s):
      pearson     correlation with no shift
      best_cc     highest correlation found shifting one signal by -max_lag_s..+max_lag_s
      best_lag_s  the shift (in seconds) where best_cc happens
                  (> 0: right is ahead, left[i + k] matches right[i])
    """
    n = len(left)

    assert len(left) == len(right), "Left and right do not have the same length"
    pearson = np.corrcoef(left, right)[0,1]

    max_lag_samples = int(max_lag_s*fs)
    best_cc, pos = -np.inf, 0

    # Sign convention: k > 0 -> left[i + k] is compared with right[i] (right is ahead).
    for k in range(-max_lag_samples, max_lag_samples +1):

        if k < 0: 
            left_part = left[:n+k]
            right_part = right[-k:]
        else: 
            left_part = left[k:]
            right_part = right[:n-k]

        cc_new = np.corrcoef(left_part, right_part)[0,1]

        if cc_new > best_cc:
              best_cc, pos = cc_new, k


    best_lag_s = pos/fs

    return pearson, best_cc, best_lag_s


def classify(best_cc, left, right, threshold=0.5, noise_floor=0.015):
    """Label one window: "not informative", "normal" or "suspect".
    left, right: the window pieces (magnitudes in g). noise_floor: g below which an arm
    counts as not moving (a still sensor reads ~0.015 g, see the 2026-11-17 test).
    threshold: placeholder, to be set from healthy recordings.
    """
    # Porta d'activitat: si cap dels dos braços es mou, la correlació és soroll.
    # Només cal que es mogui UN braç (el cas de l'ictus: un es mou i l'altre no).
    if max(left.mean(), right.mean()) <= noise_floor:
        return "not informative"
    return "suspect" if best_cc < threshold else "normal"


def load_and_sync(csv_l, csv_r):
    """Read the two wrist CSVs (columns t_s, ax, ay, az) -> two (n, 3) arrays, same length.
    ponytail: assumes both boards started recording together and sample at the same rate;
    it only trims to the shorter one. Needs a real sync (e.g. a shared tap) if they don't.
    """
    from src.models.model_asimetria import carregar_csv
    l, r = carregar_csv(csv_l), carregar_csv(csv_r)
    n = min(len(l), len(r))
    if abs(len(l) - len(r)) > 2 * FS:
        print(f"WARNING: the recordings differ by {abs(len(l) - len(r)) / FS:.1f} s, check synchronization")
    return l[["ax", "ay", "az"]].to_numpy()[:n], r[["ax", "ay", "az"]].to_numpy()[:n]


def analyze(acc_l, acc_r, fs=FS, win_s=60, step_s=30, max_lag_s=1.0,
            threshold=0.5, noise_floor=0.015, fc=0.3):
    """Whole model: raw (n, 3) arrays of both wrists -> DataFrame with one row per window."""
    mag_l, mag_r = preprocess(acc_l, fs, fc), preprocess(acc_r, fs, fc)
    rows = []
    for start_s, l, r in windows(mag_l, mag_r, fs, win_s, step_s):
        pearson, best_cc, lag = correlation(l, r, fs, max_lag_s)
        rows.append({"start_s": start_s, "pearson": pearson, "best_cc": best_cc, "best_lag_s": lag,
                     "act_l": l.mean(), "act_r": r.mean(),
                     "label": classify(best_cc, l, r, threshold, noise_floor)})
    return pd.DataFrame(rows)


def plot(df, path, threshold=0.5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"normal": "tab:green", "suspect": "tab:red", "not informative": "0.6"}
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    a1.plot(df["start_s"] / 60, df["best_cc"], "k-", lw=0.8)
    a1.scatter(df["start_s"] / 60, df["best_cc"], c=df["label"].map(colors), zorder=3)
    a1.axhline(threshold, ls="--", color="tab:red", lw=0.8)
    a1.set_ylabel("best_cc (green normal, red suspect, grey not informative)")
    a2.plot(df["start_s"] / 60, 1000 * df["act_l"], label="left")
    a2.plot(df["start_s"] / 60, 1000 * df["act_r"], label="right")
    a2.set_ylabel("mean movement (mg)")
    a2.set_xlabel("window start (min)")
    a2.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def selfcheck():
    # windows(): 60 s a 50 Hz, finestres de 20 s cada 10 s -> 5 finestres.
    x = np.arange(3000, dtype=float)
    w = windows(x, x, fs=50, win_s=20, step_s=10)
    assert [s for s, _, _ in w] == [0, 10, 20, 30, 40]
    assert all(len(l) == len(r) == 1000 for _, l, r in w)
    assert len(windows(np.zeros(3100), np.zeros(3100), 50, 20, 10)) == 5  # sobren 100 mostres
    try:
        windows(np.zeros(3000), np.zeros(2900))
        raise AssertionError("should have raised ValueError")
    except ValueError:
        pass

    # correlation(): el dret és l'esquerre desplaçat 10 mostres (0.2 s).
    a = np.random.default_rng(0).normal(size=3000)
    pearson, best_cc, best_lag_s = correlation(a, np.roll(a, 10))
    assert abs(pearson) < 0.1 and best_cc > 0.99 and abs(best_lag_s) == 0.2

    # classify()
    rng = np.random.default_rng(1)
    still, moving = np.full(3000, 0.005), np.abs(rng.normal(0.3, 0.1, 3000))
    assert classify(0.3, still, still) == "not informative"
    assert classify(0.05, moving, still) == "suspect"
    assert classify(0.9, moving, moving) == "normal"

    # pipeline sencer: 10 min; el braç dret es mou com l'esquerre els 5 primers minuts
    # i després queda quiet -> finestres de 60 s: normal, normal... suspect.
    n = 10 * 60 * FS
    env = np.repeat(rng.random(n // 250 + 1) > 0.5, 250)[:n] * 0.2
    sig = env * rng.normal(size=n)
    def acc(xmove):
        m = np.tile([0, 0, 1.0], (n, 1)) + rng.normal(0, 0.005, (n, 3))
        m[:, 0] += xmove
        return m
    right = np.roll(sig, 5)
    right[n // 2:] = 0
    df = analyze(acc(sig), acc(right), win_s=60, step_s=60)
    assert (df.loc[df.start_s < 300, "label"] == "normal").all(), df
    assert (df.loc[df.start_s >= 300, "label"] == "suspect").all(), df
    print("all OK")


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    ap = argparse.ArgumentParser(description="Inter-arm correlation. No arguments: self-check.")
    ap.add_argument("left", nargs="?", help="CSV of the left wrist")
    ap.add_argument("right", nargs="?", help="CSV of the right wrist")
    ap.add_argument("--win", type=float, default=60, help="window length (s)")
    ap.add_argument("--step", type=float, default=30, help="step between windows (s)")
    ap.add_argument("--lag", type=float, default=1.0, help="max lag (s)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--noise", type=float, default=0.015, help="noise floor (g)")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[2] / "results")
    a = ap.parse_args()
    if not a.left:
        selfcheck()
    else:
        df = analyze(*load_and_sync(a.left, a.right), win_s=a.win, step_s=a.step,
                     max_lag_s=a.lag, threshold=a.threshold, noise_floor=a.noise)
        a.out.mkdir(parents=True, exist_ok=True)
        df.to_csv(a.out / "arm_correlation_windows.csv", index=False)
        plot(df, a.out / "arm_correlation.png", a.threshold)
        print(df.round(3).to_string(index=False))
        print(df["label"].value_counts().to_string())
        print(f"\nSaved to {a.out}")
