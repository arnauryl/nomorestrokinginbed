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

    TODO (you).
    """
    n = len(left)

    assert len(left) == len(right), "Left and right do not have the same lenghts"
    pearson = np.corrcoef(left, right)[0,1]

    max_lag_samples = int(max_lag_s*fs)

    for k in range(-max_lag_samples, max_lag_samples +1):
        if k < 0: 
            left_part = left[:n+k]
            right_part = right[-k:]
        else: 
            left_part = left[:n+k]
            right_part = right[-k:]

        cc = np.corrcoef(left_part, right_part)[0,1]

    return pearson, best_cc, best_lag_s


if __name__ == "__main__":
    # Self-check de windows(): 60 s a 50 Hz, finestres de 20 s cada 10 s -> 5 finestres.
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
    print("windows OK")
