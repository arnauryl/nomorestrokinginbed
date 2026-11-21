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

    TODO (you).
    """
    raise NotImplementedError
