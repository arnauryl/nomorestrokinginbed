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
    """acc: array (n, 3) with ax, ay, az in g. Returns array (n,) with the magnitude of
    the dynamic acceleration (gravity removed).
    """
    
    """
    Utilitzem un HPF per diferents motius. 

    1. Un sensor sempre nota la gravetat, i el moviment que volem notar a la nit amb
    el moviment està un parell d'ordres de magnitud per sota de la gravetat. La 
    correlació ens informa de com dos senyals varien alhora, així que la correlació vindria
    donada per la gravetat, no el moviment.
    2. 
    """

    sos = butter(N=2, Wn=fc, fs=fs, btype="highpass", output="sos")

    """
    SOS: Second-Order Sections

    Idea del claude, tot i que no caldria per a un filtre d'ordre 2; també scipy ho 
    recomana com a pràctica habitual. 

    La idea és que per a filtres amb ordres més alts (4, 6, etc), si hi ha fórmules molt
    llargues, els valors exploten. És molt més estable poder fer una cadena de filtres
    de segon ordre. 
    """
    filtered_signal = sosfiltfilt(sos, acc, axis=0)
    acc_norm = np.linalg.norm(filtered_signal,axis=1)

    return acc_norm
