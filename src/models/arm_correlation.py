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

    TODO (you): high-pass each axis at fc Hz, then take the norm.
    """
    raise NotImplementedError
