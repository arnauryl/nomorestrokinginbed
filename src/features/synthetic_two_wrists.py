"""Synthetic two-wrist recordings built from the real single-board test (2026-11-17).

LEFT = the real recording. RIGHT = synthetic, built from the real movement of the same
recording so that its statistics are real, not made-up noise:
  dyn_R = COUPLING * (left movement, delayed LAG_S) + (1 - COUPLING) * (independent movement,
          the same recording circularly shifted by 1/3 of its length)
Healthy: use dyn_R as it is.   Stroke: from ONSET_S on, dyn_R is multiplied by `weak`
(1 = no weakness, 0 = paralysed). The coupling/lag values are ASSUMPTIONS, not measurements.

Usage (repo root): python -m src.features.synthetic_two_wrists  -> data/processed/synth_*.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.models.arm_correlation import FS
from src.models.model_asimetria import carregar_csv
from scipy.signal import butter, sosfiltfilt

SRC = Path(__file__).resolve().parents[2] / "data/raw/prova_inicial_USB1_20261117_163638.csv"
OUT = Path(__file__).resolve().parents[2] / "data/processed"
COUPLING, LAG_S, ONSET_S, NOISE_G = 0.7, 0.16, 360, 0.005


def build(weak=1.0, onset_s=ONSET_S, seed=0):
    """Returns (left, right) arrays (n, 3) in g."""
    acc = carregar_csv(SRC)[["ax", "ay", "az"]].to_numpy()
    n = len(acc)
    sos = butter(2, 0.3, btype="highpass", fs=FS, output="sos")
    dyn = sosfiltfilt(sos, acc, axis=0)
    grav = acc - dyn
    rng = np.random.default_rng(seed)
    dyn_r = COUPLING * np.roll(dyn, int(LAG_S * FS), axis=0) + (1 - COUPLING) * np.roll(dyn, n // 3, axis=0)
    dyn_r[int(onset_s * FS):] *= weak
    right = np.roll(grav, n // 2, axis=0) + dyn_r + rng.normal(0, NOISE_G, (n, 3))
    return acc, right


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, weak in (("healthy", 1.0), ("stroke_mild", 0.4), ("stroke_severe", 0.1)):
        for side, a in zip("LR", build(weak)):
            df = pd.DataFrame(a, columns=["ax", "ay", "az"])
            df.insert(0, "t_s", np.arange(len(a)) / FS)
            df.to_csv(OUT / f"synth_{name}_{side}.csv", index=False, float_format="%.4f")
    print("written to", OUT)
