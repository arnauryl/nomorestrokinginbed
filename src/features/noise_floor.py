"""Activity noise floor, calibrated on the still phases of the real recording.

floor = mean + 3 * std of the 1 s block means of the gravity-free magnitude (preprocess()),
taken from the phases "quiet" and "final" (board on the wrist, arm still; "taula" is excluded
because the board is off the body). Saved to results/noise_floor.json and used as the
default everywhere instead of the fixed 0.015 g.

Usage (repo root): python -m src.features.noise_floor
"""

import json
from pathlib import Path

import numpy as np

from src.models.arm_correlation import FS, preprocess
from src.models.model_asimetria import carregar_csv
from src.models.prova_inicial import FASES, MARGE_S

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/raw/prova_inicial_USB1_20261117_163638.csv"
JSON = ROOT / "results/noise_floor.json"
STILL_PHASES = ("quiet", "final")


def phase_bounds(name, margin=MARGE_S):
    """(start_s, end_s) of a phase of the 2026-11-17 script, minus the transition margin.
    Assumes the script started at second 0 (prova_inicial.py --inici 0)."""
    t = 0
    for n, _, dur in FASES:
        if n == name:
            return t + margin, t + dur - margin
        t += dur
    raise KeyError(name)


def block_means(mag, fs=FS):
    """Mean of mag in consecutive 1 s blocks (the incomplete last block is dropped)."""
    n = len(mag) // fs * fs
    return np.asarray(mag[:n]).reshape(-1, fs).mean(axis=1)


def calibrate(csv=SRC, fs=FS, n_sigma=3.0):
    acc = carregar_csv(csv)[["ax", "ay", "az"]].to_numpy()
    mag = preprocess(acc, fs)
    blocks = np.concatenate([block_means(mag[int(a * fs):int(b * fs)], fs)
                             for a, b in map(phase_bounds, STILL_PHASES)])
    mean, std = float(blocks.mean()), float(blocks.std())
    return {"floor_g": mean + n_sigma * std, "mean_g": mean, "std_g": std, "n_sigma": n_sigma,
            "n_blocks": len(blocks), "phases": list(STILL_PHASES), "source": Path(csv).name}


def load_floor():
    """Calibrated floor in g (from results/noise_floor.json; calibrates on the fly if missing)."""
    return json.loads(JSON.read_text())["floor_g"] if JSON.exists() else calibrate()["floor_g"]


if __name__ == "__main__":
    cal = calibrate()
    JSON.parent.mkdir(exist_ok=True)
    JSON.write_text(json.dumps(cal, indent=2))
    print(json.dumps(cal, indent=2))
