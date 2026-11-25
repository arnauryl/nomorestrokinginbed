import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.synthetic_nights import make_night
from src.features.window_features import activity, asymmetry
from src.models.arm_correlation import FS, correlation
from src.models.rule_detector import detect

ROOT = Path(__file__).resolve().parents[1]
FLOOR = 0.05


def wrist(n, rng, bursts=()):
    """(n, 3) accelerometer: gravity on z, sensor noise, and (start_s, dur_s) bursts of movement on x."""
    acc = np.tile([0, 0, 1.0], (n, 1)) + rng.normal(0, 0.003, (n, 3))
    for s, d in bursts:
        acc[int(s * FS):int((s + d) * FS), 0] += rng.normal(0, 0.3, int(d * FS))
    return acc


def test_pearson_ignores_scale_but_asym_does_not():
    # Why we moved from correlation to asymmetry: a weak arm that moves like the strong one
    # (right = 0.1 * left) has the same Pearson, so correlation cannot see the weakness.
    left = np.abs(np.random.default_rng(0).normal(0.2, 0.1, 3000))
    p1, _, _ = correlation(left, left)
    p2, _, _ = correlation(left, 0.1 * left)
    assert abs(p1 - p2) < 1e-9
    assert asymmetry(activity(left, 0), activity(left, 0)) == 0
    assert asymmetry(activity(left, 0), activity(0.1 * left, 0)) > 0.8


def test_still_arms_no_bouts_no_alarms():
    rng = np.random.default_rng(1)
    res = detect(wrist(10 * 60 * FS, rng), wrist(10 * 60 * FS, rng), floor=FLOOR)
    assert res["bouts"].empty and res["alarms"].empty
    assert res["curve"].empty or res["curve"][["S_left", "S_right"]].to_numpy().max() == 0


def test_bout_kinds():
    rng = np.random.default_rng(2)
    n = 3 * 60 * FS
    res = detect(wrist(n, rng, [(10, 8), (100, 8)]), wrist(n, rng, [(12, 8), (50, 8)]), floor=FLOOR)
    assert res["bouts"]["kind"].tolist() == ["both", "right_only", "left_only"]


def test_one_arm_moving_alone_alarms_the_other_side():
    rng = np.random.default_rng(3)
    n = 30 * 60 * FS
    left = wrist(n, rng, [(20 * i, 6) for i in range(90)])        # left moves every 20 s, right never
    res = detect(left, wrist(n, rng), floor=FLOOR)
    assert len(res["alarms"]) >= 1 and (res["alarms"]["side"] == "right").all()
    assert res["curve"]["S_left"].max() == 0


def test_same_seed_same_night():
    a = make_night(7, 10, hours=3, stroke=True)
    b = make_night(7, 10, hours=3, stroke=True)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and a[2] == b[2]
    assert not np.array_equal(a[0], make_night(8, 10, hours=3, stroke=True)[0])
    assert a[2]["events"] == make_night(7, 10, hours=3, stroke=False)[2]["events"]   # healthy twin has the same events


def test_prova_inicial_unchanged(tmp_path):
    subprocess.run([sys.executable, "-m", "src.models.prova_inicial", "data/raw/prova_inicial_USB1_20261117_163638.csv",
                    "--sortida", str(tmp_path)], cwd=ROOT, check=True, capture_output=True)
    for name in ("ma_per_fase.csv", "ma2_per_frequencia.csv"):
        pd.testing.assert_frame_equal(pd.read_csv(tmp_path / name), pd.read_csv(ROOT / "results" / name))


def test_small_bouts_and_turns():
    from src.models.evidence_accumulator import Params, run
    def bouts(**kw):
        row = dict(start_s=0., dur_s=10., kind="both", amp_l=0.02, amp_r=0.3, orient_l=0., orient_r=0.)
        return pd.DataFrame([{**row, **kw}] * 20).assign(start_s=lambda d: np.arange(20) * 20.)
    p = Params(h=3)
    assert len(run(bouts(), p)[1]) >= 1                                  # weak left arm in 20 normal bouts: alarm
    assert run(bouts(amp_l=0.001, amp_r=0.02), p)[0].empty               # both tiny: ignored
    assert run(bouts(orient_l=60., orient_r=60.), p)[1].empty            # body turn (no chest): weak evidence off
    assert run(bouts().assign(chest_turn=True), p)[1].empty              # chest sensor says turn
    assert len(run(bouts(orient_l=60., orient_r=10.), p)[1]) >= 1        # only one wrist rotates: not a turn
