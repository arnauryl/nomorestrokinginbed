# nomorestrokinginbed

Predictive stroke model based on accelerometers worn on both wrists during sleep. The idea is simple: when a person sleeps, both arms move from time to time, and a stroke that weakens one arm should change how the two arms move relative to each other. We want to detect that change.

This repository contains the material from the initial test of 2026-11-17 (firmware, capture scripts, one recording) and the two models we have built so far. The hardware is two XIAO BLE Sense boards (nRF52840), one on each wrist. According to the firmware, each board reads its onboard LSM6DS3 accelerometer at 50 Hz and sends every sample over Bluetooth Low Energy (BLE). The ADXL335 does not appear anywhere in the repository files.

Important: we do not have a recording with two wrists yet, because the prototypes are still being built. Everything about the inter-arm correlation has only been tested on synthetic data (see "Synthetic test" below), so it is not evidence that the model works on patients.

## Repository structure

- `firmware/xiao_ble_raw/`: board firmware (`xiao_ble_raw.ino`), which sends raw acceleration at 50 Hz. The folder has the same name as the file because the Arduino IDE requires it.
- `capture/`: PC receivers that save the board data as CSV. `receptor_serial.py` works over USB and `receptor_ble.py` over Bluetooth.
- `data/raw/`: captured CSVs, with the columns `t_s, ax, ay, az` (time in seconds and acceleration in g). Right now there are two files from the 2026-11-17 test, each from a single board.
- `data/processed/`: derived data, such as the synthetic recordings. Git ignores this folder.
- `src/models/`: the models. `model_asimetria.py` is the asymmetry ratio, `arm_correlation.py` is the inter-arm correlation, and `prova_inicial.py` is the analysis of the initial test.
- `src/features/`: `synthetic_two_wrists.py`, which builds the synthetic two-wrist recordings.
- `results/`: output tables and figures.
- `notebooks/` and `docs/`: empty for now.

## Setup

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

All the commands below are run from the root of the repository.

## How to run each script

- Capture over USB: `python capture/receptor_serial.py --etiqueta sessio --minuts 10`
- Capture over Bluetooth: `python capture/receptor_ble.py --etiqueta sessio --minuts 10`. Both scripts save to `data/raw/` unless you pass `--carpeta`.
- Analysis of the initial test: `python -m src.models.prova_inicial data/raw/prova_inicial_USB1_20261117_163638.csv`. It writes `ma_per_fase.csv`, `ma2_per_frequencia.csv` and `prova_inicial.png` to `results/`. If you started the script a few seconds after the recording began, add `--inici SECONDS`.
- Inter-arm correlation on two recordings: `python -m src.models.arm_correlation data/raw/LEFT.csv data/raw/RIGHT.csv`. The options are `--win` (window length in seconds, 60 by default), `--step` (seconds between window starts, 30), `--lag` (maximum lag in seconds, 1.0), `--threshold` (0.5), `--noise` (noise floor in g, 0.015) and `--out` (output folder, `results/`). It prints one row per window and saves `arm_correlation_windows.csv` and `arm_correlation.png`.
- Self-check of the correlation model: `python -m src.models.arm_correlation` with no arguments. It runs the tests of every function and prints `all OK`.
- Synthetic recordings: `python -m src.features.synthetic_two_wrists`. It writes `synth_healthy`, `synth_stroke_mild` and `synth_stroke_severe` (each with an `_L` and an `_R` file) to `data/processed/`.
- Firmware: open `firmware/xiao_ble_raw/xiao_ble_raw.ino` in the Arduino IDE.

## The initial test (2026-11-17)

We recorded 13 minutes with one board on the wrist at 50 Hz, doing six activities of 2 minutes each: board on the table, arm still, wrist rotations, arm up and down with a flat wrist, free movements, and arm still again. The goal was to see whether the movement index (MA2, explained below) picks up wrist rotations, how much noise the sensor has, and which sampling rate we need. The results are in `results/ma_per_fase.csv`, `results/ma2_per_frequencia.csv` and `results/prova_inicial.png`.

## Model 1: asymmetry ratio (`model_asimetria.py`)

This model follows Iacovelli et al. 2019, "Actigraphic measurement of the upper limbs movements in acute stroke patients" (J NeuroEng Rehabil 16:153, doi:10.1186/s12984-019-0603-z), which uses the method of Rabuffetti.

- Activity indices: for each arm and each 1 minute epoch we compute MA1 (the standard deviation of the acceleration magnitude, so only linear movement) and MA2 (the magnitude of the vector made of the standard deviations of x, y and z, so linear movement plus rotations). The paper finds that MA2 is the useful one for patients in bed, because it detects the pronation and supination of the forearm, which is almost the only movement a mildly paretic arm can still do.
- Asymmetry ratio (AR): one point per epoch, with MA2 of the right arm on one axis and MA2 of the left arm on the other. We fit a line through the origin, take its angle θ, and compute `AR = (45° − θ) / 45° · 100`. A value of 0 means symmetric, +100 means only the right arm moves (suspected left paresis) and −100 means only the left arm moves.
- Adaptation to sleep: during sleep the arms rarely move at the same time, so the line fit sticks to one axis and the ratio jumps to ±100 even in healthy people. We keep the same scale and angle, but compute them from the total activity of each arm over the window.
- Detector: AR in moving windows of 1 hour recalculated every 5 minutes, a baseline for each patient learned from normal nights, windows with little movement marked as not informative, whole body movements (turning in bed) removed, and an alarm only if the deviation lasts 20 minutes in the same direction.

This model has not been run on a night recording yet. `prova_inicial.py` only uses its functions for the activity indices.

## Model 2: inter-arm correlation (`arm_correlation.py`)

The idea: if one arm stops moving in sync with the other, the correlation between the movement of the two wrists drops. It works in four steps, each one a function.

1. `preprocess`: turns the raw acceleration of one wrist (`ax, ay, az`) into one number per sample that says how much the arm is moving. A sensor always feels gravity (about 1 g), and the movement we want to see during the night is one or two orders of magnitude smaller, so gravity has to go. Gravity is also not constant: when the wrist rotates, it is shared differently between the three axes. For that reason we apply a high-pass filter (a filter that blocks slow changes and lets faster ones pass) to each axis, with a cutoff of 0.3 Hz, and only then take the magnitude `sqrt(ax² + ay² + az²)`. Subtracting the mean would not be enough, because it removes a fixed value and not a slow change. We use `sosfiltfilt`, which filters forwards and backwards, so it adds no time delay between the two wrists. The price is that it cannot run in real time as it is. `results/preprocess_real_data.png` shows the result on the real recording: still phases stay close to zero and the wrist rotations are the tallest.
2. `windows`: cuts both signals at the same moments into windows (1 to 5 minutes, 60 s by default) with a step between starts (30 s by default, so consecutive windows overlap). It needs both signals to have the same length and drops the last window if it is incomplete. A window gives one correlation value, so a short episode is not diluted in a whole night, and we can see when it happened. Short windows react faster but are noisier, long windows are steadier but slower.
3. `correlation`: for each window it returns three numbers. `pearson` is the correlation with no shift, from −1 to +1, and it only compares the shape of the two signals, not their size. `best_cc` is the highest Pearson found after sliding one signal against the other by up to ±1 s, which allows for the natural delay between the arms (if the right arm repeats the left movement 0.3 s later, plain Pearson would be low even though the arms are in sync). `best_lag_s` is the shift where that maximum happens, and a positive value means the right arm is ahead. The maximum lag must stay small: with a large one, unrelated movements start to match and the score is meaninglessly high. Taking the best of many shifts also raises the value a little even for unrelated signals, so `best_cc` has to be compared with a threshold measured in the same way.
4. `classify`: labels each window as "not informative", "normal" or "suspect". First an activity gate: if neither arm is above the noise floor, the correlation is only noise, so the window is not informative. It is enough that one arm moves, because the case we look for is one arm moving and the other not. Then, if `best_cc` is below the threshold, the window is suspect.

`analyze` chains the four steps and returns a table with one row per window (start time, `pearson`, `best_cc`, `best_lag_s`, mean movement of each arm and label). `plot` draws `best_cc` over time, coloured by label, together with the movement of each arm. `load_and_sync` reads the two CSVs and trims them to the same length.

Values that still have to be set:
- Threshold (0.5): a placeholder. It should be set just below the normal range, measured on healthy recordings.
- Noise floor (0.015 g): taken from the initial test, where a still sensor reads about 15 mg. The still windows of that recording had a mean between 0.013 and 0.018 g, which is too close to the floor, because the same situation flips between "not informative" and "normal". For that board a value around 0.03 g works better, but it has to be confirmed with the new prototypes.
- Synchronisation: `load_and_sync` assumes that both boards started recording together and at the same rate, and it only trims to the shorter one. It prints a warning if they differ by more than 2 s. If the boards do not start together, we need a real synchronisation, for example a tap on both wrists at the start.

## Synthetic test

Since there is no two-wrist data, `src/features/synthetic_two_wrists.py` builds it from the real recording. The left wrist is the real recording. The right wrist is 70% of the left movement delayed by 0.16 s plus 30% of independent movement (the same recording shifted in time), with a different gravity orientation and a little sensor noise. For the stroke versions, the right arm movement is multiplied by 0.4 (mild) or 0.1 (severe) from minute 6. The 70/30 mix, the delay and the onset are our assumptions, not measurements.

We ran `analyze` with the noise floor at 0.03 g and compared the 12 informative windows after the onset. The share of windows flagged as suspect was:

- Threshold 0.3: 25% in the healthy recording, 33% in the mild stroke and 58% in the severe stroke.
- Threshold 0.5: 33% healthy, 33% mild and 67% severe.
- Threshold 0.7: 42% healthy, 50% mild and 92% severe.
- Mean `best_cc`: 0.61 healthy, 0.53 mild and 0.32 severe.

What we conclude:
- A severe weakness is partly detected: the mean correlation drops, but the healthy recording also has flagged windows, so the margin is small. One window means little, and we would need several in a row, which `classify` does not do yet.
- A mild weakness is not detected. Pearson compares shape and not size, so an arm that makes the same movements with 40% of the strength still correlates well. This comes from how the correlation works, whatever the data.
- The healthy recording has false alarms: when one arm moves while the other is still, the model reads it as a stroke pattern. Real healthy people also do this, but our way of building the right arm may exaggerate it.

The test is also circular: we wrote the right arm to follow the left, and then checked whether the model notices when it stops following. It shows that the code works and what the correlation can and cannot see, but not that it works on a patient. Only real two-wrist recordings can show that.

## Architecture: where the model should run

We want an app on a phone that, when the model sees a possible stroke, wakes the patient and asks them to do some activities that assess their cognitive abilities. The question is where the model runs. This is a proposal, we have not built or tested any of it.

How the data travels today: the firmware sends a 20 byte packet (a counter plus three consecutive samples of x, y and z in mg), about 17 packets per second per board. That is a small amount of data for BLE.

Proposal: the model runs on the phone, and the boards only stream their data.

- Both wrists are needed in one place: each board only sees its own arm and the two boards do not talk to each other, so the correlation can only be computed where both streams arrive, which is the phone.
- Room for the model: the model is small, but the board already has to sample at 50 Hz and keep Bluetooth running, and it has little memory.
- The model will keep changing: the threshold, the noise floor and the window length are still to be set. An app update is easier than reflashing two boards each time.
- The alert is on the phone: the phone wakes the patient and runs the activities, so the decision and the action are in the same place.

Alternatives:
- Board that processes everything: it would need a third device (a hub) that talks to both boards and then to the phone. It adds hardware, a power supply and one more thing to debug, so we would only consider it if the phone cannot stay connected all night.
- Step 1 on the boards: if streaming at 50 Hz drains the batteries too much, each board could run `preprocess` and send one value per second, and the phone would do the rest. Then `sosfiltfilt` has to become a live filter (`sosfilt`), which adds some delay, and we would need to check that the correlation does not change much. The clocks of the two boards still have to line up.

Risks we have not solved:
- Overnight connection: the phone has to stay connected to both boards for a whole night, and iOS and Android limit Bluetooth in the background. If the connection drops, the model must not read missing data as an arm that stopped.
- False alarms: waking a patient by mistake is costly. In the synthetic test, the healthy recording also had windows flagged as suspect, so the app would need several suspect windows in a row before it wakes anyone, and `classify` does not do that yet.
- Language: the model is in Python, so the app needs it rewritten (for example in Kotlin or Swift). It is a few lines, but it has to be checked against the Python version with the same recordings.
- Validation: none of this has been tested on real two-wrist data.

For now: raw stream to the phone, model developed in Python on recordings, and the decision about on-board processing after we know how long the batteries last.

## Not done yet

- Step (d) of the plan, a table comparing the correlation model with the asymmetry ratio (windows and mean values per phase), because it needs two-wrist data.
- Calibration of the threshold and the noise floor with healthy recordings.
- A real synchronisation between the two boards.
- Running the asymmetry ratio model on a night recording.

## Rule-based detector (`src/models/rule_detector.py`)

**All the numbers in this section come from synthetic data (datos sintéticos). They only check that the logic works; they are not clinical evidence.**

Why it replaces the correlation as the main signal: Pearson does not change if a signal is multiplied by 0.1, so it cannot measure how much each arm moves (`tests/test_rule_detector.py` shows it), and a healthy person scratching with one hand gives low correlation. The main signal is now the movement asymmetry (Iacovelli 2019, Reale 2023, Messé 2023), counted per movement bout, and evidence is accumulated over the night. Still arms add nothing, and one suspicious window wakes nobody up.

Pipeline: `load_and_sync` → `preprocess` (+ gravity) → bouts → window features → CUSUM accumulator → alarms.

- `python -m src.features.noise_floor`: noise floor = mean + 3 std of 1 s block means of the gravity-free magnitude in the "quiet" and "final" phases (about 0.050 g). Writes `results/noise_floor.json`, used by default everywhere.
- `src/features/bouts.py`: bouts (arm above the floor, gaps < 2 s merged, runs < 3 s dropped) classified `left_only` / `right_only` / `both` (overlap ≥ 50 % of the shorter), with amplitude and orientation change per wrist.
- `src/features/window_features.py`: per 60 s window (30 s step): `act_l, act_r, asym, pearson, best_cc, best_lag_s, orient_change_l/r, n_bouts_*`. Correlation is only computed in windows where an arm is active.
- `src/models/evidence_accumulator.py`: one one-sided CUSUM per side, `S = max(0, S + e − k)`, alarm when `S > h`, then reset and 30 min pause. Each parameter is documented in the file. `Params.ref_ratio` is the hook for a per-patient baseline.
- `python -m src.models.rule_detector LEFT.csv RIGHT.csv [--floor G --k K --h H --pause MIN --title T --out DIR]`: writes `rule_detector_{windows,bouts,curve,alarms}.csv` and `rule_detector.png` (activity, bouts by type, accumulator with threshold and alarms).
- `python -m src.features.synthetic_nights --seeds 0 1 --rate 5 [--stroke]`: 8 h two-wrist nights built from fragments of the real recording (CSV per wrist + JSON with seed, rate, dominance, onset, side, severity). Same seed, same night. Writes to `data/processed/synthetic_nights/` (about 45 MB per wrist).
- `python -m src.eval.evaluate_detector [--nights 50 --rates 2 5 10 20]`: 50 healthy + 50 stroke nights per bout rate, sweep of `h`, comparison with the correlation classifier (alarm after N consecutive suspect windows). Writes `results/eval_synthetic/`: `nights.csv` (one row per night), `sweep.csv`, `summary.csv`, `fa_vs_delay.png`, `fa_vs_detection.png`, `example_night.png`. Takes about 5 min.
- Tests: `python -m pytest`.

Bouts where both arms are under `min_amp` (30 mg above the floor) are ignored. In a whole-body turn (both wrists rotate more than 30°, or `chest_turn` from a chest sensor via `mark_chest_turns`) the "weak arm" evidence is switched off (`w_both_weak_turn = 0`). The evaluation also runs `rule_chest`, an IDEAL chest sensor that flags every posture change.

Chosen by eye, not fitted: `k = 0.5`, `h = 4`, evidence weights (±1, −0.5), `weak_ratio = 0.3`, `ok_ratio = 0.5`, bout minimum 3 s, 30 min pause, correlation baseline N = 5.

Prepared but not done: `align_by_clap` (sync with a clap), an optional per-patient reference in the accumulator, and the window features as inputs for a future XGBoost/RF.

## Limitations

- Everything is synthetic, built from 13 minutes of one healthy, awake person with one board. Movement amplitudes and shapes are those of that person awake.
- The movement rate during sleep is an assumption (2–20 bouts/h tested), as are the proportions of bilateral and one-arm events, the dominance range, and the stroke model (severity 5–60 %, abrupt or ramped).
- Circularity is reduced (each arm uses different fragments, stroke only scales amplitude) but not eliminated: the detector is tested against the same kind of movement that generated the nights.
- The quiet base is resampled from only 200 s of real data, so its noise bouts (about 5/h) repeat the statistics of those 200 s.
- The two boards are assumed to be synchronized (trimmed to the same length).
