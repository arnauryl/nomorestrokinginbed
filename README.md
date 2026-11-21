# nomorestrokinginbed
Predictive stroke model: accelerometers on both wrists (XIAO BLE + ADXL335).
Initial test of 2026-11-17.

## Structure
```
firmware/xiao_ble_raw/   board firmware (.ino), 50 Hz raw acceleration
capture/                 PC receivers (USB serial, BLE) -> CSV in data/raw/
data/raw/                captured CSVs (t_s, ax, ay, az in g)
data/processed/          derived data (ignored by git)
src/models/              model_asimetria.py (Iacovelli asymmetry ratio), prova_inicial.py
src/features/            (empty)
notebooks/  docs/        (empty)
results/                 output tables and figures
```

## Setup
```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run (from the repo root)
- Capture over USB: `python capture/receptor_serial.py --etiqueta sessio --minuts 10`
- Capture over BLE: `python capture/receptor_ble.py --etiqueta sessio --minuts 10`
  (both write to `data/raw/` by default; `--carpeta` to change)
- Initial test analysis (MA1/MA2 per phase, sampling rate, figure):
  `python -m src.models.prova_inicial data/raw/prova_inicial_USB1_20261117_163638.csv [--inici SECONDS]`
  -> `results/`
- `src/models/model_asimetria.py` is a library (imported, not run).
- Firmware: open `firmware/xiao_ble_raw/xiao_ble_raw.ino` in the Arduino IDE.
