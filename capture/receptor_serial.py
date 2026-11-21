"""Captura per cable USB (port sèrie), d'una o diverses plaques alhora.

Accepta els formats dels codis Arduino que ja tenim, sense tocar-los:
  - firmware/xiao_ble_raw/xiao_ble_raw.ino:  idx,ax_mg,ay_mg,az_mg        (50 Hz)
  - xiao_test.ino (prototip 1):          ax<TAB>ay<TAB>az  en g       (10 Hz)
  - adxl335_reading.ino:                 tiempo_ms,xb,yb,zb,x_g,y_g,z_g (50 Hz)
Quan la línia no porta comptador de mostres, el temps es pren del rellotge de l'ordinador.

Ús:
    python receptor_serial.py --port COM5 --etiqueta dreta_normal --minuts 30
    # tres plaques alhora (quan arribin les peces): un --port per placa, amb nom
    python receptor_serial.py --port COM5:D --port COM6:E --port COM7:P --etiqueta nit1
(A Windows el port surt a l'Arduino IDE, menú Eines > Port.)

Genera dades/<etiqueta>_<nom>_<data>.csv amb el mateix format que receptor_ble.py
(idx, t_s, t_pc, ax, ay, az en g), així que la resta del model funciona igual.
"""

import argparse
import csv
import os
import threading
import time
from datetime import datetime

import serial  # pip install pyserial

FS_XIAO_RAW = 50


def interpretar(linia):
    """Torna (idx o None, temps_ms o None, ax, ay, az en g), o None si la línia no és de dades."""
    sep = "\t" if "\t" in linia else ","
    parts = [p.strip() for p in linia.split(sep) if p.strip()]
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return None  # capçalera o text
    if len(nums) == 4:   # xiao_ble_raw: idx + mg
        return int(nums[0]), None, nums[1] / 1000, nums[2] / 1000, nums[3] / 1000
    if len(nums) == 3:   # xiao_test: g
        return None, None, nums[0], nums[1], nums[2]
    if len(nums) == 7:   # adxl335: temps_ms, bruts, g
        return None, nums[0], nums[4], nums[5], nums[6]
    return None


def gravar(port, nom, carpeta, etiqueta, final, aturar):
    fitxer = os.path.join(carpeta, f"{etiqueta}_{nom}_{datetime.now():%Y%m%d_%H%M%S}.csv")
    idx0 = t0_ms = t0_pc = None
    n = 0
    with serial.Serial(port, 115200, timeout=1) as s, open(fitxer, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "t_s", "t_pc", "ax", "ay", "az"])
        print(f"[{nom}] {port} -> {fitxer}")
        while time.time() < final and not aturar.is_set():
            r = interpretar(s.readline().decode(errors="ignore").strip())
            if r is None:
                continue
            idx, t_ms, ax, ay, az = r
            t_pc = time.time()
            if idx is not None:                       # comptador de la placa (el més precís)
                idx0 = idx if idx0 is None else idx0
                t_s = (idx - idx0) / FS_XIAO_RAW
            elif t_ms is not None:                    # temps de la placa en ms
                t0_ms = t_ms if t0_ms is None else t0_ms
                t_s = (t_ms - t0_ms) / 1000
            else:                                     # rellotge de l'ordinador
                t0_pc = t_pc if t0_pc is None else t0_pc
                t_s = t_pc - t0_pc
            w.writerow([n, f"{t_s:.3f}", f"{t_pc:.3f}", ax, ay, az])
            n += 1
            if n % 3000 == 0:
                f.flush()
                print(f"[{nom}] {t_s / 60:.1f} min, {n} mostres")
    print(f"[{nom}] fet: {n} mostres.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", action="append", required=True, help="COM5 o COM5:D (port:nom)")
    p.add_argument("--etiqueta", default="sessio")
    p.add_argument("--minuts", type=float, default=0)
    p.add_argument("--carpeta", default="data/raw")
    a = p.parse_args()
    os.makedirs(a.carpeta, exist_ok=True)
    final = time.time() + a.minuts * 60 if a.minuts > 0 else float("inf")
    aturar = threading.Event()
    fils = []
    for i, pn in enumerate(a.port):
        port, _, nom = pn.partition(":")
        fil = threading.Thread(target=gravar, args=(port, nom or f"USB{i + 1}", a.carpeta, a.etiqueta, final, aturar))
        fil.start()
        fils.append(fil)
    try:
        while any(f.is_alive() for f in fils):
            time.sleep(0.5)
    except KeyboardInterrupt:
        aturar.set()
        for f in fils:
            f.join()


if __name__ == "__main__":
    main()
