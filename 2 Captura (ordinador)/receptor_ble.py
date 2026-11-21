"""Rep les dades de la XIAO per Bluetooth (BLE) i les guarda en CSV.

Ús:
    python receptor_ble.py                         # busca "ICTUS-D" i grava fins a Ctrl+C
    python receptor_ble.py --nom ICTUS-D --etiqueta dreta_normal --minuts 30
    python receptor_ble.py --nom ICTUS-D --nom ICTUS-E   # quan tinguem dues plaques

Cada placa genera un fitxer dades/<etiqueta>_<nom>_<data>.csv amb les columnes:
    idx      número de mostra (50 per segon), reconstruït a partir del comptador de paquets
    t_s      temps de la mostra en segons des del principi (idx / 50)
    t_pc     hora de l'ordinador (epoch Unix) quan va arribar el paquet: serveix per
             alinear plaques diferents, que no comparteixen rellotge
    ax,ay,az acceleració en g

Cada minut mostra per pantalla els índexs MA1 i MA2 del minut acabat (Iacovelli 2019),
perquè es vegi en directe que el sensor respon.
"""

import argparse
import asyncio
import csv
import os
import struct
import time
from datetime import datetime

import numpy as np
from bleak import BleakClient, BleakScanner

UUID_CRU = "7a1e0003-5c3b-4f2a-9d6e-2b8c4a10f001"
FS = 50
MOSTRES_PER_PAQUET = 3
FORMAT = "<H9h"  # seq + 3 mostres x (x,y,z) en mg


class Gravacio:
    def __init__(self, nom, fitxer):
        self.nom = nom
        self.f = open(fitxer, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(["idx", "t_s", "t_pc", "ax", "ay", "az"])
        self.seq_ant = None
        self.voltes = 0
        self.seq0 = None
        self.seq_ant_abs = None
        self.reconnectat = False
        self.perduts = 0
        self.rebuts = 0
        self.minut = []  # mostres del minut en curs per al resum en directe
        self.minut_idx = 0

    def paquet(self, dades: bytearray):
        t_pc = time.time()
        if len(dades) != struct.calcsize(FORMAT):
            return
        v = struct.unpack(FORMAT, dades)
        seq = v[0]
        if self.reconnectat:
            # Després d'una desconnexió: si el comptador ha anat enrere (la placa s'ha
            # reiniciat), continuem la numeració on l'havíem deixat.
            self.reconnectat = False
            nou = seq + 65536 * self.voltes
            if nou <= self.seq_ant_abs:
                self.seq0 -= self.seq_ant_abs + 1 - nou
            self.seq_ant = None
        if self.seq_ant is not None:
            if seq < self.seq_ant and self.seq_ant - seq > 30000:
                self.voltes += 1  # el comptador de 16 bits ha donat la volta (~cada hora)
            salt = (seq + 65536 * self.voltes) - (self.seq_ant_abs + 1)
            if salt > 0:
                self.perduts += salt
        seq_abs = seq + 65536 * self.voltes
        if self.seq0 is None:
            self.seq0 = seq_abs  # la gravació comença a idx 0
        self.seq_ant, self.seq_ant_abs = seq, seq_abs
        seq_abs -= self.seq0
        self.rebuts += 1
        for i in range(MOSTRES_PER_PAQUET):
            idx = seq_abs * MOSTRES_PER_PAQUET + i
            ax, ay, az = (v[1 + 3 * i + j] / 1000.0 for j in range(3))
            self.w.writerow([idx, f"{idx / FS:.2f}", f"{t_pc:.3f}", ax, ay, az])
            self.minut.append((ax, ay, az))
        if len(self.minut) >= 60 * FS:
            a = np.array(self.minut)
            ma1 = np.linalg.norm(a, axis=1).std()
            ma2 = np.linalg.norm(a.std(axis=0))
            print(f"[{self.nom}] minut {self.minut_idx:4d}  MA1={ma1:.4f} g  MA2={ma2:.4f} g  "
                  f"paquets perduts={self.perduts}")
            self.minut, self.minut_idx = [], self.minut_idx + 1
            self.f.flush()

    def tanca(self):
        self.f.close()


async def grava_placa(nom, carpeta, etiqueta, final):
    marca = datetime.now().strftime("%Y%m%d_%H%M%S")
    fitxer = os.path.join(carpeta, f"{etiqueta}_{nom}_{marca}.csv")
    g = Gravacio(nom, fitxer)
    print(f"[{nom}] gravant a {fitxer}")
    try:
        while time.time() < final:
            print(f"[{nom}] buscant la placa...")
            dev = await BleakScanner.find_device_by_name(nom, timeout=20)
            if dev is None:
                print(f"[{nom}] no la trobo. Està encesa i a prop? Torno a provar.")
                continue
            desconnectat = asyncio.Event()
            try:
                async with BleakClient(dev, disconnected_callback=lambda c: desconnectat.set()) as client:
                    print(f"[{nom}] connectada ({dev.address}).")
                    await client.start_notify(UUID_CRU, lambda _, d: g.paquet(d))
                    while time.time() < final and not desconnectat.is_set():
                        await asyncio.sleep(0.5)
                    if client.is_connected:
                        await client.stop_notify(UUID_CRU)
            except Exception as e:  # una desconnexió no ha d'aturar la gravació
                print(f"[{nom}] error de connexió: {e}")
            if time.time() < final:
                print(f"[{nom}] desconnectada, reconnectant...")
                if g.seq_ant_abs is not None:
                    g.reconnectat = True
    finally:
        g.tanca()
        print(f"[{nom}] fitxer tancat: {g.rebuts} paquets rebuts, {g.perduts} perduts.")


async def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nom", action="append", help="nom BLE de la placa (es pot repetir)")
    p.add_argument("--etiqueta", default="sessio", help="p. ex. dreta_normal, esquerra_paresi")
    p.add_argument("--minuts", type=float, default=0, help="durada; 0 = fins a Ctrl+C")
    p.add_argument("--carpeta", default="dades")
    a = p.parse_args()
    noms = a.nom or ["ICTUS-D"]
    os.makedirs(a.carpeta, exist_ok=True)
    final = time.time() + a.minuts * 60 if a.minuts > 0 else float("inf")
    await asyncio.gather(*(grava_placa(n, a.carpeta, a.etiqueta, final) for n in noms))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Aturat per l'usuari.")
