"""Prova inicial amb una sola placa: el MA2 capta els girs de canell? quin soroll té el sensor?
quina freqüència de mostreig necessitem?

Fa servir model_asimetria.py (a la mateixa carpeta) per calcular MA1 i MA2.
La gravació es fa amb "../2 Captura (ordinador)/receptor_serial.py".

Ús (des d'aquesta carpeta):
    python prova_inicial.py "../4 Dades/prova_inicial_USB1_20261117_163638.csv"
    # si heu començat el guió uns segons després d'engegar la gravació:
    python prova_inicial.py fitxer.csv --inici 15

Guió (fases de 2 min): vegeu "00 INFORME.md". Resultats a "5 Resultats".
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)  # en aquesta còpia, model_asimetria.py és a la mateixa carpeta
from model_asimetria import carregar_csv, indexs_per_epoca  # noqa: E402

# (nom curt, descripció, durada en s)
FASES = [
    ("taula", "Placa quieta sobre la taula", 120),
    ("quiet", "Placa al canell, braç quiet i recolzat", 120),
    ("girs", "Girs de canell (pronosupinació), avantbraç recolzat", 120),
    ("lineal", "Pujar i baixar tot el braç amb el canell pla, sense girar-lo", 120),
    ("lliure", "Moviments normals lliures", 120),
    ("final", "Braç quiet i recolzat", 120),
]
EPOCA_S = 10           # èpoques curtes perquè cada fase en tingui 12
MARGE_S = 10           # es descarten 10 s a cada canvi de fase (transicions)
FREQS = [50, 33, 25, 10]


def assignar_fases(df, inici):
    df = df.copy()
    df["fase"] = None
    t = inici
    for nom, _, dur in FASES:
        dins = (df["t_s"] >= t + MARGE_S) & (df["t_s"] < t + dur - MARGE_S)
        df.loc[dins, "fase"] = nom
        t += dur
    return df


def freq_efectiva(df):
    dt = np.diff(df["t_s"].to_numpy())
    dt = dt[dt > 0]
    return 1 / np.median(dt), np.mean(dt > 1.5 * np.median(dt)) * 100


def remostrejar(df, fs):
    """Simula una placa que mostreja a fs Hz: es queda la mostra més propera a cada instant 1/fs
    (sense interpolar, per no suavitzar artificialment el soroll)."""
    t = np.arange(df["t_s"].iloc[0], df["t_s"].iloc[-1], 1 / fs)
    i = np.clip(np.searchsorted(df["t_s"].to_numpy(), t), 0, len(df) - 1)
    return df.iloc[i][["t_s", "ax", "ay", "az"]].reset_index(drop=True)


def per_fase(df, inici):
    ep = indexs_per_epoca(df, epoca_s=EPOCA_S)
    ep["t_mig"] = (ep.index + 0.5) * EPOCA_S
    fases = assignar_fases(pd.DataFrame({"t_s": ep["t_mig"]}), inici)["fase"].to_numpy()
    ep["fase"] = fases
    return ep


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv")
    p.add_argument("--inici", type=float, default=0, help="segon de la gravació on comença la fase 1")
    p.add_argument("--sortida", default=os.path.join(AQUI, "..", "5 Resultats"))
    a = p.parse_args()
    os.makedirs(a.sortida, exist_ok=True)

    df = carregar_csv(a.csv)
    fs, forats = freq_efectiva(df)
    durada = df["t_s"].iloc[-1]
    necessari = a.inici + sum(d for *_, d in FASES)
    print(f"Gravació: {durada / 60:.1f} min, {len(df)} mostres, freqüència efectiva {fs:.1f} Hz, "
          f"{forats:.1f}% de salts de temps")
    if durada < necessari - 5:
        print(f"ATENCIÓ: la gravació és més curta ({durada:.0f} s) que el guió ({necessari:.0f} s).")

    # 1. MA1 i MA2 per fase
    ep = per_fase(df, a.inici)
    files = []
    for nom, desc, _ in FASES:
        e = ep[ep["fase"] == nom]
        if e.empty:
            continue
        files.append({"fase": nom, "descripcio": desc, "epoques": len(e),
                      "MA1_mg": 1000 * e["MA1"].median(), "MA2_mg": 1000 * e["MA2"].median(),
                      "MA2/MA1": e["MA2"].median() / max(e["MA1"].median(), 1e-6)})
    taula = pd.DataFrame(files)
    taula.to_csv(os.path.join(a.sortida, "ma_per_fase.csv"), index=False)

    # 2. Freqüència de mostreig: MA2 a cada freqüència, relatiu al de la freqüència original
    freqs = [f for f in FREQS if f <= fs * 1.05]
    comp = {}
    for f in freqs:
        e = per_fase(remostrejar(df, f), a.inici)
        comp[f] = e.groupby("fase")["MA2"].median() * 1000
    comp = pd.DataFrame(comp).reindex([n for n, *_ in FASES]).dropna(how="all")
    rel = comp.div(comp[freqs[0]], axis=0) * 100
    comp.to_csv(os.path.join(a.sortida, "ma2_per_frequencia.csv"))

    # 3. Resum en text
    pd.set_option("display.width", 140)
    print("\n== MA1 i MA2 per fase (mediana de les èpoques de 10 s, en mg) ==")
    print(taula.drop(columns="descripcio").round(2).to_string(index=False))
    print(f"\n== MA2 a cada freqüència, en % del valor a {freqs[0]} Hz ==")
    print(rel.round(0).to_string())

    t = taula.set_index("fase")
    print("\n== Comprovacions ==")
    if {"taula", "quiet"} <= set(t.index):
        print(f"Soroll del sensor (MA2 a la taula): {t.loc['taula', 'MA2_mg']:.1f} mg "
              f"(model_asimetria.py suposa 7,5 mg)")
        print(f"Braç quiet: {t.loc['quiet', 'MA2_mg']:.1f} mg")
    if {"girs", "lineal"} <= set(t.index):
        ok = t.loc["girs", "MA2/MA1"] > 2 * t.loc["lineal", "MA2/MA1"]
        print(f"Girs: MA2/MA1 = {t.loc['girs', 'MA2/MA1']:.1f}; lineal: {t.loc['lineal', 'MA2/MA1']:.1f} -> "
              + ("OK, el MA2 capta els girs molt més que el MA1 (com Iacovelli)." if ok else
                 "REVISAR: no es veu la diferència esperada entre girs i moviment lineal."))
    if len(freqs) > 1:
        pitjor = rel.drop(index=["taula", "quiet", "final"], errors="ignore").drop(columns=freqs[0])
        for f in pitjor.columns:
            desv = (pitjor[f] - 100).abs().max()
            print(f"{f} Hz: en les fases amb moviment, el MA2 es desvia com a màxim un {desv:.0f}% del de {freqs[0]} Hz"
                  + (" -> suficient" if desv < 10 else " -> perd informació"))

    figura(df, ep, a, os.path.join(a.sortida, "prova_inicial.png"))
    print(f"\nResultats a {a.sortida}")


def figura(df, ep, a, fitxer):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for c, col in zip(("ax", "ay", "az"), ("tab:red", "tab:green", "tab:blue")):
        ax1.plot(df["t_s"] / 60, df[c], lw=0.5, color=col, label=c)
    ax1.set_ylabel("Acceleració (g)")
    ax1.legend(loc="upper right")
    ax2.plot(ep["t_mig"] / 60, 1000 * ep["MA1"], "o-", ms=3, label="MA1 (mòdul)")
    ax2.plot(ep["t_mig"] / 60, 1000 * ep["MA2"], "o-", ms=3, label="MA2 (components)")
    ax2.set_yscale("log")
    ax2.set_ylabel("Índex per època de 10 s (mg)")
    ax2.set_xlabel("Temps (min)")
    ax2.legend(loc="upper right")
    t = a.inici
    for i, (nom, _, dur) in enumerate(FASES):
        for ax in (ax1, ax2):
            ax.axvspan(t / 60, (t + dur) / 60, color="0.92" if i % 2 else "white", zorder=0)
        ax1.text((t + dur / 2) / 60, 1.02, nom, transform=ax1.get_xaxis_transform(), ha="center")
        t += dur
    fig.tight_layout()
    fig.savefig(fitxer, dpi=130)


if __name__ == "__main__":
    main()
