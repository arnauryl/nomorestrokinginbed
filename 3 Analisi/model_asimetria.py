"""Primer model de detecció d'asimetria motora entre braços durant el son.

Base científica: Iacovelli et al. 2019, "Actigraphic measurement of the upper limbs
movements in acute stroke patients", J NeuroEng Rehabil 16:153,
doi:10.1186/s12984-019-0603-z (PMC6894254), amb el mètode de Rabuffetti.

1. Índexs d'activitat per època d'1 minut i per braç (com el firmware del paper):
     MA1 = desviació típica del mòdul de l'acceleració            (només lineal)
     MA2 = mòdul del vector (σx, σy, σz) de les tres components    (lineal + rotacions)
   El paper troba que MA2 és el bo per a pacients al llit: detecta la pronosupinació,
   que és gairebé l'únic moviment d'un braç parètic lleu.

2. Índex d'asimetria del paper, AR2 (index_ar_paper):
   núvol de punts (MA2 dret a l'eix x, MA2 esquerre a l'eix y), recta per l'origen que
   minimitza els residus (primer vector singular, SVD), angle θ respecte de l'eix x:
       AR = (45° − θ) / 45° · 100
   0 = simètric; +100 = només es mou el dret (paresi ESQUERRA); −100 = només l'esquerre.
   Al paper, AR2 de 24 h identifica el braç parètic en el 95 % dels pacients i
   correlaciona amb el NIHSS de braç (r = 0,81).

3. Per què l'hem d'adaptar per a la nit (index_ar_son):
   durant el son els moviments són escassos i gairebé mai simultanis (un braç es mou,
   l'altre està quiet). Amb punts sobre els eixos, el primer vector singular de la SVD
   s'enganxa a l'eix del braç que ha fet el moviment més gran: en una finestra d'1 h
   l'AR salta a ±100 en persones sanes. Ho hem comprovat amb les nits simulades.
   Mantenim la mateixa escala i el mateix angle, però calculat sobre l'activitat total
   de cada braç a la finestra (suma de MA2 per sobre del soroll del sensor):
       θ = atan(Σ MA2_esq / Σ MA2_dret),  AR_son = (45° − θ) / 45° · 100
   Quan els dos braços es mouen junts (el cas del paper) dona el mateix resultat.

4. Què hi afegim per passar de "mesura de 24 h" a "alarma nocturna":
   - AR en finestres mòbils d'1 h, recalculades cada 5 min.
   - Línia base pròpia de cada pacient (nits de calibratge, ~14): la dominància manual i
     la manera de dormir de cadascú defineixen la seva normalitat.
   - Finestres amb menys de 10 minuts de moviment = "no informatives" (no es decideix).
   - Es treuen les èpoques de moviment de tot el cos (girar-se al llit), que mouen el braç
     parètic passivament. El paper mostra que treure els moviments passius millora molt
     la correlació amb el NIHSS (r de 0,41 a 0,71). Amb el pegat del pit es fa
     directament; amb dos braços sols, amb una regla aproximada.
   - Alarma només si la desviació es manté 20 min seguits en el mateix sentit.
   Com que l'AR és una proporció entre braços, la febre o l'ansietat (més moviment als
   dos braços) i el cansament (menys moviment als dos) no el desplacen: és la resposta a
   la preocupació d'en Toni sobre les falses alarmes.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

FS = 50
SOROLL_REPOS_G = 0.0075   # MA2 d'un sensor completament quiet (soroll del LSM6DS3 a 50 Hz, aprox.)


# ---------------------------------------------------------------------------
# Lectura i índexs per època
# ---------------------------------------------------------------------------

def carregar_csv(fitxer):
    """Llegeix un CSV de receptor_ble.py / receptor_serial.py (columnes t_s, ax, ay, az), amb t_s des de 0."""
    df = pd.read_csv(fitxer)[["t_s", "ax", "ay", "az"]].astype(float)
    df["t_s"] -= df["t_s"].iloc[0]   # temps des del principi de la gravació
    return df


def indexs_per_epoca(df, epoca_s=60):
    """Calcula MA1, MA2 i la postura mitjana per època. Torna un DataFrame indexat per època (minut)."""
    ep = (df["t_s"] // epoca_s).astype(int)
    a = df[["ax", "ay", "az"]].to_numpy()
    modul = np.linalg.norm(a, axis=1)
    g = pd.DataFrame({"ep": ep, "ax": a[:, 0], "ay": a[:, 1], "az": a[:, 2], "m": modul}).groupby("ep")
    std = g[["ax", "ay", "az"]].std()
    mitjana = g[["ax", "ay", "az"]].mean()
    out = pd.DataFrame({
        "MA1": g["m"].std(),
        "MA2": np.sqrt((std ** 2).sum(axis=1)),
        "gx": mitjana["ax"], "gy": mitjana["ay"], "gz": mitjana["az"],
        "n": g.size(),
    })
    # Èpoques incompletes (paquets perduts, final de gravació) fora. La freqüència es
    # dedueix de les dades: 50 Hz amb xiao_ble_raw, ~10 Hz amb el xiao_test antic.
    dt = np.median(np.diff(df["t_s"].to_numpy())) if len(df) > 1 else 1.0 / FS
    fs = 1.0 / dt if dt > 0 else FS
    return out[out["n"] >= 0.8 * epoca_s * fs]


def alinear(ep_d, ep_e, ep_p=None):
    """Uneix les èpoques del braç dret, l'esquerre i (si n'hi ha) el pit que existeixen a tots."""
    ep = ep_d.join(ep_e, lsuffix="_d", rsuffix="_e", how="inner")
    if ep_p is not None:
        ep = ep.join(ep_p.add_suffix("_p"), how="inner")
    return ep


# ---------------------------------------------------------------------------
# Índexs d'asimetria
# ---------------------------------------------------------------------------

def _ar_des_dangle(theta_graus):
    return (45.0 - theta_graus) / 45.0 * 100.0


def index_ar_paper(dreta, esquerra):
    """AR del paper (SVD, recta per l'origen). Positiu = predomina el dret."""
    X = np.column_stack([dreta, esquerra]).astype(float)
    X = X[np.all(np.isfinite(X), axis=1)]
    if len(X) < 2 or not np.any(X):
        return np.nan
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    vx, vy = np.abs(vt[0])
    return _ar_des_dangle(np.degrees(np.arctan2(vy, vx)))


def index_ar_son(dreta, esquerra, soroll=SOROLL_REPOS_G):
    """AR adaptat al son: angle de l'activitat total de cada braç per sobre del soroll."""
    d = np.maximum(np.asarray(dreta, float) - soroll, 0).sum()
    e = np.maximum(np.asarray(esquerra, float) - soroll, 0).sum()
    if d + e == 0:
        return np.nan
    return _ar_des_dangle(np.degrees(np.arctan2(e, d)))


def moviment_global(ep, llindar_fort_g=0.1, llindar_g=0.05, gir_graus=30.0):
    """Marca les èpoques de moviment de tot el cos (el braç es mou passivament).

    Amb el pegat del pit (columna MA2_p): qualsevol moviment del tronc.
    Sense pit (ara, amb una sola XIAO): (a) els dos braços es mouen fort alhora, o
    (b) es mouen moderadament i tots dos canvien d'orientació més de `gir_graus`.
    Durant el son, un moviment fort i simultani dels dos braços és gairebé sempre un gir.
    """
    if "MA2_p" in ep:
        return (ep["MA2_p"] > 0.02).to_numpy()

    def canvi_orientacio(s):
        g = ep[[f"gx_{s}", f"gy_{s}", f"gz_{s}"]].to_numpy()
        g = g / np.linalg.norm(g, axis=1, keepdims=True)
        cos = np.clip(np.sum(g[1:] * g[:-1], axis=1), -1, 1)
        return np.concatenate([[0.0], np.degrees(np.arccos(cos))])

    fort = (ep["MA2_d"] > llindar_fort_g) & (ep["MA2_e"] > llindar_fort_g)
    moderat = (ep["MA2_d"] > llindar_g) & (ep["MA2_e"] > llindar_g)
    gir = (canvi_orientacio("d") > gir_graus) & (canvi_orientacio("e") > gir_graus)
    return (fort | (moderat & gir)).to_numpy()


# ---------------------------------------------------------------------------
# AR en finestres mòbils
# ---------------------------------------------------------------------------

@dataclass
class Parametres:
    finestra_min: int = 60              # durada de cada finestra d'AR
    pas_min: int = 5                    # cada quant es recalcula
    llindar_moviment_g: float = 0.015   # MA2 a partir del qual una època compta com a "moviment"
    min_epoques_actives: int = 10       # mínim de minuts amb moviment per decidir
    percentil_llindar: float = 99.0     # llindar = aquest percentil de |AR − mitjana| a la línia base
    persistencia_min: int = 20          # la desviació s'ha de mantenir aquest temps
    metode: str = "son"                 # "son" (adaptat) o "paper" (SVD original)


def ar_finestres(ep, p: Parametres = Parametres()):
    """AR en finestres mòbils. Torna un DataFrame amb inici/fi (min), AR i èpoques actives."""
    ep = ep.sort_index()
    excl = moviment_global(ep)
    d = ep["MA2_d"].to_numpy()
    e = ep["MA2_e"].to_numpy()
    t = ep.index.to_numpy()
    f_ar = index_ar_son if p.metode == "son" else index_ar_paper
    files = []
    for inici in range(int(t.min()), int(t.max()) - p.finestra_min + 2, p.pas_min):
        dins = (t >= inici) & (t < inici + p.finestra_min)
        sel = dins & ~excl
        n_act = int((sel & ((d > p.llindar_moviment_g) | (e > p.llindar_moviment_g))).sum())
        ar = f_ar(d[sel], e[sel]) if n_act >= p.min_epoques_actives else np.nan
        files.append({"inici_min": inici, "fi_min": inici + p.finestra_min, "AR": ar,
                      "epoques_actives": n_act, "epoques_excloses": int((dins & excl).sum())})
    return pd.DataFrame(files)


# ---------------------------------------------------------------------------
# Detector amb línia base per pacient
# ---------------------------------------------------------------------------

class DetectorAsimetria:
    """Aprèn la normalitat d'un pacient i avisa si l'AR se n'allunya de manera sostinguda."""

    def __init__(self, p: Parametres = Parametres()):
        self.p = p
        self.mitjana = None
        self.llindar = None

    def calibrar(self, nits, min_finestres=20):
        """nits: llista de DataFrames d'èpoques alineades (alinear()) de nits normals."""
        ars = pd.concat([ar_finestres(n, self.p)["AR"] for n in nits]).dropna().to_numpy()
        if len(ars) < min_finestres:
            raise ValueError("Massa poques finestres informatives per calibrar (calen més nits).")
        self.mitjana = float(np.median(ars))
        self.llindar = float(np.percentile(np.abs(ars - self.mitjana), self.p.percentil_llindar))
        self.n_finestres_base = len(ars)
        return self

    def analitzar(self, ep):
        """Torna les finestres amb l'estat: normal / sospita / ALARMA / no informativa."""
        if self.mitjana is None:
            raise RuntimeError("Primer cal calibrar() amb nits normals del pacient.")
        w = ar_finestres(ep, self.p)
        desv = w["AR"] - self.mitjana
        w["desviacio"] = desv
        w["sospita"] = desv.abs() > self.llindar
        signe = np.sign(desv)
        alarma = w["sospita"].copy()
        for j in range(1, self.p.persistencia_min // self.p.pas_min + 1):
            alarma &= w["sospita"].shift(j, fill_value=False) & (signe == signe.shift(j))
        w["alarma"] = alarma
        # desviació positiva = el dret es mou més del normal -> sospita de paresi ESQUERRA
        w["costat_sospitos"] = np.where(desv > 0, "esquerre", np.where(desv < 0, "dret", ""))
        w["estat"] = np.select([w["AR"].isna(), w["alarma"], w["sospita"]],
                               ["no informativa", "ALARMA", "sospita"], default="normal")
        return w

    @staticmethod
    def primera_alarma(w):
        a = w[w["alarma"]]
        if a.empty:
            return None
        f = a.iloc[0]
        return {"minut": int(f["fi_min"]), "costat": f["costat_sospitos"], "AR": float(f["AR"])}
