"""Wspolne elementy wykresow kosztu obliczeniowego (UNS i SUP)."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt   # backend wybiera wywolujacy (skrypt: Agg, notebook: inline)
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter

TRANSITIONS = {"AB": "A--B", "AC": "A--C", "BCb": r"B--C$_b$"}
TRANSITION_MARKERS = {"AB": "o", "AC": "s", "BCb": "^"}

MODEL_PL = {
    "KMeans": "$k$-średnich",
    "Agglomerative": "Grupowanie aglomeracyjne",
    "Birch": "Birch",
    "Spectral": "Grupowanie spektralne",
    "GaussianMixture": "Mieszanina gaussowska",
    "BayesianGMM": "Bayesowska Miesz. gaussowska",
    "MeanShift": "MeanShift",
    "DBSCAN": "DBSCAN",
    "Decision Tree": "Drzewo decyzyjne",
    "Logistic Regression": "Regresja logistyczna",
    "Neural Network": "Sieć neuronowa",
    "Random Forest": "Las losowy",
    "Gradient Boosted Trees": "Drzewa wzmacniane",
    "SVM (RBF)": "SVM (RBF)",
    "kNN": "kNN",
}

STYL = {
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.5,
}

LINIA = dict(lw=1.1, ms=3.4, alpha=0.9, markeredgecolor="white",
             markeredgewidth=0.35)


def wczytaj(katalog, wzorzec, kolumny_num):
    """Sklada ramki wszystkich przejsc; `wzorzec` np. '{kod}_SUP_summary.csv'."""
    ramki = []
    for kod in TRANSITIONS:
        nazwa = wzorzec.format(kod=kod)
        for sciezka in (Path(katalog) / kod / nazwa, Path(katalog) / nazwa):
            if sciezka.exists():
                break
        else:
            print(f"BRAK: {Path(katalog) / kod / nazwa}")
            continue
        d = pd.read_csv(sciezka)
        d["transition"] = kod
        ramki.append(d)
    if not ramki:
        raise SystemExit(f"Nie znaleziono zadnych plikow w {katalog}")
    dane = pd.concat(ramki, ignore_index=True)
    for kol in kolumny_num:
        dane[kol] = pd.to_numeric(dane[kol], errors="coerce")
    return dane.dropna(subset=["model", *kolumny_num])


def palety(dane):
    modele = sorted(dane["model"].unique())
    return {m: plt.cm.tab10(i % 10) for i, m in enumerate(modele)}


def rysuj_serie(ax, dane, x_kol, y_kol, kolory, markery, marker_kol):
    """Jedna linia na (model, wartosc marker_kol). Zwraca uzyte klucze."""
    uzyte_modele, uzyte_markery = set(), set()
    for (model, klucz), d in dane.groupby(["model", marker_kol], sort=False):
        d = d.sort_values(x_kol)
        ax.plot(d[x_kol], d[y_kol], color=kolory[model],
                marker=markery[klucz], zorder=3, **LINIA)
        uzyte_modele.add(model)
        uzyte_markery.add(klucz)
    return uzyte_modele, uzyte_markery


def os_dyskretna(ax, wartosci, obrot=45):
    """Osi x nadaje tylko realnie wystepujace wartosci zamiast dekad."""
    ax.xaxis.set_major_locator(FixedLocator(sorted(wartosci)))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):d}"))
    if obrot:
        ax.tick_params(axis="x", labelrotation=obrot, labelsize=6.5)
        for e in ax.get_xticklabels():
            e.set_horizontalalignment("right")
            e.set_rotation_mode("anchor")
    ax.grid(True, axis="y", which="both", alpha=0.22, lw=0.5)
    ax.grid(True, axis="x", which="major", alpha=0.30, lw=0.5)


def legenda(fig, modele, kolory, markery, opisy_markerow, dodatkowe=(),
            ncol=4, dol=0.13):
    """Legenda budowana wylacznie z faktycznie narysowanych serii."""
    uchwyty = [Line2D([], [], color=kolory[m], marker="o",
                      label=MODEL_PL.get(m, m), **LINIA)
               for m in sorted(modele)]
    uchwyty += [Line2D([], [], color="0.35", marker=markery[k],
                       label=opisy_markerow[k], **LINIA)
                for k in opisy_markerow if k in markery]
    uchwyty += list(dodatkowe)
    fig.tight_layout(rect=[0, dol, 1, 1])
    fig.legend(handles=uchwyty, loc="lower center", ncol=ncol, frameon=False,
               bbox_to_anchor=(0.5, 0.005), handletextpad=0.4,
               columnspacing=1.2, handlelength=1.6)


def zapisz(fig, nazwa):
    for fmt in ("pdf", "png"):
        fig.savefig(f"{nazwa}.{fmt}", bbox_inches="tight",
                    dpi=300 if fmt == "png" else None)
