"""
Koszt obliczeniowy modeli nadzorowanych - 3 panele (jedno przejscie kazdy),
wspolna legenda. Bez szatkowania i bez limitow czasu.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from koszt_wspolne import (TRANSITIONS, STYL, LINIA, wczytaj, palety,
                           rysuj_serie, os_dyskretna, legenda, zapisz)

# ---------------------------------------------------------------- konfiguracja
DATA_DIR = "SUPERVISED"
WZORZEC = "{kod}_summary.csv"

X_KOL = "n_features"          # os x
Y_KOL = "runtime_model"       # os y
MARKER_KOL = "calibration"    # ksztalt znacznika

X_OPIS = "liczba cech"
Y_OPIS = "czas modelu [s]"

# ksztalty + podpisy dla wartosci MARKER_KOL
MARKERY = {"uncalibrated": "o", "calibrated": "s"}
OPISY_MARKEROW = {"uncalibrated": "bez kalibracji",
                  "calibrated": "z kalibracją"}

# --- filtrowanie modeli ------------------------------------------------------
# Zostawiamy tylko podstawowy wariant tam, gdzie liczylismy kilka.
MODELE_POMIN = []             # jawne wyjatki, gdyby byly potrzebne

# Z kazdej rodziny nazw zostaje tylko dokladne dopasowanie do przedrostka,
# wiec warianty w rodzaju "kNN kd_tree" czy "Logistic Regression C = 0.00175"
# odpadaja same - takze te, ktore dojda w przyszlosci.
WARIANTY_TYLKO_PODSTAWOWE = ["kNN", "Logistic Regression"]

plt.rcParams.update(STYL)

# ---------------------------------------------------------------- dane
dane = wczytaj(DATA_DIR, WZORZEC, [X_KOL, Y_KOL])

print("Znalezione modele:", sorted(dane["model"].unique()))

# 1) usun jawnie wskazane warianty
dane = dane[~dane["model"].isin(MODELE_POMIN)]

# 2) z rodzin wariantow zostaw tylko nazwe podstawowa
for przedrostek in WARIANTY_TYLKO_PODSTAWOWE:
    rodzina = dane["model"].str.startswith(przedrostek)
    dane = dane[~rodzina | (dane["model"] == przedrostek)]

# 3) zabezpieczenie przed zdublowanymi wierszami
klucz = ["transition", "model", X_KOL]
if MARKER_KOL in dane.columns:
    klucz.append(MARKER_KOL)
przed = len(dane)
dane = dane.drop_duplicates(subset=klucz, keep="first")
if len(dane) < przed:
    print(f"Usunieto {przed - len(dane)} zdublowanych wierszy.")

print("Modele po filtrowaniu:", sorted(dane["model"].unique()))

# brak kolumny kalibracji -> jeden ksztalt dla wszystkiego
if MARKER_KOL not in dane.columns:
    print(f"Brak kolumny '{MARKER_KOL}' - rysuje bez rozroznienia ksztaltem.")
    dane = dane.assign(_jeden=True)
    marker_kol, markery, opisy = "_jeden", {True: "o"}, {}
else:
    marker_kol, markery, opisy = MARKER_KOL, MARKERY, OPISY_MARKEROW

kolory = palety(dane)

# ---------------------------------------------------------------- rysunek
fig, axs = plt.subplots(1, 3, figsize=(7.4, 3.1), sharex=True, sharey=True)

modele, uzyte = set(), set()

for ax, (kod, nazwa) in zip(axs, TRANSITIONS.items()):
    m, u = rysuj_serie(ax, dane[dane["transition"] == kod],
                       X_KOL, Y_KOL, kolory, markery, marker_kol)
    modele |= m
    uzyte |= u

    ax.set_yscale("log")
    ax.set_title(nazwa, loc="left", pad=3)
    ax.set_xlabel(X_OPIS)
    os_dyskretna(ax, dane[X_KOL].unique(), obrot=0)

axs[0].invert_xaxis()          # 30 -> 12 cech, zgodnie z kolejnoscia redukcji
axs[0].set_ylabel(Y_OPIS)

legenda(fig, modele, kolory, {k: markery[k] for k in uzyte}, opisy, ncol=4)
zapisz(fig, "SUP_runtime_vs_features")

print("Narysowane:", sorted(modele))
