# Archiwalne materiały badawcze

Ten katalog zawiera kopie notebooków i artefaktów z wcześniejszych etapów
badania. Są zachowane jako dokumentacja wykonanych analiz, ale nie są częścią
kanonicznego pipeline'u z katalogu `phase_transitions/`.

## Zawartość

- `feature_analysis/correlation/notebooks/` — korelacje, PCA, sieci korelacji
  oraz symbolic regression (`gplearn`, `PySR`, `aifeynman`);
- `feature_analysis/correlation/results/` — zapisane wykresy i wyniki tych
  analiz;
- `parsers/` — parsery surowych plików dla przejść A–B, B–C_b i C–A.

Oryginalne notebooki pozostają również w swoich dotychczasowych katalogach
(`a_b/`, `b_cb/`, `c_a/`). Kopie są przeznaczone do archiwum i zachowują
oryginalny kod bez zmian. Część ścieżek względnych w notebookach odnosi się do
starego układu katalogów, dlatego do ponownego uruchamiania należy użyć
oryginałów albo ręcznie ustawić katalog roboczy i ścieżki danych.

## Dodatkowe zależności

Podstawowa instalacja projektu nie instaluje wszystkich zależności tych
notebooków. W zależności od analizy potrzebne są między innymi:

```text
scipy, networkx, sympy, gplearn, pysr, aifeynman
```

`PySR` wymaga dodatkowo środowiska Julia. Wyniki zapisane w `results/` można
oglądać bez instalowania tych pakietów.
