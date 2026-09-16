# Notebooki

Notebooki w tym katalogu są cienkimi punktami wejścia do kodu z
`phase_transitions/`. Instrukcja instalacji, opis danych i kompletne komendy
znajdują się w głównym [README.md](../README.md).

- `supervised_endpoint.ipynb` — demonstracja uczenia supervised na endpointach;
- `unsupervised_methods.ipynb` — demonstracja metod unsupervised;
- `knn_timing.ipynb` — wyłącznie benchmark czasowy KNN;
- `inspect_refactored_results.ipynb` — odczyt istniejących plików CSV.

KNN nie jest traktowany jako osobna metoda naukowa wyznaczania przejścia.
Konfiguracje unsupervised, które nie przechodzą walidacji endpointów zgodnej z
artykułem, pozostają w macierzy diagnostycznej, ale nie powinny być używane do
generowania krzywych naukowych V2.

Notebooki w `a_b/`, `b_cb/`, `c_a/` oraz `Results/` są starszym materiałem
referencyjnym. Nie należy ich używać jako głównego entry pointu nowego kodu.

Kopie materiałów, które dokumentują dodatkowe analizy badawcze, znajdują się
w [`notebooks/legacy/`](legacy/):

- analiza korelacji, PCA i symbolic regression;
- zapisane wykresy i wyniki tych analiz;
- parsery danych surowych dla poszczególnych przejść.

Są to materiały archiwalne, a nie drugi równoległy pipeline. Ich opis i
dodatkowe zależności znajdują się w `notebooks/legacy/README.md`.
