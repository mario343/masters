# Wyniki eksperymentów

Ten katalog zawiera wybrany, historyczny snapshot wyników obliczeń. Pliki są
wynikami eksperymentów i nie są generowane automatycznie podczas instalacji
pakietu. Kod źródłowy znajduje się poziom wyżej, w `phase_transitions/` oraz
`scripts/`.

## Struktura

```text
Results/
├── SUPERVISED/
│   ├── AB/
│   ├── AC/
│   └── BCb/
├── UNSUPERVISED/ --> uczenie na skrajnych danych
│   ├── AB/
│   ├── AC/
│   └── BCb/
└── UNSUPERVISED_V2/ --> uczenie na całych danych
    ├── AB/
    ├── AC/
    └── BCb/
```

Konwencja nazw przejść:

- `AB` — przejście A–B, parametr `Delta`;
- `BCb` — przejście B–C_b, parametr `Delta`;
- `AC` — przejście C–A, parametr `K0`.

## Pliki wynikowe

W katalogach poszczególnych przejść występują zwykle trzy rodzaje plików:

- `*_summary.csv` — jedno podsumowanie konfiguracji, w tym punkt krytyczny,
  niepewność, rozmiary danych i czasy obliczeń;
- `*_curve_mean.csv` — średnie prawdopodobieństwo/obserwabla w kolejnych
  punktach parametru przejścia;
- `*_curve_chi.csv` — podatność `chi` oraz jej błąd jackknife.

Sufiks `_STD` oznacza wariant ze standaryzacją cech. Brak tego sufiksu oznacza
wariant bez standaryzacji. Pliki `*_KNN_COMPARISON_*` są historycznym
porównaniem wariantów implementacyjnych KNN i służą głównie do pomiaru czasu.

## `SUPERVISED`

Wyniki uczenia nadzorowanego na danych endpointowych. W `summary.csv` kolumna
`model` identyfikuje model, a `calibration` rozróżnia wariant skalibrowany i
nieskalibrowany, gdy dana konfiguracja je posiada.

Modele korzystają z krzywych średniego prawdopodobieństwa oraz podatności
`chi`. Punkt krytyczny znajduje się w kolumnie odpowiedniej dla przejścia,
np. `Delta_crit` albo `K0_crit`.

## `UNSUPERVISED`

Starsze wyniki analiz nienadzorowanych. Wariant ten zawiera między innymi
wyniki z różnymi wartościami `stride`; dlatego jeden model i jeden zestaw cech
może występować w wielu wierszach `summary.csv`.

## `UNSUPERVISED_V2`

Wyniki wariantu V2, w którym model jest dopasowywany na endpointach, a potem
wykorzystywany do obliczenia krzywej na pełnym zakresie parametru. W
`summary.csv` znajdują się dodatkowo informacje takie jak:

- `fit_scope` — zakres użyty do dopasowania;
- `endpoint_accuracy` — dokładność walidacji endpointów;
- `standardized` — informacja o standaryzacji;
- `N_train` i `N_full` — rozmiary zbioru treningowego i pełnego.

V1 (`full_range`) i V2 (`endpoint_validation`) są dwiema świadomie różnymi
procedurami i nie należy traktować ich plików jako prostych duplikatów.

## Reprodukcja

Do ponownego wykonania obliczeń należy używać skryptów z katalogu `scripts/`.
Istniejące pliki wynikowe traktować jako archiwalne: skrypty głównego
pipeline'u nie powinny ich nadpisywać. Przed porównaniem wyników trzeba
sprawdzić przejście, wariant standaryzacji, liczbę cech, model oraz `stride`.
