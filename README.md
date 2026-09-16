# Analiza przejść fazowych metodami machine learning

Repozytorium zawiera pipeline'y do analizy przejść fazowych z użyciem uczenia
nadzorowanego, klasteryzacji, jackknife oraz histogram reweightingu.

Implementacja naukowa znajduje się w pakiecie `phase_transitions/`. Notebooki
są tylko cienkimi punktami wejścia albo materiałem historycznym. Wyniki CSV
nie są nadpisywane przez skrypty pipeline'u.

## 1. Wymagania

Zalecane środowisko:

- Python 3.11;
- NumPy 1.25.x;
- pandas 2.2.x;
- scikit-learn 1.4.x;
- pytest 8.x do testów;
- JupyterLab, ipykernel i Matplotlib tylko do notebooków.

SciPy jest instalowane tranzytywnie przez scikit-learn i nie trzeba dodawać go
osobno do komendy instalacyjnej.

## 2. Instalacja

Z katalogu repozytorium:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev,notebooks]"
```

Jeżeli notebooki nie są potrzebne, wystarczy:

```bash
python -m pip install -e ".[dev]"
```

Instalacja editable (`-e`) sprawia, że `phase_transitions` jest dostępne jako
pakiet, a zmiany w kodzie są widoczne bez ponownej instalacji.

Sprawdzenie środowiska:

```bash
python -c "import numpy, pandas, sklearn, phase_transitions; print('environment OK')"
python -m pytest -q
```

## 3. Dane wejściowe

Pipeline korzysta z przetworzonych archiwów NumPy:

```text
a_b/data/processed/TRAIN_30.npz
a_b/data/processed/FULL_30.npz
b_cb/data/processed/TRAIN_30.npz
b_cb/data/processed/FULL_30.npz
c_a/data/processed/TRAIN_30.npz
c_a/data/processed/FULL_30.npz
```

`TRAIN_30.npz` musi zawierać `X`, parametr przejścia oraz etykiety `y`.
`FULL_30.npz` musi zawierać `X` i parametr przejścia. `X` ma 30 kolumn.

Nazwy parametrów i lokalizacje danych są zdefiniowane w
[`phase_transitions/config.py`](phase_transitions/config.py):

- AB: parametr `Delta`;
- B–Cb: parametr `Delta`;
- C–A: parametr `K0`.

Notebooki parserów w `a_b/` i `b_cb/` są historycznymi narzędziami do
przygotowania danych. Kanoniczne skrypty wymagają obecności plików
`data/processed/*.npz` i nie generują ponownie danych surowych.

## 4. Struktura kodu

```text
phase_transitions/
  config.py          konfiguracja przejść i zestawów cech
  data.py            bezpieczne ładowanie TRAIN/FULL
  preprocessing.py   wybór cech i standaryzacja
  models.py          fabryki modeli supervised i clustererów
  validation.py      deterministyczny podział endpointów
  supervised.py      główny pipeline uczenia nadzorowanego
  unsupervised.py    V2 endpoint oraz V1 full_range
  statistics.py      wariancja, jackknife, crossing, peak
  reweighting.py     statystyki histogram reweightingu
  reporting.py       zapis summary/curve do CSV
  knn_timing.py      benchmark czasowy KNN

scripts/
  run_endpoint.py                pojedynczy eksperyment
  run_matrix.py                  macierz eksperymentów
  endpoint_validation_matrix.py  szybka macierz walidacji endpointów
  run_validated_unsupervised.py  krzywe tylko dla zaliczonych konfiguracji

tests/                            testy kontraktów i regresji
notebooks/                        notebooki demonstracyjne i archiwalne analizy
Results/REFactored/               wyniki nowego pipeline'u
```

## 5. Uruchamianie pojedynczego eksperymentu

### Uczenie nadzorowane — endpoint validation

```bash
python scripts/run_endpoint.py \
  --transition AB \
  --kind supervised \
  --method endpoint_validation \
  --model logistic_regression \
  --feature-set 30 \
  --standardize \
  --output-dir Results/REFactored/manual/supervised/AB
```

Dostępne modele supervised:

```text
logistic_regression  decision_tree  random_forest
gradient_boosting    knn            svm_rbf
mlp
```

SVM i MLP używają kalibracji, aby udostępnić `predict_proba`. W konfiguracji
kanonicznej `CalibratedClassifierCV` ma `cv=5`.

### Uczenie nienadzorowane na endpointach — V2

```bash
python scripts/run_endpoint.py \
  --transition AB \
  --kind unsupervised \
  --method endpoint_validation \
  --model kmeans \
  --feature-set 30 \
  --standardize \
  --output-dir Results/REFactored/manual/unsupervised/AB
```

Krzywa naukowa jest liczona tylko wtedy, gdy walidacja endpointów spełnia
próg z `TransitionConfig` (`0.999`). Do samego audytu wszystkich konfiguracji
służy `endpoint_validation_matrix.py`.

### Uczenie na pełnym zakresie — V1

```bash
python scripts/run_endpoint.py \
  --transition AB \
  --kind unsupervised \
  --method full_range \
  --model kmeans \
  --feature-set 30 \
  --standardize \
  --output-dir Results/REFactored/manual/full_range/AB
```

V1 jest odrębną, dodatkową metodą: klaster jest dopasowywany do pełnego
zakresu danych. Ten tryb nie wykonuje walidacji endpointów, dlatego w jego
summary `validation_accuracy` pozostaje `NaN`.

Nie uruchamiać Birch na pełnym zbiorze ze stride 1 bez osobnej decyzji
metodologicznej. W aktualnych danych może utworzyć bardzo dużą liczbę
subklastrów i zażądać setek gigabajtów pamięci.

## 6. Macierze eksperymentów

Przykład macierzy supervised:

```bash
python scripts/run_matrix.py \
  --kind supervised \
  --method endpoint_validation \
  --transitions AB BCB CA \
  --models logistic_regression random_forest svm_rbf mlp \
  --feature-sets 30 20 12 \
  --standardize \
  --output-root Results/REFactored/supervised
```

Szybka macierz walidacji endpointów unsupervised:

```bash
python scripts/endpoint_validation_matrix.py \
  --transitions AB BCB CA \
  --models kmeans gaussian_mixture bayesian_gmm meanshift birch \
  --output Results/REFactored/endpoint_validation_matrix.csv
```

Następnie można policzyć krzywe tylko dla konfiguracji, które przeszły
walidację:

```bash
python scripts/run_validated_unsupervised.py \
  --matrix Results/REFactored/endpoint_validation_matrix.csv \
  --output-root Results/REFactored/unsupervised
```

Macierze mogą działać długo. Skrypty wypisują numer konfiguracji, status,
czas i ścieżkę zapisanego pliku. Istniejące pliki wynikowe nie są nadpisywane.

## 7. KNN

KNN nie jest dodatkową metodą naukową wyznaczania przejścia. Moduł
`phase_transitions/knn_timing.py` zachowuje wyłącznie benchmark czasowy
historycznych wariantów algorytmu.

Benchmark uruchamia się z notebooka:

```bash
jupyter lab notebooks/knn_timing.ipynb
```

W notebooku należy najpierw sprawdzić stride'y i ustawić `RUN = True`.

## 8. Standaryzacja, walidacja i statystyki

- Standaryzator jest dopasowywany wyłącznie na danych użytych do fitowania
  modelu. Dane walidacyjne i pełne są tylko transformowane tym samym obiektem.
- W supervised i V2 endpoint model uczy się na endpointach, a walidacja jest
  deterministycznym podziałem blokowym 80/20, bez losowego przeplatania
  skorelowanych próbek.
- Dla V1 `full_range` standaryzacja całego zakresu jest celowa, bo cały zakres
  jest zbiorem fitowania tej metody.
- Podatność to wariancja populacyjna
  `mean(P**2) - mean(P)**2`, czyli odpowiednik `np.var(..., ddof=0)`.
- `ddof=1` jest używane tylko do wariancji replik jackknife i wyznaczenia
  błędu standardowego.

## 9. Histogram reweighting

Podstawowe, wspólne funkcje reweightingu znajdują się w
`phase_transitions/reweighting.py`. Główny historyczny pipeline reweightingu
pozostaje w notebooku:

```bash
jupyter lab Pipeline_reweighting_supervised__8_.ipynb
```

Notebook korzysta z plików prawdopodobieństw w
`probabilities_for_reweighting/`. Parametr `AUTO_SCALER=False` pozostaje
świadomą częścią dotychczasowej logiki tego pipeline'u.

## 10. Testy i kontrola zmian

Po każdej zmianie kodu uruchomić:

```bash
python -m pytest -q
python -m compileall -q phase_transitions scripts
git diff --check
```

Testy nie uruchamiają kosztownych pełnych modeli; sprawdzają kontrakty danych,
statystyki, standaryzację, walidację, zapis wyników i benchmark KNN.

Przed usunięciem starego kodu należy zachować jego wyniki CSV i porównać
przynajmniej punkt przecięcia, maksimum `chi`, rozmiary zbiorów oraz kształt
krzywych dla reprezentatywnych konfiguracji.

## 11. Czysty ZIP z kodem

Do spakowania nowego kodu bez starych notebooków, danych, duplikatów wyników
i cache użyj:

```bash
python scripts/package_source.py
```

Powstanie `phase-transitions-ml-source.zip` zawierający tylko:

- `README.md`, `pyproject.toml` i `.gitignore`;
- `phase_transitions/`;
- `scripts/`;
- `tests/`;
- notebooki demonstracyjne z `notebooks/` oraz archiwalne materiały z
  `notebooks/legacy/` dokumentujące korelacje, PCA i parsery.

Skrypt odmawia nadpisania istniejącego ZIP-a. Stare notebooki, katalogi
`Results/`, dane `data/`, pliki `probabilities_for_reweighting/` i historyczne
skrypty nie są do archiwum dodawane i można je przenieść poza repozytorium.

## 12. Porównanie kalibracji na krzywych `chi`

Do analizy starych wyników supervised uruchom:

```bash
python scripts/compare_calibration_chi.py
```

Skrypt nie uczy modeli ponownie. Czyta istniejące pliki
`Results/SUPERVISED/{AB,AC,BCb}/*_curve_chi.csv` i zapisuje do
`Results/figures_calibration_comparison_refactored/`:

- `chi_calibration_curves.pdf` — krzywe `chi` bez kalibracji i z kalibracją;
- `chi_calibration_peak_comparison.csv` — położenia maksimum, przesunięcia,
  amplitudy i ich stosunek;
- `chi_calibration_summary.png/pdf` — mapa stosunku amplitud oraz przesunięcia
  maksimum.

Na mapie amplitud `log10(amplituda kalibrowana / amplituda niekalibrowana)`:
wartości dodatnie oznaczają wzrost podatności po kalibracji, a ujemne jej
spadek. Druga mapa pokazuje bezpośrednio przesunięcie położenia maksimum.
