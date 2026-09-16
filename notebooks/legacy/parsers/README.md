# Parsery danych surowych — materiały archiwalne

Notebooki zawierają historyczne parsery plików symulacyjnych i zapisują dane
do `FULL_30.npz` oraz `TRAIN_30.npz`. Każdy parser odpowiada innemu przejściu:

- `ab_data_parser.ipynb` — A–B;
- `b_cb_data_parser.ipynb` — B–C_b;
- `c_a_data_parser.ipynb` — C–A;
- `b_cb_vt_parser.ipynb` — pomocniczy parser/eksperyment dla B–C_b.

Główny pipeline zaczyna się od gotowych plików `.npz` i nie uruchamia tych
notebooków. Kopie zachowują kod źródłowy bez zmian; oryginały znajdują się w
`a_b/`, `b_cb/` i `c_a/parsers/`.

DANE MOZNA POBRAC Z :
https://cs.if.uj.edu.pl/jstud/CDT_ML_Data/
