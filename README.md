# Heroes of the Brain 2026 — Dwie flagi w sygnale fNIRS

Nienadzorowany system wykrywania anomalii fizjologicznych (*Normative Unsupervised Anomaly Detection*) w wielokanałowych sygnałach fNIRS (Cortivision Spectrum C23) w ścieżce *Breath, Brain & Body*.

## Główna idea (`Dwie flagi: max(deficyt, rozprzężenie)`)

Każdy uczestnik otrzymuje dwie oceny w pełni ślepe na etykiety kliniczne, wyznaczane w zagnieżdżonym schemacie Leave-One-Out (`LOOCV`):

1. **Deficyt (`deficit_pct`)** — równomierna średnia kwadratowa (RMS) jednostronnych odchyleń w stronę stłumionej odpowiedzi naczyniowej i autonomicznej.
2. **Rozprzężenie (`decoupling_pct`)** — błąd rekonstrukcji PCA (`SPE` / $Q$-residual, $k=3$), wykrywający przypadki, w których pojedyncze cechy mieszczą się w normie, ale pękają relacje między układem krążenia a korą mózgową (np. `ID 38` na beta-blokerze *Nebilet*).

Ostateczny alarm to **wyższa z dwóch ocen** (`max(deficit_pct, decoupling_pct)`), ponownie skalibrowana względem rozkładu poza próbą — bez ręcznego strojenia wag.

## Główne wyniki

| Zbiór i kryterium | Próba | ROC-AUC (95% CI) | p (MW) | Próg 90 | Próg 85 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Zbiór 1: Supine — Kryt. A (leki kardiologiczne) | 4 z 48* | 0,892 [0,744; 0,994] | 0,0053 | 2/4 (FP: 3) | 3/4 (FP: 4) |
| Zbiór 1: Supine — Kryt. B (+ BMI ≥ 30) | 7 z 48* | 0,833 [0,684; 0,957] | 0,0027 | 3/7 (FP: 2) | 4/7 (FP: 3) |
| Zbiór 2: DOT Sitting — Kryt. A (astma) | 1 z 17 | 0,906 (n = 1) | 0,1098 | 1/1 (FP: 2) | 1/1 (FP: 2) |
| Zbiór 2: DOT Sitting — Kryt. B (astma + BMI ≥ 30) | 3 z 17 | 0,952 [0,827; 1,0] | 0,0097 | 2/3 (FP: 1) | 2/3 (FP: 1) |

\* Z 60 zapisanych osób w Supine 2 mają niepełny protokół (`ID 41, 59`), a 10 kontroli odpada w ślepym QC kanałów krótkich (`ID 3–11` nasycenie `850 nm`, `ID 35` zanik `760 nm`), dając czystą kohortę `N = 48` bez utraty przypadków z Kryterium A/B.

## Dokumentacja i raporty

- [`REPORT.html`](REPORT.html) — interaktywny raport z wykresami SVG dla obu kohort (główna metoda dwóch flag).
- [`REPORT.md`](REPORT.md) — wersja tekstowa raportu z tabelami wyników i zakresem wniosków.
- [`METHODOLOGY_ANOMALY_DETECTION.md`](METHODOLOGY_ANOMALY_DETECTION.md) — szczegółowy opis fizyki sygnału (QC, `TDDR`, `MBLL`, `SSR`) i matematyki modelu normatywnego.
- [`SIMPLE_DETECTORS.md`](SIMPLE_DETECTORS.md) — porównanie reguły dwóch flag z gotowymi detektorami bez nadzoru oraz klasyfikatorami z nadzorem (z testem permutacyjnym).

## Struktura kodu i odtworzenie wyników

- [`src/fnirs_preprocessing.py`](src/fnirs_preprocessing.py) — wczytywanie `.snirf`, ślepe QC (`compute_windowed_sci`), `tddr_repair`, `od_to_hb_mbll`, `ssr_clean`.
- [`src/feature_extraction.py`](src/feature_extraction.py) — ekstrakcja cech dla kohorty Supine (12 cech) i DOT Sitting (13 cech).
- [`src/normative_modeling.py`](src/normative_modeling.py) — zagnieżdżone `LOOCV`, `RobustScaler`, PCA (`T²`, `SPE`), Cross-Modal Ridge i formuły deficytu.
- [`src/two_flag_scoring.py`](src/two_flag_scoring.py) — główna metoda dwóch flag (`results/two_flag_*.csv`, `results/two_flag_summary.json`).
- [`src/make_two_flag_report.py`](src/make_two_flag_report.py) — generator raportów `REPORT.html` i `REPORT.md`.
- [`src/simple_detectors.py`](src/simple_detectors.py) — benchmark prostych detektorów (`SIMPLE_DETECTORS.md`).
- [`src/run_experiments.py`](src/run_experiments.py) — pełna ekstrakcja cech i ablacje wariantu 50/50 (`results/verification_summary.json`).

```bash
uv run python -m src.two_flag_scoring                     # wyniki dwóch flag -> results/two_flag_*
uv run python -m src.make_two_flag_report                 # raport -> REPORT.html, REPORT.md
uv run --with xgboost python -m src.simple_detectors      # benchmark prostych detektorów -> SIMPLE_DETECTORS.md
uv run pytest -v                                          # testy jednostkowe i anty-leakage
```
