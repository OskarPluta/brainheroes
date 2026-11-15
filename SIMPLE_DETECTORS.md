# Proste detektory: sprawdzenie

Czy gotowe modele z półki radzą sobie równie dobrze jak reguła dwóch flag? Wszystko liczone
metodą leave-one-out na zapisanych tabelach cech (w Zbiorze 1 po ślepym QC kanałów krótkich:
N = 48 z 60 zapisanych); wiek i BMI nie są cechami. Modele z etykietami
porównujemy z tłem z 200 permutacji etykiet przepuszczonych przez tę samą procedurę.
Miejsce = pozycja w rankingu (1 = najbardziej podejrzany, remisy liczone na niekorzyść).

Cechy: Zbiór 1 – 12 bazowych, 32 rozszerzonych; Zbiór 2 – 13 bazowych, 25 rozszerzonych.

### Zbiór 1: pozycja leżąca (4 z 48)

| Rodzina | Cechy | Model | AUC | Tło z permutacji (średnia / 95. pct) | p | Miejsca osób wyróżnionych | Ile przejrzeć |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- | :---: |
| bez etykiet | bazowe | **Dwie flagi (raport)** | **0,892** | – | – | 38: 4, 48: 7, 49: 4, 60: 16 | 16 z 48 |
| bez etykiet | bazowe | Isolation Forest | 0,585 | – | – | 38: 6, 48: 30, 49: 28, 60: 19 | 30 z 48 |
| bez etykiet | bazowe | LOF | 0,773 | – | – | 38: 2, 48: 17, 49: 9, 60: 22 | 22 z 48 |
| bez etykiet | bazowe | One-Class SVM | 0,705 | – | – | 38: 2, 48: 17, 49: 23, 60: 20 | 23 z 48 |
| bez etykiet | bazowe | Mahalanobis | 0,727 | – | – | 38: 2, 48: 17, 49: 14, 60: 25 | 25 z 48 |
| bez etykiet | bazowe | Odległość kNN | 0,636 | – | – | 38: 2, 48: 19, 49: 22, 60: 31 | 31 z 48 |
| bez etykiet | bazowe | Błąd rekonstrukcji PCA | 0,517 | – | – | 38: 2, 48: 14, 49: 33, 60: 46 | 46 z 48 |
| z etykietami | bazowe | Regresja logistyczna | 0,545 | 0,436 / 0,768 | 0,308 | 38: 48, 48: 3, 49: 12, 60: 27 | 48 z 48 |
| z etykietami | bazowe | Las losowy | 0,534 | 0,412 / 0,716 | 0,289 | 38: 34, 48: 3, 49: 47, 60: 8 | 47 z 48 |
| z etykietami | bazowe | XGBoost | 0,665 | 0,465 / 0,774 | 0,209 | 38: 25, 48: 1, 49: 35, 60: 8 | 35 z 48 |
| bez etykiet | rozszerzone | Isolation Forest | 0,67 | – | – | 38: 4, 48: 24, 49: 18, 60: 22 | 24 z 48 |
| bez etykiet | rozszerzone | LOF | 0,705 | – | – | 38: 2, 48: 19, 49: 12, 60: 29 | 29 z 48 |
| bez etykiet | rozszerzone | One-Class SVM | 0,727 | – | – | 38: 3, 48: 18, 49: 15, 60: 22 | 22 z 48 |
| bez etykiet | rozszerzone | Mahalanobis | 0,636 | – | – | 38: 2, 48: 25, 49: 11, 60: 36 | 36 z 48 |
| bez etykiet | rozszerzone | Odległość kNN | 0,58 | – | – | 38: 2, 48: 25, 49: 21, 60: 36 | 36 z 48 |
| bez etykiet | rozszerzone | Błąd rekonstrukcji PCA | 0,722 | – | – | 38: 3, 48: 9, 49: 14, 60: 33 | 33 z 48 |
| z etykietami | rozszerzone | Regresja logistyczna | 0,682 | 0,427 / 0,773 | 0,119 | 38: 48, 48: 2, 49: 6, 60: 10 | 48 z 48 |
| z etykietami | rozszerzone | Las losowy | 0,71 | 0,4 / 0,722 | 0,075 | 38: 26, 48: 1, 49: 19, 60: 15 | 26 z 48 |
| z etykietami | rozszerzone | XGBoost | 0,682 | 0,471 / 0,78 | 0,159 | 38: 26, 48: 1, 49: 12, 60: 27 | 27 z 48 |

### Zbiór 2: DOT (3 z 17)

| Rodzina | Cechy | Model | AUC | Tło z permutacji (średnia / 95. pct) | p | Miejsca osób wyróżnionych | Ile przejrzeć |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- | :---: |
| bez etykiet | bazowe | **Dwie flagi (raport)** | **0,952** | – | – | 9: 5, 12: 3, 16: 1 | 5 z 17 |
| bez etykiet | bazowe | Isolation Forest | 0,69 | – | – | 9: 7, 12: 10, 16: 2 | 10 z 17 |
| bez etykiet | bazowe | LOF | 0,619 | – | – | 9: 8, 12: 11, 16: 3 | 11 z 17 |
| bez etykiet | bazowe | One-Class SVM | 0,762 | – | – | 9: 5, 12: 9, 16: 2 | 9 z 17 |
| bez etykiet | bazowe | Mahalanobis | 0,762 | – | – | 9: 5, 12: 9, 16: 2 | 9 z 17 |
| bez etykiet | bazowe | Odległość kNN | 0,595 | – | – | 9: 9, 12: 11, 16: 3 | 11 z 17 |
| bez etykiet | bazowe | Błąd rekonstrukcji PCA | 0,81 | – | – | 9: 5, 12: 8, 16: 1 | 8 z 17 |
| z etykietami | bazowe | Regresja logistyczna | 0,667 | 0,433 / 0,857 | 0,204 | 9: 9, 12: 1, 16: 10 | 10 z 17 |
| z etykietami | bazowe | Las losowy | 0,857 | 0,368 / 0,81 | 0,045 | 9: 4, 12: 5, 16: 3 | 5 z 17 |
| z etykietami | bazowe | XGBoost | 1,0 | 0,491 / 0,857 | 0,025 | 9: 2, 12: 3, 16: 1 | 3 z 17 |
| bez etykiet | rozszerzone | Isolation Forest | 0,69 | – | – | 9: 4, 12: 7, 16: 8 | 8 z 17 |
| bez etykiet | rozszerzone | LOF | 0,5 | – | – | 9: 7, 12: 16, 16: 4 | 16 z 17 |
| bez etykiet | rozszerzone | One-Class SVM | 0,762 | – | – | 9: 5, 12: 8, 16: 3 | 8 z 17 |
| bez etykiet | rozszerzone | Mahalanobis | 0,81 | – | – | 9: 5, 12: 7, 16: 2 | 7 z 17 |
| bez etykiet | rozszerzone | Odległość kNN | 0,69 | – | – | 9: 7, 12: 9, 16: 3 | 9 z 17 |
| bez etykiet | rozszerzone | Błąd rekonstrukcji PCA | 0,738 | – | – | 9: 7, 12: 8, 16: 2 | 8 z 17 |
| z etykietami | rozszerzone | Regresja logistyczna | 0,476 | 0,425 / 0,81 | 0,453 | 9: 14, 12: 8, 16: 6 | 14 z 17 |
| z etykietami | rozszerzone | Las losowy | 0,619 | 0,382 / 0,833 | 0,229 | 9: 7, 12: 11, 16: 4 | 11 z 17 |
| z etykietami | rozszerzone | XGBoost | 1,0 | 0,486 / 0,882 | 0,015 | 9: 2, 12: 3, 16: 1 | 3 z 17 |

Odtworzenie: `uv run --with xgboost python -m src.simple_detectors`
