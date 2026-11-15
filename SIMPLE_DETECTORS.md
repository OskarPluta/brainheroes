# Proste detektory: sprawdzenie

Czy gotowe modele z półki radzą sobie równie dobrze jak reguła dwóch flag? Wszystko liczone
metodą leave-one-out na zapisanych tabelach cech; wiek i BMI nie są cechami. Modele z etykietami
porównujemy z tłem z 200 permutacji etykiet przepuszczonych przez tę samą procedurę.
Miejsce = pozycja w rankingu (1 = najbardziej podejrzany, remisy liczone na niekorzyść).

Cechy: Zbiór 1 – 12 bazowych, 32 rozszerzonych; Zbiór 2 – 13 bazowych, 25 rozszerzonych.

### Zbiór 1: pozycja leżąca (4 z 58)

| Rodzina | Cechy | Model | AUC | Tło z permutacji (średnia / 95. pct) | p | Miejsca osób wyróżnionych | Ile przejrzeć |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- | :---: |
| bez etykiet | bazowe | **Dwie flagi (raport)** | **0,926** | – | – | 38: 2, 48: 5, 49: 5, 60: 16 | 16 z 58 |
| bez etykiet | bazowe | Isolation Forest | 0,644 | – | – | 38: 11, 48: 27, 49: 31, 60: 18 | 31 z 58 |
| bez etykiet | bazowe | LOF | 0,731 | – | – | 38: 2, 48: 18, 49: 22, 60: 26 | 26 z 58 |
| bez etykiet | bazowe | One-Class SVM | 0,718 | – | – | 38: 3, 48: 19, 49: 28, 60: 21 | 28 z 58 |
| bez etykiet | bazowe | Mahalanobis | 0,713 | – | – | 38: 3, 48: 21, 49: 23, 60: 25 | 25 z 58 |
| bez etykiet | bazowe | Odległość kNN | 0,685 | – | – | 38: 4, 48: 20, 49: 24, 60: 30 | 30 z 58 |
| bez etykiet | bazowe | Błąd rekonstrukcji PCA | 0,75 | – | – | 38: 1, 48: 11, 49: 33, 60: 19 | 33 z 58 |
| z etykietami | bazowe | Regresja logistyczna | 0,593 | 0,445 / 0,732 | 0,249 | 38: 57, 48: 3, 49: 13, 60: 25 | 57 z 58 |
| z etykietami | bazowe | Las losowy | 0,639 | 0,429 / 0,764 | 0,159 | 38: 28, 48: 1, 49: 48, 60: 11 | 48 z 58 |
| z etykietami | bazowe | XGBoost | 0,731 | 0,472 / 0,782 | 0,1 | 38: 30, 48: 1, 49: 33, 60: 4 | 33 z 58 |
| bez etykiet | rozszerzone | Isolation Forest | 0,546 | – | – | 38: 13, 48: 34, 49: 25, 60: 36 | 36 z 58 |
| bez etykiet | rozszerzone | LOF | 0,602 | – | – | 38: 11, 48: 33, 49: 23, 60: 29 | 33 z 58 |
| bez etykiet | rozszerzone | One-Class SVM | 0,667 | – | – | 38: 11, 48: 19, 49: 20, 60: 32 | 32 z 58 |
| bez etykiet | rozszerzone | Mahalanobis | 0,528 | – | – | 38: 6, 48: 37, 49: 23, 60: 46 | 46 z 58 |
| bez etykiet | rozszerzone | Odległość kNN | 0,532 | – | – | 38: 11, 48: 31, 49: 26, 60: 43 | 43 z 58 |
| bez etykiet | rozszerzone | Błąd rekonstrukcji PCA | 0,731 | – | – | 38: 3, 48: 21, 49: 16, 60: 28 | 28 z 58 |
| z etykietami | rozszerzone | Regresja logistyczna | 0,625 | 0,432 / 0,728 | 0,174 | 38: 57, 48: 3, 49: 7, 60: 24 | 57 z 58 |
| z etykietami | rozszerzone | Las losowy | 0,741 | 0,429 / 0,728 | 0,05 | 38: 25, 48: 1, 49: 19, 60: 21 | 25 z 58 |
| z etykietami | rozszerzone | XGBoost | 0,792 | 0,469 / 0,774 | 0,05 | 38: 24, 48: 1, 49: 9, 60: 21 | 24 z 58 |

### Zbiór 2: DOT (3 z 17)

| Rodzina | Cechy | Model | AUC | Tło z permutacji (średnia / 95. pct) | p | Miejsca osób wyróżnionych | Ile przejrzeć |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- | :---: |
| bez etykiet | bazowe | **Dwie flagi (raport)** | **0,94** | – | – | 9: 4, 12: 3, 16: 2 | 4 z 17 |
| bez etykiet | bazowe | Isolation Forest | 0,571 | – | – | 9: 9, 12: 11, 16: 4 | 11 z 17 |
| bez etykiet | bazowe | LOF | 0,452 | – | – | 9: 15, 12: 9, 16: 5 | 15 z 17 |
| bez etykiet | bazowe | One-Class SVM | 0,69 | – | – | 9: 6, 12: 10, 16: 3 | 10 z 17 |
| bez etykiet | bazowe | Mahalanobis | 0,667 | – | – | 9: 8, 12: 9, 16: 3 | 9 z 17 |
| bez etykiet | bazowe | Odległość kNN | 0,452 | – | – | 9: 12, 12: 13, 16: 4 | 13 z 17 |
| bez etykiet | bazowe | Błąd rekonstrukcji PCA | 0,738 | – | – | 9: 5, 12: 10, 16: 2 | 10 z 17 |
| z etykietami | bazowe | Regresja logistyczna | 0,881 | 0,427 / 0,835 | 0,035 | 9: 6, 12: 1, 16: 4 | 6 z 17 |
| z etykietami | bazowe | Las losowy | 0,833 | 0,375 / 0,857 | 0,07 | 9: 4, 12: 8, 16: 1 | 8 z 17 |
| z etykietami | bazowe | XGBoost | 1,0 | 0,47 / 0,858 | 0,025 | 9: 2, 12: 3, 16: 1 | 3 z 17 |
| bez etykiet | rozszerzone | Isolation Forest | 0,738 | – | – | 9: 4, 12: 5, 16: 8 | 8 z 17 |
| bez etykiet | rozszerzone | LOF | 0,476 | – | – | 9: 7, 12: 16, 16: 5 | 16 z 17 |
| bez etykiet | rozszerzone | One-Class SVM | 0,738 | – | – | 9: 5, 12: 9, 16: 3 | 9 z 17 |
| bez etykiet | rozszerzone | Mahalanobis | 0,81 | – | – | 9: 5, 12: 7, 16: 2 | 7 z 17 |
| bez etykiet | rozszerzone | Odległość kNN | 0,69 | – | – | 9: 6, 12: 9, 16: 4 | 9 z 17 |
| bez etykiet | rozszerzone | Błąd rekonstrukcji PCA | 0,714 | – | – | 9: 6, 12: 10, 16: 2 | 10 z 17 |
| z etykietami | rozszerzone | Regresja logistyczna | 0,5 | 0,423 / 0,81 | 0,433 | 9: 14, 12: 7, 16: 6 | 14 z 17 |
| z etykietami | rozszerzone | Las losowy | 0,643 | 0,377 / 0,833 | 0,169 | 9: 5, 12: 12, 16: 4 | 12 z 17 |
| z etykietami | rozszerzone | XGBoost | 1,0 | 0,479 / 0,883 | 0,015 | 9: 2, 12: 3, 16: 1 | 3 z 17 |

Odtworzenie: `uv run --with xgboost python -m src.simple_detectors`
