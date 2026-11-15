# Dwie flagi: deficyt i rozprzężenie w sygnale fNIRS

**Projekt:** Heroes of the Brain 2026, ścieżka *Breath, Brain & Body* (Cortivision fNIRS).
Wersja z wykresami: [`REPORT.html`](REPORT.html). Poprzedni wariant (średnia 50/50) jest w historii gita.

## Pomysł

Każda osoba dostaje dwie oceny, obie liczone w pełni ślepo (bez etykiet) w zagnieżdżonym LOOCV:

- **deficyt**: równoważona średnia kwadratowa jednostronnych odchyleń w stronę stłumionej odpowiedzi,
- **rozprzężenie**: błąd rekonstrukcji PCA (SPE), czyli cechy przestają do siebie pasować.

Alarm to **wyższa z dwóch ocen**, ponownie skalibrowana na ocenach poza próbą. Niczego nie uśredniamy
i nie ma wag do strojenia. Jedyny parametr to liczba składowych PCA (k = 3).

## Wyniki (k = 3, próg 90)

| Zbiór | ROC-AUC | Powyżej progu | Alarmy wśród pozostałych | Ile osób przejrzeć, by mieć wszystkie |
| :--- | :---: | :---: | :---: | :---: |
| 1: pozycja leżąca (4 z 58) | 0,926 | 3/4 | 2 z 54 | 16 |
| 2: DOT, pozycja siedząca (3 z 17) | 0,94 | 1/3 | 1 z 14 | 4 |

### Zbiór 1: osoby na lekach kardiologicznych

| ID | Deficyt | Rozprzężenie | Alarm | Miejsce |
| :--- | :---: | :---: | :---: | :---: |
| 38 | 43,9 | 100,0 | 100,0 | 2 |
| 48 | 96,5 | 84,2 | 93,0 | 5 |
| 49 | 96,5 | 38,6 | 93,0 | 5 |
| 60 | 87,7 | 73,7 | 75,4 | 16 |

Ten sam przepis na samych cechach tętna daje AUC 0,664, na samych cechach korowych
0,766, a na pełnym profilu 0,926.

ID 38 (29 lat, beta-bloker) w analizie ograniczonej do 55 osób w wieku do 42 lat:
rozprzężenie 100,0, deficyt 46,3, miejsce 2 (remis na szczycie).

### Zbiór 2: astma i BMI ≥ 30

| ID | Deficyt | Rozprzężenie | Alarm | Miejsce |
| :--- | :---: | :---: | :---: | :---: |
| 9 | 87,5 | 68,8 | 81,2 | 4 |
| 12 | 93,8 | 43,8 | 87,5 | 3 |
| 16 | 100,0 | 100,0 | 100,0 | 2 |

Sam deficyt rezonansu daje tu AUC 1,0; flaga rozprzężenia dokłada jedną osobę bez rozpoznania.

## Stabilność względem k

| k | Zbiór 1: AUC | Zbiór 1: trafieni, alarmy | Zbiór 2: AUC | Zbiór 2: trafieni, alarmy |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 0,944 | 3/4, 1 z 54 | 0,893 | 1/3, 1 z 14 |
| 2 | 0,935 | 3/4, 2 z 54 | 0,94 | 1/3, 1 z 14 |
| 3 | 0,926 | 3/4, 2 z 54 | 0,94 | 1/3, 1 z 14 |
| 4 | 0,91 | 3/4, 3 z 54 | 0,964 | 2/3, 1 z 14 |
| 5 | 0,891 | 2/4, 4 z 54 | 0,952 | 2/3, 1 z 14 |

## Zakres wniosków

- Grupy wyróżnione mają 4 i 3 osoby. To pokaz działania, nie pomiar skuteczności.
- W Zbiorze 1 trzy z czterech osób to jedyni uczestnicy 70+ i sam wiek daje AUC 0,975.
  Leczenia nie da się tu oddzielić od wieku; przemawiają za tym tylko ID 38 i Zbiór 2 (wszyscy 20–29 lat).
- Na górze listy są też osoby bez leków. Narzędzie wskazuje, komu warto się przyjrzeć, i niczego nie rozpoznaje.
- ID 60 nie przekracza progu; ID 41 i 59 (niepełne nagrania) są poza analizą.
- Cechy tętna pochodzą z kanałów fNIRS, a w Zbiorze 2 (18 Hz) nie oceniamy ich osobno.
- Wariant, w którym rozprzężenie uwzględnia też resztę regresji serce → kora: AUC
  0,921 (Zbiór 1) i
  0,821 (Zbiór 2).

## Odtworzenie

```
python -m src.two_flag_scoring       # results/two_flag_*.csv, two_flag_summary.json
python -m src.make_two_flag_report   # REPORT.html, REPORT.md
```
