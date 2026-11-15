# Dwie flagi: deficyt i rozprzężenie w sygnale fNIRS

**Projekt:** Heroes of the Brain 2026, ścieżka *Breath, Brain & Body* (Cortivision fNIRS).
Wersja z wykresami: [`REPORT.html`](REPORT.html). Poprzedni wariant (średnia 50/50) liczy `uv run brainheroes ablation`.

## Pomysł

Każda osoba dostaje dwie oceny, obie liczone w pełni ślepo (bez etykiet) w zagnieżdżonym LOOCV:

- **deficyt**: równoważona średnia kwadratowa jednostronnych odchyleń w stronę stłumionej odpowiedzi,
- **rozprzężenie**: błąd rekonstrukcji PCA (SPE), czyli cechy przestają do siebie pasować.

Alarm to **wyższa z dwóch ocen**, ponownie skalibrowana na ocenach poza próbą. Niczego nie uśredniamy
i nie ma wag do strojenia. Jedyny parametr to liczba składowych PCA (k = 3).

## Kontrola jakości sygnału (QC, bez użycia etykiet)

W Zbiorze 1 (Supine) regresja kanałów krótkich (`SSR`) oraz pomiar tętna wymagają co najmniej jednego
ważnego kanału krótkiego (`10 mm`) i długiego w każdej sesji (`BS` i `BH`):
- zapisano: 60 osób;
- niepełny protokół (`BS` lub `BH`): 2 osoby (ID 41, 59);
- odrzucone przez automatyczne QC kanałów krótkich (`SCI < 0,70` / nasycenie `> 4,90 V` / zanik `< 0,005 V`): 10 osób z grupy kontrolnej (ID 3, 4, 5, 6, 7, 8, 9, 10, 11, 35 — u ID 3–11 nasycenie toru `850 nm` w obu sesjach, u ID 35 zanik toru `760 nm` w `BS`);
- **włączono do głównej analizy Supine:** **48 osób** (bez utraty żadnego przypadku z Kryterium A ani B).

## Wyniki (k = 3, ujednolicone kryteria A/B oraz progi 90 i 85)

- **Kryterium A**: leki kardiologiczne/hipotensyjne lub przewlekłe choroby krążeniowo-oddechowe
  (Supine: 4/48 — beta-bloker i leki na nadciśnienie, bez izotretynoiny; DOT: 1/17 — astma).
- **Kryterium B**: Kryterium A lub otyłość (BMI ≥ 30) (Supine: 7/48, DOT: 3/17).

Wartości p (jednostronny test Manna-Whitneya) są opisowe i nie są korygowane na wielokrotne porównania
(4 wiersze × 2 progi). Dla grupy jednoosobowej przedział ufności nie ma sensu, więc go nie podajemy.

| Zbiór i kryterium | ROC-AUC (95% CI) | p (MW) | Próg 90 (~10%) | Próg 85 (~15%) | Ile osób przejrzeć, by mieć wszystkie |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 1: Supine — Kryt. A (4 z 48) | 0,892 [0,744; 0,994] | 0,0053 | 2/4 (FP: 3) | 3/4 (FP: 4) | 16 |
| 1: Supine — Kryt. B (7 z 48) | 0,833 [0,684; 0,957] | 0,0027 | 3/7 (FP: 2) | 4/7 (FP: 3) | 24 |
| 2: DOT — Kryt. A (1 z 17) | 0,906 (CI: n/d, 1 osoba) | 0,1098 | 1/1 (FP: 2) | 1/1 (FP: 2) | 3 |
| 2: DOT — Kryt. B (3 z 17) | 0,952 [0,827; 1,0] | 0,0097 | 2/3 (FP: 1) | 2/3 (FP: 1) | 5 |

### Zbiór 1: osoby na lekach kardiologicznych (Kryterium A)

| ID | Deficyt | Rozprzężenie | Alarm | Miejsce |
| :--- | :---: | :---: | :---: | :---: |
| 38 | 29,8 | 97,9 | 95,7 | 4 |
| 48 | 93,6 | 68,1 | 87,2 | 7 |
| 49 | 97,9 | 34,0 | 95,7 | 4 |
| 60 | 85,1 | 4,3 | 72,3 | 16 |

Ten sam przepis na samych cechach tętna daje AUC 0,713, na samych cechach korowych
0,736, a na pełnym profilu 0,892.

ID 38 (29 lat, beta-bloker) w analizie ograniczonej do 45 osób w wieku do 42 lat:
rozprzężenie 97,7, deficyt 29,5, alarm 95,5, miejsce 4 (remis: 2 osoby z tym samym wynikiem).

### Zbiór 2: astma i BMI ≥ 30 (Kryterium B)

| ID | Deficyt | Rozprzężenie | Alarm | Miejsce |
| :--- | :---: | :---: | :---: | :---: |
| 9 | 87,5 | 68,8 | 81,2 | 5 |
| 12 | 93,8 | 62,5 | 93,8 | 3 |
| 16 | 100,0 | 100,0 | 100,0 | 1 |

Sam deficyt rezonansu daje tu AUC 1,0; flaga rozprzężenia dokłada jedną osobę bez rozpoznania.

## Stabilność względem k

| k | Zbiór 1 (Kryt. A): AUC | Zbiór 1: trafieni, alarmy (próg 90) | Zbiór 2 (Kryt. B): AUC | Zbiór 2: trafieni, alarmy (próg 90) |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 0,898 | 2/4, 3 z 44 | 0,976 | 2/3, 0 z 14 |
| 2 | 0,906 | 2/4, 2 z 44 | 0,952 | 2/3, 1 z 14 |
| 3 | 0,892 | 2/4, 3 z 44 | 0,952 | 2/3, 1 z 14 |
| 4 | 0,892 | 2/4, 3 z 44 | 0,964 | 2/3, 0 z 14 |
| 5 | 0,881 | 1/4, 4 z 44 | 0,94 | 2/3, 1 z 14 |

## Zakres wniosków

- Grupy wyróżnione mają 4 i 3 osoby (w DOT Kryt. A to 1 osoba). To pokaz działania, nie pomiar skuteczności.
- W Zbiorze 1 trzy z czterech osób z Kryt. A to jedyni uczestnicy 70+ i sam wiek daje AUC 0,983.
  Leczenia nie da się tu oddzielić od wieku; przemawiają za tym tylko ID 38 i Zbiór 2 (wszyscy 20–29 lat).
- BMI ≥ 30 w obu zbiorach to tylko kilka osób, więc nie wyciągamy wniosków o jego wpływie. W Supine dodanie
  trzech młodych osób z otyłością zmienia AUC do 0,833 (ID 2 łapie flaga rozprzężenia, ale ID 51 i 55
  nie są wykrywane). W DOT kryteria A i B różnią się o 2 osoby (1 vs 3); przy tak małych grupach różnicy AUC nie interpretujemy.
- Na górze listy są też osoby bez leków. Narzędzie wskazuje, komu warto się przyjrzeć, i niczego nie rozpoznaje.
- W Supine ID 48 jest tuż pod progiem 90 (87,2 — przekracza próg 85), a ID 60 nie przekracza progów 90/85; poza analizą są ID 41, 59 (niepełne nagrania) oraz ID 3, 4, 5, 6, 7, 8, 9, 10, 11, 35 (brak ważnych kanałów krótkich wg QC).
- Cechy tętna pochodzą z kanałów fNIRS (nie z osobnego pulsoksymetru), a w Zbiorze 2 (18 Hz) nie oceniamy ich osobno.
- Wariant, w którym rozprzężenie uwzględnia też resztę regresji serce → kora: AUC
  0,892 (Zbiór 1) i
  0,845 (Zbiór 2).

## Odtworzenie

```
uv run brainheroes score    # results/two_flag_*.csv, two_flag_summary.json
uv run brainheroes report   # docs/REPORT.html, docs/REPORT.md
```
