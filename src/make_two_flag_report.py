"""Builds REPORT.html (standalone page with charts) and REPORT.md from results/two_flag_*.

Usage:
    python -m src.two_flag_scoring        # regenerate the numbers
    python -m src.make_two_flag_report    # regenerate the report
    python -m src.make_two_flag_report --fragment out.html   # body-only variant for embedding
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.two_flag_scoring import DOT_POSITIVE_IDS, SUPINE_POSITIVE_IDS

FEATURE_LABELS = {
    "bs_hr_bpm": "tętno spoczynkowe",
    "bs_sdnn_ms": "zmienność tętna (SDNN)",
    "bs_rmssd_ms": "zmienność tętna (RMSSD)",
    "bs_lr_mayer_corr": "spójność półkul",
    "bs_mayer_amp": "amplituda fal Mayera",
    "bs_mayer_resp_ratio": "fale Mayera względem oddechowych",
    "bh_dur_max": "czas bezdechu",
    "bh_dur_est": "czas bezdechu (szac.)",
    "bh_ssr_cortical_od850": "odpowiedź kory na bezdech",
    "bh_deep_od850": "odpowiedź kanałów długich na bezdech",
    "bh_hbr_hbo_ratio": "stosunek HbR/HbO przy bezdechu",
    "bh_spatial_cv": "nierównomierność odpowiedzi",
    "bh_hr_delta": "przyspieszenie tętna przy bezdechu",
    "bh_evening_morning_ratio": "bezdech: sesja 2 względem 1",
    "rs_hr_bpm": "tętno spoczynkowe",
    "rs_sdnn_ms": "zmienność tętna (SDNN)",
    "rs_rmssd_ms": "zmienność tętna (RMSSD)",
    "spb1_gain_01hz": "wzmocnienie 0,1 Hz (sesja 1)",
    "spb_gain_med": "wzmocnienie 0,1 Hz (mediana)",
    "spb_ssr_gain_med": "wzmocnienie 0,1 Hz w korze",
    "spb_lr_coherence": "spójność półkul",
    "spb_q_factor": "ostrość rezonansu",
}
MEDICATION_LABELS = {"nebilet": "beta-bloker", "od nadcisnienia": "leki na nadciśnienie", "izotec": "lek dermatologiczny"}
DISEASE_LABELS = {"Astma": "astma", "Chorby jelitowe": "choroba jelit"}


def _why(row: pd.Series, top: int = 3) -> list[list[object]]:
    z = {c[2:]: float(row[c]) for c in row.index if c.startswith("z_")}
    order = sorted(z, key=lambda c: abs(z[c]), reverse=True)[:top]
    return [[FEATURE_LABELS.get(c, c), round(z[c], 1)] for c in order]


def _clean(value: object, labels: dict[str, str]) -> str:
    text = str(value).strip()
    return "" if text in {"-", "nan", ""} else labels.get(text, text)


def build_payload(res: Path) -> dict[str, object]:
    sup = pd.read_csv(res / "two_flag_supine.csv")
    dot = pd.read_csv(res / "two_flag_dot.csv")
    summary = json.loads((res / "two_flag_summary.json").read_text(encoding="utf-8"))

    def base(r: pd.Series, positives: list[int]) -> dict[str, object]:
        return {
            "id": int(r.ID),
            "def": round(float(r.deficit_pct), 1),
            "dec": round(float(r.decoupling_pct), 1),
            "alarm": round(float(r.alarm_pct), 1),
            "rank": int(r["rank"]),
            "flag": bool(r.alarm),
            "pos": int(r.ID) in positives,
            "why": _why(r),
        }

    sup_rows = [
        {**base(r, SUPINE_POSITIVE_IDS), "age": int(r.Age), "med": _clean(r.Medication, MEDICATION_LABELS)}
        for _, r in sup.iterrows()
    ]
    dot_rows = [
        {
            **base(r, DOT_POSITIVE_IDS),
            "bmi": round(float(r.BMI), 1),
            "dis": _clean(r.Chronic_Disease, DISEASE_LABELS),
            "g1": round(float(r.spb1_gain_01hz), 2),
            "gm": round(float(r.spb_gain_med), 2),
            "gs": round(float(r.spb_ssr_gain_med), 2),
        }
        for _, r in dot.iterrows()
    ]
    return {"sup": sup_rows, "dot": dot_rows, "sum": summary}


def render_html(payload: dict[str, object], standalone: bool = True) -> str:
    template = (Path(__file__).parent / "report_template.html").read_text(encoding="utf-8")
    template = template.replace("__DATA__", json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    head, body = template.split("<!--BODY-->")
    if not standalone:
        return head + body
    return (
        '<!doctype html>\n<html lang="pl">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"{head}</head>\n<body>\n{body}</body>\n</html>\n"
    )


def render_markdown(payload: dict[str, object]) -> str:
    s = payload["sum"]
    pl = lambda v: str(v).replace(".", ",")  # noqa: E731

    def pos_rows(block: dict[str, object]) -> str:
        return "\n".join(
            f"| {p['id']} | {pl(p['deficit_pct'])} | {pl(p['decoupling_pct'])} | {pl(p['alarm_pct'])} | {p['rank']} |"
            for p in block["positives"]
        )

    k_rows = "\n".join(
        f"| {r['k']} | {pl(r['supine_auc'])} | {r['supine_tp']}/4, {r['supine_fp']} z 54 | {pl(r['dot_auc'])} | {r['dot_tp']}/3, {r['dot_fp']} z 14 |"
        for r in s["k_sensitivity"]
    )
    sup, dot, young = s["supine"], s["dot"], s["supine_young_id38"]
    return f"""# Dwie flagi: deficyt i rozprzężenie w sygnale fNIRS

**Projekt:** Heroes of the Brain 2026, ścieżka *Breath, Brain & Body* (Cortivision fNIRS).
Wersja z wykresami: [`REPORT.html`](REPORT.html). Poprzedni wariant (średnia 50/50) jest w historii gita.

## Pomysł

Każda osoba dostaje dwie oceny, obie liczone w pełni ślepo (bez etykiet) w zagnieżdżonym LOOCV:

- **deficyt**: równoważona średnia kwadratowa jednostronnych odchyleń w stronę stłumionej odpowiedzi,
- **rozprzężenie**: błąd rekonstrukcji PCA (SPE), czyli cechy przestają do siebie pasować.

Alarm to **wyższa z dwóch ocen**, ponownie skalibrowana na ocenach poza próbą. Niczego nie uśredniamy
i nie ma wag do strojenia. Jedyny parametr to liczba składowych PCA (k = 3).

## Wyniki (k = 3, próg {int(s['threshold'])})

| Zbiór | ROC-AUC | Powyżej progu | Alarmy wśród pozostałych | Ile osób przejrzeć, by mieć wszystkie |
| :--- | :---: | :---: | :---: | :---: |
| 1: pozycja leżąca (4 z {sup['n']}) | {pl(sup['auc'])} | {sup['tp']}/4 | {sup['fp']} z {sup['n'] - sup['n_pos']} | {sup['review_to_catch_all']} |
| 2: DOT, pozycja siedząca (3 z {dot['n']}) | {pl(dot['auc'])} | {dot['tp']}/3 | {dot['fp']} z {dot['n'] - dot['n_pos']} | {dot['review_to_catch_all']} |

### Zbiór 1: osoby na lekach kardiologicznych

| ID | Deficyt | Rozprzężenie | Alarm | Miejsce |
| :--- | :---: | :---: | :---: | :---: |
{pos_rows(sup)}

Ten sam przepis na samych cechach tętna daje AUC {pl(s['supine_heart_only']['auc'])}, na samych cechach korowych
{pl(s['supine_cortex_only']['auc'])}, a na pełnym profilu {pl(sup['auc'])}.

ID 38 (29 lat, beta-bloker) w analizie ograniczonej do {young['n_young']} osób w wieku do 42 lat:
rozprzężenie {pl(young['decoupling_pct'])}, deficyt {pl(young['deficit_pct'])}, miejsce {young['rank']} (remis na szczycie).

### Zbiór 2: astma i BMI ≥ 30

| ID | Deficyt | Rozprzężenie | Alarm | Miejsce |
| :--- | :---: | :---: | :---: | :---: |
{pos_rows(dot)}

Sam deficyt rezonansu daje tu AUC {pl(dot['auc_deficit'])}; flaga rozprzężenia dokłada jedną osobę bez rozpoznania.

## Stabilność względem k

| k | Zbiór 1: AUC | Zbiór 1: trafieni, alarmy | Zbiór 2: AUC | Zbiór 2: trafieni, alarmy |
| :---: | :---: | :---: | :---: | :---: |
{k_rows}

## Zakres wniosków

- Grupy wyróżnione mają 4 i 3 osoby. To pokaz działania, nie pomiar skuteczności.
- W Zbiorze 1 trzy z czterech osób to jedyni uczestnicy 70+ i sam wiek daje AUC {pl(s['supine_age_auc'])}.
  Leczenia nie da się tu oddzielić od wieku; przemawiają za tym tylko ID 38 i Zbiór 2 (wszyscy 20–29 lat).
- Na górze listy są też osoby bez leków. Narzędzie wskazuje, komu warto się przyjrzeć, i niczego nie rozpoznaje.
- ID 60 nie przekracza progu; ID 41 i 59 (niepełne nagrania) są poza analizą.
- Cechy tętna pochodzą z kanałów fNIRS, a w Zbiorze 2 (18 Hz) nie oceniamy ich osobno.
- Wariant, w którym rozprzężenie uwzględnia też resztę regresji serce → kora: AUC
  {pl(s['variants_k3']['supine_decoupling_spe_or_ridge']['auc'])} (Zbiór 1) i
  {pl(s['variants_k3']['dot_decoupling_spe_or_ridge']['auc'])} (Zbiór 2).

## Odtworzenie

```
python -m src.two_flag_scoring       # results/two_flag_*.csv, two_flag_summary.json
python -m src.make_two_flag_report   # REPORT.html, REPORT.md
```
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fragment", type=Path, default=None, help="also write a body-only HTML variant here")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    payload = build_payload(root / "results")
    (root / "REPORT.html").write_text(render_html(payload), encoding="utf-8")
    (root / "REPORT.md").write_text(render_markdown(payload), encoding="utf-8")
    if args.fragment is not None:
        args.fragment.write_text(render_html(payload, standalone=False), encoding="utf-8")


if __name__ == "__main__":
    main()
