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

    sup, sup_b, dot, dot_a, young = s["supine"], s["supine_crit_b"], s["dot"], s["dot_crit_a"], s["supine_young_id38"]
    sup_neg = int(sup["n"] - sup["n_pos"])
    dot_neg = int(dot["n"] - dot["n_pos"])
    qc = s.get("qc_funnel", {})
    qc_enrolled = int(qc.get("enrolled", sup["n"]))
    qc_inc = qc.get("incomplete_ids", [])
    qc_fail = qc.get("qc_failed_short_channel_ids", [])
    thr_hi = int(s["threshold"])
    thr_lo = int(s.get("screening_threshold", 85))

    k_rows = "\n".join(
        f"| {r['k']} | {pl(r['supine_auc'])} | {r['supine_tp']}/{sup['n_pos']}, {r['supine_fp']} z {sup_neg} | {pl(r['dot_auc'])} | {r['dot_tp']}/{dot['n_pos']}, {r['dot_fp']} z {dot_neg} |"
        for r in s["k_sensitivity"]
    )
    # ranks use method="max", so rank == n_tied means the tie is at the very top
    n_tied = int(young.get("n_tied", 1))
    tie_note = ""
    if n_tied > 1:
        where = "na szczycie" if young["rank"] == n_tied else f"{n_tied} osoby z tym samym wynikiem"
        tie_note = f" (remis {where})" if young["rank"] == n_tied else f" (remis: {where})"

    def row_summary(label: str, block: dict[str, object]) -> str:
        ci_vals = block.get("auc_ci95")
        ci = f"[{pl(ci_vals[0])}; {pl(ci_vals[1])}]" if ci_vals else "(CI: n/d, 1 osoba)"
        return (
            f"| {label} | {pl(block['auc'])} {ci} | {pl(block['mw_p'])} | "
            f"{block['tp']}/{block['n_pos']} (FP: {block['fp']}) | "
            f"{block['tp_85']}/{block['n_pos']} (FP: {block['fp_85']}) | "
            f"{block['review_to_catch_all']} |"
        )

    inc_str = ", ".join(map(str, qc_inc)) if qc_inc else "brak"
    fail_str = ", ".join(map(str, qc_fail)) if qc_fail else "brak"

    return f"""# Dwie flagi: deficyt i rozprzężenie w sygnale fNIRS

**Projekt:** Heroes of the Brain 2026, ścieżka *Breath, Brain & Body* (Cortivision fNIRS).
Wersja z wykresami: [`REPORT.html`](REPORT.html). Poprzedni wariant (średnia 50/50) jest w historii gita.

## Pomysł

Każda osoba dostaje dwie oceny, obie liczone w pełni ślepo (bez etykiet) w zagnieżdżonym LOOCV:

- **deficyt**: równoważona średnia kwadratowa jednostronnych odchyleń w stronę stłumionej odpowiedzi,
- **rozprzężenie**: błąd rekonstrukcji PCA (SPE), czyli cechy przestają do siebie pasować.

Alarm to **wyższa z dwóch ocen**, ponownie skalibrowana na ocenach poza próbą. Niczego nie uśredniamy
i nie ma wag do strojenia. Jedyny parametr to liczba składowych PCA (k = 3).

## Kontrola jakości sygnału (QC, bez użycia etykiet)

W Zbiorze 1 (Supine) regresja kanałów krótkich (`SSR`) oraz pomiar tętna wymagają co najmniej jednego
ważnego kanału krótkiego (`10 mm`) i długiego w każdej sesji (`BS` i `BH`):
- zapisano: {qc_enrolled} osób;
- niepełny protokół (`BS` lub `BH`): {len(qc_inc)} osoby (ID {inc_str});
- odrzucone przez automatyczne QC kanałów krótkich (`SCI < 0,70` / nasycenie `> 4,90 V` / zanik `< 0,005 V`): {len(qc_fail)} osób z grupy kontrolnej (ID {fail_str} — u ID 3–11 nasycenie toru `850 nm` w obu sesjach, u ID 35 zanik toru `760 nm` w `BS`);
- **włączono do głównej analizy Supine:** **{sup['n']} osób** (bez utraty żadnego przypadku z Kryterium A ani B).

## Wyniki (k = 3, ujednolicone kryteria A/B oraz progi {thr_hi} i {thr_lo})

- **Kryterium A**: leki kardiologiczne/hipotensyjne lub przewlekłe choroby krążeniowo-oddechowe
  (Supine: {sup['n_pos']}/{sup['n']} — beta-bloker i leki na nadciśnienie, bez izotretynoiny; DOT: {dot_a['n_pos']}/{dot_a['n']} — astma).
- **Kryterium B**: Kryterium A lub otyłość (BMI ≥ 30) (Supine: {sup_b['n_pos']}/{sup_b['n']}, DOT: {dot['n_pos']}/{dot['n']}).

Wartości p (jednostronny test Manna-Whitneya) są opisowe i nie są korygowane na wielokrotne porównania
(4 wiersze × 2 progi). Dla grupy jednoosobowej przedział ufności nie ma sensu, więc go nie podajemy.

| Zbiór i kryterium | ROC-AUC (95% CI) | p (MW) | Próg {thr_hi} (~10%) | Próg {thr_lo} (~15%) | Ile osób przejrzeć, by mieć wszystkie |
| :--- | :---: | :---: | :---: | :---: | :---: |
{row_summary(f"1: Supine — Kryt. A ({sup['n_pos']} z {sup['n']})", sup)}
{row_summary(f"1: Supine — Kryt. B ({sup_b['n_pos']} z {sup_b['n']})", sup_b)}
{row_summary(f"2: DOT — Kryt. A ({dot_a['n_pos']} z {dot_a['n']})", dot_a)}
{row_summary(f"2: DOT — Kryt. B ({dot['n_pos']} z {dot['n']})", dot)}

### Zbiór 1: osoby na lekach kardiologicznych (Kryterium A)

| ID | Deficyt | Rozprzężenie | Alarm | Miejsce |
| :--- | :---: | :---: | :---: | :---: |
{pos_rows(sup)}

Ten sam przepis na samych cechach tętna daje AUC {pl(s['supine_heart_only']['auc'])}, na samych cechach korowych
{pl(s['supine_cortex_only']['auc'])}, a na pełnym profilu {pl(sup['auc'])}.

ID 38 (29 lat, beta-bloker) w analizie ograniczonej do {young['n_young']} osób w wieku do 42 lat:
rozprzężenie {pl(young['decoupling_pct'])}, deficyt {pl(young['deficit_pct'])}, alarm {pl(young['alarm_pct'])}, miejsce {young['rank']}{tie_note}.

### Zbiór 2: astma i BMI ≥ 30 (Kryterium B)

| ID | Deficyt | Rozprzężenie | Alarm | Miejsce |
| :--- | :---: | :---: | :---: | :---: |
{pos_rows(dot)}

Sam deficyt rezonansu daje tu AUC {pl(dot['auc_deficit'])}; flaga rozprzężenia dokłada jedną osobę bez rozpoznania.

## Stabilność względem k

| k | Zbiór 1 (Kryt. A): AUC | Zbiór 1: trafieni, alarmy (próg {thr_hi}) | Zbiór 2 (Kryt. B): AUC | Zbiór 2: trafieni, alarmy (próg {thr_hi}) |
| :---: | :---: | :---: | :---: | :---: |
{k_rows}

## Zakres wniosków

- Grupy wyróżnione mają {sup['n_pos']} i {dot['n_pos']} osoby (w DOT Kryt. A to {dot_a['n_pos']} osoba). To pokaz działania, nie pomiar skuteczności.
- W Zbiorze 1 trzy z czterech osób z Kryt. A to jedyni uczestnicy 70+ i sam wiek daje AUC {pl(s['supine_age_auc'])}.
  Leczenia nie da się tu oddzielić od wieku; przemawiają za tym tylko ID 38 i Zbiór 2 (wszyscy 20–29 lat).
- BMI ≥ 30 w obu zbiorach to tylko kilka osób, więc nie wyciągamy wniosków o jego wpływie. W Supine dodanie
  trzech młodych osób z otyłością zmienia AUC do {pl(sup_b['auc'])} (ID 2 łapie flaga rozprzężenia, ale ID 51 i 55
  nie są wykrywane). W DOT kryteria A i B różnią się o 2 osoby (1 vs 3); przy tak małych grupach różnicy AUC nie interpretujemy.
- Na górze listy są też osoby bez leków. Narzędzie wskazuje, komu warto się przyjrzeć, i niczego nie rozpoznaje.
- W Supine ID 48 jest tuż pod progiem {thr_hi} (87,2 — przekracza próg {thr_lo}), a ID 60 nie przekracza progów {thr_hi}/{thr_lo}; poza analizą są ID {inc_str} (niepełne nagrania) oraz ID {fail_str} (brak ważnych kanałów krótkich wg QC).
- Cechy tętna pochodzą z kanałów fNIRS (nie z osobnego pulsoksymetru), a w Zbiorze 2 (18 Hz) nie oceniamy ich osobno.
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
