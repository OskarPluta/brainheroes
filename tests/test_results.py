"""Checks against the real datasets and the committed results (skipped without the data)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from brainheroes.constants import (
    DATA_DIR,
    RESULTS_DIR,
    SUPINE_CRIT_A_IDS,
    SUPINE_CRIT_B_IDS,
    SUPINE_FEATURE_COLS,
)
from brainheroes.features import SUPINE_FEATURES_CSV, _supine_baseline
from brainheroes.preprocessing import load_snirf_recording

SUPINE_RAW = DATA_DIR / "fnirs_high_freq_supine"
needs_raw = pytest.mark.skipif(not SUPINE_RAW.exists(), reason="raw dataset not available")
needs_results = pytest.mark.skipif(not (RESULTS_DIR / SUPINE_FEATURES_CSV).exists(), reason="feature tables not built")


@needs_raw
@needs_results
def test_fresh_extraction_matches_the_cached_baseline_features():
    path = next((SUPINE_RAW / "data" / "baseline").glob("38_*.snirf"))
    rec = load_snirf_recording(path)
    short = (rec.dists_mm < 15.0) & rec.valid_pairs
    deep = (rec.dists_mm >= 22.0) & rec.valid_pairs
    fresh = _supine_baseline(rec, short, deep)
    cached = pd.read_csv(RESULTS_DIR / SUPINE_FEATURES_CSV).set_index("ID").loc[38]
    for name, value in fresh.items():
        np.testing.assert_allclose(value, cached[name], rtol=1e-6, err_msg=name)


@needs_results
def test_qc_keeps_48_complete_subjects_and_every_flagged_case():
    df = pd.read_csv(RESULTS_DIR / SUPINE_FEATURES_CSV)
    assert len(df) == 60
    analysed = df[df["qc_pass"]]
    assert len(analysed) == 48
    assert not analysed[SUPINE_FEATURE_COLS].isna().any().any()
    assert set(df.loc[~df["qc_pass"], "ID"]) == {3, 4, 5, 6, 7, 8, 9, 10, 11, 35, 41, 59}
    assert set(SUPINE_CRIT_B_IDS) <= set(analysed["ID"])
    # Criterion B is criterion A plus everyone analysed with BMI >= 30.
    obese = set(analysed.loc[analysed["BMI"] >= 30.0, "ID"])
    assert set(SUPINE_CRIT_B_IDS) == set(SUPINE_CRIT_A_IDS) | obese


@pytest.mark.skipif(not (RESULTS_DIR / "verification_summary.json").exists(), reason="ablation not run")
def test_ablation_headline_claims_still_hold():
    summary = json.loads((RESULTS_DIR / "verification_summary.json").read_text(encoding="utf-8"))
    young = summary["young_cohort_id38_audit"]
    # ID 38 is ordinary for heart-only features but stands out on decoupling.
    assert young["id38_ppg_pct"] < 50.0
    assert young["id38_fnirs_spe_pct"] >= 95.0
    assert young["id38_fnirs_cm_pct"] >= 95.0

    dot = {m["model_name"]: m for m in summary["dot_criterion_b_metrics"]}
    triage = dot["fNIRS Stage 2 Clinical Triage (Fully Blind LOOCV)"]
    assert dot["fNIRS A-Priori SPB Resonance Deficit (Blind LOOCV)"]["roc_auc"] > 0.90
    assert triage["roc_auc"] > max(0.80, dot["Virtual PPG Baseline (HR+HRV)"]["roc_auc"])
    assert triage["neg_control_fp"] == 0
    spe = dot["fNIRS Stage 1: PCA SPE / Q-residual (Out-of-Subspace)"]
    assert spe["fp_at_90"] <= 3 and spe["neg_control_fp"] == 0
    assert dot["Virtual PPG Baseline (HR+HRV)"]["fp_at_90"] <= 3

    sup = {m["model_name"]: m for m in summary["supine_criterion_a_metrics"]}
    triage = sup["fNIRS Stage 2 Clinical Triage (Fully Blind LOOCV)"]
    assert triage["roc_auc"] > sup["Virtual PPG Baseline (HR+HRV+BH)"]["roc_auc"]
    assert triage["neg_control_fp"] == 0
    assert sup["fNIRS Stage 1: PCA SPE / Q-residual (Out-of-Subspace)"]["fp_at_90"] <= 7
    assert sup["fNIRS Stage 1: Cross-Modal Ridge (Heart->Brain)"]["fp_at_90"] <= 7
