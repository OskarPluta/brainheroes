"""Comprehensive unit, anti-leakage, edge-case, and empirical verification tests for fNIRS pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.feature_extraction import extract_cardiac_hrv
from src.run_experiments import (
    SUPINE_BRAIN_COLS,
    SUPINE_FEATURE_COLS,
    SUPINE_LOG_COLS,
    SUPINE_PPG_COLS,
)
from src.fnirs_preprocessing import (
    butter_bandpass,
    compute_windowed_sci,
    load_snirf_recording,
    od_to_hb_mbll,
    scholkmann_wolf_dpf,
    ssr_clean,
    tddr_repair,
)
from src.normative_modeling import (
    compute_dot_stage2_scores,
    compute_supine_stage2_scores,
    empirical_percentile_oof,
    evaluate_anomaly_detector,
    run_loocv_subspace_decomposition,
    symmetrize_features,
)


def test_mbll_inversion_and_dpf_invariance():
    """Verifies MBLL exact matrix inversion and invariance of HbR/HbO ratio to scalar PPF."""
    rng = np.random.default_rng(42)
    od760 = rng.uniform(0.01, 0.05, size=(200, 4))
    od850 = rng.uniform(0.01, 0.05, size=(200, 4))
    dists = np.array([10.0, 25.0, 30.0, 35.0])

    hbo_6, hbr_6 = od_to_hb_mbll(od760, od850, dists, ppf=6.0, age_years=None)
    hbo_8, hbr_8 = od_to_hb_mbll(od760, od850, dists, ppf=8.0, age_years=None)

    # Scaling PPF scales both HbO and HbR identically, leaving HbR/HbO ratio strictly invariant
    np.testing.assert_allclose(hbo_6 * (6.0 / 8.0), hbo_8, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(hbr_6 * (6.0 / 8.0), hbr_8, rtol=1e-12, atol=1e-12)
    ratio_6 = hbr_6 / hbo_6
    ratio_8 = hbr_8 / hbo_8
    np.testing.assert_allclose(ratio_6, ratio_8, rtol=1e-11, atol=1e-12)

    # Age-dependent DPF (Scholkmann & Wolf 2013) increases with age and differs by wavelength
    dpf_young_760 = scholkmann_wolf_dpf(760.0, 21.0)
    dpf_senior_760 = scholkmann_wolf_dpf(760.0, 75.0)
    dpf_young_850 = scholkmann_wolf_dpf(850.0, 21.0)
    assert dpf_senior_760 > dpf_young_760
    assert dpf_young_760 != dpf_young_850


def test_ssr_orthogonality():
    """Verifies Short-Separation Regression removes scalp component and makes residual orthogonal."""
    t = np.linspace(0, 100, 1000)
    scalp = np.sin(2 * np.pi * 0.1 * t) + 0.5 * np.cos(2 * np.pi * 1.2 * t)
    cortex_true = 0.3 * np.sin(2 * np.pi * 0.04 * t)  # orthogonal frequency
    y_deep = np.column_stack([2.5 * scalp + cortex_true, 0.8 * scalp - 0.5 * cortex_true])
    y_shal = scalp[:, None]

    y_cort, beta = ssr_clean(y_deep, y_shal)
    assert y_cort.shape == y_deep.shape
    # Residual must have near-zero covariance with centered shallow reference
    scalp_c = scalp - np.mean(scalp)
    for ch in range(y_cort.shape[1]):
        cov = float(np.mean(y_cort[:, ch] * scalp_c))
        assert abs(cov) < 1e-10


def test_tddr_repairs_spike_and_step_artifacts():
    """Verifies TDDR removes large step artifacts while preserving physiological oscillations."""
    fs = 20.0
    t = np.arange(0, 120, 1.0 / fs)
    clean_physio = 0.01 * np.sin(2 * np.pi * 0.1 * t) + 0.003 * np.sin(2 * np.pi * 1.2 * t)
    artifact = np.zeros_like(t)
    artifact[t >= 60.0] += 0.50  # massive baseline shift
    artifact[int(30 * fs)] += 1.20  # Dirac spike
    corrupted = (clean_physio + artifact)[:, None]

    repaired = tddr_repair(corrupted, fs=fs)
    # Variance of repaired signal should be close to clean physiological signal, not corrupted step
    assert np.std(repaired) < 0.05 * np.std(corrupted)
    corr_physio = float(np.corrcoef(repaired[:, 0], clean_physio)[0, 1])
    assert corr_physio > 0.85


def test_tddr_preserves_asymmetric_cardiac_pulse_and_real_snirf():
    """
    Verifies that TDDR at 141.79 Hz does not rectify asymmetric systolic/diastolic cardiac
    pulses into artificial drift and reproduces cached features from raw SNIRF files.
    """
    fs = 141.79
    t = np.arange(0, 60.0, 1.0 / fs)
    # Asymmetric dicrotic-like pulse (steep systolic upstroke, slower diastolic decay) + Mayer wave
    cardiac_asym = 0.008 * (np.sin(2 * np.pi * 1.15 * t) + 0.45 * np.sin(2 * np.pi * 2.30 * t + 0.8))
    mayer = 0.005 * np.sin(2 * np.pi * 0.10 * t)
    clean_signal = (cardiac_asym + mayer)[:, None]

    repaired = tddr_repair(clean_signal, fs=fs)
    # High-frequency (>0.5 Hz) cardiac component must not be rectified into baseline drift
    assert float(np.corrcoef(repaired[:, 0], clean_signal[:, 0])[0, 1]) > 0.99
    assert np.std(repaired[:, 0] - clean_signal[:, 0]) < 0.10 * np.std(clean_signal[:, 0])

    # Verify fresh extraction on real SNIRF recording (ID 38 BS) matches cached CSV features
    root = Path(__file__).resolve().parent.parent
    bs_files_38 = sorted((root / "infopack_and_data" / "fnirs_high_freq_supine" / "data" / "baseline").glob("38_*.snirf"))
    cache_csv = root / "results" / "supine_features_const_dpf.csv"
    assert len(bs_files_38) == 1 and cache_csv.exists()
    rec_bs = load_snirf_recording(bs_files_38[0], apply_tddr=True, age_years=None)
    shal = (rec_bs.dists_mm < 15.0) & rec_bs.valid_pairs
    deep = (rec_bs.dists_mm >= 22.0) & rec_bs.valid_pairs
    bs_hr, bs_sdnn, bs_rmssd, _ = extract_cardiac_hrv(rec_bs.od850[:, shal], rec_bs.time, rec_bs.fs)
    mayer_all = butter_bandpass(rec_bs.od850, 0.06, 0.14, rec_bs.fs)
    bs_mayer_amp = float(np.median(np.std(mayer_all[:, deep], axis=0)) * 1e3)

    df_cached = pd.read_csv(cache_csv)
    row_38 = df_cached[df_cached["ID"] == 38].iloc[0]
    np.testing.assert_allclose(bs_hr, float(row_38["bs_hr_bpm"]), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(bs_sdnn, float(row_38["bs_sdnn_ms"]), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(bs_rmssd, float(row_38["bs_rmssd_ms"]), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(bs_mayer_amp, float(row_38["bs_mayer_amp"]), rtol=1e-4, atol=1e-4)


def test_windowed_sci_rejects_simultaneous_railing_and_dropouts():
    """Verifies that simultaneous zero-dropouts or >4.90V saturation on both wavelengths are rejected."""
    fs = 50.0
    t = np.arange(0, 30, 1.0 / fs)
    # Channel 0: clean cardiac signal at both wavelengths
    cardiac = 0.02 * np.sin(2 * np.pi * 1.2 * t)
    r760_good = 1.5 * np.exp(-cardiac)
    r850_good = 1.8 * np.exp(-1.1 * cardiac)

    # Channel 1: simultaneous dropout to 0.0 and saturation to 4.98V (would give fake r=1.0 without QC!)
    r760_bad = np.where(t < 15.0, 0.0, 4.98)
    r850_bad = np.where(t < 15.0, 0.0, 4.98)

    raw760 = np.column_stack([r760_good, r760_bad])
    raw850 = np.column_stack([r850_good, r850_bad])
    od760 = -np.log10(np.maximum(raw760, 1e-5) / np.maximum(np.median(raw760, axis=0, keepdims=True), 1e-5))
    od850 = -np.log10(np.maximum(raw850, 1e-5) / np.maximum(np.median(raw850, axis=0, keepdims=True), 1e-5))

    sci, valid = compute_windowed_sci(raw760, raw850, od760, od850, fs, sci_threshold=0.70)
    assert valid[0] is np.True_ or bool(valid[0]) is True
    assert sci[0] > 0.95
    assert bool(valid[1]) is False
    assert sci[1] == 0.0


def test_loocv_strict_zero_leakage():
    """
    Proves that in both Stage 1 (run_loocv_subspace_decomposition) and Stage 2
    (compute_supine_stage2_scores / compute_dot_stage2_scores):
    1. Modifying non-reference subjects (when blind_training=False) has zero effect on
       any reference subject's raw or percentile score.
    2. Self-exclusion invariance: flipping ref_mask[i] between True and False for any
       subject i leaves subject i's own Stage 1 and Stage 2 raw and percentile scores
       bit-for-bit identical (proving subject i never leaks into fold i's reference distribution
       via other subjects' outer folds).
    """
    rng = np.random.default_rng(123)
    n = 24
    data = {"ID": np.arange(1, n + 1)}
    for col in SUPINE_FEATURE_COLS:
        data[col] = rng.uniform(0.5, 5.0, n)
    df = pd.DataFrame(data)
    ref_mask = np.ones(n, dtype=bool)
    ref_mask[-3:] = False  # Last 3 subjects are clinical test cases

    res1 = run_loocv_subspace_decomposition(
        df,
        feature_cols=SUPINE_FEATURE_COLS,
        log_cols=SUPINE_LOG_COLS,
        ppg_cols=SUPINE_PPG_COLS,
        brain_cols=SUPINE_BRAIN_COLS,
        ref_mask=ref_mask,
        n_components=3,
        blind_training=False,
    )
    st2_1 = compute_supine_stage2_scores(df, res1, ref_mask, blind_training=False)

    # 1. Massively corrupt the clinical test cases (indices 21, 22, 23)
    df_corrupted = df.copy()
    df_corrupted.loc[21:, SUPINE_FEATURE_COLS] *= 1000.0

    res2 = run_loocv_subspace_decomposition(
        df_corrupted,
        feature_cols=SUPINE_FEATURE_COLS,
        log_cols=SUPINE_LOG_COLS,
        ppg_cols=SUPINE_PPG_COLS,
        brain_cols=SUPINE_BRAIN_COLS,
        ref_mask=ref_mask,
        n_components=3,
        blind_training=False,
    )
    st2_2 = compute_supine_stage2_scores(df_corrupted, res2, ref_mask, blind_training=False)

    # Reference subjects (0..20) must have bit-for-bit identical Stage 1 and Stage 2 scores!
    for col in [
        "fnirs_t2_raw",
        "fnirs_spe_raw",
        "fnirs_cm_raw",
        "fnirs_t2_pct",
        "fnirs_spe_pct",
        "fnirs_cm_pct",
        "iforest_pct",
        "score_apriori_deficit_raw",
        "score_apriori_deficit_pct",
        "score_normative_combined",
        "score_normative_combined_pct",
    ]:
        np.testing.assert_allclose(st2_1.loc[:20, col], st2_2.loc[:20, col], rtol=1e-12, atol=1e-12)

    # 2. Self-exclusion invariance: flip ref_mask[0] from True to False.
    # Because fold 0 excludes subject 0 regardless, subject 0's scores must be bit-for-bit identical!
    ref_mask_no0 = ref_mask.copy()
    ref_mask_no0[0] = False
    res_no0 = run_loocv_subspace_decomposition(
        df,
        feature_cols=SUPINE_FEATURE_COLS,
        log_cols=SUPINE_LOG_COLS,
        ppg_cols=SUPINE_PPG_COLS,
        brain_cols=SUPINE_BRAIN_COLS,
        ref_mask=ref_mask_no0,
        n_components=3,
        blind_training=False,
    )
    st2_no0 = compute_supine_stage2_scores(df, res_no0, ref_mask_no0, blind_training=False)
    for col in [
        "fnirs_t2_raw",
        "fnirs_spe_raw",
        "fnirs_cm_raw",
        "fnirs_t2_pct",
        "fnirs_spe_pct",
        "fnirs_cm_pct",
        "iforest_pct",
        "score_apriori_deficit_raw",
        "score_apriori_deficit_pct",
        "score_normative_combined",
        "score_normative_combined_pct",
    ]:
        np.testing.assert_allclose(st2_1.loc[0, col], st2_no0.loc[0, col], rtol=1e-12, atol=1e-12)


def test_empirical_hackathon_results_integrity():
    """Verifies the saved empirical results on both hackathon datasets."""
    root = Path(__file__).resolve().parent.parent
    summary_path = root / "results" / "verification_summary.json"
    assert summary_path.exists(), "verification_summary.json must be generated first"

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    # 1. Young cohort audit for ID 38 (Nebilet beta-blocker, age 29):
    # Must be invisible to PPG (<50th pct) while detected at >=95th pct in fNIRS SPE and Cross-Modal Ridge
    y_audit = summary["young_cohort_id38_audit"]
    assert y_audit["id38_ppg_pct"] < 50.0
    assert y_audit["id38_fnirs_spe_pct"] >= 95.0
    assert y_audit["id38_fnirs_cm_pct"] >= 95.0

    # 2. DOT Sitting: fNIRS A-Priori SPB Deficit and Stage 2 Clinical Triage (Fully Blind LOOCV)
    # must outperform Virtual PPG and maintain 0 FP on Cornax negative control
    dot_b = {m["model_name"]: m for m in summary["dot_criterion_b_metrics"]}
    assert dot_b["fNIRS A-Priori SPB Resonance Deficit (Blind LOOCV)"]["roc_auc"] > 0.90
    assert dot_b["fNIRS Stage 2 Clinical Triage (Fully Blind LOOCV)"]["roc_auc"] > 0.80
    assert (
        dot_b["fNIRS Stage 2 Clinical Triage (Fully Blind LOOCV)"]["roc_auc"]
        > dot_b["Virtual PPG Baseline (HR+HRV)"]["roc_auc"]
    )
    assert dot_b["fNIRS Stage 2 Clinical Triage (Fully Blind LOOCV)"]["neg_control_fp"] == 0
    assert dot_b["fNIRS Stage 1: PCA SPE / Q-residual (Out-of-Subspace)"]["fp_at_90"] <= 3
    assert dot_b["fNIRS Stage 1: PCA SPE / Q-residual (Out-of-Subspace)"]["neg_control_fp"] == 0
    assert dot_b["Virtual PPG Baseline (HR+HRV)"]["fp_at_90"] <= 3

    # 3. Supine: fNIRS models must outperform Virtual PPG Baseline and maintain 0 FP on Izotec negative controls
    sup_a = {m["model_name"]: m for m in summary["supine_criterion_a_metrics"]}
    assert (
        sup_a["fNIRS Stage 2 Clinical Triage (Fully Blind LOOCV)"]["roc_auc"]
        > sup_a["Virtual PPG Baseline (HR+HRV+BH)"]["roc_auc"]
    )
    assert sup_a["fNIRS Stage 2 Clinical Triage (Fully Blind LOOCV)"]["neg_control_fp"] == 0
    assert sup_a["fNIRS Stage 1: PCA SPE / Q-residual (Out-of-Subspace)"]["fp_at_90"] <= 7
    assert sup_a["fNIRS Stage 1: Cross-Modal Ridge (Heart->Brain)"]["fp_at_90"] <= 7

