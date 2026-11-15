"""Paths, feature sets, evaluation labels and thresholds shared by the whole pipeline."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "infopack_and_data"
RESULTS_DIR = ROOT / "results"
DOCS_DIR = ROOT / "docs"

# ---------------------------------------------------------------------------
# Feature sets fed to the models. Age, BMI and medication are never features.
# ---------------------------------------------------------------------------

SUPINE_FEATURE_COLS = [
    "bs_hr_bpm",
    "bs_sdnn_ms",
    "bs_rmssd_ms",
    "bs_lr_mayer_corr",
    "bs_mayer_amp",
    "bs_mayer_resp_ratio",
    "bh_dur_max",
    "bh_ssr_cortical_od850",
    "bh_deep_od850",
    "bh_hbr_hbo_ratio",
    "bh_spatial_cv",
    "bh_hr_delta",
]
# Right-skewed positive features that are log10-transformed before scaling.
SUPINE_LOG_COLS = [
    "bs_sdnn_ms",
    "bs_rmssd_ms",
    "bs_mayer_amp",
    "bs_mayer_resp_ratio",
    "bh_dur_max",
    "bh_ssr_cortical_od850",
    "bh_deep_od850",
    "bh_hbr_hbo_ratio",
    "bh_spatial_cv",
]
# Heart-only subset: what a pulse oximeter / smartwatch could also measure.
SUPINE_PPG_COLS = ["bs_hr_bpm", "bs_sdnn_ms", "bs_rmssd_ms", "bh_dur_max", "bh_hr_delta"]
SUPINE_BRAIN_COLS = [
    "bs_lr_mayer_corr",
    "bs_mayer_amp",
    "bs_mayer_resp_ratio",
    "bh_ssr_cortical_od850",
    "bh_deep_od850",
    "bh_hbr_hbo_ratio",
    "bh_spatial_cv",
]
SUPINE_HEART_COLS = ["bs_hr_bpm", "bs_sdnn_ms", "bs_rmssd_ms", "bh_hr_delta"]

DOT_FEATURE_COLS = [
    "rs_hr_bpm",
    "rs_sdnn_ms",
    "rs_rmssd_ms",
    "spb1_gain_01hz",
    "spb_gain_med",
    "spb_ssr_gain_med",
    "spb_lr_coherence",
    "spb_q_factor",
    "bh_dur_est",
    "bh_ssr_cortical_od850",
    "bh_hbr_hbo_ratio",
    "bh_spatial_cv",
    "bh_evening_morning_ratio",
]
DOT_LOG_COLS = [
    "rs_sdnn_ms",
    "rs_rmssd_ms",
    "spb1_gain_01hz",
    "spb_gain_med",
    "spb_ssr_gain_med",
    "bh_dur_est",
    "bh_ssr_cortical_od850",
    "bh_hbr_hbo_ratio",
    "bh_spatial_cv",
    "bh_evening_morning_ratio",
]
DOT_PPG_COLS = ["rs_hr_bpm", "rs_sdnn_ms", "rs_rmssd_ms", "bh_hr_delta"]
DOT_BRAIN_COLS = [
    "spb1_gain_01hz",
    "spb_gain_med",
    "spb_ssr_gain_med",
    "spb_lr_coherence",
    "spb_q_factor",
    "bh_ssr_cortical_od850",
    "bh_hbr_hbo_ratio",
    "bh_spatial_cv",
    "bh_evening_morning_ratio",
]

# ---------------------------------------------------------------------------
# Evaluation labels, taken from subjects_description.csv. Used only to score
# the finished rankings, never for fitting.
# ---------------------------------------------------------------------------

# Criterion A: cardiovascular medication (beta-blocker 38; antihypertensives 48, 49, 60).
# ID 59 (antihypertensive) has no breath-hold recording and is not in the analysed cohort.
SUPINE_CRIT_A_IDS = [38, 48, 49, 60]
# Criterion B: A plus obesity (BMI >= 30): IDs 2, 51, 55.
SUPINE_CRIT_B_IDS = [38, 48, 49, 60, 2, 51, 55]
# Medication without expected cardiovascular effect (isotretinoin).
SUPINE_NEG_CONTROL_IDS = [39, 43]

# Criterion A: asthma. Criterion B: asthma plus BMI >= 30 (IDs 9, 16).
DOT_CRIT_A_IDS = [12]
DOT_CRIT_B_IDS = [9, 12, 16]
# Medication without expected cardiovascular effect (gastrointestinal).
DOT_NEG_CONTROL_IDS = [17]

# ---------------------------------------------------------------------------
# Decision thresholds on out-of-fold percentiles.
# ---------------------------------------------------------------------------

ALARM_THRESHOLD = 90.0  # ~10% of a healthy cohort raises an alarm
SCREENING_THRESHOLD = 85.0  # looser screening variant (~15%)
N_COMPONENTS = 3  # PCA components; the only free parameter of the two-flag rule
