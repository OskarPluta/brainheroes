"""Complete verification, ablation, sensitivity analysis, and figure generation pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from src.feature_extraction import extract_dot_features, extract_supine_features
from src.normative_modeling import (
    compute_dot_stage2_scores,
    compute_supine_stage2_scores,
    empirical_percentile_oof,
    evaluate_anomaly_detector,
    run_loocv_subspace_decomposition,
)


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

SUPINE_PPG_COLS = [
    "bs_hr_bpm",
    "bs_sdnn_ms",
    "bs_rmssd_ms",
    "bh_dur_max",
    "bh_hr_delta",
]

SUPINE_BRAIN_COLS = [
    "bs_lr_mayer_corr",
    "bs_mayer_amp",
    "bs_mayer_resp_ratio",
    "bh_ssr_cortical_od850",
    "bh_deep_od850",
    "bh_hbr_hbo_ratio",
    "bh_spatial_cv",
]


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

DOT_PPG_COLS = [
    "rs_hr_bpm",
    "rs_sdnn_ms",
    "rs_rmssd_ms",
    "bh_hr_delta",
]

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


def run_all_verifications(root_dir: Path | str) -> Dict[str, object]:
    """Runs all experiments, methodological audits, ablations, and generates plots."""
    root = Path(root_dir)
    data_root = root / "infopack_and_data"
    res_dir = root / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    df_sup_all = extract_supine_features(
        data_root, use_age_dpf=False, cache_path=res_dir / "supine_features_const_dpf.csv"
    )
    df_sup_age_all = extract_supine_features(
        data_root, use_age_dpf=True, cache_path=res_dir / "supine_features_age_dpf.csv"
    )
    df_dot = extract_dot_features(
        data_root, use_age_dpf=False, cache_path=res_dir / "dot_features_const_dpf.csv"
    )

    # Filter complete BS+BH recordings passing dual-wavelength short+long channel QC (N=48)
    df_sup = df_sup_all[df_sup_all["qc_pass"].astype(bool)].reset_index(drop=True)
    df_sup_age = df_sup_age_all[df_sup_age_all["qc_pass"].astype(bool)].reset_index(drop=True)

    qc_failed_df = df_sup_all[df_sup_all["has_bs"] & df_sup_all["has_bh"] & (~df_sup_all["qc_pass"].astype(bool))]
    supine_qc_funnel = {
        "enrolled": int(len(df_sup_all)),
        "incomplete_ids": [int(x) for x in df_sup_all.loc[~(df_sup_all["has_bs"] & df_sup_all["has_bh"]), "ID"]],
        "qc_failed_ids": [int(x) for x in qc_failed_df["ID"]],
        "qc_failed_reasons": {int(r.ID): str(r.qc_exclusion_reason) for _, r in qc_failed_df.iterrows()},
        "analyzed": int(len(df_sup)),
    }

    # Ground-truth definitions for Supine (N=58)
    # Criterion A (Strict Cardiovascular Pharmacology/Pathology): ID 38 (Nebilet), ID 48, 49, 60 (Antihypertensives)
    sup_pos_crit_a = df_sup["ID"].isin([38, 48, 49, 60]).to_numpy()
    # Criterion B (Cardiovascular + WHO Obesity BMI >= 30.0): adds ID 2, 51, 55
    sup_pos_crit_b = (df_sup["ID"].isin([38, 48, 49, 60]) | (df_sup["BMI"] >= 30.0)).to_numpy()
    sup_neg_ctrl = df_sup["ID"].isin([39, 43]).to_numpy()  # Izotec (dermatological)
    sup_ref_mask_a = ~sup_pos_crit_a
    sup_ref_mask_b = ~sup_pos_crit_b

    # Ground-truth definitions for DOT Sitting (N=17)
    # Criterion A (Strict Cardiorespiratory Diagnosis): ID 12 (Asthma + bronchodilator)
    dot_pos_crit_a = (df_dot["ID"] == 12).to_numpy()
    # Criterion B (Cardiorespiratory + WHO Class I Obesity BMI >= 30.0): ID 12, 9, 16
    dot_pos_crit_b = (df_dot["ID"].isin([12, 9, 16])).to_numpy()
    dot_neg_ctrl = (df_dot["ID"] == 17).to_numpy()  # Cornax (GI)
    dot_ref_mask_a = ~dot_pos_crit_a
    dot_ref_mask_b = ~dot_pos_crit_b

    # Run Supine LOOCV (Fully Blind Unsupervised as primary + Normative Pre-Screened Reference)
    sup_loocv_blind = run_loocv_subspace_decomposition(
        df_sup,
        SUPINE_FEATURE_COLS,
        SUPINE_LOG_COLS,
        SUPINE_PPG_COLS,
        SUPINE_BRAIN_COLS,
        sup_ref_mask_a,
        n_components=3,
        blind_training=True,
    )
    sup_scores_blind = compute_supine_stage2_scores(df_sup, sup_loocv_blind, sup_ref_mask_a, blind_training=True)

    sup_loocv_norm = run_loocv_subspace_decomposition(
        df_sup,
        SUPINE_FEATURE_COLS,
        SUPINE_LOG_COLS,
        SUPINE_PPG_COLS,
        SUPINE_BRAIN_COLS,
        sup_ref_mask_a,
        n_components=3,
        blind_training=False,
    )
    sup_scores_norm = compute_supine_stage2_scores(df_sup, sup_loocv_norm, sup_ref_mask_a, blind_training=False)

    sup_loocv_norm_b = run_loocv_subspace_decomposition(
        df_sup,
        SUPINE_FEATURE_COLS,
        SUPINE_LOG_COLS,
        SUPINE_PPG_COLS,
        SUPINE_BRAIN_COLS,
        sup_ref_mask_b,
        n_components=3,
        blind_training=False,
    )
    sup_scores_norm_b = compute_supine_stage2_scores(df_sup, sup_loocv_norm_b, sup_ref_mask_b, blind_training=False)

    # Age-DPF sensitivity on Supine (Fully Blind LOOCV)
    sup_loocv_agedpf = run_loocv_subspace_decomposition(
        df_sup_age,
        SUPINE_FEATURE_COLS,
        SUPINE_LOG_COLS,
        SUPINE_PPG_COLS,
        SUPINE_BRAIN_COLS,
        sup_ref_mask_a,
        n_components=3,
        blind_training=True,
    )
    sup_scores_agedpf = compute_supine_stage2_scores(
        df_sup_age, sup_loocv_agedpf, sup_ref_mask_a, blind_training=True
    )

    # Run DOT Sitting LOOCV (Fully Blind Unsupervised as primary + Normative Pre-Screened Reference)
    dot_loocv_blind = run_loocv_subspace_decomposition(
        df_dot,
        DOT_FEATURE_COLS,
        DOT_LOG_COLS,
        DOT_PPG_COLS,
        DOT_BRAIN_COLS,
        dot_ref_mask_b,
        n_components=3,
        blind_training=True,
    )
    dot_scores_blind = compute_dot_stage2_scores(df_dot, dot_loocv_blind, dot_ref_mask_b, blind_training=True)

    dot_loocv_norm_a = run_loocv_subspace_decomposition(
        df_dot,
        DOT_FEATURE_COLS,
        DOT_LOG_COLS,
        DOT_PPG_COLS,
        DOT_BRAIN_COLS,
        dot_ref_mask_a,
        n_components=3,
        blind_training=False,
    )
    dot_scores_norm_a = compute_dot_stage2_scores(df_dot, dot_loocv_norm_a, dot_ref_mask_a, blind_training=False)

    dot_loocv_norm = run_loocv_subspace_decomposition(
        df_dot,
        DOT_FEATURE_COLS,
        DOT_LOG_COLS,
        DOT_PPG_COLS,
        DOT_BRAIN_COLS,
        dot_ref_mask_b,
        n_components=3,
        blind_training=False,
    )
    dot_scores_norm = compute_dot_stage2_scores(df_dot, dot_loocv_norm, dot_ref_mask_b, blind_training=False)

    # Save full score tables from Fully Blind LOOCV (zero label usage)
    sup_merged = pd.concat([df_sup, sup_scores_blind.drop(columns=["ID"])], axis=1)
    sup_merged.to_csv(res_dir / "supine_loocv_results.csv", index=False)

    dot_merged = pd.concat([df_dot, dot_scores_blind.drop(columns=["ID"])], axis=1)
    dot_merged.to_csv(res_dir / "dot_loocv_results.csv", index=False)

    # Evaluate Supine models (Criterion A: Cardiovascular Medication/Disease)
    sup_eval_models = [
        ("Virtual PPG Baseline (HR+HRV+BH)", "ppg_comb_raw", "ppg_pct", sup_scores_blind),
        ("Isolation Forest (Unsupervised Baseline)", "iforest_raw", "iforest_pct", sup_scores_blind),
        ("fNIRS Stage 1: Hotelling's T^2 (Within-Subspace)", "fnirs_t2_raw", "fnirs_t2_pct", sup_scores_blind),
        ("fNIRS Stage 1: PCA SPE / Q-residual (Out-of-Subspace)", "fnirs_spe_raw", "fnirs_spe_pct", sup_scores_blind),
        ("fNIRS Stage 1: Cross-Modal Ridge (Heart->Brain)", "fnirs_cm_raw", "fnirs_cm_pct", sup_scores_blind),
        ("fNIRS Stage 1: Subspace Max (T2/SPE/CM)", "fnirs_subspace_pct", "fnirs_subspace_pct", sup_scores_blind),
        ("fNIRS A-Priori Physiological Deficit (Blind LOOCV)", "score_apriori_deficit_raw", "score_apriori_deficit_pct", sup_scores_blind),
        ("fNIRS Stage 2 Clinical Triage (Fully Blind LOOCV)", "score_normative_combined", "score_normative_combined_pct", sup_scores_blind),
        ("fNIRS Stage 2 Clinical Triage (Normative Pre-Screened LOOCV)", "score_normative_combined", "score_normative_combined_pct", sup_scores_norm),
        ("fNIRS Stage 2 Clinical Triage (Age-DPF Sensitivity, Blind)", "score_normative_combined", "score_normative_combined_pct", sup_scores_agedpf),
    ]

    sup_metrics_a = []
    for name, rcol, pcol, s_df in sup_eval_models:
        m = evaluate_anomaly_detector(
            s_df[rcol].to_numpy(),
            s_df[pcol].to_numpy(),
            sup_pos_crit_a,
            sup_neg_ctrl,
            name,
        )
        sup_metrics_a.append(m.__dict__)

    # Evaluate Supine Criterion B (adding BMI >= 30: ID 2, 51, 55)
    sup_metrics_b = []
    for name, rcol, pcol, s_df in [
        ("Virtual PPG Baseline (Blind LOOCV)", "ppg_comb_raw", "ppg_pct", sup_scores_blind),
        ("fNIRS Stage 1: PCA SPE (Blind LOOCV)", "fnirs_spe_raw", "fnirs_spe_pct", sup_scores_blind),
        ("fNIRS Stage 2 Clinical Triage (Fully Blind LOOCV)", "score_normative_combined", "score_normative_combined_pct", sup_scores_blind),
        ("fNIRS Stage 2 Clinical Triage (Normative Pre-Screened LOOCV)", "score_normative_combined", "score_normative_combined_pct", sup_scores_norm_b),
    ]:
        m = evaluate_anomaly_detector(
            s_df[rcol].to_numpy(),
            s_df[pcol].to_numpy(),
            sup_pos_crit_b,
            sup_neg_ctrl,
            name,
        )
        sup_metrics_b.append(m.__dict__)

    # Evaluate DOT Sitting models (Criterion B: Asthma ID 12 + Obesity BMI>=30 ID 9, 16; and Criterion A: Asthma ID 12 only)
    dot_eval_models_b = [
        ("Virtual PPG Baseline (HR+HRV)", "ppg_comb_raw", "ppg_pct", dot_scores_blind),
        ("Isolation Forest (Unsupervised Baseline)", "iforest_raw", "iforest_pct", dot_scores_blind),
        ("fNIRS Stage 1: Hotelling's T^2 (Within-Subspace)", "fnirs_t2_raw", "fnirs_t2_pct", dot_scores_blind),
        ("fNIRS Stage 1: PCA SPE / Q-residual (Out-of-Subspace)", "fnirs_spe_raw", "fnirs_spe_pct", dot_scores_blind),
        ("fNIRS Stage 1: Cross-Modal Ridge (Heart->Brain)", "fnirs_cm_raw", "fnirs_cm_pct", dot_scores_blind),
        ("fNIRS Stage 1: Subspace Max (T2/SPE/CM)", "fnirs_subspace_pct", "fnirs_subspace_pct", dot_scores_blind),
        ("fNIRS A-Priori SPB Resonance Deficit (Blind LOOCV)", "score_apriori_spb_deficit_raw", "score_apriori_spb_deficit_pct", dot_scores_blind),
        ("fNIRS Stage 2 Clinical Triage (Fully Blind LOOCV)", "score_normative_combined", "score_normative_combined_pct", dot_scores_blind),
        ("fNIRS Stage 2 Clinical Triage (Normative Pre-Screened LOOCV)", "score_normative_combined", "score_normative_combined_pct", dot_scores_norm),
    ]

    dot_metrics_b = []
    for name, rcol, pcol, s_df in dot_eval_models_b:
        mb = evaluate_anomaly_detector(
            s_df[rcol].to_numpy(),
            s_df[pcol].to_numpy(),
            dot_pos_crit_b,
            dot_neg_ctrl,
            name,
        )
        dot_metrics_b.append(mb.__dict__)

    dot_eval_models_a = [
        ("Virtual PPG Baseline (HR+HRV)", "ppg_comb_raw", "ppg_pct", dot_scores_blind),
        ("Isolation Forest (Unsupervised Baseline)", "iforest_raw", "iforest_pct", dot_scores_blind),
        ("fNIRS Stage 1: Hotelling's T^2 (Within-Subspace)", "fnirs_t2_raw", "fnirs_t2_pct", dot_scores_blind),
        ("fNIRS Stage 1: PCA SPE / Q-residual (Out-of-Subspace)", "fnirs_spe_raw", "fnirs_spe_pct", dot_scores_blind),
        ("fNIRS Stage 1: Cross-Modal Ridge (Heart->Brain)", "fnirs_cm_raw", "fnirs_cm_pct", dot_scores_blind),
        ("fNIRS Stage 1: Subspace Max (T2/SPE/CM)", "fnirs_subspace_pct", "fnirs_subspace_pct", dot_scores_blind),
        ("fNIRS A-Priori SPB Resonance Deficit (Blind LOOCV)", "score_apriori_spb_deficit_raw", "score_apriori_spb_deficit_pct", dot_scores_blind),
        ("fNIRS Stage 2 Clinical Triage (Fully Blind LOOCV)", "score_normative_combined", "score_normative_combined_pct", dot_scores_blind),
        ("fNIRS Stage 2 Clinical Triage (Normative Pre-Screened LOOCV)", "score_normative_combined", "score_normative_combined_pct", dot_scores_norm_a),
    ]
    dot_metrics_a = []
    for name, rcol, pcol, s_df in dot_eval_models_a:
        ma = evaluate_anomaly_detector(
            s_df[rcol].to_numpy(),
            s_df[pcol].to_numpy(),
            dot_pos_crit_a,
            dot_neg_ctrl,
            name,
        )
        dot_metrics_a.append(ma.__dict__)

    # Sensitivity analysis across k = 1..5 PCA components on Supine (Fully Blind LOOCV)
    k_sensitivity = []
    idx_38 = int(df_sup.index[df_sup["ID"] == 38][0])
    for k in [1, 2, 3, 4, 5]:
        s_k = run_loocv_subspace_decomposition(
            df_sup,
            SUPINE_FEATURE_COLS,
            SUPINE_LOG_COLS,
            SUPINE_PPG_COLS,
            SUPINE_BRAIN_COLS,
            sup_ref_mask_a,
            n_components=k,
            blind_training=True,
        )
        s_k2 = compute_supine_stage2_scores(df_sup, s_k, sup_ref_mask_a, blind_training=True)
        auc_spe = float(roc_auc_score(sup_pos_crit_a, s_k["fnirs_spe_raw"]))
        auc_comb = float(roc_auc_score(sup_pos_crit_a, s_k2["score_normative_combined"]))
        k_sensitivity.append(
            {
                "k": k,
                "id38_t2_pct": round(float(s_k.loc[idx_38, "fnirs_t2_pct"]), 1),
                "id38_spe_pct": round(float(s_k.loc[idx_38, "fnirs_spe_pct"]), 1),
                "id38_combined_pct": round(float(s_k2.loc[idx_38, "score_normative_combined_pct"]), 1),
                "cohort_spe_auc": round(auc_spe, 3),
                "cohort_combined_auc": round(auc_comb, 3),
            }
        )

    # Partial-recording subjects evaluation:
    # 1. ID 59 (68 yo, antihypertensive medication, BS-only) evaluated on QC-passing BS-only features
    df_bs_only = df_sup_all[df_sup_all["has_bs"] & df_sup_all["bs_qc_ok"].astype(bool)].reset_index(drop=True)
    bs_cols = ["bs_hr_bpm", "bs_sdnn_ms", "bs_rmssd_ms", "bs_lr_mayer_corr", "bs_mayer_amp", "bs_mayer_resp_ratio"]
    bs_ref = ~df_bs_only["ID"].isin([38, 48, 49, 59, 60]).to_numpy()
    bs_loocv = run_loocv_subspace_decomposition(
        df_bs_only,
        bs_cols,
        ["bs_sdnn_ms", "bs_rmssd_ms", "bs_mayer_amp", "bs_mayer_resp_ratio"],
        ["bs_hr_bpm", "bs_sdnn_ms", "bs_rmssd_ms"],
        ["bs_lr_mayer_corr", "bs_mayer_amp", "bs_mayer_resp_ratio"],
        bs_ref,
        n_components=2,
        blind_training=False,
    )
    idx_59 = int(df_bs_only.index[df_bs_only["ID"] == 59][0])
    id59_bs_result = {
        "ID": 59,
        "Age": 68,
        "Medication": "od nadcisnienia",
        "bs_mayer_resp_ratio_z": round(float(bs_loocv.loc[idx_59, "z_bs_mayer_resp_ratio"]), 2),
        "bs_mayer_amp_z": round(float(bs_loocv.loc[idx_59, "z_bs_mayer_amp"]), 2),
        "fnirs_subspace_pct": round(float(bs_loocv.loc[idx_59, "fnirs_subspace_pct"]), 1),
    }

    # 2. ID 41 (23 yo, "ozempikowaty", BMI 34.54, BH-only) evaluated on QC-passing BH-only features
    df_bh_only = df_sup_all[df_sup_all["has_bh"] & df_sup_all["bh_qc_ok"].astype(bool)].reset_index(drop=True)
    bh_cols = ["bh_dur_max", "bh_ssr_cortical_od850", "bh_deep_od850", "bh_hbr_hbo_ratio", "bh_spatial_cv", "bh_hr_delta"]
    bh_ref = ~df_bh_only["ID"].isin([38, 41, 48, 49, 60]).to_numpy()
    bh_loocv = run_loocv_subspace_decomposition(
        df_bh_only,
        bh_cols,
        ["bh_dur_max", "bh_ssr_cortical_od850", "bh_deep_od850", "bh_hbr_hbo_ratio", "bh_spatial_cv"],
        ["bh_dur_max", "bh_hr_delta"],
        ["bh_ssr_cortical_od850", "bh_deep_od850", "bh_hbr_hbo_ratio", "bh_spatial_cv"],
        bh_ref,
        n_components=2,
        blind_training=False,
    )
    idx_41 = int(df_bh_only.index[df_bh_only["ID"] == 41][0])
    id41_bh_result = {
        "ID": 41,
        "Age": 23,
        "BMI": 34.54,
        "Medication": "ozempikowaty",
        "bh_ssr_cortical_z": round(float(bh_loocv.loc[idx_41, "z_bh_ssr_cortical_od850"]), 2),
        "bh_hbr_hbo_ratio_z": round(float(bh_loocv.loc[idx_41, "z_bh_hbr_hbo_ratio"]), 2),
        "fnirs_t2_pct": round(float(bh_loocv.loc[idx_41, "fnirs_t2_pct"]), 1),
        "fnirs_spe_pct": round(float(bh_loocv.loc[idx_41, "fnirs_spe_pct"]), 1),
    }

    # Confounder Audit: Young-Only Cohort (Ages 18-42, N=45) testing ID 38 (Nebilet, age 29) without senior age confounding
    young_mask = (df_sup["Age"] <= 42).to_numpy()
    df_young = df_sup[young_mask].reset_index(drop=True)
    y_pos_38 = (df_young["ID"] == 38).to_numpy()
    y_ref_38 = ~y_pos_38
    young_loocv = run_loocv_subspace_decomposition(
        df_young,
        SUPINE_FEATURE_COLS,
        SUPINE_LOG_COLS,
        SUPINE_PPG_COLS,
        SUPINE_BRAIN_COLS,
        y_ref_38,
        n_components=3,
        blind_training=True,
    )
    young_scores = compute_supine_stage2_scores(df_young, young_loocv, y_ref_38, blind_training=True)
    idx_38_y = int(df_young.index[df_young["ID"] == 38][0])
    young_id38_audit = {
        "n_young": len(df_young),
        "id38_ppg_pct": round(float(young_scores.loc[idx_38_y, "ppg_pct"]), 1),
        "id38_fnirs_t2_pct": round(float(young_scores.loc[idx_38_y, "fnirs_t2_pct"]), 1),
        "id38_fnirs_spe_pct": round(float(young_scores.loc[idx_38_y, "fnirs_spe_pct"]), 1),
        "id38_fnirs_cm_pct": round(float(young_scores.loc[idx_38_y, "fnirs_cm_pct"]), 1),
        "id38_combined_pct": round(float(young_scores.loc[idx_38_y, "score_normative_combined_pct"]), 1),
        "id38_rank_spe": int(np.sum(young_scores["fnirs_spe_raw"] >= young_scores.loc[idx_38_y, "fnirs_spe_raw"])),
        "id38_rank_combined": int(
            np.sum(young_scores["score_normative_combined"] >= young_scores.loc[idx_38_y, "score_normative_combined"])
        ),
    }

    # Generate visual summary plots
    generate_summary_figures(sup_merged, dot_merged, res_dir)

    summary = {
        "supine_qc_funnel": supine_qc_funnel,
        "supine_criterion_a_metrics": sup_metrics_a,
        "supine_criterion_b_metrics": sup_metrics_b,
        "dot_criterion_b_metrics": dot_metrics_b,
        "dot_criterion_a_metrics": dot_metrics_a,
        "pca_k_sensitivity": k_sensitivity,
        "young_cohort_id38_audit": young_id38_audit,
        "partial_recording_id59_bs": id59_bs_result,
        "partial_recording_id41_bh": id41_bh_result,
    }
    with open(res_dir / "verification_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def generate_summary_figures(sup_df: pd.DataFrame, dot_df: pd.DataFrame, res_dir: Path) -> None:
    """Generates diagnostic figures comparing Stage 1 Subspace Decomposition, PPG vs fNIRS, and Stage 2 Triage."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Panel A: Supine Stage 1 Subspace Decomposition (Hotelling's T^2 vs PCA SPE / Q-residual)
    ax = axes[0, 0]
    is_neb = sup_df["ID"] == 38
    is_hyp = sup_df["ID"].isin([48, 49, 60])
    is_neg_ctrl = sup_df["ID"].isin([39, 43])
    is_hyper_healthy = sup_df["ID"].isin([1, 13, 16, 56])
    is_other = ~(is_neb | is_hyp | is_neg_ctrl | is_hyper_healthy)

    ax.scatter(
        sup_df.loc[is_other, "fnirs_t2_pct"],
        sup_df.loc[is_other, "fnirs_spe_pct"],
        c="#7f8c8d",
        alpha=0.65,
        s=45,
        label="Zdrowa kohorta referencyjna",
    )
    ax.scatter(
        sup_df.loc[is_hyper_healthy, "fnirs_t2_pct"],
        sup_df.loc[is_hyper_healthy, "fnirs_spe_pct"],
        c="#f39c12",
        s=75,
        marker="^",
        label="Zdrowi hiper-reaktywni (długi BH / wysokie HRV)",
    )
    ax.scatter(
        sup_df.loc[is_neg_ctrl, "fnirs_t2_pct"],
        sup_df.loc[is_neg_ctrl, "fnirs_spe_pct"],
        c="#2ecc71",
        s=85,
        marker="s",
        label="Negatywna kontrola lekowa (Izotec ID 39, 43)",
    )
    ax.scatter(
        sup_df.loc[is_hyp, "fnirs_t2_pct"],
        sup_df.loc[is_hyp, "fnirs_spe_pct"],
        c="#e74c3c",
        s=95,
        marker="o",
        label="Seniorzy: Nadciśnienie (ID 48, 49, 60)",
    )
    ax.scatter(
        sup_df.loc[is_neb, "fnirs_t2_pct"],
        sup_df.loc[is_neb, "fnirs_spe_pct"],
        c="#8e44ad",
        s=150,
        marker="*",
        label="Beta-bloker: Nebilet (ID 38, 29 lat)",
    )
    for _, r in sup_df[is_neb | is_hyp | is_neg_ctrl | is_hyper_healthy].iterrows():
        ax.annotate(
            f"ID {int(r['ID'])}",
            (r["fnirs_t2_pct"], r["fnirs_spe_pct"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8.5,
        )
    ax.axhline(90, color="#c0392b", linestyle="--", alpha=0.5)
    ax.axvline(90, color="#2980b9", linestyle="--", alpha=0.5)
    ax.set_xlabel("Hotelling's T² Percentile (Wewnątrz podprzestrzeni PCA) [%]")
    ax.set_ylabel("PCA SPE / Q-residual Percentile (Poza podprzestrzenią) [%]")
    ax.set_title("A. Zbiór 1 (Supine): Dekompozycja Podprzestrzeni PCA (T² vs SPE)")
    ax.legend(loc="lower left", fontsize=7.5)
    ax.grid(True, alpha=0.25)

    # Panel B: Supine Virtual PPG vs fNIRS Stage 2 Clinical Triage
    ax = axes[0, 1]
    ax.scatter(sup_df.loc[is_other, "ppg_pct"], sup_df.loc[is_other, "score_normative_combined_pct"], c="#7f8c8d", alpha=0.65, s=45)
    ax.scatter(sup_df.loc[is_hyper_healthy, "ppg_pct"], sup_df.loc[is_hyper_healthy, "score_normative_combined_pct"], c="#f39c12", s=75, marker="^")
    ax.scatter(sup_df.loc[is_neg_ctrl, "ppg_pct"], sup_df.loc[is_neg_ctrl, "score_normative_combined_pct"], c="#2ecc71", s=85, marker="s")
    ax.scatter(sup_df.loc[is_hyp, "ppg_pct"], sup_df.loc[is_hyp, "score_normative_combined_pct"], c="#e74c3c", s=95, marker="o")
    ax.scatter(sup_df.loc[is_neb, "ppg_pct"], sup_df.loc[is_neb, "score_normative_combined_pct"], c="#8e44ad", s=150, marker="*")
    for _, r in sup_df[is_neb | is_hyp | is_neg_ctrl | is_hyper_healthy].iterrows():
        ax.annotate(
            f"ID {int(r['ID'])}",
            (r["ppg_pct"], r["score_normative_combined_pct"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8.5,
        )
    ax.axhline(90, color="#c0392b", linestyle="--", alpha=0.6, label="Próg 90% (fNIRS Triage)")
    ax.axvline(90, color="#2980b9", linestyle="--", alpha=0.6, label="Próg 90% (Virtual PPG)")
    ax.set_xlabel("Virtual PPG Baseline Percentile (Tętno + HRV + Czas BH) [%]")
    ax.set_ylabel("fNIRS Normative Clinical Triage Percentile (LOOCV) [%]")
    ax.set_title("B. Zbiór 1 (Supine): Virtual PPG vs Wielokanałowy fNIRS")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.25)

    # Panel C: DOT Sitting 0.1 Hz Paced Breathing Resonance vs Evening/Afternoon BH Ratio
    ax = axes[1, 0]
    is_asthma = dot_df["ID"] == 12
    is_obese = dot_df["ID"].isin([9, 16])
    is_cornax = dot_df["ID"] == 17
    is_dot_healthy = ~(is_asthma | is_obese | is_cornax)

    ax.scatter(
        dot_df.loc[is_dot_healthy, "spb1_gain_01hz"],
        dot_df.loc[is_dot_healthy, "bh_evening_morning_ratio"],
        c="#7f8c8d",
        s=60,
        alpha=0.75,
        label="Zdrowa kohorta DOT (BMI < 30)",
    )
    ax.scatter(
        dot_df.loc[is_cornax, "spb1_gain_01hz"],
        dot_df.loc[is_cornax, "bh_evening_morning_ratio"],
        c="#2ecc71",
        s=95,
        marker="s",
        label="Negatywna kontrola (Cornax ID 17)",
    )
    ax.scatter(
        dot_df.loc[is_obese, "spb1_gain_01hz"],
        dot_df.loc[is_obese, "bh_evening_morning_ratio"],
        c="#e67e22",
        s=105,
        marker="D",
        label="Otyłość I st. BMI ≥ 30 (ID 9, 16)",
    )
    ax.scatter(
        dot_df.loc[is_asthma, "spb1_gain_01hz"],
        dot_df.loc[is_asthma, "bh_evening_morning_ratio"],
        c="#e74c3c",
        s=140,
        marker="*",
        label="Astma oskrzelowa (ID 12)",
    )
    for _, r in dot_df.iterrows():
        ax.annotate(
            f"ID {int(r['ID'])}",
            (r["spb1_gain_01hz"], r["bh_evening_morning_ratio"]),
            xytext=(5, 3),
            textcoords="offset points",
            fontsize=8.5,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Wzmocnienie rezonansu Mayera 0.1 Hz w SPB1 względem RS1 (skala log)")
    ax.set_ylabel("Stosunek odpowiedzi hipoksyjnej Wieczór / Popołudnie (BH2 / BH1)")
    ax.set_title("C. Zbiór 2 (DOT Sitting): Rezonans Oddechowy 0.1 Hz vs Koszt Hipoksyjny")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25)

    # Panel D: DOT Sitting Virtual PPG vs fNIRS Stage 2 Clinical Triage
    ax = axes[1, 1]
    ax.scatter(dot_df.loc[is_dot_healthy, "ppg_pct"], dot_df.loc[is_dot_healthy, "score_normative_combined_pct"], c="#7f8c8d", s=60, alpha=0.75)
    ax.scatter(dot_df.loc[is_cornax, "ppg_pct"], dot_df.loc[is_cornax, "score_normative_combined_pct"], c="#2ecc71", s=95, marker="s")
    ax.scatter(dot_df.loc[is_obese, "ppg_pct"], dot_df.loc[is_obese, "score_normative_combined_pct"], c="#e67e22", s=105, marker="D")
    ax.scatter(dot_df.loc[is_asthma, "ppg_pct"], dot_df.loc[is_asthma, "score_normative_combined_pct"], c="#e74c3c", s=140, marker="*")
    for _, r in dot_df.iterrows():
        ax.annotate(
            f"ID {int(r['ID'])}",
            (r["ppg_pct"], r["score_normative_combined_pct"]),
            xytext=(5, 3),
            textcoords="offset points",
            fontsize=8.5,
        )
    ax.axhline(90, color="#c0392b", linestyle="--", alpha=0.6)
    ax.axvline(90, color="#2980b9", linestyle="--", alpha=0.6)
    ax.set_xlabel("Virtual PPG Baseline Percentile [%]")
    ax.set_ylabel("fNIRS Normative Clinical Triage Percentile (LOOCV) [%]")
    ax.set_title("D. Zbiór 2 (DOT Sitting): Virtual PPG vs Wielokanałowy fNIRS")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    fig.savefig(res_dir / "anomaly_detection_verification.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    summary = run_all_verifications(Path(__file__).resolve().parent.parent)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
