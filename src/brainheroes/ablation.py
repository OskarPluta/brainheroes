"""Ablations around the earlier 50/50 variant and the single-score baselines.

The earlier variant averaged a Stage 1 subspace percentile (max of T^2, SPE, ridge) with
the deficit percentile. This module keeps it for comparison and evaluates every
intermediate score (virtual PPG, Isolation Forest, T^2, SPE, ridge, deficit, 50/50) in
the fully blind setting and with a label-screened reference group. It also covers PCA
k-sensitivity, the age-dependent DPF, the young-only audit of ID 38 and the two
participants with a single session. Writes results/verification_summary.json,
results/*_loocv_results.csv and results/anomaly_detection_verification.png.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from .constants import (
    ALARM_THRESHOLD,
    DOT_BRAIN_COLS,
    DOT_CRIT_A_IDS,
    DOT_CRIT_B_IDS,
    DOT_FEATURE_COLS,
    DOT_LOG_COLS,
    DOT_NEG_CONTROL_IDS,
    DOT_PPG_COLS,
    RESULTS_DIR,
    SUPINE_BRAIN_COLS,
    SUPINE_CRIT_A_IDS,
    SUPINE_CRIT_B_IDS,
    SUPINE_FEATURE_COLS,
    SUPINE_LOG_COLS,
    SUPINE_NEG_CONTROL_IDS,
    SUPINE_PPG_COLS,
)
from .features import load_dot, load_supine, qc_funnel
from .normative import DeficitFn, LoocvResult, dot_deficit, percentile_of, run_loocv, supine_deficit


@dataclass
class EvaluationMetrics:
    model_name: str
    roc_auc: float
    pr_auc: float
    recall_at_90: float
    precision_at_90: float
    f1_at_90: float
    fp_at_90: int
    tp_at_90: int
    n_pos: int
    n_neg: int
    mannwhitney_p: float
    neg_control_fp: int


def evaluate_detector(
    scores: np.ndarray,
    pct_scores: np.ndarray,
    y_true: np.ndarray,
    neg_control: np.ndarray,
    model_name: str,
    pct_threshold: float = ALARM_THRESHOLD,
) -> EvaluationMetrics:
    """Threshold-free metrics on `scores` and thresholded metrics on `pct_scores`."""
    y = np.asarray(y_true, dtype=int)
    _, mw_p = stats.mannwhitneyu(scores[y == 1], scores[y == 0], alternative="greater")
    preds = (pct_scores >= pct_threshold).astype(int)
    return EvaluationMetrics(
        model_name=model_name,
        roc_auc=float(roc_auc_score(y, scores)),
        pr_auc=float(average_precision_score(y, scores)),
        recall_at_90=float(recall_score(y, preds, zero_division=0)),
        precision_at_90=float(precision_score(y, preds, zero_division=0)),
        f1_at_90=float(f1_score(y, preds, zero_division=0)),
        fp_at_90=int(np.sum((preds == 1) & (y == 0))),
        tp_at_90=int(np.sum((preds == 1) & (y == 1))),
        n_pos=int(np.sum(y == 1)),
        n_neg=int(np.sum(y == 0)),
        mannwhitney_p=float(mw_p),
        neg_control_fp=int(np.sum(preds[neg_control] == 1)) if np.any(neg_control) else 0,
    )


def combined_5050(loocv: LoocvResult, deficit_fn: DeficitFn) -> pd.DataFrame:
    """Add the deficit and the 50/50 (subspace + deficit) scores, both calibrated out of fold."""
    res = loocv.scores.copy()
    idx = loocv.col_idx
    deficit_raw = deficit_fn(loocv.z, idx)
    subspace = res["fnirs_subspace_pct"].to_numpy()
    deficit_pct, combined_raw, combined_pct = (np.zeros(len(res)) for _ in range(3))
    for i, ref in enumerate(loocv.references):
        deficit_ref = deficit_fn(ref.z, idx)
        deficit_pct[i] = percentile_of(deficit_raw[i], deficit_ref)
        combined_ref = 0.5 * percentile_of(ref.subspace_pct, ref.subspace_pct) + 0.5 * percentile_of(deficit_ref, deficit_ref)
        combined_raw[i] = 0.5 * subspace[i] + 0.5 * deficit_pct[i]
        combined_pct[i] = percentile_of(combined_raw[i], combined_ref)
    res["deficit_raw"] = deficit_raw
    res["deficit_pct"] = deficit_pct
    res["combined_raw"] = combined_raw
    res["combined_pct"] = combined_pct
    return res


def _supine(df: pd.DataFrame, ref_mask: np.ndarray | None = None, k: int = 3) -> pd.DataFrame:
    loocv = run_loocv(
        df, SUPINE_FEATURE_COLS, SUPINE_LOG_COLS, SUPINE_PPG_COLS, SUPINE_BRAIN_COLS,
        ref_mask=ref_mask, n_components=k, isolation_forest=True,
    )  # fmt: skip
    return combined_5050(loocv, supine_deficit)


def _dot(df: pd.DataFrame, ref_mask: np.ndarray | None = None) -> pd.DataFrame:
    loocv = run_loocv(df, DOT_FEATURE_COLS, DOT_LOG_COLS, DOT_PPG_COLS, DOT_BRAIN_COLS, ref_mask=ref_mask, isolation_forest=True)
    return combined_5050(loocv, dot_deficit)


# (model name, raw-score column, percentile column) of every evaluated score.
SINGLE_SCORES = [
    ("Isolation Forest (Unsupervised Baseline)", "iforest_raw", "iforest_pct"),
    ("fNIRS Stage 1: Hotelling's T^2 (Within-Subspace)", "fnirs_t2_raw", "fnirs_t2_pct"),
    ("fNIRS Stage 1: PCA SPE / Q-residual (Out-of-Subspace)", "fnirs_spe_raw", "fnirs_spe_pct"),
    ("fNIRS Stage 1: Cross-Modal Ridge (Heart->Brain)", "fnirs_cm_raw", "fnirs_cm_pct"),
    ("fNIRS Stage 1: Subspace Max (T2/SPE/CM)", "fnirs_subspace_pct", "fnirs_subspace_pct"),
]
COMBINED = ("combined_raw", "combined_pct")
DEFICIT = ("deficit_raw", "deficit_pct")


def _evaluate(models: list[tuple[str, str, str, pd.DataFrame]], y: np.ndarray, neg: np.ndarray) -> list[dict]:
    return [asdict(evaluate_detector(s[raw].to_numpy(), s[pct].to_numpy(), y, neg, name)) for name, raw, pct, s in models]


def _id_row(df: pd.DataFrame, sid: int) -> int:
    return int(df.index[df["ID"] == sid][0])


def run(results_dir: Path = RESULTS_DIR) -> dict[str, object]:
    sup_all, sup = load_supine(results_dir)
    _, sup_age = load_supine(results_dir, age_dpf=True)
    dot = load_dot(results_dir)

    sup_a = sup["ID"].isin(SUPINE_CRIT_A_IDS).to_numpy()
    sup_b = sup["ID"].isin(SUPINE_CRIT_B_IDS).to_numpy()
    sup_neg = sup["ID"].isin(SUPINE_NEG_CONTROL_IDS).to_numpy()
    dot_a = dot["ID"].isin(DOT_CRIT_A_IDS).to_numpy()
    dot_b = dot["ID"].isin(DOT_CRIT_B_IDS).to_numpy()
    dot_neg = dot["ID"].isin(DOT_NEG_CONTROL_IDS).to_numpy()

    sup_blind = _supine(sup)
    sup_norm_a = _supine(sup, ref_mask=~sup_a)
    sup_norm_b = _supine(sup, ref_mask=~sup_b)
    sup_age_dpf = _supine(sup_age)
    dot_blind = _dot(dot)
    dot_norm_a = _dot(dot, ref_mask=~dot_a)
    dot_norm_b = _dot(dot, ref_mask=~dot_b)

    sup_merged = pd.concat([sup, sup_blind.drop(columns=["ID"])], axis=1)
    dot_merged = pd.concat([dot, dot_blind.drop(columns=["ID"])], axis=1)
    sup_merged.to_csv(results_dir / "supine_loocv_results.csv", index=False)
    dot_merged.to_csv(results_dir / "dot_loocv_results.csv", index=False)

    triage = "fNIRS Stage 2 Clinical Triage"
    sup_models_a = [
        ("Virtual PPG Baseline (HR+HRV+BH)", "ppg_comb_raw", "ppg_pct", sup_blind),
        *[(name, raw, pct, sup_blind) for name, raw, pct in SINGLE_SCORES],
        ("fNIRS A-Priori Physiological Deficit (Blind LOOCV)", *DEFICIT, sup_blind),
        (f"{triage} (Fully Blind LOOCV)", *COMBINED, sup_blind),
        (f"{triage} (Normative Pre-Screened LOOCV)", *COMBINED, sup_norm_a),
        (f"{triage} (Age-DPF Sensitivity, Blind)", *COMBINED, sup_age_dpf),
    ]
    sup_models_b = [
        ("Virtual PPG Baseline (Blind LOOCV)", "ppg_comb_raw", "ppg_pct", sup_blind),
        ("fNIRS Stage 1: PCA SPE (Blind LOOCV)", "fnirs_spe_raw", "fnirs_spe_pct", sup_blind),
        (f"{triage} (Fully Blind LOOCV)", *COMBINED, sup_blind),
        (f"{triage} (Normative Pre-Screened LOOCV)", *COMBINED, sup_norm_b),
    ]

    def dot_models(norm: pd.DataFrame) -> list[tuple[str, str, str, pd.DataFrame]]:
        return [
            ("Virtual PPG Baseline (HR+HRV)", "ppg_comb_raw", "ppg_pct", dot_blind),
            *[(name, raw, pct, dot_blind) for name, raw, pct in SINGLE_SCORES],
            ("fNIRS A-Priori SPB Resonance Deficit (Blind LOOCV)", *DEFICIT, dot_blind),
            (f"{triage} (Fully Blind LOOCV)", *COMBINED, dot_blind),
            (f"{triage} (Normative Pre-Screened LOOCV)", *COMBINED, norm),
        ]

    # PCA k-sensitivity (fully blind).
    i38 = _id_row(sup, 38)
    k_sensitivity = []
    for k in [1, 2, 3, 4, 5]:
        s = _supine(sup, k=k)
        k_sensitivity.append(
            {
                "k": k,
                "id38_t2_pct": round(float(s.loc[i38, "fnirs_t2_pct"]), 1),
                "id38_spe_pct": round(float(s.loc[i38, "fnirs_spe_pct"]), 1),
                "id38_combined_pct": round(float(s.loc[i38, "combined_pct"]), 1),
                "cohort_spe_auc": round(float(roc_auc_score(sup_a, s["fnirs_spe_raw"])), 3),
                "cohort_combined_auc": round(float(roc_auc_score(sup_a, s["combined_raw"])), 3),
            }
        )

    # ID 38 within the young cohort only, away from the 70+ group.
    young = sup[sup["Age"] <= 42].reset_index(drop=True)
    young_scores = _supine(young)
    y38 = young_scores.loc[_id_row(young, 38)]
    young_audit = {
        "n_young": len(young),
        "id38_ppg_pct": round(float(y38["ppg_pct"]), 1),
        "id38_fnirs_t2_pct": round(float(y38["fnirs_t2_pct"]), 1),
        "id38_fnirs_spe_pct": round(float(y38["fnirs_spe_pct"]), 1),
        "id38_fnirs_cm_pct": round(float(y38["fnirs_cm_pct"]), 1),
        "id38_combined_pct": round(float(y38["combined_pct"]), 1),
        "id38_rank_spe": int(np.sum(young_scores["fnirs_spe_raw"] >= y38["fnirs_spe_raw"])),
        "id38_rank_combined": int(np.sum(young_scores["combined_raw"] >= y38["combined_raw"])),
    }

    generate_figure(sup_merged, dot_merged, results_dir)

    summary = {
        "supine_qc_funnel": qc_funnel(sup_all),
        "supine_criterion_a_metrics": _evaluate(sup_models_a, sup_a, sup_neg),
        "supine_criterion_b_metrics": _evaluate(sup_models_b, sup_b, sup_neg),
        "dot_criterion_b_metrics": _evaluate(dot_models(dot_norm_b), dot_b, dot_neg),
        "dot_criterion_a_metrics": _evaluate(dot_models(dot_norm_a), dot_a, dot_neg),
        "pca_k_sensitivity": k_sensitivity,
        "young_cohort_id38_audit": young_audit,
        **_single_session_audits(sup_all),
    }
    (results_dir / "verification_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _single_session_audits(sup_all: pd.DataFrame) -> dict[str, dict]:
    """ID 59 has only a baseline and ID 41 only a breath-hold session; score each on what it has.

    These use a label-screened reference group (all criterion-A people and the subject
    itself excluded), so they are normative, not blind.
    """
    bs = sup_all[sup_all["has_bs"] & sup_all["bs_qc_ok"]].reset_index(drop=True)
    bs_loocv = run_loocv(
        bs,
        ["bs_hr_bpm", "bs_sdnn_ms", "bs_rmssd_ms", "bs_lr_mayer_corr", "bs_mayer_amp", "bs_mayer_resp_ratio"],
        ["bs_sdnn_ms", "bs_rmssd_ms", "bs_mayer_amp", "bs_mayer_resp_ratio"],
        ["bs_hr_bpm", "bs_sdnn_ms", "bs_rmssd_ms"],
        ["bs_lr_mayer_corr", "bs_mayer_amp", "bs_mayer_resp_ratio"],
        ref_mask=~bs["ID"].isin([*SUPINE_CRIT_A_IDS, 59]).to_numpy(),
        n_components=2,
    ).scores.loc[_id_row(bs, 59)]

    bh = sup_all[sup_all["has_bh"] & sup_all["bh_qc_ok"]].reset_index(drop=True)
    bh_loocv = run_loocv(
        bh,
        ["bh_dur_max", "bh_ssr_cortical_od850", "bh_deep_od850", "bh_hbr_hbo_ratio", "bh_spatial_cv", "bh_hr_delta"],
        ["bh_dur_max", "bh_ssr_cortical_od850", "bh_deep_od850", "bh_hbr_hbo_ratio", "bh_spatial_cv"],
        ["bh_dur_max", "bh_hr_delta"],
        ["bh_ssr_cortical_od850", "bh_deep_od850", "bh_hbr_hbo_ratio", "bh_spatial_cv"],
        ref_mask=~bh["ID"].isin([*SUPINE_CRIT_A_IDS, 41]).to_numpy(),
        n_components=2,
    ).scores.loc[_id_row(bh, 41)]

    meta = sup_all.set_index("ID")
    return {
        "partial_recording_id59_bs": {
            "ID": 59,
            "Age": int(meta.loc[59, "Age"]),
            "Medication": str(meta.loc[59, "Medication"]),
            "bs_mayer_resp_ratio_z": round(float(bs_loocv["z_bs_mayer_resp_ratio"]), 2),
            "bs_mayer_amp_z": round(float(bs_loocv["z_bs_mayer_amp"]), 2),
            "fnirs_subspace_pct": round(float(bs_loocv["fnirs_subspace_pct"]), 1),
        },
        "partial_recording_id41_bh": {
            "ID": 41,
            "Age": int(meta.loc[41, "Age"]),
            "BMI": float(meta.loc[41, "BMI"]),
            "Medication": str(meta.loc[41, "Medication"]),
            "bh_ssr_cortical_z": round(float(bh_loocv["z_bh_ssr_cortical_od850"]), 2),
            "bh_hbr_hbo_ratio_z": round(float(bh_loocv["z_bh_hbr_hbo_ratio"]), 2),
            "fnirs_t2_pct": round(float(bh_loocv["fnirs_t2_pct"]), 1),
            "fnirs_spe_pct": round(float(bh_loocv["fnirs_spe_pct"]), 1),
        },
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

# (IDs, colour, size, marker, legend label); remaining subjects are drawn grey.
SUPINE_GROUPS = [
    ([1, 13, 16, 56], "#f39c12", 75, "^", "Zdrowi hiper-reaktywni (długi BH / wysokie HRV)"),
    (SUPINE_NEG_CONTROL_IDS, "#2ecc71", 85, "s", "Negatywna kontrola lekowa (Izotec ID 39, 43)"),
    ([48, 49, 60], "#e74c3c", 95, "o", "Seniorzy: Nadciśnienie (ID 48, 49, 60)"),
    ([38], "#8e44ad", 150, "*", "Beta-bloker: Nebilet (ID 38, 29 lat)"),
]
DOT_GROUPS = [
    (DOT_NEG_CONTROL_IDS, "#2ecc71", 95, "s", "Negatywna kontrola (Cornax ID 17)"),
    ([9, 16], "#e67e22", 105, "D", "Otyłość I st. BMI ≥ 30 (ID 9, 16)"),
    ([12], "#e74c3c", 140, "*", "Astma oskrzelowa (ID 12)"),
]


def _scatter(ax, df, x, y, groups, rest_label, rest_size, annotate_all, legend):
    grouped = df["ID"].isin([i for g in groups for i in g[0]])
    ax.scatter(
        df.loc[~grouped, x], df.loc[~grouped, y], c="#7f8c8d", alpha=0.7, s=rest_size, label=rest_label if legend else None
    )
    for ids, color, size, marker, label in groups:
        sel = df["ID"].isin(ids)
        ax.scatter(df.loc[sel, x], df.loc[sel, y], c=color, s=size, marker=marker, label=label if legend else None)
    for _, r in (df if annotate_all else df[grouped]).iterrows():
        ax.annotate(f"ID {int(r['ID'])}", (r[x], r[y]), xytext=(5, 4), textcoords="offset points", fontsize=8.5)
    ax.grid(True, alpha=0.25)


def generate_figure(sup: pd.DataFrame, dot: pd.DataFrame, out_dir: Path) -> None:
    """2x2 overview: supine T^2 vs SPE, PPG vs 50/50 for both cohorts, DOT resonance vs BH ratio."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    ax = axes[0, 0]
    _scatter(ax, sup, "fnirs_t2_pct", "fnirs_spe_pct", SUPINE_GROUPS, "Zdrowa kohorta referencyjna", 45, False, True)
    ax.axhline(90, color="#c0392b", linestyle="--", alpha=0.5)
    ax.axvline(90, color="#2980b9", linestyle="--", alpha=0.5)
    ax.set(xlabel="Hotelling's T² Percentile (Wewnątrz podprzestrzeni PCA) [%]",
           ylabel="PCA SPE / Q-residual Percentile (Poza podprzestrzenią) [%]",
           title="A. Zbiór 1 (Supine): Dekompozycja Podprzestrzeni PCA (T² vs SPE)")  # fmt: skip
    ax.legend(loc="lower left", fontsize=7.5)

    for ax, df, groups, size, annotate_all, title in [
        (axes[0, 1], sup, SUPINE_GROUPS, 45, False, "B. Zbiór 1 (Supine): Virtual PPG vs Wielokanałowy fNIRS"),
        (axes[1, 1], dot, DOT_GROUPS, 60, True, "D. Zbiór 2 (DOT Sitting): Virtual PPG vs Wielokanałowy fNIRS"),
    ]:
        _scatter(ax, df, "ppg_pct", "combined_pct", groups, None, size, annotate_all, False)
        ax.axhline(90, color="#c0392b", linestyle="--", alpha=0.6, label="Próg 90% (fNIRS Triage)")
        ax.axvline(90, color="#2980b9", linestyle="--", alpha=0.6, label="Próg 90% (Virtual PPG)")
        ax.set(
            xlabel="Virtual PPG Baseline Percentile [%]",
            ylabel="fNIRS Normative Clinical Triage Percentile (LOOCV) [%]",
            title=title,
        )
    axes[0, 1].legend(loc="lower right", fontsize=8)

    ax = axes[1, 0]
    _scatter(ax, dot, "spb1_gain_01hz", "bh_evening_morning_ratio", DOT_GROUPS, "Zdrowa kohorta DOT (BMI < 30)", 60, True, True)
    ax.set_xscale("log")
    ax.set(xlabel="Wzmocnienie rezonansu Mayera 0.1 Hz w SPB1 względem RS1 (skala log)",
           ylabel="Stosunek odpowiedzi hipoksyjnej Wieczór / Popołudnie (BH2 / BH1)",
           title="C. Zbiór 2 (DOT Sitting): Rezonans Oddechowy 0.1 Hz vs Koszt Hipoksyjny")  # fmt: skip
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_dir / "anomaly_detection_verification.png", dpi=180)
    plt.close(fig)
