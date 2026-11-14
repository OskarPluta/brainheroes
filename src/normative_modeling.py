"""Normative Unsupervised Anomaly Detection (LOOCV): PCA T^2, SPE (Q-residual), Cross-Modal Ridge, and Clinical Triage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import RobustScaler


@dataclass
class EvaluationMetrics:
    """Summary metrics for an anomaly detection score against objective clinical labels."""

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


def symmetrize_features(df: pd.DataFrame, cols: Sequence[str], log_cols: Sequence[str]) -> np.ndarray:
    """Applies log10(max(x, eps)) transform to skewed positive ratio/amplitude features."""
    mat = np.zeros((len(df), len(cols)), dtype=np.float64)
    log_set = set(log_cols)
    for j, c in enumerate(cols):
        vals = df[c].to_numpy(dtype=np.float64, copy=True)
        if c in log_set:
            vals = np.log10(np.maximum(vals, 1e-4))
        mat[:, j] = vals
    return mat


def empirical_percentile_oof(val: float, ref_dist: np.ndarray) -> float:
    """Computes out-of-fold empirical percentile (0..100) of val against ref_dist."""
    return float(np.mean(ref_dist <= val) * 100.0)


def _robust_scale_fit_transform(X_tr: np.ndarray, X_te: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fits median/IQR RobustScaler on X_tr and transforms both X_tr and X_te."""
    med = np.median(X_tr, axis=0)
    q25, q75 = np.percentile(X_tr, [25.0, 75.0], axis=0)
    iqr = np.where((q75 - q25) > 1e-12, q75 - q25, 1.0)
    return (X_tr - med) / iqr, (X_te - med) / iqr


def _fit_eval_subspace_single(
    X_f_tr: np.ndarray,
    X_f_te: np.ndarray,
    X_p_tr: np.ndarray,
    X_p_te: np.ndarray,
    X_b_tr: np.ndarray,
    X_b_te: np.ndarray,
    k_full: int,
    k_ppg: int,
    ridge_alpha: float,
) -> tuple[np.ndarray, float, float, float, float, float, float, float]:
    """
    Fits RobustScaler, PCA, and Cross-Modal Ridge exclusively on training fold arrays
    and evaluates out-of-sample statistics on a single held-out test vector.
    """
    Z_tr, z_te = _robust_scale_fit_transform(X_f_tr, X_f_te)

    mean_z = np.mean(Z_tr, axis=0)
    Zc_tr = Z_tr - mean_z
    zc_te = z_te - mean_z
    _, s_f, vt_f = np.linalg.svd(Zc_tr, full_matrices=False)
    V_k = vt_f[:k_full].T
    lambdas = np.maximum((s_f[:k_full] ** 2) / max(len(Z_tr) - 1, 1), 1e-8)

    t_te = zc_te @ V_k
    t2_te = float(np.sum((t_te**2) / lambdas))
    z_te_hat = mean_z + t_te @ V_k.T
    spe_te = float(np.sum((z_te - z_te_hat) ** 2))

    Zp_tr, zp_te = _robust_scale_fit_transform(X_p_tr, X_p_te)
    Zb_tr, zb_te = _robust_scale_fit_transform(X_b_tr, X_b_te)

    mean_zp = np.mean(Zp_tr, axis=0)
    mean_zb = np.mean(Zb_tr, axis=0)
    Zpc_tr = Zp_tr - mean_zp
    Zbc_tr = Zb_tr - mean_zb
    p_dim = Zpc_tr.shape[1]
    W_ridge = np.linalg.solve(Zpc_tr.T @ Zpc_tr + ridge_alpha * np.eye(p_dim), Zpc_tr.T @ Zbc_tr)
    zb_te_pred = mean_zb + (zp_te - mean_zp) @ W_ridge
    cm_te = float(np.mean((zb_te - zb_te_pred) ** 2))

    _, s_p, vt_p = np.linalg.svd(Zpc_tr, full_matrices=False)
    Vp_k = vt_p[:k_ppg].T
    lam_p = np.maximum((s_p[:k_ppg] ** 2) / max(len(Zp_tr) - 1, 1), 1e-8)
    tp_te = (zp_te - mean_zp) @ Vp_k
    pt2_te = float(np.sum((tp_te**2) / lam_p))
    zp_te_hat = mean_zp + tp_te @ Vp_k.T
    pspe_te = float(np.sum((zp_te - zp_te_hat) ** 2))
    pcomb_te = pt2_te + pspe_te
    pmaxz_te = float(np.max(np.abs(zp_te)))

    return z_te, t2_te, spe_te, cm_te, pt2_te, pspe_te, pcomb_te, pmaxz_te


def run_loocv_subspace_decomposition(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    log_cols: Sequence[str],
    ppg_cols: Sequence[str],
    brain_cols: Sequence[str],
    ref_mask: np.ndarray,
    n_components: int = 3,
    ridge_alpha: float = 1.0,
    blind_training: bool = False,
) -> pd.DataFrame:
    """
    Runs strict Leave-One-Out Cross-Validation (LOOCV) with nested out-of-fold
    reference distribution calibration (eliminating in-sample residual shrinkage
    and guaranteeing zero leakage from subject i into fold i's reference distribution):
      - Virtual PPG Baseline T^2, SPE, and combined score (using only ppg_cols)
      - Multi-Sensor fNIRS Hotelling's T^2 (within-subspace distance)
      - Multi-Sensor fNIRS PCA Reconstruction Error SPE / Q-residual (outside-subspace distance)
      - Cross-Modal Ridge Residual (predicting brain_cols from ppg_cols)
      - Isolation Forest baseline score and out-of-fold percentile
      - Out-of-fold robust Z-scores for all features

    If blind_training=False (Normative Modeling), fits scalers/PCA/Ridge on ref_mask excluding i.
    If blind_training=True (Fully Blind Unsupervised), fits on all N-1 subjects excluding i.
    """
    n = len(df)
    X_full = symmetrize_features(df, feature_cols, log_cols)
    X_ppg = symmetrize_features(df, ppg_cols, log_cols)
    X_brain = symmetrize_features(df, brain_cols, log_cols)

    k_full = min(n_components, len(feature_cols) - 1)
    k_ppg = min(min(2, n_components), len(ppg_cols) - 1)

    t2_raw = np.zeros(n, dtype=np.float64)
    spe_raw = np.zeros(n, dtype=np.float64)
    cm_raw = np.zeros(n, dtype=np.float64)
    ppg_t2_raw = np.zeros(n, dtype=np.float64)
    ppg_spe_raw = np.zeros(n, dtype=np.float64)
    ppg_comb_raw = np.zeros(n, dtype=np.float64)
    ppg_maxz_raw = np.zeros(n, dtype=np.float64)
    iforest_raw = np.zeros(n, dtype=np.float64)

    t2_pct = np.zeros(n, dtype=np.float64)
    spe_pct = np.zeros(n, dtype=np.float64)
    cm_pct = np.zeros(n, dtype=np.float64)
    ppg_pct = np.zeros(n, dtype=np.float64)
    iforest_pct = np.zeros(n, dtype=np.float64)
    subspace_max_pct = np.zeros(n, dtype=np.float64)

    z_oof = np.zeros_like(X_full)
    inner_oof_records: list[dict[str, np.ndarray]] = []

    for i in range(n):
        if blind_training:
            train_mask = np.ones(n, dtype=bool)
        else:
            train_mask = ref_mask.copy()
        train_mask[i] = False
        idx_tr = np.where(train_mask)[0]

        z_i, t2_i, spe_i, cm_i, pt2_i, pspe_i, pcomb_i, pmaxz_i = _fit_eval_subspace_single(
            X_full[idx_tr],
            X_full[i],
            X_ppg[idx_tr],
            X_ppg[i],
            X_brain[idx_tr],
            X_brain[i],
            k_full,
            k_ppg,
            ridge_alpha,
        )
        z_oof[i] = z_i
        t2_raw[i] = t2_i
        spe_raw[i] = spe_i
        cm_raw[i] = cm_i
        ppg_t2_raw[i] = pt2_i
        ppg_spe_raw[i] = pspe_i
        ppg_comb_raw[i] = pcomb_i
        ppg_maxz_raw[i] = pmaxz_i

        # Nested LOO within idx_tr so reference residuals are strictly out-of-sample
        # and have zero dependence on test subject i.
        m_tr = len(idx_tr)
        z_tr_oof = np.zeros((m_tr, X_full.shape[1]), dtype=np.float64)
        t2_tr_oof = np.zeros(m_tr, dtype=np.float64)
        spe_tr_oof = np.zeros(m_tr, dtype=np.float64)
        cm_tr_oof = np.zeros(m_tr, dtype=np.float64)
        pcomb_tr_oof = np.zeros(m_tr, dtype=np.float64)

        for m in range(m_tr):
            sub_tr = np.delete(idx_tr, m)
            sub_te = idx_tr[m]
            zm, t2m, spem, cmm, _, _, pcombm, _ = _fit_eval_subspace_single(
                X_full[sub_tr],
                X_full[sub_te],
                X_ppg[sub_tr],
                X_ppg[sub_te],
                X_brain[sub_tr],
                X_brain[sub_te],
                k_full,
                k_ppg,
                ridge_alpha,
            )
            z_tr_oof[m] = zm
            t2_tr_oof[m] = t2m
            spe_tr_oof[m] = spem
            cm_tr_oof[m] = cmm
            pcomb_tr_oof[m] = pcombm

        t2_pct[i] = empirical_percentile_oof(t2_i, t2_tr_oof)
        spe_pct[i] = empirical_percentile_oof(spe_i, spe_tr_oof)
        cm_pct[i] = empirical_percentile_oof(cm_i, cm_tr_oof)
        ppg_pct[i] = empirical_percentile_oof(pcomb_i, pcomb_tr_oof)

        t2_tr_pct = np.array([empirical_percentile_oof(v, t2_tr_oof) for v in t2_tr_oof])
        spe_tr_pct = np.array([empirical_percentile_oof(v, spe_tr_oof) for v in spe_tr_oof])
        cm_tr_pct = np.array([empirical_percentile_oof(v, cm_tr_oof) for v in cm_tr_oof])
        subspace_train = np.maximum(t2_tr_pct, np.maximum(spe_tr_pct, cm_tr_pct))
        sub_i = max(t2_pct[i], spe_pct[i], cm_pct[i])
        subspace_max_pct[i] = empirical_percentile_oof(sub_i, subspace_train)

        inner_oof_records.append(
            {
                "z_tr_oof": z_tr_oof,
                "t2_tr_pct": t2_tr_pct,
                "spe_tr_pct": spe_tr_pct,
                "cm_tr_oof": cm_tr_oof,
            }
        )

        # Isolation Forest baseline
        Z_train_scaled, _ = _robust_scale_fit_transform(X_full[idx_tr], X_full[i])
        iso = IsolationForest(n_estimators=100, random_state=42)
        iso.fit(Z_train_scaled)
        if_tr = -iso.score_samples(Z_train_scaled)
        if_i = float(-iso.score_samples(z_i[None, :])[0])
        iforest_raw[i] = if_i
        iforest_pct[i] = empirical_percentile_oof(if_i, if_tr)

    out = df[["ID"]].copy()
    for j, c in enumerate(feature_cols):
        out[f"z_{c}"] = z_oof[:, j]

    out["ppg_t2_raw"] = ppg_t2_raw
    out["ppg_spe_raw"] = ppg_spe_raw
    out["ppg_comb_raw"] = ppg_comb_raw
    out["ppg_maxz_raw"] = ppg_maxz_raw
    out["ppg_pct"] = ppg_pct

    out["fnirs_t2_raw"] = t2_raw
    out["fnirs_t2_pct"] = t2_pct
    out["fnirs_spe_raw"] = spe_raw
    out["fnirs_spe_pct"] = spe_pct
    out["fnirs_cm_raw"] = cm_raw
    out["fnirs_cm_pct"] = cm_pct
    out["fnirs_subspace_pct"] = subspace_max_pct
    out["iforest_raw"] = iforest_raw
    out["iforest_pct"] = iforest_pct
    out.attrs["inner_oof"] = inner_oof_records
    out.attrs["feature_cols"] = list(feature_cols)
    return out


def _supine_stage2_formulas(
    z_mat: np.ndarray,
    cm_vec: np.ndarray,
    col_idx: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluates Dataset 1 (Supine) directional physiological deficit formulas on z-score matrix."""
    d_sdnn = np.maximum(0.0, -z_mat[:, col_idx["bs_sdnn_ms"]])
    d_mayer_low = np.maximum(0.0, -z_mat[:, col_idx["bs_mayer_amp"]])
    d_mayer_ratio_low = np.maximum(0.0, -z_mat[:, col_idx["bs_mayer_resp_ratio"]])
    d_lr_low = np.maximum(0.0, -z_mat[:, col_idx["bs_lr_mayer_corr"]])
    d_uncoupling = np.maximum(0.0, z_mat[:, col_idx["bh_ssr_cortical_od850"]] - z_mat[:, col_idx["bh_hr_delta"]])
    d_hr_blunted = np.maximum(0.0, -z_mat[:, col_idx["bh_hr_delta"]])

    apriori_raw = np.sqrt(
        (d_sdnn**2 + d_mayer_low**2 + d_mayer_ratio_low**2 + d_lr_low**2 + d_uncoupling**2 + d_hr_blunted**2) / 6.0
    )
    benign_hyper_excess = np.maximum(0.0, z_mat[:, col_idx["bs_rmssd_ms"]]) + np.maximum(
        0.0, z_mat[:, col_idx["bh_dur_max"]] - 0.5
    )
    uncoupling_residual = np.sqrt(cm_vec) / (1.0 + benign_hyper_excess)
    deficit_rms = np.sqrt(
        (d_sdnn**2 + d_mayer_low**2 + d_mayer_ratio_low**2 + d_lr_low**2 + d_uncoupling**2) / 5.0
    )
    combined_raw = deficit_rms + 0.65 * uncoupling_residual
    return apriori_raw, deficit_rms, combined_raw


def compute_supine_stage2_scores(
    df: pd.DataFrame,
    loocv_df: pd.DataFrame,
    ref_mask: np.ndarray,
    blind_training: bool = False,
) -> pd.DataFrame:
    """
    Computes directional physiological deficit scores for Dataset 1 (Supine) strictly out-of-fold
    with zero leakage from subject i into fold i's reference distribution:
      1. `score_apriori_deficit`: Equal-weight RMS of standard physiological insufficiency axes
         (reduced autonomic variability SDNN, suppressed Mayer oscillations & Mayer/resp ratio,
          reduced L-R coherence, and blunted HR reactivity relative to cortical hypercapnic response).
      2. `score_normative_combined`: Combines Stage 1 out-of-subspace correlation break (Cross-Modal)
         gated against benign vagal/apneic hyper-reactivity with Stage 2 directional deficit.
    """
    res = loocv_df.copy()
    n = len(df)
    fcols = list(loocv_df.attrs.get("feature_cols", []))
    col_idx = {c: j for j, c in enumerate(fcols)}
    z_outer = np.column_stack([res[f"z_{c}"].to_numpy() for c in fcols])
    cm_outer = res["fnirs_cm_raw"].to_numpy()

    apriori_raw, deficit_rms, combined_raw = _supine_stage2_formulas(z_outer, cm_outer, col_idx)

    apriori_pct = np.zeros(n, dtype=np.float64)
    deficit_pct = np.zeros(n, dtype=np.float64)
    combined_pct = np.zeros(n, dtype=np.float64)

    inner_oof = loocv_df.attrs.get("inner_oof")
    for i in range(n):
        if inner_oof is not None:
            rec_i = inner_oof[i]
            ap_tr, def_tr, comb_tr = _supine_stage2_formulas(rec_i["z_tr_oof"], rec_i["cm_tr_oof"], col_idx)
        else:
            tmask = np.ones(n, dtype=bool) if blind_training else ref_mask.copy()
            tmask[i] = False
            ap_tr, def_tr, comb_tr = apriori_raw[tmask], deficit_rms[tmask], combined_raw[tmask]

        apriori_pct[i] = empirical_percentile_oof(apriori_raw[i], ap_tr)
        deficit_pct[i] = empirical_percentile_oof(deficit_rms[i], def_tr)
        combined_pct[i] = empirical_percentile_oof(combined_raw[i], comb_tr)

    res["score_apriori_deficit_raw"] = apriori_raw
    res["score_apriori_deficit_pct"] = apriori_pct
    res["score_deficit_rms"] = deficit_rms
    res["score_deficit_pct"] = deficit_pct
    res["score_normative_combined"] = combined_raw
    res["score_normative_combined_pct"] = combined_pct
    return res


def _dot_stage2_formulas(
    z_mat: np.ndarray,
    col_idx: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluates Dataset 2 (DOT Sitting) directional deficit formulas on z-score matrix."""
    d_spb1 = np.maximum(0.0, -z_mat[:, col_idx["spb1_gain_01hz"]])
    d_spb_med = np.maximum(0.0, -z_mat[:, col_idx["spb_gain_med"]])
    d_ssr_gain = np.maximum(0.0, -z_mat[:, col_idx["spb_ssr_gain_med"]])
    d_q = np.maximum(0.0, -z_mat[:, col_idx["spb_q_factor"]])

    apriori_spb_raw = np.sqrt((d_spb1**2 + d_spb_med**2 + d_ssr_gain**2 + d_q**2) / 4.0)
    d_eve_cost = np.maximum(0.0, z_mat[:, col_idx["bh_evening_morning_ratio"]])
    deficit_rms = np.sqrt((d_spb1**2 + d_spb_med**2 + d_ssr_gain**2 + d_q**2 + d_eve_cost**2) / 5.0)
    return apriori_spb_raw, deficit_rms


def compute_dot_stage2_scores(
    df: pd.DataFrame,
    loocv_df: pd.DataFrame,
    ref_mask: np.ndarray,
    blind_training: bool = False,
) -> pd.DataFrame:
    """
    Computes directional physiological deficit scores for Dataset 2 (DOT Sitting) strictly out-of-fold
    with zero leakage from subject i into fold i's reference distribution:
      1. `score_apriori_spb_deficit`: Purely a-priori cardiorespiratory resonance deficit
         (blunted 0.1 Hz Mayer amplification in SPB1 & median SPB, blunted cortical SSR gain,
          and reduced resonance Q-factor), without any evening/morning BH ratio term.
      2. `score_normative_combined`: Full Stage 1 + Stage 2 directional triage including
         evening hypoxic cost surge (`bh_evening_morning_ratio`) as in METHODOLOGY_ANOMALY_DETECTION.md.
    """
    res = loocv_df.copy()
    n = len(df)
    fcols = list(loocv_df.attrs.get("feature_cols", []))
    col_idx = {c: j for j, c in enumerate(fcols)}
    z_outer = np.column_stack([res[f"z_{c}"].to_numpy() for c in fcols])

    apriori_spb_raw, deficit_rms = _dot_stage2_formulas(z_outer, col_idx)

    apriori_pct = np.zeros(n, dtype=np.float64)
    deficit_pct = np.zeros(n, dtype=np.float64)
    combined_raw = np.zeros(n, dtype=np.float64)
    combined_pct = np.zeros(n, dtype=np.float64)

    inner_oof = loocv_df.attrs.get("inner_oof")
    for i in range(n):
        if inner_oof is not None:
            rec_i = inner_oof[i]
            ap_tr, def_tr = _dot_stage2_formulas(rec_i["z_tr_oof"], col_idx)
            sub_train = np.maximum(rec_i["t2_tr_pct"], rec_i["spe_tr_pct"])
        else:
            tmask = np.ones(n, dtype=bool) if blind_training else ref_mask.copy()
            tmask[i] = False
            ap_tr, def_tr = apriori_spb_raw[tmask], deficit_rms[tmask]
            sub_train = np.maximum(res["fnirs_t2_pct"].to_numpy()[tmask], res["fnirs_spe_pct"].to_numpy()[tmask])

        apriori_pct[i] = empirical_percentile_oof(apriori_spb_raw[i], ap_tr)
        deficit_pct[i] = empirical_percentile_oof(deficit_rms[i], def_tr)

        def_train_pct = np.array([empirical_percentile_oof(v, def_tr) for v in def_tr])
        comb_train = 0.35 * sub_train + 0.65 * def_train_pct

        sub_i = max(float(res["fnirs_t2_pct"].iloc[i]), float(res["fnirs_spe_pct"].iloc[i]))
        comb_i = 0.35 * sub_i + 0.65 * deficit_pct[i]
        combined_raw[i] = comb_i
        combined_pct[i] = empirical_percentile_oof(comb_i, comb_train)

    res["score_apriori_spb_deficit_raw"] = apriori_spb_raw
    res["score_apriori_spb_deficit_pct"] = apriori_pct
    res["score_deficit_rms"] = deficit_rms
    res["score_deficit_pct"] = deficit_pct
    res["score_normative_combined"] = combined_raw
    res["score_normative_combined_pct"] = combined_pct
    return res


def evaluate_anomaly_detector(
    scores: np.ndarray,
    pct_scores: np.ndarray,
    y_true: np.ndarray,
    neg_control_mask: np.ndarray,
    model_name: str,
    pct_threshold: float = 90.0,
) -> EvaluationMetrics:
    """Computes threshold-free (ROC-AUC, PR-AUC, Mann-Whitney U) and thresholded (@90th pct) metrics."""
    y_bin = np.asarray(y_true, dtype=int)
    roc = float(roc_auc_score(y_bin, scores))
    pr = float(average_precision_score(y_bin, scores))

    pos_scores = scores[y_bin == 1]
    neg_scores = scores[y_bin == 0]
    _, mw_p = stats.mannwhitneyu(pos_scores, neg_scores, alternative="greater")

    preds = (pct_scores >= pct_threshold).astype(int)
    tp = int(np.sum((preds == 1) & (y_bin == 1)))
    fp = int(np.sum((preds == 1) & (y_bin == 0)))
    rec = float(recall_score(y_bin, preds, zero_division=0))
    prec = float(precision_score(y_bin, preds, zero_division=0))
    f1 = float(f1_score(y_bin, preds, zero_division=0))
    neg_ctrl_fp = int(np.sum(preds[neg_control_mask] == 1)) if np.any(neg_control_mask) else 0

    return EvaluationMetrics(
        model_name=model_name,
        roc_auc=roc,
        pr_auc=pr,
        recall_at_90=rec,
        precision_at_90=prec,
        f1_at_90=f1,
        fp_at_90=fp,
        tp_at_90=tp,
        n_pos=int(np.sum(y_bin == 1)),
        n_neg=int(np.sum(y_bin == 0)),
        mannwhitney_p=float(mw_p),
        neg_control_fp=neg_ctrl_fp,
    )
