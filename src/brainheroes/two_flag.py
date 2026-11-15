"""Two-flag scoring: a person is as anomalous as their most anomalous axis.

Each person gets two fully blind, out-of-fold percentiles and the alarm is the higher one:

  * deficit    - equal-weight RMS of one-sided deviations towards a blunted response
  * decoupling - PCA reconstruction error (SPE / Q-residual): the usual relations
                 between features break, even if every single feature looks ordinary

Averaging the two would dilute people who are abnormal on one axis only. The maximum is
re-calibrated against the nested-LOOCV reference, so a threshold of 90 again means
roughly 10% alarms in a typical cohort. No clinical label is used anywhere.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from .constants import (
    ALARM_THRESHOLD,
    DOT_BRAIN_COLS,
    DOT_CRIT_A_IDS,
    DOT_CRIT_B_IDS,
    DOT_FEATURE_COLS,
    DOT_LOG_COLS,
    DOT_PPG_COLS,
    N_COMPONENTS,
    RESULTS_DIR,
    SCREENING_THRESHOLD,
    SUPINE_BRAIN_COLS,
    SUPINE_CRIT_A_IDS,
    SUPINE_CRIT_B_IDS,
    SUPINE_FEATURE_COLS,
    SUPINE_HEART_COLS,
    SUPINE_LOG_COLS,
    SUPINE_PPG_COLS,
)
from .features import load_dot, load_supine, qc_funnel
from .normative import DeficitFn, cortex_deficit, dot_deficit, heart_deficit, percentile_of, run_loocv, supine_deficit

YOUNG_MAX_AGE = 42  # everyone in the supine cohort up to 42; isolates ID 38 from the 70+ group


def two_flag_scores(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    log_cols: Sequence[str],
    deficit_fn: DeficitFn,
    ppg_cols: Sequence[str] | None = None,
    brain_cols: Sequence[str] | None = None,
    n_components: int = N_COMPONENTS,
    decoupling: str = "spe",
) -> pd.DataFrame:
    """Fully blind nested-LOOCV two-flag scores, one row per subject.

    Columns: `deficit_pct`, `decoupling_pct`, `alarm_raw` (the higher of the two),
    `alarm_pct` (alarm_raw re-calibrated out of fold), `alarm`, `rank`, `z_<feature>`.
    `decoupling="spe_cm"` also lets the heart -> cortex ridge residual raise the
    decoupling flag (reported only as a variant).
    """
    if decoupling not in ("spe", "spe_cm"):
        raise ValueError(f"unknown decoupling variant: {decoupling}")
    feature_cols = list(feature_cols)
    loocv = run_loocv(
        df,
        feature_cols,
        log_cols,
        feature_cols if ppg_cols is None else ppg_cols,
        feature_cols if brain_cols is None else brain_cols,
        n_components=n_components,
    )
    idx = loocv.col_idx
    deficit_raw = deficit_fn(loocv.z, idx)
    spe_pct = loocv.scores["fnirs_spe_pct"].to_numpy()
    cm_pct = loocv.scores["fnirs_cm_pct"].to_numpy()

    n = len(df)
    deficit_pct, decoupling_pct, alarm_raw, alarm_pct = (np.zeros(n) for _ in range(4))
    for i, ref in enumerate(loocv.references):
        deficit_ref = deficit_fn(ref.z, idx)
        decoupling_ref = ref.spe_pct
        decoupling_pct[i] = spe_pct[i]
        if decoupling == "spe_cm":
            decoupling_ref = np.maximum(decoupling_ref, ref.cm_pct)
            decoupling_pct[i] = max(spe_pct[i], cm_pct[i])
        deficit_pct[i] = percentile_of(deficit_raw[i], deficit_ref)
        alarm_raw[i] = max(deficit_pct[i], decoupling_pct[i])
        alarm_ref = np.maximum(percentile_of(deficit_ref, deficit_ref), decoupling_ref)
        alarm_pct[i] = percentile_of(alarm_raw[i], alarm_ref)

    out = df[["ID"]].copy()
    out["deficit_pct"] = deficit_pct
    out["decoupling_pct"] = decoupling_pct
    out["alarm_raw"] = alarm_raw
    out["alarm_pct"] = alarm_pct
    out["alarm"] = alarm_pct >= ALARM_THRESHOLD
    # Ties count against us: a tied group all get the worst rank in the group.
    out["rank"] = out["alarm_raw"].rank(ascending=False, method="max").astype(int)
    for c in feature_cols:
        out[f"z_{c}"] = loocv.scores[f"z_{c}"].to_numpy()
    return out


def bootstrap_auc_ci(y: np.ndarray, scores: np.ndarray, n_boot: int = 2000, seed: int = 42) -> list[float]:
    """95% percentile-bootstrap CI of ROC-AUC (resamples without both classes are skipped)."""
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        if 0 < y[idx].sum() < len(y):
            aucs.append(float(roc_auc_score(y[idx], scores[idx])))
    return [round(float(np.percentile(aucs, 2.5)), 3), round(float(np.percentile(aucs, 97.5)), 3)]


def summarize(scores: pd.DataFrame, positive_ids: Sequence[int]) -> dict[str, object]:
    """Ranking and threshold metrics of a two-flag table against metadata-flagged IDs."""
    y = scores["ID"].isin(positive_ids).to_numpy()
    raw = scores["alarm_raw"].to_numpy()
    flag = scores["alarm"].to_numpy()
    flag_85 = (scores["alarm_pct"] >= SCREENING_THRESHOLD).to_numpy()
    pos = scores[y]
    _, mw_p = stats.mannwhitneyu(raw[y], raw[~y], alternative="greater")
    return {
        "n": len(scores),
        "n_pos": int(y.sum()),
        "auc": round(float(roc_auc_score(y, raw)), 3),
        # With a single positive every resample either drops it or is degenerate.
        "auc_ci95": bootstrap_auc_ci(y, raw) if y.sum() >= 2 else None,
        "mw_p": round(float(mw_p), 4),
        "auc_deficit": round(float(roc_auc_score(y, scores["deficit_pct"])), 3),
        "auc_decoupling": round(float(roc_auc_score(y, scores["decoupling_pct"])), 3),
        "tp": int((flag & y).sum()),
        "fp": int((flag & ~y).sum()),
        "tp_85": int((flag_85 & y).sum()),
        "fp_85": int((flag_85 & ~y).sum()),
        "positives": [
            {
                "id": int(r.ID),
                "deficit_pct": round(float(r.deficit_pct), 1),
                "decoupling_pct": round(float(r.decoupling_pct), 1),
                "alarm_pct": round(float(r.alarm_pct), 1),
                "rank": int(r.rank),
            }
            for r in pos.itertuples()
        ],
        "review_to_catch_all": int(pos["rank"].max()),
    }


def supine_scores(df: pd.DataFrame, k: int = N_COMPONENTS, decoupling: str = "spe") -> pd.DataFrame:
    return two_flag_scores(
        df, SUPINE_FEATURE_COLS, SUPINE_LOG_COLS, supine_deficit, SUPINE_PPG_COLS, SUPINE_BRAIN_COLS, k, decoupling
    )


def dot_scores(df: pd.DataFrame, k: int = N_COMPONENTS, decoupling: str = "spe") -> pd.DataFrame:
    return two_flag_scores(df, DOT_FEATURE_COLS, DOT_LOG_COLS, dot_deficit, DOT_PPG_COLS, DOT_BRAIN_COLS, k, decoupling)


def run(results_dir: Path = RESULTS_DIR) -> dict[str, object]:
    """Score both cohorts, run the ablations, write results/two_flag_* and return the summary."""
    sup_all, sup = load_supine(results_dir)
    dot = load_dot(results_dir)

    sup_main = supine_scores(sup)
    dot_main = dot_scores(dot)
    heart = two_flag_scores(sup, SUPINE_HEART_COLS, SUPINE_LOG_COLS, heart_deficit, n_components=2)
    cortex = two_flag_scores(sup, SUPINE_BRAIN_COLS, SUPINE_LOG_COLS, cortex_deficit, n_components=3)

    young = sup[sup["Age"] <= YOUNG_MAX_AGE].reset_index(drop=True)
    young_scores = supine_scores(young)
    y38 = young_scores[young_scores["ID"] == 38].iloc[0]

    sup_meta = sup[["ID", "Age", "BMI", "Medication"]]
    dot_meta = dot[["ID", "Age", "BMI", "Chronic_Disease", "Medication", "spb1_gain_01hz", "spb_gain_med", "spb_ssr_gain_med"]]
    sup_main.merge(sup_meta, on="ID").to_csv(results_dir / "two_flag_supine.csv", index=False)
    dot_main.merge(dot_meta, on="ID").to_csv(results_dir / "two_flag_dot.csv", index=False)

    k_sensitivity = []
    for k in [1, 2, 3, 4, 5]:
        s = summarize(sup_main if k == N_COMPONENTS else supine_scores(sup, k), SUPINE_CRIT_A_IDS)
        d = summarize(dot_main if k == N_COMPONENTS else dot_scores(dot, k), DOT_CRIT_B_IDS)
        k_sensitivity.append(
            {"k": k, "supine_auc": s["auc"], "supine_tp": s["tp"], "supine_fp": s["fp"],
             "dot_auc": d["auc"], "dot_tp": d["tp"], "dot_fp": d["fp"]}
        )  # fmt: skip

    summary: dict[str, object] = {
        "threshold": ALARM_THRESHOLD,
        "screening_threshold": SCREENING_THRESHOLD,
        "qc_funnel": qc_funnel(sup_all),
        "supine": summarize(sup_main, SUPINE_CRIT_A_IDS),
        "supine_crit_b": summarize(sup_main, SUPINE_CRIT_B_IDS),
        "supine_heart_only": summarize(heart, SUPINE_CRIT_A_IDS),
        "supine_cortex_only": summarize(cortex, SUPINE_CRIT_A_IDS),
        "supine_age_auc": round(float(roc_auc_score(sup["ID"].isin(SUPINE_CRIT_A_IDS), sup["Age"])), 3),
        "supine_young_id38": {
            "n_young": len(young),
            "deficit_pct": round(float(y38.deficit_pct), 1),
            "decoupling_pct": round(float(y38.decoupling_pct), 1),
            "alarm_pct": round(float(y38.alarm_pct), 1),
            "rank": int(y38["rank"]),
            "n_tied": int((young_scores["alarm_raw"] == y38["alarm_raw"]).sum()),
        },
        "dot": summarize(dot_main, DOT_CRIT_B_IDS),
        "dot_crit_a": summarize(dot_main, DOT_CRIT_A_IDS),
        "k_sensitivity": k_sensitivity,
        "variants_k3": {
            "supine_decoupling_spe_or_ridge": summarize(supine_scores(sup, decoupling="spe_cm"), SUPINE_CRIT_A_IDS),
            "dot_decoupling_spe_or_ridge": summarize(dot_scores(dot, decoupling="spe_cm"), DOT_CRIT_B_IDS),
        },
    }
    (results_dir / "two_flag_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
