"""Two-flag anomaly scoring: a person is as anomalous as their most anomalous axis.

Instead of averaging two different kinds of anomaly into one number, each person gets
two out-of-fold percentiles and the alarm is the higher of the two:

  * deficit    - equal-weight RMS of one-sided deviations towards a blunted response
  * decoupling - PCA reconstruction error (SPE / Q-residual): the usual relations
                 between features break, even if every single feature looks ordinary

Everything is fully blind (no clinical labels anywhere) and calibrated against
out-of-fold reference scores from the nested leave-one-out loop.

Run `python -m src.two_flag_scoring` to regenerate results/two_flag_*.{csv,json}
from the cached feature tables.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.normative_modeling import (
    _dot_stage2_formulas,
    _supine_stage2_formulas,
    empirical_percentile_oof,
    run_loocv_subspace_decomposition,
)
from src.run_experiments import (
    DOT_BRAIN_COLS,
    DOT_FEATURE_COLS,
    DOT_LOG_COLS,
    DOT_PPG_COLS,
    SUPINE_BRAIN_COLS,
    SUPINE_FEATURE_COLS,
    SUPINE_LOG_COLS,
    SUPINE_PPG_COLS,
)

DeficitFn = Callable[[np.ndarray, dict[str, int]], np.ndarray]

SUPINE_HEART_COLS = ["bs_hr_bpm", "bs_sdnn_ms", "bs_rmssd_ms", "bh_hr_delta"]
SUPINE_CORTEX_COLS = list(SUPINE_BRAIN_COLS)

SUPINE_POSITIVE_IDS = [38, 48, 49, 60]
DOT_POSITIVE_IDS = [9, 12, 16]
ALARM_THRESHOLD = 90.0


def _one_sided_rms(z_mat: np.ndarray, col_idx: dict[str, int], low_cols: Sequence[str]) -> np.ndarray:
    parts = [np.maximum(0.0, -z_mat[:, col_idx[c]]) ** 2 for c in low_cols]
    return np.sqrt(np.sum(parts, axis=0) / len(parts))


def heart_deficit(z_mat: np.ndarray, col_idx: dict[str, int]) -> np.ndarray:
    """Heart-only deficit: low HRV (SDNN, RMSSD) and blunted heart-rate reaction to breath hold."""
    return _one_sided_rms(z_mat, col_idx, ["bs_sdnn_ms", "bs_rmssd_ms", "bh_hr_delta"])


def cortex_deficit(z_mat: np.ndarray, col_idx: dict[str, int]) -> np.ndarray:
    """Cortex-only deficit: weak Mayer waves, low Mayer/respiration ratio, low left-right coherence."""
    return _one_sided_rms(z_mat, col_idx, ["bs_mayer_amp", "bs_mayer_resp_ratio", "bs_lr_mayer_corr"])


def two_flag_scores(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    log_cols: Sequence[str],
    deficit_fn: DeficitFn,
    ppg_cols: Sequence[str] | None = None,
    brain_cols: Sequence[str] | None = None,
    n_components: int = 3,
    decoupling: str = "spe",
) -> pd.DataFrame:
    """
    Fully blind nested-LOOCV two-flag scores.

    Returns one row per subject with `deficit_pct`, `decoupling_pct`, `alarm_raw`
    (the higher of the two) and `alarm_pct` (alarm_raw re-calibrated against the
    out-of-fold reference, so a threshold of 90 again means roughly 10% alarms).
    `decoupling="spe_cm"` additionally takes the heart->cortex ridge residual into
    the decoupling flag (kept only as a transparency variant).
    """
    feature_cols = list(feature_cols)
    ppg = list(ppg_cols) if ppg_cols is not None else feature_cols
    brain = list(brain_cols) if brain_cols is not None else feature_cols
    n = len(df)
    loocv = run_loocv_subspace_decomposition(
        df,
        feature_cols,
        log_cols,
        ppg,
        brain,
        np.ones(n, dtype=bool),
        n_components=n_components,
        blind_training=True,
    )
    col_idx = {c: j for j, c in enumerate(feature_cols)}
    z_outer = np.column_stack([loocv[f"z_{c}"].to_numpy() for c in feature_cols])
    deficit_raw = deficit_fn(z_outer, col_idx)

    deficit_pct = np.zeros(n)
    decoupling_pct = np.zeros(n)
    alarm_raw = np.zeros(n)
    alarm_pct = np.zeros(n)
    for i, rec in enumerate(loocv.attrs["inner_oof"]):
        def_ref = deficit_fn(rec["z_tr_oof"], col_idx)
        def_ref_pct = np.array([empirical_percentile_oof(v, def_ref) for v in def_ref])
        dec_ref_pct = rec["spe_tr_pct"]
        dec_i = float(loocv["fnirs_spe_pct"].iloc[i])
        if decoupling == "spe_cm":
            cm_ref = rec["cm_tr_oof"]
            cm_ref_pct = np.array([empirical_percentile_oof(v, cm_ref) for v in cm_ref])
            dec_ref_pct = np.maximum(dec_ref_pct, cm_ref_pct)
            dec_i = max(dec_i, float(loocv["fnirs_cm_pct"].iloc[i]))

        deficit_pct[i] = empirical_percentile_oof(deficit_raw[i], def_ref)
        decoupling_pct[i] = dec_i
        alarm_raw[i] = max(deficit_pct[i], dec_i)
        alarm_pct[i] = empirical_percentile_oof(alarm_raw[i], np.maximum(def_ref_pct, dec_ref_pct))

    out = df[["ID"]].copy()
    out["deficit_pct"] = deficit_pct
    out["decoupling_pct"] = decoupling_pct
    out["alarm_raw"] = alarm_raw
    out["alarm_pct"] = alarm_pct
    out["alarm"] = alarm_pct >= ALARM_THRESHOLD
    out["rank"] = out["alarm_raw"].rank(ascending=False, method="max").astype(int)
    for j, c in enumerate(feature_cols):
        out[f"z_{c}"] = z_outer[:, j]
    return out


def summarize(scores: pd.DataFrame, positive_ids: Sequence[int]) -> dict[str, object]:
    """Ranking and threshold summary of a two-flag score table against a list of flagged-by-metadata IDs."""
    y = scores["ID"].isin(positive_ids).to_numpy()
    flag = scores["alarm"].to_numpy()
    pos = scores[y]
    return {
        "n": int(len(scores)),
        "n_pos": int(y.sum()),
        "auc": round(float(roc_auc_score(y, scores["alarm_raw"])), 3),
        "auc_deficit": round(float(roc_auc_score(y, scores["deficit_pct"])), 3),
        "auc_decoupling": round(float(roc_auc_score(y, scores["decoupling_pct"])), 3),
        "tp": int((flag & y).sum()),
        "fp": int((flag & ~y).sum()),
        "positives": [
            {
                "id": int(r.ID),
                "deficit_pct": round(float(r.deficit_pct), 1),
                "decoupling_pct": round(float(r.decoupling_pct), 1),
                "alarm_pct": round(float(r.alarm_pct), 1),
                "rank": int(r["rank"]),
            }
            for _, r in pos.iterrows()
        ],
        "review_to_catch_all": int(pos["rank"].max()),
    }


def run_all(root_dir: Path | str) -> dict[str, object]:
    root = Path(root_dir)
    res = root / "results"
    sup_all = pd.read_csv(res / "supine_features_const_dpf.csv")
    sup = sup_all[sup_all["has_bs"] & sup_all["has_bh"]].reset_index(drop=True)
    dot = pd.read_csv(res / "dot_features_const_dpf.csv")

    def supine_full(frame: pd.DataFrame, k: int = 3, decoupling: str = "spe") -> pd.DataFrame:
        return two_flag_scores(
            frame, SUPINE_FEATURE_COLS, SUPINE_LOG_COLS, _supine_stage2_formulas,
            SUPINE_PPG_COLS, SUPINE_BRAIN_COLS, n_components=k, decoupling=decoupling,
        )

    def dot_full(k: int = 3, decoupling: str = "spe") -> pd.DataFrame:
        return two_flag_scores(
            dot, DOT_FEATURE_COLS, DOT_LOG_COLS, _dot_stage2_formulas,
            DOT_PPG_COLS, DOT_BRAIN_COLS, n_components=k, decoupling=decoupling,
        )

    sup_scores = supine_full(sup)
    dot_scores = dot_full()
    heart = two_flag_scores(sup, SUPINE_HEART_COLS, SUPINE_LOG_COLS, heart_deficit, n_components=2)
    cortex = two_flag_scores(sup, SUPINE_CORTEX_COLS, SUPINE_LOG_COLS, cortex_deficit, n_components=3)

    young = sup[sup["Age"] <= 42].reset_index(drop=True)
    young_scores = supine_full(young)
    y38 = young_scores[young_scores["ID"] == 38].iloc[0]

    meta_sup = sup[["ID", "Age", "BMI", "Medication"]]
    meta_dot = dot[["ID", "Age", "BMI", "Chronic_Disease", "Medication", "spb1_gain_01hz", "spb_gain_med", "spb_ssr_gain_med"]]
    sup_scores.merge(meta_sup, on="ID").to_csv(res / "two_flag_supine.csv", index=False)
    dot_scores.merge(meta_dot, on="ID").to_csv(res / "two_flag_dot.csv", index=False)

    y_sup = sup["ID"].isin(SUPINE_POSITIVE_IDS).to_numpy()
    summary: dict[str, object] = {
        "threshold": ALARM_THRESHOLD,
        "supine": summarize(sup_scores, SUPINE_POSITIVE_IDS),
        "supine_heart_only": summarize(heart, SUPINE_POSITIVE_IDS),
        "supine_cortex_only": summarize(cortex, SUPINE_POSITIVE_IDS),
        "supine_age_auc": round(float(roc_auc_score(y_sup, sup["Age"])), 3),
        "supine_young_id38": {
            "n_young": int(len(young)),
            "deficit_pct": round(float(y38.deficit_pct), 1),
            "decoupling_pct": round(float(y38.decoupling_pct), 1),
            "alarm_pct": round(float(y38.alarm_pct), 1),
            "rank": int(y38["rank"]),
        },
        "dot": summarize(dot_scores, DOT_POSITIVE_IDS),
        "k_sensitivity": [],
        "variants_k3": {},
    }
    for k in [1, 2, 3, 4, 5]:
        s = sup_scores if k == 3 else supine_full(sup, k)
        d = dot_scores if k == 3 else dot_full(k)
        ss, ds = summarize(s, SUPINE_POSITIVE_IDS), summarize(d, DOT_POSITIVE_IDS)
        summary["k_sensitivity"].append(
            {"k": k, "supine_auc": ss["auc"], "supine_tp": ss["tp"], "supine_fp": ss["fp"],
             "dot_auc": ds["auc"], "dot_tp": ds["tp"], "dot_fp": ds["fp"]}
        )
    summary["variants_k3"] = {
        "supine_decoupling_spe_or_ridge": summarize(supine_full(sup, 3, "spe_cm"), SUPINE_POSITIVE_IDS),
        "dot_decoupling_spe_or_ridge": summarize(dot_full(3, "spe_cm"), DOT_POSITIVE_IDS),
    }
    with open(res / "two_flag_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


if __name__ == "__main__":
    print(json.dumps(run_all(Path(__file__).resolve().parent.parent), indent=2, ensure_ascii=False))
