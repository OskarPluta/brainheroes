"""Nested leave-one-out normative modeling.

For every held-out subject i the model (robust scaler, PCA, ridge) is fit on the
others. To turn i's raw scores into percentiles we need reference scores that are
just as out-of-sample, so an inner leave-one-out loop is run over the training
subjects of fold i. Nothing fit in fold i ever sees subject i.

Scores per held-out subject:
  * Hotelling's T^2 - distance inside the PCA subspace
  * SPE / Q-residual - PCA reconstruction error, i.e. broken relations between features
  * cross-modal ridge residual - how badly heart features predict cortical features
  * a heart-only ("virtual PPG") T^2 + SPE baseline
  * out-of-fold robust z-scores of every feature, used by the deficit scores
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

DeficitFn = Callable[[np.ndarray, dict[str, int]], np.ndarray]


def log_transform(df: pd.DataFrame, cols: Sequence[str], log_cols: Sequence[str]) -> np.ndarray:
    """Feature matrix with log10(max(x, 1e-4)) applied to the right-skewed columns."""
    log_set = set(log_cols)
    mat = np.zeros((len(df), len(cols)), dtype=np.float64)
    for j, c in enumerate(cols):
        vals = df[c].to_numpy(dtype=np.float64, copy=True)
        mat[:, j] = np.log10(np.maximum(vals, 1e-4)) if c in log_set else vals
    return mat


def percentile_of(values: np.ndarray | float, ref: np.ndarray) -> np.ndarray | float:
    """Empirical percentile (0-100) of each value within `ref`: share of `ref` that is <= value."""
    if np.ndim(values) == 0:
        return float(np.mean(ref <= values) * 100.0)
    return np.mean(ref[None, :] <= np.asarray(values)[:, None], axis=1) * 100.0


def robust_scale(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Median/IQR scaling fit on `x_train`, applied to both arrays."""
    med = np.median(x_train, axis=0)
    q25, q75 = np.percentile(x_train, [25.0, 75.0], axis=0)
    iqr = np.where((q75 - q25) > 1e-12, q75 - q25, 1.0)
    return (x_train - med) / iqr, (x_test - med) / iqr


def _pca_scores(z_train: np.ndarray, z_test: np.ndarray, k: int) -> tuple[float, float]:
    """(Hotelling's T^2, SPE) of one test vector against a PCA fit on `z_train`."""
    mean = np.mean(z_train, axis=0)
    _, s, vt = np.linalg.svd(z_train - mean, full_matrices=False)
    v_k = vt[:k].T
    lambdas = np.maximum((s[:k] ** 2) / max(len(z_train) - 1, 1), 1e-8)
    t = (z_test - mean) @ v_k
    return float(np.sum((t**2) / lambdas)), float(np.sum((z_test - (mean + t @ v_k.T)) ** 2))


class FoldScores(NamedTuple):
    z: np.ndarray  # robust z-scores of the test subject
    t2: float
    spe: float
    cm: float  # cross-modal ridge residual
    ppg_t2: float
    ppg_spe: float
    ppg_maxz: float

    @property
    def ppg_comb(self) -> float:
        return self.ppg_t2 + self.ppg_spe


@dataclass(frozen=True)
class FeatureViews:
    """Log-transformed matrices for the full profile and its heart / brain subsets."""

    full: np.ndarray
    ppg: np.ndarray
    brain: np.ndarray

    def fit_and_score(self, train: np.ndarray, test: int, k_full: int, k_ppg: int, ridge_alpha: float) -> FoldScores:
        z_tr, z_te = robust_scale(self.full[train], self.full[test])
        t2, spe = _pca_scores(z_tr, z_te, k_full)

        zp_tr, zp_te = robust_scale(self.ppg[train], self.ppg[test])
        zb_tr, zb_te = robust_scale(self.brain[train], self.brain[test])
        mean_p, mean_b = np.mean(zp_tr, axis=0), np.mean(zb_tr, axis=0)
        zpc_tr = zp_tr - mean_p
        w = np.linalg.solve(zpc_tr.T @ zpc_tr + ridge_alpha * np.eye(zpc_tr.shape[1]), zpc_tr.T @ (zb_tr - mean_b))
        cm = float(np.mean((zb_te - (mean_b + (zp_te - mean_p) @ w)) ** 2))

        ppg_t2, ppg_spe = _pca_scores(zp_tr, zp_te, k_ppg)
        return FoldScores(z_te, t2, spe, cm, ppg_t2, ppg_spe, float(np.max(np.abs(zp_te))))


@dataclass
class InnerReference:
    """Out-of-fold scores of the training subjects of one outer fold."""

    z: np.ndarray  # (m, p) robust z-scores
    t2_pct: np.ndarray
    spe_pct: np.ndarray
    cm: np.ndarray

    @property
    def cm_pct(self) -> np.ndarray:
        return percentile_of(self.cm, self.cm)

    @property
    def subspace_pct(self) -> np.ndarray:
        """Each training subject's max(T^2, SPE, ridge) percentile."""
        return np.maximum(self.t2_pct, np.maximum(self.spe_pct, self.cm_pct))


@dataclass
class LoocvResult:
    scores: pd.DataFrame  # one row per subject: ID, z_<feature>, raw scores and percentiles
    feature_cols: list[str]
    references: list[InnerReference]  # references[i] calibrates subject i

    @property
    def col_idx(self) -> dict[str, int]:
        return {c: j for j, c in enumerate(self.feature_cols)}

    @property
    def z(self) -> np.ndarray:
        return self.scores[[f"z_{c}" for c in self.feature_cols]].to_numpy()


def run_loocv(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    log_cols: Sequence[str],
    ppg_cols: Sequence[str],
    brain_cols: Sequence[str],
    ref_mask: np.ndarray | None = None,
    n_components: int = 3,
    ridge_alpha: float = 1.0,
    isolation_forest: bool = False,
) -> LoocvResult:
    """Nested leave-one-out scores for every subject in `df`.

    With `ref_mask=None` (fully blind) each fold trains on all other subjects. With a
    mask, each fold trains only on the masked reference subjects other than i
    (normative modeling on a pre-screened reference group).
    """
    feature_cols = list(feature_cols)
    n = len(df)
    views = FeatureViews(
        log_transform(df, feature_cols, log_cols),
        log_transform(df, ppg_cols, log_cols),
        log_transform(df, brain_cols, log_cols),
    )
    k_full = min(n_components, len(feature_cols) - 1)
    k_ppg = min(min(2, n_components), len(ppg_cols) - 1)
    pool = np.ones(n, dtype=bool) if ref_mask is None else np.asarray(ref_mask, dtype=bool)

    outer: list[FoldScores] = []
    references: list[InnerReference] = []
    cols: dict[str, list[float]] = {k: [] for k in ("t2_pct", "spe_pct", "cm_pct", "ppg_pct", "subspace_pct")}
    iforest_raw, iforest_pct = [], []
    for i in range(n):
        train_mask = pool.copy()
        train_mask[i] = False
        train = np.flatnonzero(train_mask)
        fold = views.fit_and_score(train, i, k_full, k_ppg, ridge_alpha)
        outer.append(fold)

        inner = [views.fit_and_score(np.delete(train, m), train[m], k_full, k_ppg, ridge_alpha) for m in range(len(train))]
        t2_ref = np.array([f.t2 for f in inner])
        spe_ref = np.array([f.spe for f in inner])
        cm_ref = np.array([f.cm for f in inner])
        ppg_ref = np.array([f.ppg_comb for f in inner])
        ref = InnerReference(
            np.array([f.z for f in inner]), percentile_of(t2_ref, t2_ref), percentile_of(spe_ref, spe_ref), cm_ref
        )
        references.append(ref)

        t2_pct, spe_pct, cm_pct = percentile_of(fold.t2, t2_ref), percentile_of(fold.spe, spe_ref), percentile_of(fold.cm, cm_ref)
        cols["t2_pct"].append(t2_pct)
        cols["spe_pct"].append(spe_pct)
        cols["cm_pct"].append(cm_pct)
        cols["ppg_pct"].append(percentile_of(fold.ppg_comb, ppg_ref))
        cols["subspace_pct"].append(percentile_of(max(t2_pct, spe_pct, cm_pct), ref.subspace_pct))

        if isolation_forest:
            z_train, _ = robust_scale(views.full[train], views.full[i])
            forest = IsolationForest(n_estimators=100, random_state=42).fit(z_train)
            raw = float(-forest.score_samples(fold.z[None, :])[0])
            iforest_raw.append(raw)
            iforest_pct.append(percentile_of(raw, -forest.score_samples(z_train)))

    out = df[["ID"]].copy()
    z = np.array([f.z for f in outer])
    for j, c in enumerate(feature_cols):
        out[f"z_{c}"] = z[:, j]
    out["ppg_t2_raw"] = [f.ppg_t2 for f in outer]
    out["ppg_spe_raw"] = [f.ppg_spe for f in outer]
    out["ppg_comb_raw"] = [f.ppg_comb for f in outer]
    out["ppg_maxz_raw"] = [f.ppg_maxz for f in outer]
    out["ppg_pct"] = cols["ppg_pct"]
    out["fnirs_t2_raw"] = [f.t2 for f in outer]
    out["fnirs_t2_pct"] = cols["t2_pct"]
    out["fnirs_spe_raw"] = [f.spe for f in outer]
    out["fnirs_spe_pct"] = cols["spe_pct"]
    out["fnirs_cm_raw"] = [f.cm for f in outer]
    out["fnirs_cm_pct"] = cols["cm_pct"]
    out["fnirs_subspace_pct"] = cols["subspace_pct"]
    if isolation_forest:
        out["iforest_raw"] = iforest_raw
        out["iforest_pct"] = iforest_pct
    return LoocvResult(out, feature_cols, references)


# ---------------------------------------------------------------------------
# Deficit scores: equal-weight RMS of one-sided deviations towards a blunted
# response. Higher-than-typical values never add to a deficit.
# ---------------------------------------------------------------------------


def _low(z: np.ndarray, idx: dict[str, int], col: str) -> np.ndarray:
    return np.maximum(0.0, -z[:, idx[col]])


def _rms(parts: list[np.ndarray]) -> np.ndarray:
    total = parts[0] ** 2
    for p in parts[1:]:
        total = total + p**2
    return np.sqrt(total / len(parts))


def supine_deficit(z: np.ndarray, idx: dict[str, int]) -> np.ndarray:
    """Low HRV, weak Mayer waves (absolute and vs. respiration), low left-right coherence,
    a cortical breath-hold response out of proportion to the heart-rate response, and a
    blunted heart-rate response."""
    return _rms(
        [
            _low(z, idx, "bs_sdnn_ms"),
            _low(z, idx, "bs_mayer_amp"),
            _low(z, idx, "bs_mayer_resp_ratio"),
            _low(z, idx, "bs_lr_mayer_corr"),
            np.maximum(0.0, z[:, idx["bh_ssr_cortical_od850"]] - z[:, idx["bh_hr_delta"]]),
            _low(z, idx, "bh_hr_delta"),
        ]
    )


def dot_deficit(z: np.ndarray, idx: dict[str, int]) -> np.ndarray:
    """Blunted 0.1 Hz amplification under paced breathing (first run, median run, cortex
    after short-channel regression) and a less sharp resonance peak."""
    return _rms([_low(z, idx, c) for c in ("spb1_gain_01hz", "spb_gain_med", "spb_ssr_gain_med", "spb_q_factor")])


def heart_deficit(z: np.ndarray, idx: dict[str, int]) -> np.ndarray:
    """Heart-only ablation: low HRV and a blunted heart-rate response to breath holding."""
    return _rms([_low(z, idx, c) for c in ("bs_sdnn_ms", "bs_rmssd_ms", "bh_hr_delta")])


def cortex_deficit(z: np.ndarray, idx: dict[str, int]) -> np.ndarray:
    """Cortex-only ablation: weak Mayer waves, low Mayer/respiration ratio, low coherence."""
    return _rms([_low(z, idx, c) for c in ("bs_mayer_amp", "bs_mayer_resp_ratio", "bs_lr_mayer_corr")])
