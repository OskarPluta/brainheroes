"""Nested LOOCV: no leakage of a held-out subject into its own calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from brainheroes.ablation import combined_5050
from brainheroes.constants import SUPINE_BRAIN_COLS, SUPINE_FEATURE_COLS, SUPINE_LOG_COLS, SUPINE_PPG_COLS
from brainheroes.normative import cortex_deficit, heart_deficit, percentile_of, run_loocv, supine_deficit

SCORE_COLS = [
    "fnirs_t2_raw", "fnirs_spe_raw", "fnirs_cm_raw", "fnirs_t2_pct", "fnirs_spe_pct", "fnirs_cm_pct",
    "iforest_pct", "deficit_raw", "deficit_pct", "combined_raw", "combined_pct",
]  # fmt: skip


def _random_cohort(n: int = 24, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"ID": np.arange(1, n + 1), **{c: rng.uniform(0.5, 5.0, n) for c in SUPINE_FEATURE_COLS}})


def _scores(df: pd.DataFrame, ref_mask: np.ndarray) -> pd.DataFrame:
    loocv = run_loocv(
        df, SUPINE_FEATURE_COLS, SUPINE_LOG_COLS, SUPINE_PPG_COLS, SUPINE_BRAIN_COLS, ref_mask=ref_mask, isolation_forest=True
    )
    return combined_5050(loocv, supine_deficit)


def test_percentile_of_matches_the_scalar_definition():
    ref = np.array([3.0, 1.0, 2.0, 2.0])
    assert percentile_of(2.0, ref) == 75.0
    np.testing.assert_array_equal(percentile_of(np.array([0.0, 2.0, 9.0]), ref), [0.0, 75.0, 100.0])


def test_non_reference_subjects_cannot_move_reference_scores():
    df = _random_cohort()
    ref_mask = np.ones(len(df), dtype=bool)
    ref_mask[-3:] = False
    before = _scores(df, ref_mask)
    corrupted = df.copy()
    corrupted.loc[21:, SUPINE_FEATURE_COLS] *= 1000.0
    after = _scores(corrupted, ref_mask)
    for col in SCORE_COLS:
        np.testing.assert_allclose(before.loc[:20, col], after.loc[:20, col], rtol=1e-12, atol=1e-12)


def test_subject_never_enters_its_own_reference():
    """Fold i excludes i anyway, so dropping i from the reference pool must not change i's scores."""
    df = _random_cohort()
    ref_mask = np.ones(len(df), dtype=bool)
    ref_mask[-3:] = False
    without_0 = ref_mask.copy()
    without_0[0] = False
    a, b = _scores(df, ref_mask), _scores(df, without_0)
    for col in SCORE_COLS:
        np.testing.assert_allclose(a.loc[0, col], b.loc[0, col], rtol=1e-12, atol=1e-12)


def test_deficits_ignore_better_than_typical_values():
    idx = {c: j for j, c in enumerate(SUPINE_FEATURE_COLS)}
    high = np.full((3, len(SUPINE_FEATURE_COLS)), 2.5)
    for fn in (heart_deficit, cortex_deficit):
        assert np.all(fn(high, idx) == 0.0)
        assert np.all(fn(-high, idx) > 0.0)
