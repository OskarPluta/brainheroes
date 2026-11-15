"""Tests for the two-flag (deficit / decoupling) scoring on synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.normative_modeling import _supine_stage2_formulas
from src.run_experiments import SUPINE_BRAIN_COLS, SUPINE_FEATURE_COLS, SUPINE_LOG_COLS, SUPINE_PPG_COLS
from src.two_flag_scoring import cortex_deficit, heart_deficit, summarize, two_flag_scores


def _synthetic(n: int = 18, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {"ID": np.arange(1, n + 1)}
    for col in SUPINE_FEATURE_COLS:
        data[col] = rng.uniform(0.5, 5.0, n)
    return pd.DataFrame(data)


def _score(df: pd.DataFrame) -> pd.DataFrame:
    return two_flag_scores(
        df, SUPINE_FEATURE_COLS, SUPINE_LOG_COLS, _supine_stage2_formulas, SUPINE_PPG_COLS, SUPINE_BRAIN_COLS
    )


def test_alarm_is_the_higher_flag_and_never_an_average():
    scores = _score(_synthetic())
    np.testing.assert_allclose(scores["alarm_raw"], np.maximum(scores["deficit_pct"], scores["decoupling_pct"]))
    for col in ["deficit_pct", "decoupling_pct", "alarm_raw", "alarm_pct"]:
        assert scores[col].between(0.0, 100.0).all()
    assert (scores["alarm"] == (scores["alarm_pct"] >= 90.0)).all()
    # A higher alarm_raw can never get a lower rank number than a smaller one.
    ordered = scores.sort_values("alarm_raw", ascending=False)
    assert ordered["rank"].is_monotonic_increasing


def test_scores_do_not_depend_on_any_label_or_row_order():
    df = _synthetic()
    a = _score(df)
    shuffled = df.sample(frac=1.0, random_state=1).reset_index(drop=True)
    b = _score(shuffled).set_index("ID").loc[df["ID"]].reset_index()
    for col in ["deficit_pct", "decoupling_pct", "alarm_raw", "alarm_pct"]:
        np.testing.assert_allclose(a[col], b[col], rtol=1e-9, atol=1e-9)


def test_planted_decoupling_is_caught_without_any_deficit():
    """A subject whose features break the cohort's correlation pattern must be flagged by decoupling."""
    rng = np.random.default_rng(3)
    n = 30
    latent = rng.normal(size=n)
    data = {"ID": np.arange(1, n + 1)}
    for col in SUPINE_FEATURE_COLS:
        data[col] = np.exp(0.4 * latent + 0.03 * rng.normal(size=n))
    df = pd.DataFrame(data)
    # Everyone else moves along one shared direction; subject 0 moves against it on two features.
    df.loc[0, "bs_mayer_amp"] = float(df["bs_mayer_amp"].quantile(0.95))
    df.loc[0, "bh_deep_od850"] = float(df["bh_deep_od850"].quantile(0.05))
    df.loc[0, "bs_sdnn_ms"] = float(df["bs_sdnn_ms"].quantile(0.95))
    scores = _score(df)
    assert scores.loc[0, "decoupling_pct"] >= 95.0
    assert bool(scores.loc[0, "alarm"])


def test_one_sided_deficits_ignore_better_than_typical_values():
    cols = {c: j for j, c in enumerate(SUPINE_FEATURE_COLS)}
    high = np.full((3, len(SUPINE_FEATURE_COLS)), 2.5)
    assert np.all(heart_deficit(high, cols) == 0.0)
    assert np.all(cortex_deficit(high, cols) == 0.0)
    low = -high
    assert np.all(heart_deficit(low, cols) > 0.0)
    assert np.all(cortex_deficit(low, cols) > 0.0)


def test_summarize_counts_and_review_depth():
    scores = pd.DataFrame(
        {
            "ID": [1, 2, 3, 4, 5],
            "deficit_pct": [100.0, 20.0, 95.0, 10.0, 30.0],
            "decoupling_pct": [10.0, 90.0, 20.0, 15.0, 40.0],
            "alarm_raw": [100.0, 90.0, 95.0, 15.0, 40.0],
            "alarm_pct": [100.0, 80.0, 95.0, 5.0, 30.0],
            "alarm": [True, False, True, False, False],
            "rank": [1, 3, 2, 5, 4],
        }
    )
    out = summarize(scores, [1, 2])
    assert out["tp"] == 1 and out["fp"] == 1
    assert out["review_to_catch_all"] == 3
    assert out["auc"] == round(5 / 6, 3)
