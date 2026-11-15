"""Two-flag scoring on synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from brainheroes.constants import SUPINE_FEATURE_COLS
from brainheroes.two_flag import summarize, supine_scores


def _synthetic(n: int = 18, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"ID": np.arange(1, n + 1), **{c: rng.uniform(0.5, 5.0, n) for c in SUPINE_FEATURE_COLS}})


def test_alarm_is_the_higher_flag_not_an_average():
    scores = supine_scores(_synthetic())
    np.testing.assert_allclose(scores["alarm_raw"], np.maximum(scores["deficit_pct"], scores["decoupling_pct"]))
    for col in ["deficit_pct", "decoupling_pct", "alarm_raw", "alarm_pct"]:
        assert scores[col].between(0.0, 100.0).all()
    assert (scores["alarm"] == (scores["alarm_pct"] >= 90.0)).all()
    assert scores.sort_values("alarm_raw", ascending=False)["rank"].is_monotonic_increasing


def test_scores_do_not_depend_on_row_order():
    df = _synthetic()
    a = supine_scores(df)
    shuffled = df.sample(frac=1.0, random_state=1).reset_index(drop=True)
    b = supine_scores(shuffled).set_index("ID").loc[df["ID"]].reset_index()
    for col in ["deficit_pct", "decoupling_pct", "alarm_raw", "alarm_pct"]:
        np.testing.assert_allclose(a[col], b[col], rtol=1e-9, atol=1e-9)


def test_planted_decoupling_is_caught_without_any_deficit():
    """A subject who breaks the cohort's correlation pattern must be flagged by decoupling."""
    rng = np.random.default_rng(3)
    n = 30
    latent = rng.normal(size=n)
    df = pd.DataFrame(
        {"ID": np.arange(1, n + 1), **{c: np.exp(0.4 * latent + 0.03 * rng.normal(size=n)) for c in SUPINE_FEATURE_COLS}}
    )
    # Everyone moves along one shared direction; subject 0 moves against it on three features.
    df.loc[0, "bs_mayer_amp"] = float(df["bs_mayer_amp"].quantile(0.95))
    df.loc[0, "bh_deep_od850"] = float(df["bh_deep_od850"].quantile(0.05))
    df.loc[0, "bs_sdnn_ms"] = float(df["bs_sdnn_ms"].quantile(0.95))
    scores = supine_scores(df)
    assert scores.loc[0, "decoupling_pct"] >= 95.0
    assert bool(scores.loc[0, "alarm"])


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
