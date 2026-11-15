"""Sanity check: do plain off-the-shelf detectors do as well as the two-flag rule?

Two families, both evaluated leave-one-subject-out on the cached feature tables:

  * supervised  - logistic regression, random forest, XGBoost. These see the labels
                  of the other subjects. With 3-4 positives the honest yardstick is a
                  label-permutation null run through the identical procedure.
  * unsupervised - Isolation Forest, LOF, One-Class SVM, Mahalanobis, kNN distance,
                  PCA reconstruction error. Fully blind, like the two-flag rule.

Each is run on the base feature set used by the normative pipeline and on an
engineered set (all extracted signal features, log-scaled, plus ratios).
Age and BMI are never used as features.

    uv run --with xgboost python -m src.simple_detectors
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.svm import OneClassSVM

from src.run_experiments import DOT_FEATURE_COLS, SUPINE_FEATURE_COLS
from src.two_flag_scoring import DOT_POSITIVE_IDS, SUPINE_POSITIVE_IDS

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None

N_PERMUTATIONS = 200
EPS = 1e-9

SUPINE_SIGNAL_COLS = [
    "bs_hr_bpm", "bs_sdnn_ms", "bs_rmssd_ms", "bs_pulse_amp", "bs_lr_mayer_corr", "bs_mayer_amp",
    "bs_cort_mayer_amp", "bs_resp_amp", "bs_mayer_resp_ratio", "bh_dur_max", "bh_dur_mean",
    "bh_ssr_cortical_od850", "bh_deep_od850", "bh_shal_od850", "bh_hbr_hbo_ratio", "bh_od760_850_ratio",
    "bh_spatial_cv", "bh_hr_delta", "bh_hr_std", "bh_cort_hr_coupling",
]
DOT_SIGNAL_COLS = [
    "rs_hr_bpm", "rs_sdnn_ms", "rs_rmssd_ms", "rs_pulse_amp", "spb1_gain_01hz", "spb_gain_med",
    "spb_ssr_gain_med", "spb_lr_coherence", "spb_q_factor", "bh_hr_delta", "bh_ssr_cortical_od850",
    "bh_deep_od850", "bh_short10s_ssr_amp", "bh_hbr_hbo_ratio", "bh_od760_850_ratio", "bh_spatial_cv",
    "bh_evening_morning_ratio",
]
NO_LOG = {"bs_hr_bpm", "rs_hr_bpm", "bs_lr_mayer_corr", "spb_lr_coherence", "spb_q_factor", "bh_hr_delta"}


def _log_scaled(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in cols:
        out[c] = df[c] if c in NO_LOG else np.log10(np.maximum(df[c], 1e-4))
    return out


def engineer_supine(df: pd.DataFrame) -> pd.DataFrame:
    """All extracted supine signal features (log-scaled) plus physiologically motivated ratios."""
    x = _log_scaled(df, SUPINE_SIGNAL_COLS)
    r = lambda a, b: np.log10((df[a] + EPS) / (df[b] + EPS))  # noqa: E731
    x["r_rmssd_sdnn"] = r("bs_rmssd_ms", "bs_sdnn_ms")
    x["r_sdnn_per_beat"] = np.log10(df["bs_sdnn_ms"] * df["bs_hr_bpm"] / 60000.0 + EPS)
    x["r_mayer_per_pulse"] = r("bs_mayer_amp", "bs_pulse_amp")
    x["r_resp_per_pulse"] = r("bs_resp_amp", "bs_pulse_amp")
    x["r_cortical_share_mayer"] = r("bs_cort_mayer_amp", "bs_mayer_amp")
    x["r_cortical_share_bh"] = r("bh_ssr_cortical_od850", "bh_deep_od850")
    x["r_deep_per_shallow_bh"] = r("bh_deep_od850", "bh_shal_od850")
    x["r_cortical_per_second"] = r("bh_ssr_cortical_od850", "bh_dur_mean")
    x["r_bh_per_mayer"] = r("bh_deep_od850", "bs_mayer_amp")
    x["r_hold_consistency"] = r("bh_dur_mean", "bh_dur_max")
    x["d_hr_delta_per_second"] = df["bh_hr_delta"] / (df["bh_dur_mean"] + EPS)
    x["d_hr_delta_rel"] = df["bh_hr_delta"] / (df["bs_hr_bpm"] + EPS)
    return x


def engineer_dot(df: pd.DataFrame) -> pd.DataFrame:
    """All extracted DOT signal features (log-scaled) plus ratios. `bh_dur_est` is left out (clipped estimator)."""
    x = _log_scaled(df, DOT_SIGNAL_COLS)
    r = lambda a, b: np.log10((df[a] + EPS) / (df[b] + EPS))  # noqa: E731
    x["r_rmssd_sdnn"] = r("rs_rmssd_ms", "rs_sdnn_ms")
    x["r_sdnn_per_beat"] = np.log10(df["rs_sdnn_ms"] * df["rs_hr_bpm"] / 60000.0 + EPS)
    x["r_gain_first_vs_median"] = r("spb1_gain_01hz", "spb_gain_med")
    x["r_cortical_share_gain"] = r("spb_ssr_gain_med", "spb_gain_med")
    x["r_cortical_share_bh"] = r("bh_ssr_cortical_od850", "bh_deep_od850")
    x["r_short_vs_max_hold"] = r("bh_short10s_ssr_amp", "bh_ssr_cortical_od850")
    x["r_bh_per_pulse"] = r("bh_deep_od850", "rs_pulse_amp")
    x["d_hr_delta_rel"] = df["bh_hr_delta"] / (df["rs_hr_bpm"] + EPS)
    return x


def _standardize(x_tr: np.ndarray, x_te: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    med = np.median(x_tr, axis=0)
    q25, q75 = np.percentile(x_tr, [25.0, 75.0], axis=0)
    iqr = np.where((q75 - q25) > 1e-12, q75 - q25, 1.0)
    return (x_tr - med) / iqr, (x_te - med) / iqr


# ---------- supervised ----------

def _fit_logreg(x: np.ndarray, y: np.ndarray):
    return LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000).fit(x, y)


def _fit_forest(x: np.ndarray, y: np.ndarray):
    return RandomForestClassifier(
        n_estimators=200, min_samples_leaf=2, class_weight="balanced_subsample", random_state=0, n_jobs=1
    ).fit(x, y)


def _fit_xgb(x: np.ndarray, y: np.ndarray):
    pos = max(int(y.sum()), 1)
    return XGBClassifier(
        n_estimators=120, max_depth=2, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=(len(y) - pos) / pos, random_state=0, n_jobs=1, verbosity=0,
    ).fit(x, y)


SUPERVISED: dict[str, Callable] = {"Regresja logistyczna": _fit_logreg, "Las losowy": _fit_forest}
if XGBClassifier is not None:
    SUPERVISED["XGBoost"] = _fit_xgb


def loo_supervised(x: np.ndarray, y: np.ndarray, fit: Callable) -> np.ndarray:
    """Out-of-fold probability for each subject from a model trained on everyone else."""
    n = len(y)
    scores = np.zeros(n)
    for i in range(n):
        tr = np.arange(n) != i
        x_tr, x_te = _standardize(x[tr], x[i : i + 1])
        scores[i] = fit(x_tr, y[tr]).predict_proba(x_te)[0, 1]
    return scores


def _permuted_auc(x: np.ndarray, y: np.ndarray, fit: Callable, seed: int) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_perm = np.random.default_rng(seed).permutation(y)
        return float(roc_auc_score(y_perm, loo_supervised(x, y_perm, fit)))


# ---------- unsupervised ----------

def _mahalanobis(x_tr: np.ndarray, x_te: np.ndarray) -> float:
    cov = LedoitWolf().fit(x_tr)
    return float(cov.mahalanobis(x_te)[0])


def _knn_distance(x_tr: np.ndarray, x_te: np.ndarray) -> float:
    k = min(5, len(x_tr) - 1)
    return float(NearestNeighbors(n_neighbors=k).fit(x_tr).kneighbors(x_te)[0].mean())


def _pca_error(x_tr: np.ndarray, x_te: np.ndarray) -> float:
    pca = PCA(n_components=3).fit(x_tr)
    return float(np.sum((x_te - pca.inverse_transform(pca.transform(x_te))) ** 2))


UNSUPERVISED: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "Isolation Forest": lambda a, b: float(-IsolationForest(n_estimators=300, random_state=0).fit(a).score_samples(b)[0]),
    "LOF": lambda a, b: float(
        -LocalOutlierFactor(n_neighbors=min(10, len(a) - 1), novelty=True).fit(a).score_samples(b)[0]
    ),
    "One-Class SVM": lambda a, b: float(-OneClassSVM(nu=0.1, gamma="scale").fit(a).score_samples(b)[0]),
    "Mahalanobis": _mahalanobis,
    "Odległość kNN": _knn_distance,
    "Błąd rekonstrukcji PCA": _pca_error,
}


def loo_unsupervised(x: np.ndarray, score: Callable[[np.ndarray, np.ndarray], float]) -> np.ndarray:
    n = len(x)
    out = np.zeros(n)
    for i in range(n):
        tr = np.arange(n) != i
        x_tr, x_te = _standardize(x[tr], x[i : i + 1])
        out[i] = score(x_tr, x_te)
    return out


# ---------- evaluation ----------

def _describe(scores: np.ndarray, y: np.ndarray, ids: np.ndarray) -> dict[str, object]:
    rank = pd.Series(scores).rank(ascending=False, method="max").astype(int).to_numpy()
    return {
        "auc": round(float(roc_auc_score(y, scores)), 3),
        "ranks": {int(i): int(r) for i, r in zip(ids[y], rank[y])},
        "review_to_catch_all": int(rank[y].max()),
    }


def evaluate(
    df: pd.DataFrame, feature_sets: dict[str, pd.DataFrame], positive_ids: list[int]
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    ids = df["ID"].to_numpy()
    y = df["ID"].isin(positive_ids).to_numpy()
    rows: list[dict[str, object]] = []
    table = pd.DataFrame({"ID": ids, "positive": y})
    for set_name, frame in feature_sets.items():
        x = frame.to_numpy(dtype=float)
        for name, score in UNSUPERVISED.items():
            s = loo_unsupervised(x, score)
            rows.append({"family": "bez etykiet", "features": set_name, "model": name, **_describe(s, y, ids)})
            table[f"{name} | {set_name}"] = s
        for name, fit in SUPERVISED.items():
            s = loo_supervised(x, y.astype(int), fit)
            null = np.array(
                Parallel(n_jobs=-1)(delayed(_permuted_auc)(x, y.astype(int), fit, seed) for seed in range(N_PERMUTATIONS))
            )
            desc = _describe(s, y, ids)
            rows.append(
                {
                    "family": "z etykietami", "features": set_name, "model": name, **desc,
                    "null_mean": round(float(null.mean()), 3),
                    "null_p95": round(float(np.percentile(null, 95)), 3),
                    "p_perm": round(float((1 + np.sum(null >= desc["auc"])) / (1 + len(null))), 3),
                }
            )
            table[f"{name} | {set_name}"] = s
    return rows, table


def _markdown(title: str, rows: list[dict[str, object]], reference: dict[str, object], n: int) -> str:
    pl = lambda v: str(v).replace(".", ",")  # noqa: E731
    ranks = lambda d: ", ".join(f"{k}: {v}" for k, v in d.items())  # noqa: E731
    ref_ranks = {p["id"]: p["rank"] for p in reference["positives"]}
    lines = [
        f"### {title}",
        "",
        "| Rodzina | Cechy | Model | AUC | Tło z permutacji (średnia / 95. pct) | p | Miejsca osób wyróżnionych | Ile przejrzeć |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :--- | :---: |",
        f"| bez etykiet | bazowe | **Dwie flagi (raport)** | **{pl(reference['auc'])}** | – | – | {ranks(ref_ranks)} | {reference['review_to_catch_all']} z {n} |",
    ]
    for r in rows:
        null = f"{pl(r['null_mean'])} / {pl(r['null_p95'])}" if "null_mean" in r else "–"
        p = pl(r["p_perm"]) if "p_perm" in r else "–"
        lines.append(
            f"| {r['family']} | {r['features']} | {r['model']} | {pl(r['auc'])} | {null} | {p} | {ranks(r['ranks'])} | {r['review_to_catch_all']} z {n} |"
        )
    return "\n".join(lines) + "\n"


def run_all(root_dir: Path | str) -> dict[str, object]:
    root = Path(root_dir)
    res = root / "results"
    sup_all = pd.read_csv(res / "supine_features_const_dpf.csv")
    sup = sup_all[sup_all["has_bs"] & sup_all["has_bh"]].reset_index(drop=True)
    dot = pd.read_csv(res / "dot_features_const_dpf.csv")
    two_flag = json.loads((res / "two_flag_summary.json").read_text(encoding="utf-8"))

    sup_sets = {"bazowe": _log_scaled(sup, SUPINE_FEATURE_COLS), "rozszerzone": engineer_supine(sup)}
    dot_sets = {"bazowe": _log_scaled(dot, DOT_FEATURE_COLS), "rozszerzone": engineer_dot(dot)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sup_rows, sup_table = evaluate(sup, sup_sets, SUPINE_POSITIVE_IDS)
        dot_rows, dot_table = evaluate(dot, dot_sets, DOT_POSITIVE_IDS)

    sup_table.to_csv(res / "simple_detectors_supine.csv", index=False)
    dot_table.to_csv(res / "simple_detectors_dot.csv", index=False)
    summary = {
        "n_permutations": N_PERMUTATIONS,
        "n_features": {
            "supine": {k: int(v.shape[1]) for k, v in sup_sets.items()},
            "dot": {k: int(v.shape[1]) for k, v in dot_sets.items()},
        },
        "supine": sup_rows,
        "dot": dot_rows,
    }
    (res / "simple_detectors_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    md = (
        "# Proste detektory: sprawdzenie\n\n"
        "Czy gotowe modele z półki radzą sobie równie dobrze jak reguła dwóch flag? Wszystko liczone\n"
        "metodą leave-one-out na zapisanych tabelach cech; wiek i BMI nie są cechami. Modele z etykietami\n"
        f"porównujemy z tłem z {N_PERMUTATIONS} permutacji etykiet przepuszczonych przez tę samą procedurę.\n"
        "Miejsce = pozycja w rankingu (1 = najbardziej podejrzany, remisy liczone na niekorzyść).\n\n"
        f"Cechy: Zbiór 1 – {summary['n_features']['supine']['bazowe']} bazowych, "
        f"{summary['n_features']['supine']['rozszerzone']} rozszerzonych; Zbiór 2 – "
        f"{summary['n_features']['dot']['bazowe']} bazowych, {summary['n_features']['dot']['rozszerzone']} rozszerzonych.\n\n"
        + _markdown("Zbiór 1: pozycja leżąca (4 z 58)", sup_rows, two_flag["supine"], len(sup))
        + "\n"
        + _markdown("Zbiór 2: DOT (3 z 17)", dot_rows, two_flag["dot"], len(dot))
        + "\nOdtworzenie: `uv run --with xgboost python -m src.simple_detectors`\n"
    )
    (root / "SIMPLE_DETECTORS.md").write_text(md, encoding="utf-8")
    return summary


if __name__ == "__main__":
    run_all(Path(__file__).resolve().parent.parent)
    print((Path(__file__).resolve().parent.parent / "SIMPLE_DETECTORS.md").read_text(encoding="utf-8"))
