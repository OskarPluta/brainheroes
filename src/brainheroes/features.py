"""Subject-level features for both cohorts.

Dataset 1, `fnirs_high_freq_supine`: 60 people at 141.79 Hz, a resting baseline (BS)
and a breath-hold session (BH). Dataset 2, `dot_montage_sitting`: 17 people, 190
channels at 18.27 Hz, four resting (RS) and four 0.1 Hz paced-breathing (SPB) runs and
two breath-hold sessions (BH1, BH2).

Every feature is computed from the optical signal only. Demographics are copied into
the table for evaluation and reporting, never used to compute features.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

from .constants import DATA_DIR, RESULTS_DIR
from .preprocessing import (
    CARDIAC_BAND_HZ,
    DROPOUT_V,
    SATURATION_V,
    SnirfRecording,
    butter_bandpass,
    butter_lowpass,
    load_snirf_recording,
    ssr_clean,
)

MAYER_BAND_HZ = (0.06, 0.14)
RESP_BAND_HZ = (0.16, 0.40)
SPB_BAND_HZ = (0.07, 0.13)  # around the 0.1 Hz paced-breathing frequency
HOLD_LOWPASS_HZ = 0.12

SUPINE_SHORT_MM, SUPINE_LONG_MM = 15.0, 22.0  # short channels are exactly 10 mm
DOT_SHORT_MM, DOT_LONG_MM = 20.0, 25.0

# Fallback when too few heart beats are found (healthy-adult resting values).
DEFAULT_HR_HRV = (70.0, 50.0, 50.0)
# Paced-breathing segments of every SPB run (s).
SPB_WINDOWS_S = ((30.0, 80.0), (140.0, 190.0), (250.0, 300.0))
# DOT stimulus codes: short (10 s) and maximal breath holds; the code differs by session.
DOT_SHORT_HOLD_CODES = ("72", "76")
DOT_MAX_HOLD_CODES = ("74", "78")

SessionExtractor = Callable[[SnirfRecording, np.ndarray, np.ndarray], dict[str, float]]

SUPINE_FEATURES_CSV = "supine_features_const_dpf.csv"
SUPINE_AGE_DPF_FEATURES_CSV = "supine_features_age_dpf.csv"
DOT_FEATURES_CSV = "dot_features_const_dpf.csv"


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def _normalized_pulse(od_short: np.ndarray, fs: float) -> np.ndarray:
    """Cardiac-band signal averaged over short channels after scaling each to unit SD."""
    pulse_ch = butter_bandpass(od_short, *CARDIAC_BAND_HZ, fs, order=3)
    return np.mean(pulse_ch / np.maximum(np.std(pulse_ch, axis=0, keepdims=True), 1e-8), axis=1)


def _find_beats(pulse: np.ndarray, fs: float, prominence_ref: np.ndarray | None = None) -> np.ndarray:
    """Systolic peaks at least 0.42 s apart (HR < 143 bpm)."""
    ref = pulse if prominence_ref is None else prominence_ref
    prom = max(float(np.std(ref) * 0.40), 1e-6)
    peaks, _ = signal.find_peaks(pulse, distance=max(int(0.42 * fs), 2), prominence=prom)
    return peaks


def extract_cardiac_hrv(
    od850_shallow: np.ndarray,
    t: np.ndarray,
    fs: float,
    max_duration_s: float | None = None,
) -> tuple[float, float, float, float]:
    """Pulse rate (bpm), SDNN (ms), RMSSD (ms) and pulse amplitude (1e-3 OD) from short channels."""
    if max_duration_s is not None:
        mask = t <= max_duration_s
        od850_shallow, t = od850_shallow[mask], t[mask]

    pulse_ch = butter_bandpass(od850_shallow, *CARDIAC_BAND_HZ, fs, order=3)
    if pulse_ch.ndim == 2:
        pulse_amp = float(np.median(np.std(pulse_ch, axis=0)) * 1e3)
        pulse = np.mean(pulse_ch / np.maximum(np.std(pulse_ch, axis=0, keepdims=True), 1e-8), axis=1)
    else:
        pulse_amp = float(np.std(pulse_ch) * 1e3)
        pulse = pulse_ch / max(float(np.std(pulse_ch)), 1e-8)

    peaks = _find_beats(pulse, fs)
    if len(peaks) < 5:
        return (*DEFAULT_HR_HRV, pulse_amp)
    ibi = np.diff(t[peaks]) * 1000.0
    ibi = ibi[(ibi >= 380.0) & (ibi <= 1550.0)]
    if len(ibi) < 4:
        return (*DEFAULT_HR_HRV, pulse_amp)

    med_ibi = float(np.median(ibi))
    ibi_clean = ibi[np.abs(ibi - med_ibi) < 0.32 * med_ibi]
    if len(ibi_clean) < 4:
        ibi_clean = ibi
    hr_bpm = float(60000.0 / np.median(ibi_clean))
    sdnn_ms = float(np.std(ibi_clean))
    diffs = np.diff(ibi_clean)
    diffs = diffs[np.abs(diffs) < 300.0]
    rmssd_ms = float(np.sqrt(np.mean(diffs**2))) if len(diffs) > 0 else sdnn_ms
    return hr_bpm, sdnn_ms, rmssd_ms, pulse_amp


def hr_reaction(
    pulse: np.ndarray, t: np.ndarray, fs: float, before: np.ndarray, during: np.ndarray
) -> tuple[float, float] | None:
    """Heart-rate rise during a hold: (90th pct of beat-to-beat HR minus pre-hold HR, SD of HR)."""
    p_pre = _find_beats(pulse[before], fs, prominence_ref=pulse)
    p_hold = _find_beats(pulse[during], fs, prominence_ref=pulse)
    if len(p_pre) <= 3 or len(p_hold) <= 3:
        return None
    hr_pre = float(60.0 / np.median(np.diff(t[before][p_pre])))
    hr_inst = 60.0 / np.diff(t[during][p_hold])
    hr_inst = hr_inst[(hr_inst > 40.0) & (hr_inst < 150.0)]
    if len(hr_inst) <= 2:
        return None
    return float(np.percentile(hr_inst, 90) - hr_pre), float(np.std(hr_inst))


def interhemispheric_coherence(band_sig: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
    """Median Pearson correlation over all left x right deep-channel pairs."""
    corrs = [
        float(np.corrcoef(band_sig[:, li], band_sig[:, ri])[0, 1])
        for li in np.flatnonzero(left)
        for ri in np.flatnonzero(right)
        if np.std(band_sig[:, li]) > 1e-7 and np.std(band_sig[:, ri]) > 1e-7
    ]
    return float(np.median(corrs)) if corrs else 0.0


def estimate_unmarked_hold_duration(lp_deep_od850: np.ndarray, t: np.ndarray, onset: float, max_search_s: float = 85.0) -> float:
    """Breath-hold duration when the end marker is missing.

    Takes the time of peak hypercapnic vasodilation (OD850 maximum) minus a ~4.5 s
    transit delay (the median offset on 118 marked holds).
    """
    mask = (t >= onset + 5.0) & (t <= min(float(t[-1]), onset + max_search_s))
    if np.sum(mask) < 10:
        return max(10.0, float(t[-1] - onset - 5.0))
    sig = lp_deep_od850[mask]
    sig_1d = np.median(sig - sig[:1, :], axis=1) if sig.ndim == 2 else sig - sig[0]
    peak_t = float((t[mask] - onset)[int(np.argmax(sig_1d))])
    return float(np.clip(peak_t - 4.5, 8.0, max_search_s))


def classify_qc_failure(rec: SnirfRecording, group: np.ndarray, prefix: str) -> str:
    """Why no channel in a distance group passed QC, e.g. `bs_short_850nm_saturated`."""
    idx = np.flatnonzero(group)
    if len(idx) == 0:
        return f"{prefix}_no_channels"
    p99_850 = np.percentile(rec.raw850[:, idx], 99, axis=0) > SATURATION_V
    p99_760 = np.percentile(rec.raw760[:, idx], 99, axis=0) > SATURATION_V
    p01_760 = np.percentile(rec.raw760[:, idx], 1, axis=0) < DROPOUT_V
    p01_850 = np.percentile(rec.raw850[:, idx], 1, axis=0) < DROPOUT_V
    checks = [("850nm_saturated", p99_850), ("760nm_saturated", p99_760), ("760nm_dropout", p01_760), ("850nm_dropout", p01_850)]
    reasons = [name for name, failed in checks if failed.any()]
    # Unless one hardware fault explains every channel, some failed on coupling (SCI) alone.
    if not any(failed.all() for _, failed in checks):
        reasons.append("low_sci")
    return f"{prefix}_" + "+".join(reasons)


def _fallback(mask: np.ndarray, default: np.ndarray) -> np.ndarray:
    """`mask`, or `default` if no channel survives (DOT keeps running on unvalidated channels)."""
    return mask if np.any(mask) else default


def _lowpassed(rec: SnirfRecording) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fs = rec.fs
    return tuple(butter_lowpass(x, HOLD_LOWPASS_HZ, fs) for x in (rec.od850, rec.od760, rec.hbo, rec.hbr))


def _window(t: np.ndarray, start: float, end: float) -> np.ndarray:
    return (t >= start) & (t <= min(float(t[-1]), end))


def _mean_or(values: list[float], default: float) -> float:
    return float(np.mean(values)) if values else default


def _demographics(row: pd.Series, id_col: str) -> dict[str, object]:
    num = lambda col: float(str(row[col]).replace(",", "."))
    height_m = num("Wzrost") / 100.0
    weight_kg = num("Waga")
    return {
        "ID": int(row[id_col]),
        "Sex": str(row["Płeć"]).strip(),
        "Age": float(row["Wiek"]),
        "Height_cm": height_m * 100.0,
        "Weight_kg": weight_kg,
        "BMI": round(weight_kg / height_m**2, 2),
        "Fitzpatrick": str(row["Fitzpatrick"]).strip(),
    }


# ---------------------------------------------------------------------------
# Dataset 1: supine
# ---------------------------------------------------------------------------

SUPINE_BS_FEATURES = [
    "bs_hr_bpm", "bs_sdnn_ms", "bs_rmssd_ms", "bs_pulse_amp", "bs_lr_mayer_corr",
    "bs_mayer_amp", "bs_cort_mayer_amp", "bs_resp_amp", "bs_mayer_resp_ratio",
]  # fmt: skip
SUPINE_BH_FEATURES = [
    "bh_dur_max", "bh_dur_mean", "bh_dur_hold3_est", "bh_ssr_cortical_od850", "bh_deep_od850",
    "bh_shal_od850", "bh_hbr_hbo_ratio", "bh_od760_850_ratio", "bh_spatial_cv", "bh_hr_delta",
    "bh_hr_std", "bh_cort_hr_coupling",
]  # fmt: skip


def _supine_baseline(rec: SnirfRecording, short: np.ndarray, deep: np.ndarray) -> dict[str, float]:
    """Resting features: pulse/HRV from short channels, Mayer and respiratory waves from deep ones."""
    left, right = deep & (rec.x_pos_mm < 0), deep & (rec.x_pos_mm > 0)
    hr, sdnn, rmssd, pulse_amp = extract_cardiac_hrv(rec.od850[:, short], rec.time, rec.fs)

    mayer = butter_bandpass(rec.od850, *MAYER_BAND_HZ, rec.fs)
    resp = butter_bandpass(rec.od850, *RESP_BAND_HZ, rec.fs)
    mayer_cortex, _ = ssr_clean(mayer[:, deep], mayer[:, short])
    return {
        "bs_hr_bpm": hr,
        "bs_sdnn_ms": sdnn,
        "bs_rmssd_ms": rmssd,
        "bs_pulse_amp": pulse_amp,
        "bs_lr_mayer_corr": interhemispheric_coherence(mayer, left, right),
        "bs_mayer_amp": float(np.median(np.std(mayer[:, deep], axis=0)) * 1e3),
        "bs_cort_mayer_amp": float(np.median(np.std(mayer_cortex, axis=0)) * 1e3),
        "bs_resp_amp": float(np.median(np.std(resp[:, deep], axis=0)) * 1e3),
        "bs_mayer_resp_ratio": float(np.median(np.std(mayer[:, deep], axis=0) / (np.std(resp[:, deep], axis=0) + 1e-9))),
    }


def _supine_breath_hold(rec: SnirfRecording, short: np.ndarray, deep: np.ndarray) -> dict[str, float]:
    """Per-hold hemodynamic and heart-rate responses, averaged over holds.

    Stimulus "2" marks a hold onset and "1" its end; a missing end marker is estimated
    from the signal (`bh_dur_hold3_est`).
    """
    t, fs = rec.time, rec.fs
    lp850, lp760, lphbo, lphbr = _lowpassed(rec)
    ssr850, _ = ssr_clean(lp850[:, deep], lp850[:, short])
    pulse = _normalized_pulse(rec.od850[:, short], fs)
    ends = rec.stims.get("1", np.array([]))

    durs, cort, deep_amp, shal_amp, hbr_hbo, od_ratio, sp_cv, hr_delta, hr_std = ([] for _ in range(9))
    dur_est = np.nan
    for onset in rec.stims.get("2", np.array([])):
        after = ends[ends > onset + 1.0]
        if len(after) > 0:
            dur = float(after[0] - onset)
            durs.append(dur)
        else:
            dur = dur_est = estimate_unmarked_hold_duration(lp850[:, deep], t, float(onset))

        before = (t >= onset - 15.0) & (t <= onset - 2.0)
        response = _window(t, onset + 2.0, onset + dur + 10.0)
        hold = _window(t, onset + 2.0, onset + dur)
        if np.sum(before) <= 10 or np.sum(response) <= 10:
            continue
        cort.append(float(np.median(np.ptp(ssr850[response], axis=0) * 1e3)))
        d_ch = np.ptp(lp850[response][:, deep], axis=0) * 1e3
        deep_amp.append(float(np.median(d_ch)))
        shal_amp.append(float(np.median(np.ptp(lp850[response][:, short], axis=0)) * 1e3))
        sp_cv.append(float(np.std(d_ch) / (np.mean(d_ch) + 1e-9)))
        hbr_hbo.append(
            float(np.median(np.ptp(lphbr[response][:, deep], axis=0) / (np.ptp(lphbo[response][:, deep], axis=0) + 1e-9)))
        )
        od_ratio.append(
            float(np.median(np.ptp(lp760[response][:, deep], axis=0) / (np.ptp(lp850[response][:, deep], axis=0) + 1e-9)))
        )
        reaction = hr_reaction(pulse, t, fs, before, hold)
        if reaction is not None:
            hr_delta.append(reaction[0])
            hr_std.append(reaction[1])

    out = {
        "bh_dur_max": float(max(durs)) if durs else dur_est,
        "bh_dur_mean": float(np.mean(durs)) if durs else dur_est,
        "bh_dur_hold3_est": dur_est,
        "bh_ssr_cortical_od850": float(np.mean(cort)),
        "bh_deep_od850": float(np.mean(deep_amp)),
        "bh_shal_od850": float(np.mean(shal_amp)),
        "bh_hbr_hbo_ratio": float(np.mean(hbr_hbo)),
        "bh_od760_850_ratio": float(np.mean(od_ratio)),
        "bh_spatial_cv": float(np.mean(sp_cv)),
        "bh_hr_delta": _mean_or(hr_delta, 0.0),
        "bh_hr_std": _mean_or(hr_std, 1.0),
    }
    out["bh_cort_hr_coupling"] = float(out["bh_ssr_cortical_od850"] / (out["bh_hr_std"] + 0.5))
    return out


@dataclass
class _Session:
    """QC outcome and features of one supine session (empty if the file is missing)."""

    n_short: int = 0
    n_deep: int = 0
    qc_ok: bool = False
    sci: float | None = None
    reasons: list[str] = field(default_factory=list)
    features: dict[str, float] = field(default_factory=dict)


def _supine_session(path: Path | None, prefix: str, age: float | None, extract: SessionExtractor) -> _Session:
    """Load one session, run QC and, if both short and deep channels survive, extract features."""
    if path is None:
        return _Session()
    rec = load_snirf_recording(path, age_years=age)
    short_all, deep_all = rec.dists_mm < SUPINE_SHORT_MM, rec.dists_mm >= SUPINE_LONG_MM
    short, deep = short_all & rec.valid_pairs, deep_all & rec.valid_pairs
    out = _Session(n_short=int(short.sum()), n_deep=int(deep.sum()), sci=float(np.mean(rec.sci)))
    out.qc_ok = out.n_short > 0 and out.n_deep > 0
    if not short.any():
        out.reasons.append(classify_qc_failure(rec, short_all, f"{prefix}_short"))
    if not deep.any():
        out.reasons.append(classify_qc_failure(rec, deep_all, f"{prefix}_deep"))
    if out.qc_ok:
        out.features = extract(rec, short, deep)
    return out


def extract_supine_features(data_root: Path | str = DATA_DIR, use_age_dpf: bool = False) -> pd.DataFrame:
    """One row per supine participant with demographics, QC outcome and signal features."""
    base = Path(data_root) / "fnirs_high_freq_supine"
    meta = pd.read_csv(base / "subjects_description.csv")
    rows = []
    for _, mrow in meta.iterrows():
        demo = _demographics(mrow, "Nr osoby badanej")
        sid = demo["ID"]
        age = demo["Age"] if use_age_dpf else None
        bs_files = sorted((base / "data" / "baseline").glob(f"{sid}_*.snirf"))
        bh_files = sorted((base / "data" / "BH").glob(f"*_{sid}_*.snirf"))
        bs = _supine_session(bs_files[0] if bs_files else None, "bs", age, _supine_baseline)
        bh = _supine_session(bh_files[0] if bh_files else None, "bh", age, _supine_breath_hold)

        reasons = ["missing_bs"] * (not bs_files) + ["missing_bh"] * (not bh_files) + bs.reasons + bh.reasons
        scis = [x.sci for x in (bs, bh) if x.sci is not None]
        row = {
            **demo,
            "Medication": str(mrow["Przyjmowane leki"]).strip(),
            "Physical_Activity": float(str(mrow["Ocena aktywności fizycznej"]).replace(",", ".")),
            "has_bs": bool(bs_files),
            "has_bh": bool(bh_files),
            "bs_n_valid_short": bs.n_short,
            "bs_n_valid_deep": bs.n_deep,
            "bh_n_valid_short": bh.n_short,
            "bh_n_valid_deep": bh.n_deep,
            "bs_qc_ok": bs.qc_ok,
            "bh_qc_ok": bh.qc_ok,
            "qc_pass": bool(bs_files) and bool(bh_files) and bs.qc_ok and bh.qc_ok,
            "qc_exclusion_reason": ";".join(reasons),
            "mean_sci": float(np.mean(scis)) if scis else np.nan,
        }
        for name in SUPINE_BS_FEATURES:
            row[name] = bs.features.get(name, np.nan)
        for name in SUPINE_BH_FEATURES:
            row[name] = bh.features.get(name, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Dataset 2: DOT sitting
# ---------------------------------------------------------------------------


def _dot_groups(rec: SnirfRecording) -> tuple[np.ndarray, np.ndarray]:
    short_all, deep_all = rec.dists_mm < DOT_SHORT_MM, rec.dists_mm >= DOT_LONG_MM
    return _fallback(short_all & rec.valid_pairs, short_all), _fallback(deep_all & rec.valid_pairs, deep_all)


def _band_power_fraction(x: np.ndarray, fs: float, band: tuple[float, float], total: tuple[float, float]) -> np.ndarray:
    f, pxx = signal.welch(x, fs=fs, nperseg=min(int(60.0 * fs), x.shape[0]), axis=0)
    in_band = (f >= band[0]) & (f <= band[1])
    in_total = (f >= total[0]) & (f <= total[1])
    return np.trapezoid(pxx[in_band], f[in_band], axis=0) / (np.trapezoid(pxx[in_total], f[in_total], axis=0) + 1e-12)


def _dot_rest_and_paced(rs: SnirfRecording, spb: SnirfRecording) -> dict[str, float]:
    """One RS/SPB pair: resting HRV and how much paced breathing amplifies 0.1 Hz oscillations."""
    short_r, deep_r = _dot_groups(rs)
    short_s, deep_s = _dot_groups(spb)
    deep_all = spb.dists_mm >= DOT_LONG_MM
    left_all, right_all = deep_all & (spb.x_pos_mm < -5.0), deep_all & (spb.x_pos_mm > 5.0)
    left = _fallback(left_all & spb.valid_pairs, left_all)
    right = _fallback(right_all & spb.valid_pairs, right_all)

    hr, sdnn, rmssd, pulse_amp = extract_cardiac_hrv(rs.od850[:, short_r], rs.time, rs.fs, max_duration_s=600.0)

    rest = butter_bandpass(rs.od850[rs.time <= 600.0], *SPB_BAND_HZ, rs.fs)
    rest_cortex, _ = ssr_clean(rest[:, deep_r], rest[:, short_r])
    paced_mask = np.zeros_like(spb.time, dtype=bool)
    for start, end in SPB_WINDOWS_S:
        paced_mask |= (spb.time >= start) & (spb.time <= end)
    paced = butter_bandpass(spb.od850, *SPB_BAND_HZ, spb.fs)
    paced_cortex, _ = ssr_clean(paced[:, deep_s], paced[:, short_s])

    var_ratio = lambda a, b: float(np.median(np.var(a, axis=0))) / (float(np.median(np.var(b, axis=0))) + 1e-12)
    q = _band_power_fraction(spb.od850[:, deep_s], spb.fs, (0.08, 0.12), (0.03, 0.30))
    return {
        "hr": hr,
        "sdnn": sdnn,
        "rmssd": rmssd,
        "pulse_amp": pulse_amp,
        "gain": var_ratio(paced[paced_mask][:, deep_s], rest[:, deep_r]),
        "ssr_gain": var_ratio(paced_cortex[paced_mask], rest_cortex),
        "lr_coherence": interhemispheric_coherence(paced[paced_mask], left, right),
        "q_factor": float(np.median(q)),
    }


def _dot_breath_hold(rec: SnirfRecording) -> dict[str, list[float]]:
    """Responses to the short (10 s) and maximal holds of one session; holds have no end marker."""
    t, fs = rec.time, rec.fs
    short, deep = _dot_groups(rec)
    lp850, lp760, lphbo, lphbr = _lowpassed(rec)
    ssr850, _ = ssr_clean(lp850[:, deep], lp850[:, short])
    pulse = _normalized_pulse(rec.od850[:, short], fs)
    out: dict[str, list[float]] = {k: [] for k in ("short10s", "dur", "deep", "ssr", "sp_cv", "hbr_hbo", "od_ratio", "hr_delta")}

    short_onsets = next((rec.stims[c] for c in DOT_SHORT_HOLD_CODES if c in rec.stims), np.array([]))
    for onset in short_onsets:
        response = _window(t, onset + 2.0, onset + 18.0)
        if np.sum(response) > 5:
            out["short10s"].append(float(np.median(np.ptp(ssr850[response], axis=0)) * 1e3))

    max_onsets = next((rec.stims[c] for c in DOT_MAX_HOLD_CODES if c in rec.stims), np.array([]))
    for i, onset in enumerate(max_onsets):
        cap = 65.0 if i + 1 >= len(max_onsets) else min(65.0, float(max_onsets[i + 1] - onset - 5.0))
        out["dur"].append(estimate_unmarked_hold_duration(lp850[:, deep], t, float(onset), max_search_s=cap))
        before = (t >= onset - 15.0) & (t <= onset - 2.0)
        response = _window(t, onset + 2.0, onset + 60.0)
        if np.sum(response) <= 10:
            continue
        d_ch = np.ptp(lp850[response][:, deep], axis=0) * 1e3
        out["deep"].append(float(np.median(d_ch)))
        out["ssr"].append(float(np.median(np.ptp(ssr850[response], axis=0)) * 1e3))
        out["sp_cv"].append(float(np.std(d_ch) / (np.mean(d_ch) + 1e-9)))
        out["hbr_hbo"].append(
            float(np.median(np.ptp(lphbr[response][:, deep], axis=0) / (np.ptp(lphbo[response][:, deep], axis=0) + 1e-9)))
        )
        out["od_ratio"].append(
            float(np.median(np.ptp(lp760[response][:, deep], axis=0) / (np.ptp(lp850[response][:, deep], axis=0) + 1e-9)))
        )
        reaction = hr_reaction(pulse, t, fs, before, response)
        if reaction is not None:
            out["hr_delta"].append(reaction[0])
    return out


def _load_dot_run(base: Path, sid: int, kind: str, name: str, age: float | None) -> SnirfRecording:
    return load_snirf_recording(min((base / "data" / kind).glob(f"{sid:02d}_*_{name}.snirf")), age_years=age)


def extract_dot_features(data_root: Path | str = DATA_DIR, use_age_dpf: bool = False) -> pd.DataFrame:
    """One row per DOT participant with demographics and signal features."""
    base = Path(data_root) / "dot_montage_sitting"
    meta = pd.read_csv(base / "subjects_description.csv")
    rows = []
    for _, mrow in meta.iterrows():
        demo = _demographics(mrow, "ID")
        sid = demo["ID"]
        age = demo["Age"] if use_age_dpf else None
        runs, holds, scis = [], [], []
        for s in range(1, 5):
            rs = _load_dot_run(base, sid, "RS", f"RS{s}", age)
            spb = _load_dot_run(base, sid, "SPB", f"SPB{s}", age)
            scis += [float(np.mean(rs.sci)), float(np.mean(spb.sci))]
            runs.append(_dot_rest_and_paced(rs, spb))
        for b in (1, 2):
            bh = _load_dot_run(base, sid, "BH", f"BH{b}", age)
            scis.append(float(np.mean(bh.sci)))
            holds.append(_dot_breath_hold(bh))

        def run(key: str, runs: list[dict] = runs) -> list[float]:
            return [r[key] for r in runs]

        def hold(key: str, holds: list[dict] = holds) -> list[float]:
            return [v for h in holds for v in h[key]]

        session_deep = [_mean_or(h["deep"], np.nan) for h in holds]
        rows.append(
            {
                **demo,
                "Chronic_Disease": str(mrow["Choroby przewlekłe"]).strip(),
                "Medication": str(mrow["Przyjmowane leki"]).strip(),
                "Physical_Activity": float(str(mrow["Ocena aktywności fizycznej"]).replace(",", ".")),
                "Treadmill_Speed": float(str(mrow["Prędkość bieżni"]).replace(",", ".")),
                "mean_sci": float(np.mean(scis)),
                "rs_hr_bpm": float(np.median(run("hr"))),
                "rs_sdnn_ms": float(np.median(run("sdnn"))),
                "rs_rmssd_ms": float(np.median(run("rmssd"))),
                "rs_pulse_amp": float(np.median(run("pulse_amp"))),
                "spb1_gain_01hz": float(runs[0]["gain"]),
                "spb_gain_med": float(np.median(run("gain"))),
                "spb_ssr_gain_med": float(np.median(run("ssr_gain"))),
                "spb_lr_coherence": float(np.median(run("lr_coherence"))),
                "spb_q_factor": float(np.median(run("q_factor"))),
                "bh_dur_est": float(np.median(hold("dur"))),
                "bh_hr_delta": _mean_or(hold("hr_delta"), 0.0),
                "bh_ssr_cortical_od850": float(np.mean(hold("ssr"))),
                "bh_deep_od850": float(np.mean(hold("deep"))),
                "bh_short10s_ssr_amp": float(np.mean([np.mean(h["short10s"]) for h in holds if h["short10s"]])),
                "bh_hbr_hbo_ratio": float(np.mean(hold("hbr_hbo"))),
                "bh_od760_850_ratio": float(np.mean(hold("od_ratio"))),
                "bh_spatial_cv": float(np.mean(hold("sp_cv"))),
                "bh_evening_morning_ratio": float(session_deep[1] / (session_deep[0] + 1e-9)),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cached tables
# ---------------------------------------------------------------------------


def build_feature_tables(data_root: Path | str = DATA_DIR, results_dir: Path = RESULTS_DIR, force: bool = False) -> None:
    """Extract all three feature tables from raw SNIRF files (skips existing CSVs unless `force`)."""
    results_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (SUPINE_FEATURES_CSV, extract_supine_features, False),
        (SUPINE_AGE_DPF_FEATURES_CSV, extract_supine_features, True),
        (DOT_FEATURES_CSV, extract_dot_features, False),
    ]
    for name, extract, use_age_dpf in jobs:
        path = results_dir / name
        if path.exists() and not force:
            print(f"keep    {path}")
            continue
        extract(data_root, use_age_dpf=use_age_dpf).to_csv(path, index=False)
        print(f"wrote   {path}")


def qc_funnel(supine_all: pd.DataFrame) -> dict[str, object]:
    """Who was recorded, who lacks a session, who failed channel QC, and how many remain."""
    complete = supine_all["has_bs"] & supine_all["has_bh"]
    failed = supine_all[complete & ~supine_all["qc_pass"]]
    return {
        "enrolled": len(supine_all),
        "incomplete_ids": [int(x) for x in supine_all.loc[~complete, "ID"]],
        "qc_failed_ids": [int(x) for x in failed["ID"]],
        "qc_failed_reasons": {int(r.ID): str(r.qc_exclusion_reason) for r in failed.itertuples()},
        "analyzed": int(supine_all["qc_pass"].sum()),
    }


def load_supine(results_dir: Path = RESULTS_DIR, age_dpf: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(everyone recorded, the analysed cohort that has both sessions and passes QC)."""
    supine_all = pd.read_csv(results_dir / (SUPINE_AGE_DPF_FEATURES_CSV if age_dpf else SUPINE_FEATURES_CSV))
    return supine_all, supine_all[supine_all["qc_pass"]].reset_index(drop=True)


def load_dot(results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    return pd.read_csv(results_dir / DOT_FEATURES_CSV)
