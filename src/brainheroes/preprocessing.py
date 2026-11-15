"""Signal preprocessing: SNIRF loading, channel quality control, TDDR, MBLL and short-separation regression."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy import signal

CARDIAC_BAND_HZ = (0.7, 2.5)
SATURATION_V = 4.90  # detector rails at ~5 V
DROPOUT_V = 0.005  # dark-current floor
SCI_THRESHOLD = 0.70  # Pollonini et al. (2014)


@dataclass
class SnirfRecording:
    """A dual-wavelength recording after missing-sample repair, TDDR and MBLL.

    Channel arrays are (time, pair). `raw*` are the repaired intensities (V),
    `od*` the motion-corrected optical densities, `hbo`/`hbr` concentrations (uM).
    """

    time: np.ndarray
    fs: float
    raw760: np.ndarray
    raw850: np.ndarray
    od760: np.ndarray
    od850: np.ndarray
    hbo: np.ndarray
    hbr: np.ndarray
    dists_mm: np.ndarray  # source-detector separation per pair
    x_pos_mm: np.ndarray  # pair midpoint; < 0 left hemisphere, > 0 right
    sci: np.ndarray
    valid_pairs: np.ndarray
    stims: dict[str, np.ndarray]


def butter_bandpass(sig: np.ndarray, low: float, high: float, fs: float, order: int = 3) -> np.ndarray:
    """Zero-phase Butterworth band-pass along axis 0 (cut-offs clipped below Nyquist)."""
    nyq = 0.5 * fs
    low_c = max(1e-4, min(low, nyq * 0.95))
    high_c = max(low_c + 1e-4, min(high, nyq * 0.98))
    sos = signal.butter(order, [low_c, high_c], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, sig, axis=0)


def butter_lowpass(sig: np.ndarray, high: float, fs: float, order: int = 3) -> np.ndarray:
    """Zero-phase Butterworth low-pass along axis 0."""
    high_c = max(1e-4, min(high, 0.5 * fs * 0.98))
    sos = signal.butter(order, high_c, btype="lowpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, sig, axis=0)


def tddr_repair(od: np.ndarray, fs: float, n_iter: int = 10) -> np.ndarray:
    """Temporal Derivative Distribution Repair (Fishburn et al., 2019).

    Single-sample hardware glitches (> 25 MAD and > 0.05 OD) are clamped first so the
    low-pass filter does not ring on them. TDDR then reweights the derivative of the
    < 0.5 Hz component with Tukey's biweight; the faster cardiac/respiratory component
    is added back untouched.
    """
    if od.shape[0] < 10:
        return od.copy()

    raw_deriv = np.diff(od, axis=0, prepend=od[:1])
    med_r = np.median(raw_deriv, axis=0, keepdims=True)
    sigma_r = 1.4826 * np.median(np.abs(raw_deriv - med_r), axis=0, keepdims=True) + 1e-12
    is_glitch = (np.abs(raw_deriv - med_r) > 25.0 * sigma_r) & (np.abs(raw_deriv - med_r) > 0.05)
    if np.any(is_glitch):
        clean_deriv = np.where(is_glitch, med_r, raw_deriv)
        od = od[:1] + np.cumsum(clean_deriv, axis=0) - clean_deriv[:1]

    low = butter_lowpass(od, min(0.5, fs * 0.4), fs, order=3)
    high = od - low
    deriv = np.diff(low, axis=0, prepend=low[:1])
    w = np.ones_like(deriv)
    for _ in range(n_iter):
        mu = np.sum(w * deriv, axis=0, keepdims=True) / (np.sum(w, axis=0, keepdims=True) + 1e-12)
        dev = np.abs(deriv - mu)
        sigma = 1.4826 * np.median(dev, axis=0, keepdims=True) + 1e-12
        u = dev / (4.685 * sigma)
        w = np.where(u < 1.0, (1.0 - u**2) ** 2, 0.0)
    repaired_low = np.cumsum(w * (deriv - mu), axis=0)
    repaired_low -= np.median(repaired_low, axis=0, keepdims=True)
    return repaired_low + high


def scholkmann_wolf_dpf(wavelength_nm: float, age_years: float) -> float:
    """Age- and wavelength-dependent differential pathlength factor (Scholkmann & Wolf, 2013)."""
    age = float(np.clip(age_years, 18.0, 85.0))
    lam = float(wavelength_nm)
    return float(223.3 + 0.05624 * age**0.8493 - 5.723e-7 * lam**3 + 0.001245 * lam**2 - 0.9025 * lam)


# Molar extinction coefficients, 1/(cm*mM). Rows: 760 nm, 850 nm; columns: HbO, HbR.
_EXTINCTION = np.array([[1.4865865, 3.843707], [2.5263910, 1.798643]], dtype=np.float64)
_EXTINCTION_INV = np.linalg.inv(_EXTINCTION)


def od_to_hb_mbll(
    od760: np.ndarray,
    od850: np.ndarray,
    dists_mm: np.ndarray,
    ppf: float = 6.0,
    age_years: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Modified Beer-Lambert law: optical density at 760/850 nm -> HbO, HbR (uM).

    By default a constant pathlength factor is used so no survey data (age) leaks into
    the signal features; pass `age_years` for the Scholkmann & Wolf DPF instead.
    """
    d_cm = np.maximum(np.asarray(dists_mm, dtype=np.float64) / 10.0, 0.5)
    if age_years is None:
        dpf760 = dpf850 = ppf
    else:
        dpf760 = scholkmann_wolf_dpf(760.0, age_years)
        dpf850 = scholkmann_wolf_dpf(850.0, age_years)
    od760_norm = od760 / (d_cm[None, :] * dpf760)
    od850_norm = od850 / (d_cm[None, :] * dpf850)
    hbo = (_EXTINCTION_INV[0, 0] * od760_norm + _EXTINCTION_INV[0, 1] * od850_norm) * 1000.0
    hbr = (_EXTINCTION_INV[1, 0] * od760_norm + _EXTINCTION_INV[1, 1] * od850_norm) * 1000.0
    return hbo, hbr


def ssr_clean(y_deep: np.ndarray, y_shallow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Short-separation regression (Saager & Berger 2005; Gagnon et al. 2011).

    Regresses the (median) shallow signal out of every deep channel with OLS on centred
    data and returns `(cortical residual, beta)`.
    """
    y_d = np.asarray(y_deep, dtype=np.float64)
    y_s = np.asarray(y_shallow, dtype=np.float64)
    x = np.median(y_s, axis=1, keepdims=True) if y_s.ndim == 2 else y_s[:, None]
    x_c = x - np.mean(x, axis=0, keepdims=True)
    if y_d.ndim == 1:
        y_c = y_d[:, None] - np.mean(y_d)
        beta = np.sum(x_c * y_c, axis=0) / (np.sum(x_c**2, axis=0) + 1e-12)
        return (y_c - x_c * beta[None, :])[:, 0], beta
    y_c = y_d - np.mean(y_d, axis=0, keepdims=True)
    beta = np.sum(x_c * y_c, axis=0) / (np.sum(x_c**2, axis=0) + 1e-12)
    return y_c - x_c * beta[None, :], beta


def hardware_failure(raw760: np.ndarray, raw850: np.ndarray) -> np.ndarray:
    """Per-pair saturation / dropout flags from raw intensities (any wavelength)."""
    return (
        (np.percentile(raw760, 99, axis=0) > SATURATION_V)
        | (np.percentile(raw850, 99, axis=0) > SATURATION_V)
        | (np.percentile(raw760, 1, axis=0) < DROPOUT_V)
        | (np.percentile(raw850, 1, axis=0) < DROPOUT_V)
        | (np.median(raw760, axis=0) < 0.01)
        | (np.median(raw850, axis=0) < 0.01)
    )


def compute_windowed_sci(
    raw760: np.ndarray,
    raw850: np.ndarray,
    od760: np.ndarray,
    od850: np.ndarray,
    fs: float,
    sci_threshold: float = SCI_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """Scalp coupling index (Pollonini et al., 2014) and per-pair validity.

    SCI is the median over 5 s windows of the 760/850 nm correlation in the cardiac
    band. A pair is invalid if it saturates, drops out, is flat, or has SCI below the
    threshold; invalid hardware gets SCI = 0.
    """
    n_pairs = raw760.shape[1]
    c760 = butter_bandpass(od760, *CARDIAC_BAND_HZ, fs, order=3)
    c850 = butter_bandpass(od850, *CARDIAC_BAND_HZ, fs, order=3)
    win = max(int(5.0 * fs), 10)
    n_win = od760.shape[0] // win
    if n_win > 0:
        w1 = c760[: n_win * win].reshape(n_win, win, n_pairs)
        w2 = c850[: n_win * win].reshape(n_win, win, n_pairs)
        w1 = w1 - np.mean(w1, axis=1, keepdims=True)
        w2 = w2 - np.mean(w2, axis=1, keepdims=True)
        num = np.sum(w1 * w2, axis=1)
        den = np.sqrt(np.sum(w1**2, axis=1) * np.sum(w2**2, axis=1)) + 1e-12
        med_sci = np.median(num / den, axis=0)
    else:
        med_sci = np.zeros(n_pairs, dtype=np.float64)

    flat = (np.std(od760, axis=0) < 1e-5) | (np.std(od850, axis=0) < 1e-5)
    broken = hardware_failure(raw760, raw850) | flat
    sci = np.where(broken, 0.0, med_sci)
    valid = ~broken & (sci >= sci_threshold)
    return sci, valid


def _read_name(group: h5py.Group) -> str:
    name = group["name"][()]
    return name.decode("utf-8") if isinstance(name, bytes) else str(name)


def load_snirf_recording(path: Path | str, ppf: float = 6.0, age_years: float | None = None) -> SnirfRecording:
    """Load a SNIRF file and run missing-sample repair, SCI quality control, TDDR and MBLL."""
    with h5py.File(path, "r") as f:
        nirs = f["nirs"]
        raw = nirs["data1/dataTimeSeries"][:].astype(np.float64, copy=True)
        t = nirs["data1/time"][:].astype(np.float64, copy=True)
        src = nirs["probe/sourcePos3D"][:].astype(np.float64)
        det = nirs["probe/detectorPos3D"][:].astype(np.float64)
        n_ch = raw.shape[1]
        n_pairs = n_ch // 2  # measurement lists: all 760 nm pairs, then the same pairs at 850 nm
        dists = np.zeros(n_pairs)
        x_pos = np.zeros(n_pairs)
        for i in range(n_pairs):
            ml = nirs[f"data1/measurementList{i + 1}"]
            si = int(np.squeeze(ml["sourceIndex"][()])) - 1
            di = int(np.squeeze(ml["detectorIndex"][()])) - 1
            dists[i] = float(np.linalg.norm(src[si] - det[di]))
            x_pos[i] = float(0.5 * (src[si, 0] + det[di, 0]))

        stims = {_read_name(nirs[k]): np.sort(nirs[f"{k}/data"][:, 0].astype(np.float64)) for k in nirs if k.startswith("stim")}
        tags = nirs["metaDataTags"]
        missing = tags["missing_sample"][:].astype(np.float64) if "missing_sample" in tags else np.zeros(len(t))

    # Interpolate missing samples and dark dropouts before the log transform.
    for ch in range(n_ch):
        bad = (missing > 0) | (raw[:, ch] <= 1e-4) | ~np.isfinite(raw[:, ch])
        if np.any(bad) and not np.all(bad):
            raw[bad, ch] = np.interp(t[bad], t[~bad], raw[~bad, ch])

    eps = 1e-5
    od = -np.log10(np.maximum(raw, eps) / np.maximum(np.median(raw, axis=0, keepdims=True), eps))
    fs = float(1.0 / np.median(np.diff(t)))
    raw760, raw850 = raw[:, :n_pairs], raw[:, n_pairs:]
    sci, valid = compute_windowed_sci(raw760, raw850, od[:, :n_pairs], od[:, n_pairs:], fs)

    od = tddr_repair(od, fs)
    od760, od850 = od[:, :n_pairs], od[:, n_pairs:]
    hbo, hbr = od_to_hb_mbll(od760, od850, dists, ppf=ppf, age_years=age_years)
    return SnirfRecording(
        time=t,
        fs=fs,
        raw760=raw760,
        raw850=raw850,
        od760=od760,
        od850=od850,
        hbo=hbo,
        hbr=hbr,
        dists_mm=dists,
        x_pos_mm=x_pos,
        sci=sci,
        valid_pairs=valid,
        stims=stims,
    )
