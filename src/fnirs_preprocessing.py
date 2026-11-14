"""Core fNIRS physical preprocessing: SNIRF loading, SCI quality control, TDDR, MBLL, and SSR."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import numpy as np
from scipy import signal


@dataclass
class SnirfRecording:
    """Container for a preprocessed multi-channel dual-wavelength SNIRF recording."""

    path: Path
    time: np.ndarray
    fs: float
    raw760: np.ndarray
    raw850: np.ndarray
    od760: np.ndarray
    od850: np.ndarray
    hbo: np.ndarray
    hbr: np.ndarray
    dists_mm: np.ndarray
    x_pos_mm: np.ndarray
    sci: np.ndarray
    valid_pairs: np.ndarray
    stims: Dict[str, np.ndarray]
    missing_fraction: float
    aux: Dict[str, np.ndarray]


def butter_bandpass(sig: np.ndarray, low: float, high: float, fs: float, order: int = 3) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter along axis 0."""
    nyq = 0.5 * fs
    low_c = max(1e-4, min(low, nyq * 0.95))
    high_c = max(low_c + 1e-4, min(high, nyq * 0.98))
    sos = signal.butter(order, [low_c, high_c], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, sig, axis=0)


def butter_lowpass(sig: np.ndarray, high: float, fs: float, order: int = 3) -> np.ndarray:
    """Zero-phase Butterworth lowpass filter along axis 0."""
    nyq = 0.5 * fs
    high_c = max(1e-4, min(high, nyq * 0.98))
    sos = signal.butter(order, high_c, btype="lowpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, sig, axis=0)


def tddr_repair(od: np.ndarray, fs: float, n_iter: int = 10) -> np.ndarray:
    """
    Temporal Derivative Distribution Repair (Fishburn et al., 2019).
    Parameter-free IRLS motion-artifact correction using Tukey's biweight
    on low-frequency (< 0.5 Hz) temporal derivatives, preceded by a strict
    hardware-discontinuity clamp (> 25 MAD and > 0.05 OD/sample) that prevents
    IIR filter ringing on extreme step/Dirac glitches while leaving physiological
    cardiac and respiratory waveforms 100% untouched.
    """
    if od.shape[0] < 10:
        return od.copy()

    # Pre-filter hardware glitch clamp: only clamp non-physiological single-sample
    # jumps (> 25 MAD and > 0.05 OD) so physiological cardiac upstrokes have w=1.0
    # identically and never suffer asymmetric rectification drift.
    raw_deriv = np.diff(od, axis=0, prepend=od[:1])
    med_r = np.median(raw_deriv, axis=0, keepdims=True)
    sigma_r = 1.4826 * np.median(np.abs(raw_deriv - med_r), axis=0, keepdims=True) + 1e-12
    is_glitch = (np.abs(raw_deriv - med_r) > 25.0 * sigma_r) & (np.abs(raw_deriv - med_r) > 0.05)
    if np.any(is_glitch):
        clean_deriv = np.where(is_glitch, med_r, raw_deriv)
        od_in = od[:1] + np.cumsum(clean_deriv, axis=0) - clean_deriv[:1]
    else:
        od_in = od

    # Standard Fishburn et al. (2019) TDDR on low-frequency (< 0.5 Hz) component
    cutoff = min(0.5, fs * 0.4)
    low = butter_lowpass(od_in, cutoff, fs, order=3)
    high = od_in - low
    deriv = np.diff(low, axis=0, prepend=low[:1])
    w = np.ones_like(deriv)
    for _ in range(n_iter):
        mu = np.sum(w * deriv, axis=0, keepdims=True) / (np.sum(w, axis=0, keepdims=True) + 1e-12)
        dev = np.abs(deriv - mu)
        sigma = 1.4826 * np.median(dev, axis=0, keepdims=True) + 1e-12
        u = dev / (4.685 * sigma)
        w = np.where(u < 1.0, (1.0 - u**2) ** 2, 0.0)
    repaired_deriv = w * (deriv - mu)
    repaired_low = np.cumsum(repaired_deriv, axis=0)
    repaired_low -= np.median(repaired_low, axis=0, keepdims=True)

    return repaired_low + high


def scholkmann_wolf_dpf(wavelength_nm: float, age_years: float) -> float:
    """
    General equation for age- and wavelength-dependent Differential Pathlength Factor (DPF)
    in the human frontal head (Scholkmann & Wolf, 2013, J. Biomed. Opt. 18(10), 105004).
    """
    alpha = 223.3
    beta = 0.05624
    gamma = 0.8493
    delta = -5.723e-7
    eta = 0.001245
    zeta = -0.9025
    age_clamped = float(np.clip(age_years, 18.0, 85.0))
    lam = float(wavelength_nm)
    return float(alpha + beta * (age_clamped**gamma) + delta * (lam**3) + eta * (lam**2) + zeta * lam)


def od_to_hb_mbll(
    od760: np.ndarray,
    od850: np.ndarray,
    dists_mm: np.ndarray,
    ppf: float = 6.0,
    age_years: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Modified Beer-Lambert Law (MBLL) converting dual-wavelength optical density
    (760 nm and 850 nm) into relative concentration changes of oxygenated (HbO)
    and deoxygenated (HbR) hemoglobin in micromolar (uM).

    If age_years is None (default), uses constant PPF=6.0 to prevent any survey
    metadata (age) from leaking into blind signal features. If age_years is provided,
    applies the Scholkmann & Wolf (2013) wavelength- and age-specific DPF.
    """
    # Molar extinction coefficients in 1 / (cm * mM) at 760 nm and 850 nm
    # Row 0: 760 nm [HbO, HbR], Row 1: 850 nm [HbO, HbR]
    ext_mat = np.array(
        [
            [1.4865865, 3.843707],
            [2.5263910, 1.798643],
        ],
        dtype=np.float64,
    )
    ext_inv = np.linalg.inv(ext_mat)
    d_cm = np.maximum(np.asarray(dists_mm, dtype=np.float64) / 10.0, 0.5)

    if age_years is not None:
        dpf760 = scholkmann_wolf_dpf(760.0, age_years)
        dpf850 = scholkmann_wolf_dpf(850.0, age_years)
        od760_norm = od760 / (d_cm[None, :] * dpf760)
        od850_norm = od850 / (d_cm[None, :] * dpf850)
    else:
        od760_norm = od760 / (d_cm[None, :] * ppf)
        od850_norm = od850 / (d_cm[None, :] * ppf)

    # Convert mM -> uM (* 1000.0)
    hbo = (ext_inv[0, 0] * od760_norm + ext_inv[0, 1] * od850_norm) * 1000.0
    hbr = (ext_inv[1, 0] * od760_norm + ext_inv[1, 1] * od850_norm) * 1000.0
    return hbo, hbr


def ssr_clean(y_deep: np.ndarray, y_shallow: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Short-Separation Regression (SSR, Saager & Berger 2005; Gagnon et al. 2011).
    Removes superficial scalp/systemic hemodynamics from deep cortical channels via OLS:
        y_cortex(t) = y_deep(t) - (beta_0 + beta_1 * y_shallow(t))
    Returns (y_cortex, beta_1).
    """
    y_d = np.asarray(y_deep, dtype=np.float64)
    y_s = np.asarray(y_shallow, dtype=np.float64)
    if y_s.ndim == 2:
        x = np.median(y_s, axis=1, keepdims=True)
    else:
        x = y_s[:, None]
    x_c = x - np.mean(x, axis=0, keepdims=True)
    if y_d.ndim == 1:
        y_c = y_d[:, None] - np.mean(y_d)
        beta1 = np.sum(x_c * y_c, axis=0) / (np.sum(x_c**2, axis=0) + 1e-12)
        res = (y_c - x_c * beta1[None, :])[:, 0]
        return res, beta1
    y_c = y_d - np.mean(y_d, axis=0, keepdims=True)
    beta1 = np.sum(x_c * y_c, axis=0) / (np.sum(x_c**2, axis=0) + 1e-12)
    res = y_c - x_c * beta1[None, :]
    return res, beta1


def compute_windowed_sci(
    raw760: np.ndarray,
    raw850: np.ndarray,
    od760: np.ndarray,
    od850: np.ndarray,
    fs: float,
    sci_threshold: float = 0.70,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes Pollonini et al. (2014) windowed Scalp Coupling Index (SCI) in the cardiac
    band (0.7-2.5 Hz) and flags channels with hardware saturation (> 4.90 V),
    dark-current dropout (< 0.005 V), or low cardiac coupling (SCI < sci_threshold).
    """
    n_pairs = raw760.shape[1]
    sci = np.zeros(n_pairs, dtype=np.float64)
    valid = np.ones(n_pairs, dtype=bool)

    c760 = butter_bandpass(od760, 0.7, 2.5, fs, order=3)
    c850 = butter_bandpass(od850, 0.7, 2.5, fs, order=3)
    win = max(int(5.0 * fs), 10)
    n_win = od760.shape[0] // win

    if n_win > 0:
        w1 = c760[: n_win * win].reshape(n_win, win, n_pairs)
        w2 = c850[: n_win * win].reshape(n_win, win, n_pairs)
        w1_c = w1 - np.mean(w1, axis=1, keepdims=True)
        w2_c = w2 - np.mean(w2, axis=1, keepdims=True)
        num = np.sum(w1_c * w2_c, axis=1)
        den = np.sqrt(np.sum(w1_c**2, axis=1) * np.sum(w2_c**2, axis=1)) + 1e-12
        med_sci = np.median(num / den, axis=0)
    else:
        med_sci = np.zeros(n_pairs, dtype=np.float64)

    p99_1 = np.percentile(raw760, 99, axis=0)
    p99_2 = np.percentile(raw850, 99, axis=0)
    p01_1 = np.percentile(raw760, 1, axis=0)
    p01_2 = np.percentile(raw850, 1, axis=0)
    med_1 = np.median(raw760, axis=0)
    med_2 = np.median(raw850, axis=0)
    std_od1 = np.std(od760, axis=0)
    std_od2 = np.std(od850, axis=0)

    for j in range(n_pairs):
        if (
            p99_1[j] > 4.90
            or p99_2[j] > 4.90
            or p01_1[j] < 0.005
            or p01_2[j] < 0.005
            or med_1[j] < 0.01
            or med_2[j] < 0.01
            or std_od1[j] < 1e-5
            or std_od2[j] < 1e-5
        ):
            valid[j] = False
            sci[j] = 0.0
        else:
            sci[j] = float(med_sci[j])
            if sci[j] < sci_threshold:
                valid[j] = False

    return sci, valid


def load_snirf_recording(
    path: Path | str,
    apply_tddr: bool = True,
    ppf: float = 6.0,
    age_years: Optional[float] = None,
    load_aux: bool = False,
) -> SnirfRecording:
    """Loads and preprocesses a SNIRF file with missing-sample repair, SCI QC, TDDR, and MBLL."""
    path = Path(path)
    with h5py.File(path, "r") as f:
        raw = f["nirs/data1/dataTimeSeries"][:].astype(np.float64, copy=True)
        t = f["nirs/data1/time"][:].astype(np.float64, copy=True)
        fs = float(1.0 / np.median(np.diff(t)))
        s3d = f["nirs/probe/sourcePos3D"][:].astype(np.float64)
        d3d = f["nirs/probe/detectorPos3D"][:].astype(np.float64)
        n_ch = raw.shape[1]
        n_pairs = n_ch // 2
        dists = np.zeros(n_pairs, dtype=np.float64)
        x_pos = np.zeros(n_pairs, dtype=np.float64)
        for i in range(n_pairs):
            ml = f[f"nirs/data1/measurementList{i + 1}"]
            si = int(np.squeeze(ml["sourceIndex"][()])) - 1
            di = int(np.squeeze(ml["detectorIndex"][()])) - 1
            dists[i] = float(np.linalg.norm(s3d[si] - d3d[di]))
            x_pos[i] = float(0.5 * (s3d[si, 0] + d3d[di, 0]))

        stims: Dict[str, np.ndarray] = {}
        for k in f["nirs"].keys():
            if k.startswith("stim"):
                name = f[f"nirs/{k}/name"][()]
                if isinstance(name, bytes):
                    name = name.decode("utf-8")
                stims[str(name)] = np.sort(f[f"nirs/{k}/data"][:, 0].astype(np.float64))

        if "missing_sample" in f["nirs/metaDataTags"]:
            miss = f["nirs/metaDataTags/missing_sample"][:].astype(np.float64)
        else:
            miss = np.zeros(len(t), dtype=np.float64)

        aux: Dict[str, np.ndarray] = {}
        if load_aux:
            for k in sorted(f["nirs"].keys()):
                if k.startswith("aux"):
                    name = f[f"nirs/{k}/name"][()]
                    if isinstance(name, bytes):
                        name = name.decode("utf-8")
                    aux[str(name)] = np.squeeze(f[f"nirs/{k}/dataTimeSeries"][:].astype(np.float64))

    missing_fraction = float(np.mean(miss > 0))

    # Interpolate missing samples and dark dropouts before log transform
    for ch in range(n_ch):
        bad = (miss > 0) | (raw[:, ch] <= 1e-4) | ~np.isfinite(raw[:, ch])
        if np.any(bad) and not np.all(bad):
            raw[bad, ch] = np.interp(t[bad], t[~bad], raw[~bad, ch])

    raw760 = raw[:, :n_pairs]
    raw850 = raw[:, n_pairs:]

    eps = 1e-5
    med = np.maximum(np.median(raw, axis=0, keepdims=True), eps)
    od = -np.log10(np.maximum(raw, eps) / med)
    od760_raw = od[:, :n_pairs]
    od850_raw = od[:, n_pairs:]

    sci, valid_pairs = compute_windowed_sci(raw760, raw850, od760_raw, od850_raw, fs)

    if apply_tddr:
        od_repaired = tddr_repair(od, fs)
        od760 = od_repaired[:, :n_pairs]
        od850 = od_repaired[:, n_pairs:]
    else:
        od760 = od760_raw
        od850 = od850_raw

    hbo, hbr = od_to_hb_mbll(od760, od850, dists, ppf=ppf, age_years=age_years)

    return SnirfRecording(
        path=path,
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
        valid_pairs=valid_pairs,
        stims=stims,
        missing_fraction=missing_fraction,
        aux=aux,
    )
