"""Signal-processing building blocks on synthetic data."""

from __future__ import annotations

import numpy as np

from brainheroes.preprocessing import compute_windowed_sci, od_to_hb_mbll, scholkmann_wolf_dpf, ssr_clean, tddr_repair


def test_mbll_scales_with_pathlength_and_keeps_hbr_hbo_ratio():
    rng = np.random.default_rng(42)
    od760 = rng.uniform(0.01, 0.05, size=(200, 4))
    od850 = rng.uniform(0.01, 0.05, size=(200, 4))
    dists = np.array([10.0, 25.0, 30.0, 35.0])
    hbo_6, hbr_6 = od_to_hb_mbll(od760, od850, dists, ppf=6.0)
    hbo_8, hbr_8 = od_to_hb_mbll(od760, od850, dists, ppf=8.0)
    np.testing.assert_allclose(hbo_6 * 6.0 / 8.0, hbo_8, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(hbr_6 * 6.0 / 8.0, hbr_8, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(hbr_6 / hbo_6, hbr_8 / hbo_8, rtol=1e-11, atol=1e-12)


def test_age_dpf_grows_with_age_and_depends_on_wavelength():
    assert scholkmann_wolf_dpf(760.0, 75.0) > scholkmann_wolf_dpf(760.0, 21.0)
    assert scholkmann_wolf_dpf(760.0, 21.0) != scholkmann_wolf_dpf(850.0, 21.0)


def test_ssr_residual_is_orthogonal_to_the_shallow_signal():
    t = np.linspace(0, 100, 1000)
    scalp = np.sin(2 * np.pi * 0.1 * t) + 0.5 * np.cos(2 * np.pi * 1.2 * t)
    cortex = 0.3 * np.sin(2 * np.pi * 0.04 * t)
    deep = np.column_stack([2.5 * scalp + cortex, 0.8 * scalp - 0.5 * cortex])
    residual, _ = ssr_clean(deep, scalp[:, None])
    assert residual.shape == deep.shape
    for ch in range(residual.shape[1]):
        assert abs(float(np.mean(residual[:, ch] * (scalp - scalp.mean())))) < 1e-10


def test_tddr_removes_step_and_spike_but_keeps_physiology():
    fs = 20.0
    t = np.arange(0, 120, 1.0 / fs)
    physio = 0.01 * np.sin(2 * np.pi * 0.1 * t) + 0.003 * np.sin(2 * np.pi * 1.2 * t)
    artifact = np.where(t >= 60.0, 0.5, 0.0)
    artifact[int(30 * fs)] += 1.2
    corrupted = (physio + artifact)[:, None]
    repaired = tddr_repair(corrupted, fs=fs)
    assert np.std(repaired) < 0.05 * np.std(corrupted)
    assert np.corrcoef(repaired[:, 0], physio)[0, 1] > 0.85


def test_tddr_does_not_rectify_an_asymmetric_cardiac_pulse():
    fs = 141.79
    t = np.arange(0, 60.0, 1.0 / fs)
    pulse = 0.008 * (np.sin(2 * np.pi * 1.15 * t) + 0.45 * np.sin(2 * np.pi * 2.30 * t + 0.8))
    clean = (pulse + 0.005 * np.sin(2 * np.pi * 0.10 * t))[:, None]
    repaired = tddr_repair(clean, fs=fs)
    assert np.corrcoef(repaired[:, 0], clean[:, 0])[0, 1] > 0.99
    assert np.std(repaired[:, 0] - clean[:, 0]) < 0.10 * np.std(clean[:, 0])


def test_sci_rejects_railing_and_dropout_even_when_wavelengths_correlate():
    fs = 50.0
    t = np.arange(0, 30, 1.0 / fs)
    cardiac = 0.02 * np.sin(2 * np.pi * 1.2 * t)
    # Channel 1 drops to 0 V and then rails at 4.98 V on both wavelengths: perfectly
    # correlated, so it would get SCI = 1 without the hardware checks.
    railing = np.where(t < 15.0, 0.0, 4.98)
    raw760 = np.column_stack([1.5 * np.exp(-cardiac), railing])
    raw850 = np.column_stack([1.8 * np.exp(-1.1 * cardiac), railing])
    od = lambda raw: -np.log10(np.maximum(raw, 1e-5) / np.maximum(np.median(raw, axis=0, keepdims=True), 1e-5))
    sci, valid = compute_windowed_sci(raw760, raw850, od(raw760), od(raw850), fs)
    assert bool(valid[0]) and sci[0] > 0.95
    assert not bool(valid[1]) and sci[1] == 0.0
