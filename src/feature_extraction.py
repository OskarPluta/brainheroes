"""Feature extraction for Supine (Dataset 1) and DOT Sitting (Dataset 2) fNIRS cohorts."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal

from src.fnirs_preprocessing import (
    butter_bandpass,
    butter_lowpass,
    load_snirf_recording,
    ssr_clean,
)


def extract_cardiac_hrv(
    od850_shallow: np.ndarray,
    t: np.ndarray,
    fs: float,
    max_duration_s: Optional[float] = None,
) -> Tuple[float, float, float, float]:
    """
    Extracts peripheral pulse rate (bpm), SDNN (ms), RMSSD (ms), and pulse amplitude (x1e3 OD)
    from the cardiac band (0.7 - 2.5 Hz) of short-separation channels.
    """
    if max_duration_s is not None:
        mask = t <= max_duration_s
        sig_in = od850_shallow[mask]
        t_in = t[mask]
    else:
        sig_in = od850_shallow
        t_in = t

    # Bandpass each shallow channel first, then average normalized cardiac wave for peak picking
    pulse_ch = butter_bandpass(sig_in, 0.7, 2.5, fs, order=3)
    if pulse_ch.ndim == 2:
        pulse_amp = float(np.median(np.std(pulse_ch, axis=0)) * 1e3)
        stds = np.maximum(np.std(pulse_ch, axis=0, keepdims=True), 1e-8)
        pulse = np.mean(pulse_ch / stds, axis=1)
    else:
        pulse_amp = float(np.std(pulse_ch) * 1e3)
        pulse = pulse_ch / max(float(np.std(pulse_ch)), 1e-8)

    min_dist = max(int(0.42 * fs), 2)
    prom = max(float(np.std(pulse) * 0.40), 1e-6)
    peaks, _ = signal.find_peaks(pulse, distance=min_dist, prominence=prom)
    if len(peaks) < 5:
        return 70.0, 50.0, 50.0, pulse_amp

    ibi = np.diff(t_in[peaks]) * 1000.0
    ibi = ibi[(ibi >= 380.0) & (ibi <= 1550.0)]
    if len(ibi) < 4:
        return 70.0, 50.0, 50.0, pulse_amp

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


def compute_interhemispheric_coherence(
    band_sig: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
) -> float:
    """Computes median Pearson correlation across all Left x Right deep channel pairs."""
    left_idx = np.where(left_mask)[0]
    right_idx = np.where(right_mask)[0]
    if len(left_idx) == 0 or len(right_idx) == 0:
        return 0.0
    corrs = []
    for li in left_idx:
        for ri in right_idx:
            s_l = band_sig[:, li]
            s_r = band_sig[:, ri]
            if np.std(s_l) > 1e-7 and np.std(s_r) > 1e-7:
                corrs.append(float(np.corrcoef(s_l, s_r)[0, 1]))
    return float(np.median(corrs)) if corrs else 0.0


def estimate_unmarked_hold_duration(
    lp_deep_od850: np.ndarray,
    t: np.ndarray,
    onset: float,
    max_search_s: float = 85.0,
) -> float:
    """
    Signal-driven estimation of breath-hold duration when the end marker is absent.
    Uses the time of peak hypercapnic cerebral vasodilation (OD850 maximum) minus
    the physiological transit delay (~4.5 s median offset validated on 118 marked holds).
    """
    end_t = min(float(t[-1]), onset + max_search_s)
    mask = (t >= onset + 5.0) & (t <= end_t)
    if np.sum(mask) < 10:
        return max(10.0, float(t[-1] - onset - 5.0))
    sig = lp_deep_od850[mask]
    if sig.ndim == 2:
        sig_rel = sig - sig[:1, :]
        sig_1d = np.median(sig_rel, axis=1)
    else:
        sig_1d = sig - sig[0]
    t_sub = t[mask] - onset
    pk_time = float(t_sub[int(np.argmax(sig_1d))])
    est_dur = float(np.clip(pk_time - 4.5, 8.0, max_search_s))
    return est_dur


def _classify_qc_failure(rec, dist_mask: np.ndarray, prefix: str) -> str:
    """Describes why zero channels passed quality control in a distance group."""
    idx = np.where(dist_mask)[0]
    if len(idx) == 0:
        return f"{prefix}_no_channels"
    p99_850 = np.percentile(rec.raw850[:, idx], 99, axis=0)
    p99_760 = np.percentile(rec.raw760[:, idx], 99, axis=0)
    p01_760 = np.percentile(rec.raw760[:, idx], 1, axis=0)
    p01_850 = np.percentile(rec.raw850[:, idx], 1, axis=0)
    reasons: List[str] = []
    if np.any(p99_850 > 4.90):
        reasons.append("850nm_saturated")
    if np.any(p99_760 > 4.90):
        reasons.append("760nm_saturated")
    if np.any(p01_760 < 0.005):
        reasons.append("760nm_dropout")
    if np.any(p01_850 < 0.005):
        reasons.append("850nm_dropout")
    if not reasons or not (
        np.all(p99_850 > 4.90)
        or np.all(p99_760 > 4.90)
        or np.all(p01_760 < 0.005)
        or np.all(p01_850 < 0.005)
    ):
        if "low_sci" not in reasons and not (
            np.all(p99_850 > 4.90)
            or np.all(p99_760 > 4.90)
            or np.all(p01_760 < 0.005)
            or np.all(p01_850 < 0.005)
        ):
            reasons.append("low_sci")
    return f"{prefix}_" + "+".join(reasons)


def extract_supine_features(
    data_root: Path | str,
    use_age_dpf: bool = False,
    cache_path: Optional[Path | str] = None,
) -> pd.DataFrame:
    """
    Extracts subject-level physical and physiological features for Dataset 1
    (fnirs_high_freq_supine, 60 participants at 141.79 Hz).
    """
    if cache_path is not None and Path(cache_path).exists():
        return pd.read_csv(cache_path)

    base_dir = Path(data_root) / "fnirs_high_freq_supine"
    meta = pd.read_csv(base_dir / "subjects_description.csv")

    rows: List[Dict[str, object]] = []
    for _, mrow in meta.iterrows():
        sid = int(mrow["Nr osoby badanej"])
        age = float(mrow["Wiek"])
        height_m = float(str(mrow["Wzrost"]).replace(",", ".")) / 100.0
        weight_kg = float(str(mrow["Waga"]).replace(",", "."))
        bmi = weight_kg / (height_m**2)
        pa_score = float(str(mrow["Ocena aktywności fizycznej"]).replace(",", "."))
        med = str(mrow["Przyjmowane leki"]).strip()

        bs_files = sorted((base_dir / "data" / "baseline").glob(f"{sid}_*.snirf"))
        bh_files = sorted((base_dir / "data" / "BH").glob(f"*_{sid}_*.snirf"))

        has_bs = len(bs_files) > 0
        has_bh = len(bh_files) > 0

        age_arg = age if use_age_dpf else None

        bs_hr, bs_sdnn, bs_rmssd, bs_pulse_amp = np.nan, np.nan, np.nan, np.nan
        bs_lr_mayer_corr, bs_mayer_amp, bs_resp_amp, bs_mayer_resp_ratio = np.nan, np.nan, np.nan, np.nan
        bs_cort_mayer_amp = np.nan
        bs_n_valid_short, bs_n_valid_deep = 0, 0
        bs_qc_ok = False
        qc_reasons: List[str] = []
        sci_list: List[float] = []

        if not has_bs:
            qc_reasons.append("missing_bs")
        if not has_bh:
            qc_reasons.append("missing_bh")

        if has_bs:
            rec_bs = load_snirf_recording(bs_files[0], apply_tddr=True, age_years=age_arg)
            sci_list.append(float(np.mean(rec_bs.sci)))
            dists = rec_bs.dists_mm
            x_pos = rec_bs.x_pos_mm
            val = rec_bs.valid_pairs

            shal = (dists < 15.0) & val
            deep = (dists >= 22.0) & val
            bs_n_valid_short = int(np.sum(shal))
            bs_n_valid_deep = int(np.sum(deep))
            bs_qc_ok = (bs_n_valid_short > 0) and (bs_n_valid_deep > 0)
            if bs_n_valid_short == 0:
                qc_reasons.append(_classify_qc_failure(rec_bs, dists < 15.0, "bs_short"))
            if bs_n_valid_deep == 0:
                qc_reasons.append(_classify_qc_failure(rec_bs, dists >= 22.0, "bs_deep"))

            if bs_qc_ok:
                left_d = (dists >= 22.0) & (x_pos < 0) & val
                right_d = (dists >= 22.0) & (x_pos > 0) & val

                bs_hr, bs_sdnn, bs_rmssd, bs_pulse_amp = extract_cardiac_hrv(
                    rec_bs.od850[:, shal], rec_bs.time, rec_bs.fs
                )

                # Filter per channel first, then aggregate across valid channels
                mayer_all = butter_bandpass(rec_bs.od850, 0.06, 0.14, rec_bs.fs)
                resp_all = butter_bandpass(rec_bs.od850, 0.16, 0.40, rec_bs.fs)

                bs_lr_mayer_corr = compute_interhemispheric_coherence(mayer_all, left_d, right_d)
                mayer_cort_ch, _ = ssr_clean(mayer_all[:, deep], mayer_all[:, shal])

                bs_mayer_amp = float(np.median(np.std(mayer_all[:, deep], axis=0)) * 1e3)
                bs_cort_mayer_amp = float(np.median(np.std(mayer_cort_ch, axis=0)) * 1e3)
                bs_resp_amp = float(np.median(np.std(resp_all[:, deep], axis=0)) * 1e3)
                bs_mayer_resp_ratio = float(
                    np.median(np.std(mayer_all[:, deep], axis=0) / (np.std(resp_all[:, deep], axis=0) + 1e-9))
                )

        bh_dur_max, bh_dur_mean, bh_dur_hold3_est = np.nan, np.nan, np.nan
        bh_ssr_cortical_od850, bh_deep_od850, bh_shal_od850 = np.nan, np.nan, np.nan
        bh_hbr_hbo_ratio, bh_od760_850_ratio, bh_spatial_cv = np.nan, np.nan, np.nan
        bh_hr_delta, bh_hr_std, bh_cort_hr_coupling = np.nan, np.nan, np.nan
        bh_n_valid_short, bh_n_valid_deep = 0, 0
        bh_qc_ok = False

        if has_bh:
            rec_bh = load_snirf_recording(bh_files[0], apply_tddr=True, age_years=age_arg)
            sci_list.append(float(np.mean(rec_bh.sci)))
            dists = rec_bh.dists_mm
            val = rec_bh.valid_pairs
            shal = (dists < 15.0) & val
            deep = (dists >= 22.0) & val
            bh_n_valid_short = int(np.sum(shal))
            bh_n_valid_deep = int(np.sum(deep))
            bh_qc_ok = (bh_n_valid_short > 0) and (bh_n_valid_deep > 0)
            if bh_n_valid_short == 0:
                qc_reasons.append(_classify_qc_failure(rec_bh, dists < 15.0, "bh_short"))
            if bh_n_valid_deep == 0:
                qc_reasons.append(_classify_qc_failure(rec_bh, dists >= 22.0, "bh_deep"))

            if bh_qc_ok:
                lp850 = butter_lowpass(rec_bh.od850, 0.12, rec_bh.fs)
                lp760 = butter_lowpass(rec_bh.od760, 0.12, rec_bh.fs)
                lphbo = butter_lowpass(rec_bh.hbo, 0.12, rec_bh.fs)
                lphbr = butter_lowpass(rec_bh.hbr, 0.12, rec_bh.fs)

                ssr850, _ = ssr_clean(lp850[:, deep], lp850[:, shal])
                pulse_ch = butter_bandpass(rec_bh.od850[:, shal], 0.7, 2.5, rec_bh.fs)
                pulse_bh = np.mean(pulse_ch / np.maximum(np.std(pulse_ch, axis=0, keepdims=True), 1e-8), axis=1)

                s1 = rec_bh.stims.get("1", np.array([]))
                s2 = rec_bh.stims.get("2", np.array([]))

                marked_durs: List[float] = []
                cort_amps: List[float] = []
                deep_amps: List[float] = []
                shal_amps: List[float] = []
                hbr_hbo_rats: List[float] = []
                od760_850_rats: List[float] = []
                sp_cvs: List[float] = []
                hr_deltas: List[float] = []
                hr_stds: List[float] = []

                for onset in s2:
                    after = s1[s1 > onset + 1.0]
                    if len(after) > 0:
                        dur = float(after[0] - onset)
                        marked_durs.append(dur)
                    else:
                        dur = estimate_unmarked_hold_duration(lp850[:, deep], rec_bh.time, float(onset))
                        bh_dur_hold3_est = dur

                    b_mask = (rec_bh.time >= onset - 15.0) & (rec_bh.time <= onset - 2.0)
                    r_mask = (rec_bh.time >= onset + 2.0) & (
                        rec_bh.time <= min(float(rec_bh.time[-1]), onset + dur + 10.0)
                    )
                    h_mask = (rec_bh.time >= onset + 2.0) & (
                        rec_bh.time <= min(float(rec_bh.time[-1]), onset + dur)
                    )

                    if np.sum(b_mask) > 10 and np.sum(r_mask) > 10:
                        c_ch = np.ptp(ssr850[r_mask], axis=0) * 1e3
                        cort_amps.append(float(np.median(c_ch)))
                        d_ch = np.ptp(lp850[r_mask][:, deep], axis=0) * 1e3
                        deep_amps.append(float(np.median(d_ch)))
                        shal_amps.append(float(np.median(np.ptp(lp850[r_mask][:, shal], axis=0)) * 1e3))
                        sp_cvs.append(float(np.std(d_ch) / (np.mean(d_ch) + 1e-9)))

                        hbo_ch = np.ptp(lphbo[r_mask][:, deep], axis=0)
                        hbr_ch = np.ptp(lphbr[r_mask][:, deep], axis=0)
                        hbr_hbo_rats.append(float(np.median(hbr_ch / (hbo_ch + 1e-9))))

                        od760_ch = np.ptp(lp760[r_mask][:, deep], axis=0)
                        od850_ch = np.ptp(lp850[r_mask][:, deep], axis=0)
                        od760_850_rats.append(float(np.median(od760_ch / (od850_ch + 1e-9))))

                        min_d = max(int(0.42 * rec_bh.fs), 2)
                        prom = max(float(np.std(pulse_bh) * 0.40), 1e-6)
                        p_pre, _ = signal.find_peaks(pulse_bh[b_mask], distance=min_d, prominence=prom)
                        p_h, _ = signal.find_peaks(pulse_bh[h_mask], distance=min_d, prominence=prom)
                        if len(p_pre) > 3 and len(p_h) > 3:
                            hr_pre = float(60.0 / np.median(np.diff(rec_bh.time[b_mask][p_pre])))
                            hr_inst = 60.0 / np.diff(rec_bh.time[h_mask][p_h])
                            hr_inst = hr_inst[(hr_inst > 40.0) & (hr_inst < 150.0)]
                            if len(hr_inst) > 2:
                                hr_deltas.append(float(np.percentile(hr_inst, 90) - hr_pre))
                                hr_stds.append(float(np.std(hr_inst)))

                bh_dur_max = float(max(marked_durs)) if marked_durs else bh_dur_hold3_est
                bh_dur_mean = float(np.mean(marked_durs)) if marked_durs else bh_dur_hold3_est
                bh_ssr_cortical_od850 = float(np.mean(cort_amps))
                bh_deep_od850 = float(np.mean(deep_amps))
                bh_shal_od850 = float(np.mean(shal_amps))
                bh_hbr_hbo_ratio = float(np.mean(hbr_hbo_rats))
                bh_od760_850_ratio = float(np.mean(od760_850_rats))
                bh_spatial_cv = float(np.mean(sp_cvs))
                bh_hr_delta = float(np.mean(hr_deltas)) if hr_deltas else 0.0
                bh_hr_std = float(np.mean(hr_stds)) if hr_stds else 1.0
                bh_cort_hr_coupling = float(bh_ssr_cortical_od850 / (bh_hr_std + 0.5))

        rows.append(
            {
                "ID": sid,
                "Sex": str(mrow["Płeć"]).strip(),
                "Age": age,
                "Height_cm": height_m * 100.0,
                "Weight_kg": weight_kg,
                "BMI": round(bmi, 2),
                "Fitzpatrick": str(mrow["Fitzpatrick"]).strip(),
                "Medication": med,
                "Physical_Activity": pa_score,
                "has_bs": has_bs,
                "has_bh": has_bh,
                "bs_n_valid_short": bs_n_valid_short,
                "bs_n_valid_deep": bs_n_valid_deep,
                "bh_n_valid_short": bh_n_valid_short,
                "bh_n_valid_deep": bh_n_valid_deep,
                "bs_qc_ok": bs_qc_ok,
                "bh_qc_ok": bh_qc_ok,
                "qc_pass": has_bs and has_bh and bs_qc_ok and bh_qc_ok,
                "qc_exclusion_reason": ";".join(qc_reasons),
                "mean_sci": float(np.mean(sci_list)) if sci_list else np.nan,
                "bs_hr_bpm": bs_hr,
                "bs_sdnn_ms": bs_sdnn,
                "bs_rmssd_ms": bs_rmssd,
                "bs_pulse_amp": bs_pulse_amp,
                "bs_lr_mayer_corr": bs_lr_mayer_corr,
                "bs_mayer_amp": bs_mayer_amp,
                "bs_cort_mayer_amp": bs_cort_mayer_amp,
                "bs_resp_amp": bs_resp_amp,
                "bs_mayer_resp_ratio": bs_mayer_resp_ratio,
                "bh_dur_max": bh_dur_max,
                "bh_dur_mean": bh_dur_mean,
                "bh_dur_hold3_est": bh_dur_hold3_est,
                "bh_ssr_cortical_od850": bh_ssr_cortical_od850,
                "bh_deep_od850": bh_deep_od850,
                "bh_shal_od850": bh_shal_od850,
                "bh_hbr_hbo_ratio": bh_hbr_hbo_ratio,
                "bh_od760_850_ratio": bh_od760_850_ratio,
                "bh_spatial_cv": bh_spatial_cv,
                "bh_hr_delta": bh_hr_delta,
                "bh_hr_std": bh_hr_std,
                "bh_cort_hr_coupling": bh_cort_hr_coupling,
            }
        )

    df = pd.DataFrame(rows)
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
    return df


def extract_dot_features(
    data_root: Path | str,
    use_age_dpf: bool = False,
    cache_path: Optional[Path | str] = None,
) -> pd.DataFrame:
    """
    Extracts subject-level physical and physiological features for Dataset 2
    (dot_montage_sitting, 17 participants, 190 channels at 18.27 Hz across RS1..4, SPB1..4, BH1..2).
    """
    if cache_path is not None and Path(cache_path).exists():
        return pd.read_csv(cache_path)

    base_dir = Path(data_root) / "dot_montage_sitting"
    meta = pd.read_csv(base_dir / "subjects_description.csv")

    rows: List[Dict[str, object]] = []
    for _, mrow in meta.iterrows():
        sid = int(mrow["ID"])
        age = float(mrow["Wiek"])
        height_m = float(str(mrow["Wzrost"]).replace(",", ".")) / 100.0
        weight_kg = float(str(mrow["Waga"]).replace(",", "."))
        bmi = weight_kg / (height_m**2)
        pa_score = float(str(mrow["Ocena aktywności fizycznej"]).replace(",", "."))
        treadmill = float(str(mrow["Prędkość bieżni"]).replace(",", "."))
        disease = str(mrow["Choroby przewlekłe"]).strip()
        med = str(mrow["Przyjmowane leki"]).strip()

        age_arg = age if use_age_dpf else None

        rs_hrs, rs_sdnns, rs_rmssds, rs_pamps = [], [], [], []
        gains_01, ssr_gains, lr_cohs, q_facs = [], [], [], []
        sci_list: List[float] = []

        for s_idx in range(1, 5):
            p_rs = sorted((base_dir / "data" / "RS").glob(f"{sid:02d}_*_RS{s_idx}.snirf"))[0]
            p_spb = sorted((base_dir / "data" / "SPB").glob(f"{sid:02d}_*_SPB{s_idx}.snirf"))[0]

            rec_r = load_snirf_recording(p_rs, apply_tddr=True, age_years=age_arg)
            rec_s = load_snirf_recording(p_spb, apply_tddr=True, age_years=age_arg)
            sci_list.extend([float(np.mean(rec_r.sci)), float(np.mean(rec_s.sci))])

            dists = rec_r.dists_mm
            x_pos = rec_r.x_pos_mm
            shal_r = (dists < 20.0) & rec_r.valid_pairs
            deep_r = (dists >= 25.0) & rec_r.valid_pairs
            if not np.any(shal_r):
                shal_r = dists < 20.0
            if not np.any(deep_r):
                deep_r = dists >= 25.0

            shal_s = (dists < 20.0) & rec_s.valid_pairs
            deep_s = (dists >= 25.0) & rec_s.valid_pairs
            if not np.any(shal_s):
                shal_s = dists < 20.0
            if not np.any(deep_s):
                deep_s = dists >= 25.0

            left_s = (dists >= 25.0) & (x_pos < -5.0) & rec_s.valid_pairs
            right_s = (dists >= 25.0) & (x_pos > 5.0) & rec_s.valid_pairs
            if not np.any(left_s):
                left_s = (dists >= 25.0) & (x_pos < -5.0)
            if not np.any(right_s):
                right_s = (dists >= 25.0) & (x_pos > 5.0)

            hr, sdnn, rmssd, pamp = extract_cardiac_hrv(
                rec_r.od850[:, shal_r], rec_r.time, rec_r.fs, max_duration_s=600.0
            )
            rs_hrs.append(hr)
            rs_sdnns.append(sdnn)
            rs_rmssds.append(rmssd)
            rs_pamps.append(pamp)

            # 0.1 Hz Mayer resonance gain per channel first, then spatial median
            rmask = rec_r.time <= 600.0
            r_bp = butter_bandpass(rec_r.od850[rmask], 0.07, 0.13, rec_r.fs)
            r_cort, _ = ssr_clean(r_bp[:, deep_r], r_bp[:, shal_r])

            smask = (
                ((rec_s.time >= 30.0) & (rec_s.time <= 80.0))
                | ((rec_s.time >= 140.0) & (rec_s.time <= 190.0))
                | ((rec_s.time >= 250.0) & (rec_s.time <= 300.0))
            )
            s_bp = butter_bandpass(rec_s.od850, 0.07, 0.13, rec_s.fs)
            s_cort, _ = ssr_clean(s_bp[:, deep_s], s_bp[:, shal_s])

            var_s_deep = float(np.median(np.var(s_bp[smask][:, deep_s], axis=0)))
            var_r_deep = float(np.median(np.var(r_bp[:, deep_r], axis=0)))
            gains_01.append(var_s_deep / (var_r_deep + 1e-12))

            var_s_cort = float(np.median(np.var(s_cort[smask], axis=0)))
            var_r_cort = float(np.median(np.var(r_cort, axis=0)))
            ssr_gains.append(var_s_cort / (var_r_cort + 1e-12))

            lr_cohs.append(compute_interhemispheric_coherence(s_bp[smask], left_s, right_s))

            f_w, pxx = signal.welch(
                rec_s.od850[:, deep_s],
                fs=rec_s.fs,
                nperseg=min(int(60.0 * rec_s.fs), len(rec_s.time)),
                axis=0,
            )
            p_01 = np.trapezoid(pxx[(f_w >= 0.08) & (f_w <= 0.12)], f_w[(f_w >= 0.08) & (f_w <= 0.12)], axis=0)
            p_tot = np.trapezoid(pxx[(f_w >= 0.03) & (f_w <= 0.30)], f_w[(f_w >= 0.03) & (f_w <= 0.30)], axis=0)
            q_facs.append(float(np.median(p_01 / (p_tot + 1e-12))))

        bh_deep_amps, bh_ssr_amps, bh_hbr_hbo, bh_od_rat, bh_sp_cvs = [], [], [], [], []
        bh_short10s_amps, bh_est_durs, bh_hr_deltas = [], [], []
        bh_session_deep: List[float] = []

        for b_idx in [1, 2]:
            p_bh = sorted((base_dir / "data" / "BH").glob(f"{sid:02d}_*_BH{b_idx}.snirf"))[0]
            rec_b = load_snirf_recording(p_bh, apply_tddr=True, age_years=age_arg)
            sci_list.append(float(np.mean(rec_b.sci)))

            dists = rec_b.dists_mm
            shal_b = (dists < 20.0) & rec_b.valid_pairs
            deep_b = (dists >= 25.0) & rec_b.valid_pairs
            if not np.any(shal_b):
                shal_b = dists < 20.0
            if not np.any(deep_b):
                deep_b = dists >= 25.0

            lp850 = butter_lowpass(rec_b.od850, 0.12, rec_b.fs)
            lp760 = butter_lowpass(rec_b.od760, 0.12, rec_b.fs)
            lphbo = butter_lowpass(rec_b.hbo, 0.12, rec_b.fs)
            lphbr = butter_lowpass(rec_b.hbr, 0.12, rec_b.fs)
            ssr850, _ = ssr_clean(lp850[:, deep_b], lp850[:, shal_b])
            pulse_ch = butter_bandpass(rec_b.od850[:, shal_b], 0.7, 2.5, rec_b.fs)
            pulse_b = np.mean(pulse_ch / np.maximum(np.std(pulse_ch, axis=0, keepdims=True), 1e-8), axis=1)

            short_onsets = rec_b.stims.get("72", rec_b.stims.get("76", np.array([])))
            s10_list = []
            for onset in short_onsets:
                rmask = (rec_b.time >= onset + 2.0) & (rec_b.time <= min(float(rec_b.time[-1]), onset + 18.0))
                if np.sum(rmask) > 5:
                    s10_list.append(float(np.median(np.ptp(ssr850[rmask], axis=0)) * 1e3))
            if s10_list:
                bh_short10s_amps.append(float(np.mean(s10_list)))

            max_onsets = rec_b.stims.get("74", rec_b.stims.get("78", np.array([])))
            sess_deep = []
            for idx_o, onset in enumerate(max_onsets):
                cap = 65.0
                if idx_o + 1 < len(max_onsets):
                    cap = min(cap, float(max_onsets[idx_o + 1] - onset - 5.0))
                est_dur = estimate_unmarked_hold_duration(lp850[:, deep_b], rec_b.time, float(onset), max_search_s=cap)
                bh_est_durs.append(est_dur)
                b_mask = (rec_b.time >= onset - 15.0) & (rec_b.time <= onset - 2.0)
                r_mask = (rec_b.time >= onset + 2.0) & (rec_b.time <= min(float(rec_b.time[-1]), onset + 60.0))
                if np.sum(r_mask) > 10:
                    d_ch = np.ptp(lp850[r_mask][:, deep_b], axis=0) * 1e3
                    sess_deep.append(float(np.median(d_ch)))
                    bh_deep_amps.append(float(np.median(d_ch)))
                    bh_ssr_amps.append(float(np.median(np.ptp(ssr850[r_mask], axis=0)) * 1e3))
                    bh_sp_cvs.append(float(np.std(d_ch) / (np.mean(d_ch) + 1e-9)))
                    hbo_ch = np.ptp(lphbo[r_mask][:, deep_b], axis=0)
                    hbr_ch = np.ptp(lphbr[r_mask][:, deep_b], axis=0)
                    bh_hbr_hbo.append(float(np.median(hbr_ch / (hbo_ch + 1e-9))))
                    od760_ch = np.ptp(lp760[r_mask][:, deep_b], axis=0)
                    od850_ch = np.ptp(lp850[r_mask][:, deep_b], axis=0)
                    bh_od_rat.append(float(np.median(od760_ch / (od850_ch + 1e-9))))

                    min_d = max(int(0.42 * rec_b.fs), 2)
                    prom = max(float(np.std(pulse_b) * 0.40), 1e-6)
                    p_pre, _ = signal.find_peaks(pulse_b[b_mask], distance=min_d, prominence=prom)
                    p_h, _ = signal.find_peaks(pulse_b[r_mask], distance=min_d, prominence=prom)
                    if len(p_pre) > 3 and len(p_h) > 3:
                        hr_pre = float(60.0 / np.median(np.diff(rec_b.time[b_mask][p_pre])))
                        hr_inst = 60.0 / np.diff(rec_b.time[r_mask][p_h])
                        hr_inst = hr_inst[(hr_inst > 40.0) & (hr_inst < 150.0)]
                        if len(hr_inst) > 2:
                            bh_hr_deltas.append(float(np.percentile(hr_inst, 90) - hr_pre))

            bh_session_deep.append(float(np.mean(sess_deep)) if sess_deep else np.nan)

        rows.append(
            {
                "ID": sid,
                "Sex": str(mrow["Płeć"]).strip(),
                "Age": age,
                "Height_cm": height_m * 100.0,
                "Weight_kg": weight_kg,
                "BMI": round(bmi, 2),
                "Fitzpatrick": str(mrow["Fitzpatrick"]).strip(),
                "Chronic_Disease": disease,
                "Medication": med,
                "Physical_Activity": pa_score,
                "Treadmill_Speed": treadmill,
                "mean_sci": float(np.mean(sci_list)),
                "rs_hr_bpm": float(np.median(rs_hrs)),
                "rs_sdnn_ms": float(np.median(rs_sdnns)),
                "rs_rmssd_ms": float(np.median(rs_rmssds)),
                "rs_pulse_amp": float(np.median(rs_pamps)),
                "spb1_gain_01hz": float(gains_01[0]),
                "spb_gain_med": float(np.median(gains_01)),
                "spb_ssr_gain_med": float(np.median(ssr_gains)),
                "spb_lr_coherence": float(np.median(lr_cohs)),
                "spb_q_factor": float(np.median(q_facs)),
                "bh_dur_est": float(np.median(bh_est_durs)),
                "bh_hr_delta": float(np.mean(bh_hr_deltas)) if bh_hr_deltas else 0.0,
                "bh_ssr_cortical_od850": float(np.mean(bh_ssr_amps)),
                "bh_deep_od850": float(np.mean(bh_deep_amps)),
                "bh_short10s_ssr_amp": float(np.mean(bh_short10s_amps)),
                "bh_hbr_hbo_ratio": float(np.mean(bh_hbr_hbo)),
                "bh_od760_850_ratio": float(np.mean(bh_od_rat)),
                "bh_spatial_cv": float(np.mean(bh_sp_cvs)),
                "bh_evening_morning_ratio": float(bh_session_deep[1] / (bh_session_deep[0] + 1e-9)),
            }
        )

    df = pd.DataFrame(rows)
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
    return df
