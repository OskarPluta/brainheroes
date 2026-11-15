# brainheroes: two flags in the fNIRS signal

Blind detection of physiological anomalies in prefrontal fNIRS recordings, built for the
*Breath, Brain & Body* track of **Heroes of the Brain 2026** (Cortivision Spectrum C23 data).

The approach is normative modeling: learn what typical physiology looks like from the cohort
itself, with no clinical labels, and rank people by how far they deviate from it. Medication,
diagnoses and BMI from the questionnaire are used only afterwards, to check whether the ranking
puts the right people at the top.

## The idea

Each person gets two out-of-fold percentiles, both computed in nested leave-one-out
cross-validation without any labels:

| Flag | What it measures | Catches |
| :--- | :--- | :--- |
| **Deficit** | RMS of *one-sided* robust z-scores in the "blunted" direction: low HRV, weak Mayer waves, low left–right coherence, weak heart-rate reaction to breath holding (supine); weak 0.1 Hz amplification under paced breathing (DOT) | suppressed vascular and autonomic responses |
| **Decoupling** | PCA reconstruction error (SPE / Q-residual, k = 3) of the full feature profile | every single feature looks normal, but the usual relations between heart and cortex are broken |

The alarm is **the higher of the two**, re-calibrated against the out-of-fold reference so that
a threshold of 90 means roughly 10% alarms in a typical cohort. Taking the maximum instead of the
average matters: ID 38 (29 years old, beta-blocker) is abnormal only on decoupling (97.7th
percentile among the 45 people aged ≤ 42) and has an unremarkable deficit (29.5). An average would
hide that. There are no weights to tune; the only free parameter is the number of PCA
components, and the results are stable for k = 1…5.

## Results

Main analysis, k = 3, thresholds at the 90th and 85th out-of-fold percentile.

| Cohort and criterion | Flagged by metadata | ROC-AUC [95% CI] | p (MW) | Alarms ≥ 90 | Alarms ≥ 85 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Supine – A: cardiovascular drugs | 4 of 48 | 0.892 [0.744, 0.994] | 0.005 | 2/4 hit, 3 false | 3/4 hit, 4 false |
| Supine – B: A or BMI ≥ 30 | 7 of 48 | 0.833 [0.684, 0.957] | 0.003 | 3/7 hit, 2 false | 4/7 hit, 3 false |
| DOT – A: asthma | 1 of 17 | 0.906 (n = 1) | 0.110 | 1/1 hit, 2 false | 1/1 hit, 2 false |
| DOT – B: asthma or BMI ≥ 30 | 3 of 17 | 0.952 [0.827, 1.000] | 0.010 | 2/3 hit, 1 false | 2/3 hit, 1 false |

On the same supine cohort, off-the-shelf unsupervised detectors (Isolation Forest, LOF,
One-Class SVM, Mahalanobis, kNN, PCA error) reach AUC 0.52–0.77, and supervised models trained
on the other subjects' labels do not beat a label-permutation null. On DOT, with only 3
positives, supervised XGBoost reaches AUC 1.0 (permutation p ≈ 0.02). Details are in
[`docs/SIMPLE_DETECTORS.md`](docs/SIMPLE_DETECTORS.md).

**Read these numbers with care.**

- The positive groups have 1 to 7 people. This demonstrates the method; it does not measure
  its accuracy. p-values are descriptive and not corrected for multiple comparisons.
- Three of the four supine Criterion A subjects are the only participants over 70, and age
  alone gives AUC 0.983. In this cohort treatment cannot be separated from age. The evidence
  that the method detects more than age comes from ID 38 and from the DOT cohort, where
  everyone is 20–29.
- The tool points to people worth a closer look. It does not diagnose: healthy people also
  reach the top of the list.

Cohort definition: 60 people were recorded supine. Two have only one session (IDs 41, 59), and
10 controls fail the label-blind short-channel QC (IDs 3–11: 850 nm detector saturated; ID 35:
760 nm dropout). That leaves N = 48, and no Criterion A or B case is lost.

## Quick start

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                          # add --extra benchmark to include XGBoost in the benchmark
uv run brainheroes all           # full pipeline (feature extraction runs only if the CSVs are missing)
uv run pytest                    # tests
```

The extracted feature tables are committed in `results/`, so everything after feature
extraction runs without the raw data. Steps can also be run one at a time:

| Command | Writes | Time |
| :--- | :--- | ---: |
| `brainheroes features [--force]` | `results/*_features_*.csv` from raw SNIRF | ~3 min |
| `brainheroes score` | `results/two_flag_{supine,dot}.csv`, `results/two_flag_summary.json` | ~1 min |
| `brainheroes report` | `docs/REPORT.html`, `docs/REPORT.md` | 1 s |
| `brainheroes benchmark` | `docs/SIMPLE_DETECTORS.md`, `results/simple_detectors_*` | ~11 min on 16 cores |
| `brainheroes ablation` | `results/verification_summary.json`, `results/*_loocv_results.csv`, figure | ~1 min |

Global options `--data`, `--results` and `--docs` override the default directories.

### Data

The raw recordings are not part of this repository (Cortivision's terms limit their use to the
hackathon). Feature extraction expects the hackathon info pack at `infopack_and_data/`:

```
infopack_and_data/
├── fnirs_high_freq_supine/      # 60 people, 141.79 Hz, resting baseline (BS) + breath holds (BH)
│   ├── subjects_description.csv
│   └── data/{baseline,BH}/*.snirf
└── dot_montage_sitting/         # 17 people, 190 channels at 18.27 Hz: RS1–4, SPB1–4, BH1–2
    ├── subjects_description.csv
    └── data/{RS,SPB,BH}/*.snirf
```

## How it works

```
SNIRF ─► missing-sample repair ─► optical density ─► channel QC (SCI ≥ 0.70, no saturation/dropout)
      ─► TDDR motion correction ─► MBLL (HbO/HbR) ─► short-separation regression (scalp vs. cortex)
      ─► subject features (12 supine, 13 DOT) ─► log10 for skewed features
      ─► nested LOOCV: robust scaling + PCA + ridge fit without subject i
      ─► deficit and decoupling percentiles ─► alarm = max(deficit, decoupling)
```

- **Quality control is label-blind.** Supine subjects need at least one valid short (10 mm)
  and one valid long channel in both sessions, otherwise short-separation regression and pulse
  detection are impossible. They are excluded rather than imputed.
- **Features** come from the optical signal only: pulse rate and HRV from the cardiac band of
  short channels; Mayer-wave and respiratory amplitude, left–right coherence, breath-hold
  response amplitude (deep, cortical after regression, HbR/HbO ratio, spatial spread) and the
  heart-rate reaction to holds. DOT adds the 0.1 Hz gain and resonance sharpness under paced
  breathing. A constant pathlength factor keeps age out of the features; the age-dependent DPF
  is tested as a sensitivity analysis.
- **Nested leave-one-out.** For each subject i, everything is fit on the others. To turn
  i's raw scores into percentiles, an inner leave-one-out over the training subjects produces
  equally out-of-sample reference scores. Subject i never enters its own calibration, and
  the tests check this explicitly.

The full method, with the signal physics and the math, is in
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) (Polish).

## Repository layout

```
src/brainheroes/
├── cli.py             # `brainheroes` command
├── constants.py       # paths, feature sets, evaluation labels, thresholds
├── preprocessing.py   # SNIRF loading, SCI quality control, TDDR, MBLL, short-separation regression
├── features.py        # per-subject features for both cohorts, QC funnel, cached tables
├── normative.py       # nested LOOCV (T², SPE, ridge, virtual-PPG baseline) and deficit scores
├── two_flag.py        # main method: max(deficit, decoupling), summaries, ablations
├── report.py          # docs/REPORT.{html,md} (+ report_template.html)
├── detectors.py       # off-the-shelf detectors and supervised baselines with permutation test
└── ablation.py        # earlier 50/50 variant, single-score baselines, k- and DPF-sensitivity
results/               # feature tables and all computed scores (committed)
docs/                  # generated reports and methodology (Polish)
tests/                 # unit, leakage and results checks (data-dependent tests skip without the data)
```

## Documentation

- [`docs/REPORT.html`](docs/REPORT.html): interactive report with charts for both cohorts.
- [`docs/REPORT.md`](docs/REPORT.md): text version with all result tables.
- [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md): signal processing and modeling in detail.
- [`docs/SIMPLE_DETECTORS.md`](docs/SIMPLE_DETECTORS.md): comparison with standard detectors.
