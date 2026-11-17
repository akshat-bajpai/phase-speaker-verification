# Who Said "Stop"? — GMM-UBM speaker verification with phase-based features

A text-dependent speaker-verification system for the competition *Who Said
"Stop"?*.  Every numerical operation — feature extraction, k-means, GMM/EM
training, MAP adaptation, log-likelihood-ratio scoring, score normalisation and
all metrics (EER/FAR/FRR/DET/min-DCF/bootstrap CIs) — is implemented in
**PyTorch** using only `torch` and `torchaudio`.  NumPy, SciPy, scikit-learn,
librosa and TensorFlow are *not* used anywhere in the pipeline.  `matplotlib` is
used solely to draw figures from values that torch has already computed.

**Headline result:** `1.02%` EER on the labelled development trials
(95% bootstrap CI 0.49–1.68%), using instantaneous-frequency-deviation features,
a 128-component diagonal-covariance UBM and mean-only MAP adaptation with
`r = 16`.

---

## 1. Requirements and setup

```bash
python3 -m venv .venv
.venv/bin/pip install torch torchaudio torchcodec matplotlib
```

* `torch` / `torchaudio` — the whole pipeline.
* `torchcodec` — only the WAV decoder behind `torchaudio.load`.  If it is
  unavailable the code falls back automatically to a reader built from Python's
  standard-library `wave` module plus `torch.frombuffer`
  (`svcore/data.py::_load_wave_stdlib`), so the project runs with no codec
  installed.
* `matplotlib` — figures only.

Unpack the competition archive so the tree looks like:

```
data/
  audio/audio/<file_id>.wav      3601 recordings, 16 kHz mono 16-bit
  background.csv  dev_enrol.csv  dev_trials.csv
  eval_enrol.csv  eval_trials.csv  sample_submission.csv
```

## 2. Reproducing everything

```bash
./run_all.sh                 # ~25 min on an 8-core CPU, no GPU needed
```

or step by step:

| command | produces |
|---|---|
| `python scripts/01_features.py` | feature definition, VAD retention, shift-invariance audit |
| `python scripts/02_gmm_behaviour.py` | Task 2: covariance structures, BIC/AIC, 25 initialisations, degeneracy + remedies, PCA plot |
| `python scripts/03_train_ubm.py` | Task 3: mixture-count sweep, convergence, held-out likelihood, timing |
| `python scripts/04_enrol_score.py` | Tasks 4–5: relevance factor, adaptation variants, DET/ROC, operating points, `results/dev_results.csv` |
| `python scripts/05_ablations.py` | Task 6: all ablations incl. the magnitude-leakage compliance audit |
| `python scripts/06_make_submission.py` | `submissions/submission_final.csv` (T-norm, primary) |
| `python scripts/06_make_submission.py --norm none --tag rawllr` | `submissions/submission_rawllr.csv` (raw LLR, backup) |
| `python tests/test_modules.py` | 49 analytic module checks → `results/log_tests.txt` |

Outputs land in `figures/` (PNG), `results/` (text logs + JSON) and
`submissions/`.  Feature caches and UBM checkpoints are written to `cache/`;
delete that directory to force a full recomputation.

## 3. Reproducibility

* **Global seed `2026`** (`svcore/config.py::SEED`) — the same seed the
  organisers used to split the data.  It drives the background speaker split,
  k-means/EM initialisation, frame subsampling and the bootstrap.
* **Device / precision**: CPU, `float32` features, `float64` for EM, MAP and
  scoring (set in `GMMConfig`).  No GPU is required and no result depends on
  one.
* **Stopping criterion**: EM stops when the relative change of the mean
  log-likelihood falls below `1e-4`, capped at 100 iterations.
* **Background splits are by speaker**, never by file
  (`svcore/data.py::background_splits`): 90% of the 786 background speakers
  (1287 recordings) train the UBM, the remaining 10% (149 recordings) are the
  held-out set used for every held-out-likelihood number, and 150 UBM-split
  speakers with ≥3 recordings form the T-norm cohort.
* Evaluation recordings are used **only** as the designated evaluation
  enrolment utterances and as evaluation test recordings.  Nothing — not the
  normalisation statistics, not the UBM, not the cohort — is fitted on them.

## 4. Code layout

```
svcore/
  config.py      FeatureConfig / GMMConfig / EnrolConfig / ScoreConfig, paths, seed
  data.py        protocol CSVs (stdlib csv), speaker-disjoint splits, WAV loading
  features.py    framing, STFT, energy VAD, the four phase representations,
                 band aggregation, DCT, deltas, CMVN
  normalise.py   background-fitted global standardisation
  gmm.py         GMM with full/diag/tied/spherical covariance, k-means++ init,
                 EM, BIC/AIC, occupancy and conditioning diagnostics
  map_adapt.py   Baum-Welch statistics, MAP mean/weight/covariance adaptation,
                 train-from-scratch baseline
  scoring.py     average per-frame LLR, batched mean-bank scorer, top-C scoring,
                 T-norm / Z-norm / ZT-norm
  metrics.py     ROC/DET points, EER, FAR/FRR, min-DCF, AUC, bootstrap CIs
  system.py      the selected configuration + helpers shared by the scripts
  plots.py       matplotlib figures (visualisation only)
scripts/         01..06, one per task group
report.md        the written report
```

## 5. The system in one page

1. **Front end.** 25 ms Hann window, **3 ms hop**, `n_fft = 512`.
2. **Phase representation — instantaneous-frequency deviation (IFD).**
   `Δ(t,k) = princarg(φ(t,k) − φ(t−1,k) − 2πk·hop/n_fft)`: the temporal
   (baseband) phase difference.  A constant time shift of the waveform
   multiplies every bin by `exp(−j2πkd/N)`, which is identical in consecutive
   frames and therefore **cancels exactly** — essential here, because the
   organisers prepended 0–30 ms of random leading audio to every clip.
3. **Band aggregation.** Magnitude-weighted *circular* mean of `e^{jΔ}` inside
   32 mel bands, kept as the `(cos, sin)` pair — i.e. the real and imaginary
   parts of each band's mean resultant vector.  Averaging the unit vector rather
   than the angle is what makes the step immune to phase wrapping, and the
   magnitude weight is a *reliability* weight only: with a constant phase the
   output is that phase whatever the weights, so no magnitude information can
   enter the feature value.
4. **Post-processing.** First-order regression deltas on contiguous frames, then
   an energy VAD (frames within 30 dB of the loudest frame), then
   standardisation by a global mean/variance fitted on background frames only.
   **Final feature dimension: 128** (32 bands × {cos, sin} × {static, delta}).
5. **UBM.** 128-component diagonal-covariance GMM, k-means++ initialisation,
   EM in `float64` with a variance floor of `1e-3`, trained on all 1287
   background UBM-split recordings.
6. **Enrolment.** Mean-only MAP adaptation, `α_k = n_k/(n_k + 16)`; components
   with no enrolment occupancy keep their UBM mean exactly.
7. **Scoring.** Average per-frame log-likelihood ratio
   `(1/T) Σ_t [log p(x_t|S) − log p(x_t|UBM)]`, computed with `logsumexp`.
   Optional T-norm against the 150-speaker background cohort.

Magnitude enters the pipeline in exactly three documented places: the energy
VAD, the band-aggregation reliability weight, and numerical floors.  It is never
a model input feature.  `scripts/05_ablations.py` section B runs the controls
that verify this empirically — including the audit that caused the
modified-group-delay feature to be **rejected** from the final system (see
`report.md` §4).

## 5b. Version 4 (current submission)

Earlier versions were selected on the official dev list (196 genuine trials)
and lost 1–3 pp on the leaderboard.  v4 is selected on an **eval-mirroring
bench** (`svcore/bench.py`): 10 dev enrolment rotations (1,960 genuine trials)
plus 4-fold background cross-validation whose UBMs never see the fold's
speakers (368 genuine trials), with trial lists built by the organisers' recipe.

| system | bench EER |
|---|---|
| v3 recipe (submitted earlier, public LB 2.36%) | 1.05% |
| best single system + v4 scoring | 0.99% |
| **v4**: 3 IFD + 3 FDPD members, adaptive s-norm with enrolled cohort, competitor penalty | **0.69%** |

```bash
python scripts/08_bench_sweep.py members      # bench one sweep (results/bench_sweeps.jsonl)
python scripts/10_bench_fusion.py <a> <b> ...  # fusions of saved bench systems
python scripts/11_bench_greedy.py              # greedy member selection
python scripts/09_make_submission_v4.py --tag v4   # -> submissions/submission_v4.csv
```

**Final notebook:** `akspai418_BT24056_a4.ipynb` contains the whole final
pipeline in one self-contained file (torch/torchaudio only, no `svcore` import).
Run it from the repository root, or on Kaggle with the competition data attached.
It writes `submissions/final_submission.csv`, which is identical to the
script output `submission_v4.csv`.  A cold run takes about 10 min on 8 CPU
cores; features and UBMs are cached in `nb_cache/`.

Configuration: `svcore/system4.py`.  Scoring for a trial uses only background
data, the claimed speaker's designated enrolment recordings, the test recording
itself, and the other *enrolled* models of the same split (designated enrolment
data) as extra utterance-side cohort models.  No other evaluation test recording
influences a trial's score.

## 6. Deliverables

| deliverable | path |
|---|---|
| **final notebook** (self-contained, reproduces the final submission) | `akspai418_BT24056_a4.ipynb` |
| **final Kaggle submission** (written by the notebook) | `submissions/final_submission.csv` (`trial_id, score`) |
| final development scores | `results/dev_results_final.csv` (`claimed_id, test_file, label, score`) |
| v1 development scores | `results/dev_results.csv` |
| v1 submissions (superseded) | `submissions/submission_final.csv`, `submissions/submission_rawllr.csv` |
| module verification | `tests/test_modules.py` → `results/log_tests.txt` |
| report | `report.md` |
| figures | `figures/*.png` |
| raw numbers | `results/*.json`, `results/log_*.txt` |

## 7. Attribution

* D. A. Reynolds, T. F. Quatieri, R. B. Dunn, "Speaker Verification Using
  Adapted Gaussian Mixture Models", *Digital Signal Processing* 10(1–3), 2000 —
  the GMM-UBM/MAP framework, the relevance-factor form of the mean update, and
  top-C fast scoring.
* R. M. Hegde, H. A. Murthy, V. R. R. Gadde, "Significance of the Modified Group
  Delay Feature in Speech Recognition", *IEEE TASLP* 15(1), 2007 — the modified
  group delay function (implemented, audited, and then rejected; see §4 of the
  report).
* R. Auckenthaler, M. Carey, H. Lloyd-Thomas, "Score Normalization for
  Text-Independent Speaker Verification Systems", *Digital Signal Processing*
  10(1–3), 2000 — Z-norm / T-norm.
* P. Warden, "Speech Commands: A Dataset for Limited-Vocabulary Speech
  Recognition", 2018 (CC BY 4.0) — the source corpus.
* k-means++ seeding follows Arthur & Vassilvitskii (2007).

All code in `svcore/` and `scripts/` was written for this project; no code was
copied from an external implementation.
