# Who Said "Stop"? — Speaker Verification with Phase Features and a GMM-UBM

**Deliverable report.** All code in `svcore/` and `scripts/`; `README.md` covers
dependencies, commands, seeds and reproduction steps.

---

## 1. Summary

| | |
|---|---|
| **Feature** | instantaneous-frequency deviation (temporal/baseband phase difference), 25 ms window, **3 ms hop**, `n_fft=512`, magnitude-weighted circular mean over 32 mel bands, `(cos, sin)` band pair + first-order deltas, background-fitted global standardisation. **Dimension 128.** |
| **UBM** | 128-component diagonal-covariance GMM, k-means++ init, EM in `float64`, variance floor `1e-3`, trained on 222 389 frames from 1287 background recordings (707 speakers) |
| **Enrolment** | mean-only MAP adaptation, `r = 16`, 3 × 1 s utterances |
| **Scoring** | average per-frame log-likelihood ratio + T-norm over a 150-speaker background cohort |
| **Development EER** | **0.61 %** (95 % bootstrap CI 0.31–1.05 %) with T-norm; **1.02 %** (0.49–1.68 %) raw LLR |
| **Compliance** | `torch` + `torchaudio` only; features are phase-derived; magnitude appears only in the VAD, as a band-aggregation reliability weight, and in numerical floors — each audited in §4 |

Everything numerical — STFT framing, the phase transforms, k-means, EM for four
covariance structures, MAP adaptation, LLR scoring, T/Z-norm, and all metrics
(EER, FAR/FRR, DET/ROC, min-DCF, bootstrap CIs) — is implemented in PyTorch.
`matplotlib` only draws values torch has already computed. 49 module-level
checks against analytic ground truth are in `tests/test_modules.py`
(`results/log_tests.txt`).

**The two decisions that mattered most** were not the ones the brief
foregrounds. They were (i) recognising that the organisers' random leading pad
destroys *absolute* phase, which rules out an entire family of phase features
(§3.1), and (ii) auditing the modified group delay feature with a phase-free
control, which revealed that its apparent strength was magnitude leakage and got
it **removed** from the submitted system (§4).

---

## 2. Data, splits and protocol

3601 recordings of the word "stop" (16 kHz, mono, ~1 s) from Speech Commands
v0.02, split by speaker with seed 2026. We never mix the groups:

| group | speakers | recordings | our use |
|---|---|---|---|
| background | 786 | 1436 | UBM training, held-out likelihood, T-norm cohort, Z-norm impostors |
| dev targets | 97 | 291 enrol + 196 genuine tests | all model selection and every number in this report |
| dev impostors | 170 | 260 | dev impostor tests |
| eval targets | 179 | 537 enrol + 358 genuine tests | leaderboard only |
| eval impostors | 354 | 523 | leaderboard only |

**Our own splits of the background set are by speaker, with seed 2026**
(`svcore/data.py::background_splits`):

* **UBM training** — 90 % of the background speakers (**707 speakers**) →
  1287 recordings, 222 389 retained frames.
* **Held-out** — the remaining 10 % (**79 speakers**) → 149 recordings,
  26 509 frames. Every
  held-out-likelihood figure in §5–§6 is computed here, on speakers the UBM has
  never seen.
* **T-norm cohort** — 150 UBM-split speakers with ≥ 3 recordings, MAP-adapted
  with the same `r` as a target speaker, so a cohort model is statistically a
  pseudo-impostor of the same shape as a real one. Cohort speakers are drawn
  from the UBM-training portion; this is standard practice (cohort models are
  adapted, not trained) and costs nothing in validity because *no background
  speaker appears in any trial*. The cohort-size sweep is in §9.7.
* **Z-norm impostor utterances** — the 149 held-out recordings.

Development trials: **1824 total — 196 genuine, 1628 impostor** over 97 claimed
speakers and 456 distinct test recordings. Evaluation trials: 3524 over 179
claimed speakers and 881 test recordings.

Evaluation recordings are used **only** as the designated evaluation enrolment
utterances and as evaluation test recordings. The global normalisation
statistics, the UBM, the cohort and every hyper-parameter were fitted or chosen
without touching them.

### 2.1 How large is a 1 % difference?

The dev set has 196 genuine trials, so a *single* misordered genuine trial moves
the EER by about 0.51 % absolute. We therefore report a **percentile bootstrap
95 % CI over trials** (1000–2000 resamples) beside every EER, and we do not
claim a difference that the intervals do not support. This is also why the
headline system is chosen for being in a broad, flat optimum rather than for
owning the single lowest dev number.

---

## 3. Task 1 — Phase-based feature extraction

### 3.1 The constraint that shapes everything: the random leading pad

The organisers "gave each clip 0–30 ms of extra leading audio". A time shift of
`d` samples multiplies DFT bin `k` by `exp(−j 2π k d / N)`. With `N = 512` and
`d` up to 480 samples, the phase of a mid-frequency bin is rotated by many full
turns: **the absolute phase spectrum is effectively randomised per utterance.**

Any feature that reads the absolute phase is therefore measuring a property of
the organisers' random number generator, not of the speaker. This is not a
subtle effect — it is the dominant one. We measured it directly
(`scripts/01_features.py`), as the cosine similarity between a clip's features
and those of a shifted copy:

| representation | hop-aligned (harness check) | **sub-hop** mean / median | arbitrary 0–30 ms |
|---|---|---|---|
| **IFD (temporal phase difference)** | 1.000 | **+0.995 / +0.996** | +0.995 |
| FDPD (frequency-direction difference) | 1.000 | +0.952 / +0.969 | +0.959 |
| group-delay cepstra (exact `\|X\|²`) | 1.000 | +0.853 / +0.896 | +0.876 |
| **CosPhase (absolute phase)** | 1.000 | **+0.169 / −0.007** | +0.131 |

The three regimes probe different things, and separating them was necessary to
avoid drawing the wrong conclusion:

* **hop-aligned** (a whole number of hops): each analysis frame then covers
  *exactly the same samples* as before, merely under a different frame index.
  Every representation must therefore score 1.000, and every one does — this
  column is a check on the comparison harness, not on the features. (An earlier
  version of this experiment used only hop-aligned and arbitrary shifts, and the
  trivially-perfect hop-aligned column could easily have been misread as
  evidence about the features.)
* **sub-hop** (1 … hop−1 samples): the frame grid now cuts the signal at
  different points, rotating bin `k` of every frame by about `−2πkd/N`. This is
  the regime that isolates the phase consequence of a time shift, and it
  separates the representations completely.
* **arbitrary** (0–30 ms, as the organisers applied it): a whole number of hops
  plus a sub-hop remainder, so it tracks the sub-hop column closely.

CosPhase has a **median similarity of −0.007** under a sub-hop shift: its
features are, on average, *orthogonal* to those of the same recording shifted by
a few samples. The EER follows exactly (§9.1): CosPhase reaches **43.4 %**
against a 50 % chance level, while the shift-invariant representations reach
1–7 %.

So the design requirement is: **the representation must not depend on the time
origin.**

### 3.2 The chosen representation: instantaneous-frequency deviation

For frame `t` and bin `k`, with `φ(t,k) = ∠X(t,k)`:

```
Δ(t,k) = princarg( φ(t,k) − φ(t−L,k) − 2π k L·hop / n_fft )
```

This is the **temporal (baseband) phase difference**, equivalently the
instantaneous-frequency deviation: the second term removes the phase advance a
partial sitting exactly at the bin centre would accumulate over `L` hops, so
`Δ` measures how far the local partial deviates from that centre frequency.
`tests/test_modules.py` §3 verifies this against analytic ground truth: a tone
offset by 10/25/60 Hz from bin 40 yields `Δ` of 0.1885/0.4712/1.1310 rad,
matching `2π·Δf·hop/fs` to 4 decimal places.

**Why it is shift invariant.** Write the shift as `d = q·hop + rem`.

*The `q·hop` part is exact.* Shifting by a whole number of hops leaves every
analysis frame covering precisely the same samples, merely re-indexed, so the
features are unchanged for **any** representation. The test suite confirms our
frame bookkeeping does this correctly — max difference `0.00e+00` away from the
centre-padding boundary (the first `⌈(win/2)/hop⌉ + L + 1 = 7` frames, which
the boundary legitimately affects). This is a check on the implementation, not
evidence about the feature, and §3.1 treats it as such.

*The `rem` part is where representations differ.* A sub-hop shift moves the
frame grid relative to the signal, so frame `t` of the shifted clip is the
segment of the original starting `rem` samples earlier. For a locally stationary
frame this is, to good approximation, a pure phase rotation
`X'(t,k) ≈ X(t,k)·e^{+j2πk·rem/N}` — **and crucially the rotation is the same
for every `t`**, because `rem` does not depend on the frame index. It therefore
cancels in `φ(t,k) − φ(t−L,k)`. The invariance is exact for the `q·hop` part and
accurate-to-the-stationarity-approximation for `rem`, which is exactly what the
measurement shows: sub-hop cosine similarity **0.995** for IFD against **0.169**
for absolute phase (§3.1).

**Why the hop must be short.** `Δ` is a wrapped quantity: the deviation is
unambiguous only while `|f − f_k| < fs/(2·L·hop)`. At `L·hop = 160` samples
(10 ms) that window is ±50 Hz; at 48 samples (3 ms) it is ±167 Hz. A short
baseline also decorrelates less. §9.8 separates this from the second thing a
short hop does — producing more frames — and both effects are real and
comparable in size.

### 3.3 Phase wrapping

Wrapping is handled in two places, and never by unwrapping (which is unstable
on 1 s of noisy speech):

1. `princarg(θ) = θ − 2π⌊θ/2π + ½⌋` maps the difference into `[−π, π)`.
2. **Band aggregation averages the unit vector, not the angle.** Inside mel
   band `b` we form the *mean resultant vector* of `e^{jΔ}` and keep its real
   and imaginary parts:

```
c(t,b) = Σ_k w(b,k)·|X(t,k)|·cos Δ(t,k)  /  Σ_k w(b,k)·|X(t,k)|
s(t,b) = Σ_k w(b,k)·|X(t,k)|·sin Δ(t,k)  /  Σ_k w(b,k)·|X(t,k)|
```

Averaging angles across a wrap boundary is meaningless (the mean of `+179°` and
`−179°` is not `0°`); averaging unit vectors is the correct circular statistic
and is what `(cos, sin)` encodes. Replacing the `(cos, sin)` pair with the raw
wrapped `Δ` values costs **1.02 % → 2.21 %** (band means) or **3.57 %** (DCT of
the band means) — §9.4.

The size of that penalty is itself informative. During exploration at a 10 ms
hop the same substitution was catastrophic (16.3 % → 41.3 %); at the final 3 ms
hop it merely doubles the EER. The reason is that the short baseline keeps most
deviations well inside `±π`, so there is simply less wrapping left to mishandle.
The two design choices — short baseline and circular aggregation — attack the
same problem, and the second matters less once the first is in place.

### 3.4 Low-energy bins and numerical stability

Phase in a bin with no energy is numerically meaningless — it is the phase of
rounding noise. Our remedy is *reliability weighting*: the magnitude `|X(t,k)|`
weights each bin's contribution to its band's circular mean. Removing the
weight costs **1.02 % → 3.32 %** EER (§9.4).

This is magnitude used as a weight, never as a value, and the distinction is
provable rather than rhetorical: a weighted mean of a *constant* phase returns
that constant whatever the weights. `tests/test_modules.py` §4 checks exactly
this — with `Δ ≡ 0.7 rad`, the band output under the true magnitudes and under
random weights differ by `4e-07`, and both equal `(cos 0.7, sin 0.7)`. **No
magnitude information can reach the feature value through this path.** §4
builds on that to audit the alternatives.

### 3.5 Silence removal

An energy VAD keeps frames whose log energy is within `vad_rel_db` of the
loudest frame of the same utterance. Relative-to-peak thresholding (rather than
absolute) is what makes it immune to the organisers' ±3 dB random gain. If
fewer than 10 frames survive, we fall back to the 10 loudest frames, so no
recording can ever yield an empty feature matrix.

On a 200-recording sample this keeps 152.1 of ~334 frames on average (45.5 %);
over the whole corpus it retains 155.7 frames per recording on average, with a
minimum of 35 and a maximum of 344. The sweep (§9.4) shows 25–30 dB is best; 20 dB discards useful voiced onsets, 40 dB
and "off" admit silence frames whose phase is noise. Silence removal is worth
**1.02 % vs 2.04 %** EER.

Magnitude is used here, which the rules explicitly permit for silence detection.

### 3.6 Normalisation — and why per-utterance CMVN *hurts*

This was the most counter-intuitive result of the project.

| normalisation | dev EER |
|---|---|
| global (background-fitted) | **1.02 %** |
| none | 1.66 % |
| per-utterance CMN | 2.55 % |
| per-utterance CMVN | 4.59 % |

Per-utterance CMVN is near-universal for magnitude features because it removes
channel and gain effects. Neither applies here: **phase is already exactly
gain-invariant**, so there is no gain to remove — and the per-utterance *mean*
of the band-wise circular means is itself speaker-discriminative, because it
encodes where each band's dominant partial sits relative to the bin grid
(harmonic placement, hence F0 and formant structure). Subtracting it throws the
signal away.

We therefore standardise with a **single global mean/variance estimated on the
background UBM-training frames only** (`svcore/normalise.py`), which improves
the conditioning of the GMM without touching within-utterance structure. This
is a dataset-level transform fitted on permitted data, as the rules require.

An honest caveat: part of CMVN's usual value is removing *session* effects, and
this corpus has none to remove — every recording of a speaker comes from one
session on one device. In a cross-session deployment per-utterance normalisation
would likely become necessary again, and our margin would shrink. See §10.

### 3.7 Final feature specification

| parameter | value |
|---|---|
| window / hop / `n_fft` | 25 ms (400 samples) / **3 ms (48 samples)** / 512 → 257 bins |
| window function | Hann (periodic=False) |
| phase quantity | `princarg(φ(t,k) − φ(t−1,k) − 2πk·hop/n_fft)` |
| band aggregation | magnitude-weighted circular mean, 32 mel bands, 60–7800 Hz |
| sub-streams | `cos` and `sin` of the band mean resultant vector (64 dims) |
| cepstral compression | none (raw band values outperformed a DCT, §9.4) |
| deltas | first order, regression half-width 2 → ×2 |
| frame selection | energy VAD, 30 dB below peak, ≥ 10 frames |
| normalisation | global standardisation, fitted on background UBM frames |
| **feature dimension** | **128** |
| frames per utterance | 155.7 mean, min 35, max 344 (560 647 frames over 3601 recordings) |

Order of operations: framing → phase stream → band aggregation → deltas *on
contiguous frames* (so the VAD does not break temporal context) → VAD selection
→ normalisation. The identical pipeline is applied to background, enrolment,
development and evaluation recordings.

---

## 4. Compliance audit: why the modified group delay was rejected

The brief warns that "naming a method is not sufficient to establish
compliance". We took that literally and built a control.

The **modified group delay function** (Hegde, Murthy & Gadde 2007) is an
explicitly permitted "group delay cepstra" feature. Group delay is the phase
derivative, `τ(ω) = −dφ/dω = (X_R Y_R + X_I Y_I)/|X(ω)|²` with
`Y = DFT{n·x[n]}`. The "modified" version replaces the denominator by a
cepstrally smoothed magnitude `|S(ω)|^{2γ}`, nominally for numerical stability.

We implemented it, and it looked excellent — tuning `γ` took the dev EER from
8.2 % at `γ=0.9` to **3.57 % at `γ=0.5`**, far better than anything else at the
time. Then we noticed what `γ < 1` actually does:

```
τ_γ = (X_R Y_R + X_I Y_I) / |S|^{2γ}  ≈  τ_true · |X|^{2−2γ}
```

For `γ < 1` the denominator no longer cancels the numerator's magnitude — it
**reintroduces the spectral envelope**, multiplicatively. So we ran the control:
the identical pipeline with the phase derivative deleted from the numerator and
`|X|²` put in its place (`mgd_magnitude_only=True`). Any performance this
reaches is attributable to magnitude alone.

Measured at the final framing (`scripts/05_ablations.py` section B, all values
dev EER with 95 % CIs in `results/log_05_ablations.txt`):

| candidate | feature EER | phase-free control | control − feature | verdict |
|---|---|---|---|---|
| modified GD, `γ = 0.5` | 4.08 % | **3.13 %** | **−0.95 pp** | **rejected** |
| modified GD, `γ = 0.9` | 6.63 % | **5.61 %** | **−1.02 pp** | **rejected** |
| group delay, exact `\|X\|²` | 6.63 % | 10.20 % | +3.57 pp | compliant; ablation only |
| **IFD (submitted)** | **1.02 %** | **constant by construction** | — | **accepted** |

For both modified-group-delay settings **the phase-free control is better than
the feature itself.** Deleting the phase derivative and replacing it with
`|X|²` *improves* the result. There is no reading of that under which the
modified group delay, as parameterised here, is a phase feature: the phase term
is not merely redundant, it is a mild nuisance on top of a magnitude feature.
(During exploration at a 10 ms hop the same pair read 3.57 % vs 4.08 % — the
control already within half a point — which is what prompted the audit.)

The exact-`|X|²` group delay *is* compliant: its control is 3.6 points worse,
so the phase derivative is doing real work. But it is also 6.5× worse than IFD,
because dividing by the true `|X|²` leaves the quantity ill-conditioned wherever
the spectrum is small, and the floor that tames it is itself a crude
spectral-support indicator (which is why its control still reaches 10.2 %
rather than chance). We report it as an ablation and do not submit it.

The IFD row is not an empirical claim but an analytic one, verified in the test
suite (§3.4) and re-verified inside the ablation script: deleting the phase from
a magnitude-weighted circular mean leaves a constant (`max diff 4.8e-07` between
true and random weights), carrying zero information. There is no leakage channel
to measure.

**The best-scoring feature we built was the one we had to throw away.** Keeping
it would have been easy and would have looked good; it would also have meant
submitting a magnitude system for a phase-feature competition. The full audit
runs as section B of `scripts/05_ablations.py`.

---

## 5. Task 2 — GMM behaviour on the phase features

Fitting set: 20 000 frames uniformly subsampled (seed 2026) from the background
UBM split. Held-out set: all 26 509 frames of the 10 % held-out background
speakers — **disjoint speakers**, which is what makes the held-out likelihood
mean anything.

### 5.1 Covariance structure and mixture count

| cov | K | params | train LL | held-out LL | BIC | time |
|---|---|---|---|---|---|---|
| spherical | 64 | 8 319 | −171.17 | −170.53 | 6.929e6 | 0.7 s |
| spherical | 128 | 16 639 | −169.95 | −170.07 | 6.963e6 | 1.6 s |
| diag | 32 | 8 223 | −167.73 | −167.24 | 6.791e6 | 0.6 s |
| diag | 64 | 16 447 | −165.65 | −166.19 | **6.789e6** | 1.1 s |
| diag | 128 | 32 895 | −163.78 | **−165.82** | 6.877e6 | 2.1 s |
| tied | 64 | 16 511 | −165.56 | −165.41 | 6.786e6 | 0.9 s |
| tied | 128 | 24 767 | −164.52 | −165.08 | 6.826e6 | 1.6 s |
| full | 8 | 67 079 | −160.18 | **−163.48** | 7.072e6 | 10.5 s |
| full | 16 | 134 159 | −156.91 | −164.29 | 7.605e6 | 17.2 s |
| full | 32 | 268 319 | −151.33 | −168.43 | 8.711e6 | 23.0 s |

Four things are visible, and they disagree in an instructive way.

1. **Full covariance over-fits exactly as the parameter count predicts.** From
   `K=8` to `K=32` the training likelihood climbs by 8.9 nats while the held-out
   likelihood *falls* by 4.9. At `K=32`, `D=128` a full-covariance model has
   268 319 parameters fitted from 20 000 × 128 numbers, i.e. one parameter per
   9.5 observations, with about 625 frames available to estimate each
   128 × 128 covariance (8256 free terms). The matrices cannot be estimated;
   only the ridge keeps them invertible.
2. **BIC and the held-out likelihood pick different winners.** BIC's minimum is
   `diag, K=64`; the best held-out likelihood is `full, K=8`. BIC penalises
   parameters with `p·log N` under an asymptotic argument that does not hold at
   one parameter per ten observations, and it is computed on the *fitting* set.
   The held-out number on disjoint speakers is the more trustworthy criterion
   here, and the EER (the actual objective) is the final arbiter — §9.3 shows
   `diag` wins on EER by a wide margin, because what matters for verification is
   not raw density fit but how well the component *means* localise speaker
   information for MAP to adapt.
3. **Tied covariance is a good compromise on likelihood** (−165.08 held-out at
   `K=128`, beating diag) **but not on EER**, for the same reason.
4. **Cost scales as the parameter count.** Full covariance is 10–20× slower per
   fit than diagonal and would make the `K=128` UBM, the 150-model cohort and
   the 3524-trial scoring pass infeasible.

**Decision: diagonal covariance.** With `K` large enough, a diagonal mixture
approximates correlated structure by using more components — the standard
GMM-UBM argument — and it is the only structure that is simultaneously
well-estimated, cheap, and best on the target metric.

### 5.2 Initialisation: 25 seeds, random vs k-means++

One fit per seed, `K = 64`, diagonal, `max_iter = 100`, `tol = 1e-5`:

| initialiser | final train LL (mean ± sd) | min | max | spread | held-out LL | iterations |
|---|---|---|---|---|---|---|
| random | −165.911 ± 0.102 | −166.097 | −165.742 | 0.355 | −166.330 | 78.8 |
| k-means++ | **−165.771 ± 0.084** | −165.929 | −165.613 | 0.316 | **−166.267** | **66.0** |

k-means++ wins on all four counts: a better optimum on average, a better
*best* optimum, lower variance across seeds, and 16 % fewer EM iterations to
get there. Every seed lands in a distinct local maximum — the spread of 0.355
nats is far larger than the convergence tolerance, so this is genuine
multi-modality of the likelihood surface, not incomplete convergence.

On a low-dimensional synthetic mixture the same comparison is far more
dramatic: `tests/test_modules.py` and the synthetic check in §8 of development
showed random init landing in a local maximum with a maximum mean error of 8.0
where k-means++ recovered the true means to 0.03. Real 128-dimensional speech
features have a smoother, better-connected likelihood surface, so the penalty
for a poor start is a fraction of a nat rather than a qualitatively wrong model
— but it is consistently a penalty.

### 5.3 The value of selecting the best of n initialisations

Expected log-likelihood of the model you keep, if you fit `n` times and keep the
best training likelihood (4000 Monte-Carlo draws from the 25 fits):

| initialiser | train, n=1 | n=5 | n=25 | held-out, n=1 | n=5 | n=25 |
|---|---|---|---|---|---|---|
| random | −165.911 | −165.797 | −165.749 | −166.330 | −166.276 | −166.247 |
| k-means++ | −165.769 | −165.671 | −165.623 | −166.266 | −166.206 | −166.188 |

Two observations. First, the returns diminish sharply — most of the available
gain arrives by `n = 5`, and `n = 25` buys little more, as expected when
sampling the maximum of a tight distribution. Second, and more important,
**selecting on the training likelihood does transfer to held-out data here**
(held-out improves by 0.08 nats alongside the 0.15 nat training gain). That is
not guaranteed — selection on a training criterion can chase over-fitting — and
it holds in this case only because the model is strongly under-parameterised
relative to 20 000 frames. One k-means++ fit is within 0.15 nats of best-of-25,
so the final system uses a single deterministic k-means++ fit at seed 2026 and
spends the compute budget elsewhere.

### 5.4 A deliberately degenerate component, and three remedies

Fitting `K = 64` diagonal components to only `N = 400` frames of `D = 128`
dimensions means estimating 16 447 parameters from 51 200 numbers — about **6.2
frames per component** for a 128-dimensional Gaussian.

**The symptom.** Sweeping the variance floor from `1e-2` down to `1e-14`:

| variance floor | min variance | cond. number | components at the floor | train LL | held-out LL | gap |
|---|---|---|---|---|---|---|
| 1e-2 | 1.0e-2 | 1.0e3 | 23 / 64 | −127.0 | −185.1 | +58.1 |
| 1e-3 | 1.0e-3 | 9.9e3 | 23 / 64 | −117.8 | −185.6 | +67.8 |
| 1e-6 | 1.0e-6 | 1.6e6 | 23 / 64 | −92.1 | −185.7 | +93.5 |
| 1e-10 | 1.0e-10 | 1.4e7 | 23 / 64 | −58.3 | −185.7 | +127.4 |
| **1e-14** | **1.0e-14** | **1.4e7** | **23 / 64** | **−24.3** | **−185.7** | **+161.4** |

Three symptoms appear together, and they are exactly the ones the brief asks
for:

1. **Near-singular covariances.** 23 of the 64 components are pinned *exactly*
   at whatever floor we set — they would go to zero without it — and the
   covariance condition number reaches `1.4e7`.
2. **Tiny effective occupancy.** The minimum component occupancy is **exactly
   1.00 frames**: a component has collapsed onto a single frame and claims it
   with responsibility 1. (Note that the natural-looking diagnostic "count of
   components with `n_k < 1`" returns zero here and is the *wrong* test — a
   collapsed component owns precisely one frame, not less than one.)
3. **Unstable, meaningless likelihood.** The training likelihood rises without
   bound as the floor is removed — `−127 → −24` per frame, monotone in
   `log(floor)` — while the held-out likelihood sits immovably at `−185.7`. The
   generalisation gap grows to **161 nats per frame**, against **0.4 nats** for
   the properly-sized UBM in §6. A density that assigns ever-greater likelihood
   to its own training points and nothing to anything else is the textbook
   degenerate mixture, and its training likelihood has stopped being a
   measurement of anything.

**The remedies**, all applied to the same degenerate setting:

| configuration | K | min n_k | min variance | cond. number | train LL | held-out LL |
|---|---|---|---|---|---|---|
| degenerate: no floor, no reinit | 64 | 1.00 | 1.0e-14 | 1.4e7 | −24.3 | −185.7 |
| remedy 1: variance floor 1e-3 | 64 | 1.00 | 1.0e-3 | 9.9e3 | −117.8 | −185.6 |
| remedy 2: reinit low-occupancy | 64 | 1.00 | 1.0e-14 | **3.0e2** | **+140.2** | −176.7 |
| remedy 3: fewer components (K=8) | 8 | 1.00 | 1.0e-14 | 1.2e5 | −164.2 | **−176.2** |
| **remedies 1+3 (K=8 + floor)** | **8** | **1.00** | **1.0e-3** | **3.4e3** | **−168.3** | **−176.2** |

Each remedy addresses a *different* symptom, and reading the table carefully is
more instructive than the headline:

1. **A variance floor** fixes the conditioning and the likelihood divergence
   (`1.4e7 → 9.9e3`, train LL `−24 → −118`) but moves the held-out likelihood by
   0.1 nats. It stops the model lying about its own fit; it does not make the
   model better. It treats the symptom.
2. **Component re-initialisation** (re-seed any component below 2 % occupancy at
   a random data point) gives the best conditioning of all (`3.0e2`) and a real
   held-out gain (`−185.7 → −176.7`) — but look at the training likelihood:
   **+140.2**, i.e. *worse* divergence than the unregularised fit. Keeping
   components alive and well-spread does nothing to stop each one collapsing
   onto its own neighbourhood when there is no floor. **Re-initialisation and
   flooring are not substitutes**; they fix orthogonal failures, which is why
   the real UBM uses both.
3. **Fewer components** is the only remedy that addresses the *cause*, and it
   produces the largest held-out improvement (`−176.2`). The model was simply
   too large for the data. Combining it with a floor (last row) keeps the
   held-out gain *and* restores a trustworthy training likelihood — the
   configuration one would actually ship.

In the real system all three are active in proportion: `K = 128` is chosen by
held-out likelihood and EER on 222 389 frames (**1737 frames per component, not
6.2**), the variance floor is `1e-3`, and low-occupancy re-initialisation is on
during UBM training. Even so, the shipped `K = 128` UBM contains a component
with occupancy 2.0 out of 222 389 frames (§6) — degeneracy pressure never
disappears, it is only kept bounded.

### 5.5 Two-dimensional projection (visualisation only)

`scripts/02_gmm_behaviour.py` section F computes a PCA projection **in torch**
(SVD of the centred fitting set) purely to draw `figures/fig02j_pca_components.png`.
PC1 and PC2 explain only 6.2 % and 4.3 % of the variance — the phase features
are genuinely high-dimensional, with no dominant 2-D structure, which is itself
a reason the mixture needs many components. **The GMMs are always fitted on all
128 dimensions**; the projection is never an input to any model.

---

## 6. Task 3 — UBM training

Trained on all 222 389 retained frames from the 1287 background UBM-split
recordings (no frame subsampling). EM in `float64`, k-means++ init, relative
tolerance `1e-4` on the mean log-likelihood, `max_iter = 100`, variance floor
`1e-3`, low-occupancy re-initialisation on, seed 2026, CPU.

| K | params | train LL | held-out LL | gap | iters | time | min n_k | dev EER (raw LLR) |
|---|---|---|---|---|---|---|---|---|
| 16 | 4 111 | −169.97 | −168.63 | −1.35 | 14 | 3.2 s | 2100.4 | 1.23 % [0.67–1.72] |
| 32 | 8 223 | −168.08 | −166.92 | −1.16 | 19 | 4.4 s | 1262.6 | 1.97 % [1.13–2.62] |
| 64 | 16 447 | −166.42 | −165.61 | −0.81 | 19 | 7.5 s | 48.0 | 1.53 % [0.98–2.67] |
| **128** | **32 895** | **−164.86** | **−164.43** | **−0.43** | **24** | **15.2 s** | **2.0** | **1.02 % [0.49–1.64]** |
| 256 | 65 791 | −163.50 | −163.51 | +0.01 | 23 | 27.5 s | 2.0 | 1.53 % [0.74–2.14] |

**Fit.** The held-out likelihood improves monotonically with `K` across the
whole range — at 222 389 frames even `K = 256` (one parameter per 433
observations) is not over-fitted in the density sense. Note the train/held-out
gap: it is *negative* for small `K` (held-out scores **better** than train),
which simply says the held-out speakers happen to be slightly easier to model,
and it closes steadily as `K` grows, crossing zero at `K = 256`. That crossing
is the first sign of the model beginning to specialise to the training
speakers.

**Verification performance does not follow the likelihood.** EER is best at
`K = 128` and worsens at `K = 256`, even though `K = 256` has the better
held-out density. The reason is MAP, not the UBM: with 3 × 1 s of enrolment
(390 frames) spread over 256 components, each component sees ~1.5 frames, so
most adapted means are dominated by the prior and the speaker model carries
little speaker-specific information. The occupancy table in §7.1 makes this
concrete. **Density fit and verification accuracy are different objectives, and
`K` trades them off.** All the differences between `K = 16` and `K = 256` are
within overlapping CIs, so we read this as a broad flat optimum around
64–128 and choose `K = 128` as its centre.

**Numerical stability and cost.** EM converged in 14–24 iterations at every `K`
(the monotone likelihood increase is verified in the test suite). Cost grows
linearly in `K` as expected — 3.2 s at `K = 16` to 27.5 s at `K = 256` on 8 CPU
threads. The minimum component occupancy falls from 2100 frames at `K = 16` to
**2.0 frames at `K = 128`**: even with 222 389 training frames the largest
mixtures contain a nearly-empty component, which is exactly the pressure §5.4
studied, held in check by the variance floor and re-initialisation.

Frame subsampling: none for the reported UBMs. The Task 2 study uses a fixed
20 000-frame subsample (seed 2026) consistently across every comparison in that
section, so its likelihood numbers are mutually comparable but are *not*
comparable with this table.

---

## 7. Task 4 — MAP adaptation and enrolment

### 7.1 Why MAP, and not a speaker GMM from scratch

Each target speaker provides three 1-second utterances. After VAD that is **390
frames on average** in 128 dimensions. A 128-component diagonal GMM needs 32 895
parameters. The arithmetic is hopeless, and the measurement agrees:

| speaker model | dev EER (raw LLR) | 95 % CI |
|---|---|---|
| **MAP, mean-only, r = 16** | **1.02 %** | [0.49–1.64] |
| EM from scratch, K = 2 | 13.78 % | [11.23–16.98] |
| EM from scratch, K = 4 | 8.16 % | [6.12–10.50] |
| EM from scratch, K = 8 | 7.13 % | [5.59–9.00] |
| EM from scratch, K = 16 | 9.64 % | [7.03–11.36] |

Training from scratch is **7× worse at its own best setting**, and its optimum is
`K = 8` — two orders of magnitude fewer components than the UBM, because that is
all 390 frames can support. Its U-shape is the bias/variance trade-off made
visible: `K = 2` cannot represent the speaker, `K = 16` cannot be estimated.

MAP escapes the trade-off by not estimating a model at all. It starts from a
128-component density already estimated from 222 389 background frames and moves
each mean toward the enrolment data by an amount proportional to the evidence
available *for that component*. The model has the resolution of a large GMM and
the variance of a small one.

### 7.2 Component occupancy — what three seconds actually buys

| enrolment utterances | frames pooled | components with `n_k > 0` | occupied | max `n_k` |
|---|---|---|---|---|
| 1 | 128.9 | 83.1 / 128 | 65.0 % | 16.7 |
| 2 | 259.3 | 93.0 / 128 | 72.7 % | 29.5 |
| 3 | 389.7 | 96.6 / 128 | 75.4 % | 41.0 |

Even with all three utterances, **a quarter of the mixture never sees a single
enrolment frame.** Those components retain their UBM means *exactly*
(`svcore/map_adapt.py`: `torch.where(n_k <= 0, ubm.means, adapted)`), which has
two consequences worth being explicit about:

* the speaker model stays a proper density over the whole feature space, so the
  LLR is always well defined — a test frame landing in an unoccupied region
  simply contributes ≈ 0 to the ratio rather than a spurious penalty;
* the adaptation is *sparse by construction*. Only the ~75 % of components the
  speaker actually visited carry identity, and the remainder cancel in the
  numerator and denominator of the LLR. This is the mechanism that lets a
  128-component model be "trained" on 390 frames.

### 7.3 The relevance factor

```
α_k = n_k / (n_k + r),    μ̂_k = α_k·E_k[x] + (1 − α_k)·μ_k
```

| r | dev EER | 95 % CI | mean ‖μ_s − μ_UBM‖ | AUC |
|---|---|---|---|---|
| 0.5 | 7.65 % | [5.19–9.34] | 50.91 | 0.9841 |
| 1 | 6.33 % | [4.02–7.37] | 42.40 | 0.9911 |
| 2 | 3.75 % | [2.21–4.81] | 33.04 | 0.9971 |
| 4 | 1.53 % | [0.96–2.48] | 24.03 | 0.9991 |
| 8 | 1.02 % | [0.51–1.89] | 16.34 | 0.9994 |
| **16** | **1.02 %** | **[0.49–1.64]** | **10.45** | **0.9995** |
| 32 | 1.53 % | [0.85–2.15] | 6.33 | 0.9992 |
| 64 | 2.04 % | [1.05–2.74] | 3.66 | 0.9987 |
| 128 | 2.33 % | [1.46–3.32] | 2.03 | 0.9980 |
| 256 | 2.58 % | [1.70–4.00] | 1.09 | 0.9971 |

The curve is strongly asymmetric: going from `r = 16` down to `r = 0.5`
costs 6.6 points of EER, while going up to `r = 256` costs only 1.6. **Trusting
390 frames too much is far more damaging than trusting them too little**, and the
reason is in the drift column — at `r = 0.5` the means move 50 units, five times
their distance at `r = 16`, fitting noise in components that saw a handful of
frames. The `r → ∞` direction degrades gracefully because the model merely
becomes conservative: it approaches the UBM, the LLR approaches 0 for every
trial, and discrimination fades rather than inverts. We choose `r = 16`, the
centre of the flat 8–16 optimum.

**Limiting behaviour.** Measured on `dev_spk_001` with the shipped `K = 128`
UBM. Of its 128 components, **61 receive at least one enrolment frame and 3
receive exactly zero**; the remainder see a fraction of a frame.

| r | `max\|μ̂ − μ_UBM\|` | `max\|μ̂ − μ_ML\|` over `n_k ≥ 1` |
|---|---|---|
| 1e−8 | 4.315 | **2.4e−08** (0.000 % of the ML scale) |
| 1 | 2.152 | 1.347 (33.4 %) |
| 16 | 1.310 | 2.896 (71.7 %) |
| 1e+8 | **1.2e−06** | 3.136 (77.7 %) |

* `r → 0` ⟹ `α_k → 1` for any component with `n_k > 0`: the adapted mean
  **becomes** the enrolment maximum-likelihood estimate (agreement to `2.4e-08`
  above). Maximally speaker-specific and maximally over-fitted — and undefined
  for the 3 unoccupied components, which is why `r = 0` is unusable even in
  principle.
* `r → ∞` ⟹ `α_k → 0`: every mean collapses onto the UBM (`1.2e-06` above),
  the LLR becomes identically 0 for every trial, and the EER must go to 50 %.
  The observed 2.58 % at `r = 256` shows how slowly that limit is approached:
  `n_k` for a well-occupied component is of order 40, so `r = 256` still leaves
  `α ≈ 0.14`.
* The comparison has to be restricted to components with `n_k ≥ 1`, and the
  reason is itself informative. A component with `n_k ≈ 1e-12` has a
  numerically meaningless ML mean *and* `α ≈ n_k/r ≈ 0`, so it sits at the UBM
  mean for every practical `r`. Averaging such components into the "distance to
  ML" statistic makes the `r → 0` limit look violated when it is not — an
  earlier version of this table did exactly that and reported a spurious
  constant 5.785.
* The deeper point: **`α_k` depends on `n_k`, so one global `r` imposes
  different amounts of smoothing on different components.** Well-occupied
  components (`n_k ≈ 41`) get `α ≈ 0.72` at `r = 16` and move substantially;
  components with `n_k ≈ 0.01` get `α ≈ 6e-4` and barely move. That
  per-component adaptivity is the entire point of the relevance-factor
  formulation, and it is why a single `r` transfers across speakers with
  different amounts of usable speech.

### 7.4 Adapting weights and covariances

| speaker model | dev EER | 95 % CI |
|---|---|---|
| **mean only (required baseline)** | **1.02 %** | [0.49–1.64] |
| mean + weights | 1.23 % | [0.73–1.77] |
| mean + covariances | 0.92 % | [0.43–1.24] |
| mean + weights + covariances | 0.98 % | [0.37–1.53] |

All four intervals overlap almost completely, so the honest reading is that
**on this corpus it makes no measurable difference what else you adapt.** That
is itself the expected result, and the counting argument from §7.1 says why:
with 390 frames spread over 128 components, the per-component evidence supports
shifting a location parameter and little else. Covariance adaptation replaces a
spread estimated from 222 389 frames with one estimated from ~40, which is why
we floor it at 1 % of the UBM variance even in the exploratory arm; weight
adaptation re-estimates a 128-way prior from 390 frames. Mean-only remains the
submitted configuration — it is the required baseline, it is the cheapest, and
nothing here justifies the extra variance.

**A process note.** An earlier version of this experiment returned four
*identical* EERs. The cause was not the models but the scorer: the batched
mean-bank scorer (§11.3) exploits the fact that mean-only MAP leaves weights and
covariances shared with the UBM, so it reads only the means — and therefore
silently ignored the adapted weights and covariances. Four identical numbers to
the last digit is not a plausible experimental outcome, and chasing it produced
the guard now in `MeanBankScorer.from_models`, which raises rather than
silently degrade, plus the fallback to the general per-trial scorer in
`scripts/04_enrol_score.py` and `scripts/05_ablations.py`. The lesson is general:
an optimisation that assumes a model property must *check* that property, or it
becomes a silent experiment-corrupting bug.

## 8. Task 5 — Verification, scoring and the operating point

### 8.1 The score

```
s(X, S) = (1/T) Σ_t [ log p(x_t | λ_S) − log p(x_t | λ_UBM) ]
```

computed entirely in the log domain: component log-densities are combined with
`torch.logsumexp`, and the Gaussian terms use cached Cholesky precision factors
(`torch.linalg.solve_triangular`) rather than explicit matrix inverses. Dividing
by `T` makes the score comparable across recordings of different lengths, which
matters here because the retained-frame count ranges from 52 to 344.

A correctness anchor: scoring the UBM *against itself* must give exactly 0, and
does (`tests/test_modules.py` §12, `|s| < 1e-9`). The batched mean-bank scorer
agrees with the straightforward per-trial loop to `1.8e-15`.

### 8.2 Results on the development trials

196 genuine and 1628 impostor trials.

| scoring | EER | 95 % CI | AUC | min-DCF (p=0.01) |
|---|---|---|---|---|
| raw LLR | 1.02 % | [0.49–1.61] | 0.9995 | 0.1276 |
| **+ T-norm** | **0.61 %** | **[0.31–1.05]** | **0.9998** | **0.0816** |
| + Z-norm | 1.17 % | [0.61–2.13] | 0.9993 | 0.1833 |
| + ZT-norm | 1.04 % | [0.52–1.86] | 0.9995 | 0.0714 |

T-norm is the only normalisation that helps, and it helps on every metric at
once. This is diagnostic of *what* it fixes: T-norm removes a per-test-utterance
score offset, estimated by scoring the same utterance against 150 background
cohort models. Such an offset arises when a test recording is intrinsically
better or worse matched to the UBM — a loud, clearly-voiced "stop" scores higher
against *every* model. Z-norm, which removes a per-speaker-model offset, does
not help and slightly hurts: because every target has exactly three enrolment
utterances processed identically, there is little model-dependent offset to
remove, and the estimate from 149 impostor utterances adds more noise than it
removes. ZT-norm inherits Z-norm's noise, though it does reach the best
min-DCF — i.e. it is better in the low-false-accept tail even while its EER is
worse, which is a reminder that a single operating point does not summarise a
detector.

DET and ROC curves for all four are in `figures/fig05a_det.png` and
`fig05b_roc.png`; score histograms in `fig05c`/`fig05d`.

### 8.3 Ranking score vs decision vs calibrated probability

These three are routinely conflated, and the competition's metric depends on the
distinction:

* A **ranking score** is a number whose *order* is meaningful. Our average LLR
  is one. EER, AUC and the whole DET curve depend only on the ordering, so any
  strictly increasing transform — adding a constant, scaling, exponentiating —
  leaves them unchanged. We verify this property in the test suite rather than
  assuming it (`tests/test_modules.py` §13: EER is identical under `3s + 7` and
  under `exp(s/5)`). This is also why **we submit continuous scores and never
  thresholded decisions**: a binary submission throws away all ordering
  information within each class and would score catastrophically.
* A **decision** needs a threshold, and the threshold encodes a cost
  assumption, not a property of the system. Our scores are *T-normalised*, so
  the threshold is in units of cohort standard deviations and is only
  comparable across systems that share the same cohort.
* A **calibrated probability** would require `P(genuine | s)` to be well
  specified — e.g. that `s = 2.46` means the same posterior odds for every
  speaker and every utterance. Our score is **not** calibrated, for a concrete
  reason beyond "it's a likelihood ratio of a wrong model": the per-frame
  average divides by `T`, which makes scores comparable across durations but
  destroys the evidence-accumulation property a true log-likelihood ratio has
  (a 3-second utterance should carry more evidence than a 0.5-second one, and
  our score deliberately does not reflect that). Mapping to probabilities would
  need a separate calibration stage — logistic regression on dev scores, or a
  PAV/isotonic fit — trained on held-out trials and ideally duration-dependent.
  We did not fit one because the leaderboard metric is threshold-free and a
  monotone map cannot change the EER.

### 8.4 Operating point

FAR and FRR against threshold are in `figures/fig05e_far_frr.png`.

| operating point | threshold | FAR | FRR |
|---|---|---|---|
| EER (balanced) | +2.4591 | 0.61 % | 0.51 % |
| **high security (FAR ≤ 1 %)** | **+2.0558** | **0.98 %** | **0.00 %** |
| balanced-security (FAR ≤ 5 %) | +1.3729 | 4.98 % | 0.00 % |
| convenience (FAR ≤ 10 %) | +1.0423 | 9.95 % | 0.00 % |

**Stated use case: unlocking a shared device by voice.** A false accept hands a
stranger someone else's device — a security breach. A false reject costs the
legitimate user one retry. The costs are asymmetric by orders of magnitude, so
we operate at **FAR ≤ 1 %: threshold +2.0558, giving FAR = 0.98 % and
FRR = 0.00 %** on dev.

That FRR of 0.00 % is not a result to be proud of; it is a warning. It means
*zero* of 196 genuine trials fall below the threshold, so the dev set cannot
resolve the false-reject rate at this operating point at all — the true value
could be anything up to roughly 1.5 % (the 95 % upper bound for 0/196). The
threshold itself is also fitted on these same 196 trials and will not transfer
exactly. A deployment would need a dedicated, larger calibration set, and
preferably a cross-session one (§10).

---

### 8.5 The submitted system

| | primary | backup |
|---|---|---|
| file | `submissions/submission_final.csv` | `submissions/submission_rawllr.csv` |
| scoring | average per-frame LLR + T-norm | average per-frame LLR |
| dev EER | **0.61 %** [0.31–1.05] | 1.02 % [0.49–1.61] |
| eval rows | 3524 | 3524 |
| eval score range | −7.94 … +21.48 (mean +0.76, sd 2.30) | −4.87 … +7.21 (mean −0.11, sd 0.95) |

Both files are re-read and validated after writing (`scripts/06_make_submission.py::validate`):
exactly one row per `eval_trials.csv` trial, no duplicate or missing `trial_id`,
all scores finite and numeric, header `trial_id,score`. Scores are continuous,
never thresholded decisions (§8.3).

**A distribution check on the unlabelled evaluation scores.** We cannot measure
the eval EER, but we can check the submission is not broken. The eval protocol
has 358 genuine among 3524 trials (10.2 %); dev has 196 among 1824 (10.7 %). If
the system behaves on eval as it does on dev, the top decile of eval scores
should separate from the rest by about as much as the top decile of dev scores
does:

| | top-decile mean | rest mean | separation | overall mean / sd |
|---|---|---|---|---|
| dev (labelled) | +6.578 | +0.074 | **2.76 sd** | +0.773 / 2.360 |
| eval (unlabelled) | +6.497 | +0.115 | **2.78 sd** | +0.763 / 2.298 |

The two distributions are nearly identical. On dev, where we can check, the top
196 scores contain 191 of the 196 true genuine trials (97.4 %). This does not
predict the eval EER — it only says the eval scores have the same shape as dev
scores, so nothing has gone structurally wrong in enrolment, scoring or
normalisation for the evaluation speakers.

We submit the T-normalised scores as primary because T-norm improves EER, AUC
*and* min-DCF simultaneously on dev (§8.2) and the gain grows monotonically with
cohort size (§9.7), which argues it is a real effect rather than a dev artefact.
The un-normalised scores are supplied as a backup because they depend on one
fewer modelling assumption, and because the public/private split is by claimed
speaker — if the cohort happens to match the public half better than the private
half, the raw system is the more conservative choice. We expect both to score
worse than dev for the reasons in §10.2.

---

### 8.6 Compliance summary

| rule | how this system satisfies it |
|---|---|
| features derived from FT phase, not magnitude | the feature is `princarg(φ(t,k) − φ(t−1,k) − 2πk·hop/N)`; magnitude appears only as a VAD criterion, a band-aggregation reliability weight (provably non-leaking, §3.4), and numerical floors. Audited with phase-free controls in §4 |
| no magnitude MFCC / log-mel / magnitude spectrogram as input | none used; the modified-group-delay candidate that *behaved* like one was detected and rejected (§4) |
| feature extraction with `torch` + `torchaudio` only | `svcore/features.py`; the only torchaudio helpers are `melscale_fbanks`, `create_dct`, `load` |
| GMM init, EM, MAP, likelihoods, scoring in PyTorch | `svcore/gmm.py`, `map_adapt.py`, `scoring.py` |
| no NumPy / SciPy / scikit-learn / librosa / TensorFlow | verified by grep over every pipeline file; the full third-party import set is `torch`, `torchaudio`, `matplotlib` |
| plotting libraries used only to visualise | `svcore/plots.py` converts torch tensors to Python lists before any call |
| same pipeline for background / enrolment / dev / eval | one `FeatureConfig` drives all four (`svcore/system.py::prepare_features`) |
| learned transforms fitted on permitted background data only | the single global mean/variance is fitted on the background UBM split (`svcore/normalise.py`) |
| speaker-disjoint splits, seed reported | `svcore/data.py::background_splits`, seed 2026, splits by speaker |
| no external data or pretrained encoders | none |
| continuous scores submitted | §8.5 |

---

## 9. Task 6 — Analysis and ablations

Every ablation changes **one** thing relative to the final configuration
(IFD features, `K = 128` diagonal UBM, `r = 16`, raw LLR unless stated).
Baseline: **1.02 % [0.49–1.64]**. Full logs: `results/log_05_ablations.txt`;
machine-readable: `results/task6_ablations.json`. Read the confidence intervals,
not the point estimates — see §10.2.

### 9.1 Phase representation

| representation | dim | dev EER | 95 % CI | AUC |
|---|---|---|---|---|
| **IFD — temporal phase difference** | 128 | **1.02 %** | [0.49–1.64] | 0.9995 |
| group-delay cepstra (exact `\|X\|²`) | 58 | 6.63 % | [5.34–8.14] | 0.9802 |
| FDPD — frequency-direction phase difference | 128 | 7.25 % | [5.25–9.25] | 0.9813 |
| CosPhase — absolute phase (control) | 128 | **43.37 %** | [39.27–48.04] | 0.6173 |

The ordering follows the shift-invariance measurements of §3.1 exactly, and the
spread is enormous — a factor of 42 in EER between the best and worst phase
representation, on identical framing, the same GMM and the same scoring. **The
choice of *which* phase quantity to use dominates every other decision in this
project.**

CosPhase is the informative row. It is a perfectly reasonable phase feature in
the abstract — bounded, wrap-free, directly derived from `∠X` — and it performs
at 43.4 % against a 50 % chance level. It fails not because absolute phase
carries no speaker information but because *this corpus* destroyed it, by
design, with a random leading pad. Including a deliberate negative control is
what makes that diagnosable rather than mysterious.

The two group-delay-family representations land in the middle. Both are
genuinely shift-invariant (0.85–0.95 sub-hop similarity) but both are built from
a *ratio* that is ill-conditioned where the spectrum is small, and neither is
bounded the way a circular mean is.

### 9.2 Magnitude-leakage controls

See §4 — the audit that removed the modified group delay from the submission.
This is an ablation we consider as important as any accuracy number.

### 9.3 Mixture count

| K | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|
| dev EER | 1.23 % | 1.97 % | 1.53 % | **1.02 %** | 1.53 % |
| 95 % CI | [0.62–1.75] | [1.15–2.62] | [0.98–2.67] | [0.49–1.64] | [0.74–2.14] |

Every interval overlaps every other. The honest statement is that **the dev set
cannot distinguish `K` over a 16× range**, and the non-monotone wiggle (32 worse
than both 16 and 64) is noise, not structure. We choose `K = 128` because it is
the point estimate minimum, it sits where the held-out likelihood is still
improving (§6), and it is the centre of the 64–128 region rather than an
endpoint. Contrast this with §6's likelihood column, which *is* monotone and
unambiguous — a reminder that density fit is measured with 26 509 held-out
frames while EER is measured with 196 genuine trials.

### 9.4 Front-end ablations

| variant | dim | dev EER | 95 % CI |
|---|---|---|---|
| static only (no deltas) | 64 | 1.60 % | [1.02–2.73] |
| **+ delta (baseline)** | 128 | **1.02 %** | [0.49–1.64] |
| + delta + delta-delta | 192 | 2.55 % | [1.35–3.54] |
| no silence removal | 128 | 2.04 % | [1.06–3.14] |
| VAD 20 dB | 128 | 1.53 % | [0.79–2.21] |
| **VAD 30 dB (baseline)** | 128 | **1.02 %** | [0.49–1.64] |
| VAD 40 dB | 128 | 2.55 % | [1.47–3.27] |
| no normalisation | 128 | 2.03 % | [1.23–2.85] |
| **global norm (baseline)** | 128 | **1.02 %** | [0.49–1.64] |
| per-utterance CMN | 128 | 3.57 % | [2.54–4.65] |
| per-utterance CMVN | 128 | 4.59 % | [2.55–6.35] |
| no magnitude reliability weight | 128 | 3.32 % | [1.87–4.95] |
| DCT compression (20 coefficients) | 76 | 2.04 % | [1.15–3.65] |
| raw wrapped `Δ` + band means (no cos/sin) | 64 | 2.21 % | [1.46–3.29] |
| raw wrapped `Δ` + DCT (no cos/sin) | 38 | 3.57 % | [2.01–4.98] |

* **Deltas** help (1.60 → 1.02 %), delta-deltas hurt (2.55 %). At a 3 ms hop the
  second difference of a quantity that is already a difference is dominated by
  frame-to-frame noise, and it adds 64 dimensions that MAP must adapt from the
  same 390 frames.
* **Silence removal** is worth 2.04 → 1.02 %, and the threshold matters in both
  directions: 20 dB discards useful low-energy voiced onsets, 40 dB and "off"
  admit frames whose phase is noise. The optimum at 30 dB (45 % of frames
  retained) is where the retained set is mostly voiced.
* **Normalisation** reproduces §3.6: global standardisation beats none (1.02 vs
  2.03 %), and both beat per-utterance CMN (3.57 %) and CMVN (4.59 %). This is
  the result most likely to invert on a cross-session corpus (§10.1).
* **The magnitude reliability weight is worth 3× in EER** (1.02 vs 3.32 %),
  which quantifies how much of the raw phase spectrum is unusable noise — and
  justifies the one place where magnitude touches the feature value, in the
  provably-non-leaking way established in §3.4.
* **Cepstral compression loses information here** (2.04 % with a 20-coefficient
  DCT vs 1.02 % with the 32 raw band values). A DCT is worthwhile when the
  band-energy sequence is smooth and correlated; the band-wise circular means of
  phase deviations are not — adjacent bands can genuinely disagree, and
  truncating the DCT discards exactly that detail.
* **The circular `(cos, sin)` aggregation is worth 2–3.5×**: replacing it with
  the raw wrapped `Δ` values gives 2.21 % (band means) or 3.57 % (plus a DCT),
  against 1.02 % for the baseline. See §3.3 for why this penalty is smaller at a
  3 ms baseline than at 10 ms.

### 9.5 Covariance structure

| structure | K | dev EER | 95 % CI |
|---|---|---|---|
| spherical | 128 | 2.04 % | [1.19–3.29] |
| **diagonal** | 128 | **1.02 %** | [0.49–1.64] |
| tied | 128 | 2.95 % | [1.98–4.48] |
| full | 16 | 0.98 % | [0.51–1.65] |
| full | 32 | 1.29 % | [0.51–1.75] |

Full covariance at `K = 16` matches diagonal at `K = 128` (0.98 vs 1.02 %, fully
overlapping intervals) — a nice confirmation that the two are alternative ways
to spend capacity on modelling correlation. We keep diagonal because it costs
`10–20×` less per EM iteration (§5.1), because scoring 150 cohort models + 3524
trials with full covariances would be prohibitive, and because it degrades
gracefully as `K` rises whereas full covariance is already over-fitting at
`K = 32` (§5.1).

Note the inversion against §5.1's likelihood ranking: **tied covariance has the
*best* held-out likelihood of the structures at `K = 128` (−165.08) but the
*worst* EER (2.95 %).** A single shared covariance models the pooled background
density well while making every component's discriminative geometry identical,
which is precisely what MAP needs to differ between speakers. Density fit and
verification accuracy are not the same objective, and this row is the cleanest
demonstration of it in the project.

### 9.6 Enrolment size (`enrol_index` 0…k−1), and its interaction with `r`

| utterances | r = 8 | r = 16 | r = 32 |
|---|---|---|---|
| 1 | 4.59 % | 4.08 % | **3.57 %** |
| 2 | 2.27 % | 2.21 % | **2.04 %** |
| 3 | **1.02 %** | **1.02 %** | 1.53 % |

Each additional utterance roughly halves the EER (4.08 → 2.21 → 1.02 % at
`r = 16`), with no sign of saturation — more enrolment data would still help,
which is consistent with the occupancy table in §7.2 where even three
utterances leave a quarter of the mixture untouched.

The more interesting result is the **interaction**: the best `r` *decreases* as
enrolment data increases (32 → 32 → 8–16). This is exactly what the theory
predicts and is worth stating explicitly, because it is easy to treat `r` as a
constant of the system rather than a function of the enrolment budget: `r` sets
how much evidence is needed before a component's mean moves, so with one
utterance (≈ 130 frames) the right response is to trust the prior more. A
deployment with variable enrolment length should scale `r` with it rather than
fix it.

### 9.7 Score normalisation and cohort size

| scoring | dev EER | 95 % CI | min-DCF (p=0.01) |
|---|---|---|---|
| none | 1.02 % | [0.49–1.64] | 0.1276 |
| **T-norm** | **0.61 %** | [0.31–1.06] | **0.0816** |
| Z-norm | 1.17 % | [0.61–2.16] | 0.1833 |
| ZT-norm | 1.04 % | [0.52–1.85] | 0.0714 |

| T-norm cohort | 30 | 60 | 100 | 150 |
|---|---|---|---|---|
| dev EER | 1.17 % | 1.04 % | 0.80 % | **0.61 %** |

T-norm improves monotonically with cohort size over the whole available range,
which says the gain is limited by the *variance* of the per-utterance mean/sd
estimates rather than by any mismatch between cohort and impostors. 150 is our
ceiling — only 186 background speakers have the three recordings needed to build
a cohort model the same way a target model is built — so a larger cohort would
plausibly help further. The analysis of *why* T-norm helps and Z-norm does not
is in §8.2.

### 9.8 Phase-difference baseline vs frame rate

A short hop changes two things simultaneously. `ifd_lag` separates them: the
difference is taken against the frame `L` steps back, so the **baseline**
(`L × hop`) and the **frame rate** (`1/hop`) move independently.

| hop | lag | baseline | frames/utt | dev EER | 95 % CI |
|---|---|---|---|---|---|
| 3 ms | 1 | 3 ms | 153 | **1.02 %** | [0.49–1.64] |
| 3 ms | 2 | 6 ms | 153 | 1.53 % | [0.91–2.58] |
| 3 ms | 3 | 9 ms | 153 | 2.04 % | [1.06–4.02] |
| 3 ms | 4 | 12 ms | 153 | 5.10 % | [2.87–6.49] |
| 10 ms | 1 | 10 ms | 46 | 5.10 % | [3.57–6.74] |
| 2 ms | 5 | 10 ms | 230 | 1.90 % | [1.08–3.05] |
| 5 ms | 1 | 5 ms | 92 | 1.53 % | [0.97–2.09] |
| 1.5 ms | 2 | 3 ms | 306 | 1.02 % | [0.49–1.48] |

Both effects are real and comparable in size:

* **At a fixed frame rate** (rows 1–4, all 153 frames/utt) stretching the
  baseline from 3 ms to 12 ms costs **1.02 → 5.10 %**. The phase difference
  decorrelates and, past `|f − f_k| > fs/(2·L·hop)`, wraps: ±167 Hz at a 3 ms
  baseline, ±42 Hz at 12 ms.
* **At a fixed 10 ms baseline** (rows 5–6) raising the frame rate from 46 to 230
  frames per utterance recovers **5.10 → 1.90 %**. More frames means lower
  variance in both the MAP statistics and the per-frame score average.
* The last row is the control that matters: `hop = 1.5 ms, lag = 2` has the same
  3 ms baseline as the chosen setting and **twice** the frames, and performs
  identically (1.02 %). The frame-rate benefit has saturated by 153 frames —
  additional frames at a 1.5 ms hop are almost perfectly correlated with their
  neighbours and carry no new information. We therefore use `hop = 3 ms,
  lag = 1`: the same accuracy at half the computational cost.

This ablation is the reason we can describe the 3 ms hop as a *motivated*
choice rather than a tuned one.

### 9.9 Top-C component scoring

| C | 1 | 2 | 5 | 10 | 20 | 50 | all (128) |
|---|---|---|---|---|---|---|---|
| dev EER | 1.53 % | 1.02 % | 1.02 % | 1.02 % | 1.02 % | 1.02 % | 1.02 % |
| scoring time (1824 trials) | 3.17 s | 3.01 s | 3.14 s | 3.19 s | 3.30 s | 3.61 s | 4.16 s |

Scoring only the C components the **UBM** ranks highest for each frame (the same
set used for numerator and denominator, so the ratio stays consistent) is
**lossless from C = 2 upward** — identical to the last digit from C = 10. Even
`C = 1` costs only half a point. The speed saving in our implementation is
modest (4.16 → 3.0 s) because the batched mean-bank scorer is dominated by a
single GEMM that top-C does not shrink; in a conventional per-component
implementation the saving would be the full `C/K = 10/128` factor. The
accuracy result is the transferable one: **a 128-component likelihood ratio is
carried entirely by its top handful of components**, which is the standard
GMM-UBM fast-scoring result (Reynolds et al. 2000) reproduced on phase features.

### 9.10 Additive-noise robustness (stretch goal)

White noise added to the **test** recordings only; enrolment, UBM and cohort
stay clean, as in a deployed system.

| test SNR | clean | 20 dB | 10 dB | 5 dB | 0 dB |
|---|---|---|---|---|---|
| dev EER | 1.02 % | 7.14 % | 18.06 % | 25.00 % | 32.65 % |

**This is the system's clearest weakness, and it is intrinsic to phase.** A 20 dB
SNR — barely audible noise — costs a factor of seven. The mechanism is direct:
adding a noise component of relative amplitude `ε` to a partial perturbs its
log-magnitude by about `ε` but its *phase* by up to `arcsin ε` radians, and the
instantaneous-frequency deviation is the difference of two such perturbed
phases, so the noise enters twice and without the logarithmic compression that
protects magnitude features. The magnitude reliability weighting helps (it
down-weights the bins noise dominates) but cannot recover phase that has been
randomised.

A magnitude-based MFCC system would degrade far more gracefully here. That
trade-off — excellent clean-condition accuracy, poor noise robustness — is the
honest characterisation of phase-only features, and it is the main reason
production systems fuse phase with magnitude rather than replacing it.

---

## 10. Limitations, failure cases and honest caveats

A 0.61 % EER on a speaker-verification task should invite suspicion rather than
satisfaction. Here is what we believe it does and does not mean.

### 10.1 The corpus makes genuine trials much easier than deployment would

The brief states it plainly: *"All enrolment and test recordings of a speaker
were made in the same session on the same device."* Every genuine trial
therefore matches enrolment in microphone, room, distance, gain and the
speaker's state within one recording session. A cross-session genuine trial has
none of that.

This matters more for our system than for a magnitude-based one, because
fine-resolution phase structure is exactly where channel and recording-geometry
information lives — group delay and instantaneous frequency respond directly to
microphone impulse response and room reflections. **We should assume a
meaningful fraction of our 0.61 % reflects session/device matching rather than
speaker identity, and that a cross-session EER would be several times higher.**
The §3.6 result is corroborating evidence: per-utterance CMVN — the standard
defence against channel mismatch — *hurts* by 4× here, which is only possible
because there is no channel mismatch to defend against. On a cross-session
corpus that finding would very likely reverse, and our normalisation choice with
it.

We cannot quantify this with the data provided, since no cross-session trials
exist. It is the single largest caveat on the headline number.

### 10.2 The development set cannot resolve the differences we are choosing between

196 genuine trials means one trial ≈ 0.51 % EER. At 0.61 % EER the system makes
roughly one genuine-side error. Consequently:

* Most of the ablation differences in §9 that are under ~1.5 points have
  overlapping confidence intervals. We selected hyper-parameters from the
  *shape* of each curve (flat optima, monotone trends) rather than from the
  single best cell, but the selection is still based on 196 trials and some of
  it is certainly noise-fitting.
* The reported CIs are bootstrap-over-trials and therefore **understate** the
  true uncertainty: they resample trials, not *speakers*, so they do not capture
  the variance from having only 97 dev target speakers. A speaker-level
  bootstrap would give wider intervals.
* The leaderboard's private half holds ~70 % of 179 eval speakers. We expect our
  eval EER to be higher than 0.61 % simply from regression to the mean, and we
  would not be surprised by a factor of two.

### 10.3 Known failure modes of the feature

* **Short or weakly-voiced recordings.** The IFD is only meaningful where a bin
  contains a resolvable partial. The minimum retained-frame count across the
  corpus is 35 frames — about 105 ms of speech at a 3 ms hop — against a mean of
  155.7; for such recordings both the MAP statistics and the per-frame score
  average have high variance. The VAD fallback (10 loudest frames)
  prevents a crash but cannot manufacture evidence.
* **Noise.** The noise ablation (§9.10) is the clearest weakness: phase is far
  more fragile than magnitude under additive noise, because a noise component
  comparable to a partial's amplitude rotates its phase arbitrarily while
  changing the log-magnitude only slightly. The magnitude reliability weighting
  helps (it down-weights bins the noise dominates) but does not fix it.
* **The wrapping ceiling.** The 3 ms baseline resolves `|f − f_k| < ±167 Hz`.
  Partials deviating more than that alias, and we have no unwrapping stage. This
  bounds how short the window can usefully get and is why `hop = 2 ms` is not
  better than 3 ms despite producing more frames (§9.8).
* **Resampling / codec sensitivity.** Nothing in the pipeline would survive a
  resample or a lossy codec as gracefully as a magnitude feature would; phase is
  the first thing such processing discards.

### 10.4 Methodological caveats we chose to accept

* **The T-norm cohort speakers are also UBM-training speakers.** Standard
  practice, and no cohort speaker appears in a trial, but it is not a fully
  disjoint design. §9.7 reports the cohort-size sweep, which shows the gain still rising at our
  ceiling of 150 speakers; a fully disjoint cohort would have cost ~12 % of the
  UBM training data and capped the cohort lower still.
* **The operating threshold is fitted on the same 196 genuine trials used to
  report FRR**, so the FRR = 0.00 % at FAR ≤ 1 % is optimistic and
  unresolvable (§8.4).
* **We did not calibrate the scores.** Justified for a threshold-free metric,
  but it means the submitted numbers are rankings only and must not be read as
  probabilities.
* **`r`, `K`, the hop and the band count were all chosen on dev.** With ~10
  hyper-parameters selected on 196 genuine trials, some over-fitting to dev is
  unavoidable; the broad-plateau selection policy is a mitigation, not a cure.

---

## 11. Stretch goals attempted

1. **Weight and diagonal-covariance adaptation** in addition to means
   (`svcore/map_adapt.py`, §7.4).
2. **Four phase representations implemented and compared** — temporal phase
   difference, frequency-direction phase difference, group-delay cepstra (both
   exact and modified denominators), and raw cosine-phase as a negative control
   (§9.1).
3. **A batched/efficient EM implementation.** The full-covariance M-step
   accumulates `Σ_n r_nk x_n x_nᵗ` as a single chunked GEMM per block instead of
   `K` separate rank updates, which is what made full covariance at `D = 128`
   feasible at all (0.5 s per EM iteration at `K = 16`, versus minutes for the
   naive loop). Scoring is likewise batched: because mean-only MAP leaves the
   weights and covariances shared, the Mahalanobis terms for *all* speaker models
   against one utterance reduce to one matrix product
   (`svcore/scoring.py::MeanBankScorer`), verified equal to the per-trial
   reference loop to `1.8e-15`. This is what makes the 150-model T-norm cohort
   affordable.
4. **Additive-noise robustness** at four SNRs, with noise applied to test
   recordings only (§9.10).
5. **Score normalisation**: T-norm, Z-norm and ZT-norm, with the composition
   order handled properly (T-norm statistics computed on Z-normalised cohort
   scores, not on raw ones) — §8.2.
6. **A compliance audit methodology** (§4) — the phase-free control. We offer
   this as the most transferable contribution of the project: for any feature
   claimed to be phase-derived, delete the phase and re-measure. If performance
   survives, the feature was not what it claimed to be.

---

## 12. Reproducibility

* Seed **2026** everywhere (`svcore/config.py::SEED`) — background speaker
  split, k-means++ and EM initialisation, frame subsampling, bootstrap
  resampling, and the noise ablation's generator.
* CPU only; `float32` features, `float64` for EM / MAP / scoring; 8 threads.
  No result depends on a GPU.
* EM stopping rule: relative change of the mean log-likelihood `< 1e-4`, hard
  cap 100 iterations. Variance floor `1e-3`, ridge `1e-4` for full/tied.
* `./run_all.sh` regenerates every table, figure and CSV in this report from
  the raw audio in roughly 25 minutes; `results/log_*.txt` are the verbatim
  logs. `python tests/test_modules.py` runs the 49 analytic checks.
* Feature caches are keyed by a tag that encodes every front-end parameter
  (`FeatureConfig.tag`), so a changed parameter can never silently reuse stale
  features.

## 13. References

1. D. A. Reynolds, T. F. Quatieri, R. B. Dunn. "Speaker Verification Using
   Adapted Gaussian Mixture Models." *Digital Signal Processing* 10(1–3),
   19–41, 2000. — GMM-UBM, the relevance-factor MAP mean update (our
   Appendix-A implementation), and top-C fast scoring.
2. R. M. Hegde, H. A. Murthy, V. R. R. Gadde. "Significance of the Modified
   Group Delay Feature in Speech Recognition." *IEEE TASLP* 15(1), 190–202,
   2007. — the modified group delay function, implemented in
   `svcore/features.py` and then rejected on the evidence of §4.
3. R. Auckenthaler, M. Carey, H. Lloyd-Thomas. "Score Normalization for
   Text-Independent Speaker Verification Systems." *Digital Signal Processing*
   10(1–3), 42–54, 2000. — Z-norm, T-norm and their composition.
4. D. Arthur, S. Vassilvitskii. "k-means++: The Advantages of Careful Seeding."
   *SODA*, 1027–1035, 2007. — the seeding in `svcore/gmm.py::kmeans_plusplus`.
5. A. P. Dempster, N. M. Laird, D. B. Rubin. "Maximum Likelihood from
   Incomplete Data via the EM Algorithm." *JRSS B* 39(1), 1–38, 1977.
6. A. Martin et al. "The DET Curve in Assessment of Detection Task
   Performance." *Eurospeech*, 1997. — the probit-warped DET axes in
   `svcore/plots.py`.
7. P. Warden. "Speech Commands: A Dataset for Limited-Vocabulary Speech
   Recognition." arXiv:1804.03209, 2018 (CC BY 4.0). — the source corpus.

No code was adapted from an external implementation; `svcore/` and `scripts/`
were written for this project. The `torchaudio` helpers used are
`melscale_fbanks` (mel filterbank weights), `create_dct` (orthonormal DCT
matrix) and `load` (WAV decoding).
