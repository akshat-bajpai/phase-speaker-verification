"""The v4 system: members and scoring, selected on the eval-mirroring bench.

Selection signal: ``svcore/bench.py`` -- 10 dev enrolment rotations (1,960
genuine trials) + 4-fold background cross-validation with fold-held-out UBMs
(368 genuine trials), trial lists built with the organisers' recipe.  Earlier
versions were tuned on the official dev list alone (196 genuine trials) and
lost 1-3 pp on the leaderboard; see report.md.

Bench (mean of pooled dev-rotation EER and background-CV EER):
    v3 recipe (4 members, background as-norm, z-score average)   1.05 %
    best single system (IFD A) + asnormE + penalty                0.99 %
    this system                                                    0.69 %
      (dev rotations 0.56 %, background CV 0.82 %, official dev 0.51 %)

Members -- two phase representations, three resolutions each:
  * IFD  (temporal phase difference, svcore/features._stream_ifd)
  * FDPD (frequency-direction phase difference, _stream_fdpd)
Every member: 128-component diagonal UBM on the background UBM split,
mean-only MAP with r = 16, average per-frame LLR.

Rejected on the bench (see results/bench_sweeps.jsonl): K = 64/256,
LDA (utterance- or frame-level), temporal smoothing, |X|^p weighting with
p != 1, mean+variance MAP, larger cohorts, RPS (21.8 % alone), exact
group-delay cepstra (no fusion gain, and its floored |X|^2 denominator leaks
magnitude in low-energy bins).
Round 5 (scripts/12_bench_new.py, results/exp_new.log), all within bench noise
of 0.69 % or worse: temporal-segment (text-dependent) UBMs with 2/3 segments
(0.66-0.74 % when fused), mixed time-frequency phase derivative TFD (13.8 %
alone, hurts fusion), 2 ms hop, delta-deltas, FDPD 12 ms window, 20 dB VAD.
Best cherry-picked 4-way addition reached 0.63 % -- not adopted (selection
noise on ~30 combinations, four extra members).

Scoring
-------
* ``asnormE`` -- adaptive s-norm (top-N cohort statistics).  Model side:
  the claimed speaker's model scored on background cohort utterances.
  Utterance side: the test file scored by the background cohort models *and*
  by the other enrolled models of the same split.
* members are averaged with block weights 0.7 (IFD) / 0.3 (FDPD);
* competitor penalty (``lam``, ``delta``) -- see bench.competitor_penalty.
"""
from __future__ import annotations

from typing import List

from .bench import SystemSpec
from .system2 import GMM as _G, MEMBERS as _M2

A = _M2[0]          # IFD, n_fft 1024, 16 ms Hann, 3 ms hop, VAD 25 dB, 32 mel bands

MEMBERS: List[SystemSpec] = [
    SystemSpec(_M2[0], _G, 16.0, "IFD  n1024 w16"),
    SystemSpec(_M2[1], _G, 16.0, "IFD  n2048 w25"),
    SystemSpec(_M2[2], _G, 16.0, "IFD  n1024 w20"),
    SystemSpec(A.replace(stream="fdpd"), _G, 16.0, "FDPD n1024 w16"),
    SystemSpec(_M2[3], _G, 16.0, "FDPD n1024 w20"),
    SystemSpec(A.replace(stream="fdpd", n_fft=2048, win_ms=32.0), _G, 16.0,
               "FDPD n2048 w32"),
]

SCORING = {
    "scheme": "asnormE",
    "top": 100,
    "weights": [0.7 / 3] * 3 + [0.3 / 3] * 3,
    "lam": 0.5,
    "delta": -1.0,
    "cohort_min": 3,
    "cohort_size": 150,
}
