"""The revised system (v2), selected on the multi-protocol signal.

Why there is a v2
-----------------
Version 1 was tuned against the 196 genuine trials of the official development
list and reached 0.61 % dev EER -- but 3.70 % on the public leaderboard.  The
gap was selection bias, not a bug: ~10 hyper-parameters chosen against 196
trials.

``svcore/protocols.py`` adds two further speaker-disjoint protocols (dev-LOO,
291 genuine; background, 92 genuine with a UBM retrained to exclude them).  On
those, the v1 system scores 3.67 % and 4.11 % -- i.e. they predict the
leaderboard, and the dev number was the outlier.  Every choice below was made on
the mean of the three protocols, and only changes that help on all of them were
kept.

What changed
------------
1. **Finer frequency sampling.**  ``n_fft`` 512 -> 1024/2048 with a *shorter*
   analysis window.  Zero-padding oversamples the spectrum, so each mel band
   averages the phase deviation over more (and better-conditioned) bins.
   Single largest gain: 2.93 % -> 2.10 %.
2. **Tighter VAD** (30 dB -> 25 dB below peak).
3. **Four-way fusion** of complementary phase systems -- three IFD resolutions
   plus the frequency-direction phase difference, which is weak alone (7.8 %)
   but decorrelated enough to help.
4. **T-norm kept**, now verified on the honest protocols (2.10 % -> 1.50 %),
   where Z-norm is confirmed harmful.
"""
from __future__ import annotations

from typing import List

from .config import EnrolConfig, FeatureConfig, GMMConfig, ScoreConfig
from .system import FINAL_FEATURE as _BASE

#: Fusion members.  Equal weights, so there is nothing fitted at the fusion
#: stage and no extra opportunity to over-fit the selection protocols.
MEMBERS: List[FeatureConfig] = [
    _BASE.replace(n_fft=1024, win_ms=16.0, vad_rel_db=25.0),   # A
    _BASE.replace(n_fft=2048, win_ms=25.0, vad_rel_db=25.0),   # B
    _BASE.replace(n_fft=1024, win_ms=20.0, vad_rel_db=25.0),   # C
    _BASE.replace(stream="fdpd", n_fft=1024, win_ms=20.0,
                  vad_rel_db=25.0),                            # FDPD (diverse)
]

MEMBER_NAMES = ["IFD n_fft=1024 win=16ms",
                "IFD n_fft=2048 win=25ms",
                "IFD n_fft=1024 win=20ms",
                "FDPD n_fft=1024 win=20ms"]

GMM = GMMConfig(n_components=128, cov_type="diag", init="kmeans", max_iter=100,
                tol=1e-4, var_floor=1e-3, reg_covar=1e-4, dtype="float64")
ENROL = EnrolConfig(relevance_factor=16.0, n_enrol=3, adapt_means=True)
SCORE = ScoreConfig(top_c=0, norm="tnorm", cohort_size=150)
