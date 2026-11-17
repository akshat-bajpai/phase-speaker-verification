"""Global configuration for the GMM-UBM speaker verification system.

Only Python's standard library is used here (dataclasses / os / json), as
permitted by the competition rules.  All numerical work lives in torch.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field, replace

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
AUDIO_DIR = os.path.join(DATA_DIR, "audio", "audio")
CACHE_DIR = os.path.join(ROOT, "cache")
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(ROOT, "figures")
SUBMISSIONS_DIR = os.path.join(ROOT, "submissions")

for _d in (CACHE_DIR, RESULTS_DIR, FIGURES_DIR, SUBMISSIONS_DIR):
    os.makedirs(_d, exist_ok=True)

# Global seed.  The organisers split the data with seed 2026; we reuse it for
# every split *we* create (background UBM / held-out / cohort) so the whole
# pipeline is reproducible.
SEED = 2026

SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# Feature configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FeatureConfig:
    """Everything that defines the feature pipeline.

    `stream` selects the phase representation:
      * ``"mgdcc"``   -- modified group-delay cepstral coefficients
      * ``"ifd"``     -- (weighted) instantaneous-frequency deviation, i.e. the
                         temporal phase difference / baseband phase difference
      * ``"fdpd"``    -- frequency-direction phase difference (finite-difference
                         group delay)
      * ``"cosphase"``-- cosine of the raw phase spectrum + DCT (negative
                         control: not shift invariant)
      * ``"rps"``     -- relative phase shift w.r.t. the lowest voiced bin
    """

    stream: str = "ifd"

    # --- framing -----------------------------------------------------------
    sample_rate: int = SAMPLE_RATE
    win_ms: float = 25.0
    hop_ms: float = 10.0
    n_fft: int = 512
    window: str = "hann"
    center: bool = True
    preemph: float = 0.0           # phase features: no pre-emphasis by default

    # --- voice activity detection (magnitude is used ONLY here + as a
    #     reliability weight; never as a model input feature) --------------
    vad: bool = True
    vad_rel_db: float = 30.0       # keep frames within X dB of the loudest frame
    vad_min_frames: int = 10       # fall back to the top-N loudest frames

    # --- phase -> cepstrum compression ------------------------------------
    n_bands: int = 24              # mel/bark bands for band aggregation
    band_type: str = "mel"         # "mel" | "linear" | "none"
    mag_weight: bool = True        # magnitude-weighted band aggregation
    mag_power: float = 1.0         # reliability weight = |X| ** mag_power
    n_ceps: int = 20               # DCT coefficients retained per sub-stream
    keep_c0: bool = False          # c0 is a shift/offset term -> usually dropped
    f_min: float = 60.0
    f_max: float = 7800.0

    # --- modified group delay shaping parameters ---------------------------
    mgd_alpha: float = 0.4         # |tau|^alpha compression
    mgd_gamma: float = 0.9         # smoothed-magnitude denominator exponent
    mgd_lifter: int = 24           # cepstral-smoothing cutoff for |X|
    mgd_denom: str = "exact"       # "exact" -> |X|^2 (true group delay, phase-only)
                                   # "smoothed" -> |S|^{2 gamma} (Hegde et al. MGD)
    mgd_floor_db: float = 50.0     # per-frame magnitude floor (dB below frame peak)
    mgd_magnitude_only: bool = False  # DIAGNOSTIC: drop phase, keep |X|^2/|S|^2g

    # --- ifd sub-streams ---------------------------------------------------
    ifd_cossin: bool = True        # use cos/sin of the wrapped difference
    ifd_lag: int = 1               # difference against the frame `lag` steps back;
                                   # decouples the phase-difference baseline
                                   # (lag * hop) from the frame rate

    # --- relative phase shift (rps) -----------------------------------------
    rps_harmonics: int = 16        # harmonics 1..H; features use h = 2..H
    rps_f0_min: float = 70.0
    rps_f0_max: float = 400.0
    rps_voicing: float = 0.5       # min harmonic-sieve coherence to keep a frame
    rps_mode: str = "rps"          # "rps": phi_h - h phi_1
                                   # "d2":  phi_{h+1} - 2 phi_h + phi_{h-1}

    # --- optional voiced-frame selection (any stream) ----------------------
    voicing_mask: float = 0.0      # >0: also drop frames whose phase-domain
                                   # harmonic-sieve coherence is below this

    # --- post processing ---------------------------------------------------
    smooth: int = 0                # moving-average half-width (frames) applied
                                   # to the static phase features before deltas
    deltas: int = 1                # 0 = static, 1 = +delta, 2 = +delta-delta
    delta_width: int = 2           # regression window half-width
    cmvn: str = "global"           # "none" | "m" | "mv" (per-utterance)
                                   # | "global" (background-fitted standardisation,
                                   #   applied outside the per-file cache)

    def frame_lengths(self) -> tuple[int, int]:
        win = int(round(self.win_ms * self.sample_rate / 1000.0))
        hop = int(round(self.hop_ms * self.sample_rate / 1000.0))
        return win, hop

    @property
    def tag(self) -> str:
        """Short, filesystem-safe identity used for the feature cache."""
        win, hop = self.frame_lengths()
        parts = [
            self.stream,
            f"w{win}h{hop}n{self.n_fft}",
            f"{self.band_type}{self.n_bands}" if self.band_type != "none" else "nob",
            f"c{self.n_ceps}{'+c0' if self.keep_c0 else ''}",
            f"d{self.deltas}",
            f"cmvn-{self.cmvn}",
            f"vad{int(self.vad)}-{self.vad_rel_db:g}",
            f"mw{int(self.mag_weight)}",
        ]
        if self.stream == "mgdcc":
            parts.append(f"a{self.mgd_alpha:g}g{self.mgd_gamma:g}l{self.mgd_lifter}"
                         f"_{self.mgd_denom}{self.mgd_floor_db:g}"
                         + ("_MAGONLY" if self.mgd_magnitude_only else ""))
        if self.stream == "ifd":
            parts.append(f"cs{int(self.ifd_cossin)}lag{self.ifd_lag}")
        if self.stream == "rps":
            parts.append(f"H{self.rps_harmonics}f{self.rps_f0_min:g}-{self.rps_f0_max:g}"
                         f"v{self.rps_voicing:g}{self.rps_mode}")
        if self.smooth:
            parts.append(f"sm{self.smooth}")
        if self.mag_power != 1.0:
            parts.append(f"mp{self.mag_power:g}")
        if self.window != "hann":
            parts.append(f"win-{self.window}")
        if self.voicing_mask and self.stream != "rps":
            parts.append(f"vm{self.voicing_mask:g}")
        return "_".join(parts)

    def to_dict(self):
        return asdict(self)

    def replace(self, **kw) -> "FeatureConfig":
        return replace(self, **kw)


# ---------------------------------------------------------------------------
# GMM / UBM configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GMMConfig:
    n_components: int = 128
    cov_type: str = "diag"          # "full" | "diag" | "tied" | "spherical"
    init: str = "kmeans"            # "random" | "kmeans"
    max_iter: int = 100
    tol: float = 1e-4               # rel. change of mean log-likelihood
    var_floor: float = 1e-3         # additive floor on variances (post-CMVN scale)
    reg_covar: float = 1e-4         # ridge added to full/tied covariances
    kmeans_iter: int = 25
    min_weight: float = 1e-6        # occupancy floor before re-initialisation
    seed: int = SEED
    dtype: str = "float64"
    device: str = "cpu"

    def to_dict(self):
        return asdict(self)

    def replace(self, **kw) -> "GMMConfig":
        return replace(self, **kw)


@dataclass(frozen=True)
class EnrolConfig:
    relevance_factor: float = 16.0
    n_enrol: int = 3                # use enrol_index 0..n_enrol-1
    adapt_means: bool = True
    adapt_weights: bool = False
    adapt_covars: bool = False


@dataclass(frozen=True)
class ScoreConfig:
    top_c: int = 0                  # 0 = use all components
    norm: str = "none"              # "none" | "tnorm" | "znorm" | "ztnorm"
    cohort_size: int = 150


def save_json(obj, path):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)
    return path
