"""The configuration of the reported system, plus helpers shared by the scripts.

Every number here was selected on the **labelled development trials only**.
Evaluation recordings are used exclusively as (a) the designated evaluation
enrolment utterances and (b) the test recordings of the evaluation trials.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from . import cache, config as C, data, metrics as M, normalise as NZ, scoring as SC
from .config import EnrolConfig, FeatureConfig, GMMConfig, ScoreConfig
from .gmm import GMM

# ---------------------------------------------------------------------------
# selected configuration
# ---------------------------------------------------------------------------

#: Instantaneous-frequency-deviation features: the temporal (baseband) phase
#: difference, aggregated as a magnitude-weighted circular mean in 32 mel bands,
#: kept as the raw (cos, sin) band pair plus first-order deltas.
FINAL_FEATURE = FeatureConfig(
    stream="ifd",
    win_ms=25.0, hop_ms=3.0, n_fft=512, window="hann",
    n_bands=32, band_type="mel", mag_weight=True,
    n_ceps=0,                 # keep the raw band values (no DCT)
    ifd_cossin=True, ifd_lag=1,
    vad=True, vad_rel_db=30.0,
    deltas=1, delta_width=2,
    cmvn="global",            # background-fitted standardisation
)

FINAL_GMM = GMMConfig(n_components=128, cov_type="diag", init="kmeans",
                      max_iter=100, tol=1e-4, var_floor=1e-3, reg_covar=1e-4,
                      seed=C.SEED, dtype="float64", device="cpu")

FINAL_ENROL = EnrolConfig(relevance_factor=16.0, n_enrol=3, adapt_means=True)

FINAL_SCORE = ScoreConfig(top_c=0, norm="tnorm", cohort_size=150)

#: Streams compared in the feature ablation (Task 6).
COMPARISON_STREAMS: Dict[str, FeatureConfig] = {
    "IFD (temporal phase difference)": FINAL_FEATURE,
    "FDPD (finite-difference group delay)": FINAL_FEATURE.replace(stream="fdpd"),
    "Group-delay cepstra (exact |X|^2)": FINAL_FEATURE.replace(
        stream="mgdcc", n_bands=48, n_ceps=30, mgd_denom="exact"),
    "CosPhase (absolute phase, control)": FINAL_FEATURE.replace(stream="cosphase"),
}


# ---------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------


def splits():
    """``(ubm_files, heldout_files, cohort_spk2files)`` -- speaker disjoint."""
    return data.background_splits(heldout_frac=0.10,
                                  cohort_size=FINAL_SCORE.cohort_size,
                                  cohort_min_files=3, seed=C.SEED)


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------


def prepare_features(cfg: FeatureConfig = FINAL_FEATURE, verbose: bool = True
                     ) -> Tuple[Dict[str, torch.Tensor], Optional[NZ.GlobalNorm]]:
    """Build/load the feature cache and apply background-fitted normalisation."""
    feats = cache.build(cfg, verbose=verbose)
    norm = None
    if cfg.cmvn == "global":
        ubm_files, _, _ = splits()
        feats, norm = NZ.fit_and_apply(feats, ubm_files)
    return feats, norm


# ---------------------------------------------------------------------------
# UBM
# ---------------------------------------------------------------------------


def ubm_path(cfg: FeatureConfig, gcfg: GMMConfig) -> str:
    return os.path.join(C.CACHE_DIR,
                        f"ubm_{cfg.tag}_K{gcfg.n_components}_{gcfg.cov_type}"
                        f"_{gcfg.init}_s{gcfg.seed}.pt")


def train_ubm(feats: Dict[str, torch.Tensor], cfg: FeatureConfig = FINAL_FEATURE,
              gcfg: GMMConfig = FINAL_GMM, max_frames: int = 0,
              use_cache: bool = True, verbose: bool = False) -> GMM:
    """Train (or load) the UBM on pooled background frames."""
    path = ubm_path(cfg, gcfg)
    if use_cache and os.path.exists(path):
        return GMM.from_state_dict(torch.load(path, map_location="cpu",
                                              weights_only=False), gcfg.device)
    ubm_files, _, _ = splits()
    X = cache.pooled(feats, ubm_files, max_frames=max_frames, seed=gcfg.seed)
    dtype = torch.float64 if gcfg.dtype == "float64" else torch.float32
    ubm = GMM(gcfg.n_components, X.shape[1], gcfg.cov_type, dtype=dtype,
              device=gcfg.device)
    ubm.fit(X, gcfg, verbose=verbose)
    if use_cache:
        torch.save(ubm.state_dict(), path)
    return ubm


# ---------------------------------------------------------------------------
# full verification system
# ---------------------------------------------------------------------------


def score_split(split: str, feats: Dict[str, torch.Tensor], ubm: GMM,
                ecfg: EnrolConfig = FINAL_ENROL, scfg: ScoreConfig = FINAL_SCORE,
                cohort_models: Optional[Dict[str, GMM]] = None,
                impostor_files: Optional[Sequence[str]] = None
                ) -> Tuple[List[dict], torch.Tensor, torch.Tensor]:
    """Enrol the speakers of ``split`` and score its trials.

    Returns ``(trials, raw_scores, normalised_scores)``.
    """
    trials = data.trials(split)
    emap = data.enrol_map(split, ecfg.n_enrol)
    test_files = sorted({t["test_file"] for t in trials})
    models = SC.enrol_speakers(ubm, feats, emap, ecfg)

    S_raw, mi, fi = SC.score_matrix(ubm, models, feats, test_files, scfg.top_c)
    raw = SC.gather_trial_scores(S_raw, mi, fi, trials)

    if scfg.norm == "none":
        return trials, raw, raw
    if cohort_models is None or impostor_files is None:
        raise ValueError("score normalisation needs cohort models and impostor files")
    S_n, mi, fi = SC.normalise_scores(ubm, models, cohort_models, feats, test_files,
                                      impostor_files, scfg.norm, scfg.top_c)
    return trials, raw, SC.gather_trial_scores(S_n, mi, fi, trials)


def build_cohort(feats: Dict[str, torch.Tensor], ubm: GMM,
                 ecfg: EnrolConfig = FINAL_ENROL) -> Tuple[Dict[str, GMM], List[str]]:
    """MAP-adapted background cohort models + the held-out impostor utterances."""
    _, held_files, cohort = splits()
    cohort_models = SC.enrol_speakers(ubm, feats, cohort, ecfg)
    return cohort_models, held_files


def dev_eer(feats, ubm, ecfg: EnrolConfig = FINAL_ENROL,
            scfg: ScoreConfig = ScoreConfig(norm="none"), cohort_models=None,
            impostor_files=None) -> Tuple[float, torch.Tensor, List[int]]:
    trials, raw, sc = score_split("dev", feats, ubm, ecfg, scfg, cohort_models,
                                  impostor_files)
    y = [t["label"] for t in trials]
    return M.eer(sc, y)[0], sc, y
