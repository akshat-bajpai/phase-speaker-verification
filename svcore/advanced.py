"""Scoring refinements that stay inside the GMM-UBM / LLR framework.

* :func:`symmetric_score_matrix` -- bidirectional scoring.  The standard LLR
  asks "how well does the test utterance fit the claimed speaker's adapted
  model?".  With only ~390 enrolment frames the speaker model is itself a noisy
  estimate, so the reverse question -- "how well do the claimed speaker's
  enrolment frames fit a model adapted from the test utterance?" -- carries
  genuinely complementary evidence.  Both terms are average per-frame
  log-likelihood ratios against the same UBM; we average them.

* :func:`fuse` -- linear fusion of several systems' score matrices after
  standardising each, used to combine complementary phase-feature streams
  (an explicit stretch goal of the brief).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .config import EnrolConfig
from .gmm import GMM
from .map_adapt import map_adapt
from .scoring import MeanBankScorer, score_matrix, ubm_utt_stats


def adapt_per_utterance(ubm: GMM, feats: Dict[str, torch.Tensor],
                        files: Sequence[str], ecfg: EnrolConfig,
                        relevance_factor: Optional[float] = None
                        ) -> Tuple[torch.Tensor, Dict[str, int]]:
    """MAP-adapt one model per *test utterance*.  Returns stacked means."""
    means, index = [], {}
    for i, f in enumerate(files):
        X = feats[f]
        m = map_adapt(ubm, X.to(ubm.dtype), ecfg, relevance_factor)
        means.append(m.means)
        index[f] = i
    return torch.stack(means, dim=0), index


def reverse_score_matrix(ubm: GMM, feats: Dict[str, torch.Tensor],
                         enrol_map: Dict[str, List[str]],
                         test_files: Sequence[str], ecfg: EnrolConfig,
                         test_relevance: Optional[float] = None,
                         dtype: torch.dtype = torch.float32
                         ) -> Tuple[torch.Tensor, Dict[str, int], Dict[str, int]]:
    """(n_speakers, n_test_files) average per-frame LLR of the SPEAKER's
    enrolment frames under a model adapted from each TEST utterance."""
    means, fi = adapt_per_utterance(ubm, feats, test_files, ecfg, test_relevance)
    # a lighter-precision copy of the UBM keeps this O(spk x files) pass affordable
    light = GMM(ubm.K, ubm.D, ubm.cov_type, dtype=dtype)
    light.weights = ubm.weights.to(dtype)
    light.means = ubm.means.to(dtype)
    light.covars = ubm.covars.to(dtype)
    light._invalidate()
    bank = MeanBankScorer(light, means.to(dtype), list(test_files))

    speakers = sorted(enrol_map)
    S = torch.zeros(len(speakers), len(test_files), dtype=dtype)
    for i, spk in enumerate(speakers):
        Xe = torch.cat([feats[f] for f in enrol_map[spk]], dim=0).to(dtype)
        if Xe.shape[0] == 0:
            continue
        ubm_ll, _ = ubm_utt_stats(light, Xe, 0)
        S[i] = bank.utt_loglik(Xe) - ubm_ll
    return S.to(torch.float64), {s: i for i, s in enumerate(speakers)}, fi


def symmetric_score_matrix(ubm: GMM, models: Dict[str, GMM],
                           feats: Dict[str, torch.Tensor],
                           enrol_map: Dict[str, List[str]],
                           test_files: Sequence[str], ecfg: EnrolConfig,
                           weight: float = 0.5,
                           test_relevance: Optional[float] = None,
                           top_c: int = 0):
    """``weight * forward + (1 - weight) * reverse``, both as average LLRs."""
    Sf, mi, fi = score_matrix(ubm, models, feats, test_files, top_c)
    Sr, mi_r, fi_r = reverse_score_matrix(ubm, feats, enrol_map, test_files, ecfg,
                                          test_relevance)
    # align the reverse matrix onto the forward matrix's ordering
    rows = torch.tensor([mi_r[s] for s in sorted(mi, key=lambda k: mi[k])])
    cols = torch.tensor([fi_r[f] for f in sorted(fi, key=lambda k: fi[k])])
    Sr = Sr[rows][:, cols]
    return weight * Sf + (1.0 - weight) * Sr, mi, fi


def zscore(S: torch.Tensor) -> torch.Tensor:
    """Standardise a whole score matrix (a monotone map, so EER-neutral alone)."""
    return (S - S.mean()) / S.std().clamp_min(1e-9)


def fuse(mats: Sequence[torch.Tensor], weights: Optional[Sequence[float]] = None
         ) -> torch.Tensor:
    """Weighted sum of standardised score matrices (equal weights by default)."""
    w = list(weights) if weights is not None else [1.0 / len(mats)] * len(mats)
    tot = sum(w)
    out = None
    for wi, S in zip(w, mats):
        term = (wi / tot) * zscore(S)
        out = term if out is None else out + term
    return out


# ---------------------------------------------------------------------------
# adaptive symmetric score normalisation (as-norm)
# ---------------------------------------------------------------------------


def _top_stats(S: torch.Tensor, top_n: int, dim: int):
    """Mean/std of the ``top_n`` largest entries along ``dim``."""
    n = min(top_n, S.shape[dim]) if top_n else S.shape[dim]
    v = torch.topk(S, n, dim=dim).values
    return v.mean(dim=dim), v.std(dim=dim, unbiased=True).clamp_min(1e-6)


def asnorm_matrix(ubm, target_models, cohort_models, feats,
                  test_files, cohort_files, top_n: int = 100,
                  symmetric: bool = True):
    """Adaptive symmetric score normalisation.

    Plain T-norm standardises a trial by the mean/std of *all* cohort models
    scored on that test utterance.  Adaptive normalisation uses only the ``top_n``
    cohort entries -- the cohort models that actually resemble the utterance --
    which estimates the local impostor distribution where the decision is made
    instead of a global one dominated by irrelevant speakers.

    ``symmetric`` averages the utterance-side and the speaker-side
    normalisation (the standard as-norm1); set it False for the utterance side
    only.

    Returns ``(S_norm, model_index, file_index)``.
    """
    from .scoring import score_matrix
    S, mi, fi = score_matrix(ubm, target_models, feats, test_files, 0)
    # cohort models vs each test utterance -> per-utterance impostor distribution
    Et, _, fi_t = score_matrix(ubm, cohort_models, feats, test_files, 0)
    cols = torch.tensor([fi_t[f] for f in sorted(fi, key=lambda k: fi[k])])
    Et = Et[:, cols]
    mu_e, sd_e = _top_stats(Et, top_n, dim=0)                 # (n_test,)
    out = (S - mu_e.unsqueeze(0)) / sd_e.unsqueeze(0)

    if symmetric:
        # target models vs cohort utterances -> per-speaker impostor distribution
        Es, mi_s, _ = score_matrix(ubm, target_models, feats, cohort_files, 0)
        rows = torch.tensor([mi_s[s] for s in sorted(mi, key=lambda k: mi[k])])
        Es = Es[rows]
        mu_s, sd_s = _top_stats(Es, top_n, dim=1)             # (n_models,)
        out = 0.5 * (out + (S - mu_s.unsqueeze(1)) / sd_s.unsqueeze(1))
    return out, mi, fi
