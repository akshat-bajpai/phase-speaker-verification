"""Trial scoring: average per-frame log-likelihood ratio, top-C scoring and
score normalisation (T-norm / Z-norm / ZT-norm).  PyTorch only.

The batched scorer exploits the structure of mean-only MAP adaptation: all
speaker models share the UBM weights and covariances and differ only in their
means, so the Mahalanobis terms for *every* speaker model against one test
utterance reduce to a single matrix product.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .config import EnrolConfig, ScoreConfig
from .gmm import GMM
from .map_adapt import map_adapt


# ---------------------------------------------------------------------------
# enrolment
# ---------------------------------------------------------------------------


def enrol_speakers(ubm: GMM, feats: Dict[str, torch.Tensor],
                   enrol_map: Dict[str, List[str]],
                   cfg: EnrolConfig = EnrolConfig(),
                   relevance_factor: Optional[float] = None) -> Dict[str, GMM]:
    """MAP-adapt one model per speaker from its (first ``cfg.n_enrol``) utterances."""
    models: Dict[str, GMM] = {}
    for spk, files in enrol_map.items():
        X = torch.cat([feats[f] for f in files[:cfg.n_enrol]], dim=0)
        models[spk] = map_adapt(ubm, X, cfg, relevance_factor)
    return models


# ---------------------------------------------------------------------------
# batched mean-only scorer
# ---------------------------------------------------------------------------


class MeanBankScorer:
    """Scores many mean-only-adapted models against one utterance at a time.

    ``means``: (M, K, D) stacked speaker means sharing ``ubm`` weights/covars.
    Only the ``diag`` covariance type is supported by the fast path (the
    covariance structure actually used by the final system); other structures
    fall back to :func:`llr_scores_loop`.
    """

    def __init__(self, ubm: GMM, means: torch.Tensor, keys: Sequence[str]):
        if ubm.cov_type != "diag":
            raise NotImplementedError("MeanBankScorer requires a diagonal UBM")
        self.ubm = ubm
        self.keys = list(keys)
        self.index = {k: i for i, k in enumerate(self.keys)}
        self.dtype, self.device = ubm.dtype, ubm.device
        M, K, D = means.shape
        self.M, self.K, self.D = M, K, D
        inv_var = (1.0 / ubm.covars)                                   # (K,D)
        self.log_w = torch.log(ubm.weights)                            # (K,)
        self.log_norm = -0.5 * (D * math.log(2 * math.pi)
                                + torch.log(ubm.covars).sum(1))        # (K,)
        self.inv_var = inv_var
        mu_over_var = means * inv_var.unsqueeze(0)                     # (M,K,D)
        self.mu_over_var = mu_over_var.reshape(M * K, D).contiguous()
        self.maha_const = (means.pow(2) * inv_var.unsqueeze(0)).sum(2)  # (M,K)

    @classmethod
    def from_models(cls, ubm: GMM, models: Dict[str, GMM]) -> "MeanBankScorer":
        """Stack mean-only-adapted models.

        The fast path is valid *only* if every model shares the UBM's weights and
        covariances, which is what mean-only MAP guarantees.  We check it rather
        than assume it: silently ignoring adapted weights/covariances would make
        any weight- or covariance-adaptation experiment read back as mean-only.
        Use :func:`llr_scores_loop` for models that differ in those parameters.
        """
        keys = list(models.keys())
        for k in keys:
            m = models[k]
            if (m.cov_type != ubm.cov_type
                    or not torch.allclose(m.weights, ubm.weights)
                    or not torch.allclose(m.covars, ubm.covars)):
                raise ValueError(
                    f"model '{k}' does not share the UBM weights/covariances; "
                    "MeanBankScorer handles mean-only adaptation. Use "
                    "llr_scores_loop() for weight- or covariance-adapted models.")
        means = torch.stack([models[k].means for k in keys], dim=0)
        return cls(ubm, means, keys)

    def frame_logprob(self, X: torch.Tensor) -> torch.Tensor:
        """(T,D) -> (T, M, K) log w_k + log N(x_t | mu^m_k, Sigma_k)."""
        X = X.to(dtype=self.dtype, device=self.device)
        T = X.shape[0]
        quad = X.pow(2) @ self.inv_var.t()                             # (T,K)
        cross = X @ self.mu_over_var.t()                               # (T,M*K)
        cross = cross.view(T, self.M, self.K)
        maha = quad.unsqueeze(1) - 2.0 * cross + self.maha_const.unsqueeze(0)
        return self.log_norm + self.log_w - 0.5 * maha                 # (T,M,K)

    def utt_loglik(self, X: torch.Tensor, top_idx: Optional[torch.Tensor] = None
                   ) -> torch.Tensor:
        """Mean per-frame log-likelihood of ``X`` under every model: (M,).

        If ``top_idx`` (T,C) is given, only those components are summed -- the
        standard fast-scoring approximation, with the component set chosen by the
        UBM and shared by all models so the ratio stays consistent.
        """
        lw = self.frame_logprob(X)                                     # (T,M,K)
        if top_idx is not None:
            idx = top_idx.unsqueeze(1).expand(-1, self.M, -1)          # (T,M,C)
            lw = torch.gather(lw, 2, idx)
        return torch.logsumexp(lw, dim=2).mean(dim=0)                  # (M,)


def ubm_utt_stats(ubm: GMM, X: torch.Tensor, top_c: int = 0
                  ) -> Tuple[float, Optional[torch.Tensor]]:
    """Mean per-frame UBM log-likelihood and (optionally) top-C indices."""
    lw = ubm.log_weighted(X.to(dtype=ubm.dtype, device=ubm.device))    # (T,K)
    if top_c and top_c < ubm.K:
        top = torch.topk(lw, top_c, dim=1)
        return float(torch.logsumexp(top.values, dim=1).mean()), top.indices
    return float(torch.logsumexp(lw, dim=1).mean()), None


# ---------------------------------------------------------------------------
# score matrices
# ---------------------------------------------------------------------------


def score_matrix(ubm: GMM, models: Dict[str, GMM], feats: Dict[str, torch.Tensor],
                 test_files: Sequence[str], top_c: int = 0
                 ) -> Tuple[torch.Tensor, Dict[str, int], Dict[str, int]]:
    """Average per-frame LLR of every (model, test file) pair.

    Returns ``(S (n_models, n_files), model_index, file_index)``.
    """
    bank = MeanBankScorer.from_models(ubm, models)
    files = list(test_files)
    S = torch.zeros(bank.M, len(files), dtype=bank.dtype, device=bank.device)
    for j, f in enumerate(files):
        X = feats[f]
        if X.shape[0] == 0:
            continue
        ubm_ll, top_idx = ubm_utt_stats(ubm, X, top_c)
        S[:, j] = bank.utt_loglik(X, top_idx) - ubm_ll
    return S, bank.index, {f: j for j, f in enumerate(files)}


def llr_scores_loop(ubm: GMM, models: Dict[str, GMM],
                    feats: Dict[str, torch.Tensor], trials: Sequence[dict],
                    top_c: int = 0) -> torch.Tensor:
    """Reference (slow) scorer that works for any covariance structure."""
    out = []
    for t in trials:
        X = feats[t["test_file"]]
        if X.shape[0] == 0:
            out.append(0.0)
            continue
        ubm_ll, top_idx = ubm_utt_stats(ubm, X, top_c)
        spk = models[t["claimed_id"]]
        lw = spk.log_weighted(X)
        if top_idx is not None:
            lw = torch.gather(lw, 1, top_idx)
        out.append(float(torch.logsumexp(lw, dim=1).mean()) - ubm_ll)
    return torch.tensor(out, dtype=torch.float64)


def gather_trial_scores(S: torch.Tensor, mi: Dict[str, int], fi: Dict[str, int],
                        trials: Sequence[dict]) -> torch.Tensor:
    rows = torch.tensor([mi[t["claimed_id"]] for t in trials])
    cols = torch.tensor([fi[t["test_file"]] for t in trials])
    return S[rows, cols].to(torch.float64)


# ---------------------------------------------------------------------------
# score normalisation
# ---------------------------------------------------------------------------


def _zstats(S: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Row-wise (per-model) mean/std -- the Z-norm statistics."""
    return S.mean(dim=1), S.std(dim=1, unbiased=True).clamp_min(1e-6)


def _tstats(S: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Column-wise (per-test-utterance) mean/std -- the T-norm statistics."""
    return S.mean(dim=0), S.std(dim=0, unbiased=True).clamp_min(1e-6)


def normalise_scores(ubm: GMM, target_models: Dict[str, GMM],
                     cohort_models: Dict[str, GMM],
                     feats: Dict[str, torch.Tensor], test_files: Sequence[str],
                     impostor_files: Sequence[str], mode: str = "tnorm",
                     top_c: int = 0) -> Tuple[torch.Tensor, Dict[str, int], Dict[str, int]]:
    """Return a normalised score matrix (n_target_models, n_test_files).

    * ``tnorm``  -- subtract/divide by the per-utterance statistics of the
      MAP-adapted background cohort models scored on that utterance.
      Compensates for utterance-dependent score offsets.
    * ``znorm``  -- subtract/divide by the per-model statistics obtained by
      scoring that speaker model against held-out background utterances.
      Compensates for speaker-model-dependent offsets.
    * ``ztnorm`` -- Z-norm first, then T-norm computed on Z-normalised cohort
      scores (the standard composition order).
    """
    S, mi, fi = score_matrix(ubm, target_models, feats, test_files, top_c)
    if mode == "none":
        return S, mi, fi

    if mode in ("znorm", "ztnorm"):
        Sz, _, _ = score_matrix(ubm, target_models, feats, impostor_files, top_c)
        zmu, zsd = _zstats(Sz)
        S = (S - zmu.unsqueeze(1)) / zsd.unsqueeze(1)

    if mode in ("tnorm", "ztnorm"):
        Ct, _, _ = score_matrix(ubm, cohort_models, feats, test_files, top_c)
        if mode == "ztnorm":
            Cz, _, _ = score_matrix(ubm, cohort_models, feats, impostor_files, top_c)
            czmu, czsd = _zstats(Cz)
            Ct = (Ct - czmu.unsqueeze(1)) / czsd.unsqueeze(1)
        tmu, tsd = _tstats(Ct)
        S = (S - tmu.unsqueeze(0)) / tsd.unsqueeze(0)
    return S, mi, fi
