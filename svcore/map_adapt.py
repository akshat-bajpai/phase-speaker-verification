"""MAP adaptation of a UBM to a target speaker (PyTorch only).

Mean-only adaptation is the required baseline (Reynolds, Quatieri & Dunn 2000,
"Speaker Verification Using Adapted Gaussian Mixture Models", Digital Signal
Processing 10(1-3)).  Weight and diagonal-covariance adaptation are provided as
the optional extension described in Task 4.
"""
from __future__ import annotations

from typing import Tuple

import torch

from .config import EnrolConfig
from .gmm import GMM


def sufficient_statistics(ubm: GMM, X: torch.Tensor, chunk: int = 32768
                          ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Zeroth/first/second-order Baum-Welch statistics of ``X`` w.r.t. ``ubm``.

    Returns ``(n_k (K,), F_k (K,D), S_k (K,D))`` where
    ``n_k = sum_t gamma_k(t)``, ``F_k = sum_t gamma_k(t) x_t`` and
    ``S_k = sum_t gamma_k(t) x_t^2`` (element-wise square, for diagonal
    covariance adaptation).
    """
    K, D = ubm.K, ubm.D
    n_k = torch.zeros(K, dtype=ubm.dtype, device=ubm.device)
    F_k = torch.zeros(K, D, dtype=ubm.dtype, device=ubm.device)
    S_k = torch.zeros(K, D, dtype=ubm.dtype, device=ubm.device)
    X = X.to(dtype=ubm.dtype, device=ubm.device)
    for s in range(0, X.shape[0], chunk):
        xb = X[s:s + chunk]
        lw = ubm.log_weighted(xb)
        gamma = torch.exp(lw - torch.logsumexp(lw, dim=1, keepdim=True))   # (n,K)
        n_k += gamma.sum(0)
        F_k += gamma.t() @ xb
        S_k += gamma.t() @ xb.pow(2)
    return n_k, F_k, S_k


def map_adapt(ubm: GMM, X: torch.Tensor, cfg: EnrolConfig = EnrolConfig(),
              relevance_factor: float | None = None) -> GMM:
    """Return a speaker GMM adapted from ``ubm`` on enrolment frames ``X``.

    ``alpha_k = n_k / (n_k + r)`` interpolates between the UBM prior
    (``alpha -> 0``, i.e. ``r -> inf`` or no enrolment occupancy) and the
    maximum-likelihood estimate from the enrolment data (``alpha -> 1``, i.e.
    ``r -> 0``).  Components with ``n_k = 0`` keep their UBM parameters exactly.
    """
    r = cfg.relevance_factor if relevance_factor is None else relevance_factor
    spk = ubm.clone()
    if X.shape[0] == 0:
        return spk
    n_k, F_k, S_k = sufficient_statistics(ubm, X)
    T = X.shape[0]

    if cfg.adapt_means:
        alpha = (n_k / (n_k + r)).unsqueeze(1) if r > 0 else torch.ones_like(n_k).unsqueeze(1)
        Ex = F_k / n_k.clamp_min(1e-10).unsqueeze(1)
        spk.means = alpha * Ex + (1.0 - alpha) * ubm.means
        # zero-occupancy components fall back to the UBM mean exactly
        zero = (n_k <= 0).unsqueeze(1)
        spk.means = torch.where(zero, ubm.means, spk.means)

    if cfg.adapt_weights:
        a_w = n_k / (n_k + r) if r > 0 else torch.ones_like(n_k)
        w = a_w * (n_k / T) + (1.0 - a_w) * ubm.weights
        spk.weights = (w / w.sum()).clamp_min(1e-12)
        spk.weights = spk.weights / spk.weights.sum()

    if cfg.adapt_covars:
        if ubm.cov_type != "diag":
            raise NotImplementedError("covariance adaptation implemented for diag only")
        a_v = (n_k / (n_k + r)).unsqueeze(1) if r > 0 else torch.ones_like(n_k).unsqueeze(1)
        Ex2 = S_k / n_k.clamp_min(1e-10).unsqueeze(1)
        var = (a_v * Ex2 + (1.0 - a_v) * (ubm.covars + ubm.means.pow(2))
               - spk.means.pow(2))
        floor = ubm.covars * 1e-2
        spk.covars = torch.maximum(var, floor)
        spk.covars = torch.where((n_k <= 0).unsqueeze(1), ubm.covars, spk.covars)

    spk._invalidate()
    return spk


def train_from_scratch(X: torch.Tensor, n_components: int, gmm_cfg) -> GMM:
    """Baseline for Task 4: a speaker GMM trained by EM on enrolment data only."""
    dtype = torch.float64 if gmm_cfg.dtype == "float64" else torch.float32
    m = GMM(n_components, X.shape[1], gmm_cfg.cov_type, dtype=dtype,
            device=gmm_cfg.device)
    return m.fit(X, gmm_cfg)
