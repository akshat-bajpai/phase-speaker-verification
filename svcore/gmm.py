"""Gaussian mixture models with EM, implemented in PyTorch only.

Supports four covariance structures -- ``full``, ``diag``, ``tied``,
``spherical`` -- random and k-means(++) initialisation, variance flooring /
ridge regularisation, BIC/AIC, and numerically stable log-domain likelihoods
(``torch.logsumexp``, Cholesky factorisations instead of explicit inverses).
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

import torch

from .config import GMMConfig

LOG_2PI = math.log(2.0 * math.pi)
COV_TYPES = ("full", "diag", "tied", "spherical")


# ---------------------------------------------------------------------------
# k-means (PyTorch) -- used as an initialiser for EM
# ---------------------------------------------------------------------------


def kmeans_plusplus(X: torch.Tensor, k: int, generator: torch.Generator) -> torch.Tensor:
    """k-means++ seeding.  X: (N, D) -> centres (k, D)."""
    N = X.shape[0]
    idx = int(torch.randint(N, (1,), generator=generator, device=X.device))
    centres = [X[idx]]
    d2 = (X - centres[0]).pow(2).sum(1)
    for _ in range(1, k):
        probs = d2.clamp_min(0)
        total = probs.sum()
        if not torch.isfinite(total) or total <= 0:
            nxt = int(torch.randint(N, (1,), generator=generator, device=X.device))
        else:
            nxt = int(torch.multinomial(probs / total, 1, generator=generator))
        centres.append(X[nxt])
        d2 = torch.minimum(d2, (X - X[nxt]).pow(2).sum(1))
    return torch.stack(centres)


def kmeans(X: torch.Tensor, k: int, n_iter: int = 25, seed: int = 0,
           init: str = "++", chunk: int = 16384) -> Tuple[torch.Tensor, torch.Tensor]:
    """Lloyd's algorithm.  Returns ``(centres (k,D), assignments (N,))``."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    N = X.shape[0]
    if init == "++":
        C = kmeans_plusplus(X, k, g)
    else:
        sel = torch.randperm(N, generator=g)[:k]
        C = X[sel].clone()
    assign = torch.zeros(N, dtype=torch.long, device=X.device)
    for _ in range(n_iter):
        for s in range(0, N, chunk):
            xb = X[s:s + chunk]
            d = (xb.pow(2).sum(1, keepdim=True) - 2.0 * xb @ C.t()
                 + C.pow(2).sum(1))
            assign[s:s + chunk] = d.argmin(1)
        counts = torch.bincount(assign, minlength=k).to(X.dtype)
        newC = torch.zeros_like(C).index_add_(0, assign, X)
        empty = counts == 0
        newC = newC / counts.clamp_min(1.0).unsqueeze(1)
        if bool(empty.any()):
            # re-seed empty clusters at random data points
            n_e = int(empty.sum())
            sel = torch.randperm(N, generator=g)[:n_e]
            newC[empty] = X[sel]
        shift = (newC - C).pow(2).sum().sqrt()
        C = newC
        if float(shift) < 1e-6:
            break
    return C, assign


# ---------------------------------------------------------------------------
# GMM
# ---------------------------------------------------------------------------


class GMM:
    """A Gaussian mixture model.  All state lives in plain torch tensors."""

    def __init__(self, n_components: int, dim: int, cov_type: str = "diag",
                 dtype: torch.dtype = torch.float64, device: str = "cpu"):
        if cov_type not in COV_TYPES:
            raise ValueError(f"cov_type must be one of {COV_TYPES}")
        self.K, self.D, self.cov_type = n_components, dim, cov_type
        self.dtype, self.device = dtype, torch.device(device)
        self.weights = torch.full((self.K,), 1.0 / self.K, dtype=dtype, device=device)
        self.means = torch.zeros(self.K, dim, dtype=dtype, device=device)
        self.covars = self._empty_cov()
        self._cache: Dict[str, torch.Tensor] = {}
        self.history: List[float] = []
        self.fit_seconds: float = 0.0
        self.n_iter_run: int = 0
        self.converged: bool = False

    # -- shapes ------------------------------------------------------------
    def _empty_cov(self) -> torch.Tensor:
        if self.cov_type == "diag":
            return torch.ones(self.K, self.D, dtype=self.dtype, device=self.device)
        if self.cov_type == "full":
            return torch.eye(self.D, dtype=self.dtype, device=self.device).repeat(self.K, 1, 1)
        if self.cov_type == "tied":
            return torch.eye(self.D, dtype=self.dtype, device=self.device)
        return torch.ones(self.K, dtype=self.dtype, device=self.device)  # spherical

    @property
    def n_parameters(self) -> int:
        """Free parameters: weights (K-1) + means (K*D) + covariance terms."""
        n_cov = {"diag": self.K * self.D,
                 "full": self.K * self.D * (self.D + 1) // 2,
                 "tied": self.D * (self.D + 1) // 2,
                 "spherical": self.K}[self.cov_type]
        return (self.K - 1) + self.K * self.D + n_cov

    # -- cached precision factors -----------------------------------------
    def _invalidate(self):
        self._cache.clear()

    def _prepare(self):
        """Pre-compute Cholesky precision factors and log-normalisers."""
        if "log_norm" in self._cache:
            return
        if self.cov_type == "diag":
            var = self.covars
            self._cache["inv_var"] = 1.0 / var
            self._cache["log_norm"] = -0.5 * (self.D * LOG_2PI + torch.log(var).sum(1))
            self._cache["mu_over_var"] = self.means / var
            self._cache["maha_const"] = (self.means.pow(2) / var).sum(1)
        elif self.cov_type == "spherical":
            var = self.covars.unsqueeze(1)
            self._cache["inv_var"] = 1.0 / var
            self._cache["log_norm"] = -0.5 * (self.D * LOG_2PI
                                              + self.D * torch.log(self.covars))
        elif self.cov_type == "tied":
            L = torch.linalg.cholesky(self.covars)
            # P = L^{-T}; Sigma^{-1} = P P^T,  whiten: (x-mu) @ P
            P = torch.linalg.solve_triangular(
                L, torch.eye(self.D, dtype=self.dtype, device=self.device),
                upper=False).t()
            self._cache["P"] = P
            logdet = 2.0 * torch.log(torch.diagonal(L)).sum()
            self._cache["log_norm"] = (-0.5 * (self.D * LOG_2PI + logdet)
                                       ).expand(self.K).clone()
            self._cache["mu_w"] = self.means @ P
        else:  # full
            L = torch.linalg.cholesky(self.covars)                 # (K,D,D)
            eye = torch.eye(self.D, dtype=self.dtype, device=self.device)
            P = torch.linalg.solve_triangular(
                L, eye.expand(self.K, self.D, self.D), upper=False).transpose(1, 2)
            self._cache["P"] = P
            logdet = 2.0 * torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(1)
            self._cache["log_norm"] = -0.5 * (self.D * LOG_2PI + logdet)
            self._cache["mu_w"] = torch.einsum("kd,kde->ke", self.means, P)

    # -- densities ---------------------------------------------------------
    def log_prob_components(self, X: torch.Tensor) -> torch.Tensor:
        """(N,D) -> (N,K) log N(x_n | mu_k, Sigma_k) (no mixture weights)."""
        X = X.to(dtype=self.dtype, device=self.device)
        self._prepare()
        ln = self._cache["log_norm"]
        if self.cov_type == "diag":
            maha = (X.pow(2) @ self._cache["inv_var"].t()
                    - 2.0 * X @ self._cache["mu_over_var"].t()
                    + self._cache["maha_const"])
            return ln - 0.5 * maha
        if self.cov_type == "spherical":
            sq = (X.pow(2).sum(1, keepdim=True) - 2.0 * X @ self.means.t()
                  + self.means.pow(2).sum(1))
            return ln - 0.5 * sq * self._cache["inv_var"].squeeze(1)
        if self.cov_type == "tied":
            Xw = X @ self._cache["P"]
            muw = self._cache["mu_w"]
            sq = (Xw.pow(2).sum(1, keepdim=True) - 2.0 * Xw @ muw.t()
                  + muw.pow(2).sum(1))
            return ln - 0.5 * sq
        # full
        Xw = torch.einsum("nd,kde->nke", X, self._cache["P"])       # (N,K,D)
        diff = Xw - self._cache["mu_w"].unsqueeze(0)
        return ln - 0.5 * diff.pow(2).sum(2)

    def log_weighted(self, X: torch.Tensor) -> torch.Tensor:
        """(N,K) log w_k + log N(x|k)."""
        return self.log_prob_components(X) + torch.log(self.weights).unsqueeze(0)

    def log_likelihood(self, X: torch.Tensor, chunk: int = 32768,
                       top_c: int = 0) -> torch.Tensor:
        """(N,) per-frame log p(x).  ``top_c>0`` keeps only the C best components."""
        out = []
        for s in range(0, X.shape[0], chunk):
            lw = self.log_weighted(X[s:s + chunk])
            if top_c and top_c < self.K:
                lw = torch.topk(lw, top_c, dim=1).values
            out.append(torch.logsumexp(lw, dim=1))
        return torch.cat(out) if out else torch.zeros(0, dtype=self.dtype)

    def mean_log_likelihood(self, X: torch.Tensor, chunk: int = 32768) -> float:
        if X.shape[0] == 0:
            return float("nan")
        return float(self.log_likelihood(X, chunk).mean())

    def bic(self, X: torch.Tensor) -> float:
        """BIC = -2 * total loglik + p * log N  (lower is better)."""
        N = X.shape[0]
        ll = float(self.log_likelihood(X).sum())
        return -2.0 * ll + self.n_parameters * math.log(N)

    def aic(self, X: torch.Tensor) -> float:
        ll = float(self.log_likelihood(X).sum())
        return -2.0 * ll + 2.0 * self.n_parameters

    # -- initialisation ----------------------------------------------------
    def initialise(self, X: torch.Tensor, cfg: GMMConfig, seed: Optional[int] = None):
        seed = cfg.seed if seed is None else seed
        X = X.to(dtype=self.dtype, device=self.device)
        g = torch.Generator(device="cpu").manual_seed(seed)
        N = X.shape[0]
        global_var = X.var(dim=0, unbiased=False).clamp_min(cfg.var_floor)
        if cfg.init == "kmeans":
            C, assign = kmeans(X, self.K, cfg.kmeans_iter, seed)
            self.means = C
            counts = torch.bincount(assign, minlength=self.K).to(self.dtype)
            self.weights = (counts + 1e-6)
            self.weights = self.weights / self.weights.sum()
            resp = torch.zeros(N, self.K, dtype=self.dtype, device=self.device)
            resp[torch.arange(N, device=self.device), assign] = 1.0
            self._set_covariances(X, resp, counts.clamp_min(1.0), cfg)
        elif cfg.init == "random":
            sel = torch.randperm(N, generator=g)[:self.K]
            self.means = X[sel].clone()
            self.weights = torch.full((self.K,), 1.0 / self.K,
                                      dtype=self.dtype, device=self.device)
            self._set_isotropic(global_var)
        else:
            raise ValueError(cfg.init)
        self._invalidate()

    def _set_isotropic(self, global_var: torch.Tensor):
        if self.cov_type == "diag":
            self.covars = global_var.unsqueeze(0).repeat(self.K, 1).contiguous()
        elif self.cov_type == "spherical":
            self.covars = torch.full((self.K,), float(global_var.mean()),
                                     dtype=self.dtype, device=self.device)
        elif self.cov_type == "tied":
            self.covars = torch.diag(global_var)
        else:
            self.covars = torch.diag(global_var).repeat(self.K, 1, 1)

    # -- EM ---------------------------------------------------------------
    def _e_step(self, X: torch.Tensor, chunk: int) -> Tuple[torch.Tensor, float]:
        """Returns (responsibilities (N,K), mean log-likelihood)."""
        N = X.shape[0]
        resp = torch.empty(N, self.K, dtype=self.dtype, device=self.device)
        total = 0.0
        for s in range(0, N, chunk):
            lw = self.log_weighted(X[s:s + chunk])
            lse = torch.logsumexp(lw, dim=1, keepdim=True)
            resp[s:s + chunk] = torch.exp(lw - lse)
            total += float(lse.sum())
        return resp, total / N

    def _set_covariances(self, X, resp, Nk, cfg: GMMConfig):
        if self.cov_type == "diag":
            ex2 = (resp.t() @ X.pow(2)) / Nk.unsqueeze(1)
            var = ex2 - self.means.pow(2)
            self.covars = var.clamp_min(0) + cfg.var_floor
        elif self.cov_type == "spherical":
            ex2 = (resp.t() @ X.pow(2)) / Nk.unsqueeze(1)
            var = (ex2 - self.means.pow(2)).clamp_min(0).mean(1)
            self.covars = var + cfg.var_floor
        elif self.cov_type == "tied":
            S = X.t() @ X - (self.means * Nk.unsqueeze(1)).t() @ self.means
            S = S / Nk.sum()
            S = 0.5 * (S + S.t())
            self.covars = S + (cfg.reg_covar + cfg.var_floor) * torch.eye(
                self.D, dtype=self.dtype, device=self.device)
        else:  # full
            # Batched second moments: S_k = sum_n r_nk x_n x_n^T accumulated as a
            # single matrix product per chunk, which is far faster than looping
            # over components (one GEMM instead of K rank-updates).
            eye = torch.eye(self.D, dtype=self.dtype, device=self.device)
            D = self.D
            acc = torch.zeros(self.K, D * D, dtype=self.dtype, device=self.device)
            step = max(1, min(X.shape[0], max(1, (1 << 22) // max(1, D * D))))
            for s in range(0, X.shape[0], step):
                xb = X[s:s + step]
                outer = (xb.unsqueeze(2) * xb.unsqueeze(1)).reshape(xb.shape[0], D * D)
                acc += resp[s:s + step].t() @ outer
            covs = acc.view(self.K, D, D) / Nk.view(-1, 1, 1)
            covs = covs - self.means.unsqueeze(2) * self.means.unsqueeze(1)
            covs = 0.5 * (covs + covs.transpose(1, 2))
            self.covars = covs + (cfg.reg_covar + cfg.var_floor) * eye

    def _m_step(self, X: torch.Tensor, resp: torch.Tensor, cfg: GMMConfig,
                generator: torch.Generator, reinit: bool) -> int:
        N = X.shape[0]
        Nk = resp.sum(0)
        n_reinit = 0
        if reinit:
            bad = Nk < max(cfg.min_weight * N, 1.0)
            if bool(bad.any()):
                n_reinit = int(bad.sum())
                sel = torch.randperm(N, generator=generator)[:n_reinit]
                resp[:, bad] = 0.0
                resp[sel, torch.nonzero(bad).squeeze(1)] = 1.0
                Nk = resp.sum(0)
        Nk_safe = Nk.clamp_min(1e-10)
        self.weights = (Nk / N).clamp_min(1e-12)
        self.weights = self.weights / self.weights.sum()
        self.means = (resp.t() @ X) / Nk_safe.unsqueeze(1)
        self._set_covariances(X, resp, Nk_safe, cfg)
        self._invalidate()
        return n_reinit

    def fit(self, X: torch.Tensor, cfg: GMMConfig, seed: Optional[int] = None,
            chunk: int = 32768, verbose: bool = False,
            reinit_degenerate: bool = True) -> "GMM":
        """Run EM to convergence.  Records ``history`` (mean log-lik per iter)."""
        X = X.to(dtype=self.dtype, device=self.device)
        seed = cfg.seed if seed is None else seed
        g = torch.Generator(device="cpu").manual_seed(seed + 1)
        t0 = time.time()
        self.initialise(X, cfg, seed=seed)
        self.history, prev = [], -float("inf")
        self.converged = False
        for it in range(cfg.max_iter):
            resp, mll = self._e_step(X, chunk)
            self.history.append(mll)
            self._m_step(X, resp, cfg, g, reinit_degenerate)
            self.n_iter_run = it + 1
            if verbose and (it % 10 == 0 or it == cfg.max_iter - 1):
                print(f"    iter {it:3d}  mean loglik {mll:+.5f}")
            if it > 0:
                denom = max(abs(prev), 1e-12)
                if abs(mll - prev) / denom < cfg.tol:
                    self.converged = True
                    prev = mll
                    break
            prev = mll
        # final E-step so that `history[-1]` reflects the returned parameters
        _, mll = self._e_step(X, chunk)
        self.history.append(mll)
        self.fit_seconds = time.time() - t0
        return self

    # -- occupancy diagnostics --------------------------------------------
    def occupancy(self, X: torch.Tensor, chunk: int = 32768) -> torch.Tensor:
        """Soft counts n_k over X (used to detect degenerate components)."""
        Nk = torch.zeros(self.K, dtype=self.dtype, device=self.device)
        for s in range(0, X.shape[0], chunk):
            lw = self.log_weighted(X[s:s + chunk])
            Nk += torch.exp(lw - torch.logsumexp(lw, 1, keepdim=True)).sum(0)
        return Nk

    def min_variance(self) -> float:
        if self.cov_type == "diag":
            return float(self.covars.min())
        if self.cov_type == "spherical":
            return float(self.covars.min())
        if self.cov_type == "tied":
            return float(torch.linalg.eigvalsh(self.covars).min())
        return float(torch.linalg.eigvalsh(self.covars).min())

    def condition_number(self) -> float:
        if self.cov_type in ("diag", "spherical"):
            c = self.covars if self.cov_type == "diag" else self.covars.unsqueeze(1)
            return float((c.max(dim=-1).values / c.min(dim=-1).values.clamp_min(1e-300)).max())
        ev = torch.linalg.eigvalsh(self.covars)
        return float((ev.max(dim=-1).values / ev.min(dim=-1).values.clamp_min(1e-300)).max())

    # -- (de)serialisation -------------------------------------------------
    def state_dict(self) -> dict:
        return {"K": self.K, "D": self.D, "cov_type": self.cov_type,
                "weights": self.weights, "means": self.means, "covars": self.covars,
                "history": self.history, "fit_seconds": self.fit_seconds,
                "n_iter_run": self.n_iter_run, "converged": self.converged,
                "dtype": str(self.dtype)}

    @classmethod
    def from_state_dict(cls, sd: dict, device: str = "cpu") -> "GMM":
        dtype = {"torch.float64": torch.float64,
                 "torch.float32": torch.float32}[sd.get("dtype", "torch.float64")]
        m = cls(sd["K"], sd["D"], sd["cov_type"], dtype=dtype, device=device)
        m.weights = sd["weights"].to(dtype=dtype, device=device)
        m.means = sd["means"].to(dtype=dtype, device=device)
        m.covars = sd["covars"].to(dtype=dtype, device=device)
        m.history = sd.get("history", [])
        m.fit_seconds = sd.get("fit_seconds", 0.0)
        m.n_iter_run = sd.get("n_iter_run", 0)
        m.converged = sd.get("converged", False)
        m._invalidate()
        return m

    def clone(self) -> "GMM":
        m = GMM(self.K, self.D, self.cov_type, self.dtype, str(self.device))
        m.weights = self.weights.clone()
        m.means = self.means.clone()
        m.covars = self.covars.clone()
        return m
