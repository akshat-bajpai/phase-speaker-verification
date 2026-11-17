"""Dataset-level feature normalisation, fitted on permitted background data only.

Per-utterance CMVN is the usual choice for magnitude features because it removes
channel and gain effects.  Phase-difference features are already invariant to
gain, and their per-utterance mean is itself speaker-discriminative (it encodes
the average deviation of each band's dominant partial from the bin centre, i.e.
harmonic placement).  Removing it therefore *destroys* information on this
corpus.  We instead standardise with a single global mean/variance estimated on
the background UBM-training frames, which only improves the conditioning of the
GMM without touching within-utterance structure.
"""
from __future__ import annotations

from typing import Dict, Sequence

import torch


class GlobalNorm:
    """Affine standardisation ``(x - mean) / std`` with globally pooled statistics."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean, self.std = mean, std

    @classmethod
    def fit(cls, X: torch.Tensor, eps: float = 1e-5) -> "GlobalNorm":
        return cls(X.mean(dim=0), X.std(dim=0, unbiased=False).clamp_min(eps))

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 0:
            return x
        return (x - self.mean.to(x.dtype)) / self.std.to(x.dtype)

    def apply_all(self, feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: self.apply(v) for k, v in feats.items()}

    def state_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_state_dict(cls, sd: dict) -> "GlobalNorm":
        return cls(sd["mean"], sd["std"])


def fit_and_apply(feats: Dict[str, torch.Tensor], fit_files: Sequence[str],
                  dtype: torch.dtype = torch.float32):
    """Fit on ``fit_files`` (background only) and return ``(feats_norm, norm)``."""
    X = torch.cat([feats[f] for f in fit_files if feats[f].shape[0] > 0], dim=0).to(dtype)
    norm = GlobalNorm.fit(X)
    return norm.apply_all(feats), norm
