"""Speaker-discriminative linear feature transforms, fitted on background only.

The brief allows "any learned feature transformation ... fitted using only
permitted background-training data".  The transform is a fixed linear map of
the phase features, so the GMM input remains a (linear) function of phase.

``lda_utt``   LDA whose within-class scatter is computed from *utterance-mean*
              feature vectors of background speakers with >= 2 recordings,
              and whose between-class scatter is computed from speaker means.
              Every speaker says the same word, so within-utterance (phonetic)
              variation is common to all speakers; what separates speakers is
              the session-stable offset of each band's phase deviation.  The
              utterance-mean scatter isolates exactly that.
``lda_frame`` the textbook frame-level LDA (within-class scatter includes all
              within-utterance phonetic variation), for comparison.

The returned map is ``y = (x - m) @ W`` with ``W`` (D, d).  Within-speaker
covariance of the projected utterance means is the identity, so the projected
dimensions are also ordered and scaled by discriminability.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch


def _scatter_utt(feats: Dict[str, torch.Tensor], spk2files: Dict[str, List[str]],
                 min_files: int = 2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    means, labels = [], []
    for i, (s, files) in enumerate(sorted(spk2files.items())):
        u = [feats[f].double().mean(0) for f in files if feats[f].shape[0] > 0]
        if len(u) < min_files:
            continue
        for v in u:
            means.append(v)
            labels.append(i)
    U = torch.stack(means)
    lab = torch.tensor(labels)
    mu = U.mean(0)
    uniq = lab.unique()
    spm = torch.stack([U[lab == c].mean(0) for c in uniq])
    cnt = torch.tensor([(lab == c).sum() for c in uniq], dtype=torch.float64)
    Wc = torch.cat([U[lab == c] - spm[i] for i, c in enumerate(uniq)])
    Sw = Wc.t() @ Wc / max(1, Wc.shape[0] - len(uniq))
    Bc = (spm - mu) * cnt.sqrt().unsqueeze(1)
    Sb = Bc.t() @ Bc / cnt.sum()
    return Sw, Sb, mu


def _scatter_frame(feats, spk2files, min_files: int = 2):
    D = next(iter(feats.values())).shape[1]
    Sw = torch.zeros(D, D, dtype=torch.float64)
    spm, cnt = [], []
    n = 0
    for s, files in sorted(spk2files.items()):
        X = [feats[f].double() for f in files if feats[f].shape[0] > 0]
        if len(X) < min_files:
            continue
        X = torch.cat(X)
        m = X.mean(0)
        Xc = X - m
        Sw += Xc.t() @ Xc
        n += X.shape[0]
        spm.append(m)
        cnt.append(X.shape[0])
    spm = torch.stack(spm)
    cnt = torch.tensor(cnt, dtype=torch.float64)
    mu = (spm * cnt.unsqueeze(1)).sum(0) / cnt.sum()
    Bc = (spm - mu) * cnt.sqrt().unsqueeze(1)
    return Sw / n, Bc.t() @ Bc / cnt.sum(), mu


def fit_lda(feats: Dict[str, torch.Tensor], spk2files: Dict[str, List[str]],
            kind: str = "utt", dim: int = 0, shrink: float = 0.05,
            min_files: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(m, W)`` such that ``y = (x - m) @ W``.

    ``shrink`` regularises the within-class scatter towards its scaled
    identity, ``Sw <- (1-a) Sw + a tr(Sw)/D I``, which keeps the solution stable
    when there are only a few hundred speakers.
    """
    if kind == "utt":
        Sw, Sb, mu = _scatter_utt(feats, spk2files, min_files)
    elif kind == "frame":
        Sw, Sb, mu = _scatter_frame(feats, spk2files, min_files)
    else:
        raise ValueError(kind)
    D = Sw.shape[0]
    Sw = (1 - shrink) * Sw + shrink * torch.trace(Sw) / D * torch.eye(D, dtype=Sw.dtype)
    # whiten Sw, then diagonalise the whitened Sb
    ew, Vw = torch.linalg.eigh(Sw)
    Wh = Vw @ torch.diag(ew.clamp_min(1e-10).rsqrt())
    eb, Vb = torch.linalg.eigh(Wh.t() @ Sb @ Wh)
    order = torch.argsort(eb, descending=True)
    W = Wh @ Vb[:, order]
    if dim and dim < D:
        W = W[:, :dim]
    return mu.float(), W.float()


def apply(feats: Dict[str, torch.Tensor], m: torch.Tensor, W: torch.Tensor
          ) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in feats.items():
        out[k] = (v - m) @ W if v.shape[0] else v.new_zeros(0, W.shape[1])
    return out
