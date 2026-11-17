"""Verification metrics: EER, FAR/FRR, ROC/DET, min-DCF, bootstrap CIs.

Implemented with torch tensors only (plotting scripts convert to Python lists).
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import torch


def _as_tensor(x) -> torch.Tensor:
    return x.detach().to(torch.float64) if isinstance(x, torch.Tensor) \
        else torch.tensor(list(x), dtype=torch.float64)


def roc_points(scores, labels) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(thresholds, far, frr)`` for the rule *accept iff score > theta*.

    Thresholds are the sorted unique scores plus -inf (accept all); the arrays
    are monotone: ``far`` decreasing, ``frr`` increasing.
    """
    s, y = _as_tensor(scores), _as_tensor(labels)
    n_gen = float((y == 1).sum())
    n_imp = float((y == 0).sum())
    if n_gen == 0 or n_imp == 0:
        raise ValueError("need both genuine and impostor trials")
    thr = torch.unique(s, sorted=True)
    thr = torch.cat([torch.tensor([-float("inf")], dtype=torch.float64), thr])
    # counts of scores > theta via searchsorted on sorted score vectors
    gs = torch.sort(s[y == 1]).values
    is_ = torch.sort(s[y == 0]).values
    # number of elements <= theta
    n_gen_le = torch.searchsorted(gs, thr, right=True).to(torch.float64)
    n_imp_le = torch.searchsorted(is_, thr, right=True).to(torch.float64)
    frr = n_gen_le / n_gen                      # genuine rejected (score <= thr)
    far = (n_imp - n_imp_le) / n_imp            # impostors accepted (score > thr)
    return thr, far, frr


def eer(scores, labels) -> Tuple[float, float]:
    """Equal error rate with linear interpolation.  Returns ``(eer, threshold)``."""
    thr, far, frr = roc_points(scores, labels)
    diff = frr - far
    idx = int(torch.nonzero(diff >= 0)[0]) if bool((diff >= 0).any()) else len(thr) - 1
    if idx == 0:
        return float(max(far[0], frr[0])), float(thr[0])
    f0, r0, t0 = float(far[idx - 1]), float(frr[idx - 1]), float(thr[idx - 1])
    f1, r1, t1 = float(far[idx]), float(frr[idx]), float(thr[idx])
    d0, d1 = r0 - f0, r1 - f1
    if d1 == d0:
        return 0.5 * (f1 + r1), t1
    w = -d0 / (d1 - d0)
    e = (f0 + w * (f1 - f0) + r0 + w * (r1 - r0)) / 2.0
    t = t0 + w * (t1 - t0) if math.isfinite(t0) else t1
    return float(e), float(t)


def far_frr_at(scores, labels, threshold: float) -> Tuple[float, float]:
    s, y = _as_tensor(scores), _as_tensor(labels)
    acc = s > threshold
    far = float((acc & (y == 0)).sum()) / float((y == 0).sum())
    frr = float((~acc & (y == 1)).sum()) / float((y == 1).sum())
    return far, frr


def threshold_at_far(scores, labels, target_far: float) -> float:
    """Smallest threshold whose FAR <= ``target_far``."""
    thr, far, _ = roc_points(scores, labels)
    ok = torch.nonzero(far <= target_far)
    return float(thr[int(ok[0])]) if len(ok) else float(thr[-1])


def min_dcf(scores, labels, p_target: float = 0.01, c_miss: float = 1.0,
            c_fa: float = 1.0) -> Tuple[float, float]:
    """Minimum detection cost (NIST-style), normalised by the trivial system."""
    thr, far, frr = roc_points(scores, labels)
    dcf = c_miss * p_target * frr + c_fa * (1 - p_target) * far
    norm = min(c_miss * p_target, c_fa * (1 - p_target))
    i = int(dcf.argmin())
    return float(dcf[i] / norm), float(thr[i])


def auc(scores, labels) -> float:
    """Area under the ROC curve (= P(score_gen > score_imp), ties counted half)."""
    s, y = _as_tensor(scores), _as_tensor(labels)
    order = torch.argsort(s)
    ranks = torch.empty_like(s)
    ranks[order] = torch.arange(1, len(s) + 1, dtype=torch.float64)
    # average ranks over ties
    uniq, inv, counts = torch.unique(s, return_inverse=True, return_counts=True)
    sums = torch.zeros(len(uniq), dtype=torch.float64).index_add_(0, inv, ranks)
    ranks = (sums / counts)[inv]
    n1 = float((y == 1).sum()); n0 = float((y == 0).sum())
    r1 = float(ranks[y == 1].sum())
    return (r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def eer_bootstrap_ci(scores, labels, n_boot: int = 1000, alpha: float = 0.05,
                     seed: int = 2026) -> Tuple[float, float, float]:
    """Percentile bootstrap CI for the EER (resampling trials with replacement)."""
    s, y = _as_tensor(scores), _as_tensor(labels)
    n = len(s)
    g = torch.Generator().manual_seed(seed)
    vals: List[float] = []
    for _ in range(n_boot):
        idx = torch.randint(n, (n,), generator=g)
        ys, ss = y[idx], s[idx]
        if int((ys == 1).sum()) < 2 or int((ys == 0).sum()) < 2:
            continue
        vals.append(eer(ss, ys)[0])
    v = torch.tensor(vals, dtype=torch.float64).sort().values
    lo = float(v[int(alpha / 2 * len(v))])
    hi = float(v[min(len(v) - 1, int((1 - alpha / 2) * len(v)))])
    return float(v.mean()), lo, hi


def det_axis(p: torch.Tensor) -> torch.Tensor:
    """Normal-deviate warp used by DET plots (probit of the error rate)."""
    p = p.clamp(1e-6, 1 - 1e-6)
    return torch.special.ndtri(p)


def summarise(scores, labels, n_boot: int = 1000) -> Dict[str, float]:
    e, t = eer(scores, labels)
    dcf, t_dcf = min_dcf(scores, labels)
    _, lo, hi = eer_bootstrap_ci(scores, labels, n_boot=n_boot)
    y = _as_tensor(labels)
    return {"eer": e, "eer_threshold": t, "eer_ci_lo": lo, "eer_ci_hi": hi,
            "auc": auc(scores, labels), "min_dcf_p01": dcf, "dcf_threshold": t_dcf,
            "n_genuine": int((y == 1).sum()), "n_impostor": int((y == 0).sum())}
