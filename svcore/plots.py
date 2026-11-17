"""Figure helpers.

matplotlib is used *solely* to visualise results, as the rules permit.  Every
function takes torch tensors / Python lists and converts to plain lists before
plotting, so no array library participates in any computation.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from . import config as C
from . import metrics as M

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 140, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
    "axes.spines.right": False, "figure.autolayout": True,
})

PALETTE = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2",
           "#be185d", "#4d7c0f"]


def _l(x) -> List[float]:
    if isinstance(x, torch.Tensor):
        return [float(v) for v in x.flatten()]
    return [float(v) for v in x]


def save(fig, name: str) -> str:
    path = os.path.join(C.FIGURES_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {path}")
    return path


# ---------------------------------------------------------------------------


def det_curve(score_sets: Dict[str, tuple], name: str, title: str = "DET curve"):
    """``score_sets``: label -> (scores, labels).  DET axes = probit-warped."""
    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    ticks = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6]
    for i, (lab, (s, y)) in enumerate(score_sets.items()):
        _, far, frr = M.roc_points(s, y)
        e, _ = M.eer(s, y)
        xs = _l(M.det_axis(far)); ys = _l(M.det_axis(frr))
        ax.plot(xs, ys, color=PALETTE[i % len(PALETTE)], lw=1.6,
                label=f"{lab} (EER {e*100:.2f}%)")
    lo, hi = M.det_axis(torch.tensor([ticks[0], ticks[-1]]))
    ax.plot([float(lo), float(hi)], [float(lo), float(hi)], ls=":", c="0.5", lw=1,
            label="EER line")
    tl = _l(M.det_axis(torch.tensor(ticks)))
    ax.set_xticks(tl); ax.set_xticklabels([f"{t*100:g}" for t in ticks])
    ax.set_yticks(tl); ax.set_yticklabels([f"{t*100:g}" for t in ticks])
    ax.set_xlim(float(lo), float(hi)); ax.set_ylim(float(lo), float(hi))
    ax.set_xlabel("False accept rate (%)"); ax.set_ylabel("False reject rate (%)")
    ax.set_title(title); ax.legend(fontsize=7.5, loc="upper right")
    return save(fig, name)


def roc_curve(score_sets: Dict[str, tuple], name: str, title: str = "ROC curve"):
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    for i, (lab, (s, y)) in enumerate(score_sets.items()):
        _, far, frr = M.roc_points(s, y)
        ax.plot(_l(far), _l(1.0 - frr), color=PALETTE[i % len(PALETTE)], lw=1.6,
                label=f"{lab} (AUC {M.auc(s,y):.3f})")
    ax.plot([0, 1], [0, 1], ls=":", c="0.5", lw=1)
    ax.set_xlabel("False accept rate"); ax.set_ylabel("True accept rate (1 - FRR)")
    ax.set_title(title); ax.legend(fontsize=7.5, loc="lower right")
    return save(fig, name)


def far_frr_vs_threshold(scores, labels, name: str, eer_thr: Optional[float] = None,
                         extra: Optional[Dict[str, float]] = None):
    thr, far, frr = M.roc_points(scores, labels)
    keep = torch.isfinite(thr)
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.plot(_l(thr[keep]), _l(far[keep]), color=PALETTE[1], lw=1.6, label="FAR")
    ax.plot(_l(thr[keep]), _l(frr[keep]), color=PALETTE[0], lw=1.6, label="FRR")
    if eer_thr is not None:
        ax.axvline(eer_thr, ls="--", c="0.4", lw=1, label=f"EER thr {eer_thr:.3f}")
    for i, (lab, t) in enumerate((extra or {}).items()):
        ax.axvline(t, ls=":", c=PALETTE[(i + 2) % len(PALETTE)], lw=1.2, label=lab)
    ax.set_xlabel("threshold on the average per-frame LLR")
    ax.set_ylabel("error rate"); ax.set_title("FAR / FRR vs decision threshold")
    ax.legend(fontsize=7.5)
    return save(fig, name)


def score_hist(scores, labels, name: str, title: str = "Score distributions",
               thr: Optional[float] = None):
    s = torch.as_tensor(scores, dtype=torch.float64)
    y = torch.as_tensor(labels)
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    bins = 45
    ax.hist(_l(s[y == 0]), bins=bins, alpha=0.65, color=PALETTE[1], density=True,
            label=f"impostor (n={int((y==0).sum())})")
    ax.hist(_l(s[y == 1]), bins=bins, alpha=0.65, color=PALETTE[0], density=True,
            label=f"genuine (n={int((y==1).sum())})")
    if thr is not None:
        ax.axvline(thr, ls="--", c="0.3", lw=1.2, label="EER threshold")
    ax.set_xlabel("average per-frame log-likelihood ratio"); ax.set_ylabel("density")
    ax.set_title(title); ax.legend(fontsize=7.5)
    return save(fig, name)


def lines(xs, series: Dict[str, Sequence[float]], name: str, xlabel: str,
          ylabel: str, title: str, logx: bool = False, marker: str = "o",
          yerr: Optional[Dict[str, Sequence[float]]] = None, figsize=(5.0, 3.5)):
    fig, ax = plt.subplots(figsize=figsize)
    for i, (lab, ys) in enumerate(series.items()):
        c = PALETTE[i % len(PALETTE)]
        if yerr and lab in yerr:
            lo = [y - e[0] for y, e in zip(_l(ys), yerr[lab])]
            hi = [e[1] - y for y, e in zip(_l(ys), yerr[lab])]
            ax.errorbar(_l(xs), _l(ys), yerr=[lo, hi], color=c, lw=1.5,
                        marker=marker, ms=4, capsize=2.5, label=lab)
        else:
            ax.plot(_l(xs), _l(ys), color=c, lw=1.5, marker=marker, ms=4, label=lab)
    if logx:
        ax.set_xscale("log", base=2)
        ax.set_xticks(_l(xs)); ax.set_xticklabels([f"{int(v)}" for v in _l(xs)])
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    if len(series) > 1:
        ax.legend(fontsize=7.5)
    return save(fig, name)


def hist(values, name: str, xlabel: str, title: str, bins: int = 20,
         vlines: Optional[Dict[str, float]] = None):
    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    ax.hist(_l(values), bins=bins, color=PALETTE[0], alpha=0.8, edgecolor="white")
    for i, (lab, v) in enumerate((vlines or {}).items()):
        ax.axvline(v, ls="--", lw=1.3, color=PALETTE[(i + 1) % len(PALETTE)], label=lab)
    ax.set_xlabel(xlabel); ax.set_ylabel("count"); ax.set_title(title)
    if vlines:
        ax.legend(fontsize=7.5)
    return save(fig, name)


def convergence(histories: Dict[str, Sequence[float]], name: str,
                title: str = "EM convergence", ylabel: str = "mean log-likelihood",
                alpha: float = 1.0):
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    for i, (lab, h) in enumerate(histories.items()):
        ax.plot(range(1, len(h) + 1), _l(h), lw=1.2, alpha=alpha,
                color=PALETTE[i % len(PALETTE)], label=lab if alpha == 1.0 else None)
    ax.set_xlabel("EM iteration"); ax.set_ylabel(ylabel); ax.set_title(title)
    if alpha == 1.0 and len(histories) <= 10:
        ax.legend(fontsize=7.5)
    return save(fig, name)


def convergence_many(histories: Sequence[Sequence[float]], name: str,
                     title: str, labels: Optional[Sequence[str]] = None,
                     colors: Optional[Sequence[str]] = None):
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    seen = set()
    for i, h in enumerate(histories):
        lab = labels[i] if labels else None
        c = colors[i] if colors else PALETTE[0]
        show = lab if lab not in seen else None
        seen.add(lab)
        ax.plot(range(1, len(h) + 1), _l(h), lw=1.0, alpha=0.7, color=c, label=show)
    ax.set_xlabel("EM iteration"); ax.set_ylabel("mean log-likelihood")
    ax.set_title(title)
    if labels:
        ax.legend(fontsize=7.5)
    return save(fig, name)


def scatter2d(Z: torch.Tensor, name: str, title: str, colour=None,
              centres: Optional[torch.Tensor] = None, cbar_label: str = ""):
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    if colour is None:
        ax.scatter(_l(Z[:, 0]), _l(Z[:, 1]), s=3, alpha=0.3, color=PALETTE[0],
                   linewidths=0)
    else:
        sc = ax.scatter(_l(Z[:, 0]), _l(Z[:, 1]), s=3, alpha=0.45, c=_l(colour),
                        cmap="viridis", linewidths=0)
        fig.colorbar(sc, ax=ax, label=cbar_label, shrink=0.85)
    if centres is not None:
        ax.scatter(_l(centres[:, 0]), _l(centres[:, 1]), s=26, marker="x",
                   color=PALETTE[1], linewidths=1.1, label="component means")
        ax.legend(fontsize=7.5)
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2"); ax.set_title(title)
    return save(fig, name)


def bars(labels: Sequence[str], values: Sequence[float], name: str, ylabel: str,
         title: str, yerr=None, figsize=(5.4, 3.4), rotate: int = 20,
         annotate: bool = True):
    fig, ax = plt.subplots(figsize=figsize)
    xs = list(range(len(labels)))
    err = None
    if yerr is not None:
        err = [[v - e[0] for v, e in zip(_l(values), yerr)],
               [e[1] - v for v, e in zip(_l(values), yerr)]]
    ax.bar(xs, _l(values), color=[PALETTE[i % len(PALETTE)] for i in xs],
           alpha=0.85, yerr=err, capsize=3)
    ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=rotate, ha="right" if rotate else "center")
    ax.set_ylabel(ylabel); ax.set_title(title)
    if annotate:
        for x, v in zip(xs, _l(values)):
            ax.text(x, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    return save(fig, name)
