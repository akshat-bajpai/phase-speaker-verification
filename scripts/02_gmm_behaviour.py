#!/usr/bin/env python
"""Task 2 -- understanding GMM behaviour on the phase features.

Experiments (all in PyTorch):
  A. covariance structure x mixture count: parameter count, BIC/AIC on a fixed
     fitting set and average log-likelihood on a speaker-disjoint held-out
     background split;
  B. >= 20 random initialisations, one fit per seed -> distribution of final
     log-likelihoods;
  C. random vs k-means(++) initialisation;
  D. the value of selecting the best of n initialisations;
  E. a deliberately degenerate component, its symptoms, and three remedies;
  F. a 2-D PCA projection (computed in torch) for visualisation only.

Usage:  python scripts/02_gmm_behaviour.py [--n-fit 20000] [--seeds 25]
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from svcore import cache, config as C, data, plots, system
from svcore.config import GMMConfig
from svcore.gmm import GMM, kmeans


def table(rows, headers):
    w = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
         for i, h in enumerate(headers)]
    line = "  ".join(h.ljust(w[i]) for i, h in enumerate(headers))
    print(line); print("-" * len(line))
    for r in rows:
        print("  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))


def torch_pca(X: torch.Tensor, k: int = 2):
    """PCA by SVD of the centred data.  Returns ``(Z, mean, components)``.

    Used for VISUALISATION ONLY -- the GMMs are always fitted on the full
    feature vectors, never on this projection.
    """
    mu = X.mean(dim=0, keepdim=True)
    Xc = X - mu
    # economy SVD; V columns are the principal directions
    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    comps = Vh[:k]
    return Xc @ comps.t(), mu, comps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-fit", type=int, default=20000,
                    help="frames in the fitting set (uniform subsample, seed 2026)")
    ap.add_argument("--seeds", type=int, default=25, help="initialisations per arm")
    ap.add_argument("--skip-full", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(8)

    print("=" * 78)
    print("TASK 2 -- GMM BEHAVIOUR ON PHASE FEATURES")
    print("=" * 78)

    feats, _ = system.prepare_features(verbose=True)
    ubm_files, held_files, _ = system.splits()
    X = cache.pooled(feats, ubm_files, max_frames=args.n_fit, seed=C.SEED)
    Xh = cache.pooled(feats, held_files)
    D = X.shape[1]
    print(f"\nFitting set   : {X.shape[0]} frames x {D} dims "
          f"(uniform subsample of the {len(ubm_files)} background UBM-split "
          f"recordings, seed {C.SEED})")
    print(f"Held-out set  : {Xh.shape[0]} frames from {len(held_files)} recordings "
          f"of speakers DISJOINT from the fitting set (seed {C.SEED})")

    results = {"n_fit": X.shape[0], "n_heldout": Xh.shape[0], "dim": D}

    # ------------------------------------------------------------------ A
    print("\n" + "-" * 78)
    print("A. COVARIANCE STRUCTURE x MIXTURE COUNT")
    print("-" * 78)
    rows, covA = [], {}
    Ks = [8, 16, 32, 64, 128]
    for cov in ("spherical", "diag", "tied", "full"):
        ks = ([8, 16, 32] if cov == "full" else Ks)
        if cov == "full" and args.skip_full:
            continue
        for K in ks:
            g = GMMConfig(n_components=K, cov_type=cov, init="kmeans",
                          max_iter=60, tol=1e-5)
            m = GMM(K, D, cov).fit(X, g)
            tr = m.mean_log_likelihood(X)
            ho = m.mean_log_likelihood(Xh)
            rows.append([cov, K, m.n_parameters, f"{tr:+.3f}", f"{ho:+.3f}",
                         f"{m.bic(X):.3e}", f"{m.aic(X):.3e}", m.n_iter_run,
                         f"{m.fit_seconds:.1f}s"])
            covA.setdefault(cov, []).append(
                {"K": K, "params": m.n_parameters, "train_ll": tr, "heldout_ll": ho,
                 "bic": m.bic(X), "aic": m.aic(X), "iters": m.n_iter_run,
                 "seconds": m.fit_seconds})
            print(f"  {cov:9s} K={K:3d}  params={m.n_parameters:7d}  "
                  f"train LL={tr:+8.3f}  held-out LL={ho:+8.3f}  "
                  f"BIC={m.bic(X):.4e}  {m.fit_seconds:5.1f}s", flush=True)
    print()
    table(rows, ["cov", "K", "params", "train LL", "held-out LL", "BIC", "AIC",
                 "iters", "time"])
    results["covariance_study"] = covA

    plots.lines(Ks, {c: [r["bic"] for r in covA[c]] + [None] * (len(Ks) - len(covA[c]))
                     for c in covA if len(covA[c]) == len(Ks)},
                "fig02a_bic_vs_k.png", "mixture components K", "BIC (lower is better)",
                f"BIC on the fitting set (N={X.shape[0]}, D={D})", logx=True)
    plots.lines(Ks, {c: [r["heldout_ll"] for r in covA[c]]
                     for c in covA if len(covA[c]) == len(Ks)},
                "fig02b_heldout_vs_k.png", "mixture components K",
                "mean held-out log-likelihood",
                "Held-out likelihood on disjoint background speakers", logx=True)
    allcov = sorted(covA)
    plots.bars([f"{c}\nK={r['K']}" for c in allcov for r in covA[c] if r["K"] == 32],
               [r["params"] for c in allcov for r in covA[c] if r["K"] == 32],
               "fig02c_params_k32.png", "free parameters",
               f"Parameter count at K=32 (D={D})", rotate=0)

    # ------------------------------------------------------------------ B/C
    print("\n" + "-" * 78)
    print(f"B/C. INITIALISATION: {args.seeds} SEEDS, RANDOM vs K-MEANS (K=64, diag)")
    print("-" * 78)
    K_INIT = 64
    arms, hist_curves, curve_colours, curve_labels = {}, [], [], []
    for init in ("random", "kmeans"):
        lls, hos, times, iters = [], [], [], []
        for s in range(args.seeds):
            g = GMMConfig(n_components=K_INIT, cov_type="diag", init=init,
                          max_iter=100, tol=1e-5, seed=C.SEED + s)
            m = GMM(K_INIT, D, "diag").fit(X, g, seed=C.SEED + s)
            lls.append(m.history[-1]); hos.append(m.mean_log_likelihood(Xh))
            times.append(m.fit_seconds); iters.append(m.n_iter_run)
            hist_curves.append(m.history)
            curve_colours.append(plots.PALETTE[0] if init == "random" else plots.PALETTE[2])
            curve_labels.append(init)
        t = torch.tensor(lls)
        arms[init] = {"final_ll": lls, "heldout_ll": hos, "seconds": times,
                      "iters": iters}
        print(f"  {init:7s}: final train LL  mean={t.mean():+.4f}  sd={t.std():.4f}  "
              f"min={t.min():+.4f}  max={t.max():+.4f}  spread={float(t.max()-t.min()):.4f}")
        th = torch.tensor(hos)
        print(f"           held-out  LL  mean={th.mean():+.4f}  sd={th.std():.4f}  "
              f"best={th.max():+.4f}")
        print(f"           iterations    mean={sum(iters)/len(iters):.1f}   "
              f"time mean={sum(times)/len(times):.1f}s")
    results["init_study"] = arms

    plots.hist(arms["random"]["final_ll"], "fig02d_init_ll_random.png",
               "final mean log-likelihood",
               f"{args.seeds} random initialisations (K={K_INIT}, diag)", bins=15,
               vlines={"best": max(arms["random"]["final_ll"]),
                       "k-means mean": sum(arms["kmeans"]["final_ll"]) / args.seeds})
    fig_sets = {"random init": arms["random"]["final_ll"],
                "k-means++ init": arms["kmeans"]["final_ll"]}
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for i, (lab, vals) in enumerate(fig_sets.items()):
        ax.hist([float(v) for v in vals], bins=14, alpha=0.65,
                color=plots.PALETTE[i * 2], label=lab, edgecolor="white")
    ax.set_xlabel("final mean log-likelihood"); ax.set_ylabel("count")
    ax.set_title(f"Initialisation vs EM optimum ({args.seeds} seeds each)")
    ax.legend(fontsize=7.5)
    plots.save(fig, "fig02e_init_compare.png")
    plots.convergence_many(hist_curves, "fig02f_convergence.png",
                           f"EM convergence, {args.seeds} seeds per initialiser",
                           labels=curve_labels, colors=curve_colours)

    # ------------------------------------------------------------------ D
    print("\n" + "-" * 78)
    print("D. VALUE OF SELECTING THE BEST OF n INITIALISATIONS")
    print("-" * 78)
    g = torch.Generator().manual_seed(C.SEED)
    rows, curves = [], {}
    for init in ("random", "kmeans"):
        lls = torch.tensor(arms[init]["final_ll"], dtype=torch.float64)
        hos = torch.tensor(arms[init]["heldout_ll"], dtype=torch.float64)
        exp_best, exp_ho = [], []
        for n in range(1, args.seeds + 1):
            draws = torch.randint(len(lls), (4000, n), generator=g)
            picked = lls[draws].argmax(dim=1)
            chosen = draws.gather(1, picked.unsqueeze(1)).squeeze(1)
            exp_best.append(float(lls[chosen].mean()))
            exp_ho.append(float(hos[chosen].mean()))
        curves[f"{init}: train LL"] = exp_best
        curves[f"{init}: held-out LL"] = exp_ho
        rows.append([init, f"{exp_best[0]:+.4f}", f"{exp_best[4]:+.4f}",
                     f"{exp_best[-1]:+.4f}", f"{exp_ho[0]:+.4f}",
                     f"{exp_ho[4]:+.4f}", f"{exp_ho[-1]:+.4f}"])
    table(rows, ["init", "train n=1", "n=5", f"n={args.seeds}",
                 "held n=1", "n=5", f"n={args.seeds}"])
    plots.lines(list(range(1, args.seeds + 1)), curves, "fig02g_best_of_n.png",
                "number of initialisations n", "expected log-likelihood of the pick",
                "Selecting the best of n initialisations", marker="")
    results["best_of_n"] = curves

    # ------------------------------------------------------------------ E
    print("\n" + "-" * 78)
    print("E. A DELIBERATELY DEGENERATE COMPONENT, AND REMEDIES")
    print("-" * 78)
    N_SMALL, K_BIG = 400, 64
    Xs = X[:N_SMALL]
    n_par = K_BIG * D * 2 + K_BIG - 1
    print(f"  Over-parameterised fit: K={K_BIG} components, N={N_SMALL} frames, "
          f"D={D} dims")
    print(f"  -> {n_par} free parameters estimated from {N_SMALL*D} numbers "
          f"({n_par/(N_SMALL*D):.2f} parameters per observation),")
    print(f"     i.e. about {N_SMALL/K_BIG:.1f} frames per component for a "
          f"{D}-dimensional diagonal Gaussian.\n")

    def diagnose(m, Xf):
        """Symptoms of a degenerate mixture."""
        occ = m.occupancy(Xf)
        floored = int((m.covars <= m.covars.min() * 1.0000001).any(dim=1).sum())
        return {"min_weight": float(m.weights.min()),
                "min_occupancy": float(occ.min()),
                "n_occ_lt1": int((occ < 1.0).sum()),
                "n_floored": floored,
                "min_variance": m.min_variance(),
                "condition_number": m.condition_number(),
                "train_ll": m.mean_log_likelihood(Xf),
                "heldout_ll": m.mean_log_likelihood(Xh)}

    # --- E.1 the symptom: likelihood diverges as the variance floor is removed
    print("  E.1  The symptom: with no variance floor, components collapse onto")
    print("       individual frames and the TRAINING likelihood diverges while the")
    print("       held-out likelihood collapses.\n")
    floors = [1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2]
    rows, floor_curve = [], {"train": [], "held-out": []}
    for vf in floors:
        g2 = GMMConfig(n_components=K_BIG, cov_type="diag", init="kmeans",
                       max_iter=60, tol=1e-6, var_floor=vf)
        m = GMM(K_BIG, D, "diag").fit(Xs, g2, reinit_degenerate=False)
        d = diagnose(m, Xs)
        rows.append([f"{vf:.0e}", f"{d['min_variance']:.2e}",
                     f"{d['condition_number']:.2e}", d["n_occ_lt1"], d["n_floored"],
                     f"{d['train_ll']:+.1f}", f"{d['heldout_ll']:+.1f}",
                     f"{d['train_ll']-d['heldout_ll']:+.1f}"])
        floor_curve["train"].append(d["train_ll"])
        floor_curve["held-out"].append(d["heldout_ll"])
        print(f"    floor={vf:.0e}  min var={d['min_variance']:.2e}  "
              f"cond={d['condition_number']:.2e}  components with n_k<1: "
              f"{d['n_occ_lt1']:2d}/{K_BIG}  at the floor: {d['n_floored']:2d}  "
              f"train LL={d['train_ll']:+9.1f}  held-out LL={d['heldout_ll']:+8.1f}",
              flush=True)
    print()
    table(rows, ["variance floor", "min variance", "cond number", "n_k<1",
                 "at floor", "train LL", "held-out LL", "gap"])
    results["degeneracy_floor_sweep"] = {"floors": floors, **floor_curve}
    plots.lines([math.log10(f) for f in floors],
                {"training set (N=400)": floor_curve["train"],
                 "held-out background": floor_curve["held-out"]},
                "fig02h_degeneracy_floor.png", "log10(variance floor)",
                "mean log-likelihood per frame",
                f"Removing the variance floor (K={K_BIG}, N={N_SMALL})")

    # --- E.2 the remedies
    print("\n  E.2  Three remedies, each applied to the same degenerate setting")
    print("       (K=64 on 400 frames).  'reinit' re-seeds any component whose")
    print("       occupancy falls below 2% of the frames at a random data point.\n")
    arms_e = [
        ("degenerate: no floor, no reinit",
         dict(n_components=K_BIG, var_floor=1e-14, min_weight=0.0), False),
        ("remedy 1: variance floor 1e-3",
         dict(n_components=K_BIG, var_floor=1e-3, min_weight=0.0), False),
        ("remedy 2: reinit low-occupancy",
         dict(n_components=K_BIG, var_floor=1e-14, min_weight=0.02), True),
        ("remedy 3: fewer components (K=8)",
         dict(n_components=8, var_floor=1e-14, min_weight=0.0), False),
        ("remedies 1+3 (K=8 + floor)",
         dict(n_components=8, var_floor=1e-3, min_weight=0.0), False),
    ]
    rows, deg = [], {}
    for name, kw, reinit in arms_e:
        K = kw.pop("n_components")
        g2 = GMMConfig(cov_type="diag", init="kmeans", max_iter=60, tol=1e-6,
                       n_components=K, **kw)
        m = GMM(K, D, "diag").fit(Xs, g2, reinit_degenerate=reinit)
        d = diagnose(m, Xs)
        rows.append([name, K, f"{d['min_weight']:.2e}", f"{d['min_occupancy']:.2f}",
                     d["n_occ_lt1"], f"{d['min_variance']:.2e}",
                     f"{d['condition_number']:.2e}", f"{d['train_ll']:+.1f}",
                     f"{d['heldout_ll']:+.1f}"])
        d["K"] = K
        d["weights_sorted"] = sorted(float(v) for v in m.weights)
        deg[name] = d
        print(f"  {name:34s} min w={d['min_weight']:.2e}  min n_k={d['min_occupancy']:5.2f}  "
              f"n_k<1: {d['n_occ_lt1']:2d}  min var={d['min_variance']:.2e}  "
              f"cond={d['condition_number']:.2e}  train LL={d['train_ll']:+9.1f}  "
              f"held-out LL={d['heldout_ll']:+8.1f}", flush=True)
    print()
    table(rows, ["configuration", "K", "min weight", "min n_k", "n_k<1",
                 "min variance", "cond number", "train LL", "held-out LL"])
    results["degeneracy"] = deg

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for i, (name, d) in enumerate(deg.items()):
        wv = d["weights_sorted"]
        ax.semilogy([j / (len(wv) - 1) for j in range(len(wv))], wv, marker="o",
                    ms=2.5, lw=1.2, color=plots.PALETTE[i], label=name)
    ax.axhline(1.0 / N_SMALL, ls=":", c="0.4", lw=1,
               label=f"one frame (1/{N_SMALL})")
    ax.set_xlabel("component rank (normalised)")
    ax.set_ylabel("mixture weight (log scale)")
    ax.set_title(f"Component weights under degeneracy and its remedies")
    ax.legend(fontsize=6.5)
    plots.save(fig, "fig02i_degenerate_weights.png")

    # ------------------------------------------------------------------ F
    print("\n" + "-" * 78)
    print("F. 2-D PCA PROJECTION (VISUALISATION ONLY)")
    print("-" * 78)
    Z, mu, comps = torch_pca(X[:6000], 2)
    ev = (Z.var(dim=0, unbiased=False) / X[:6000].var(dim=0, unbiased=False).sum())
    print(f"  variance explained by PC1/PC2: {float(ev[0])*100:.1f}% / "
          f"{float(ev[1])*100:.1f}%  (the GMMs are fitted on all {D} dims)")
    g3 = GMMConfig(n_components=32, cov_type="diag", init="kmeans", max_iter=60)
    m32 = GMM(32, D, "diag").fit(X, g3)
    means_proj = (m32.means - mu) @ comps.t()
    resp_top = m32.log_weighted(X[:6000]).argmax(dim=1).to(torch.float64)
    plots.scatter2d(Z, "fig02j_pca_components.png",
                    f"Phase features in 2-D PCA (K=32 means overlaid)",
                    colour=resp_top, centres=means_proj,
                    cbar_label="most likely component")
    results["pca_variance_explained"] = [float(ev[0]), float(ev[1])]

    C.save_json(results, os.path.join(C.RESULTS_DIR, "task2_gmm_behaviour.json"))
    print(f"\n[saved] {os.path.join(C.RESULTS_DIR, 'task2_gmm_behaviour.json')}")


if __name__ == "__main__":
    main()
