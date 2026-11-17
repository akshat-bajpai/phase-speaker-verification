#!/usr/bin/env python
"""Task 3 -- train the GMM-UBM on the permitted background speakers.

Sweeps the mixture count (16...256) and the covariance structure, reporting
convergence behaviour, training time, held-out average log-likelihood on
speaker-disjoint background data, and the resulting development EER.

Usage:  python scripts/03_train_ubm.py [--ks 16,32,64,128,256] [--max-frames 0]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from svcore import cache, config as C, data, metrics as M, plots, system
from svcore.config import EnrolConfig, GMMConfig, ScoreConfig
from svcore.gmm import GMM


def table(rows, headers):
    w = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
         for i, h in enumerate(headers)]
    line = "  ".join(h.ljust(w[i]) for i, h in enumerate(headers))
    print(line); print("-" * len(line))
    for r in rows:
        print("  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="16,32,64,128,256")
    ap.add_argument("--covs", default="diag")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="0 = use every background frame")
    args = ap.parse_args()
    torch.set_num_threads(8)
    Ks = [int(k) for k in args.ks.split(",")]
    covs = args.covs.split(",")

    print("=" * 78)
    print("TASK 3 -- UBM TRAINING")
    print("=" * 78)

    feats, norm = system.prepare_features(verbose=True)
    ubm_files, held_files, _ = system.splits()
    X = cache.pooled(feats, ubm_files, max_frames=args.max_frames, seed=C.SEED)
    Xh = cache.pooled(feats, held_files)
    D = X.shape[1]
    bkg = data.background()
    n_spk = len({r["speaker_id"] for r in bkg})
    print(f"\nBackground speakers        : {n_spk} ({len(bkg)} recordings)")
    print(f"UBM-training split         : {len(ubm_files)} recordings, "
          f"{X.shape[0]} frames x {D} dims")
    print(f"Held-out split (disjoint)  : {len(held_files)} recordings, "
          f"{Xh.shape[0]} frames   [seed {C.SEED}]")
    print(f"Frame subsampling          : "
          f"{'none' if not args.max_frames else f'{args.max_frames} frames'}")
    print(f"EM settings                : init={system.FINAL_GMM.init}, "
          f"max_iter={system.FINAL_GMM.max_iter}, "
          f"rel. tol={system.FINAL_GMM.tol:g}, "
          f"variance floor={system.FINAL_GMM.var_floor:g}, "
          f"dtype={system.FINAL_GMM.dtype}")

    dev_tr = data.trials("dev")
    y = [t["label"] for t in dev_tr]
    rows, out, curves = [], [], {}
    for cov in covs:
        for K in Ks:
            gcfg = system.FINAL_GMM.replace(n_components=K, cov_type=cov)
            ubm = system.train_ubm(feats, system.FINAL_FEATURE, gcfg)
            tr = ubm.mean_log_likelihood(X)
            ho = ubm.mean_log_likelihood(Xh)
            e, sc, _ = system.dev_eer(feats, ubm, system.FINAL_ENROL,
                                      ScoreConfig(norm="none"))
            _, lo, hi = M.eer_bootstrap_ci(sc, y, n_boot=500)
            occ = ubm.occupancy(X)
            rows.append([cov, K, ubm.n_parameters, f"{tr:+.3f}", f"{ho:+.3f}",
                         f"{tr-ho:+.3f}", ubm.n_iter_run, ubm.converged,
                         f"{ubm.fit_seconds:.1f}s", f"{float(occ.min()):.1f}",
                         f"{e*100:.2f}%", f"[{lo*100:.2f}-{hi*100:.2f}]"])
            out.append({"cov": cov, "K": K, "params": ubm.n_parameters,
                        "train_ll": tr, "heldout_ll": ho, "iters": ubm.n_iter_run,
                        "converged": ubm.converged, "seconds": ubm.fit_seconds,
                        "min_occupancy": float(occ.min()), "dev_eer": e,
                        "dev_eer_ci": [lo, hi], "history": ubm.history})
            curves[f"{cov} K={K}"] = ubm.history
            print(f"  {cov:5s} K={K:4d}  params={ubm.n_parameters:7d}  "
                  f"train LL={tr:+8.3f}  held-out LL={ho:+8.3f}  "
                  f"iters={ubm.n_iter_run:3d}  {ubm.fit_seconds:6.1f}s  "
                  f"dev EER={e*100:5.2f}% [{lo*100:.2f}-{hi*100:.2f}]", flush=True)

    print()
    table(rows, ["cov", "K", "params", "train LL", "held-out LL", "gap", "iters",
                 "conv", "time", "min n_k", "dev EER", "95% CI"])

    for cov in covs:
        sub = [r for r in out if r["cov"] == cov]
        if len(sub) < 2:
            continue
        plots.lines([r["K"] for r in sub],
                    {"train": [r["train_ll"] for r in sub],
                     "held-out (disjoint speakers)": [r["heldout_ll"] for r in sub]},
                    f"fig03a_ll_vs_k_{cov}.png", "mixture components K",
                    "mean log-likelihood per frame",
                    f"UBM fit vs mixture count ({cov} covariance)", logx=True)
        plots.lines([r["K"] for r in sub],
                    {"dev EER (%)": [r["dev_eer"] * 100 for r in sub]},
                    f"fig03b_eer_vs_k_{cov}.png", "mixture components K",
                    "development EER (%)",
                    f"Verification performance vs mixture count ({cov})",
                    logx=True,
                    yerr={"dev EER (%)": [[r["dev_eer_ci"][0] * 100,
                                           r["dev_eer_ci"][1] * 100] for r in sub]})
        plots.lines([r["K"] for r in sub],
                    {"training time (s)": [r["seconds"] for r in sub]},
                    f"fig03c_time_vs_k_{cov}.png", "mixture components K",
                    "EM training time (s)", f"Training cost vs K ({cov})", logx=True)
    plots.convergence(curves, "fig03d_convergence.png",
                      "UBM EM convergence (mean log-likelihood per iteration)")

    C.save_json({"feature": system.FINAL_FEATURE.to_dict(),
                 "gmm": system.FINAL_GMM.to_dict(),
                 "n_background_speakers": n_spk,
                 "n_ubm_recordings": len(ubm_files),
                 "n_ubm_frames": X.shape[0],
                 "n_heldout_frames": Xh.shape[0],
                 "sweep": out},
                os.path.join(C.RESULTS_DIR, "task3_ubm.json"))
    print(f"\n[saved] {os.path.join(C.RESULTS_DIR, 'task3_ubm.json')}")
    print(f"[saved] UBM checkpoints in {C.CACHE_DIR}/ubm_*.pt")


if __name__ == "__main__":
    main()
