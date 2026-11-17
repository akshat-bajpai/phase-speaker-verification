#!/usr/bin/env python
"""Greedy forward selection of equal-weight fusion members on the bench.

Starts from the best single system and repeatedly adds the saved system
(cache/benchmats) that most lowers the bench score (mean of pooled dev-rotation
and background-CV EER), under adaptive s-norm + competitor penalty.  Equal
weights: the only thing selected is membership.
"""
from __future__ import annotations

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from svcore import bench as B, config as C


def main(max_members: int = 8, lam: float = 0.25):
    torch.set_num_threads(4)
    files = sorted(glob.glob(os.path.join(C.CACHE_DIR, "benchmats", "*.pt")))
    names = [os.path.basename(f)[:-3] for f in files]
    mats = [torch.load(f, weights_only=False)["mats"] for f in files]
    b = B.Bench(verbose=False)
    fn = B.fused_fn("asnormE", 50, lam=lam)
    single = {i: B.evaluate_fusion(b, [mats[i]], fn) for i in range(len(mats))}
    for i in sorted(single, key=lambda i: single[i]["score"]):
        print(f"single {single[i]['score']*100:5.2f}%  {names[i]}")
    chosen = [min(single, key=lambda i: single[i]["score"])]
    best = single[chosen[0]]
    print(f"\nstart: {names[chosen[0]]}  {B.fmt(best)}", flush=True)
    while len(chosen) < max_members:
        trial = {}
        for i in range(len(mats)):
            if i in chosen:
                continue
            trial[i] = B.evaluate_fusion(b, [mats[j] for j in chosen + [i]], fn)
        i = min(trial, key=lambda k: trial[k]["score"])
        print(f"+ {names[i][:60]:60s} {B.fmt(trial[i])}", flush=True)
        if trial[i]["score"] >= best["score"] - 1e-9:
            print("no further improvement; stopping")
            break
        chosen.append(i)
        best = trial[i]
    print("\nSELECTED:")
    for i in chosen:
        print("  ", names[i])
    print(B.fmt(best))


if __name__ == "__main__":
    main()
