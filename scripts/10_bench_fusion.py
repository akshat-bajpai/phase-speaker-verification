#!/usr/bin/env python
"""Evaluate equal-weight fusions of saved bench systems (cache/benchmats/*.pt).

Usage:  python scripts/10_bench_fusion.py <substr1> <substr2> ...
Each argument selects one saved system by a substring of its file name.
Every non-empty subset of size >= 2 is scored with adaptive s-norm (enrolled
cohort) with and without the closed-set competitor penalty.
"""
from __future__ import annotations

import glob
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from svcore import bench as B, config as C


def main():
    torch.set_num_threads(4)
    files = sorted(glob.glob(os.path.join(C.CACHE_DIR, "benchmats", "*.pt")))
    sel = []
    for a in sys.argv[1:]:
        hit = [f for f in files if a in os.path.basename(f)]
        if len(hit) != 1:
            raise SystemExit(f"'{a}' matches {len(hit)} files: {hit}")
        sel.append(hit[0])
    blobs = [torch.load(f, weights_only=False) for f in sel]
    names = [os.path.basename(f)[:-3] for f in sel]
    b = B.Bench(verbose=False)
    for n, bl in zip(names, blobs):
        r = b.evaluate(bl["mats"], lambda m: B.competitor_penalty(
            B.normalise(m, "asnormE", 50), 0.25, -1.0))
        print(f"single  {n[:70]:70s} {B.fmt(r)}", flush=True)
    for k in range(2, len(blobs) + 1):
        for combo in itertools.combinations(range(len(blobs)), k):
            ms = [blobs[i]["mats"] for i in combo]
            for lam in (0.0, 0.25):
                fn = B.fused_fn("asnormE", 50, lam=lam)
                r = B.evaluate_fusion(b, ms, fn)
                print(f"fuse{k} {'+'.join(str(i) for i in combo):8s} pen={lam:<4g} "
                      f"{B.fmt(r)}", flush=True)


if __name__ == "__main__":
    main()
