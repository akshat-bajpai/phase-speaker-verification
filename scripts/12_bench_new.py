#!/usr/bin/env python
"""Round-5 candidates on the bench, alone and as additions to the v4 fusion.

Candidates
  * temporal-segment (text-dependent) systems: the VAD-retained frames of every
    utterance are split into n equal consecutive segments, one UBM + MAP per
    segment, segment LLRs averaged (svcore/bench: transform "seg:i/n");
  * TFD -- mixed time-frequency phase derivative (features._stream_tfd);
  * IFD variants: 2 ms hop, delta-deltas, tighter VAD; FDPD with a 12 ms window.

Usage:  python scripts/12_bench_new.py [group ...]      (groups: seg, feat)
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from svcore import bench as B, config as C, system2, system4

A = system2.MEMBERS[0]
G = system2.GMM
CFG = system4.SCORING


def mpath(spec) -> str:
    tag = (f"{spec.ftag}_K{spec.gmm.n_components}{spec.gmm.cov_type}"
           f"_r{spec.relevance:g}" + (f"_rv{spec.r_var:g}" if spec.r_var else ""))
    d = os.path.join(C.CACHE_DIR, "benchmats")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, tag + ".pt")


def get(b, spec):
    p = mpath(spec)
    if os.path.exists(p):
        return torch.load(p, weights_only=False)["mats"]
    t0 = time.time()
    m = b.mats(spec)
    torch.save({"label": spec.label, "mats": m}, p)
    b._mats.clear()
    print(f"   computed {spec.label} [{time.time()-t0:.0f}s]", flush=True)
    return m


def seg_specs(fc, n, K, label, gm=G):
    return [B.SystemSpec(fc, gm.replace(n_components=K), 16.0, f"{label} seg{i}/{n}",
                         transform=f"seg:{i}/{n}") for i in range(n)]


def candidates(groups):
    out = []
    if "seg" in groups:
        out += [
            ("A seg2 K128", seg_specs(A, 2, 128, "A")),
            ("A seg3 K64", seg_specs(A, 3, 64, "A")),
            ("A seg3 K128", seg_specs(A, 3, 128, "A")),
        ]
    if "feat" in groups:
        out += [
            ("TFD n1024 w16", [B.SystemSpec(A.replace(stream="tfd"), G, 16.0, "TFD n1024 w16")]),
            ("A hop2", [B.SystemSpec(A.replace(hop_ms=2.0), G, 16.0, "A hop2")]),
            ("A dd", [B.SystemSpec(A.replace(deltas=2), G, 16.0, "A dd")]),
            ("FDPD n1024 w12", [B.SystemSpec(A.replace(stream="fdpd", win_ms=12.0), G, 16.0,
                                             "FDPD n1024 w12")]),
            ("A vad20", [B.SystemSpec(A.replace(vad_rel_db=20.0), G, 16.0, "A vad20")]),
        ]
    if "segall" in groups:
        for spec in system4.MEMBERS:
            out.append((f"{spec.label} seg2", seg_specs(spec.feature, 2, 128, spec.label)))
    return out


def main():
    torch.set_num_threads(8)
    groups = sys.argv[1:] or ["seg", "feat"]
    b = B.Bench(verbose=False)
    base = [get(b, s) for s in system4.MEMBERS]
    lam, dl = CFG["lam"], CFG["delta"]
    v4 = B.fused_fn("asnormE", 100, CFG["weights"], lam, dl)
    print(f"v4 reference        {B.fmt(B.evaluate_fusion(b, base, v4))}", flush=True)
    single = lambda m: B.competitor_penalty(B.normalise(m, "asnormE", 100), lam, dl)
    print(f"A alone             {B.fmt(b.evaluate(base[0], single))}", flush=True)
    for label, specs in candidates(groups):
        ms = [get(b, s) for s in specs]
        mats = ms[0] if len(ms) == 1 else B.average_mats(ms)
        torch.save({"label": label, "mats": mats},
                   os.path.join(C.CACHE_DIR, "benchmats",
                                "combo_" + label.replace(" ", "_") + ".pt"))
        print(f"{label:20s}{B.fmt(b.evaluate(mats, single))}", flush=True)
        for w in (0.1, 0.2, 0.3, 0.5):
            wts = [x * (1 - w) for x in CFG["weights"]] + [w]
            r = B.evaluate_fusion(b, base + [mats], B.fused_fn("asnormE", 100, wts, lam, dl))
            print(f"  v4 + {label:14s} w={w:<4g} {B.fmt(r)}", flush=True)


if __name__ == "__main__":
    main()
