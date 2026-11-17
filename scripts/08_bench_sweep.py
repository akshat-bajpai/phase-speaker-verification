#!/usr/bin/env python
"""Sweep system configurations on the eval-mirroring bench (svcore/bench.py).

Usage:  python scripts/08_bench_sweep.py <sweep-name>

Each sweep is a named list of SystemSpecs; every result line is appended to
results/bench_sweeps.jsonl so nothing is lost if a long sweep is interrupted.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from svcore import bench as B, config as C, system2
from svcore.config import GMMConfig

A = system2.MEMBERS[0]          # IFD n_fft=1024 win=16 ms vad 25 dB
G = system2.GMM


def specs(name):
    if name == "mix":
        out = []
        for K in (64, 256, 512):
            for r in (8.0, 16.0, 32.0):
                out.append(B.SystemSpec(A, G.replace(n_components=K), r, f"A K={K} r={r:g}"))
        return out
    if name == "feat":
        out = []
        for lab, fc in [
            ("A n2048 w16", A.replace(n_fft=2048)),
            ("A n4096 w16", A.replace(n_fft=4096)),
            ("A bands48", A.replace(n_bands=48)),
            ("A bands24", A.replace(n_bands=24)),
            ("A w12", A.replace(win_ms=12.0)),
            ("A lag2", A.replace(ifd_lag=2)),
            ("A fmax4k", A.replace(f_max=4000.0)),
            ("A linear32", A.replace(band_type="linear")),
            ("A fmin20", A.replace(f_min=20.0)),
            ("A fmin20 fmax8k", A.replace(f_min=20.0, f_max=8000.0)),
            ("A fmin20 bands40", A.replace(f_min=20.0, n_bands=40)),
        ]:
            out.append(B.SystemSpec(fc, G, 16.0, lab))
        return out
    if name == "smooth":
        return [B.SystemSpec(A.replace(smooth=h), G, 16.0, f"A smooth{h}")
                for h in (1, 2, 4, 8)]
    if name == "rps":
        R = A.replace(stream="rps", win_ms=32.0, hop_ms=5.0, n_fft=2048,
                      n_bands=32, n_ceps=0)
        return [B.SystemSpec(fc, G.replace(n_components=K), 16.0, lab)
                for lab, fc, K in [
                    ("RPS w32 v0.5", R, 128),
                    ("RPS w32 v0.3", R.replace(rps_voicing=0.3), 128),
                    ("RPS w40 v0.3", R.replace(win_ms=40.0, rps_voicing=0.3), 128),
                    ("D2 w32 v0.3", R.replace(rps_mode="d2", rps_voicing=0.3), 128),
                    ("RPS w32 v0.3 K64", R.replace(rps_voicing=0.3), 64),
                ]]
    if name == "members":            # previous fusion members, for fusion tests
        return [B.SystemSpec(fc, G, 16.0, lab) for lab, fc in
                zip(system2.MEMBER_NAMES, system2.MEMBERS)]
    if name == "fusioncands":
        return [B.SystemSpec(fc, G, 16.0, lab) for lab, fc in [
            ("A magpow0.5", A.replace(mag_power=0.5)),
            ("A n2048 w16", A.replace(n_fft=2048)),
            ("FDPD n1024 w16", A.replace(stream="fdpd")),
            ("A lag2", A.replace(ifd_lag=2)),
        ]] + [B.SystemSpec(A, G.replace(n_components=K), r, f"A K={K} r={r:g}")
              for K, r in ((16, 16.0), (4, 16.0), (32, 16.0))]
    if name == "feat2":
        return [B.SystemSpec(fc, G, 16.0, lab) for lab, fc in [
            ("A fmin20", A.replace(f_min=20.0)),
            ("A linear32", A.replace(band_type="linear")),
            ("A w12", A.replace(win_ms=12.0)),
            ("A bands24", A.replace(n_bands=24)),
        ]]
    if name == "gd":
        F = A.replace(stream="fdpd")
        R = A.replace(stream="rps", win_ms=32.0, hop_ms=5.0, n_fft=2048, rps_voicing=0.3)
        return [B.SystemSpec(fc, G, 16.0, lab) for lab, fc in [
            ("FDPD n1024 w25", F.replace(win_ms=25.0)),
            ("FDPD n2048 w32", F.replace(n_fft=2048, win_ms=32.0)),
            ("GDCC exact n1024 w25", A.replace(stream="mgdcc", win_ms=25.0, n_bands=48,
                                                n_ceps=30, mgd_denom="exact")),
            ("FDPD n1024 w20 vad30", F.replace(win_ms=20.0, vad_rel_db=30.0)),
            ("RPS w32 v0.3", R),
        ]]
    if name == "var":
        return [B.SystemSpec(A, G, 16.0, f"A r=16 rvar={rv:g}", r_var=rv)
                for rv in (16.0, 64.0, 256.0)]
    if name == "weight":
        return [B.SystemSpec(fc, G, 16.0, lab) for lab, fc in [
            ("A magpow2", A.replace(mag_power=2.0)),
            ("A magpow0.5", A.replace(mag_power=0.5)),
            ("A magpow3", A.replace(mag_power=3.0)),
            ("A blackman", A.replace(window="blackman")),
            ("A blackman w20", A.replace(window="blackman", win_ms=20.0)),
        ]]
    if name == "subband":
        return [B.SystemSpec(A, G, 16.0, f"A {t}", transform=t)
                for t in ("bands:0-16", "bands:16-32", "bands:0-12", "bands:12-24",
                          "bands:20-32")]
    if name == "voiced":
        return [B.SystemSpec(A.replace(voicing_mask=v), G, 16.0, f"A voiced>{v:g}")
                for v in (0.2, 0.35)]
    if name == "lda":
        return [B.SystemSpec(A, G, 16.0, f"A {t}", transform=t)
                for t in ("lda_utt:0", "lda_utt:96", "lda_utt:64", "lda_frame:0",
                          "lda_frame:64")]
    raise SystemExit(f"unknown sweep {name}")


SCHEMES = [("raw", "raw", 0), ("tnorm", "tnorm", 0), ("asnorm100", "asnorm", 100),
           ("asnormE50", "asnormE", 50), ("asnormE100", "asnormE", 100)]


def mats_path(spec) -> str:
    """Saved bench matrices for one system (for offline fusion experiments)."""
    tag = (f"{spec.ftag}_K{spec.gmm.n_components}{spec.gmm.cov_type}"
           f"_r{spec.relevance:g}" + (f"_rv{spec.r_var:g}" if spec.r_var else ""))
    d = os.path.join(C.CACHE_DIR, "benchmats")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, tag + ".pt")


def main():
    torch.set_num_threads(8)
    name = sys.argv[1]
    b = B.Bench(verbose=False)
    out_path = os.path.join(C.RESULTS_DIR, "bench_sweeps.jsonl")
    for spec in specs(name):
        t0 = time.time()
        try:
            mats = b.mats(spec)
        except Exception as e:                       # keep sweeping
            print(f"{spec.label}: FAILED {e!r}", flush=True)
            continue
        torch.save({"label": spec.label, "mats": mats},
                   mats_path(spec))
        for sname, scheme, top in SCHEMES:
            r = b.evaluate(mats, lambda m: B.normalise(m, scheme, top))
            print(f"{spec.label:22s} {sname:11s} {B.fmt(r)}", flush=True)
            with open(out_path, "a") as fh:
                fh.write(json.dumps({"sweep": name, "label": spec.label,
                                     "norm": sname, **r}) + "\n")
        print(f"   [{time.time()-t0:.0f}s]", flush=True)
        b._mats.clear()


if __name__ == "__main__":
    main()
