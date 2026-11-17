#!/usr/bin/env python
"""Task 1 -- phase-based feature extraction.

Builds the feature cache for every representation compared in the report and
prints the diagnostics that establish *compliance* as well as correctness:

1. feature dimension, VAD retention and frame counts;
2. a time-shift invariance test that quantifies how each representation
   survives the organisers' random 0-30 ms leading pad;
3. a magnitude-leakage control: the identical pipeline with the phase term
   deleted.  A representation is only a genuine phase feature if the control
   collapses to chance.

Usage:  python scripts/01_features.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from svcore import cache, config as C, data, features as F, plots, system
from svcore.config import FeatureConfig


def table(rows, headers):
    w = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
         for i, h in enumerate(headers)]
    line = "  ".join(h.ljust(w[i]) for i, h in enumerate(headers))
    print(line); print("-" * len(line))
    for r in rows:
        print("  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))


def shift_invariance(cfg: FeatureConfig, n_files: int = 60, seed: int = C.SEED):
    """Cosine similarity between features of a clip and of a shifted copy.

    Three shift regimes, because they probe different things:

    ``hop-aligned``  a whole number of hops.  Each analysis frame then covers
      *exactly the same samples* as before, only under a different frame index,
      so EVERY representation must score 1.000.  This is a check on the
      comparison harness (and on our frame-index alignment), not on the feature.
    ``sub-hop``      1..hop-1 samples.  The frame grid now cuts the signal at
      different points, which rotates bin ``k`` of each frame by about
      ``-2 pi k d / N``.  This isolates the *phase* consequence of a time shift
      and is the regime that separates the representations.
    ``arbitrary``    a random 0-30 ms shift, exactly as the organisers applied
      it: a whole number of hops plus a sub-hop remainder.
    """
    _, hop = cfg.frame_lengths()
    ids = data.all_file_ids()
    rnd = random.Random(seed)
    sel = rnd.sample(ids, n_files)
    out = {}
    for mode in ("hop-aligned", "sub-hop", "arbitrary"):
        sims = []
        for fid in sel:
            w = data.load_wav(fid)
            if mode == "hop-aligned":
                d = hop * rnd.randint(1, max(1, int(0.030 * cfg.sample_rate) // hop))
            elif mode == "sub-hop":
                d = rnd.randint(1, max(1, hop - 1))
            else:
                d = rnd.randint(1, int(0.030 * cfg.sample_rate))
            w2 = torch.cat([w[:d].flip(0), w])          # mirrored leading pad
            a, b = F.extract(w, cfg), F.extract(w2, cfg)
            n = min(a.shape[0], b.shape[0])
            if n < 5:
                continue
            off = b.shape[0] - a.shape[0]
            aa, bb = a[:n], b[max(0, off):max(0, off) + n]
            sims.append(float((aa * bb).sum() / (aa.norm() * bb.norm()).clamp_min(1e-9)))
        t = torch.tensor(sims)
        out[mode] = (float(t.mean()), float(t.median()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the invariance test")
    args = ap.parse_args()

    torch.set_num_threads(8)
    print("=" * 78)
    print("TASK 1 -- PHASE-BASED FEATURE EXTRACTION")
    print("=" * 78)

    cfg = system.FINAL_FEATURE
    win, hop = cfg.frame_lengths()
    print(f"\nFinal feature definition ({cfg.stream}):")
    print(f"  framing            {cfg.win_ms:g} ms window ({win} samples), "
          f"{cfg.hop_ms:g} ms hop ({hop} samples), {cfg.window} window, "
          f"n_fft={cfg.n_fft} ({cfg.n_fft//2+1} frequency bins)")
    print(f"  phase quantity     princarg(phi(t,k) - phi(t-{cfg.ifd_lag},k) "
          f"- 2*pi*k*{cfg.ifd_lag}*hop/n_fft)   [instantaneous-frequency deviation]")
    print(f"  band aggregation   magnitude-weighted circular mean over "
          f"{cfg.n_bands} {cfg.band_type} bands ({cfg.f_min:g}-{cfg.f_max:g} Hz)")
    print(f"  sub-streams        cos and sin of the band-wise mean resultant vector")
    print(f"  compression        {'none (raw band values)' if cfg.n_ceps == 0 else f'DCT-II, {cfg.n_ceps} coefficients'}")
    print(f"  deltas             order {cfg.deltas} (regression half-width {cfg.delta_width})")
    print(f"  normalisation      {cfg.cmvn}")
    print(f"  VAD                energy within {cfg.vad_rel_db:g} dB of the loudest "
          f"frame (min {cfg.vad_min_frames} frames)")
    print(f"  FEATURE DIMENSION  {F.feature_dim(cfg)}")

    # ---- build every cache ------------------------------------------------
    print("\n--- feature caches ---")
    rows = []
    stats = {}
    for name, fc in system.COMPARISON_STREAMS.items():
        t0 = time.time()
        feats = cache.build(fc, verbose=False)
        n = cache.frame_counts(feats, list(feats.keys()))
        dim = next(iter(feats.values())).shape[1]
        rows.append([name, fc.stream, dim, f"{float(n.mean()):.1f}",
                     int(n.min()), int(n.max()), int(n.sum()),
                     f"{time.time()-t0:.1f}s"])
        stats[name] = {"dim": dim, "frames_mean": float(n.mean()),
                       "frames_min": int(n.min()), "frames_total": int(n.sum())}
    table(rows, ["representation", "stream", "dim", "frames/utt", "min", "max",
                 "total frames", "time"])

    # ---- VAD retention ----------------------------------------------------
    print("\n--- VAD retention vs threshold (final framing) ---")
    ids = data.all_file_ids()
    rnd = random.Random(C.SEED)
    sel = rnd.sample(ids, 200)
    rows = []
    for db in (15, 20, 25, 30, 40, 60):
        c = cfg.replace(vad_rel_db=db)
        kept, total = [], []
        for fid in sel:
            _, _, logE = F.stft_pair(data.load_wav(fid), c, False)
            kept.append(int(F.vad_mask(logE, c).sum())); total.append(int(logE.numel()))
        k = torch.tensor(kept, dtype=torch.float32)
        rows.append([f"{db} dB", f"{float(k.mean()):.1f}", int(k.min()), int(k.max()),
                     f"{100*float(k.mean())/ (sum(total)/len(total)):.1f}%"])
    table(rows, ["threshold", "frames kept", "min", "max", "% of frames"])

    # ---- shift invariance -------------------------------------------------
    if not args.quick:
        print("\n--- time-shift invariance (the organisers' random leading pad) ---")
        print("cosine similarity between a clip's features and a shifted copy's;")
        print("1.0 = perfectly invariant, 0.0 = the representation is scrambled")
        rows = []
        for name, fc in system.COMPARISON_STREAMS.items():
            r = shift_invariance(fc)
            rows.append([name, f"{r['hop-aligned'][0]:+.3f}",
                         f"{r['sub-hop'][0]:+.3f}", f"{r['sub-hop'][1]:+.3f}",
                         f"{r['arbitrary'][0]:+.3f}", f"{r['arbitrary'][1]:+.3f}"])
            for k in ("hop-aligned", "sub-hop", "arbitrary"):
                stats[name][f"shift_sim_{k}"] = r[k][0]
        table(rows, ["representation", "hop-aligned (harness check)",
                     "sub-hop mean", "median", "arbitrary mean", "median"])
        plots.bars([n.split(" (")[0] for n in stats],
                   [stats[n]["shift_sim_sub-hop"] for n in stats],
                   "fig01_shift_invariance.png",
                   "cosine similarity (sub-hop shift)",
                   "Invariance to a sub-hop time shift", rotate=25)

    C.save_json({"final_feature": cfg.to_dict(),
                 "feature_dim": F.feature_dim(cfg),
                 "streams": {k: v.to_dict() for k, v in system.COMPARISON_STREAMS.items()},
                 "stats": stats},
                os.path.join(C.RESULTS_DIR, "task1_features.json"))
    print(f"\n[saved] {os.path.join(C.RESULTS_DIR, 'task1_features.json')}")


if __name__ == "__main__":
    main()
