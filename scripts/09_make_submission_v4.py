#!/usr/bin/env python
"""v4 Kaggle submission, built with exactly the code validated on the bench.

Per fusion member (svcore/system4.py):
  phase features -> background-fitted normalisation (+ optional LDA, also fitted
  on background only) -> UBM on the background UBM split -> mean-only MAP
  (relevance r) -> average per-frame LLR.
Scores are then adaptive-s-normalised (svcore/bench.normalise), averaged over
members, and optionally passed through the closed-set competitor penalty.

Information used for an evaluation trial (claimed speaker S, test file X):
  * the UBM / cohort / normalisation statistics -- background data only;
  * S's three designated enrolment recordings;
  * X itself;
  * the other *enrolled* evaluation models (designated enrolment data), which
    serve as extra cohort models on the utterance side of s-norm and in the
    competitor penalty.
No other evaluation test recording influences a trial's score, and nothing is
fitted on evaluation test recordings.

Usage:  python scripts/09_make_submission_v4.py [--tag v4]
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from svcore import bench as B, config as C, data, metrics as M, system4


def validate(path, trials):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    want = [t["trial_id"] for t in trials]
    ids = [r["trial_id"] for r in rows]
    assert [*rows[0]] == ["trial_id", "score"], f"bad header {[*rows[0]]}"
    assert len(ids) == len(want) and len(set(ids)) == len(ids), "row/id count"
    assert set(ids) == set(want), "trial_id set mismatch"
    for r in rows:
        assert math.isfinite(float(r["score"])), f"non-finite {r['trial_id']}"
    print(f"  [validated] {len(rows)} rows, unique ids, finite scores, header ok")


def split_mats(bench, spec, pool, split):
    """Mats for an official split (dev or eval) under one system."""
    feats = bench.features_for(spec.feature, pool, spec.transform)
    ubm = bench.ubm(spec.feature, spec.gmm, pool, feats, spec.ftag)
    fs = B.FastScorer(ubm)
    trials = data.trials(split)
    emap = data.enrol_map(split, 3)
    tests = sorted({t["test_file"] for t in trials})
    means, keys = B.map_means(ubm, feats, emap, spec.relevance)
    cmeans, _ = B.map_means(ubm, feats, pool.cohort, spec.relevance)
    coh_utts = [f for v in pool.cohort.values() for f in v]
    S = fs.matrix(means, feats, tests)
    Ct = fs.matrix(cmeans, feats, tests)
    Es = fs.matrix(means, feats, coh_utts)
    mi = {k: i for i, k in enumerate(keys)}
    fi = {f: j for j, f in enumerate(tests)}
    return trials, B.Mats(S, Ct, Es, None, mi, fi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v4")
    args = ap.parse_args()
    torch.set_num_threads(8)
    cfg = system4.SCORING
    print("=" * 78)
    print(f"SUBMISSION {args.tag}: {len(system4.MEMBERS)} member(s), {cfg}")
    print("=" * 78)
    bench = B.Bench(verbose=False, cohort_min=cfg["cohort_min"],
                    cohort_size=cfg["cohort_size"])
    pool = bench.pools[0]          # main UBM split + cohort (background only)
    print(f"UBM files {len(pool.ubm_files)}, cohort speakers {len(pool.cohort)}")

    fn = B.fused_fn(cfg["scheme"], cfg["top"], cfg.get("weights"),
                    cfg.get("lam", 0.0), cfg.get("delta", -1.0),
                    **cfg.get("kw", {}))
    out = {}
    for split in ("dev", "eval"):
        ms, trials = [], None
        for spec in system4.MEMBERS:
            t0 = time.time()
            trials, m = split_mats(bench, spec, pool, split)
            ms.append(m)
            if split == "dev":
                s1 = B.normalise(m, cfg["scheme"], cfg["top"], **cfg.get("kw", {}))
                rows = torch.tensor([m.mi[t["claimed_id"]] for t in trials])
                cols = torch.tensor([m.fi[t["test_file"]] for t in trials])
                e = M.eer(s1[rows, cols], [int(t["label"]) for t in trials])[0]
                print(f"  {spec.label:28s} dev EER {e*100:5.2f}%   [{time.time()-t0:.0f}s]",
                      flush=True)
        Sn = fn(ms)
        m0 = ms[0]
        rows = torch.tensor([m0.mi[t["claimed_id"]] for t in trials])
        cols = torch.tensor([m0.fi[t["test_file"]] for t in trials])
        out[split] = (trials, Sn[rows, cols].double())

    dev_tr, dev_s = out["dev"]
    y = [int(t["label"]) for t in dev_tr]
    summ = M.summarise(dev_s, y, n_boot=2000)
    print(f"\nFUSED dev EER = {summ['eer']*100:.2f}% "
          f"(95% CI {summ['eer_ci_lo']*100:.2f}-{summ['eer_ci_hi']*100:.2f}%), "
          f"AUC = {summ['auc']:.5f}")
    ev_tr, ev_s = out["eval"]
    print(f"eval scores: n={len(ev_s)} mean={float(ev_s.mean()):+.3f} "
          f"sd={float(ev_s.std()):.3f} finite={bool(torch.isfinite(ev_s).all())}")

    path = os.path.join(C.SUBMISSIONS_DIR, f"submission_{args.tag}.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["trial_id", "score"])
        for t, s in zip(ev_tr, ev_s.tolist()):
            w.writerow([t["trial_id"], f"{s:.6f}"])
    print(f"[saved] {path}")
    validate(path, ev_tr)
    dpath = os.path.join(C.RESULTS_DIR, f"dev_results_{args.tag}.csv")
    with open(dpath, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["claimed_id", "test_file", "label", "score"])
        for t, s in zip(dev_tr, dev_s.tolist()):
            w.writerow([t["claimed_id"], t["test_file"], t["label"], f"{s:.6f}"])
    print(f"[saved] {dpath}")
    C.save_json({"tag": args.tag, "scoring": cfg,
                 "members": [{"label": s.label, "feature": s.feature.to_dict(),
                              "gmm": s.gmm.to_dict(), "relevance": s.relevance,
                              "transform": s.transform} for s in system4.MEMBERS],
                 "dev_summary": summ},
                os.path.join(C.RESULTS_DIR, f"submission_{args.tag}.json"))


if __name__ == "__main__":
    main()
