#!/usr/bin/env python
"""Revised (v2) Kaggle submission: fused, T-normalised phase systems.

Pipeline, per fusion member:
  features -> UBM (background only) -> MAP-adapted speaker models
  -> average per-frame LLR -> T-norm against the background cohort
Then: standardise each member's score vector and average with equal weights.

v1 reached 0.61 % on dev but 3.70 % on the public leaderboard -- a selection-bias
gap, since ~10 hyper-parameters had been chosen against only 196 genuine dev
trials.  v2 was selected on the mean of three speaker-disjoint protocols
(dev, dev-LOO, background) totalling 579 genuine trials; see svcore/system2.py.

Usage:  python scripts/07_make_submission_v2.py [--tag v2] [--no-tnorm]
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

from svcore import (advanced as A, cache, config as C, data, metrics as M,
                    normalise as NZ, scoring as SC, system, system2)
from svcore.config import ScoreConfig
from svcore.gmm import GMM


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--no-tnorm", action="store_true")
    ap.add_argument("--asnorm", type=int, default=0,
                    help="use adaptive score normalisation with this top-N "
                         "(overrides T-norm)")
    args = ap.parse_args()
    torch.set_num_threads(8)
    norm = "none" if args.no_tnorm else system2.SCORE.norm
    if args.asnorm:
        norm = f"as-norm(top-{args.asnorm})"
    scfg = ScoreConfig(top_c=0, norm=norm, cohort_size=system2.SCORE.cohort_size)

    print("=" * 78)
    print(f"SUBMISSION v2 -- {len(system2.MEMBERS)}-way fusion, score norm = {norm}")
    print("=" * 78)

    ubm_files, held_files, cohort = system.splits()
    dev_tr, eval_tr = data.trials("dev"), data.trials("eval")
    dev_y = [t["label"] for t in dev_tr]
    dev_tests = sorted({t["test_file"] for t in dev_tr})
    eval_tests = sorted({t["test_file"] for t in eval_tr})
    dev_emap = data.enrol_map("dev", system2.ENROL.n_enrol)
    eval_emap = data.enrol_map("eval", system2.ENROL.n_enrol)
    print(f"\ndev : {len(dev_tr)} trials ({sum(dev_y)} genuine), {len(dev_emap)} targets")
    print(f"eval: {len(eval_tr)} trials, {len(eval_emap)} targets, "
          f"{len(eval_tests)} test recordings")

    dev_parts, eval_parts = [], []
    for name, fc in zip(system2.MEMBER_NAMES, system2.MEMBERS):
        t0 = time.time()
        feats = cache.build(fc, verbose=False)
        if fc.cmvn == "global":
            feats, _ = NZ.fit_and_apply(feats, ubm_files)
        ubm = system.train_ubm(feats, fc, system2.GMM)
        cohort_models = (SC.enrol_speakers(ubm, feats, cohort, system2.ENROL)
                         if norm != "none" else None)
        cohort_utts = [f for v in cohort.values() for f in v]

        def run(emap, trials, tests):
            models = SC.enrol_speakers(ubm, feats, emap, system2.ENROL)
            if args.asnorm:
                S, mi, fi = A.asnorm_matrix(ubm, models, cohort_models, feats,
                                            tests, cohort_utts, args.asnorm, True)
            elif norm == "none":
                S, mi, fi = SC.score_matrix(ubm, models, feats, tests, 0)
            else:
                S, mi, fi = SC.normalise_scores(ubm, models, cohort_models, feats,
                                                tests, held_files, norm, 0)
            return SC.gather_trial_scores(S, mi, fi, trials)

        ds = run(dev_emap, dev_tr, dev_tests)
        es = run(eval_emap, eval_tr, eval_tests)
        dev_parts.append(ds)
        eval_parts.append(es)
        e, _ = M.eer(ds, dev_y)
        print(f"  {name:26s} dim={next(iter(feats.values())).shape[1]:3d}  "
              f"K={ubm.K}  dev EER={e*100:5.2f}%   [{time.time()-t0:.0f}s]", flush=True)

    dev_f = A.fuse(dev_parts)
    eval_f = A.fuse(eval_parts)
    summ = M.summarise(dev_f, dev_y, n_boot=2000)
    print(f"\nFUSED dev EER = {summ['eer']*100:.2f}% "
          f"(95% CI {summ['eer_ci_lo']*100:.2f}-{summ['eer_ci_hi']*100:.2f}%), "
          f"AUC = {summ['auc']:.4f}")
    print(f"eval scores: mean={float(eval_f.mean()):+.4f} sd={float(eval_f.std()):.4f} "
          f"min={float(eval_f.min()):+.4f} max={float(eval_f.max()):+.4f} "
          f"finite={bool(torch.isfinite(eval_f).all())}")

    out = os.path.join(C.SUBMISSIONS_DIR, f"submission_{args.tag}.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["trial_id", "score"])
        for t, s in zip(eval_tr, eval_f.tolist()):
            w.writerow([t["trial_id"], f"{s:.6f}"])
    print(f"\n[saved] {out}")
    validate(out, eval_tr)

    dev_csv = os.path.join(C.RESULTS_DIR, f"dev_results_{args.tag}.csv")
    with open(dev_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["claimed_id", "test_file", "label", "score"])
        for t, s in zip(dev_tr, dev_f.tolist()):
            w.writerow([t["claimed_id"], t["test_file"], t["label"], f"{s:.6f}"])
    print(f"[saved] {dev_csv}")
    C.save_json({"tag": args.tag, "norm": norm,
                 "members": [m.to_dict() for m in system2.MEMBERS],
                 "member_names": system2.MEMBER_NAMES,
                 "dev_summary": summ},
                os.path.join(C.RESULTS_DIR, f"submission_{args.tag}.json"))


if __name__ == "__main__":
    main()
