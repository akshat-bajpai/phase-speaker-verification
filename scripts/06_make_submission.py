#!/usr/bin/env python
"""Produce the Kaggle submission for the evaluation trials.

Runs the final system end to end: background-fitted feature normalisation, the
UBM trained on background speakers, MAP-adapted evaluation speaker models from
the three designated evaluation enrolment recordings, average per-frame LLR
scoring, and (optionally) T-norm against the background cohort.

Evaluation recordings are used ONLY as enrolment/test material -- never to fit
the normalisation statistics, the UBM or anything else.

Usage:  python scripts/06_make_submission.py [--norm tnorm] [--tag v1]
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from svcore import config as C, data, metrics as M, scoring as SC, system
from svcore.config import ScoreConfig


def validate(path: str, trials) -> None:
    """Re-read the file and check every rule on the competition's Evaluation page."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [*rows[0]] == ["trial_id", "score"], f"bad header: {[*rows[0]]}"
    ids = [r["trial_id"] for r in rows]
    want = [t["trial_id"] for t in trials]
    assert len(ids) == len(want), f"{len(ids)} rows for {len(want)} trials"
    assert len(set(ids)) == len(ids), "duplicate trial_id"
    assert set(ids) == set(want), "trial_id set does not match eval_trials.csv"
    for r in rows:
        v = float(r["score"])
        assert math.isfinite(v), f"non-finite score for {r['trial_id']}"
    print(f"  [validated] {len(rows)} rows, unique ids, all scores finite, "
          f"header correct")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--norm", default=None, choices=["none", "tnorm", "znorm", "ztnorm"])
    ap.add_argument("--tag", default="final")
    args = ap.parse_args()
    torch.set_num_threads(8)

    print("=" * 78)
    print("FINAL SYSTEM -> KAGGLE SUBMISSION")
    print("=" * 78)

    norm = args.norm or system.FINAL_SCORE.norm
    scfg = ScoreConfig(top_c=system.FINAL_SCORE.top_c, norm=norm,
                       cohort_size=system.FINAL_SCORE.cohort_size)

    feats, _ = system.prepare_features(verbose=True)
    ubm = system.train_ubm(feats, verbose=False)
    cohort_models, impostor_files = system.build_cohort(feats, ubm)
    print(f"\nFeature  : {system.FINAL_FEATURE.tag}")
    print(f"UBM      : K={ubm.K} {ubm.cov_type}, D={ubm.D}, "
          f"{ubm.n_iter_run} EM iterations, {ubm.fit_seconds:.1f}s")
    print(f"Enrolment: MAP mean adaptation, r={system.FINAL_ENROL.relevance_factor:g}, "
          f"{system.FINAL_ENROL.n_enrol} utterances")
    print(f"Scoring  : average per-frame LLR, "
          f"top_c={'all' if not scfg.top_c else scfg.top_c}, norm={scfg.norm}")

    # ---- development check -------------------------------------------------
    dev_tr, dev_raw, dev_sc = system.score_split("dev", feats, ubm,
                                                 system.FINAL_ENROL, scfg,
                                                 cohort_models, impostor_files)
    y = [t["label"] for t in dev_tr]
    summ = M.summarise(dev_sc, y, n_boot=2000)
    print(f"\nDEV  EER = {summ['eer']*100:.2f}% "
          f"(95% CI {summ['eer_ci_lo']*100:.2f}-{summ['eer_ci_hi']*100:.2f}%), "
          f"AUC = {summ['auc']:.4f}, "
          f"{summ['n_genuine']} genuine / {summ['n_impostor']} impostor trials")

    # ---- evaluation --------------------------------------------------------
    eval_tr = data.trials("eval")
    emap = data.enrol_map("eval", system.FINAL_ENROL.n_enrol)
    test_files = sorted({t["test_file"] for t in eval_tr})
    print(f"\nEVAL protocol: {len(emap)} enrolled speakers, {len(eval_tr)} trials, "
          f"{len(test_files)} test recordings")

    models = SC.enrol_speakers(ubm, feats, emap, system.FINAL_ENROL)
    if scfg.norm == "none":
        S, mi, fi = SC.score_matrix(ubm, models, feats, test_files, scfg.top_c)
    else:
        S, mi, fi = SC.normalise_scores(ubm, models, cohort_models, feats,
                                        test_files, impostor_files, scfg.norm,
                                        scfg.top_c)
    scores = SC.gather_trial_scores(S, mi, fi, eval_tr)
    print(f"Eval scores: mean={float(scores.mean()):+.4f}, "
          f"sd={float(scores.std()):.4f}, "
          f"min={float(scores.min()):+.4f}, max={float(scores.max()):+.4f}, "
          f"finite={bool(torch.isfinite(scores).all())}")

    out = os.path.join(C.SUBMISSIONS_DIR, f"submission_{args.tag}.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["trial_id", "score"])
        for t, s in zip(eval_tr, scores.tolist()):
            w.writerow([t["trial_id"], f"{s:.6f}"])
    print(f"\n[saved] {out}")
    validate(out, eval_tr)

    C.save_json({"tag": args.tag, "norm": scfg.norm,
                 "feature": system.FINAL_FEATURE.to_dict(),
                 "gmm": system.FINAL_GMM.to_dict(),
                 "enrol": {"relevance_factor": system.FINAL_ENROL.relevance_factor,
                           "n_enrol": system.FINAL_ENROL.n_enrol},
                 "dev_summary": summ,
                 "eval_score_stats": {"mean": float(scores.mean()),
                                      "sd": float(scores.std()),
                                      "min": float(scores.min()),
                                      "max": float(scores.max())}},
                os.path.join(C.RESULTS_DIR, f"submission_{args.tag}.json"))
    print(f"[saved] {os.path.join(C.RESULTS_DIR, f'submission_{args.tag}.json')}")


if __name__ == "__main__":
    main()
