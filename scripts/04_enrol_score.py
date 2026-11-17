#!/usr/bin/env python
"""Tasks 4 & 5 -- MAP adaptation, enrolment, and verification scoring.

Task 4: relevance-factor study, the limiting behaviour of ``alpha_k``, the
effect of zero-occupancy components, mean-only vs weight/covariance adaptation,
and MAP vs training a speaker GMM from scratch.

Task 5: average per-frame log-likelihood-ratio scoring, DET/ROC curves, EER with
a bootstrap confidence interval, an operating point chosen for a stated use
case, and the development results CSV.

Usage:  python scripts/04_enrol_score.py
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from svcore import cache, config as C, data, metrics as M, plots, scoring as SC, system
from svcore.config import EnrolConfig, GMMConfig, ScoreConfig
from svcore.gmm import GMM
from svcore.map_adapt import map_adapt, sufficient_statistics, train_from_scratch


def table(rows, headers):
    w = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
         for i, h in enumerate(headers)]
    line = "  ".join(h.ljust(w[i]) for i, h in enumerate(headers))
    print(line); print("-" * len(line))
    for r in rows:
        print("  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-boot", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(8)
    n_boot = 0 if args.no_boot else 2000

    print("=" * 78)
    print("TASKS 4 & 5 -- MAP ADAPTATION, ENROLMENT AND SCORING")
    print("=" * 78)

    feats, _ = system.prepare_features(verbose=True)
    ubm = system.train_ubm(feats)
    ubm_files, held_files, cohort = system.splits()
    dev_tr = data.trials("dev")
    y = [t["label"] for t in dev_tr]
    emap = data.enrol_map("dev", 3)
    test_files = sorted({t["test_file"] for t in dev_tr})
    results = {}

    print(f"\nUBM: K={ubm.K}, {ubm.cov_type} covariance, D={ubm.D}")
    print(f"Dev protocol: {len(emap)} enrolled speakers, {len(dev_tr)} trials "
          f"({sum(y)} genuine, {len(y)-sum(y)} impostor), "
          f"{len(test_files)} distinct test recordings")

    # ------------------------------------------------------------------ 4a
    print("\n" + "-" * 78)
    print("4a. ENROLMENT STATISTICS AND COMPONENT OCCUPANCY")
    print("-" * 78)
    rows = []
    for n_enrol in (1, 2, 3):
        em = data.enrol_map("dev", n_enrol)
        nz, tot, occs = [], [], []
        for spk, files in em.items():
            Xe = torch.cat([feats[f] for f in files], dim=0)
            n_k, _, _ = sufficient_statistics(ubm, Xe)
            nz.append(float((n_k > 1e-3).sum()))
            tot.append(Xe.shape[0])
            occs.append(float(n_k.max()))
        t_nz = torch.tensor(nz); t_tot = torch.tensor(tot, dtype=torch.float32)
        rows.append([n_enrol, f"{float(t_tot.mean()):.1f}",
                     f"{float(t_nz.mean()):.1f}/{ubm.K}",
                     f"{100*float(t_nz.mean())/ubm.K:.1f}%",
                     f"{float(torch.tensor(occs).mean()):.1f}"])
    table(rows, ["utterances", "frames pooled", "components with n_k>0",
                 "occupied", "max n_k"])
    print("\n  Components with zero enrolment occupancy keep their UBM mean exactly,")
    print("  so a speaker model always remains a proper density over the whole")
    print("  feature space -- this is what makes 3 x 1 s of enrolment usable.")
    results["occupancy"] = rows

    # ------------------------------------------------------------------ 4b
    print("\n" + "-" * 78)
    print("4b. RELEVANCE FACTOR r")
    print("-" * 78)
    rs = [0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256]
    rows, rsum = [], []
    for r in rs:
        ecfg = EnrolConfig(relevance_factor=float(r), n_enrol=3)
        models = SC.enrol_speakers(ubm, feats, emap, ecfg)
        S, mi, fi = SC.score_matrix(ubm, models, feats, test_files)
        s = SC.gather_trial_scores(S, mi, fi, dev_tr)
        e, thr = M.eer(s, y)
        # how far did the means actually move?
        drift = torch.stack([(models[k].means - ubm.means).norm() for k in models]).mean()
        _, lo, hi = M.eer_bootstrap_ci(s, y, n_boot=400)
        rows.append([r, f"{e*100:.2f}%", f"[{lo*100:.2f}-{hi*100:.2f}]",
                     f"{float(drift):.3f}", f"{M.auc(s,y):.4f}"])
        rsum.append({"r": r, "eer": e, "ci": [lo, hi], "mean_drift": float(drift)})
        print(f"  r={r:6.1f}  dev EER={e*100:5.2f}%  [{lo*100:4.2f}-{hi*100:5.2f}]  "
              f"mean ||mu_s - mu_ubm||={float(drift):6.3f}", flush=True)
    table(rows, ["r", "dev EER", "95% CI", "mean mean-drift", "AUC"])
    plots.lines([r["r"] for r in rsum],
                {"dev EER (%)": [r["eer"] * 100 for r in rsum]},
                "fig04a_eer_vs_r.png", "relevance factor r", "development EER (%)",
                "MAP relevance factor vs EER", logx=True,
                yerr={"dev EER (%)": [[c["ci"][0] * 100, c["ci"][1] * 100]
                                      for c in rsum]})
    results["relevance_factor"] = rsum

    print("\n  Limiting behaviour of alpha_k = n_k / (n_k + r):")
    Xe = torch.cat([feats[f] for f in emap["dev_spk_001"]], dim=0)
    n_k, F_k, _ = sufficient_statistics(ubm, Xe)
    ml = F_k / n_k.clamp_min(1e-10).unsqueeze(1)
    # "well-occupied" means n_k >= 1 frame.  Components with n_k of order 1e-12
    # have a numerically meaningless ML mean AND alpha ~ n_k/r ~ 0, so they stay
    # at the UBM mean for any practical r -- they must not be mixed into this
    # comparison.
    occ = n_k >= 1.0
    print(f"    (K={ubm.K}; {int(occ.sum())} components have n_k >= 1 frame, "
          f"{int((n_k <= 0).sum())} have n_k == 0 exactly)")
    for r in (1e-8, 1.0, 16.0, 1e8):
        m = map_adapt(ubm, Xe, EnrolConfig(), relevance_factor=r)
        d_ubm = float((m.means - ubm.means).abs().max())
        d_ml = float((m.means[occ] - ml[occ]).abs().max())
        d_ml_rel = d_ml / float(ml[occ].abs().max())
        print(f"    r={r:>9.0e}: max|mu-mu_UBM|={d_ubm:9.3e}   "
              f"max|mu-mu_ML| over n_k>=1 ={d_ml:9.3e} "
              f"({d_ml_rel*100:6.3f}% of the ML scale)")
    print("    r -> 0   : alpha -> 1, the mean becomes the enrolment ML estimate")
    print("               (maximally specific, maximally over-fitted)")
    print("    r -> inf : alpha -> 0, the speaker model collapses onto the UBM")
    print("               (the LLR then tends to 0 for every trial)")

    # ------------------------------------------------------------------ 4c
    print("\n" + "-" * 78)
    print("4c. WHAT TO ADAPT, AND MAP vs TRAINING FROM SCRATCH")
    print("-" * 78)
    rows, adapt_res = [], {}
    variants = [
        ("mean only (required baseline)", EnrolConfig(relevance_factor=16.0)),
        ("mean + weights", EnrolConfig(relevance_factor=16.0, adapt_weights=True)),
        ("mean + covariances", EnrolConfig(relevance_factor=16.0, adapt_covars=True)),
        ("mean + weights + covariances",
         EnrolConfig(relevance_factor=16.0, adapt_weights=True, adapt_covars=True)),
    ]
    for name, ecfg in variants:
        models = SC.enrol_speakers(ubm, feats, emap, ecfg)
        if ecfg.adapt_weights or ecfg.adapt_covars:
            # these models no longer share the UBM weights/covariances, so the
            # batched mean-bank scorer does not apply
            s = SC.llr_scores_loop(ubm, models, feats, dev_tr)
        else:
            S, mi, fi = SC.score_matrix(ubm, models, feats, test_files)
            s = SC.gather_trial_scores(S, mi, fi, dev_tr)
        e, _ = M.eer(s, y)
        _, lo, hi = M.eer_bootstrap_ci(s, y, n_boot=400)
        rows.append([name, f"{e*100:.2f}%", f"[{lo*100:.2f}-{hi*100:.2f}]"])
        adapt_res[name] = {"eer": e, "ci": [lo, hi]}
        print(f"  {name:32s} dev EER={e*100:5.2f}%  [{lo*100:4.2f}-{hi*100:5.2f}]",
              flush=True)

    print("\n  Speaker GMMs trained from scratch on the 3 enrolment utterances:")
    for K_s in (2, 4, 8, 16):
        models = {}
        for spk, files in emap.items():
            Xs = torch.cat([feats[f] for f in files], dim=0).to(torch.float64)
            g = GMMConfig(n_components=K_s, cov_type="diag", init="kmeans",
                          max_iter=50, tol=1e-5, var_floor=1e-3)
            models[spk] = GMM(K_s, ubm.D, "diag").fit(Xs, g)
        s = SC.llr_scores_loop(ubm, models, feats, dev_tr)
        e, _ = M.eer(s, y)
        _, lo, hi = M.eer_bootstrap_ci(s, y, n_boot=400)
        rows.append([f"from scratch, K={K_s}", f"{e*100:.2f}%",
                     f"[{lo*100:.2f}-{hi*100:.2f}]"])
        adapt_res[f"scratch_K{K_s}"] = {"eer": e, "ci": [lo, hi]}
        print(f"  {'from scratch, K=' + str(K_s):32s} dev EER={e*100:5.2f}%  "
              f"[{lo*100:4.2f}-{hi*100:5.2f}]", flush=True)
    print()
    table(rows, ["speaker model", "dev EER", "95% CI"])
    plots.bars([r[0] for r in rows], [float(r[1].rstrip('%')) for r in rows],
               "fig04b_adaptation_variants.png", "development EER (%)",
               "Speaker-model construction", rotate=30, figsize=(6.4, 3.8))
    results["adaptation_variants"] = adapt_res

    # ------------------------------------------------------------------ 5
    print("\n" + "-" * 78)
    print("5. VERIFICATION AND SCORING (final configuration)")
    print("-" * 78)
    cohort_models, impostor_files = system.build_cohort(feats, ubm)
    print(f"  T-norm cohort: {len(cohort_models)} background speakers "
          f"(3 utterances each, MAP-adapted with the same r)")
    print(f"  Z-norm impostor utterances: {len(impostor_files)} held-out "
          f"background recordings")

    score_sets, srows = {}, {}
    for norm in ("none", "tnorm", "znorm", "ztnorm"):
        scfg = ScoreConfig(top_c=system.FINAL_SCORE.top_c, norm=norm)
        _, raw, sc = system.score_split("dev", feats, ubm, system.FINAL_ENROL, scfg,
                                        cohort_models, impostor_files)
        summ = M.summarise(sc, y, n_boot=n_boot or 200)
        score_sets[{"none": "raw LLR", "tnorm": "T-norm", "znorm": "Z-norm",
                    "ztnorm": "ZT-norm"}[norm]] = (sc, y)
        srows[norm] = summ
        print(f"  {norm:7s}: EER={summ['eer']*100:5.2f}%  "
              f"[95% CI {summ['eer_ci_lo']*100:4.2f}-{summ['eer_ci_hi']*100:5.2f}]  "
              f"AUC={summ['auc']:.4f}  minDCF(p=0.01)={summ['min_dcf_p01']:.4f}",
              flush=True)
    results["score_norms"] = srows

    plots.det_curve(score_sets, "fig05a_det.png",
                    "DET curves on the development trials")
    plots.roc_curve(score_sets, "fig05b_roc.png",
                    "ROC curves on the development trials")

    # final system scores
    _, raw, final_scores = system.score_split("dev", feats, ubm, system.FINAL_ENROL,
                                              system.FINAL_SCORE, cohort_models,
                                              impostor_files)
    e, thr = M.eer(final_scores, y)
    plots.score_hist(final_scores, y, "fig05c_score_hist.png",
                     f"Final system ({system.FINAL_SCORE.norm}) score distributions",
                     thr=thr)
    plots.score_hist(raw, y, "fig05d_score_hist_raw.png",
                     "Raw average per-frame LLR", thr=M.eer(raw, y)[1])

    print("\n  Operating points (final system):")
    op_rows = []
    far_e, frr_e = M.far_frr_at(final_scores, y, thr)
    op_rows.append(["EER threshold (balanced)", f"{thr:+.4f}", f"{far_e*100:.2f}%",
                    f"{frr_e*100:.2f}%"])
    extra = {}
    for name, target in [("high security (FAR<=1%)", 0.01),
                         ("balanced-security (FAR<=5%)", 0.05),
                         ("convenience (FAR<=10%)", 0.10)]:
        t = M.threshold_at_far(final_scores, y, target)
        far, frr = M.far_frr_at(final_scores, y, t)
        op_rows.append([name, f"{t:+.4f}", f"{far*100:.2f}%", f"{frr*100:.2f}%"])
        extra[name.split(" (")[0]] = t
    table(op_rows, ["operating point", "threshold", "FAR", "FRR"])
    results["operating_points"] = op_rows
    plots.far_frr_vs_threshold(final_scores, y, "fig05e_far_frr.png", eer_thr=thr,
                               extra=extra)

    print("\n  Chosen operating point for the stated use case")
    print("  ('unlocking a shared device by voice -- a false accept is a security")
    print("   breach, a false reject is a retry'):  FAR <= 1%")
    t_use = M.threshold_at_far(final_scores, y, 0.01)
    far_u, frr_u = M.far_frr_at(final_scores, y, t_use)
    print(f"    threshold = {t_use:+.4f}  ->  FAR = {far_u*100:.2f}%, "
          f"FRR = {frr_u*100:.2f}%")
    results["chosen_operating_point"] = {"threshold": t_use, "far": far_u, "frr": frr_u}

    # ---- dev results CSV --------------------------------------------------
    out_csv = os.path.join(C.RESULTS_DIR, "dev_results.csv")
    with open(out_csv, "w", newline="") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["claimed_id", "test_file", "label", "score"])
        for t, s in zip(dev_tr, final_scores.tolist()):
            wtr.writerow([t["claimed_id"], t["test_file"], t["label"], f"{s:.6f}"])
    print(f"\n[saved] {out_csv}  ({len(dev_tr)} rows)")

    summ = M.summarise(final_scores, y, n_boot=n_boot or 200)
    print(f"\nFINAL DEV RESULT: EER = {summ['eer']*100:.2f}% "
          f"(95% bootstrap CI {summ['eer_ci_lo']*100:.2f}-{summ['eer_ci_hi']*100:.2f}%), "
          f"AUC = {summ['auc']:.4f}")
    results["final_dev"] = summ
    C.save_json(results, os.path.join(C.RESULTS_DIR, "task45_enrol_score.json"))
    print(f"[saved] {os.path.join(C.RESULTS_DIR, 'task45_enrol_score.json')}")


if __name__ == "__main__":
    main()
