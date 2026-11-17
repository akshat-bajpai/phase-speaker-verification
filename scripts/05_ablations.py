#!/usr/bin/env python
"""Task 6 -- analysis and ablations (development trials only).

Each ablation changes ONE thing with respect to the final configuration.
Bootstrap 95% confidence intervals are reported throughout because the dev set
holds only 196 genuine trials, so a single misordered trial moves the EER by
about 0.5% absolute.

Ablations: phase representation; magnitude-leakage compliance controls; mixture
count; covariance structure; deltas; silence removal; feature normalisation;
enrolment size (1/2/3 utterances); relevance factor; score normalisation; top-C
scoring; phase-difference baseline vs frame rate; additive-noise robustness.

Usage:  python scripts/05_ablations.py [--only name1,name2]
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from svcore import (cache, config as C, data, features as F, metrics as M,
                    normalise as NZ, plots, scoring as SC, system)
from svcore.config import EnrolConfig, FeatureConfig, GMMConfig, ScoreConfig
from svcore.gmm import GMM

N_BOOT = 400


def table(rows, headers):
    w = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
         for i, h in enumerate(headers)]
    line = "  ".join(h.ljust(w[i]) for i, h in enumerate(headers))
    print(line); print("-" * len(line))
    for r in rows:
        print("  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))


class Bench:
    """Runs one end-to-end system and reports the dev EER."""

    def __init__(self):
        self.dev_tr = data.trials("dev")
        self.y = [t["label"] for t in self.dev_tr]
        self.test_files = sorted({t["test_file"] for t in self.dev_tr})
        self.ubm_files, self.held_files, self.cohort = system.splits()
        self._feat_cache = {}

    def features(self, fcfg: FeatureConfig):
        if fcfg.tag not in self._feat_cache:
            feats = cache.build(fcfg, verbose=False)
            if fcfg.cmvn == "global":
                feats, _ = NZ.fit_and_apply(feats, self.ubm_files)
            self._feat_cache[fcfg.tag] = feats
        return self._feat_cache[fcfg.tag]

    def run(self, fcfg=None, gcfg=None, ecfg=None, scfg=None, feats=None,
            n_enrol=None, top_c=None):
        fcfg = fcfg or system.FINAL_FEATURE
        gcfg = gcfg or system.FINAL_GMM
        ecfg = ecfg or system.FINAL_ENROL
        scfg = scfg or ScoreConfig(norm="none")
        if n_enrol is not None:
            ecfg = EnrolConfig(relevance_factor=ecfg.relevance_factor,
                               n_enrol=n_enrol, adapt_means=ecfg.adapt_means)
        if top_c is not None:
            scfg = ScoreConfig(top_c=top_c, norm=scfg.norm,
                               cohort_size=scfg.cohort_size)
        feats = feats if feats is not None else self.features(fcfg)
        ubm = system.train_ubm(feats, fcfg, gcfg)
        emap = data.enrol_map("dev", ecfg.n_enrol)
        models = SC.enrol_speakers(ubm, feats, emap, ecfg)
        t0 = time.time()
        # The batched mean-bank scorer requires a diagonal UBM (and mean-only
        # adaptation).  For the covariance-structure ablation we fall back to the
        # general per-trial scorer, which handles every covariance type.
        fast = (ubm.cov_type == "diag" and not ecfg.adapt_weights
                and not ecfg.adapt_covars)
        if not fast:
            if scfg.norm != "none":
                raise ValueError("score normalisation requires the diagonal "
                                 "mean-only fast path")
            s = SC.llr_scores_loop(ubm, models, feats, self.dev_tr, scfg.top_c)
            dt = time.time() - t0
        else:
            if scfg.norm == "none":
                S, mi, fi = SC.score_matrix(ubm, models, feats, self.test_files,
                                            scfg.top_c)
            else:
                cohort_models = SC.enrol_speakers(ubm, feats, self.cohort, ecfg)
                S, mi, fi = SC.normalise_scores(ubm, models, cohort_models, feats,
                                                self.test_files, self.held_files,
                                                scfg.norm, scfg.top_c)
            dt = time.time() - t0
            s = SC.gather_trial_scores(S, mi, fi, self.dev_tr)
        e, _ = M.eer(s, self.y)
        _, lo, hi = M.eer_bootstrap_ci(s, self.y, n_boot=N_BOOT)
        return {"eer": e, "ci": [lo, hi], "auc": M.auc(s, self.y),
                "dim": next(iter(feats.values())).shape[1], "score_seconds": dt,
                "scores": s}


def fmt(r):
    return f"{r['eer']*100:5.2f}%  [{r['ci'][0]*100:4.2f}-{r['ci'][1]*100:5.2f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    torch.set_num_threads(8)
    only = set(a for a in args.only.split(",") if a)
    def want(n):
        return not only or n in only

    print("=" * 78)
    print("TASK 6 -- ANALYSIS AND ABLATIONS (development trials)")
    print("=" * 78)
    B = Bench()
    print(f"\nDev trials: {len(B.dev_tr)} ({sum(B.y)} genuine, "
          f"{len(B.y)-sum(B.y)} impostor) over {len(set(t['claimed_id'] for t in B.dev_tr))} "
          f"claimed speakers and {len(B.test_files)} test recordings")
    print("Baseline = the final configuration; every row changes exactly one thing.")
    out = {}

    base = B.run()
    print(f"\nBASELINE: dev EER = {fmt(base)}  (dim={base['dim']})")
    out["baseline"] = {k: v for k, v in base.items() if k != "scores"}

    # ---------------------------------------------------------------- A
    if want("representation"):
        print("\n" + "-" * 78)
        print("A. PHASE REPRESENTATION vs EER  (same framing, same GMM)")
        print("-" * 78)
        rows, res = [], {}
        for name, fc in system.COMPARISON_STREAMS.items():
            r = B.run(fcfg=fc)
            rows.append([name, fc.stream, r["dim"], fmt(r), f"{r['auc']:.4f}"])
            res[name] = {k: v for k, v in r.items() if k != "scores"}
            print(f"  {name:40s} dim={r['dim']:3d}  EER={fmt(r)}", flush=True)
        table(rows, ["representation", "stream", "dim", "dev EER / 95% CI", "AUC"])
        out["representation"] = res
        plots.bars([n.split(" (")[0] for n in res],
                   [res[n]["eer"] * 100 for n in res], "fig06a_representation.png",
                   "development EER (%)", "Phase representation vs EER", rotate=25,
                   yerr=[[res[n]["ci"][0] * 100, res[n]["ci"][1] * 100] for n in res])

    # ---------------------------------------------------------------- B
    if want("compliance"):
        print("\n" + "-" * 78)
        print("B. MAGNITUDE-LEAKAGE CONTROLS (compliance audit)")
        print("-" * 78)
        print("  For each candidate we also run the IDENTICAL pipeline with the phase")
        print("  term deleted.  A genuine phase feature must collapse to chance; if the")
        print("  control matches the full feature, the feature is really a magnitude")
        print("  feature and must be rejected.")
        gd = system.FINAL_FEATURE.replace(stream="mgdcc", n_bands=48, n_ceps=30)
        rows, res = [], {}
        cands = [
            ("IFD (submitted system)", system.FINAL_FEATURE, None),
            ("group delay, exact |X|^2", gd.replace(mgd_denom="exact"),
             gd.replace(mgd_denom="exact", mgd_magnitude_only=True)),
            ("modified GD, gamma=0.9", gd.replace(mgd_denom="smoothed", mgd_gamma=0.9),
             gd.replace(mgd_denom="smoothed", mgd_gamma=0.9, mgd_magnitude_only=True)),
            ("modified GD, gamma=0.5", gd.replace(mgd_denom="smoothed", mgd_gamma=0.5),
             gd.replace(mgd_denom="smoothed", mgd_gamma=0.5, mgd_magnitude_only=True)),
        ]
        for name, fc, ctrl in cands:
            r = B.run(fcfg=fc)
            if ctrl is not None:
                rc = B.run(fcfg=ctrl)
                ctrl_s = fmt(rc)
                gap = (rc["eer"] - r["eer"]) * 100
            else:
                # the IFD control is analytic: a magnitude-weighted circular mean of
                # a constant phase returns that phase whatever the weights, so
                # deleting the phase leaves a constant feature (zero information).
                ctrl_s = "constant by construction"
                gap = float("nan")
            rows.append([name, fmt(r), ctrl_s,
                         "n/a" if math.isnan(gap) else f"{gap:+.2f} pp"])
            res[name] = {"eer": r["eer"], "ci": r["ci"],
                         "control_eer": None if ctrl is None else rc["eer"]}
            print(f"  {name:28s} feature EER={fmt(r)}   phase-free control: {ctrl_s}",
                  flush=True)
        table(rows, ["candidate", "feature EER", "magnitude-only control",
                     "control - feature"])
        out["compliance"] = res

        # empirical confirmation that the IFD aggregation cannot leak magnitude
        cfg = system.FINAL_FEATURE
        w = data.load_wav(data.all_file_ids()[0])
        X, _, logE = F.stft_pair(w, cfg, False)
        theta = torch.full_like(X.real, 0.7)
        z1 = F._weighted_band_cos_sin(theta, X.abs(), cfg)
        z2 = F._weighted_band_cos_sin(theta, torch.rand_like(X.abs()), cfg)
        print(f"\n  Analytic check: with a CONSTANT phase of 0.7 rad, the band "
              f"aggregation returns\n  the same value under the true magnitudes and "
              f"under random weights: max diff = {float((z1-z2).abs().max()):.2e}")
        print(f"  (= cos/sin of 0.7: {math.cos(0.7):.4f}/{math.sin(0.7):.4f}, "
              f"observed {float(z1[0,0]):.4f}/{float(z1[0,cfg.n_bands]):.4f})")
        out["compliance_analytic_maxdiff"] = float((z1 - z2).abs().max())

    # ---------------------------------------------------------------- C
    if want("mixtures"):
        print("\n" + "-" * 78)
        print("C. MIXTURE COUNT vs EER")
        print("-" * 78)
        res = {}
        for K in (16, 32, 64, 128, 256):
            r = B.run(gcfg=system.FINAL_GMM.replace(n_components=K))
            res[K] = {k: v for k, v in r.items() if k != "scores"}
            print(f"  K={K:4d}  EER={fmt(r)}", flush=True)
        out["mixtures"] = res
        ks = sorted(res)
        plots.lines(ks, {"dev EER (%)": [res[k]["eer"] * 100 for k in ks]},
                    "fig06b_mixtures.png", "mixture components K",
                    "development EER (%)", "Mixture count vs EER", logx=True,
                    yerr={"dev EER (%)": [[res[k]["ci"][0] * 100,
                                           res[k]["ci"][1] * 100] for k in ks]})

    # ---------------------------------------------------------------- D
    if want("covariance"):
        print("\n" + "-" * 78)
        print("D. COVARIANCE STRUCTURE vs EER")
        print("-" * 78)
        res = {}
        for cov, K in (("spherical", 128), ("diag", 128), ("tied", 128),
                       ("full", 16), ("full", 32)):
            lab = f"{cov} (K={K})"
            r = B.run(gcfg=system.FINAL_GMM.replace(n_components=K, cov_type=cov),
                      scfg=ScoreConfig(norm="none"))
            res[lab] = {k: v for k, v in r.items() if k != "scores"}
            print(f"  {lab:16s} EER={fmt(r)}", flush=True)
        out["covariance"] = res
        plots.bars(list(res), [res[k]["eer"] * 100 for k in res],
                   "fig06c_covariance.png", "development EER (%)",
                   "Covariance structure vs EER", rotate=20,
                   yerr=[[res[k]["ci"][0] * 100, res[k]["ci"][1] * 100] for k in res])

    # ---------------------------------------------------------------- E
    if want("frontend"):
        print("\n" + "-" * 78)
        print("E. FRONT-END ABLATIONS (deltas / silence removal / normalisation)")
        print("-" * 78)
        fc = system.FINAL_FEATURE
        variants = [
            ("static only (no deltas)", fc.replace(deltas=0)),
            ("+ delta (baseline)", fc),
            ("+ delta + delta-delta", fc.replace(deltas=2)),
            ("no silence removal", fc.replace(vad=False)),
            ("VAD 20 dB", fc.replace(vad_rel_db=20.0)),
            ("VAD 30 dB (baseline)", fc),
            ("VAD 40 dB", fc.replace(vad_rel_db=40.0)),
            ("no normalisation", fc.replace(cmvn="none")),
            ("global norm (baseline)", fc),
            ("per-utterance CMN", fc.replace(cmvn="m")),
            ("per-utterance CMVN", fc.replace(cmvn="mv")),
            ("no magnitude reliability weight", fc.replace(mag_weight=False)),
            ("DCT compression (20 coeff)", fc.replace(n_ceps=20)),
            ("raw wrapped Delta + bands (no cos/sin)",
             fc.replace(ifd_cossin=False, n_ceps=0)),
            ("raw wrapped Delta + DCT (no cos/sin)",
             fc.replace(ifd_cossin=False, n_ceps=20)),
        ]
        rows, res = [], {}
        for name, c in variants:
            r = B.run(fcfg=c)
            rows.append([name, r["dim"], fmt(r)])
            res[name] = {k: v for k, v in r.items() if k != "scores"}
            print(f"  {name:34s} dim={r['dim']:3d}  EER={fmt(r)}", flush=True)
        table(rows, ["front-end variant", "dim", "dev EER / 95% CI"])
        out["frontend"] = res

    # ---------------------------------------------------------------- F
    if want("enrolsize"):
        print("\n" + "-" * 78)
        print("F. ENROLMENT SIZE (enrol_index 0..k-1) vs EER")
        print("-" * 78)
        res = {}
        for n in (1, 2, 3):
            for r_fac in (8.0, 16.0, 32.0):
                r = B.run(ecfg=EnrolConfig(relevance_factor=r_fac, n_enrol=n),
                          n_enrol=n)
                res[f"n={n},r={r_fac:g}"] = {k: v for k, v in r.items() if k != "scores"}
                print(f"  {n} utterance(s), r={r_fac:4.0f}  EER={fmt(r)}", flush=True)
        out["enrol_size"] = res
        ns = [1, 2, 3]
        plots.lines(ns, {f"r={rf:g}": [res[f"n={n},r={rf:g}"]["eer"] * 100 for n in ns]
                         for rf in (8.0, 16.0, 32.0)},
                    "fig06d_enrol_size.png", "enrolment utterances",
                    "development EER (%)", "Enrolment size vs EER")

    # ---------------------------------------------------------------- G
    if want("scorenorm"):
        print("\n" + "-" * 78)
        print("G. SCORE NORMALISATION")
        print("-" * 78)
        res = {}
        for norm in ("none", "tnorm", "znorm", "ztnorm"):
            r = B.run(scfg=ScoreConfig(norm=norm))
            res[norm] = {k: v for k, v in r.items() if k != "scores"}
            print(f"  {norm:7s}  EER={fmt(r)}", flush=True)
        out["score_norm"] = res

        print("\n  T-norm cohort size (background speakers with >=3 recordings):")
        feats = B.features(system.FINAL_FEATURE)
        ubm = system.train_ubm(feats, system.FINAL_FEATURE, system.FINAL_GMM)
        emap = data.enrol_map("dev", 3)
        models = SC.enrol_speakers(ubm, feats, emap, system.FINAL_ENROL)
        res_c = {}
        for cs in (30, 60, 100, 150):
            _, _, coh = data.background_splits(cohort_size=cs)
            cm = SC.enrol_speakers(ubm, feats, coh, system.FINAL_ENROL)
            S, mi, fi = SC.normalise_scores(ubm, models, cm, feats, B.test_files,
                                            B.held_files, "tnorm")
            s = SC.gather_trial_scores(S, mi, fi, B.dev_tr)
            e, _ = M.eer(s, B.y)
            _, lo, hi = M.eer_bootstrap_ci(s, B.y, n_boot=N_BOOT)
            res_c[cs] = {"eer": e, "ci": [lo, hi], "n_cohort": len(cm)}
            print(f"    cohort={cs:4d} speakers  EER={e*100:5.2f}%  "
                  f"[{lo*100:4.2f}-{hi*100:5.2f}]", flush=True)
        out["tnorm_cohort_size"] = res_c

    # ---------------------------------------------------------------- H
    if want("topc"):
        print("\n" + "-" * 78)
        print("H. TOP-C COMPONENT SCORING (speed / accuracy trade-off)")
        print("-" * 78)
        rows, res = [], {}
        for c in (1, 2, 5, 10, 20, 50, 0):
            r = B.run(top_c=c)
            lab = "all" if c == 0 else str(c)
            rows.append([lab, fmt(r), f"{r['score_seconds']:.2f}s"])
            res[lab] = {k: v for k, v in r.items() if k != "scores"}
            print(f"  top-C={lab:3s}  EER={fmt(r)}  scoring time "
                  f"{r['score_seconds']:.2f}s", flush=True)
        table(rows, ["C", "dev EER / 95% CI", "scoring time (1824 trials)"])
        out["top_c"] = res

    # ---------------------------------------------------------------- I
    if want("baseline"):
        print("\n" + "-" * 78)
        print("I. PHASE-DIFFERENCE BASELINE vs FRAME RATE")
        print("-" * 78)
        print("  A short hop changes two things at once: the baseline (lag x hop) of")
        print("  the phase difference, and the number of frames per utterance.")
        print("  `ifd_lag` separates them.")
        fc = system.FINAL_FEATURE
        rows, res = [], {}
        for hop, lag in ((3.0, 1), (3.0, 2), (3.0, 3), (3.0, 4),
                         (5.0, 1), (10.0, 1), (2.0, 5), (1.5, 2)):
            c = fc.replace(hop_ms=hop, ifd_lag=lag)
            feats = B.features(c)
            nf = float(cache.frame_counts(feats, B.test_files).mean())
            r = B.run(fcfg=c, feats=feats)
            rows.append([f"{hop:g}", lag, f"{hop*lag:g}", f"{nf:.0f}", fmt(r)])
            res[f"hop{hop:g}_lag{lag}"] = {"hop_ms": hop, "lag": lag,
                                           "baseline_ms": hop * lag, "frames": nf,
                                           "eer": r["eer"], "ci": r["ci"]}
            print(f"  hop={hop:4.1f}ms lag={lag} -> baseline={hop*lag:4.1f}ms, "
                  f"{nf:5.1f} frames/utt   EER={fmt(r)}", flush=True)
        table(rows, ["hop (ms)", "lag", "baseline (ms)", "frames/utt",
                     "dev EER / 95% CI"])
        out["baseline_vs_framerate"] = res

    # ---------------------------------------------------------------- J
    if want("noise"):
        print("\n" + "-" * 78)
        print("J. ADDITIVE-NOISE ROBUSTNESS (stretch goal)")
        print("-" * 78)
        print("  White noise added to the TEST recordings only; enrolment, the UBM and")
        print("  the cohort stay clean, as in a deployed system.")
        fc = system.FINAL_FEATURE
        clean = B.features(fc)
        ubm = system.train_ubm(clean, fc, system.FINAL_GMM)
        emap = data.enrol_map("dev", 3)
        models = SC.enrol_speakers(ubm, clean, emap, system.FINAL_ENROL)
        _, nrm = system.prepare_features(fc, verbose=False)
        rows, res = [], {}
        for snr in (None, 20, 10, 5, 0):
            noisy = dict(clean)
            if snr is not None:
                g = torch.Generator().manual_seed(C.SEED)
                for fid in B.test_files:
                    w = data.load_wav(fid)
                    p = w.pow(2).mean()
                    n = torch.randn(w.shape, generator=g) * (p / (10 ** (snr / 10))).sqrt()
                    f = F.extract(w + n, fc)
                    noisy[fid] = nrm.apply(f) if nrm is not None else f
            S, mi, fi = SC.score_matrix(ubm, models, noisy, B.test_files)
            s = SC.gather_trial_scores(S, mi, fi, B.dev_tr)
            e, _ = M.eer(s, B.y)
            _, lo, hi = M.eer_bootstrap_ci(s, B.y, n_boot=N_BOOT)
            lab = "clean" if snr is None else f"{snr} dB"
            rows.append([lab, f"{e*100:5.2f}%", f"[{lo*100:4.2f}-{hi*100:5.2f}]"])
            res[lab] = {"eer": e, "ci": [lo, hi]}
            print(f"  test SNR {lab:6s}  EER={e*100:5.2f}%  "
                  f"[{lo*100:4.2f}-{hi*100:5.2f}]", flush=True)
        table(rows, ["test SNR", "dev EER", "95% CI"])
        out["noise"] = res
        labs = [k for k in res]
        plots.bars(labs, [res[k]["eer"] * 100 for k in labs],
                   "fig06e_noise.png", "development EER (%)",
                   "Robustness to additive noise on the test recording", rotate=0)

    C.save_json(out, os.path.join(C.RESULTS_DIR, "task6_ablations.json"))
    print(f"\n[saved] {os.path.join(C.RESULTS_DIR, 'task6_ablations.json')}")


if __name__ == "__main__":
    main()
