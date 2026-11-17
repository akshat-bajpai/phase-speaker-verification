"""Run one system configuration across every validation protocol.

A modelling choice is only adopted if it helps on the pooled signal, not just on
the 196 genuine trials of the official development list.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from . import cache, config as C, data, metrics as M, normalise as NZ
from . import advanced as A, protocols as P, scoring as SC, system
from .config import EnrolConfig, FeatureConfig, GMMConfig, ScoreConfig
from .gmm import GMM


class Validator:
    """Caches features, UBMs and protocols so many systems can be compared fast."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.ubm_files, self.held_files, self.cohort = system.splits()
        self.dev_tr = data.trials("dev")
        self.dev_y = [t["label"] for t in self.dev_tr]
        self.dev_emap = data.enrol_map("dev", 3)
        self.dev_tests = sorted({t["test_file"] for t in self.dev_tr})
        self.loo_rot, self.loo_n = P.dev_loo()
        self.bkg_emap, self.bkg_tr, self.bkg_excl = P.background_protocol()
        self.bkg_tests = sorted({t["test_file"] for t in self.bkg_tr})
        self.bkg_ubm_files = [f for f in self.ubm_files if f not in self.bkg_excl]
        self._feats: Dict[str, dict] = {}
        self._ubm: Dict[str, GMM] = {}

    # -- caches ------------------------------------------------------------
    def features(self, fcfg: FeatureConfig):
        if fcfg.tag not in self._feats:
            f = cache.build(fcfg, verbose=self.verbose)
            if fcfg.cmvn == "global":
                f, _ = NZ.fit_and_apply(f, self.ubm_files)
            self._feats[fcfg.tag] = f
        return self._feats[fcfg.tag]

    def ubm(self, fcfg: FeatureConfig, gcfg: GMMConfig, exclude_bkg: bool = False) -> GMM:
        key = f"{fcfg.tag}|{gcfg.n_components}|{gcfg.cov_type}|{gcfg.seed}|{int(exclude_bkg)}"
        if key in self._ubm:
            return self._ubm[key]
        if not exclude_bkg:
            m = system.train_ubm(self.features(fcfg), fcfg, gcfg)
        else:
            path = os.path.join(C.CACHE_DIR, f"ubmX_{fcfg.tag}_K{gcfg.n_components}"
                                             f"_{gcfg.cov_type}_s{gcfg.seed}.pt")
            if os.path.exists(path):
                m = GMM.from_state_dict(torch.load(path, map_location="cpu",
                                                   weights_only=False))
            else:
                X = cache.pooled(self.features(fcfg), self.bkg_ubm_files)
                dt = torch.float64 if gcfg.dtype == "float64" else torch.float32
                m = GMM(gcfg.n_components, X.shape[1], gcfg.cov_type, dtype=dt)
                m.fit(X, gcfg)
                torch.save(m.state_dict(), path)
        self._ubm[key] = m
        return m

    # -- scoring -----------------------------------------------------------
    def cohort_utterances(self):
        """Enrolment recordings of the cohort speakers, as impostor utterances."""
        return [f for v in self.cohort.values() for f in v]

    def matrix(self, ubm, feats, emap, test_files, ecfg, scfg,
               sym: float = 1.0, test_relevance=None, cohort_models=None,
               asnorm_top: int = 0, asnorm_sym: bool = True):
        """Score matrix for one protocol; ``sym < 1`` mixes in reverse scoring."""
        models = SC.enrol_speakers(ubm, feats, emap, ecfg)
        if asnorm_top:
            cm = cohort_models or SC.enrol_speakers(ubm, feats, self.cohort, ecfg)
            return A.asnorm_matrix(ubm, models, cm, feats, test_files,
                                   self.cohort_utterances(), asnorm_top,
                                   asnorm_sym)
        if sym < 1.0:
            S, mi, fi = A.symmetric_score_matrix(ubm, models, feats, emap,
                                                 test_files, ecfg, weight=sym,
                                                 test_relevance=test_relevance,
                                                 top_c=scfg.top_c)
            return S, mi, fi
        if scfg.norm == "none":
            return SC.score_matrix(ubm, models, feats, test_files, scfg.top_c)
        cm = cohort_models or SC.enrol_speakers(ubm, feats, self.cohort, ecfg)
        return SC.normalise_scores(ubm, models, cm, feats, test_files,
                                   self.held_files, scfg.norm, scfg.top_c)

    def _score(self, ubm, feats, emap, trials, test_files, ecfg, scfg,
               cohort_models=None, sym=1.0, test_relevance=None,
               asnorm_top=0, asnorm_sym=True):
        if asnorm_top or sym < 1.0 or scfg.norm == "none":
            S, mi, fi = self.matrix(ubm, feats, emap, test_files, ecfg, scfg,
                                    sym, test_relevance, cohort_models,
                                    asnorm_top, asnorm_sym)
            return SC.gather_trial_scores(S, mi, fi, trials)
        models = SC.enrol_speakers(ubm, feats, emap, ecfg)
        if False:
            pass
        cm = cohort_models or SC.enrol_speakers(ubm, feats, self.cohort, ecfg)
        S, mi, fi = SC.normalise_scores(ubm, models, cm, feats, test_files,
                                        self.held_files, scfg.norm, scfg.top_c)
        return SC.gather_trial_scores(S, mi, fi, trials)

    def run(self, fcfg: Optional[FeatureConfig] = None,
            gcfg: Optional[GMMConfig] = None, ecfg: Optional[EnrolConfig] = None,
            scfg: Optional[ScoreConfig] = None, protocols=("dev", "loo", "bkg"),
            n_boot: int = 300, sym: float = 1.0, test_relevance=None,
            asnorm_top: int = 0, asnorm_sym: bool = True) -> Dict[str, dict]:
        fcfg = fcfg or system.FINAL_FEATURE
        gcfg = gcfg or system.FINAL_GMM
        ecfg = ecfg or system.FINAL_ENROL
        scfg = scfg or ScoreConfig(norm="none")
        feats = self.features(fcfg)
        out: Dict[str, dict] = {}

        if "dev" in protocols:
            ubm = self.ubm(fcfg, gcfg)
            s = self._score(ubm, feats, self.dev_emap, self.dev_tr, self.dev_tests,
                            ecfg, scfg, sym=sym, test_relevance=test_relevance,
                            asnorm_top=asnorm_top, asnorm_sym=asnorm_sym)
            out["dev"] = _summ(s, self.dev_y, n_boot)

        if "loo" in protocols:
            ubm = self.ubm(fcfg, gcfg)
            e2 = EnrolConfig(relevance_factor=ecfg.relevance_factor, n_enrol=2,
                             adapt_means=ecfg.adapt_means)
            ss, yy = [], []
            for emap, trials in self.loo_rot:
                tf = sorted({t["test_file"] for t in trials})
                ss.append(self._score(ubm, feats, emap, trials, tf, e2, scfg,
                                      sym=sym, test_relevance=test_relevance,
                                      asnorm_top=asnorm_top, asnorm_sym=asnorm_sym))
                yy += [t["label"] for t in trials]
            out["loo"] = _summ(torch.cat(ss), yy, n_boot)

        if "bkg" in protocols:
            ubmx = self.ubm(fcfg, gcfg, exclude_bkg=True)
            y = [t["label"] for t in self.bkg_tr]
            s = self._score(ubmx, feats, self.bkg_emap, self.bkg_tr,
                            self.bkg_tests, ecfg, scfg, sym=sym,
                            test_relevance=test_relevance,
                            asnorm_top=asnorm_top, asnorm_sym=asnorm_sym)
            out["bkg"] = _summ(s, y, n_boot)

        out["mean_eer"] = sum(out[k]["eer"] for k in protocols) / len(protocols)
        out["n_genuine_total"] = sum(out[k]["n_gen"] for k in protocols)
        return out


    def run_fusion(self, fcfgs, gcfg=None, ecfg=None, scfg=None,
                   weights=None, protocols=("dev", "loo", "bkg"),
                   n_boot: int = 300, sym: float = 1.0,
                   asnorm_top: int = 0, asnorm_sym: bool = True) -> Dict[str, dict]:
        """Linear fusion of several systems' standardised score matrices."""
        from .config import EnrolConfig as _E
        gcfg = gcfg or system.FINAL_GMM
        ecfg = ecfg or system.FINAL_ENROL
        scfg = scfg or ScoreConfig(norm="none")
        out: Dict[str, dict] = {}

        def fused(emap, trials, test_files, ec, exclude_bkg=False):
            mats = []
            for fc in fcfgs:
                feats = self.features(fc)
                ubm = self.ubm(fc, gcfg, exclude_bkg=exclude_bkg)
                S, mi, fi = self.matrix(ubm, feats, emap, test_files, ec, scfg,
                                        sym, None, None, asnorm_top, asnorm_sym)
                mats.append(SC.gather_trial_scores(S, mi, fi, trials))
            return A.fuse(mats, weights)

        if "dev" in protocols:
            s = fused(self.dev_emap, self.dev_tr, self.dev_tests, ecfg)
            out["dev"] = _summ(s, self.dev_y, n_boot)
        if "loo" in protocols:
            e2 = _E(relevance_factor=ecfg.relevance_factor, n_enrol=2,
                    adapt_means=ecfg.adapt_means)
            ss, yy = [], []
            for emap, trials in self.loo_rot:
                tf = sorted({t["test_file"] for t in trials})
                ss.append(fused(emap, trials, tf, e2))
                yy += [t["label"] for t in trials]
            out["loo"] = _summ(torch.cat(ss), yy, n_boot)
        if "bkg" in protocols:
            y = [t["label"] for t in self.bkg_tr]
            s = fused(self.bkg_emap, self.bkg_tr, self.bkg_tests, ecfg,
                      exclude_bkg=True)
            out["bkg"] = _summ(s, y, n_boot)
        out["mean_eer"] = sum(out[k]["eer"] for k in protocols) / len(protocols)
        out["n_genuine_total"] = sum(out[k]["n_gen"] for k in protocols)
        return out


def _summ(s: torch.Tensor, y: Sequence[int], n_boot: int) -> dict:
    e, _ = M.eer(s, y)
    d = {"eer": e, "auc": M.auc(s, y), "n_gen": int(sum(y))}
    if n_boot:
        _, lo, hi = M.eer_bootstrap_ci(s, y, n_boot=n_boot)
        d["ci"] = [lo, hi]
    return d


def fmt(r: Dict[str, dict], protocols=("dev", "loo", "bkg")) -> str:
    parts = [f"{k}={r[k]['eer']*100:5.2f}%" for k in protocols]
    return "  ".join(parts) + f"   MEAN={r['mean_eer']*100:5.2f}%"


    # -- fusion ------------------------------------------------------------
