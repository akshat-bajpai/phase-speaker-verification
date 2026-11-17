"""Eval-mirroring validation bench (v4).

Why
---
The official dev list has 196 genuine trials, so one misordered trial moves
the EER by ~0.5 %.  Earlier submissions were tuned against it and lost 1-3 pp on
the leaderboard.  This bench rebuilds the *exact* eval trial construction on
two speaker-disjoint pools so a modelling choice is judged on far more
genuine trials:

``devrot``  The 97 dev targets each have 5 distinct takes (3 enrolment + 2
            genuine-test).  Every one of the C(5,3) = 10 ways of choosing the 3
            enrolment takes is a rotation; the 2 remaining takes are genuine
            tests.  Trials follow the organisers' recipe: each target test file
            is paired with its own speaker once and with 3 other targets; each
            of the 260 dev impostor-speaker files is paired with 4 random
            targets.  Rotation 0 uses enrol_index 0,1,2 (the official enrolment).
            1,940 genuine trials (correlated across rotations).

``bkgcv``   4-fold cross-validation over *background* speakers.  For fold f
            the UBM, cohort and global normalisation are re-fitted on the other
            three folds only; the fold's 4-take speakers are targets (enrol 3,
            test 1, 4 rotations) and its 1-3-take speakers are impostor
            speakers that the UBM has never seen -- exactly as in eval.  368
            genuine trials from 92 further speakers.

Everything is torch; the trial lists are built with a seeded torch.Generator.
"""
from __future__ import annotations

import itertools
import math
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from . import cache, config as C, data, metrics as M, normalise as NZ
from .config import EnrolConfig, FeatureConfig, GMMConfig
from .gmm import GMM
from .map_adapt import sufficient_statistics


# ---------------------------------------------------------------------------
# trial construction (mirrors the organisers' recipe)
# ---------------------------------------------------------------------------


def _make_trials(target_tests: Dict[str, str], external: Sequence[str],
                 targets: Sequence[str], g: torch.Generator,
                 n_imp_target: int = 3, n_imp_external: int = 4) -> List[dict]:
    """target_tests: test_file -> true speaker."""
    tg = list(targets)
    out: List[dict] = []

    def pick(n, exclude):
        pool = [t for t in tg if t != exclude]
        n = min(n, len(pool))
        idx = torch.randperm(len(pool), generator=g)[:n].tolist()
        return [pool[i] for i in idx]

    for f in sorted(target_tests):
        s = target_tests[f]
        out.append({"claimed_id": s, "test_file": f, "label": 1})
        for c in pick(n_imp_target, s):
            out.append({"claimed_id": c, "test_file": f, "label": 0})
    for f in sorted(external):
        for c in pick(n_imp_external, None):
            out.append({"claimed_id": c, "test_file": f, "label": 0})
    return out


@dataclass
class Rotation:
    emap: Dict[str, List[str]]          # speaker -> enrolment files
    trials: List[dict]
    test_files: List[str]


@dataclass
class Pool:
    """A set of rotations sharing one UBM-training set and one cohort."""
    name: str
    ubm_files: List[str]
    cohort: Dict[str, List[str]]        # cohort speaker -> files (>=3)
    rotations: List[Rotation]
    all_files: List[str]                # every test-able file of the pool


def _cohort_from(ubm_files: Sequence[str], min_files: int, size: int
                 ) -> Dict[str, List[str]]:
    """Cohort speakers drawn from the UBM-training speakers only."""
    keep = set(ubm_files)
    spk = data.speaker_files([r for r in data.background() if r["file_id"] in keep])
    cand = sorted(s for s, v in spk.items() if len(v) >= min_files)
    if size:
        cand = cand[:size]
    return {s: spk[s][:3] for s in cand}


def dev_rotations(seed: int = C.SEED, cohort_min: int = 3,
                  cohort_size: int = 150) -> Pool:
    enrol_rows = data.enrol("dev")
    by_spk: Dict[str, Dict[int, str]] = {}
    for r in enrol_rows:
        by_spk.setdefault(r["speaker_id"], {})[int(r["enrol_index"])] = r["file_id"]
    trials = data.trials("dev")
    gen_tests: Dict[str, List[str]] = {}
    for t in trials:
        if int(t["label"]) == 1:
            gen_tests.setdefault(t["claimed_id"], []).append(t["test_file"])
    owned = {f for v in gen_tests.values() for f in v}
    external = sorted({t["test_file"] for t in trials} - owned)
    takes = {s: [by_spk[s][0], by_spk[s][1], by_spk[s][2]] + sorted(gen_tests[s])[:2]
             for s in sorted(by_spk)}
    # the one dev speaker with 4 genuine test files keeps the extra two as
    # additional genuine tests in every rotation
    extra = {s: sorted(gen_tests[s])[2:] for s in by_spk if len(gen_tests[s]) > 2}

    g = torch.Generator().manual_seed(seed + 404)
    rots: List[Rotation] = []
    targets = sorted(takes)
    for combo in itertools.combinations(range(5), 3):
        emap = {s: [takes[s][i] for i in combo] for s in targets}
        tt = {}
        for s in targets:
            for i in range(5):
                if i not in combo:
                    tt[takes[s][i]] = s
            for f in extra.get(s, []):
                tt[f] = s
        if combo == (0, 1, 2):
            tr = [dict(t, label=int(t["label"])) for t in trials]   # official list
        else:
            tr = _make_trials(tt, external, targets, g)
        rots.append(Rotation(emap, tr, sorted({t["test_file"] for t in tr})))
    all_files = sorted({f for v in takes.values() for f in v}
                       | {f for v in extra.values() for f in v} | set(external))
    ubm_files, _, _ = data.background_splits(0.10, 150, 3, C.SEED)
    cohort = _cohort_from(ubm_files, cohort_min, cohort_size)
    return Pool("devrot", ubm_files, cohort, rots, all_files)


def bkg_folds(n_folds: int = 4, seed: int = C.SEED, cohort_min: int = 3,
              cohort_size: int = 150) -> List[Pool]:
    spk2files = data.speaker_files(data.background())
    spks = sorted(spk2files)
    g = torch.Generator().manual_seed(seed + 77)
    perm = torch.randperm(len(spks), generator=g).tolist()
    fold_of = {spks[i]: k % n_folds for k, i in enumerate(perm)}
    pools: List[Pool] = []
    for f in range(n_folds):
        in_fold = [s for s in spks if fold_of[s] == f]
        out_fold = [s for s in spks if fold_of[s] != f]
        ubm_files = [x for s in out_fold for x in spk2files[s]]
        cohort = _cohort_from(ubm_files, cohort_min, cohort_size)
        targets = [s for s in in_fold if len(spk2files[s]) >= 4]
        external = [x for s in in_fold if len(spk2files[s]) < 4 for x in spk2files[s]]
        gr = torch.Generator().manual_seed(seed + 1000 + f)
        rots = []
        for k in range(4):
            emap = {s: [x for i, x in enumerate(spk2files[s][:4]) if i != k]
                    for s in targets}
            tt = {spk2files[s][k]: s for s in targets}
            tr = _make_trials(tt, external, targets, gr)
            rots.append(Rotation(emap, tr, sorted({t["test_file"] for t in tr})))
        all_files = sorted({x for s in in_fold for x in spk2files[s]})
        pools.append(Pool(f"bkg{f}", ubm_files, cohort, rots, all_files))
    return pools


# ---------------------------------------------------------------------------
# fast mean-only scoring of many models against many files
# ---------------------------------------------------------------------------


class FastScorer:
    """Average per-frame LLR of mean-only-MAP models (shared covariances).

    Supports ``diag`` and ``full`` UBMs.  ``float32`` is ample here: per-frame
    log-likelihoods are O(100) and the LLR is averaged over ~100 frames, so the
    rounding error is ~1e-5.
    """

    def __init__(self, ubm: GMM, dtype=torch.float32):
        assert ubm.cov_type in ("diag", "full")
        self.ubm, self.dtype, self.full = ubm, dtype, ubm.cov_type == "full"
        self.K, self.D = ubm.K, ubm.D
        self.log_w = torch.log(ubm.weights).to(dtype)
        if self.full:
            ubm._prepare()
            self.P = ubm._cache["P"].to(dtype)                       # (K,D,D)
            self.log_norm = ubm._cache["log_norm"].to(dtype)
        else:
            self.inv_var = (1.0 / ubm.covars).to(dtype)
            self.log_norm = (-0.5 * (self.D * math.log(2 * math.pi)
                                     + torch.log(ubm.covars).sum(1))).to(dtype)
        self.ubm_means = ubm.means.to(dtype)

    def frame_ll(self, X: torch.Tensor, means: torch.Tensor,
                 var: Optional[torch.Tensor] = None) -> torch.Tensor:
        """X (T,D), means (M,K,D) [, var (M,K,D)] -> per-frame log-lik (T,M)."""
        Mn = means.shape[0]
        X = X.to(self.dtype)
        if var is not None:                      # per-model diagonal variances
            iv = (1.0 / var).to(self.dtype)                          # (M,K,D)
            ln = (-0.5 * (self.D * math.log(2 * math.pi)
                          + torch.log(var).sum(2))).to(self.dtype)   # (M,K)
            quad = (X.pow(2) @ iv.reshape(Mn * self.K, self.D).t()).view(-1, Mn, self.K)
            cross = (X @ (means * iv).reshape(Mn * self.K, self.D).t()).view(-1, Mn, self.K)
            mc = (means.pow(2) * iv).sum(2)
            lw = ln + self.log_w - 0.5 * (quad - 2 * cross + mc.unsqueeze(0))
            return torch.logsumexp(lw, dim=2)
        if self.full:
            Xw = torch.einsum("td,kde->kte", X, self.P)              # (K,T,D)
            nu = torch.einsum("mkd,kde->kme", means, self.P)         # (K,M,D)
            quad = Xw.pow(2).sum(2).t()                              # (T,K)
            cross = torch.bmm(Xw, nu.transpose(1, 2)).permute(1, 2, 0)  # (T,M,K)
            mc = nu.pow(2).sum(2).t()                                # (M,K)
        else:
            quad = X.pow(2) @ self.inv_var.t()                       # (T,K)
            mov = (means * self.inv_var).reshape(Mn * self.K, self.D)
            cross = (X @ mov.t()).view(-1, Mn, self.K)
            mc = (means.pow(2) * self.inv_var).sum(2)                # (M,K)
        lw = self.log_norm + self.log_w - 0.5 * (quad.unsqueeze(1) - 2 * cross
                                                 + mc.unsqueeze(0))
        return torch.logsumexp(lw, dim=2)

    def ubm_frame_ll(self, X: torch.Tensor) -> torch.Tensor:
        return self.frame_ll(X, self.ubm_means.unsqueeze(0))[:, 0]

    def matrix(self, means: torch.Tensor, feats: Dict[str, torch.Tensor],
               files: Sequence[str], chunk_models: int = 64,
               var: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(M, n_files) average per-frame LLR."""
        Mn = means.shape[0]
        S = torch.zeros(Mn, len(files), dtype=torch.float64)
        means = means.to(self.dtype)
        for j, f in enumerate(files):
            X = feats[f]
            if X.shape[0] == 0:
                continue
            u = self.ubm_frame_ll(X)
            for a in range(0, Mn, chunk_models):
                ll = self.frame_ll(X, means[a:a + chunk_models],
                                   None if var is None else var[a:a + chunk_models])
                S[a:a + chunk_models, j] = (ll - u.unsqueeze(1)).mean(0).double()
        return S


def map_mean_var(ubm: GMM, feats: Dict[str, torch.Tensor],
                 emap: Dict[str, List[str]], r: float, r_var: float,
                 floor: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Mean + diagonal-variance MAP (Reynolds 2000, eqs. 12-14) with separate
    relevance factors; variances floored at ``floor`` x the UBM variance."""
    keys = sorted(emap)
    mus, vs = [], []
    for s in keys:
        X = torch.cat([feats[f] for f in emap[s]], 0).to(ubm.dtype)
        if X.shape[0] == 0:
            mus.append(ubm.means.clone()); vs.append(ubm.covars.clone()); continue
        n, F, S2 = sufficient_statistics(ubm, X)
        nz = n.clamp_min(1e-10).unsqueeze(1)
        a = (n / (n + r)).unsqueeze(1)
        av = (n / (n + r_var)).unsqueeze(1)
        mu = a * F / nz + (1 - a) * ubm.means
        var = av * S2 / nz + (1 - av) * (ubm.covars + ubm.means.pow(2)) - mu.pow(2)
        var = torch.maximum(var, floor * ubm.covars)
        keep = (n > 0).unsqueeze(1)
        mus.append(torch.where(keep, mu, ubm.means))
        vs.append(torch.where(keep, var, ubm.covars))
    return torch.stack(mus), torch.stack(vs), keys


def map_means(ubm: GMM, feats: Dict[str, torch.Tensor],
              emap: Dict[str, List[str]], r: float) -> Tuple[torch.Tensor, List[str]]:
    """Mean-only MAP (Reynolds 2000) for every speaker; (M,K,D) stacked means."""
    keys = sorted(emap)
    out = []
    for s in keys:
        X = torch.cat([feats[f] for f in emap[s]], 0).to(ubm.dtype)
        if X.shape[0] == 0:
            out.append(ubm.means.clone())
            continue
        n, F, _ = sufficient_statistics(ubm, X)
        a = (n / (n + r)).unsqueeze(1)
        Ex = F / n.clamp_min(1e-10).unsqueeze(1)
        mu = torch.where((n > 0).unsqueeze(1), a * Ex + (1 - a) * ubm.means, ubm.means)
        out.append(mu)
    return torch.stack(out, 0), keys


# ---------------------------------------------------------------------------
# score normalisation on matrices
# ---------------------------------------------------------------------------


def _top_mean_std(E: torch.Tensor, top: int, dim: int):
    n = E.shape[dim] if not top else min(top, E.shape[dim])
    v = torch.topk(E, n, dim=dim).values
    return v.mean(dim), v.std(dim).clamp_min(1e-6)


@dataclass
class Mats:
    """Every matrix a normalisation scheme may need, for one rotation."""
    S: torch.Tensor                # targets x files
    Ct: torch.Tensor               # cohort models x files
    Es: torch.Tensor               # targets x cohort utterances
    Cc: Optional[torch.Tensor]     # cohort models x cohort utterances
    mi: Dict[str, int]
    fi: Dict[str, int]


def normalise(m: Mats, scheme: str, top: int = 0, **kw) -> torch.Tensor:
    """Return a normalised (targets x files) matrix.

    raw      -- the average per-frame LLR
    tnorm    -- per-utterance stats of the background cohort models
    znorm    -- per-model stats on background cohort utterances
    snorm    -- 0.5 (z + t)
    asnorm   -- adaptive s-norm: top-``top`` cohort entries on each side
    asnormE  -- adaptive s-norm whose utterance-side cohort is the background
                cohort models PLUS the other enrolled target models of the split
    """
    S = m.S
    if scheme == "raw":
        return S
    if scheme == "tnorm":
        mu, sd = _top_mean_std(m.Ct, top, 0)
        return (S - mu) / sd
    if scheme == "znorm":
        mu, sd = _top_mean_std(m.Es, top, 1)
        return (S - mu.unsqueeze(1)) / sd.unsqueeze(1)
    if scheme in ("snorm", "asnorm"):
        t = top if scheme == "asnorm" else 0
        mu_t, sd_t = _top_mean_std(m.Ct, t, 0)
        mu_z, sd_z = _top_mean_std(m.Es, t, 1)
        return 0.5 * ((S - mu_t) / sd_t + (S - mu_z.unsqueeze(1)) / sd_z.unsqueeze(1))
    if scheme == "asnormE":
        # utterance side: cohort = background cohort models + other targets
        n_t = S.shape[0]
        eye = torch.eye(n_t, dtype=torch.bool)
        outs = []
        mu_z, sd_z = _top_mean_std(m.Es, top, 1)
        zpart = (S - mu_z.unsqueeze(1)) / sd_z.unsqueeze(1)
        # per target row i, utterance cohort for column j = Ct[:,j] + S[~i, j]
        # (computed column-wise with the target itself masked out)
        big = torch.cat([m.Ct, S], 0)                            # (Nc+Nt, F)
        tpart = torch.empty_like(S)
        Nc = m.Ct.shape[0]
        for i in range(n_t):
            keep = torch.ones(big.shape[0], dtype=torch.bool)
            keep[Nc + i] = False
            mu, sd = _top_mean_std(big[keep], top, 0)
            tpart[i] = (S[i] - mu) / sd
        w = kw.get("wt", 0.5)
        return w * tpart + (1 - w) * zpart
    raise ValueError(scheme)


# ---------------------------------------------------------------------------
# the bench
# ---------------------------------------------------------------------------


@dataclass
class SystemSpec:
    feature: FeatureConfig
    gmm: GMMConfig
    relevance: float = 16.0
    label: str = ""
    transform: str = ""          # "" | "lda_utt:<dim>" | "lda_frame:<dim>"
    r_var: float = 0.0           # >0: also MAP-adapt diagonal variances

    @property
    def ftag(self) -> str:
        return self.feature.tag + (f"_T{self.transform.replace(':', '').replace('/', 'of')}"
                                   if self.transform else "")


class Bench:
    def __init__(self, n_folds: int = 4, verbose: bool = True,
                 cohort_min: int = 3, cohort_size: int = 150):
        self.verbose = verbose
        self.cohort_cfg = (cohort_min, cohort_size)
        self.pools: List[Pool] = ([dev_rotations(C.SEED, cohort_min, cohort_size)]
                                  + bkg_folds(n_folds, C.SEED, cohort_min, cohort_size))
        self._feat_raw: Dict[str, Dict[str, torch.Tensor]] = {}
        self._mats: Dict[Tuple, List[List[Mats]]] = {}

    # -- features ------------------------------------------------------------
    def raw_features(self, fc: FeatureConfig) -> Dict[str, torch.Tensor]:
        if fc.tag not in self._feat_raw:
            self._feat_raw[fc.tag] = cache.build(fc, verbose=self.verbose)
        return self._feat_raw[fc.tag]

    def features_for(self, fc: FeatureConfig, pool: Pool,
                     transform: str = "") -> Dict[str, torch.Tensor]:
        """Features for one pool; every fitted statistic uses pool.ubm_files."""
        f = self.raw_features(fc)
        if fc.cmvn == "global":
            f, _ = NZ.fit_and_apply(f, pool.ubm_files)
        if transform.startswith("seg:"):
            # text-dependent temporal segment i of n: retained frames of each
            # utterance split into n equal consecutive parts (segment index from
            # the frame's position among the VAD-retained frames)
            i, n = (int(v) for v in transform[4:].split("/"))
            out = {}
            for k, v in f.items():
                T = v.shape[0]
                pos = torch.div(torch.arange(T) * n, max(T, 1), rounding_mode="floor")
                out[k] = v[pos == i]
            return out
        if transform.startswith("bands:"):
            # sub-band stream: keep (cos, sin, dcos, dsin) of bands lo..hi-1 of a
            # 32-band IFD layout [cos | sin | dcos | dsin]
            lo, hi = (int(v) for v in transform[6:].split("-"))
            nb = fc.n_bands
            idx = torch.cat([torch.arange(q * nb + lo, q * nb + hi) for q in range(4)])
            return {k: v[:, idx] for k, v in f.items()}
        if transform:
            from . import transforms as TR
            kind, _, dim = transform.partition(":")
            spk = data.speaker_files([r for r in data.background()
                                      if r["file_id"] in set(pool.ubm_files)])
            m, W = TR.fit_lda(f, spk, kind.split("_")[1], int(dim or 0))
            f = TR.apply(f, m, W)
            f, _ = NZ.fit_and_apply(f, pool.ubm_files)
        return f

    # -- UBM -----------------------------------------------------------------
    def ubm(self, fc: FeatureConfig, gc: GMMConfig, pool: Pool,
            feats: Dict[str, torch.Tensor], ftag: str = "") -> GMM:
        tag = (f"benchubm_{pool.name if pool.name.startswith('bkg') else 'main'}"
               f"_{ftag or fc.tag}_K{gc.n_components}_{gc.cov_type}_vf{gc.var_floor:g}"
               f"_s{gc.seed}.pt")
        path = os.path.join(C.CACHE_DIR, tag)
        if os.path.exists(path):
            return GMM.from_state_dict(torch.load(path, map_location="cpu",
                                                  weights_only=False))
        X = cache.pooled(feats, pool.ubm_files)
        dt = torch.float64 if gc.dtype == "float64" else torch.float32
        m = GMM(gc.n_components, X.shape[1], gc.cov_type, dtype=dt)
        m.fit(X, gc)
        torch.save(m.state_dict(), path)
        return m

    # -- matrices ------------------------------------------------------------
    def mats(self, spec: SystemSpec, need_cohort: bool = True) -> List[List[Mats]]:
        key = (spec.ftag, spec.gmm.n_components, spec.gmm.cov_type,
               spec.gmm.var_floor, spec.gmm.seed, spec.relevance, need_cohort,
               self.cohort_cfg, spec.r_var)
        if key in self._mats:
            return self._mats[key]
        out = []
        for pool in self.pools:
            feats = self.features_for(spec.feature, pool, spec.transform)
            ubm = self.ubm(spec.feature, spec.gmm, pool, feats, spec.ftag)
            fs = FastScorer(ubm)
            coh_utts = [f for v in pool.cohort.values() for f in v]
            def adapt(emap):
                if spec.r_var:
                    mu, var, keys = map_mean_var(ubm, feats, emap, spec.relevance,
                                                 spec.r_var)
                    return mu, var, keys
                mu, keys = map_means(ubm, feats, emap, spec.relevance)
                return mu, None, keys

            if need_cohort:
                cmeans, cvar, _ = adapt(pool.cohort)
                Ct_all = fs.matrix(cmeans, feats, pool.all_files, var=cvar)
                fidx_all = {f: j for j, f in enumerate(pool.all_files)}
            rl = []
            for rot in pool.rotations:
                means, var, keys = adapt(rot.emap)
                S = fs.matrix(means, feats, rot.test_files, var=var)
                fi = {f: j for j, f in enumerate(rot.test_files)}
                mi = {k: i for i, k in enumerate(keys)}
                if need_cohort:
                    cols = torch.tensor([fidx_all[f] for f in rot.test_files])
                    Ct = Ct_all[:, cols]
                    Es = fs.matrix(means, feats, coh_utts, var=var)
                else:
                    Ct = Es = torch.zeros(0)
                rl.append(Mats(S, Ct, Es, None, mi, fi))
            out.append(rl)
        self._mats[key] = out
        return out

    # -- reverse (test-adapted) scoring ---------------------------------------
    def rev_mats(self, spec: SystemSpec, test_r: float = 0.0) -> List[List[Mats]]:
        """Reverse LLR matrices: rows = speakers' pooled enrolment frames,
        columns = models MAP-adapted on single test utterances.

        S[i, j]  = avg per-frame LLR of target i's enrolment frames under the
                   model adapted on test file j;
        Ct[c, j] = the same for cohort speaker c (utterance-side cohort);
        Es[i, u] = target i's frames under models adapted on cohort utterances.
        """
        r_test = test_r or spec.relevance
        key = ("rev", spec.ftag, spec.gmm.n_components, spec.gmm.cov_type,
               spec.gmm.var_floor, spec.gmm.seed, spec.relevance, r_test,
               self.cohort_cfg)
        if key in self._mats:
            return self._mats[key]
        out = []
        for pool in self.pools:
            feats = self.features_for(spec.feature, pool, spec.transform)
            ubm = self.ubm(spec.feature, spec.gmm, pool, feats, spec.ftag)
            fs = FastScorer(ubm)
            coh_utts = [f for v in pool.cohort.values() for f in v]
            single = {f: [f] for f in pool.all_files}
            fmeans, fkeys = map_means(ubm, feats, single, r_test)
            fidx = {k: i for i, k in enumerate(fkeys)}
            umeans, _ = map_means(ubm, feats, {f: [f] for f in coh_utts}, r_test)

            def rows_vs(models, spk2files):
                keys = sorted(spk2files)
                R = torch.zeros(len(keys), models.shape[0], dtype=torch.float64)
                for i, s in enumerate(keys):
                    X = torch.cat([feats[f] for f in spk2files[s]], 0)
                    if X.shape[0] == 0:
                        continue
                    u = fs.ubm_frame_ll(X)
                    for a in range(0, models.shape[0], 64):
                        ll = fs.frame_ll(X, models[a:a + 64].to(fs.dtype))
                        R[i, a:a + 64] = (ll - u.unsqueeze(1)).mean(0).double()
                return R, keys

            Ct_all, _ = rows_vs(fmeans, pool.cohort)
            rl = []
            for rot in pool.rotations:
                cols = torch.tensor([fidx[f] for f in rot.test_files])
                S_all, keys = rows_vs(fmeans[cols], rot.emap)
                Es, _ = rows_vs(umeans, rot.emap)
                mi = {k: i for i, k in enumerate(keys)}
                fi = {f: j for j, f in enumerate(rot.test_files)}
                rl.append(Mats(S_all, Ct_all[:, cols], Es, None, mi, fi))
            out.append(rl)
        self._mats[key] = out
        return out

    # -- evaluation ----------------------------------------------------------
    def evaluate(self, mats: List[List[Mats]],
                 fn: Callable[[Mats], torch.Tensor]) -> Dict[str, float]:
        """Pooled EER per protocol group + the official-dev EER."""
        res = {}
        groups = {"devrot": [], "bkgcv": []}
        for pool, rl in zip(self.pools, mats):
            for r_i, (rot, m) in enumerate(zip(pool.rotations, rl)):
                Sn = fn(m)
                rows = torch.tensor([m.mi[t["claimed_id"]] for t in rot.trials])
                cols = torch.tensor([m.fi[t["test_file"]] for t in rot.trials])
                s = Sn[rows, cols].double()
                y = [int(t["label"]) for t in rot.trials]
                g = "devrot" if pool.name == "devrot" else "bkgcv"
                groups[g].append((s, y))
                if pool.name == "devrot" and r_i == 0:
                    res["dev"] = M.eer(s, y)[0]
        for g, lst in groups.items():
            s = torch.cat([a for a, _ in lst])
            y = [v for _, b in lst for v in b]
            res[g] = M.eer(s, y)[0]
            res[g + "_mean"] = sum(M.eer(a, b)[0] for a, b in lst) / len(lst)
            res[g + "_auc"] = M.auc(s, y)
        res["score"] = 0.5 * (res["devrot"] + res["bkgcv"])
        return res


def fmt(r: Dict[str, float]) -> str:
    return (f"dev={r['dev']*100:5.2f}%  devrot={r['devrot']*100:5.2f}% "
            f"(rotmean {r['devrot_mean']*100:5.2f}%)  bkgcv={r['bkgcv']*100:5.2f}% "
            f"(rotmean {r['bkgcv_mean']*100:5.2f}%)  ==> {r['score']*100:5.2f}%")


# ---------------------------------------------------------------------------
# fusion and closed-set competitor penalty
# ---------------------------------------------------------------------------


def competitor_penalty(Sn: torch.Tensor, lam: float = 0.25,
                       delta: float = -1.0) -> torch.Tensor:
    """Penalise a trial when another *enrolled* model of the same split explains
    the test utterance better than the claimed one.

    ``Sn`` is a normalised (targets x files) matrix.  For entry (i, j) let
    ``c_ij = max_{i' != i} Sn[i', j]``; the returned score is
    ``Sn - lam * relu(c_ij - Sn + delta)``.  Uses only the test utterance and
    the enrolled models -- the same information as a T-norm whose cohort is
    the other enrolled speakers.
    """
    if Sn.shape[0] < 2 or lam == 0:
        return Sn
    top2 = torch.topk(Sn, 2, dim=0).values
    other = torch.where(Sn >= top2[0:1], top2[1:2], top2[0:1])
    return Sn - lam * torch.relu(other - Sn + delta)


def fused_fn(scheme: str, top: int, weights: Optional[Sequence[float]] = None,
             lam: float = 0.0, delta: float = -1.0, **kw):
    """Build an ``evaluate`` callback over a *list* of Mats (one per system)."""
    def fn(ms: Sequence[Mats]) -> torch.Tensor:
        w = list(weights) if weights is not None else [1.0] * len(ms)
        out = sum(wi * normalise(m, scheme, top, **kw) for wi, m in zip(w, ms))
        out = out / sum(w)
        return competitor_penalty(out, lam, delta) if lam else out
    return fn


def zip_mats(all_mats: Sequence[List[List[Mats]]]) -> List[List[List[Mats]]]:
    """[system][pool][rot] -> [pool][rot][system] (each element: list of Mats)."""
    return [[list(ms) for ms in zip(*pools)] for pools in zip(*all_mats)]


class _Multi:
    """Adapter so Bench.evaluate can read mi/fi from the first system."""

    def __init__(self, ms: List[Mats]):
        self.ms = ms
        self.mi, self.fi = ms[0].mi, ms[0].fi


def evaluate_fusion(bench: "Bench", all_mats, fn) -> Dict[str, float]:
    z = zip_mats(all_mats)
    wrapped = [[_Multi(ms) for ms in pool] for pool in z]
    return bench.evaluate(wrapped, lambda mm: fn(mm.ms))


def average_mats(all_mats: Sequence[List[List[Mats]]]) -> List[List[Mats]]:
    """Element-wise mean of the matrices of several systems that share trials
    (used to combine temporal-segment systems into one segmented system:
    with equal-size segments this is the per-frame average LLR)."""
    out = []
    for pools in zip(*all_mats):
        rl = []
        for ms in zip(*pools):
            n = len(ms)
            avg = lambda name: (None if getattr(ms[0], name) is None else
                                sum(getattr(m, name) for m in ms) / n)
            rl.append(Mats(avg("S"), avg("Ct"), avg("Es"), avg("Cc"), ms[0].mi, ms[0].fi))
        out.append(rl)
    return out
