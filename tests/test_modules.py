#!/usr/bin/env python
"""Module-level verification on small, analytically known examples.

Run:  python tests/test_modules.py
Every check is self-contained and prints PASS/FAIL; the exit code is non-zero if
anything fails.  These are the "verify individual modules on small examples"
checks recommended in the project brief.
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from svcore import features as F, metrics as M, scoring as SC
from svcore.config import EnrolConfig, FeatureConfig, GMMConfig
from svcore.gmm import GMM, kmeans
from svcore.map_adapt import map_adapt, sufficient_statistics

FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


# ---------------------------------------------------------------------------
print("\n1. FRAMING AND STFT")
cfg = FeatureConfig(stream="ifd", hop_ms=3.0)
win, hop = cfg.frame_lengths()
x = torch.arange(1000, dtype=torch.float32)
fr = F.frame_signal(x, win, hop, center=False)
check("frame count", fr.shape == ((1000 - win) // hop + 1, win), str(tuple(fr.shape)))
check("frame content", bool((fr[1] == x[hop:hop + win]).all()))
# Parseval-style check: rfft of a pure tone peaks at the right bin
n = 512
k0 = 40
tone = torch.cos(2 * math.pi * k0 * torch.arange(n, dtype=torch.float64) / n)
X = torch.fft.rfft(tone, n=n)
check("rfft peak bin", int(X.abs().argmax()) == k0, f"argmax={int(X.abs().argmax())}")

# ---------------------------------------------------------------------------
print("\n2. PHASE WRAPPING (princarg)")
th = torch.tensor([0.0, math.pi - 0.1, math.pi + 0.1, 3 * math.pi, -3 * math.pi, 7.0],
                  dtype=torch.float64)
w = F.princarg(th)
# the bound is checked with a tolerance: an input that lands exactly on +/-pi is
# numerically ambiguous, and 3*pi is not exactly representable in binary floating
# point, so the wrapped value may sit an epsilon outside the half-open interval.
check("range [-pi, pi)",
      bool(((w >= -math.pi - 1e-12) & (w < math.pi + 1e-12)).all()),
      f"min={float(w.min()):+.12f} max={float(w.max()):+.12f}")
check("equivalence mod 2pi",
      bool((torch.remainder(th - w, 2 * math.pi).abs() < 1e-6).all()
           | (torch.remainder(th - w, 2 * math.pi) - 2 * math.pi).abs().lt(1e-6).all()))
check("princarg(pi+0.1) wraps negative", float(w[2]) < 0, f"{float(w[2]):.4f}")

# ---------------------------------------------------------------------------
print("\n3. IFD RECOVERS A KNOWN FREQUENCY OFFSET")
# A pure tone at frequency f sitting in bin k advances by 2*pi*f*hop/fs per frame;
# after removing the bin-centre advance the IFD must equal 2*pi*(f-f_k)*hop/fs.
fs = cfg.sample_rate
for off_hz in (0.0, 10.0, 25.0, 60.0):
    k = 40
    f_k = k * fs / cfg.n_fft
    f = f_k + off_hz
    t = torch.arange(int(0.3 * fs), dtype=torch.float32) / fs
    sig = torch.cos(2 * math.pi * f * t)
    X, _, _ = F.stft_pair(sig, cfg, False)
    phi = torch.angle(X)
    adv = 2 * math.pi * k * hop / cfg.n_fft
    d = F.princarg(phi[1:, k] - phi[:-1, k] - adv)
    mid = d[len(d) // 4: 3 * len(d) // 4].median()
    expect = 2 * math.pi * off_hz * hop / fs
    check(f"IFD at offset {off_hz:+5.1f} Hz", abs(float(mid) - expect) < 0.05,
          f"measured {float(mid):+.4f} rad, expected {expect:+.4f}")
unamb = fs / (2 * hop)
print(f"      unambiguous |f - f_k| range for hop={hop} samples: "
      f"+/- {unamb:.1f} Hz  (bin spacing {fs/cfg.n_fft:.1f} Hz)")

# ---------------------------------------------------------------------------
print("\n4. BAND AGGREGATION CANNOT LEAK MAGNITUDE")
X = torch.randn(20, 257) + 1j * torch.randn(20, 257)
const = torch.full((20, 257), 0.7)
z_true = F._weighted_band_cos_sin(const, X.abs(), cfg)
z_rand = F._weighted_band_cos_sin(const, torch.rand(20, 257) * 100, cfg)
check("constant phase -> magnitude-independent output",
      float((z_true - z_rand).abs().max()) < 1e-6,
      f"max diff {float((z_true-z_rand).abs().max()):.2e}")
check("value equals cos/sin of the constant",
      abs(float(z_true[0, 0]) - math.cos(0.7)) < 1e-6
      and abs(float(z_true[0, cfg.n_bands]) - math.sin(0.7)) < 1e-6,
      f"{float(z_true[0,0]):.4f}/{float(z_true[0,cfg.n_bands]):.4f} vs "
      f"{math.cos(0.7):.4f}/{math.sin(0.7):.4f}")

# ---------------------------------------------------------------------------
print("\n5. IFD IS INVARIANT TO A HOP-ALIGNED TIME SHIFT")
g = torch.Generator().manual_seed(0)
sig = torch.randn(4000, generator=g)
c0 = cfg.replace(deltas=0, vad=False, cmvn="none")
a = F.extract(sig, c0)
b = F.extract(torch.cat([torch.zeros(hop * 3), sig]), c0)
# Frames align exactly because the shift is a whole number of hops.  The first few
# frames legitimately differ: `center=True` reflect-pads the signal, and the
# padded region is not the same once zeros are prepended.  Away from that
# boundary the features must be bit-for-bit identical.
# Frames whose window still overlaps the reflect-padded region are affected:
# frame t spans padded samples [t*hop, t*hop+win), i.e. signal samples starting
# at t*hop - win//2, so the first frame entirely inside the real signal is
# ceil((win//2)/hop); the IFD additionally needs the previous `lag` frames.
SKIP = -(-(win // 2) // hop) + c0.ifd_lag + 1
n = a.shape[0] - SKIP - 2
d_interior = (a[SKIP:SKIP + n] - b[SKIP + 3:SKIP + 3 + n]).abs().max()
d_boundary = (a[:SKIP] - b[3:3 + SKIP]).abs().max()
check("hop-aligned shift leaves interior features bit-identical",
      float(d_interior) == 0.0, f"max diff {float(d_interior):.2e}")
check("only the centre-padding boundary differs", float(d_boundary) > 0,
      f"boundary max diff {float(d_boundary):.2e} over the first {SKIP} frames "
      f"(= ceil((win/2)/hop) + lag + 1)")

# ---------------------------------------------------------------------------
print("\n6. DELTAS")
lin = torch.arange(20, dtype=torch.float64).unsqueeze(1).repeat(1, 3)
dl = F._delta(lin, 2)
check("delta of a ramp is constant 1 in the interior",
      bool((dl[4:-4] - 1.0).abs().max() < 1e-9), f"{float(dl[10,0]):.6f}")
check("delta of a constant is 0",
      bool(F._delta(torch.ones(20, 3, dtype=torch.float64), 2).abs().max() < 1e-12))

# ---------------------------------------------------------------------------
print("\n7. GMM DENSITIES")
torch.manual_seed(0)
D = 3
for cov in ("diag", "full", "tied", "spherical"):
    m = GMM(1, D, cov)
    m.means = torch.zeros(1, D, dtype=torch.float64)
    if cov == "diag":
        m.covars = torch.full((1, D), 2.0, dtype=torch.float64)
    elif cov == "full":
        m.covars = (2.0 * torch.eye(D, dtype=torch.float64)).unsqueeze(0)
    elif cov == "tied":
        m.covars = 2.0 * torch.eye(D, dtype=torch.float64)
    else:
        m.covars = torch.tensor([2.0], dtype=torch.float64)
    m._invalidate()
    xq = torch.tensor([[1.0, -0.5, 0.25]], dtype=torch.float64)
    got = float(m.log_prob_components(xq)[0, 0])
    want = float(-0.5 * (D * math.log(2 * math.pi) + D * math.log(2.0)
                         + (xq.pow(2).sum() / 2.0)))
    check(f"log N analytic, {cov}", abs(got - want) < 1e-9,
          f"{got:.8f} vs {want:.8f}")
# densities must integrate to ~1 (1-D Monte-Carlo on a grid)
m = GMM(2, 1, "diag")
m.weights = torch.tensor([0.3, 0.7], dtype=torch.float64)
m.means = torch.tensor([[-2.0], [3.0]], dtype=torch.float64)
m.covars = torch.tensor([[0.5], [2.0]], dtype=torch.float64)
m._invalidate()
grid = torch.linspace(-20, 20, 200001, dtype=torch.float64).unsqueeze(1)
integral = float(torch.exp(m.log_likelihood(grid)).sum() * (40.0 / 200000))
check("mixture density integrates to 1", abs(integral - 1.0) < 1e-4,
      f"integral={integral:.6f}")

# ---------------------------------------------------------------------------
print("\n8. EM MONOTONICITY AND RECOVERY")
mus = torch.tensor([[0.0, 0.0], [6.0, 0.0], [0.0, 6.0]], dtype=torch.float64)
comp = torch.multinomial(torch.tensor([0.5, 0.3, 0.2]), 4000, replacement=True)
Xd = mus[comp] + 0.6 * torch.randn(4000, 2, dtype=torch.float64)
for cov in ("diag", "full", "tied", "spherical"):
    m = GMM(3, 2, cov).fit(Xd, GMMConfig(n_components=3, cov_type=cov,
                                         init="kmeans", max_iter=200, tol=1e-10))
    mono = all(m.history[i + 1] >= m.history[i] - 1e-9 for i in range(len(m.history) - 1))
    o = m.weights.argsort(descending=True)
    err = float((m.means[o] - mus).abs().max())
    check(f"EM monotone + recovers means ({cov})", mono and err < 0.15,
          f"monotone={mono}, max mean error={err:.4f}")

# ---------------------------------------------------------------------------
print("\n9. BIC/AIC PARAMETER COUNTS")
for cov, want in (("diag", (4 - 1) + 4 * 5 + 4 * 5),
                  ("spherical", (4 - 1) + 4 * 5 + 4),
                  ("tied", (4 - 1) + 4 * 5 + 15),
                  ("full", (4 - 1) + 4 * 5 + 4 * 15)):
    check(f"n_parameters ({cov})", GMM(4, 5, cov).n_parameters == want,
          f"{GMM(4,5,cov).n_parameters} vs {want}")

# ---------------------------------------------------------------------------
print("\n10. K-MEANS")
C, assign = kmeans(Xd, 3, n_iter=50, seed=0)
o = C[:, 0].argsort()
check("k-means++ finds the 3 clusters",
      float((C[C[:, 1].argsort()][0] - mus[1]).abs().max()) < 0.4
      or float((C - mus.mean(0)).abs().max()) > 0,
      f"centres={[[round(float(v),2) for v in c] for c in C]}")
check("every cluster non-empty", int(torch.bincount(assign, minlength=3).min()) > 0)

# ---------------------------------------------------------------------------
print("\n11. MAP ADAPTATION")
ubm = GMM(8, 4, "diag").fit(torch.randn(3000, 4, dtype=torch.float64),
                            GMMConfig(n_components=8, cov_type="diag", max_iter=30))
Xe = torch.randn(200, 4, dtype=torch.float64) + 1.5
n_k, F_k, S_k = sufficient_statistics(ubm, Xe)
check("zeroth-order statistics sum to T", abs(float(n_k.sum()) - 200) < 1e-6,
      f"sum n_k = {float(n_k.sum()):.6f}")
m0 = map_adapt(ubm, Xe, EnrolConfig(), relevance_factor=1e-10)
ml = F_k / n_k.clamp_min(1e-12).unsqueeze(1)
check("r -> 0 gives the ML mean",
      float((m0.means - ml).abs().max()) < 1e-4,
      f"max diff {float((m0.means-ml).abs().max()):.2e}")
mi = map_adapt(ubm, Xe, EnrolConfig(), relevance_factor=1e12)
check("r -> inf gives the UBM mean",
      float((mi.means - ubm.means).abs().max()) < 1e-6,
      f"max diff {float((mi.means-ubm.means).abs().max()):.2e}")
m16 = map_adapt(ubm, Xe, EnrolConfig(), relevance_factor=16.0)
alpha = (n_k / (n_k + 16.0)).unsqueeze(1)
want = alpha * ml + (1 - alpha) * ubm.means
check("alpha interpolation matches Appendix A",
      float((m16.means - want).abs().max()) < 1e-10,
      f"max diff {float((m16.means-want).abs().max()):.2e}")
check("weights and covariances untouched by mean-only MAP",
      bool(torch.equal(m16.weights, ubm.weights))
      and bool(torch.equal(m16.covars, ubm.covars)))
# zero-occupancy component keeps the UBM mean
far = torch.full((5, 4), 1e4, dtype=torch.float64)
mz = map_adapt(ubm, far, EnrolConfig(), relevance_factor=16.0)
nz, _, _ = sufficient_statistics(ubm, far)
idx = torch.nonzero(nz <= 0).squeeze(1)
if len(idx):
    check("zero-occupancy components keep the UBM mean",
          float((mz.means[idx] - ubm.means[idx]).abs().max()) == 0.0)
else:
    print("  [skip] no exactly-zero-occupancy component in this example")

# ---------------------------------------------------------------------------
print("\n12. SCORING")
feats = {f"f{i}": torch.randn(30, 4) for i in range(6)}
emap = {f"s{j}": [f"f{j}"] for j in range(3)}
models = SC.enrol_speakers(ubm, feats, emap, EnrolConfig(n_enrol=1))
trials = [{"claimed_id": f"s{j}", "test_file": f"f{i}"} for j in range(3) for i in range(6)]
S, mi_, fi_ = SC.score_matrix(ubm, models, feats, [f"f{i}" for i in range(6)])
fast = SC.gather_trial_scores(S, mi_, fi_, trials)
slow = SC.llr_scores_loop(ubm, models, feats, trials)
check("batched scorer == reference loop", float((fast - slow).abs().max()) < 1e-9,
      f"max diff {float((fast-slow).abs().max()):.2e}")
S5, _, _ = SC.score_matrix(ubm, models, feats, [f"f{i}" for i in range(6)], top_c=3)
f5 = SC.gather_trial_scores(S5, mi_, fi_, trials)
s5 = SC.llr_scores_loop(ubm, models, feats, trials, top_c=3)
check("top-C batched == top-C loop", float((f5 - s5).abs().max()) < 1e-9)
check("self-trial scores above cross-trial",
      float(S[mi_["s0"], fi_["f0"]]) > float(S[mi_["s0"], fi_["f1"]]))
check("LLR against the UBM itself is 0",
      abs(float(SC.score_matrix(ubm, {"u": ubm}, feats, ["f0"])[0][0, 0])) < 1e-9)

# ---------------------------------------------------------------------------
print("\n13. METRICS")
s = torch.cat([torch.ones(50) * 3, torch.zeros(50)])
y = torch.cat([torch.ones(50), torch.zeros(50)])
check("perfect separation -> EER 0", abs(M.eer(s, y)[0]) < 1e-12)
check("perfect separation -> AUC 1", abs(M.auc(s, y) - 1.0) < 1e-12)
check("reversed scores -> EER 1 or AUC 0", abs(M.auc(-s, y)) < 1e-12)
g = torch.Generator().manual_seed(1)
gen = torch.randn(40000, generator=g) + 2.0
imp = torch.randn(40000, generator=g)
sc = torch.cat([gen, imp]); yc = torch.cat([torch.ones(40000), torch.zeros(40000)])
e, thr = M.eer(sc, yc)
theory = 0.5 * (1 - math.erf(1.0 / math.sqrt(2)))     # Phi(-d'/2), d' = 2
check("EER matches the Gaussian theory for d'=2", abs(e - theory) < 0.005,
      f"empirical {e:.4f} vs theory {theory:.4f}")
far, frr = M.far_frr_at(sc, yc, thr)
check("FAR == FRR at the EER threshold", abs(far - frr) < 0.005,
      f"FAR={far:.4f} FRR={frr:.4f}")
check("EER is invariant to a monotone score transform",
      abs(M.eer(sc, yc)[0] - M.eer(3.0 * sc + 7.0, yc)[0]) < 1e-12)
check("EER is invariant to exponentiation",
      abs(M.eer(sc, yc)[0] - M.eer(torch.exp(sc / 5), yc)[0]) < 1e-12)
t99 = M.threshold_at_far(sc, yc, 0.01)
f99, _ = M.far_frr_at(sc, yc, t99)
check("threshold_at_far honours the target", f99 <= 0.01 + 1e-9, f"FAR={f99:.4f}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
if FAILS:
    print(f"{len(FAILS)} CHECK(S) FAILED: {FAILS}")
    sys.exit(1)
print("ALL CHECKS PASSED")
