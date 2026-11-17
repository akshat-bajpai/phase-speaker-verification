"""Phase-based feature extraction, implemented with **torch + torchaudio only**.

Design notes (see the report for the full discussion)
-----------------------------------------------------
The organisers prepended 0-30 ms of random leading audio to every clip.  A time
shift of ``d`` samples multiplies bin ``k`` of the DFT by ``exp(-j 2 pi k d / N)``,
so the *absolute* phase spectrum is effectively randomised per utterance.  Every
feature here is therefore built from a phase representation that is invariant (or
made invariant) to the time origin:

* ``ifd``      temporal phase difference / baseband phase difference.  The
               ``exp(-j2 pi k d/N)`` term is identical in consecutive frames and
               cancels exactly in the difference -> exactly shift invariant.
* ``mgdcc``    modified group delay ``tau(w) = -dphi/dw``.  A shift adds the
               constant ``d`` to ``tau`` at every frequency, which we remove with
               a magnitude-weighted per-frame mean before compression.
* ``fdpd``     frequency-direction phase difference (finite-difference group
               delay), likewise centred by its per-frame circular mean.
* ``cosphase`` cosine of the raw phase spectrum.  Kept deliberately as a
               *negative control*: it is not shift invariant and is expected to
               perform near chance.

Role of magnitude (permitted by the rules, documented here):
  1. frame energy for the energy VAD;
  2. a *reliability weight* when averaging phase quantities inside a
     filterbank band -- phase in a near-empty bin is numerically meaningless;
  3. the smoothed denominator ``|S(w)|^{2 gamma}`` that is part of the standard
     modified-group-delay definition.
Magnitude is never used as a model input feature.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torchaudio

from .config import FeatureConfig

EPS = 1e-10


# ---------------------------------------------------------------------------
# cached front-end matrices
# ---------------------------------------------------------------------------
_CACHE: Dict[tuple, torch.Tensor] = {}


def _window(name: str, win: int, dtype, device) -> torch.Tensor:
    key = ("win", name, win, dtype, device)
    if key not in _CACHE:
        if name == "hann":
            w = torch.hann_window(win, periodic=False, dtype=dtype, device=device)
        elif name == "hamming":
            w = torch.hamming_window(win, periodic=False, dtype=dtype, device=device)
        elif name == "blackman":
            w = torch.blackman_window(win, periodic=False, dtype=dtype, device=device)
        elif name == "rect":
            w = torch.ones(win, dtype=dtype, device=device)
        else:
            raise ValueError(f"unknown window {name}")
        _CACHE[key] = w
    return _CACHE[key]


def _fbank(cfg: FeatureConfig, n_freq: int, dtype, device) -> Optional[torch.Tensor]:
    """(n_bands, n_freq) band-aggregation weights, rows summing to 1."""
    if cfg.band_type == "none":
        return None
    key = ("fb", cfg.band_type, cfg.n_bands, n_freq, cfg.f_min, cfg.f_max,
           cfg.sample_rate, dtype, device)
    if key not in _CACHE:
        if cfg.band_type == "mel":
            fb = torchaudio.functional.melscale_fbanks(
                n_freqs=n_freq, f_min=cfg.f_min, f_max=cfg.f_max,
                n_mels=cfg.n_bands, sample_rate=cfg.sample_rate, norm=None,
                mel_scale="htk").t()                       # (n_bands, n_freq)
        elif cfg.band_type == "linear":
            edges = torch.linspace(cfg.f_min, cfg.f_max, cfg.n_bands + 1)
            freqs = torch.linspace(0, cfg.sample_rate / 2, n_freq)
            fb = torch.stack([((freqs >= edges[b]) & (freqs < edges[b + 1])).to(torch.float32)
                              for b in range(cfg.n_bands)])
        else:
            raise ValueError(cfg.band_type)
        fb = fb.to(dtype=dtype, device=device)
        fb = fb / fb.sum(dim=1, keepdim=True).clamp_min(EPS)
        _CACHE[key] = fb
    return _CACHE[key]


def _dct_matrix(n_out: int, n_in: int, dtype, device) -> torch.Tensor:
    """Orthonormal DCT-II matrix of shape (n_in, n_out)."""
    key = ("dct", n_out, n_in, dtype, device)
    if key not in _CACHE:
        m = torchaudio.functional.create_dct(n_mfcc=n_out, n_mels=n_in, norm="ortho")
        _CACHE[key] = m.to(dtype=dtype, device=device)
    return _CACHE[key]


# ---------------------------------------------------------------------------
# framing + STFT (hand-rolled so that the n*x[n] transform needed by the
# group-delay features shares exactly the same framing)
# ---------------------------------------------------------------------------


def frame_signal(x: torch.Tensor, win: int, hop: int, center: bool = True) -> torch.Tensor:
    """(num_samples,) -> (n_frames, win) with reflect padding when ``center``."""
    if center:
        pad = win // 2
        x = torch.nn.functional.pad(x.view(1, 1, -1), (pad, pad), mode="reflect").view(-1)
    if x.numel() < win:
        x = torch.nn.functional.pad(x, (0, win - x.numel()))
    return x.unfold(0, win, hop)


def stft_pair(x: torch.Tensor, cfg: FeatureConfig, need_ramp: bool
              ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Return ``(X, Y, frame_logE)``.

    ``X`` is the complex STFT ``(n_frames, n_freq)``; ``Y`` is the STFT of
    ``n * x[n]`` (same framing/window) used by the group-delay features; and
    ``frame_logE`` is the log frame energy used by the VAD.
    """
    win, hop = cfg.frame_lengths()
    if cfg.preemph > 0:
        x = torch.cat([x[:1], x[1:] - cfg.preemph * x[:-1]])
    frames = frame_signal(x, win, hop, cfg.center)                 # (T, win)
    w = _window(cfg.window, win, frames.dtype, frames.device)
    fw = frames * w
    logE = torch.log10(fw.pow(2).sum(dim=1) + EPS) * 10.0          # dB
    X = torch.fft.rfft(fw, n=cfg.n_fft, dim=1)
    Y = None
    if need_ramp:
        ramp = torch.arange(win, dtype=frames.dtype, device=frames.device)
        Y = torch.fft.rfft(fw * ramp, n=cfg.n_fft, dim=1)
    return X, Y, logE


# ---------------------------------------------------------------------------
# VAD (magnitude used here only)
# ---------------------------------------------------------------------------


def vad_mask(logE: torch.Tensor, cfg: FeatureConfig) -> torch.Tensor:
    """Boolean keep-mask over frames: energy within ``vad_rel_db`` of the peak."""
    if not cfg.vad:
        return torch.ones_like(logE, dtype=torch.bool)
    thr = logE.max() - cfg.vad_rel_db
    mask = logE > thr
    n_keep = int(mask.sum())
    if n_keep < min(cfg.vad_min_frames, logE.numel()):
        # degenerate clip: fall back to the loudest ``vad_min_frames`` frames
        k = min(cfg.vad_min_frames, logE.numel())
        idx = torch.topk(logE, k).indices
        mask = torch.zeros_like(logE, dtype=torch.bool)
        mask[idx] = True
    return mask


# ---------------------------------------------------------------------------
# phase helpers
# ---------------------------------------------------------------------------


def princarg(theta: torch.Tensor) -> torch.Tensor:
    """Wrap to [-pi, pi) -- the principal-argument operator."""
    return theta - 2.0 * math.pi * torch.floor(theta / (2.0 * math.pi) + 0.5)


def _weighted_band_cos_sin(theta: torch.Tensor, weight: torch.Tensor,
                           cfg: FeatureConfig) -> torch.Tensor:
    """Magnitude-weighted circular mean of ``theta`` inside each band.

    ``theta`` (T, F) is a wrapped phase quantity; ``weight`` (T, F) >= 0 is the
    reliability weight.  Returns ``(T, 2*n_bands)`` = [cos-part | sin-part],
    i.e. the real and imaginary parts of the band-wise mean resultant vector.
    Averaging ``e^{j theta}`` rather than ``theta`` is what makes the aggregation
    immune to phase wrapping.
    """
    fb = _fbank(cfg, theta.shape[1], theta.dtype, theta.device)
    c, s = torch.cos(theta), torch.sin(theta)
    if fb is None:
        return torch.cat([c, s], dim=1)
    num_c = (c * weight) @ fb.t()
    num_s = (s * weight) @ fb.t()
    den = weight @ fb.t()
    den = den.clamp_min(EPS)
    return torch.cat([num_c / den, num_s / den], dim=1)


def _dct_reduce(z: torch.Tensor, cfg: FeatureConfig, n_sub: int = 1) -> torch.Tensor:
    """DCT-II along frequency, keeping ``n_ceps`` coefficients per sub-stream."""
    if cfg.n_ceps <= 0:
        return z
    chunks = torch.chunk(z, n_sub, dim=1)
    out = []
    for ch in chunks:
        n_in = ch.shape[1]
        n_out = min(cfg.n_ceps, n_in)
        d = _dct_matrix(n_out, n_in, ch.dtype, ch.device)
        c = ch @ d
        out.append(c if cfg.keep_c0 else c[:, 1:])
    return torch.cat(out, dim=1)


def _cepstral_smooth(log_mag: torch.Tensor, lifter: int) -> torch.Tensor:
    """Low-time-liftered (cepstrally smoothed) log magnitude, via ortho DCT."""
    n_in = log_mag.shape[1]
    d = _dct_matrix(n_in, n_in, log_mag.dtype, log_mag.device)   # (n_in, n_in)
    c = log_mag @ d
    c = torch.cat([c[:, :lifter], torch.zeros_like(c[:, lifter:])], dim=1)
    return c @ d.t()


# ---------------------------------------------------------------------------
# the five phase streams -> static features
# ---------------------------------------------------------------------------


def _stream_ifd(X, Y, cfg) -> torch.Tensor:
    """Instantaneous-frequency deviation = temporal (baseband) phase difference.

    ``Delta(t,k) = princarg(phi(t,k) - phi(t-L,k) - 2 pi k L hop / n_fft)`` for a
    lag of ``L = cfg.ifd_lag`` frames, so the *baseline* of the difference
    (``L * hop``) can be varied independently of the frame rate.
    The expected per-frame phase advance of bin ``k`` is removed first (the
    "baseband" step), so ``Delta`` measures how far the local partial deviates
    from the bin centre frequency.  Any constant time shift of the waveform
    contributes the same ``-2 pi k d / N`` to both frames and cancels exactly.
    """
    _, hop = cfg.frame_lengths()
    lag = max(1, cfg.ifd_lag)
    phi = torch.angle(X)
    k = torch.arange(X.shape[1], dtype=phi.dtype, device=phi.device)
    advance = 2.0 * math.pi * k * (hop * lag) / cfg.n_fft
    d = princarg(phi[lag:] - phi[:-lag] - advance)
    d = torch.cat([d[:1].expand(lag, -1), d], dim=0)       # replicate at the start
    mag = X.abs()
    if cfg.mag_power != 1.0:
        mag = mag.pow(cfg.mag_power)
    w = mag if cfg.mag_weight else torch.ones_like(mag)
    if cfg.ifd_cossin:
        z = _weighted_band_cos_sin(d, w, cfg)
        return _dct_reduce(z, cfg, n_sub=2)
    fb = _fbank(cfg, d.shape[1], d.dtype, d.device)
    z = (d * w) @ fb.t() / (w @ fb.t()).clamp_min(EPS) if fb is not None else d
    return _dct_reduce(z, cfg, n_sub=1)


def _stream_mgdcc(X, Y, cfg) -> torch.Tensor:
    """Group-delay cepstral coefficients.

    The group delay is the derivative of the phase spectrum,
    ``tau(w) = -dphi/dw``, which for the windowed frame ``x`` equals

        tau(w) = (Xr Yr + Xi Yi) / |X(w)|^2 ,   Y = DFT{ n x[n] } .

    Two denominators are available:

    ``mgd_denom="exact"`` (default, used by the reported system)
        The exact ``|X(w)|^2``, i.e. the *true* group delay -- a pure function of
        the phase derivative with no magnitude envelope left in it.  Because the
        group delay is ill-conditioned where the spectrum is near zero, ``|X|^2``
        is floored at ``mgd_floor_db`` dB below the per-frame spectral peak.
        That floor is magnitude used purely as a numerical stabiliser.

    ``mgd_denom="smoothed"``
        The modified group delay of Hegde, Murthy & Gadde (2007), which replaces
        the denominator by a cepstrally smoothed magnitude ``|S(w)|^{2 gamma}``.
        Note that for ``gamma < 1`` this does *not* cancel the numerator's
        magnitude: ``tau_gamma ~ tau_true * |X|^{2-2gamma}``, so the spectral
        envelope is progressively reintroduced.  We therefore report it only as
        an ablation (see ``mgd_magnitude_only`` for the control that isolates
        exactly how much of its gain is magnitude rather than phase).

    After the derivative we remove the magnitude-weighted mean of ``tau`` over
    frequency -- this is the per-frame linear-phase (time-origin) term, so the
    representation becomes invariant to the organisers' random leading pad --
    then compress with ``sign(tau)|tau|^alpha``, aggregate inside mel bands and
    take a DCT.
    """
    mag2 = X.real.pow(2) + X.imag.pow(2)
    mag = mag2.clamp_min(0).sqrt()
    num = X.real * Y.real + X.imag * Y.imag
    if cfg.mgd_denom == "exact":
        floor = mag2.max(dim=1, keepdim=True).values * (10.0 ** (-cfg.mgd_floor_db / 10.0))
        den = torch.maximum(mag2, floor).clamp_min(EPS)
    elif cfg.mgd_denom == "smoothed":
        log_mag = torch.log(mag + EPS)
        smooth = torch.exp(_cepstral_smooth(log_mag, cfg.mgd_lifter)).clamp_min(EPS)
        den = smooth.pow(2.0 * cfg.mgd_gamma)
    else:
        raise ValueError(cfg.mgd_denom)
    if cfg.mgd_magnitude_only:
        # DIAGNOSTIC CONTROL (not a submitted system): keep the denominator's
        # effect but delete the phase derivative from the numerator, i.e. feed
        # |X|^2 / den through the identical shaping chain.  Any performance this
        # reaches is attributable to magnitude, not phase.
        tau = mag2 / den
    else:
        tau = num / den
    w = mag if cfg.mag_weight else torch.ones_like(mag)
    tau = tau - (tau * w).sum(1, keepdim=True) / w.sum(1, keepdim=True).clamp_min(EPS)
    tau_m = torch.sign(tau) * tau.abs().pow(cfg.mgd_alpha)
    fb = _fbank(cfg, tau_m.shape[1], tau_m.dtype, tau_m.device)
    if fb is not None:
        tau_m = (tau_m * w) @ fb.t() / (w @ fb.t()).clamp_min(EPS)
    return _dct_reduce(tau_m, cfg, n_sub=1)


def _stream_fdpd(X, Y, cfg) -> torch.Tensor:
    """Frequency-direction phase difference (finite-difference group delay).

    ``Delta_f(t,k) = princarg(phi(t,k+1) - phi(t,k))``, centred by its
    magnitude-weighted circular mean per frame (again removing the time-origin
    term), then aggregated as a circular mean inside each band.
    """
    phi = torch.angle(X)
    d = princarg(phi[:, 1:] - phi[:, :-1])
    mag = 0.5 * (X.abs()[:, 1:] + X.abs()[:, :-1])
    w = mag if cfg.mag_weight else torch.ones_like(mag)
    # magnitude-weighted circular mean per frame, subtracted off
    mc = (torch.cos(d) * w).sum(1, keepdim=True)
    ms = (torch.sin(d) * w).sum(1, keepdim=True)
    d = princarg(d - torch.atan2(ms, mc))
    z = _weighted_band_cos_sin(d, w, cfg)
    return _dct_reduce(z, cfg, n_sub=2)


def _stream_cosphase(X, Y, cfg) -> torch.Tensor:
    """Negative control: cosine of the *raw* phase spectrum + DCT.

    Bounded in [-1, 1] so phase wrapping is handled, but the representation
    carries the absolute time origin and is therefore scrambled by the
    organisers' random leading pad.
    """
    phi = torch.angle(X)
    mag = X.abs()
    w = mag if cfg.mag_weight else torch.ones_like(mag)
    z = _weighted_band_cos_sin(phi, w, cfg)
    return _dct_reduce(z, cfg, n_sub=2)


def instantaneous_frequency(X: torch.Tensor, cfg: FeatureConfig) -> torch.Tensor:
    """Per-bin instantaneous frequency in Hz from the temporal phase difference.

    ``IF(t,k) = (k + princarg(phi_t - phi_{t-1} - 2 pi k hop/N) * N/(2 pi hop)) fs/N``
    -- unambiguous for partials within ``fs / (2 hop)`` Hz of the bin centre.
    """
    _, hop = cfg.frame_lengths()
    N, fs = cfg.n_fft, cfg.sample_rate
    phi = torch.angle(X)
    k = torch.arange(X.shape[1], dtype=phi.dtype, device=phi.device)
    d = princarg(phi[1:] - phi[:-1] - 2.0 * math.pi * k * hop / N)
    d = torch.cat([d[:1], d], dim=0)
    return (k + d * N / (2.0 * math.pi * hop)) * fs / N


def phase_f0(X: torch.Tensor, cfg: FeatureConfig, f_lo: float = 50.0,
             f_hi: float = 2000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Phase-domain F0: a harmonic sieve on instantaneous frequencies.

    For a candidate ``c`` the coherence ``A(c) = <cos(2 pi IF/c)>_w`` is 1 when
    every reliable bin's IF sits on a multiple of ``c``; sub-octave candidates
    (``c = f0/2``) score equally, so we subtract ``B(c) = <cos(pi IF/c)>_w``,
    which is ~1 only for those.  ``w = |X|`` is a reliability weight (phase in
    near-empty bins is meaningless); the quantity being estimated is a
    function of phase only.  The grid estimate is refined by least squares
    ``IF_k ~ h_k f0``.  Returns ``(f0 (T,), coherence A (T,))``.
    """
    N, fs = cfg.n_fft, cfg.sample_rate
    IF = instantaneous_frequency(X, cfg)
    freqs = torch.arange(X.shape[1], dtype=IF.dtype) * fs / N
    band = (freqs >= f_lo) & (freqs <= f_hi)
    w = X.abs()[:, band]
    ifb = IF[:, band].clamp_min(1.0)
    wn = w / w.sum(1, keepdim=True).clamp_min(EPS)
    cand = torch.arange(cfg.rps_f0_min, cfg.rps_f0_max + 1.0, 1.0, dtype=IF.dtype)
    ratio = ifb.unsqueeze(-1) / cand                           # (T, Fb, C)
    A = (wn.unsqueeze(-1) * torch.cos(2 * math.pi * ratio)).sum(1)
    B = (wn.unsqueeze(-1) * torch.cos(math.pi * ratio)).sum(1)
    best = torch.argmax(A - B, dim=1)
    f0 = cand[best]
    coh = A.gather(1, best.unsqueeze(1)).squeeze(1)
    # least-squares refinement on bins that sit close to a harmonic
    h = torch.round(ifb / f0.unsqueeze(1))
    ok = (h >= 1) & ((ifb / f0.unsqueeze(1) - h).abs() < 0.2)
    ww = wn * ok
    f0r = (ww * ifb * h).sum(1) / (ww * h * h).sum(1).clamp_min(EPS)
    f0 = torch.where((ww.sum(1) > 0) & (f0r > 0.5 * f0) & (f0r < 2 * f0), f0r, f0)
    return f0, coh


def _stream_rps(X, Y, cfg):
    """Relative phase shift (RPS) of the harmonics.

    With ``phi_h`` the phase of harmonic ``h`` referenced to the *centre* of the
    (symmetric) analysis window -- for a stationary partial this is constant
    over the whole main lobe, so small F0 errors do not matter --

        rps:  psi_h = princarg(phi_h - h phi_1),            h = 2..H
        d2 :  psi_h = princarg(phi_{h+1} - 2 phi_h + phi_{h-1})

    A time shift ``tau`` adds ``2 pi h f0 tau`` to ``phi_h`` and therefore
    cancels in both forms, so the organisers' random leading pad has no effect.
    RPS describes the *shape* of the glottal pulse (the harmonic phase
    structure), which is complementary to the harmonic *frequencies* that the
    IFD stream encodes.  Output: ``[cos psi | sin psi]`` (wrap-free) and a
    voicing mask (sieve coherence >= ``rps_voicing``).
    """
    win, _ = cfg.frame_lengths()
    N, fs = cfg.n_fft, cfg.sample_rate
    f0, coh = phase_f0(X, cfg)
    k = torch.arange(X.shape[1], dtype=torch.float32)
    centre = (win - 1) / 2.0
    phic = torch.angle(X) + 2.0 * math.pi * k * centre / N
    H = cfg.rps_harmonics
    hs = torch.arange(1, H + 1, dtype=f0.dtype)
    kh = torch.round(hs.unsqueeze(0) * f0.unsqueeze(1) * N / fs).long()
    kh = kh.clamp(0, X.shape[1] - 1)
    ph = torch.gather(phic, 1, kh)                               # (T, H)
    if cfg.rps_mode == "rps":
        psi = princarg(ph[:, 1:] - hs[1:].unsqueeze(0) * ph[:, :1])
    elif cfg.rps_mode == "d2":
        psi = princarg(ph[:, 2:] - 2 * ph[:, 1:-1] + ph[:, :-2])
    else:
        raise ValueError(cfg.rps_mode)
    feat = torch.cat([torch.cos(psi), torch.sin(psi)], dim=1)
    voiced = coh >= cfg.rps_voicing
    return feat, voiced


def _stream_tfd(X, Y, cfg) -> torch.Tensor:
    """Mixed time-frequency phase derivative (frequency difference of the IFD).

    ``D(t,k) = princarg(Delta_t(t,k+1) - Delta_t(t,k))`` where ``Delta_t`` is the
    baseband temporal phase difference of :func:`_stream_ifd`.  Equivalently the
    frame-to-frame change of the frequency-direction phase difference, i.e. how
    the local group delay moves over time.  Built only from the shift-invariant
    IFD, so the random leading pad cancels exactly; aggregated as a weighted
    circular mean inside each band (wrap-free).
    """
    _, hop = cfg.frame_lengths()
    lag = max(1, cfg.ifd_lag)
    phi = torch.angle(X)
    k = torch.arange(X.shape[1], dtype=phi.dtype, device=phi.device)
    advance = 2.0 * math.pi * k * (hop * lag) / cfg.n_fft
    d = princarg(phi[lag:] - phi[:-lag] - advance)
    d = torch.cat([d[:1].expand(lag, -1), d], dim=0)
    dd = princarg(d[:, 1:] - d[:, :-1])
    mag = X.abs()
    m2 = torch.minimum(mag[:, 1:], mag[:, :-1])       # both bins must be reliable
    w = m2 if cfg.mag_weight else torch.ones_like(m2)
    z = _weighted_band_cos_sin(dd, w, cfg)
    return _dct_reduce(z, cfg, n_sub=2)


_STREAMS = {
    "tfd": (_stream_tfd, False),
    "rps": (_stream_rps, False),
    "ifd": (_stream_ifd, False),
    "mgdcc": (_stream_mgdcc, True),
    "fdpd": (_stream_fdpd, False),
    "cosphase": (_stream_cosphase, False),
}


# ---------------------------------------------------------------------------
# post-processing
# ---------------------------------------------------------------------------


def add_deltas(feat: torch.Tensor, order: int, width: int) -> torch.Tensor:
    """Append regression deltas (and delta-deltas) computed on contiguous frames."""
    out = [feat]
    cur = feat
    for _ in range(order):
        cur = _delta(cur, width)
        out.append(cur)
    return torch.cat(out, dim=1)


def _delta(x: torch.Tensor, width: int) -> torch.Tensor:
    """d_t = sum_{n=1..W} n (x_{t+n} - x_{t-n}) / (2 sum n^2); edges replicated."""
    T, D = x.shape
    norm = 2.0 * sum(n * n for n in range(1, width + 1))
    pad = torch.nn.functional.pad(x.t().unsqueeze(0), (width, width), mode="replicate")
    pad = pad.squeeze(0).t()                                   # (T+2W, D)
    acc = torch.zeros_like(x)
    for n in range(1, width + 1):
        acc = acc + n * (pad[width + n: width + n + T] - pad[width - n: width - n + T])
    return acc / norm


def smooth_time(x: torch.Tensor, half: int) -> torch.Tensor:
    """Centred moving average over ``2*half+1`` contiguous frames (edges
    replicated).  Applied to the band-level (cos, sin) phase-deviation values,
    i.e. it averages unit-circle quantities, so wrapping is still not an issue;
    it lowers the frame-to-frame estimation noise of the deviation."""
    if half <= 0 or x.shape[0] == 0:
        return x
    k = 2 * half + 1
    pad = torch.nn.functional.pad(x.t().unsqueeze(0), (half, half), mode="replicate")
    return torch.nn.functional.avg_pool1d(pad, k, stride=1).squeeze(0).t()


def cmvn(feat: torch.Tensor, mode: str) -> torch.Tensor:
    """Per-utterance mean / mean-variance normalisation.

    ``"global"`` is a no-op here: that variant is applied after caching by
    :mod:`svcore.normalise`, using statistics pooled over the background
    UBM-training frames only.
    """
    if mode in ("none", "global") or feat.shape[0] == 0:
        return feat
    f = feat - feat.mean(dim=0, keepdim=True)
    if mode == "mv":
        f = f / f.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-5)
    return f


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------


def extract(wav: torch.Tensor, cfg: FeatureConfig) -> torch.Tensor:
    """Waveform -> ``(n_retained_frames, feature_dim)`` float32 features.

    Order of operations: framing -> phase stream -> band aggregation -> DCT ->
    deltas (on contiguous frames, so temporal context is not broken by the VAD)
    -> VAD frame selection -> per-utterance CMVN.
    """
    fn, need_ramp = _STREAMS[cfg.stream]
    X, Y, logE = stft_pair(wav, cfg, need_ramp)
    feat = fn(X, Y, cfg)
    extra_mask = None
    if isinstance(feat, tuple):            # stream supplies its own frame mask
        feat, extra_mask = feat
    elif cfg.voicing_mask > 0:
        _, coh = phase_f0(X, cfg)
        extra_mask = coh >= cfg.voicing_mask
    if cfg.smooth > 0:
        feat = smooth_time(feat, cfg.smooth)
    if cfg.deltas > 0:
        feat = add_deltas(feat, cfg.deltas, cfg.delta_width)
    mask = vad_mask(logE, cfg)
    if extra_mask is not None:
        both = mask & extra_mask
        # never return an empty utterance: fall back to the VAD frames
        mask = both if int(both.sum()) >= min(cfg.vad_min_frames, int(mask.sum())) else mask
    feat = feat[mask]
    feat = cmvn(feat, cfg.cmvn)
    return torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).to(torch.float32)


def feature_dim(cfg: FeatureConfig) -> int:
    """Feature dimension, obtained by running one dummy second of audio."""
    g = torch.Generator().manual_seed(0)
    wav = torch.randn(cfg.sample_rate, generator=g)
    return extract(wav, cfg).shape[1]
