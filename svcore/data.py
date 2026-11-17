"""Dataset access: CSV protocol files (stdlib ``csv``) and waveform loading.

Audio I/O goes through ``torchaudio.load``; a ``wave``-based fallback (Python
standard library + ``torch.frombuffer``) keeps the pipeline runnable without
TorchCodec.  No NumPy/SciPy is involved in either path.
"""
from __future__ import annotations

import csv
import os
import wave
from typing import Dict, List, Sequence, Tuple

import torch

from . import config as C

# ---------------------------------------------------------------------------
# protocol files
# ---------------------------------------------------------------------------


def _read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def background() -> List[Dict[str, str]]:
    """[{file_id, speaker_id}] for the 786 permitted background speakers."""
    return _read_csv(os.path.join(C.DATA_DIR, "background.csv"))


def enrol(split: str) -> List[Dict[str, str]]:
    """[{file_id, speaker_id, enrol_index}] for ``split`` in {dev, eval}."""
    rows = _read_csv(os.path.join(C.DATA_DIR, f"{split}_enrol.csv"))
    for r in rows:
        r["enrol_index"] = int(r["enrol_index"])
    return rows


def trials(split: str) -> List[Dict[str, str]]:
    """[{trial_id, claimed_id, test_file[, label]}] for ``split``."""
    rows = _read_csv(os.path.join(C.DATA_DIR, f"{split}_trials.csv"))
    for r in rows:
        if "label" in r:
            r["label"] = int(r["label"])
    return rows


def enrol_map(split: str, n_enrol: int = 3) -> Dict[str, List[str]]:
    """speaker_id -> the first ``n_enrol`` enrolment file_ids, by enrol_index."""
    out: Dict[str, List[Tuple[int, str]]] = {}
    for r in enrol(split):
        if r["enrol_index"] < n_enrol:
            out.setdefault(r["speaker_id"], []).append((r["enrol_index"], r["file_id"]))
    return {spk: [f for _, f in sorted(v)] for spk, v in out.items()}


def speaker_files(rows: Sequence[Dict[str, str]]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for r in rows:
        out.setdefault(r["speaker_id"], []).append(r["file_id"])
    return {k: sorted(v) for k, v in out.items()}


# ---------------------------------------------------------------------------
# speaker-disjoint background splits (reproducible, seed = C.SEED)
# ---------------------------------------------------------------------------


def background_splits(
    heldout_frac: float = 0.10,
    cohort_size: int = 150,
    cohort_min_files: int = 3,
    seed: int = C.SEED,
):
    """Split the background speakers *by speaker*.

    Returns ``(ubm_files, heldout_files, cohort_spk2files)``.

    * ``ubm``      -- pooled frames for UBM/EM training.
    * ``heldout``  -- speaker-disjoint from ``ubm``; used only to report average
                      held-out log-likelihood (model selection, BIC/AIC check).
    * ``cohort``   -- speakers with >= ``cohort_min_files`` recordings, used to
                      build T-norm / Z-norm cohort models.  Drawn from the UBM
                      portion (standard practice: cohort models are MAP-adapted
                      pseudo-impostors, and no background speaker ever appears in
                      a trial).  ``scripts/05_ablations.py`` also reports a
                      variant with a cohort fully disjoint from UBM training.
    """
    spk2files = speaker_files(background())
    spks = sorted(spk2files)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(spks), generator=g).tolist()
    spks = [spks[i] for i in perm]

    n_held = int(round(heldout_frac * len(spks)))
    held_spk, ubm_spk = spks[:n_held], spks[n_held:]

    ubm_files = [f for s in ubm_spk for f in spk2files[s]]
    held_files = [f for s in held_spk for f in spk2files[s]]

    cohort = {s: spk2files[s][:cohort_min_files] for s in ubm_spk
              if len(spk2files[s]) >= cohort_min_files}
    cohort = {s: cohort[s] for s in sorted(cohort)[:cohort_size]} if cohort_size else cohort
    return ubm_files, held_files, cohort


# ---------------------------------------------------------------------------
# waveform loading
# ---------------------------------------------------------------------------
_USE_TORCHAUDIO = None


def _load_torchaudio(path: str) -> Tuple[torch.Tensor, int]:
    import torchaudio
    wav, sr = torchaudio.load(path)
    return wav, sr


def _load_wave_stdlib(path: str) -> Tuple[torch.Tensor, int]:
    """16-bit PCM WAV reader built from ``wave`` (stdlib) + torch only."""
    with wave.open(path, "rb") as wf:
        n_ch, width, sr, n_frames = (
            wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes())
        raw = wf.readframes(n_frames)
    if width != 2:
        raise ValueError(f"expected 16-bit PCM, got {width*8}-bit: {path}")
    pcm = torch.frombuffer(bytearray(raw), dtype=torch.int16)
    wav = pcm.to(torch.float32).div_(32768.0).view(-1, n_ch).t().contiguous()
    return wav, sr


def load_wav(file_id: str, target_sr: int = C.SAMPLE_RATE) -> torch.Tensor:
    """Return a mono float32 waveform of shape (num_samples,)."""
    global _USE_TORCHAUDIO
    path = os.path.join(C.AUDIO_DIR, f"{file_id}.wav")
    if _USE_TORCHAUDIO is None:
        try:
            _load_torchaudio(path)
            _USE_TORCHAUDIO = True
        except Exception:
            _USE_TORCHAUDIO = False
    wav, sr = _load_torchaudio(path) if _USE_TORCHAUDIO else _load_wave_stdlib(path)
    if sr != target_sr:
        raise ValueError(f"{file_id}: sample rate {sr} != {target_sr}")
    if wav.dim() == 2:
        wav = wav.mean(dim=0) if wav.shape[0] > 1 else wav[0]
    return wav.to(torch.float32)


def all_file_ids() -> List[str]:
    """Every recording referenced by any protocol file."""
    ids = {r["file_id"] for r in background()}
    for split in ("dev", "eval"):
        ids |= {r["file_id"] for r in enrol(split)}
        ids |= {r["test_file"] for r in trials(split)}
    return sorted(ids)
