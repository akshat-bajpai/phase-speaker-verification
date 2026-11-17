"""Feature extraction over the corpus, with an on-disk cache per configuration."""
from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Sequence

import torch

from . import config as C
from . import data
from .config import FeatureConfig
from .features import extract


def cache_path(cfg: FeatureConfig) -> str:
    return os.path.join(C.CACHE_DIR, f"feat_{cfg.tag}.pt")


def build(cfg: FeatureConfig, file_ids: Optional[Sequence[str]] = None,
          verbose: bool = True) -> Dict[str, torch.Tensor]:
    """Extract (or load) features for every referenced recording."""
    path = cache_path(cfg)
    if os.path.exists(path):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if blob.get("tag") == cfg.tag:
            if verbose:
                print(f"[cache] hit  {os.path.basename(path)}  "
                      f"({len(blob['feats'])} files, dim={blob['dim']})")
            return blob["feats"]
    ids = list(file_ids) if file_ids is not None else data.all_file_ids()
    feats: Dict[str, torch.Tensor] = {}
    t0 = time.time()
    for i, fid in enumerate(ids):
        feats[fid] = extract(data.load_wav(fid), cfg)
        if verbose and (i + 1) % 500 == 0:
            print(f"[cache] {i+1}/{len(ids)} files  ({time.time()-t0:.1f}s)")
    dim = next(iter(feats.values())).shape[1]
    torch.save({"tag": cfg.tag, "cfg": cfg.to_dict(), "feats": feats, "dim": dim}, path)
    if verbose:
        n = torch.tensor([f.shape[0] for f in feats.values()], dtype=torch.float32)
        print(f"[cache] built {os.path.basename(path)} in {time.time()-t0:.1f}s | "
              f"dim={dim} frames/utt mean={n.mean():.1f} min={int(n.min())} "
              f"max={int(n.max())} total={int(n.sum())}")
    return feats


def pooled(feats: Dict[str, torch.Tensor], file_ids: Sequence[str],
           max_frames: int = 0, seed: int = C.SEED,
           dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Concatenate frames of ``file_ids``; optionally subsample to ``max_frames``.

    Subsampling is a uniform random draw over the pooled frames with a fixed
    seed, so every experiment that uses the same ``max_frames`` sees the same
    frames (required for comparable likelihood numbers).
    """
    parts = [feats[f] for f in file_ids if f in feats and feats[f].shape[0] > 0]
    X = torch.cat(parts, dim=0).to(dtype)
    if max_frames and X.shape[0] > max_frames:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(X.shape[0], generator=g)[:max_frames].sort().values
        X = X[idx]
    return X


def frame_counts(feats: Dict[str, torch.Tensor], file_ids: Sequence[str]) -> torch.Tensor:
    return torch.tensor([feats[f].shape[0] for f in file_ids], dtype=torch.float32)
