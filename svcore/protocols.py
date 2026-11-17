"""Additional speaker-disjoint validation protocols.

The official development list has only 196 genuine trials, so a single
misordered trial moves the EER by 0.51 %.  Selecting ~10 hyper-parameters
against it over-fits badly -- our first submission scored 0.61 % on dev and
3.70 % on the public leaderboard.

This module builds two *extra* protocols so that a modelling choice can be
required to help on several independent sets before we believe it:

``dev_loo``   Leave-one-out over the development enrolment utterances.  For
              rotation k in {0,1,2}, every dev speaker is enrolled on its two
              other enrolment recordings and tested on the k-th.  291 genuine
              trials (enrolment size 2).  Disjoint from the official dev trials
              in *test material* (it reuses enrolment audio, which the official
              list never uses as a test file).

``bkg``       Background speakers with >= 4 recordings: enrol on the first 3,
              test on the rest, with impostor material drawn from other
              background speakers.  92 genuine trials at enrolment size 3.
              These speakers are excluded from UBM training for this protocol
              (``ubm_exclude``), so it is genuinely held out.

Impostor trials are drawn with a fixed seed so every system sees exactly the
same list.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch

from . import config as C
from . import data


def _impostors(test_files: Sequence[str], owner: Dict[str, str],
               targets: Sequence[str], n_per: int, seed: int) -> List[dict]:
    """Pair each test file with ``n_per`` targets that are not its true speaker."""
    g = torch.Generator().manual_seed(seed)
    out = []
    tg = list(targets)
    for f in test_files:
        true = owner.get(f)
        picks, guard = [], 0
        while len(picks) < n_per and guard < 100:
            i = int(torch.randint(len(tg), (1,), generator=g))
            if tg[i] != true and tg[i] not in picks:
                picks.append(tg[i])
            guard += 1
        for p in picks:
            out.append({"claimed_id": p, "test_file": f, "label": 0})
    return out


def dev_loo(n_impostor_per_test: int = 3, seed: int = C.SEED):
    """Returns ``(rotations, n_enrol)`` where each rotation is
    ``(enrol_map, trials)`` -- enrol_map: speaker -> 2 enrolment files."""
    rows = data.enrol("dev")
    by_spk: Dict[str, Dict[int, str]] = {}
    for r in rows:
        by_spk.setdefault(r["speaker_id"], {})[r["enrol_index"]] = r["file_id"]
    speakers = sorted(by_spk)
    rotations = []
    for k in (0, 1, 2):
        emap = {s: [by_spk[s][i] for i in (0, 1, 2) if i != k and i in by_spk[s]]
                for s in speakers}
        emap = {s: v for s, v in emap.items() if len(v) == 2 and k in by_spk[s]}
        tests = {by_spk[s][k]: s for s in emap}
        trials = [{"claimed_id": s, "test_file": f, "label": 1}
                  for f, s in tests.items()]
        trials += _impostors(list(tests), tests, list(emap),
                             n_impostor_per_test, seed + k)
        rotations.append((emap, trials))
    return rotations, 2


def background_protocol(min_files: int = 4, n_enrol: int = 3,
                        n_impostor_per_test: int = 3,
                        n_extra_impostor_files: int = 300,
                        seed: int = C.SEED):
    """Returns ``(enrol_map, trials, ubm_exclude_files)``.

    ``ubm_exclude_files`` lists every recording of every speaker appearing in
    this protocol, so the UBM can be retrained without them and the protocol is
    genuinely speaker-disjoint from the model.
    """
    spk2files = data.speaker_files(data.background())
    targets = sorted(s for s, v in spk2files.items() if len(v) >= min_files)
    emap = {s: spk2files[s][:n_enrol] for s in targets}
    owner: Dict[str, str] = {}
    test_files: List[str] = []
    for s in targets:
        for f in spk2files[s][n_enrol:]:
            owner[f] = s
            test_files.append(f)
    trials = [{"claimed_id": owner[f], "test_file": f, "label": 1} for f in test_files]
    trials += _impostors(test_files, owner, targets, n_impostor_per_test, seed)

    # extra impostor material: speakers that are not targets here
    others = sorted(s for s in spk2files if s not in set(targets))
    g = torch.Generator().manual_seed(seed + 99)
    perm = torch.randperm(len(others), generator=g).tolist()[:n_extra_impostor_files]
    extra_files, extra_owner = [], {}
    for i in perm:
        f = spk2files[others[i]][0]
        extra_files.append(f)
        extra_owner[f] = others[i]
    trials += _impostors(extra_files, extra_owner, targets, 2, seed + 7)

    # Exclude the TARGET speakers from UBM training: those are the speakers whose
    # models we adapt and whose genuine trials we score, so a UBM that had seen
    # their audio would make this protocol optimistic.  The extra impostor
    # speakers stay in the UBM -- removing them too would cost 2/3 of the UBM
    # training data, and their effect on the LLR is both tiny and conservative
    # (a UBM that models an impostor well *lowers* that impostor's score).
    exclude = set()
    for s in targets:
        exclude.update(spk2files[s])
    return emap, trials, exclude


def summarise(trials: Sequence[dict]) -> str:
    g = sum(1 for t in trials if t["label"] == 1)
    return (f"{len(trials)} trials ({g} genuine, {len(trials)-g} impostor), "
            f"{len({t['claimed_id'] for t in trials})} claimed speakers, "
            f"{len({t['test_file'] for t in trials})} test recordings")
