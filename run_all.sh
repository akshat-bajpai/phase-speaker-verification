#!/usr/bin/env bash
# Reproduce every reported result from scratch.
# Expects the competition data unpacked into ./data (see README.md).
set -euo pipefail
PY=${PY:-.venv/bin/python}

echo "### Module verification"              ; $PY -u tests/test_modules.py         | tee results/log_tests.txt
echo "### Task 1 -- features"               ; $PY -u scripts/01_features.py        | tee results/log_01_features.txt
echo "### Task 2 -- GMM behaviour"          ; $PY -u scripts/02_gmm_behaviour.py   | tee results/log_02_gmm.txt
echo "### Task 3 -- UBM training"           ; $PY -u scripts/03_train_ubm.py       | tee results/log_03_ubm.txt
echo "### Tasks 4 & 5 -- MAP + scoring"     ; $PY -u scripts/04_enrol_score.py     | tee results/log_04_enrol_score.txt
echo "### Task 6 -- ablations"              ; $PY -u scripts/05_ablations.py       | tee results/log_05_ablations.txt
echo "### Submission (T-norm, primary)"     ; $PY -u scripts/06_make_submission.py --norm tnorm --tag final \
                                                                                   | tee results/log_06_submission.txt
echo "### Submission (raw LLR, backup)"     ; $PY -u scripts/06_make_submission.py --norm none  --tag rawllr \
                                                                                   | tee results/log_06_submission_raw.txt
echo
echo "Done."
echo "  figures/             all report figures"
echo "  results/*.json       machine-readable results"
echo "  results/log_*.txt    verbatim logs"
echo "  results/dev_results.csv              claimed_id,test_file,label,score"
echo "  submissions/submission_final.csv     trial_id,score  <- submit this one"
echo "  submissions/submission_rawllr.csv    trial_id,score  (no score normalisation)"
