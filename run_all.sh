#!/usr/bin/env bash
# Reproduces every number and figure in the report.
# Runtime on a 2-core machine: Q1 ~2 min, Q2 ~4 min, Q3 ~75 min.
set -euo pipefail
cd "$(dirname "$0")"
python scripts/q1.py        # tree grid, bagging, OOB/disagreement/margin, clean vs masked
python scripts/q2.py        # SAMME toy round, grid, diagnostics, bagging vs boosting, masked
python scripts/q3.py        # 30 resampled training sets, bias^2 and variance
python scripts/figures.py   # all figures from results/*.json
