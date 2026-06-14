#!/usr/bin/env bash
set -euo pipefail
# One-command reproducible pipeline.
python3.11 -m pytest -q
python3.11 -m tranx.cli synth --n "${TRANX_N:-100000}"
python3.11 -m tranx.cli eval
echo "Done. See reports/leaderboard.md and reports/*.png"
