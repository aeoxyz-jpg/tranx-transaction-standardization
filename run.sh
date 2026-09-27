#!/usr/bin/env bash
set -euo pipefail
# One-command reproducible pipeline (hard-mode feed).
# Local routes only by default; set TYPESAFE_API_KEY to also score the Jev routes.
python3.11 -m pytest -q
python3.11 -m tranx.cli synth --hard --n "${TRANX_N:-100000}"
if [[ -n "${TYPESAFE_API_KEY:-}" ]]; then
  python3.11 -m tranx.cli eval --routes rules,embedding,slm_fewshot,jev_merchant,jev_slm \
    --name leaderboard_hard_jev.md --also leaderboard_hard.md=rules,embedding,slm_fewshot
else
  python3.11 -m tranx.cli eval --routes rules,embedding,slm_fewshot --name leaderboard_hard.md
fi
echo "Done. See reports/leaderboard_hard*.md and reports/*.png"
