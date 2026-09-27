#!/usr/bin/env bash
set -euo pipefail
# One-command reproducible pipeline (hard-mode feed).
# Local routes only by default; the Jev routes also run when the TypeSafe key is in
# TYPESAFE_API_KEY or the macOS keychain (service jev-api-key). The key is never echoed.
python3.11 -m pytest -q
python3.11 -m tranx.cli synth --hard --n "${TRANX_N:-100000}"
TYPESAFE_API_KEY="${TYPESAFE_API_KEY:-$(security find-generic-password -s jev-api-key -w 2>/dev/null || true)}"
if [[ -n "${TYPESAFE_API_KEY}" ]]; then
  TYPESAFE_API_KEY="${TYPESAFE_API_KEY}" python3.11 -m tranx.cli eval \
    --routes rules,embedding,slm_fewshot,jev_merchant,jev_slm,cleaner,metadata \
    --name leaderboard_hard_jev.md \
    --also leaderboard_hard.md=rules,embedding,slm_fewshot,cleaner,metadata
else
  python3.11 -m tranx.cli eval --routes rules,embedding,slm_fewshot,cleaner,metadata \
    --name leaderboard_hard.md
fi
echo "Done. See reports/leaderboard_hard*.md, reports/run/ and reports/*.png"
