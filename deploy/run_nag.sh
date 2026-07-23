#!/bin/bash
# Wrapper launchd calls every 15 minutes. Loads env, runs one nag check.
# Location-independent: derives the repo root from this script's own path, so it
# works wherever the folder was unzipped on the Mini.
# Exits 0 when there is nothing to do — that is the normal case.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${FPL_NAG_LOG:-$HOME/Library/Logs/fpl-nag.log}"

if [ ! -f "$REPO/deploy/fpl-nag.env" ]; then
  echo "$(date -u +%FT%TZ) no deploy/fpl-nag.env — run deploy/install_mini.sh first" >> "$LOG"
  exit 0
fi

set -a
# shellcheck disable=SC1091
source "$REPO/deploy/fpl-nag.env"
set +a

cd "$REPO"
exec .venv/bin/python scripts/lineup_nag.py >> "$LOG" 2>&1
