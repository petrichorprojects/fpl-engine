#!/bin/bash
# Wrapper launchd calls every 15 minutes. Loads env, runs one nag check.
# Exits quietly (0) when there is nothing to do — that is the normal case.
set -euo pipefail

REPO="/Volumes/T9/fpl-engine"

# The repo lives on an external volume. If it is not mounted (Mini asleep,
# drive unplugged), do nothing rather than error-spam the log.
if [ ! -d "$REPO" ]; then
  echo "$(date -u +%FT%TZ) T9 not mounted — skipping" >> /tmp/fpl-nag.log
  exit 0
fi

set -a
# shellcheck disable=SC1091
source "$REPO/deploy/fpl-nag.env"
set +a

cd "$REPO"
exec .venv/bin/python scripts/lineup_nag.py >> /tmp/fpl-nag.log 2>&1
