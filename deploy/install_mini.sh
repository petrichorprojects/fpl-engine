#!/bin/bash
# One-shot installer for the FPL lineup nag on the Mac Mini.
#
# After AirDropping and unzipping this folder anywhere (e.g. ~/fpl-engine):
#   cd ~/fpl-engine
#   bash deploy/install_mini.sh
#
# It creates a venv, installs the nag's deps, writes a launchd job pointing at
# THIS folder, and starts it. Re-runnable: reloads cleanly if already installed.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="com.philipp.fpl-nag"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG="$HOME/Library/Logs/fpl-nag.log"

echo "Repo:  $REPO"
echo "Plist: $PLIST"
echo

# ── 1. Python venv ───────────────────────────────────────────────────────────
PY=""
if command -v uv >/dev/null 2>&1; then
  echo "→ Creating venv with uv…"
  uv venv "$REPO/.venv"
  uv pip install --python "$REPO/.venv/bin/python" -r "$REPO/deploy/requirements-nag.txt"
  PY="$REPO/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  echo "→ uv not found; using python3 venv…"
  python3 -m venv "$REPO/.venv"
  "$REPO/.venv/bin/python" -m pip install --upgrade pip
  "$REPO/.venv/bin/python" -m pip install -r "$REPO/deploy/requirements-nag.txt"
  PY="$REPO/.venv/bin/python"
else
  echo "ERROR: neither uv nor python3 found. Install one, then re-run." >&2
  exit 1
fi

# ── 2. Env file ──────────────────────────────────────────────────────────────
if [ ! -f "$REPO/deploy/fpl-nag.env" ]; then
  cp "$REPO/deploy/fpl-nag.env.example" "$REPO/deploy/fpl-nag.env"
  echo "→ Wrote deploy/fpl-nag.env from the example. Check FPL_SLACK_USER_ID."
fi

# ── 3. Smoke test ────────────────────────────────────────────────────────────
echo "→ Dry-run smoke test (no Slack post):"
( cd "$REPO" && "$PY" scripts/lineup_nag.py --dry-run ) || {
  echo "ERROR: smoke test failed. Fix before installing the timer." >&2
  exit 1
}

# ── 4. launchd job ───────────────────────────────────────────────────────────
mkdir -p "$HOME/Library/LaunchAgents" "$HOME/Library/Logs"
cat > "$PLIST" <<PLISTEOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>$REPO/deploy/run_nag.sh</string>
  </array>
  <key>StartInterval</key><integer>900</integer>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>$LOG</string>
  <key>StandardErrorPath</key><string>$LOG</string>
</dict>
</plist>
PLISTEOF

launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"

echo
echo "✓ Installed. Runs every 15 min. Log: $LOG"
echo "  Stop:  launchctl unload $PLIST"
echo "  Tail:  tail -f $LOG"
