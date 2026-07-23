# FPL Lineup Nag — Mac Mini setup

AirDropped this folder? Three steps.

## 1. Unzip somewhere permanent

Not `/Volumes/T9` (that's the external drive). Put it on the Mini's own disk:

```bash
mkdir -p ~/fpl-engine && unzip ~/Downloads/fpl-nag.zip -d ~/fpl-engine
cd ~/fpl-engine
```

## 2. Install

```bash
bash deploy/install_mini.sh
```

This creates a venv, installs deps (`httpx`, `pandas`), runs a dry-run smoke
test, then registers a launchd job that fires every 15 minutes. Re-runnable.

## 3. Confirm your Slack id

The nag silences only when **your** screenshot appears. Default id is
`U076BVACCRH`. To confirm: post any message in #pr-bot-talk, then:

```bash
curl -s -X POST "https://petrichorprojects.app.n8n.cloud/webhook/fpl-nag-relay" \
  -H 'Content-Type: application/json' -d '{"mode":"history"}' \
  | python3 -m json.tool | grep -m1 '"user"'
```

If it differs, edit `FPL_SLACK_USER_ID` in `deploy/fpl-nag.env` and re-run the
installer.

---

**Preview the cadence** without waiting for a real deadline:

```bash
.venv/bin/python scripts/lineup_nag.py --dry-run --simulate-hours 1.5
```

**Logs:** `~/Library/Logs/fpl-nag.log`
**Stop:** `launchctl unload ~/Library/LaunchAgents/com.philipp.fpl-nag.plist`

The season opener is 21 Aug 2026, so until then every run logs "outside nag
window" and stays silent — that's correct.

## What's in this bundle

- `fpl_engine/` — deadlines, nag brain, Slack relay gateway
- `scripts/lineup_nag.py` — the runner launchd calls
- `scripts/gameweek_brief.py` — deadline + (in-season) squad brief
- `deploy/` — installer, env, launchd wrapper
- Slack I/O goes through the n8n relay "FPL Lineup Nag Relay" — no token lives
  on the Mini.
