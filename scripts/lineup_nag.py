#!/usr/bin/env python3
"""Escalating lineup reminder — nags until you post a screenshot, then stops.

Run this on a schedule (every ~15 minutes). Each run:
  1. Finds the next FPL deadline.
  2. If it is inside the nag window and you have not posted a screenshot since
     the window opened, posts a nag whose cadence tightens toward the deadline.
  3. When your screenshot appears, posts a confirmation and goes quiet until the
     next gameweek.

State lives in a small JSON file so runs are stateless from cron's point of view.

Delivery (pick one; the relay is the default and matches the other bots):
    FPL_NAG_RELAY_URL     n8n webhook, e.g.
                          https://…/webhook/fpl-nag-relay  (recommended)
  — or direct Slack, if you prefer a bot token over the relay —
    FPL_SLACK_BOT_TOKEN   Slack bot token (xoxb-…)
    FPL_SLACK_CHANNEL     Channel ID to post in (e.g. C0123ABCD)

Always:
    FPL_SLACK_USER_ID     YOUR Slack user ID (e.g. U0123ABCD) — only your image
                          silences the nag.

Usage:
    python scripts/lineup_nag.py                 # live, reads env
    python scripts/lineup_nag.py --dry-run       # print, never post
    python scripts/lineup_nag.py --dry-run --simulate-hours 1.5   # test cadence

Exit codes:
    0  normal (nagged, confirmed, or nothing due)
    1  misconfiguration or Slack error
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fpl_engine.client import FPLClient
from fpl_engine.deadlines import DeadlineTracker
from fpl_engine.nag import (
    NAG_WINDOW_HOURS,
    NagState,
    find_acknowledgement,
    render_ack_confirmation,
    render_nag,
    should_nag,
)
from fpl_engine.slack_gateway import (
    DryRunGateway,
    N8nRelayGateway,
    SlackApiGateway,
    SlackGateway,
)

UTC = _dt.timezone.utc
DEFAULT_STATE = Path(__file__).resolve().parent.parent / "data" / "nag_state.json"


def load_state(path: Path) -> NagState | None:
    if not path.exists():
        return None
    try:
        return NagState.from_dict(json.loads(path.read_text()))
    except (json.JSONDecodeError, TypeError):
        return None


def save_state(path: Path, state: NagState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.to_dict(), indent=2))


def run(
    gateway: SlackGateway,
    tracker: DeadlineTracker,
    state_path: Path,
    me: str,
    now: _dt.datetime,
) -> int:
    nxt = tracker.next_deadline(now=now)
    if nxt is None:
        print("No upcoming deadline — season over. Nothing to do.")
        return 0

    state = load_state(state_path)
    # Reset when the gameweek rolls over.
    if state is None or state.gameweek != nxt.gameweek:
        state = NagState.new(nxt.gameweek, nxt.deadline)

    hours = nxt.time_remaining(now).total_seconds() / 3600.0

    if state.acknowledged:
        print(f"GW{state.gameweek}: already acknowledged. Quiet.")
        return 0

    if hours > NAG_WINDOW_HOURS:
        print(f"GW{nxt.gameweek}: {hours:.1f}h out — outside the "
              f"{NAG_WINDOW_HOURS:.0f}h nag window. Quiet.")
        return 0

    # Window is open. Record when it first opened — the screenshot only counts
    # if it lands after this moment, so an old image cannot pre-silence a nag.
    if state.window_opened_iso is None:
        state.window_opened_iso = now.isoformat()

    since_ts = state.window_opened.timestamp() if state.window_opened else now.timestamp()

    # ── Check for the screenshot ─────────────────────────────────────────
    if me:
        messages = gateway.recent_messages(since_ts=since_ts, thread_ts=state.thread_ts)
        ack_ts = find_acknowledgement(messages, me=me, since_ts=since_ts)
        if ack_ts is not None:
            state.acknowledged = True
            state.ack_ts = str(ack_ts)
            gateway.post(render_ack_confirmation(state.gameweek), thread_ts=state.thread_ts)
            save_state(state_path, state)
            print(f"GW{state.gameweek}: screenshot received. Confirmed and quiet.")
            return 0

    # ── Decide whether to nag ────────────────────────────────────────────
    if not should_nag(hours, state.last_nag, now):
        mins = int((now - state.last_nag).total_seconds() / 60) if state.last_nag else 0
        print(f"GW{state.gameweek}: {hours:.1f}h out, last nag {mins}m ago — "
              "not due yet. Quiet.")
        return 0

    text = render_nag(
        gameweek=state.gameweek,
        hours_remaining=hours,
        countdown=nxt.human_countdown(now),
        nag_count=state.nag_count,
    )
    ts = gateway.post(text, thread_ts=state.thread_ts)
    if state.thread_ts is None:
        state.thread_ts = ts
    state.last_nag_iso = now.isoformat()
    state.nag_count += 1
    save_state(state_path, state)
    print(f"GW{state.gameweek}: nag #{state.nag_count} posted ({hours:.1f}h out).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Escalating FPL lineup nag")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print messages instead of posting to Slack")
    parser.add_argument("--state", default=str(DEFAULT_STATE),
                        help="Path to the nag state file")
    parser.add_argument("--fresh", action="store_true",
                        help="Clear the FPL API cache before reading deadlines")
    parser.add_argument("--simulate-hours", type=float, default=None,
                        help="Dry-run only: pretend the deadline is this many "
                             "hours away, to preview cadence and message tone")
    args = parser.parse_args()

    client = FPLClient()
    if args.fresh:
        client.clear_cache()
    tracker = DeadlineTracker.from_client(client)

    now = _dt.datetime.now(UTC)

    if args.simulate_hours is not None:
        if not args.dry_run:
            print("--simulate-hours requires --dry-run", file=sys.stderr)
            return 1
        nxt = tracker.next_deadline(now=now)
        if nxt:
            # Shift 'now' so the real next deadline sits simulate-hours away.
            now = nxt.deadline - _dt.timedelta(hours=args.simulate_hours)

    me = os.environ.get("FPL_SLACK_USER_ID", "")

    if args.dry_run:
        gateway: SlackGateway = DryRunGateway()
    else:
        relay = os.environ.get("FPL_NAG_RELAY_URL")
        token = os.environ.get("FPL_SLACK_BOT_TOKEN")
        channel = os.environ.get("FPL_SLACK_CHANNEL")
        if relay:
            gateway = N8nRelayGateway(webhook_url=relay)
        elif token and channel:
            gateway = SlackApiGateway(token=token, channel=channel)
        else:
            print("Set FPL_NAG_RELAY_URL (recommended), or "
                  "FPL_SLACK_BOT_TOKEN + FPL_SLACK_CHANNEL, or use --dry-run.",
                  file=sys.stderr)
            return 1
        if not me:
            print("Warning: FPL_SLACK_USER_ID unset — cannot detect your "
                  "screenshot, so the nag will never self-silence.",
                  file=sys.stderr)

    try:
        return run(gateway, tracker, Path(args.state), me=me, now=now)
    except Exception as exc:  # noqa: BLE001
        print(f"Nag run failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
