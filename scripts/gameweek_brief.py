#!/usr/bin/env python3
"""One command, one answer: what do I do before the next deadline?

Designed to be run on a schedule (see `--check-only`) and to be readable when
it lands in Slack or a terminal at 8am. The deadline block always renders, even
when the model cannot — missing a deadline is the expensive failure, so the
reminder must never depend on the modelling path succeeding.

Usage:
    python scripts/gameweek_brief.py                    # full brief
    python scripts/gameweek_brief.py --check-only       # deadline block only, fast
    python scripts/gameweek_brief.py --squad 123,456    # include transfer advice
    python scripts/gameweek_brief.py --horizon 5        # plan further ahead
    python scripts/gameweek_brief.py --out brief.md     # write to a file

Exit codes:
    0  brief produced
    1  hard failure (no network, no fixture data)
    2  deadline block produced, model unavailable (e.g. pre-season)
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fpl_engine.client import FPLClient
from fpl_engine.deadlines import DeadlineTracker
from fpl_engine.engine import FPLEngine
from fpl_engine.optimizer import Chip, GameState

# Minimum rounds of per-fixture history before the rolling features mean
# anything. Below this the model will happily fit noise and hand you a
# confident-looking squad built on three data points.
MIN_ROUNDS_FOR_MODEL = 4


def build_deadline_block(tracker: DeadlineTracker) -> str:
    nxt = tracker.next_deadline()
    if nxt is None:
        return "## Deadline\n\nNo remaining deadlines — the season is over.\n"

    remaining = nxt.time_remaining()
    hours = remaining.total_seconds() / 3600

    if hours <= 3:
        urgency = "🚨 SET YOUR LINEUP NOW"
    elif hours <= 26:
        urgency = "⚠️  Deadline within a day"
    elif hours <= 72:
        urgency = "🟡 Deadline this week"
    else:
        urgency = "🟢 Nothing due yet"

    lines = [
        "## Deadline",
        "",
        f"**{urgency}**",
        "",
        f"- **{nxt.name}** locks {nxt.local().strftime('%A %d %B at %H:%M %Z')}",
        f"- Time remaining: **{nxt.human_countdown()}**",
    ]

    rest = tracker.upcoming(4)[1:]
    if rest:
        lines.append("- Then: " + " · ".join(
            f"{d.name} {d.local().strftime('%a %d %b %H:%M')}" for d in rest
        ))

    lines.append("")
    lines.append(
        "Checklist: captain set · bench ordered · no flagged players starting · "
        "free transfer used or banked deliberately."
    )
    lines.append("")
    return "\n".join(lines)


def build_model_block(
    engine: FPLEngine,
    horizon: int,
    gamestate: GameState,
    chip: Chip,
    squad_ids: list[int] | None,
    free_transfers: int,
    bank: int,
) -> str:
    """Run the model and render the squad recommendation."""
    engine.build_features(verbose=False)
    engine.train(verbose=False)
    engine.predict(horizon=horizon, verbose=False)

    result = engine.optimize(gamestate=gamestate, chip=chip, verbose=False)
    preds = engine.predictions_df

    captain = preds[preds["element_id"] == result.captain_id].iloc[0]
    vice = preds[preds["element_id"] == result.vice_captain_id].iloc[0]

    lines = [
        f"## Recommended XI — GW {engine.target_gw}",
        "",
        f"Projected: **{result.total_xp:.1f} xP** · strategy: {gamestate.value}",
        "",
        "| Player | Pos | Team | Price | xP | Own% |",
        "|---|---|---|---|---|---|",
    ]
    for _, p in result.starting_xi.iterrows():
        mark = ""
        if p["element_id"] == result.captain_id:
            mark = " (C)"
        elif p["element_id"] == result.vice_captain_id:
            mark = " (V)"
        lines.append(
            f"| {p['name']}{mark} | {p['position']} | "
            f"{str(p.get('team_name', ''))[:12]} | £{p['price'] / 10:.1f}m | "
            f"{p['xp']:.2f} | {p.get('ownership_pct', 0):.1f}% |"
        )

    lines += [
        "",
        f"**Captain:** {captain['name']} ({captain['xp']:.2f} xP) · "
        f"**Vice:** {vice['name']} ({vice['xp']:.2f} xP)",
        "",
        "**Bench, in order:** " + " → ".join(
            f"{p['name']} ({p['xp']:.2f})" for _, p in result.bench.iterrows()
        ),
        "",
    ]

    # Fixture warnings the manager actually needs to see.
    if "n_fixtures" in preds.columns:
        squad_preds = preds[preds["element_id"].isin(result.squad["element_id"])]
        blanks = squad_preds[squad_preds["n_fixtures"] == 0]
        doubles = squad_preds[squad_preds["n_fixtures"] > 1]
        if not blanks.empty:
            lines.append("⚠️ **Blanking this GW:** " +
                         ", ".join(blanks["name"].astype(str)))
        if not doubles.empty:
            lines.append("🎯 **Doubling this GW:** " +
                         ", ".join(doubles["name"].astype(str)))
        if not blanks.empty or not doubles.empty:
            lines.append("")

    if squad_ids:
        lines += _transfer_block(engine, squad_ids, free_transfers, bank, horizon)

    return "\n".join(lines)


def _transfer_block(
    engine: FPLEngine,
    squad_ids: list[int],
    free_transfers: int,
    bank: int,
    horizon: int,
) -> list[str]:
    transfers = engine.optimize_transfers(
        current_squad_ids=squad_ids,
        free_transfers=free_transfers,
        bank=bank,
        horizon=horizon,
        verbose=False,
    )
    lines = [f"## Transfers — next {horizon} GWs", ""]
    if not transfers:
        lines += ["No transfer clears the bar. Bank the free transfer.", ""]
        return lines

    for i, t in enumerate(transfers, 1):
        tag = "costs a -4 hit" if t.get("hit") else "free"
        lines.append(
            f"{i}. **{t['out_name']} → {t['in_name']}** "
            f"· +{t['xp_gain']:.1f} xP over {horizon} GWs ({tag})"
        )
    lines.append("")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="FPL gameweek brief")
    parser.add_argument("--check-only", action="store_true",
                        help="Deadline block only — skips the 5-minute data fetch")
    parser.add_argument("--horizon", type=int, default=4,
                        help="Gameweeks to plan ahead (default: 4)")
    parser.add_argument("--gamestate", default="neutral",
                        choices=["neutral", "leading", "chasing", "mini_league"])
    parser.add_argument("--chip", default="none",
                        choices=["none", "wildcard", "free_hit",
                                 "bench_boost", "triple_captain"])
    parser.add_argument("--squad", default="",
                        help="Comma-separated element_ids of your current 15")
    parser.add_argument("--free-transfers", type=int, default=1)
    parser.add_argument("--bank", type=int, default=0,
                        help="Money in the bank, in 0.1m units")
    parser.add_argument("--fresh", action="store_true",
                        help="Clear the API cache first")
    parser.add_argument("--out", help="Write the brief to this path")
    args = parser.parse_args()

    client = FPLClient()
    if args.fresh:
        client.clear_cache()

    try:
        tracker = DeadlineTracker.from_client(client)
    except Exception as exc:
        print(f"Could not reach the FPL API: {exc}", file=sys.stderr)
        return 1

    blocks = ["# FPL Brief", "", build_deadline_block(tracker)]
    status = 0

    if not args.check_only:
        engine = FPLEngine(client=client)
        try:
            engine.fetch_data(fetch_histories=True, verbose=False)
            rounds = (
                int(engine.history_df["round"].nunique())
                if not engine.history_df.empty else 0
            )
            if rounds < MIN_ROUNDS_FOR_MODEL:
                blocks.append(
                    "## Model\n\n"
                    f"Not enough data yet — {rounds} gameweek(s) of per-fixture "
                    f"history, {MIN_ROUNDS_FOR_MODEL} needed before the rolling "
                    "features carry signal.\n\n"
                    "Until then the deadline reminder is the useful half of this "
                    "brief. Pick on fixtures and team news.\n"
                )
                status = 2
            else:
                squad_ids = (
                    [int(x) for x in args.squad.split(",") if x.strip()]
                    if args.squad else None
                )
                blocks.append(build_model_block(
                    engine=engine,
                    horizon=args.horizon,
                    gamestate=GameState(args.gamestate),
                    chip=Chip(args.chip),
                    squad_ids=squad_ids,
                    free_transfers=args.free_transfers,
                    bank=args.bank,
                ))
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            blocks.append(
                f"## Model\n\nThe model failed to run: `{exc}`\n\n"
                "The deadline above still stands — set your lineup manually.\n"
            )
            status = 2

    brief = "\n".join(blocks)
    print(brief)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(brief)
        print(f"\nWritten to {out}", file=sys.stderr)

    return status


if __name__ == "__main__":
    sys.exit(main())
