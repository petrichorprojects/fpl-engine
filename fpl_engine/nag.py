"""Lineup nag loop — escalating reminders that stop only when you prove you acted.

The problem this solves is behavioural, not analytical: a single reminder is easy
to swipe away and forget. So this keeps re-posting on a tightening cadence as the
deadline approaches, and the *only* thing that silences it is proof of action —
a screenshot of your lineup posted back into the channel.

Everything in this module is pure and network-free. The Slack I/O lives behind
the `SlackGateway` protocol in `fpl_engine.slack_gateway`, so the escalation
ladder and acknowledgement logic are unit-testable without a token.

State machine, per gameweek:

    NEW → (deadline enters nag window) → NAGGING → (screenshot posted) → DONE

`DONE` is sticky until the gameweek rolls over, at which point state resets.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Optional

UTC = _dt.timezone.utc


# ── Escalation ladder ────────────────────────────────────────────────────────
# (hours_low, hours_high, interval_minutes): while the deadline is between
# hours_low and hours_high away, nag at most once per interval. Bands tighten
# toward the deadline. Outside the widest band (>30h) nothing fires.
LADDER: list[tuple[float, float, int]] = [
    (24.0, 30.0, 360),   # 24-30h out: one early heads-up, then every 6h
    (12.0, 24.0, 360),   # 12-24h: every 6h
    (6.0, 12.0, 180),    # 6-12h:  every 3h
    (2.0, 6.0, 60),      # 2-6h:   hourly
    (0.0, 2.0, 20),      # <2h:    every 20 min — the "set it NOW" window
]

NAG_WINDOW_HOURS = LADDER[0][1]  # nothing fires until inside this many hours


def band_for(hours_remaining: float) -> Optional[tuple[float, float, int]]:
    """Return the ladder band covering `hours_remaining`, or None if outside."""
    if hours_remaining <= 0:
        return None
    for low, high, interval in LADDER:
        if low < hours_remaining <= high:
            return (low, high, interval)
    return None


def should_nag(
    hours_remaining: float,
    last_nag: Optional[_dt.datetime],
    now: _dt.datetime,
) -> bool:
    """Decide whether to post a nag right now.

    Fires when the deadline is inside the nag window and either no nag has gone
    out yet in this run of the loop, or the current band's interval has elapsed
    since the last one.
    """
    band = band_for(hours_remaining)
    if band is None:
        return False
    if last_nag is None:
        return True
    interval_min = band[2]
    elapsed_min = (now - last_nag).total_seconds() / 60.0
    return elapsed_min >= interval_min


def nag_urgency(hours_remaining: float) -> str:
    """Escalating prefix so the message itself signals how close the wire is."""
    if hours_remaining <= 2:
        return "🚨🚨 LINEUP LOCKS IN UNDER 2 HOURS"
    if hours_remaining <= 6:
        return "🚨 Lineup locks soon"
    if hours_remaining <= 12:
        return "⚠️ Deadline today"
    return "⏰ Deadline reminder"


# ── State ────────────────────────────────────────────────────────────────────

@dataclass
class NagState:
    """Per-gameweek nag state. Serialises to a small JSON file on disk."""

    gameweek: int
    deadline_iso: str
    acknowledged: bool = False
    ack_ts: Optional[str] = None
    thread_ts: Optional[str] = None      # Slack ts of the first nag (thread root)
    window_opened_iso: Optional[str] = None
    last_nag_iso: Optional[str] = None
    nag_count: int = 0

    @classmethod
    def new(cls, gameweek: int, deadline: _dt.datetime) -> "NagState":
        return cls(gameweek=gameweek, deadline_iso=deadline.isoformat())

    def to_dict(self) -> dict:
        return {
            "gameweek": self.gameweek,
            "deadline_iso": self.deadline_iso,
            "acknowledged": self.acknowledged,
            "ack_ts": self.ack_ts,
            "thread_ts": self.thread_ts,
            "window_opened_iso": self.window_opened_iso,
            "last_nag_iso": self.last_nag_iso,
            "nag_count": self.nag_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NagState":
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__})  # type: ignore[attr-defined]

    @property
    def last_nag(self) -> Optional[_dt.datetime]:
        return _parse(self.last_nag_iso)

    @property
    def window_opened(self) -> Optional[_dt.datetime]:
        return _parse(self.window_opened_iso)


def _parse(iso: Optional[str]) -> Optional[_dt.datetime]:
    if not iso:
        return None
    dt = _dt.datetime.fromisoformat(iso)
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


# ── Acknowledgement detection ────────────────────────────────────────────────

@dataclass
class SlackMessage:
    """The slice of a Slack message this module cares about."""

    ts: float
    user: str
    has_image: bool


def find_acknowledgement(
    messages: list[SlackMessage],
    me: str,
    since_ts: float,
) -> Optional[float]:
    """Return the ts of the earliest qualifying screenshot, or None.

    Qualifies when a message is from `me`, carries an image, and landed at or
    after `since_ts` (the moment the nag window opened). Only *your* image
    counts — a bot posting a chart must not silence your own reminder.
    """
    hits = [
        m.ts for m in messages
        if m.user == me and m.has_image and m.ts >= since_ts
    ]
    return min(hits) if hits else None


# ── Message rendering ────────────────────────────────────────────────────────

def render_nag(
    gameweek: int,
    hours_remaining: float,
    countdown: str,
    nag_count: int,
) -> str:
    """The nag body. Escalates in tone and states the exit condition plainly."""
    lines = [
        f"{nag_urgency(hours_remaining)} — Gameweek {gameweek}",
        f"Locks in *{countdown}*.",
        "",
        "Before it locks: captain set · bench ordered · no flagged starters.",
        "",
        "📸 *Reply to this with a screenshot of your lineup and I'll stop.*",
    ]
    if nag_count >= 3:
        lines.append("")
        lines.append(f"(reminder #{nag_count + 1} — I will keep going until you post it)")
    return "\n".join(lines)


def render_ack_confirmation(gameweek: int) -> str:
    return (
        f"✅ Locked in for Gameweek {gameweek}. Nice. "
        "I'll go quiet until the next deadline."
    )
