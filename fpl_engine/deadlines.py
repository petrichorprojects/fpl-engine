"""Gameweek deadlines — the thing that actually loses you points.

Missing a deadline costs more than any modelling error in this repo. A default
lineup with a benched captain and three flagged players is a 20-30 point swing;
a marginal xG improvement is worth fractions of a point.

FPL deadlines are *not* on a fixed weekly cadence. They are 90 minutes before the
first kickoff of the round, which moves with TV scheduling, midweek rounds, and
international breaks. The only reliable source is `deadline_time` on each event in
the bootstrap payload — which, before this module, nothing in this repo read.

Usage:

    from fpl_engine.deadlines import DeadlineTracker

    dt = DeadlineTracker.from_client(client)
    nxt = dt.next_deadline()
    print(nxt.gameweek, nxt.human_countdown())

    # One-time calendar import covering the rest of the season
    dt.write_ics("fpl-deadlines.ics")
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

UTC = _dt.timezone.utc

# Alarms fired ahead of each deadline, in hours. 24h gives you time to react to
# Friday press conferences; 2h is the last call before the lineup locks.
DEFAULT_ALARM_HOURS = (24, 2)


@dataclass(frozen=True)
class Deadline:
    """A single gameweek deadline."""

    gameweek: int
    name: str
    deadline: _dt.datetime  # timezone-aware, UTC
    finished: bool
    is_current: bool
    is_next: bool

    def time_remaining(self, now: _dt.datetime | None = None) -> _dt.timedelta:
        """Time until this deadline. Negative once it has passed."""
        now = now or _dt.datetime.now(UTC)
        return self.deadline - now

    def has_passed(self, now: _dt.datetime | None = None) -> bool:
        return self.time_remaining(now).total_seconds() <= 0

    def human_countdown(self, now: _dt.datetime | None = None) -> str:
        """Readable countdown, e.g. '2d 4h 15m' or 'PASSED 3h ago'."""
        delta = self.time_remaining(now)
        secs = int(delta.total_seconds())
        past = secs < 0
        secs = abs(secs)

        days, rem = divmod(secs, 86400)
        hours, rem = divmod(rem, 3600)
        minutes = rem // 60

        parts = []
        if days:
            parts.append(f"{days}d")
        if hours or days:
            parts.append(f"{hours}h")
        parts.append(f"{minutes}m")
        body = " ".join(parts)

        return f"PASSED {body} ago" if past else body

    def local(self, tz: _dt.tzinfo | None = None) -> _dt.datetime:
        """Deadline in local time (system timezone by default)."""
        return self.deadline.astimezone(tz)


@dataclass
class DeadlineTracker:
    """All gameweek deadlines for the season."""

    deadlines: list[Deadline]

    # ── Constructors ─────────────────────────────────────────────────────

    @classmethod
    def from_events(cls, events: list[dict] | pd.DataFrame) -> "DeadlineTracker":
        """Build from `bootstrap['events']` (list of dicts or DataFrame)."""
        if isinstance(events, pd.DataFrame):
            records = events.to_dict("records")
        else:
            records = list(events)

        out: list[Deadline] = []
        for e in records:
            raw = e.get("deadline_time")
            if not raw:
                continue
            parsed = pd.to_datetime(raw, utc=True, errors="coerce")
            if pd.isna(parsed):
                continue
            out.append(
                Deadline(
                    gameweek=int(e["id"]),
                    name=str(e.get("name", f"Gameweek {e['id']}")),
                    deadline=parsed.to_pydatetime(),
                    finished=bool(e.get("finished", False)),
                    is_current=bool(e.get("is_current", False)),
                    is_next=bool(e.get("is_next", False)),
                )
            )

        out.sort(key=lambda d: d.deadline)
        return cls(deadlines=out)

    @classmethod
    def from_client(cls, client) -> "DeadlineTracker":
        """Build from an `FPLClient`. Note: bootstrap is cached on disk, so
        call `client.clear_cache()` first if the cache may be stale."""
        return cls.from_events(client.bootstrap()["events"])

    # ── Queries ──────────────────────────────────────────────────────────

    def next_deadline(self, now: _dt.datetime | None = None) -> Deadline | None:
        """The next deadline that has not yet passed."""
        now = now or _dt.datetime.now(UTC)
        upcoming = [d for d in self.deadlines if d.deadline > now]
        return upcoming[0] if upcoming else None

    def upcoming(self, n: int = 5, now: _dt.datetime | None = None) -> list[Deadline]:
        """The next `n` deadlines."""
        now = now or _dt.datetime.now(UTC)
        return [d for d in self.deadlines if d.deadline > now][:n]

    def for_gameweek(self, gw: int) -> Deadline | None:
        for d in self.deadlines:
            if d.gameweek == gw:
                return d
        return None

    def is_urgent(self, hours: float = 26.0, now: _dt.datetime | None = None) -> bool:
        """True when the next deadline is inside the alert window.

        Default 26h so a daily morning job never skips a gameweek: a 24h window
        checked once a day can miss a deadline that lands just before the next run.
        """
        nxt = self.next_deadline(now)
        if nxt is None:
            return False
        return 0 < nxt.time_remaining(now).total_seconds() <= hours * 3600

    # ── Calendar export ──────────────────────────────────────────────────

    def to_ics(
        self,
        alarm_hours: tuple[int, ...] = DEFAULT_ALARM_HOURS,
        include_past: bool = False,
        now: _dt.datetime | None = None,
    ) -> str:
        """Render remaining deadlines as an iCalendar feed.

        Each deadline becomes a 15-minute event with one alarm per entry in
        `alarm_hours`. Import once and every remaining deadline of the season
        lands in your calendar with reminders attached.
        """
        now = now or _dt.datetime.now(UTC)
        items = self.deadlines if include_past else [
            d for d in self.deadlines if d.deadline > now
        ]

        lines = [
            "BEGIN:VCALENDAR",
            "VERSION:2.0",
            "PRODID:-//fpl-engine//Gameweek Deadlines//EN",
            "CALSCALE:GREGORIAN",
            "METHOD:PUBLISH",
            "X-WR-CALNAME:FPL Deadlines",
            "X-WR-TIMEZONE:UTC",
        ]

        for d in items:
            start = d.deadline
            end = start + _dt.timedelta(minutes=15)
            lines += [
                "BEGIN:VEVENT",
                f"UID:fpl-gw{d.gameweek}@fpl-engine",
                f"DTSTAMP:{_ics_ts(now)}",
                f"DTSTART:{_ics_ts(start)}",
                f"DTEND:{_ics_ts(end)}",
                f"SUMMARY:FPL deadline — {d.name}",
                (
                    f"DESCRIPTION:Set your lineup for {d.name}. "
                    "Check: captain\\, bench order\\, flagged players\\, "
                    "and whether anyone is suspended."
                ),
                "TRANSP:TRANSPARENT",
            ]
            for h in alarm_hours:
                lines += [
                    "BEGIN:VALARM",
                    "ACTION:DISPLAY",
                    f"DESCRIPTION:FPL deadline in {h}h — {d.name}",
                    f"TRIGGER:-PT{int(h)}H",
                    "END:VALARM",
                ]
            lines.append("END:VEVENT")

        lines.append("END:VCALENDAR")
        # RFC 5545: CRLF line endings, and content lines folded at 75 octets.
        folded = [part for line in lines for part in _fold(line)]
        return "\r\n".join(folded) + "\r\n"

    def write_ics(self, path: str | Path, **kwargs) -> Path:
        """Write the calendar feed to disk and return the path."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_ics(**kwargs))
        return p

    # ── Reporting ────────────────────────────────────────────────────────

    def summary(self, n: int = 5, tz: _dt.tzinfo | None = None) -> str:
        """Human-readable block for briefs and Slack messages."""
        nxt = self.next_deadline()
        if nxt is None:
            return "No remaining gameweek deadlines — season is over."

        lines = [
            f"NEXT DEADLINE: {nxt.name}",
            f"  {nxt.local(tz).strftime('%a %d %b, %H:%M %Z')}",
            f"  Time remaining: {nxt.human_countdown()}",
        ]

        rest = self.upcoming(n)[1:]
        if rest:
            lines.append("")
            lines.append("  Then:")
            for d in rest:
                lines.append(
                    f"    {d.name:<14} {d.local(tz).strftime('%a %d %b %H:%M')}"
                )
        return "\n".join(lines)


def _ics_ts(dt: _dt.datetime) -> str:
    """Format a datetime as an iCalendar UTC timestamp."""
    return dt.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def _fold(line: str, limit: int = 73) -> list[str]:
    """Fold a content line to RFC 5545's 75-octet limit.

    The limit counts octets, not characters, so folding is done on the UTF-8
    encoding — a naive character-count fold overruns on any line containing
    non-ASCII, and stricter parsers (Apple Calendar among them) reject the file.
    Continuation lines start with a single space, which the parser strips.
    """
    encoded = line.encode("utf-8")
    if len(encoded) <= limit + 2:
        return [line]

    parts: list[str] = []
    buf = ""
    buf_len = 0
    budget = limit

    for ch in line:
        ch_len = len(ch.encode("utf-8"))
        if buf_len + ch_len > budget:
            parts.append(buf)
            buf = ch
            buf_len = ch_len
            budget = limit - 1  # continuation lines carry a leading space
        else:
            buf += ch
            buf_len += ch_len

    if buf:
        parts.append(buf)

    return [parts[0]] + [" " + p for p in parts[1:]]


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> int:
    import argparse

    from .client import FPLClient

    parser = argparse.ArgumentParser(description="FPL gameweek deadlines")
    parser.add_argument("--ics", metavar="PATH",
                        help="Write an .ics calendar of remaining deadlines")
    parser.add_argument("--next", type=int, default=5, metavar="N",
                        help="How many upcoming deadlines to list (default: 5)")
    parser.add_argument("--fresh", action="store_true",
                        help="Clear the API cache before reading deadlines")
    args = parser.parse_args()

    client = FPLClient()
    if args.fresh:
        client.clear_cache()

    tracker = DeadlineTracker.from_client(client)
    print(tracker.summary(n=args.next))

    if args.ics:
        path = tracker.write_ics(args.ics)
        n = len(tracker.upcoming(n=99))
        print(f"\nWrote {n} deadlines to {path}")
        print("Import it into your calendar — alarms fire 24h and 2h before each.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
