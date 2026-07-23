"""Tests for gameweek deadline tracking and calendar export."""

from __future__ import annotations

import datetime as _dt

import pytest

from fpl_engine.deadlines import DeadlineTracker

UTC = _dt.timezone.utc

EVENTS = [
    {"id": 1, "name": "Gameweek 1", "deadline_time": "2026-08-21T17:30:00Z",
     "finished": True, "is_current": False, "is_next": False},
    {"id": 2, "name": "Gameweek 2", "deadline_time": "2026-08-28T17:30:00Z",
     "finished": False, "is_current": True, "is_next": False},
    # Midweek round: proof that deadlines are not on a weekly cadence.
    {"id": 3, "name": "Gameweek 3", "deadline_time": "2026-09-01T17:45:00Z",
     "finished": False, "is_current": False, "is_next": True},
    {"id": 4, "name": "Gameweek 4", "deadline_time": "2026-09-12T10:00:00Z",
     "finished": False, "is_current": False, "is_next": False},
]

NOW = _dt.datetime(2026, 8, 27, 12, 0, tzinfo=UTC)


@pytest.fixture
def tracker() -> DeadlineTracker:
    return DeadlineTracker.from_events(EVENTS)


class TestParsing:
    def test_all_events_parsed_and_sorted(self, tracker):
        assert [d.gameweek for d in tracker.deadlines] == [1, 2, 3, 4]

    def test_events_without_a_deadline_are_skipped(self):
        t = DeadlineTracker.from_events(
            EVENTS + [{"id": 5, "name": "Gameweek 5", "deadline_time": None}]
        )
        assert len(t.deadlines) == 4

    def test_deadlines_are_timezone_aware(self, tracker):
        assert all(d.deadline.tzinfo is not None for d in tracker.deadlines)


class TestQueries:
    def test_next_deadline_skips_the_past(self, tracker):
        assert tracker.next_deadline(now=NOW).gameweek == 2

    def test_next_deadline_is_not_the_same_as_current_gameweek(self, tracker):
        """FPL keeps a gameweek `is_current` until its last match ends, so the
        gameweek you are picking for is often the one after `is_current`."""
        after_gw2_locks = _dt.datetime(2026, 8, 29, 12, 0, tzinfo=UTC)
        current = next(d for d in tracker.deadlines if d.is_current)
        assert tracker.next_deadline(now=after_gw2_locks).gameweek != current.gameweek

    def test_upcoming_returns_n_in_order(self, tracker):
        assert [d.gameweek for d in tracker.upcoming(2, now=NOW)] == [2, 3]

    def test_no_deadlines_left_returns_none(self, tracker):
        end_of_season = _dt.datetime(2027, 6, 1, tzinfo=UTC)
        assert tracker.next_deadline(now=end_of_season) is None

    def test_time_remaining_and_countdown(self, tracker):
        gw2 = tracker.for_gameweek(2)
        assert gw2.time_remaining(NOW) == _dt.timedelta(hours=29, minutes=30)
        assert gw2.human_countdown(NOW) == "1d 5h 30m"
        assert not gw2.has_passed(NOW)

    def test_passed_deadline_reads_as_passed(self, tracker):
        gw1 = tracker.for_gameweek(1)
        assert gw1.has_passed(NOW)
        assert "PASSED" in gw1.human_countdown(NOW)

    def test_urgency_window(self, tracker):
        # 29.5h out — inside the default 26h window? No.
        assert not tracker.is_urgent(hours=26, now=NOW)
        # 6h before the GW2 deadline, yes.
        close = _dt.datetime(2026, 8, 28, 11, 30, tzinfo=UTC)
        assert tracker.is_urgent(hours=26, now=close)

    def test_urgency_window_is_wider_than_a_day(self, tracker):
        """A daily job checking a strict 24h window can skip a gameweek whose
        deadline falls just before the next run. 26h closes that gap."""
        assert tracker.is_urgent(hours=26, now=NOW) is False
        just_inside = _dt.datetime(2026, 8, 27, 16, 0, tzinfo=UTC)  # 25.5h out
        assert tracker.is_urgent(hours=26, now=just_inside) is True
        assert tracker.is_urgent(hours=24, now=just_inside) is False


class TestCalendarExport:
    def test_only_future_deadlines_by_default(self, tracker):
        ics = tracker.to_ics(now=NOW)
        assert ics.count("BEGIN:VEVENT") == 3
        assert "fpl-gw1@fpl-engine" not in ics

    def test_include_past_when_asked(self, tracker):
        assert tracker.to_ics(now=NOW, include_past=True).count("BEGIN:VEVENT") == 4

    def test_one_alarm_per_configured_hour(self, tracker):
        ics = tracker.to_ics(now=NOW, alarm_hours=(48, 24, 2))
        assert ics.count("BEGIN:VALARM") == 3 * 3
        assert "TRIGGER:-PT48H" in ics
        assert "TRIGGER:-PT2H" in ics

    def test_structure_is_balanced(self, tracker):
        ics = tracker.to_ics(now=NOW)
        assert ics.startswith("BEGIN:VCALENDAR")
        assert ics.rstrip().endswith("END:VCALENDAR")
        assert ics.count("BEGIN:VEVENT") == ics.count("END:VEVENT")
        assert ics.count("BEGIN:VALARM") == ics.count("END:VALARM")

    def test_crlf_line_endings(self, tracker):
        ics = tracker.to_ics(now=NOW)
        assert "\r\n" in ics
        # No bare LF: every newline must be part of a CRLF pair.
        assert ics.replace("\r\n", "") .count("\n") == 0

    def test_lines_respect_the_octet_limit(self, tracker):
        """Long DESCRIPTION lines must be folded, counting UTF-8 octets —
        unfolded non-ASCII lines are rejected by stricter parsers."""
        for line in tracker.to_ics(now=NOW).split("\r\n"):
            assert len(line.encode("utf-8")) <= 75, line

    def test_folded_lines_rejoin_to_the_original(self, tracker):
        ics = tracker.to_ics(now=NOW)
        unfolded = ics.replace("\r\n ", "")
        assert "Set your lineup for Gameweek 2." in unfolded

    def test_write_ics_creates_the_file(self, tracker, tmp_path):
        path = tracker.write_ics(tmp_path / "nested" / "fpl.ics", now=NOW)
        assert path.exists()
        assert path.read_text().startswith("BEGIN:VCALENDAR")


class TestSummary:
    def test_summary_names_the_next_deadline(self, tracker):
        # `summary` reads the real clock, so pin the tracker to future events.
        future = DeadlineTracker.from_events([{
            "id": 30, "name": "Gameweek 30",
            "deadline_time": "2099-03-01T11:00:00Z",
            "finished": False, "is_current": False, "is_next": True,
        }])
        text = future.summary()
        assert "NEXT DEADLINE" in text
        assert "Gameweek 30" in text

    def test_summary_handles_end_of_season(self):
        past = DeadlineTracker.from_events([{
            "id": 38, "name": "Gameweek 38",
            "deadline_time": "2020-05-01T11:00:00Z",
            "finished": True, "is_current": False, "is_next": False,
        }])
        assert "season is over" in past.summary()
