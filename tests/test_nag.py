"""Tests for the escalating lineup nag loop."""

from __future__ import annotations

import datetime as _dt
import json

import pytest

from fpl_engine.deadlines import DeadlineTracker
from fpl_engine.nag import (
    NagState,
    SlackMessage,
    band_for,
    find_acknowledgement,
    nag_urgency,
    render_nag,
    should_nag,
)
from fpl_engine.slack_gateway import DryRunGateway, _to_message

UTC = _dt.timezone.utc
NOW = _dt.datetime(2026, 8, 21, 12, 0, tzinfo=UTC)


def hours_ago(h: float) -> _dt.datetime:
    return NOW - _dt.timedelta(hours=h)


class TestEscalationLadder:
    def test_outside_window_no_band(self):
        assert band_for(48) is None
        assert band_for(31) is None

    def test_bands_tighten_toward_deadline(self):
        assert band_for(28)[2] == 360   # 6h cadence far out
        assert band_for(8)[2] == 180    # 3h
        assert band_for(4)[2] == 60     # hourly
        assert band_for(1)[2] == 20     # every 20 min at the wire

    def test_passed_deadline_no_band(self):
        assert band_for(0) is None
        assert band_for(-1) is None


class TestShouldNag:
    def test_no_nag_outside_window(self):
        assert should_nag(40, None, NOW) is False

    def test_first_nag_in_window_fires(self):
        assert should_nag(20, None, NOW) is True

    def test_respects_interval_within_band(self):
        # 12-24h band = 6h cadence.
        assert should_nag(20, hours_ago(1), NOW) is False   # only 1h since last
        assert should_nag(20, hours_ago(7), NOW) is True    # 7h since last

    def test_tighter_band_fires_sooner(self):
        # <2h band = 20 min cadence.
        assert should_nag(1.5, hours_ago(0.2), NOW) is False   # 12 min
        assert should_nag(1.5, hours_ago(0.5), NOW) is True    # 30 min

    def test_urgency_prefixes_escalate(self):
        assert "UNDER 2 HOURS" in nag_urgency(1)
        assert nag_urgency(20) == "⏰ Deadline reminder"


class TestAcknowledgementDetection:
    def test_my_screenshot_counts(self):
        msgs = [SlackMessage(ts=1050.0, user="U_ME", has_image=True)]
        assert find_acknowledgement(msgs, me="U_ME", since_ts=1000.0) == 1050.0

    def test_my_text_reply_does_not_count(self):
        msgs = [SlackMessage(ts=1050.0, user="U_ME", has_image=False)]
        assert find_acknowledgement(msgs, me="U_ME", since_ts=1000.0) is None

    def test_someone_elses_image_does_not_count(self):
        """A bot posting a chart must not silence your reminder."""
        msgs = [SlackMessage(ts=1050.0, user="U_BOT", has_image=True)]
        assert find_acknowledgement(msgs, me="U_ME", since_ts=1000.0) is None

    def test_old_image_before_window_does_not_count(self):
        msgs = [SlackMessage(ts=900.0, user="U_ME", has_image=True)]
        assert find_acknowledgement(msgs, me="U_ME", since_ts=1000.0) is None

    def test_earliest_qualifying_image_wins(self):
        msgs = [
            SlackMessage(ts=1200.0, user="U_ME", has_image=True),
            SlackMessage(ts=1050.0, user="U_ME", has_image=True),
        ]
        assert find_acknowledgement(msgs, me="U_ME", since_ts=1000.0) == 1050.0


class TestSlackMessageMapping:
    def test_image_file_flagged(self):
        raw = {"ts": "1.0", "user": "U_ME",
               "files": [{"mimetype": "image/png", "filetype": "png"}]}
        assert _to_message(raw).has_image is True

    def test_pdf_attachment_not_an_image(self):
        raw = {"ts": "1.0", "user": "U_ME",
               "files": [{"mimetype": "application/pdf", "filetype": "pdf"}]}
        assert _to_message(raw).has_image is False

    def test_plain_message_no_image(self):
        assert _to_message({"ts": "1.0", "user": "U_ME"}).has_image is False


class TestNagStateRoundTrip:
    def test_serialise_and_restore(self):
        s = NagState.new(5, NOW)
        s.nag_count = 3
        s.thread_ts = "1234.5678"
        restored = NagState.from_dict(json.loads(json.dumps(s.to_dict())))
        assert restored.gameweek == 5
        assert restored.nag_count == 3
        assert restored.thread_ts == "1234.5678"


# ── Integration: the loop against a fake gateway and fixed clock ─────────────

def _tracker(deadline: _dt.datetime) -> DeadlineTracker:
    return DeadlineTracker.from_events([{
        "id": 1, "name": "Gameweek 1",
        "deadline_time": deadline.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "finished": False, "is_current": False, "is_next": True,
    }])


class TestLoop:
    def test_quiet_when_far_out(self, tmp_path):
        from scripts.lineup_nag import run

        gw = DryRunGateway()
        deadline = NOW + _dt.timedelta(hours=40)
        run(gw, _tracker(deadline), tmp_path / "s.json", me="U_ME", now=NOW)
        assert gw.posted == []

    def test_nags_inside_window(self, tmp_path):
        from scripts.lineup_nag import run

        gw = DryRunGateway()
        deadline = NOW + _dt.timedelta(hours=1)
        run(gw, _tracker(deadline), tmp_path / "s.json", me="U_ME", now=NOW)
        assert len(gw.posted) == 1
        assert "Gameweek 1" in gw.posted[0][0]

    def test_does_not_double_nag_within_interval(self, tmp_path):
        from scripts.lineup_nag import run

        state = tmp_path / "s.json"
        deadline = NOW + _dt.timedelta(hours=1)  # 20-min cadence band
        gw = DryRunGateway()

        run(gw, _tracker(deadline), state, me="U_ME", now=NOW)
        # 10 minutes later: too soon.
        run(gw, _tracker(deadline), state, me="U_ME",
            now=NOW + _dt.timedelta(minutes=10))
        assert len(gw.posted) == 1

        # 25 minutes after the first: due again.
        run(gw, _tracker(deadline), state, me="U_ME",
            now=NOW + _dt.timedelta(minutes=25))
        assert len(gw.posted) == 2

    def test_screenshot_silences_and_confirms(self, tmp_path):
        from scripts.lineup_nag import run

        state = tmp_path / "s.json"
        deadline = NOW + _dt.timedelta(hours=1)

        gw = DryRunGateway()
        run(gw, _tracker(deadline), state, me="U_ME", now=NOW)
        assert len(gw.posted) == 1

        # Screenshot lands, then the loop runs again.
        later = NOW + _dt.timedelta(minutes=25)
        gw._inbound = [SlackMessage(ts=later.timestamp(), user="U_ME", has_image=True)]
        run(gw, _tracker(deadline), state, me="U_ME", now=later)

        # A confirmation went out, and state is now acknowledged.
        assert "Locked in" in gw.posted[-1][0]
        saved = json.loads(state.read_text())
        assert saved["acknowledged"] is True

        # Further runs stay silent.
        even_later = NOW + _dt.timedelta(minutes=60)
        run(gw, _tracker(deadline), state, me="U_ME", now=even_later)
        assert len(gw.posted) == 2  # no new nag

    def test_state_resets_on_new_gameweek(self, tmp_path):
        from scripts.lineup_nag import run

        state = tmp_path / "s.json"
        gw = DryRunGateway()

        # GW1 acknowledged.
        d1 = NOW + _dt.timedelta(hours=1)
        run(gw, _tracker(d1), state, me="U_ME", now=NOW)
        gw._inbound = [SlackMessage(ts=(NOW + _dt.timedelta(minutes=5)).timestamp(),
                                    user="U_ME", has_image=True)]
        run(gw, _tracker(d1), state, me="U_ME", now=NOW + _dt.timedelta(minutes=5))
        assert json.loads(state.read_text())["acknowledged"] is True

        # A GW2 deadline appears within the window. State must reset and nag.
        gw._inbound = []
        d2 = NOW + _dt.timedelta(hours=1)
        tracker2 = DeadlineTracker.from_events([{
            "id": 2, "name": "Gameweek 2",
            "deadline_time": d2.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            "finished": False, "is_current": False, "is_next": True,
        }])
        posts_before = len(gw.posted)
        run(gw, tracker2, state, me="U_ME", now=NOW + _dt.timedelta(minutes=10))
        assert len(gw.posted) == posts_before + 1
        saved = json.loads(state.read_text())
        assert saved["gameweek"] == 2
        assert saved["acknowledged"] is False
