"""End-to-end check of the wired pipeline: features → models → xP → squad.

The unit tests in `test_upcoming.py` prove the prediction frame is built
correctly. This proves `FPLEngine` actually uses it — the previous code had a
correct feature builder and still shipped skewed predictions because the
inference path took a different route through the data.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fpl_engine.engine import FPLEngine
from fpl_engine.optimizer import GameState

from .test_upcoming import (  # noqa: F401 — pytest fixtures
    N_PLAYERS,
    TARGET_GW,
    fixtures_df,
    history_df,
    players_df,
    teams_df,
)


@pytest.fixture
def engine(history_df, players_df, fixtures_df, teams_df) -> FPLEngine:
    """An engine primed with synthetic data, bypassing the network."""
    eng = FPLEngine()
    eng.players_df = players_df
    eng.fixtures_df = fixtures_df
    eng.teams_df = teams_df
    eng.history_df = history_df
    eng.current_gw = TARGET_GW - 1
    eng.target_gw = TARGET_GW
    eng.build_features(verbose=False)
    eng.train(verbose=False)
    return eng


class TestPipeline:
    def test_predict_returns_one_row_per_player(self, engine):
        preds = engine.predict(horizon=1, verbose=False)
        assert len(preds) == N_PLAYERS
        assert preds["element_id"].is_unique

    def test_predictions_carry_upcoming_fixture_context(self, engine, fixtures_df):
        """The regression guard: xP must be scored against the target GW's
        fixture, which lives in `fixture_predictions_df`."""
        engine.predict(horizon=1, verbose=False)
        detail = engine.fixture_predictions_df

        target_fixture_ids = set(
            fixtures_df[fixtures_df["gameweek"] == TARGET_GW]["fixture_id"]
        )
        assert set(detail["fixture_id"]) <= target_fixture_ids
        assert (detail["target_gw"] == TARGET_GW).all()

    def test_horizon_produces_per_gameweek_columns(self, engine):
        preds = engine.predict(horizon=2, verbose=False)
        assert f"xp_gw{TARGET_GW}" in preds.columns
        assert f"xp_gw{TARGET_GW + 1}" in preds.columns
        # The horizon total is the sum of its parts, not a scaled single week.
        expected = preds[f"xp_gw{TARGET_GW}"] + preds[f"xp_gw{TARGET_GW + 1}"]
        pd.testing.assert_series_equal(
            preds["xp_horizon"], expected, check_names=False
        )

    def test_xp_is_positive_and_finite(self, engine):
        preds = engine.predict(horizon=1, verbose=False)
        assert preds["xp"].notna().all()
        assert (preds["xp"] >= 0).all()
        assert preds["xp"].max() > 0

    def test_unavailable_players_score_zero(
        self, history_df, players_df, fixtures_df, teams_df
    ):
        players = players_df.copy()
        players.loc[players["element_id"] == 5, "status"] = "i"

        eng = FPLEngine()
        eng.players_df = players
        eng.fixtures_df = fixtures_df
        eng.teams_df = teams_df
        eng.history_df = history_df
        eng.target_gw = TARGET_GW
        eng.build_features(verbose=False)
        eng.train(verbose=False)
        preds = eng.predict(horizon=1, verbose=False)

        injured = preds[preds["element_id"] == 5].iloc[0]
        assert injured["xp"] == 0.0
        assert injured["unavailable"]

    def test_optimize_excludes_unavailable_players(
        self, history_df, players_df, fixtures_df, teams_df
    ):
        players = players_df.copy()
        # Flag one player per position so exclusion cannot make the squad illegal.
        flagged = (
            players.groupby("position").head(1)["element_id"].tolist()
        )
        players.loc[players["element_id"].isin(flagged), "status"] = "i"

        eng = FPLEngine()
        eng.players_df = players
        eng.fixtures_df = fixtures_df
        eng.teams_df = teams_df
        eng.history_df = history_df
        eng.target_gw = TARGET_GW
        eng.build_features(verbose=False)
        eng.train(verbose=False)
        eng.predict(horizon=1, verbose=False)

        result = eng.optimize(gamestate=GameState.NEUTRAL, verbose=False)
        assert not set(result.squad["element_id"]) & set(flagged)

    def test_optimize_produces_a_legal_squad(self, engine):
        engine.predict(horizon=1, verbose=False)
        result = engine.optimize(gamestate=GameState.NEUTRAL, verbose=False)

        assert len(result.squad) == 15
        assert len(result.starting_xi) == 11
        assert len(result.bench) == 4
        assert result.squad["price"].sum() <= 1000
        assert (result.squad["team_id"].value_counts() <= 3).all()
        assert result.captain_id != result.vice_captain_id

    def test_report_includes_the_deadline_when_known(self, engine):
        from fpl_engine.deadlines import DeadlineTracker

        engine.deadlines = DeadlineTracker.from_events([{
            "id": TARGET_GW,
            "name": f"Gameweek {TARGET_GW}",
            "deadline_time": "2099-01-01T11:00:00Z",
            "finished": False,
            "is_current": False,
            "is_next": True,
        }])
        engine.predict(horizon=1, verbose=False)
        report = engine.report(engine.optimize(verbose=False))

        assert "DEADLINE" in report
        assert f"Gameweek {TARGET_GW}" in report or "GAMEWEEK REPORT" in report

    def test_blank_gameweek_players_are_not_selected(
        self, history_df, players_df, fixtures_df, teams_df
    ):
        # Teams 1 and 2 blank in the target gameweek.
        fx = fixtures_df[
            ~((fixtures_df["gameweek"] == TARGET_GW)
              & (fixtures_df["home_team_id"].isin([1, 2])))
        ]

        eng = FPLEngine()
        eng.players_df = players_df
        eng.fixtures_df = fx
        eng.teams_df = teams_df
        eng.history_df = history_df
        eng.target_gw = TARGET_GW
        eng.build_features(verbose=False)
        eng.train(verbose=False)
        preds = eng.predict(horizon=1, verbose=False)

        blanking = players_df[players_df["team_id"].isin([1, 2])]["element_id"]
        blanked = preds[preds["element_id"].isin(blanking)]
        assert (blanked["n_fixtures"] == 0).all()
        assert (blanked["xp"] == 0).all()
