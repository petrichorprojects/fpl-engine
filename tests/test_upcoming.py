"""Tests for upcoming-fixture prediction frames.

These cover the train/serve skew fix: prediction rows must describe the fixture
about to be played, not the one just played.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpl_engine.points_model import PointsModel
from fpl_engine.upcoming import build_upcoming_frame

N_PLAYERS = 40
PLAYED_GWS = 8
TARGET_GW = PLAYED_GWS + 1


@pytest.fixture
def teams_df() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "id": range(1, 21),
        "name": [f"Team {i}" for i in range(1, 21)],
        "short_name": [f"T{i:02d}" for i in range(1, 21)],
        "strength_overall_home": rng.integers(1000, 1400, 20),
        "strength_overall_away": rng.integers(1000, 1400, 20),
        "strength_attack_home": rng.integers(1000, 1400, 20),
        "strength_attack_away": rng.integers(1000, 1400, 20),
        "strength_defence_home": rng.integers(1000, 1400, 20),
        "strength_defence_away": rng.integers(1000, 1400, 20),
    })


@pytest.fixture
def players_df() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    positions = ["GKP"] * 4 + ["DEF"] * 12 + ["MID"] * 14 + ["FWD"] * 10
    team_ids = [(i % 20) + 1 for i in range(N_PLAYERS)]
    return pd.DataFrame({
        "element_id": range(1, N_PLAYERS + 1),
        "name": [f"Player{i}" for i in range(1, N_PLAYERS + 1)],
        "full_name": [f"Player {i}" for i in range(1, N_PLAYERS + 1)],
        "position": positions,
        "team_id": team_ids,
        "team_name": [f"Team {t}" for t in team_ids],
        "team_short": [f"T{t:02d}" for t in team_ids],
        "price": rng.integers(40, 130, N_PLAYERS),
        "selected_pct": rng.uniform(0.1, 60.0, N_PLAYERS),
        "status": ["a"] * N_PLAYERS,
        "chance_next_round": [None] * N_PLAYERS,
        "form": rng.uniform(0, 8, N_PLAYERS),
        "news": [""] * N_PLAYERS,
        "fpl_code": range(1000, 1000 + N_PLAYERS),
    })


def _fixture_rows(gw: int, finished: bool, base_id: int, kickoff: str) -> list[dict]:
    """Ten fixtures pairing team 2k-1 vs 2k, with the venue alternating by
    gameweek so a test can tell "upcoming fixture" apart from "last fixture"."""
    rows = []
    for i in range(10):
        a, b = 2 * i + 1, 2 * i + 2
        home, away = (a, b) if gw % 2 == 0 else (b, a)
        rows.append({
            "fixture_id": base_id + i,
            "gameweek": gw,
            "home_team_id": home,
            "away_team_id": away,
            "home_score": 1 if finished else None,
            "away_score": 1 if finished else None,
            "finished": finished,
            "kickoff_time": kickoff,
            "home_difficulty": 3,
            "away_difficulty": 3,
        })
    return rows


@pytest.fixture
def fixtures_df() -> pd.DataFrame:
    rows = []
    for gw in range(1, PLAYED_GWS + 1):
        rows += _fixture_rows(gw, True, gw * 100, f"2026-01-{gw:02d}T15:00:00Z")
    # Upcoming, unplayed
    rows += _fixture_rows(TARGET_GW, False, TARGET_GW * 100, f"2026-01-{TARGET_GW:02d}T15:00:00Z")
    rows += _fixture_rows(TARGET_GW + 1, False, (TARGET_GW + 1) * 100,
                          f"2026-01-{TARGET_GW + 1:02d}T15:00:00Z")
    return pd.DataFrame(rows)


@pytest.fixture
def history_df(players_df, fixtures_df) -> pd.DataFrame:
    """Per-fixture history for played gameweeks."""
    rng = np.random.default_rng(23)
    played = fixtures_df[fixtures_df["finished"]]
    rows = []
    for _, p in players_df.iterrows():
        for _, f in played.iterrows():
            if p["team_id"] == f["home_team_id"]:
                opponent, was_home = f["away_team_id"], True
            elif p["team_id"] == f["away_team_id"]:
                opponent, was_home = f["home_team_id"], False
            else:
                continue
            minutes = int(rng.choice([0, 20, 90], p=[0.2, 0.2, 0.6]))
            rows.append({
                "element_id": p["element_id"],
                "fixture": f["fixture_id"],
                "round": f["gameweek"],
                "opponent_team": opponent,
                "was_home": was_home,
                "kickoff_time": f["kickoff_time"],
                "minutes": minutes,
                "starts": 1 if minutes >= 60 else 0,
                "total_points": int(rng.integers(0, 14)),
                "goals_scored": int(rng.integers(0, 2)),
                "assists": int(rng.integers(0, 2)),
                "clean_sheets": int(rng.integers(0, 2)),
                "goals_conceded": int(rng.integers(0, 3)),
                "saves": int(rng.integers(0, 5)),
                "bonus": int(rng.integers(0, 4)),
                "bps": int(rng.integers(0, 40)),
                "influence": float(rng.uniform(0, 50)),
                "creativity": float(rng.uniform(0, 50)),
                "threat": float(rng.uniform(0, 60)),
                "ict_index": float(rng.uniform(0, 15)),
                "expected_goals": float(rng.uniform(0, 0.8)),
                "expected_assists": float(rng.uniform(0, 0.5)),
                "expected_goal_involvements": float(rng.uniform(0, 1.2)),
                "expected_goals_conceded": float(rng.uniform(0, 2)),
                "value": int(p["price"]),
                "selected": int(rng.integers(1000, 900000)),
                "transfers_in": int(rng.integers(0, 50000)),
                "transfers_out": int(rng.integers(0, 50000)),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def frame(history_df, players_df, fixtures_df, teams_df) -> pd.DataFrame:
    return build_upcoming_frame(
        history_df=history_df,
        players_df=players_df,
        fixtures_df=fixtures_df,
        teams_df=teams_df,
        target_gws=[TARGET_GW],
    )


class TestUpcomingFixtureAlignment:
    """AC3 — prediction rows describe the upcoming fixture, not the last one."""

    def test_frame_is_not_empty(self, frame):
        assert not frame.empty
        assert frame["element_id"].nunique() == N_PLAYERS

    def test_opponent_matches_the_target_gameweek_fixture(self, frame, fixtures_df):
        """The core regression: opponent must come from the target GW."""
        target_fixtures = fixtures_df[fixtures_df["gameweek"] == TARGET_GW]

        expected = {}
        for _, f in target_fixtures.iterrows():
            expected[f["home_team_id"]] = (f["away_team_id"], True)
            expected[f["away_team_id"]] = (f["home_team_id"], False)

        for _, row in frame.iterrows():
            exp_opponent, exp_home = expected[row["team_id"]]
            assert row["opponent_team"] == exp_opponent
            assert bool(row["was_home"]) is exp_home

    def test_venue_differs_from_last_played_fixture(self, frame, history_df):
        """Home/away alternates in this fixture set, so carrying the last
        played row forward would invert `was_home` for every player."""
        last_played = (
            history_df.sort_values("round")
            .groupby("element_id")
            .last()["was_home"]
        )
        upcoming_home = frame.set_index("element_id")["was_home"].astype(bool)
        # Not merely equal-by-accident: at least some players must differ.
        differing = (upcoming_home != last_played.reindex(upcoming_home.index)).sum()
        assert differing > 0, "venue never changed — the fixture join is not working"

    def test_opponent_strength_matches_the_upcoming_opponent(self, frame, teams_df):
        strengths = teams_df.set_index("id")
        for _, row in frame.head(15).iterrows():
            opp = row["opponent_team"]
            expected = (
                strengths.loc[opp, "strength_overall_home"]
                + strengths.loc[opp, "strength_overall_away"]
            )
            assert row["opp_strength_overall"] == expected

    def test_rolling_form_includes_the_most_recent_match(self, frame, history_df):
        """The shifted-window bug: features built from the last played row
        exclude that match. The carry frame must include it."""
        player = frame.iloc[0]["element_id"]
        player_history = history_df[history_df["element_id"] == player]
        expected = player_history.sort_values("round")["total_points"].tail(3).mean()
        actual = frame[frame["element_id"] == player].iloc[0]["roll_pts_3"]
        assert actual == pytest.approx(expected, rel=1e-6)

    def test_rest_days_measured_to_the_upcoming_kickoff(self, frame):
        assert (frame["days_since_last"] >= 0).all()
        assert frame["days_since_last"].notna().all()


class TestDoublesAndBlanks:
    """AC4 — one row per fixture; blanks produce no rows."""

    def test_double_gameweek_yields_two_rows(
        self, history_df, players_df, fixtures_df, teams_df
    ):
        # Give teams 1 and 2 a second fixture in the target gameweek.
        extra = pd.DataFrame([{
            "fixture_id": 9999,
            "gameweek": TARGET_GW,
            "home_team_id": 1,
            "away_team_id": 2,
            "home_score": None,
            "away_score": None,
            "finished": False,
            "kickoff_time": f"2026-01-{TARGET_GW:02d}T19:45:00Z",
            "home_difficulty": 3,
            "away_difficulty": 3,
        }])
        fx = pd.concat([fixtures_df, extra], ignore_index=True)

        frame = build_upcoming_frame(history_df, players_df, fx, teams_df, [TARGET_GW])

        doubled = frame[frame["team_id"].isin([1, 2])]
        assert (doubled.groupby("element_id").size() == 2).all()
        assert (doubled["n_fixtures_in_gw"] == 2).all()

        single = frame[~frame["team_id"].isin([1, 2])]
        assert (single.groupby("element_id").size() == 1).all()

    def test_blank_gameweek_yields_no_rows(
        self, history_df, players_df, fixtures_df, teams_df
    ):
        # Remove the fixture involving teams 1 and 2 from the target gameweek.
        fx = fixtures_df[
            ~((fixtures_df["gameweek"] == TARGET_GW)
              & (fixtures_df["home_team_id"].isin([1, 2])))
        ]
        frame = build_upcoming_frame(history_df, players_df, fx, teams_df, [TARGET_GW])

        blanking_players = players_df[players_df["team_id"].isin([1, 2])]["element_id"]
        assert frame[frame["element_id"].isin(blanking_players)].empty
        # Everyone else still has exactly one row.
        assert frame["element_id"].nunique() == N_PLAYERS - len(blanking_players)

    def test_multi_gameweek_horizon(
        self, history_df, players_df, fixtures_df, teams_df
    ):
        frame = build_upcoming_frame(
            history_df, players_df, fixtures_df, teams_df,
            [TARGET_GW, TARGET_GW + 1],
        )
        assert set(frame["target_gw"]) == {TARGET_GW, TARGET_GW + 1}
        assert (frame.groupby("element_id").size() == 2).all()


class TestAvailabilityOverrides:
    """AC6 — hard availability facts are encoded, not inferred."""

    def test_injured_player_flagged_unavailable(
        self, history_df, players_df, fixtures_df, teams_df
    ):
        players = players_df.copy()
        players.loc[players["element_id"] == 1, "status"] = "i"
        players.loc[players["element_id"] == 2, "status"] = "s"  # suspended

        frame = build_upcoming_frame(history_df, players, fixtures_df, teams_df,
                                     [TARGET_GW])

        flagged = frame[frame["element_id"].isin([1, 2])]
        assert flagged["unavailable"].all()
        # Null chance-of-playing must resolve to 0, not the old default of 100.
        assert (flagged["chance_next_round"] == 0).all()

    def test_available_player_defaults_to_full_chance(self, frame):
        fit = frame[frame["element_id"] == 3].iloc[0]
        assert not fit["unavailable"]
        assert fit["chance_next_round"] == 100


class TestGoalkeeperLogShift:
    """AC5 — the GKP log transform reverses with the training-time offset."""

    def test_shift_is_persisted_and_reused(self, tmp_path):
        rng = np.random.default_rng(3)
        n = 300
        df = pd.DataFrame({
            "element_id": rng.integers(1, 30, n),
            "position": "GKP",
            "minutes": 90,
            # Includes negative points, which is why the shift exists at all.
            "total_points": rng.integers(-2, 12, n),
        })
        for col in PointsModel().feature_cols:
            df[col] = rng.uniform(0, 5, n)

        model = PointsModel()
        model.train(df, n_splits=2, verbose=False)

        assert "GKP" in model.log_shifts
        shift = model.log_shifts["GKP"]

        model.save(tmp_path / "points")
        reloaded = PointsModel()
        reloaded.load(tmp_path / "points")

        assert reloaded.log_shifts["GKP"] == shift

        # A prediction population with a different points distribution must not
        # change the transform — that was the bug.
        subset = df.head(20).copy()
        subset["total_points"] = 5
        before = model.predict(df.head(20))["e_pts_start"].values
        after = model.predict(subset)["e_pts_start"].values
        np.testing.assert_allclose(before, after)


class TestTransferHorizon:
    """AC7 — transfer value sums real per-gameweek xP."""

    def test_horizon_column_is_used_when_present(self):
        from fpl_engine.optimizer import FPLOptimizer

        squad = pd.DataFrame({
            "element_id": [1],
            "position": ["MID"],
            "team_id": [1],
            "price": [80],
            "name": ["Out"],
            "xp": [5.0],
            "xp_horizon": [15.0],   # 3 flat gameweeks
        })
        pool = pd.DataFrame({
            "element_id": [2],
            "position": ["MID"],
            "team_id": [5],
            "price": [80],
            "name": ["In"],
            "xp": [5.0],            # identical next GW...
            "xp_horizon": [24.0],   # ...but a far better run of fixtures
        })

        opt = FPLOptimizer()
        transfers = opt.optimize_transfers(squad, pool, free_transfers=1, horizon=3)

        assert transfers
        # Old behaviour: (5.0 - 5.0) * 3 == 0 — the fixture swing was invisible.
        assert transfers[0]["xp_gain"] == pytest.approx(9.0)
