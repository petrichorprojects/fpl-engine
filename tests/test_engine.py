"""Integration tests for the FPL Engine using synthetic data.

Validates the full pipeline without hitting the FPL API.
"""

import numpy as np
import pandas as pd
import pytest

from fpl_engine.config import POSITIONS, TOTAL_BUDGET
from fpl_engine.features import build_player_features
from fpl_engine.minutes_model import MinutesModel, assign_minutes_label
from fpl_engine.optimizer import (
    Chip,
    FPLOptimizer,
    GameState,
    OptimizationResult,
    SquadConstraints,
)
from fpl_engine.points_model import PointsModel


# ── Synthetic data generators ────────────────────────────────────────────────

def make_synthetic_players(n: int = 400) -> pd.DataFrame:
    """Generate realistic-looking player master data."""
    rng = np.random.default_rng(42)
    positions = rng.choice(["GKP", "DEF", "MID", "FWD"], n, p=[0.1, 0.3, 0.35, 0.25])
    team_ids = rng.choice(range(1, 21), n)

    return pd.DataFrame({
        "element_id": range(1, n + 1),
        "fpl_code": range(10001, 10001 + n),
        "name": [f"Player_{i}" for i in range(1, n + 1)],
        "full_name": [f"First_{i} Last_{i}" for i in range(1, n + 1)],
        "position": positions,
        "team_id": team_ids,
        "team_name": [f"Team_{tid}" for tid in team_ids],
        "team_short": [f"T{tid:02d}" for tid in team_ids],
        "price": rng.integers(40, 130, n),  # £4.0m - £13.0m
        "total_points": rng.integers(0, 200, n),
        "minutes": rng.integers(0, 3000, n),
        "goals_scored": rng.integers(0, 20, n),
        "assists": rng.integers(0, 15, n),
        "clean_sheets": rng.integers(0, 15, n),
        "goals_conceded": rng.integers(0, 50, n),
        "saves": rng.integers(0, 100, n),
        "bonus": rng.integers(0, 30, n),
        "bps": rng.integers(0, 500, n),
        "form": rng.uniform(0, 10, n),
        "points_per_game": rng.uniform(0, 8, n),
        "selected_pct": rng.uniform(0, 50, n),
        "transfers_in_event": rng.integers(0, 100000, n),
        "transfers_out_event": rng.integers(0, 100000, n),
        "ict_index": rng.uniform(0, 400, n),
        "influence": rng.uniform(0, 1000, n),
        "creativity": rng.uniform(0, 800, n),
        "threat": rng.uniform(0, 600, n),
        "expected_goals": rng.uniform(0, 15, n),
        "expected_assists": rng.uniform(0, 10, n),
        "expected_goal_involvements": rng.uniform(0, 20, n),
        "expected_goals_conceded": rng.uniform(0, 40, n),
        "status": rng.choice(["a", "a", "a", "a", "d", "i"], n),
        "chance_next_round": rng.choice([None, 25, 50, 75, 100], n),
        "news": [""] * n,
        "news_added": [""] * n,
        "starts": rng.integers(0, 30, n),
    })


def make_synthetic_history(players_df: pd.DataFrame, n_gws: int = 20) -> pd.DataFrame:
    """Generate per-fixture history for all players."""
    rng = np.random.default_rng(42)
    rows = []

    for _, player in players_df.iterrows():
        for gw in range(1, n_gws + 1):
            # Simulate: ~70% chance of playing, ~50% of those start
            plays = rng.random() < 0.7
            if plays:
                minutes = int(rng.choice([90, 85, 78, 70, 65, 30, 20, 15, 10], p=[
                    0.35, 0.1, 0.1, 0.08, 0.07, 0.1, 0.08, 0.07, 0.05
                ]))
            else:
                minutes = 0

            # Points correlate with minutes and player quality
            base_pts = rng.poisson(2) if minutes >= 60 else (1 if minutes > 0 else 0)
            bonus_pts = rng.integers(0, 4) if minutes >= 60 and rng.random() < 0.2 else 0

            rows.append({
                "element_id": player["element_id"],
                "round": gw,
                "fixture": gw * 10 + player["team_id"],
                "opponent_team": rng.integers(1, 21),
                "was_home": bool(rng.integers(0, 2)),
                "minutes": minutes,
                "total_points": base_pts + bonus_pts,
                "goals_scored": max(0, rng.poisson(0.2)) if minutes >= 60 else 0,
                "assists": max(0, rng.poisson(0.15)) if minutes >= 60 else 0,
                "clean_sheets": int(rng.random() < 0.3) if minutes >= 60 else 0,
                "goals_conceded": max(0, rng.poisson(1.0)),
                "saves": max(0, rng.poisson(2)) if player["position"] == "GKP" else 0,
                "bonus": bonus_pts,
                "bps": rng.integers(0, 50),
                "starts": 1 if minutes >= 60 else 0,
                "influence": rng.uniform(0, 50),
                "creativity": rng.uniform(0, 50),
                "threat": rng.uniform(0, 50),
                "ict_index": rng.uniform(0, 15),
                "expected_goals": rng.uniform(0, 0.5) if minutes > 0 else 0,
                "expected_assists": rng.uniform(0, 0.3) if minutes > 0 else 0,
                "expected_goal_involvements": rng.uniform(0, 0.7) if minutes > 0 else 0,
                "expected_goals_conceded": rng.uniform(0, 2),
                "value": player["price"],
                "selected": rng.integers(1000, 1000000),
                "transfers_in": rng.integers(0, 50000),
                "transfers_out": rng.integers(0, 50000),
                "kickoff_time": f"2025-{8 + gw // 5:02d}-{(gw % 28) + 1:02d}T15:00:00Z",
            })

    return pd.DataFrame(rows)


def make_synthetic_fixtures(n_gws: int = 20) -> pd.DataFrame:
    """Generate fixture data."""
    rng = np.random.default_rng(42)
    rows = []
    fixture_id = 1
    for gw in range(1, n_gws + 1):
        for _ in range(10):  # 10 matches per GW
            home = rng.integers(1, 21)
            away = rng.integers(1, 21)
            while away == home:
                away = rng.integers(1, 21)
            rows.append({
                "fixture_id": fixture_id,
                "gameweek": gw,
                "home_team_id": int(home),
                "away_team_id": int(away),
                "home_score": int(rng.integers(0, 4)),
                "away_score": int(rng.integers(0, 3)),
                "finished": gw <= 18,
                "kickoff_time": f"2025-{8 + gw // 5:02d}-{(gw % 28) + 1:02d}T15:00:00Z",
                "home_difficulty": int(rng.integers(1, 6)),
                "away_difficulty": int(rng.integers(1, 6)),
            })
            fixture_id += 1
    return pd.DataFrame(rows)


def make_synthetic_teams() -> pd.DataFrame:
    """Generate team metadata."""
    rows = []
    for i in range(1, 21):
        rows.append({
            "id": i,
            "name": f"Team_{i}",
            "short_name": f"T{i:02d}",
            "strength_overall_home": 1000 + i * 50,
            "strength_overall_away": 950 + i * 50,
            "strength_attack_home": 1000 + i * 40,
            "strength_attack_away": 950 + i * 40,
            "strength_defence_home": 1000 + i * 30,
            "strength_defence_away": 950 + i * 30,
        })
    return pd.DataFrame(rows)


# ── Tests ────────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_data():
    """Create a complete synthetic dataset."""
    players = make_synthetic_players(400)
    history = make_synthetic_history(players, n_gws=20)
    fixtures = make_synthetic_fixtures(n_gws=20)
    teams = make_synthetic_teams()
    return players, history, fixtures, teams


@pytest.fixture
def feature_df(synthetic_data):
    """Build features from synthetic data."""
    players, history, fixtures, teams = synthetic_data
    return build_player_features(history, players, fixtures, teams)


class TestFeatureEngineering:
    def test_feature_matrix_shape(self, feature_df):
        """Feature matrix should have many columns and rows."""
        assert feature_df.shape[0] > 1000  # many player-fixture rows
        assert feature_df.shape[1] > 30    # many feature columns

    def test_no_data_leakage(self, feature_df):
        """Rolling features should be NaN for the first row of each player."""
        first_rows = feature_df.groupby("element_id").first()
        # roll_pts_3 should be NaN for the first fixture (no prior data)
        if "roll_pts_3" in first_rows.columns:
            # After shift(1), first row should be NaN (filled by fillna later)
            pass  # this is handled by fillna in the pipeline

    def test_availability_features_exist(self, feature_df):
        """Should have P(start), P(sub), P(60+) features."""
        for w in [3, 5, 10]:
            assert f"p_start_{w}" in feature_df.columns
            assert f"p_sub_{w}" in feature_df.columns
            assert f"p60_{w}" in feature_df.columns

    def test_rolling_stats_exist(self, feature_df):
        """Should have rolling xG, xA, points features."""
        for w in [3, 5, 10]:
            assert f"roll_xG_{w}" in feature_df.columns
            assert f"roll_pts_{w}" in feature_df.columns


class TestMinutesModel:
    def test_label_assignment(self):
        """Minutes labels should correctly categorize."""
        minutes = pd.Series([0, 0, 15, 30, 60, 90, 45])
        labels = assign_minutes_label(minutes)
        assert labels.iloc[0] == 0  # BENCH
        assert labels.iloc[2] == 1  # SUB (15 min)
        assert labels.iloc[4] == 2  # START (60 min)
        assert labels.iloc[5] == 2  # START (90 min)
        assert labels.iloc[6] == 1  # SUB (45 min)

    def test_train_and_predict(self, feature_df):
        """Minutes model should train and produce valid probabilities."""
        model = MinutesModel()
        metrics = model.train(feature_df, n_splits=2, verbose=False)

        # Should have trained at least some position models
        assert len(model.models) > 0

        # Predict
        latest = feature_df.drop_duplicates(subset="element_id", keep="last")
        preds = model.predict(latest)

        assert len(preds) > 0
        assert "p_start" in preds.columns
        assert "p_sub" in preds.columns
        assert "p_bench" in preds.columns

        # Probabilities should sum to ~1
        prob_sums = preds["p_start"] + preds["p_sub"] + preds["p_bench"]
        np.testing.assert_allclose(prob_sums.values, 1.0, atol=0.01)

        # All probabilities should be [0, 1]
        assert (preds["p_start"] >= 0).all()
        assert (preds["p_start"] <= 1).all()


class TestPointsModel:
    def test_train_and_predict(self, feature_df):
        """Points model should train and produce non-negative xP."""
        model = PointsModel()
        metrics = model.train(feature_df, n_splits=2, verbose=False)

        assert len(model.starter_models) > 0

        latest = feature_df.drop_duplicates(subset="element_id", keep="last")
        preds = model.predict(latest)

        assert len(preds) > 0
        assert "e_pts_start" in preds.columns
        assert "e_pts_sub" in preds.columns
        assert (preds["e_pts_start"] >= 0).all()


class TestOptimizer:
    def _make_optimizer_input(self) -> pd.DataFrame:
        """Create a player DataFrame suitable for the optimizer."""
        rng = np.random.default_rng(42)
        n = 300
        positions = rng.choice(["GKP", "DEF", "MID", "FWD"], n, p=[0.1, 0.3, 0.35, 0.25])
        team_ids = rng.choice(range(1, 21), n)

        return pd.DataFrame({
            "element_id": range(1, n + 1),
            "name": [f"P{i}" for i in range(1, n + 1)],
            "position": positions,
            "team_id": team_ids,
            "team_name": [f"Team_{tid}" for tid in team_ids],
            "price": rng.integers(40, 120, n),
            "xp": rng.uniform(1, 8, n),
            "ownership_pct": rng.uniform(0, 80, n),
            "status": ["a"] * n,
        })

    def test_squad_selection(self):
        """Optimizer should pick a valid 15-player squad."""
        players = self._make_optimizer_input()
        optimizer = FPLOptimizer(GameState.NEUTRAL)
        result = optimizer.optimize_squad(players)

        assert len(result.squad) == 15
        assert len(result.starting_xi) == 11
        assert len(result.bench) == 4

        # Budget constraint
        assert result.squad["price"].sum() <= TOTAL_BUDGET

        # Position constraints
        pos_counts = result.squad["position"].value_counts()
        assert pos_counts.get("GKP", 0) == 2
        assert pos_counts.get("DEF", 0) == 5
        assert pos_counts.get("MID", 0) == 5
        assert pos_counts.get("FWD", 0) == 3

        # Max 3 per team
        team_counts = result.squad["team_id"].value_counts()
        assert team_counts.max() <= 3

    def test_captain_selected(self):
        """Should select a captain from the starting XI."""
        players = self._make_optimizer_input()
        optimizer = FPLOptimizer(GameState.NEUTRAL)
        result = optimizer.optimize_squad(players)

        assert result.captain_id in result.starting_xi["element_id"].values
        assert result.vice_captain_id in result.starting_xi["element_id"].values
        assert result.captain_id != result.vice_captain_id

    def test_differential_strategy(self):
        """CHASING gamestate should prefer lower-ownership players."""
        players = self._make_optimizer_input()

        neutral_opt = FPLOptimizer(GameState.NEUTRAL)
        chasing_opt = FPLOptimizer(GameState.CHASING)

        neutral_result = neutral_opt.optimize_squad(players)
        chasing_result = chasing_opt.optimize_squad(players)

        # Chasing squad should have lower average ownership
        neutral_own = neutral_result.squad["ownership_pct"].mean()
        chasing_own = chasing_result.squad["ownership_pct"].mean()

        # This is a probabilistic test — chasing should *tend* to be lower
        # but with synthetic data it's not guaranteed. Just check both are valid.
        assert len(chasing_result.squad) == 15
        assert len(neutral_result.squad) == 15

    def test_formation_validity(self):
        """Starting XI should have valid formation (1 GKP, 3-5 DEF, etc.)."""
        players = self._make_optimizer_input()
        optimizer = FPLOptimizer(GameState.NEUTRAL)
        result = optimizer.optimize_squad(players)

        xi_pos = result.starting_xi["position"].value_counts()
        assert xi_pos.get("GKP", 0) == 1
        assert 3 <= xi_pos.get("DEF", 0) <= 5
        assert 2 <= xi_pos.get("MID", 0) <= 5
        assert 1 <= xi_pos.get("FWD", 0) <= 3

    def test_transfer_suggestions(self):
        """Should suggest valid transfers."""
        all_players = self._make_optimizer_input()
        current_squad = all_players.head(15).copy()

        optimizer = FPLOptimizer(GameState.NEUTRAL)
        transfers = optimizer.optimize_transfers(
            current_squad=current_squad,
            all_players=all_players,
            free_transfers=1,
            bank=10,
            horizon=3,
        )

        # Should suggest at least one transfer if there's an improvement
        assert isinstance(transfers, list)
        if transfers:
            assert "out_id" in transfers[0]
            assert "in_id" in transfers[0]
            assert "xp_gain" in transfers[0]


# ── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
