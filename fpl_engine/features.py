"""Feature engineering — rolling stats, availability, opponent strength, context.

Produces the unified feature matrix consumed by both the minutes model and
the points model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import POSITION_MAP, ROLLING_WINDOWS

# ── Position-specific priors for cold-start players ──────────────────────────
POSITION_PRIORS = {
    "GKP": {"p_start": 0.50, "p_sub": 0.02, "p60": 0.48, "e_minutes": 45.0},
    "DEF": {"p_start": 0.45, "p_sub": 0.10, "p60": 0.42, "e_minutes": 42.0},
    "MID": {"p_start": 0.42, "p_sub": 0.12, "p60": 0.38, "e_minutes": 38.0},
    "FWD": {"p_start": 0.40, "p_sub": 0.12, "p60": 0.35, "e_minutes": 35.0},
}


def build_player_features(
    history_df: pd.DataFrame,
    players_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    teams_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build the full feature matrix from raw FPL data.

    Args:
        history_df: Per-fixture player stats (from get_all_player_histories).
        players_df: Player master data (from get_players_df).
        fixtures_df: Fixture list (from get_fixtures_df).
        teams_df: Team metadata (from get_teams_df).

    Returns:
        DataFrame with one row per (player, fixture) and all engineered features.
    """
    df = _prepare_history(history_df, players_df, fixtures_df, teams_df)
    df = _add_rolling_player_stats(df)
    df = _add_availability_features(df)
    df = _add_opponent_features(df, teams_df, fixtures_df)
    df = _add_fixture_congestion(df, fixtures_df)
    df = _add_market_features(df, players_df)
    df = _add_form_consistency(df)
    return df


# ── Internals ────────────────────────────────────────────────────────────────

def _prepare_history(
    history_df: pd.DataFrame,
    players_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    teams_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge raw history with player/team metadata."""
    df = history_df.copy()

    # Ensure numeric types
    numeric_cols = [
        "minutes", "total_points", "goals_scored", "assists",
        "clean_sheets", "goals_conceded", "saves", "bonus", "bps",
        "influence", "creativity", "threat", "ict_index",
        "expected_goals", "expected_assists", "expected_goal_involvements",
        "expected_goals_conceded", "value", "selected", "transfers_in",
        "transfers_out", "starts",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Merge player metadata
    player_cols = ["element_id", "name", "position", "team_id", "team_name",
                   "team_short", "fpl_code", "status", "chance_next_round",
                   "news", "price"]
    available_cols = [c for c in player_cols if c in players_df.columns]
    df = df.merge(
        players_df[available_cols],
        on="element_id",
        how="left",
        suffixes=("", "_current"),
    )

    # Parse kickoff time
    if "kickoff_time" in df.columns:
        df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], errors="coerce")

    # Sort for rolling calculations
    df = df.sort_values(["element_id", "round"]).reset_index(drop=True)

    # Derived columns
    df["was_home"] = df["was_home"].astype(int) if "was_home" in df.columns else 0
    df["started"] = (df["starts"] > 0).astype(int) if "starts" in df.columns else (df["minutes"] >= 60).astype(int)
    df["played_any"] = (df["minutes"] > 0).astype(int)
    df["played_60plus"] = (df["minutes"] >= 60).astype(int)

    return df


def _add_rolling_player_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Add shifted rolling averages for key stats."""
    df = df.copy()

    # Stats to compute rolling averages for
    roll_stats = {
        "total_points": "pts",
        "minutes": "mins",
        "goals_scored": "goals",
        "assists": "assists",
        "clean_sheets": "cs",
        "bonus": "bonus",
        "bps": "bps",
        "influence": "infl",
        "creativity": "crea",
        "threat": "threat",
        "ict_index": "ict",
        "expected_goals": "xG",
        "expected_assists": "xA",
        "expected_goal_involvements": "xGI",
        "expected_goals_conceded": "xGC",
    }

    for raw_col, short in roll_stats.items():
        if raw_col not in df.columns:
            continue
        for w in ROLLING_WINDOWS:
            col_name = f"roll_{short}_{w}"
            df[col_name] = (
                df.groupby("element_id")[raw_col]
                .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            )

    # Per-90 stats (using rolling minutes to avoid div-by-zero)
    for w in ROLLING_WINDOWS:
        mins_col = f"roll_mins_{w}"
        if mins_col in df.columns:
            safe_mins = df[mins_col].clip(lower=1)
            for stat in ["xG", "xA", "xGI", "goals", "assists", "bps"]:
                roll_col = f"roll_{stat}_{w}"
                if roll_col in df.columns:
                    df[f"per90_{stat}_{w}"] = df[roll_col] / safe_mins * 90

    return df


def _add_availability_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add P(start), P(sub), P(60+), E[minutes] as rolling features."""
    df = df.copy()

    for w in ROLLING_WINDOWS:
        # P(played any minutes)
        df[f"p_any_{w}"] = (
            df.groupby("element_id")["played_any"]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        )
        # P(started)
        df[f"p_start_{w}"] = (
            df.groupby("element_id")["started"]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        )
        # P(60+ minutes)
        df[f"p60_{w}"] = (
            df.groupby("element_id")["played_60plus"]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        )
        # E[minutes]
        df[f"e_mins_{w}"] = (
            df.groupby("element_id")["minutes"]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        )
        # Sub rate: played but didn't start
        df[f"p_sub_{w}"] = df[f"p_any_{w}"] - df[f"p_start_{w}"]

    # Fill NaN with position-specific priors
    for pos, priors in POSITION_PRIORS.items():
        mask = df["position"] == pos
        for w in ROLLING_WINDOWS:
            df.loc[mask, f"p_start_{w}"] = df.loc[mask, f"p_start_{w}"].fillna(priors["p_start"])
            df.loc[mask, f"p_sub_{w}"] = df.loc[mask, f"p_sub_{w}"].fillna(priors["p_sub"])
            df.loc[mask, f"p_any_{w}"] = df.loc[mask, f"p_any_{w}"].fillna(priors["p_start"] + priors["p_sub"])
            df.loc[mask, f"p60_{w}"] = df.loc[mask, f"p60_{w}"].fillna(priors["p60"])
            df.loc[mask, f"e_mins_{w}"] = df.loc[mask, f"e_mins_{w}"].fillna(priors["e_minutes"])

    return df


def _add_opponent_features(
    df: pd.DataFrame,
    teams_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add opponent strength and defensive quality features."""
    df = df.copy()

    # Build team strength lookup from teams_df
    team_strength = {}
    for _, t in teams_df.iterrows():
        team_strength[t["id"]] = {
            "strength_overall": t.get("strength_overall_home", 0) + t.get("strength_overall_away", 0),
            "strength_attack": t.get("strength_attack_home", 0) + t.get("strength_attack_away", 0),
            "strength_defence": t.get("strength_defence_home", 0) + t.get("strength_defence_away", 0),
        }

    # Map opponent team for each fixture in history
    if "opponent_team" in df.columns:
        for key in ["strength_overall", "strength_attack", "strength_defence"]:
            df[f"opp_{key}"] = df["opponent_team"].map(
                lambda tid: team_strength.get(tid, {}).get(key, 0)
            )

    # Opponent goals conceded per game (rolling, from fixture data)
    if not fixtures_df.empty:
        team_goals_conceded = _compute_team_rolling_goals_conceded(fixtures_df)
        if "opponent_team" in df.columns and "round" in df.columns:
            df = df.merge(
                team_goals_conceded.rename(columns={
                    "team_id": "opponent_team",
                    "gameweek": "round",
                }),
                on=["opponent_team", "round"],
                how="left",
            )

    return df


def _compute_team_rolling_goals_conceded(fixtures_df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling goals-conceded-per-game for each team."""
    finished = fixtures_df[fixtures_df["finished"] == True].copy()
    if finished.empty:
        return pd.DataFrame(columns=["team_id", "gameweek", "opp_goals_conceded_roll5"])

    # Build team-level records
    records = []
    for _, f in finished.iterrows():
        gw = f["gameweek"]
        if gw is None:
            continue
        records.append({"team_id": f["home_team_id"], "gameweek": gw,
                        "goals_conceded": f["away_score"] or 0})
        records.append({"team_id": f["away_team_id"], "gameweek": gw,
                        "goals_conceded": f["home_score"] or 0})

    tdf = pd.DataFrame(records).sort_values(["team_id", "gameweek"])
    tdf["opp_goals_conceded_roll5"] = (
        tdf.groupby("team_id")["goals_conceded"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    return tdf[["team_id", "gameweek", "opp_goals_conceded_roll5"]]


def _add_fixture_congestion(df: pd.DataFrame, fixtures_df: pd.DataFrame) -> pd.DataFrame:
    """Add days-since-last-match and matches-in-next-7-days features."""
    df = df.copy()

    if "kickoff_time" not in df.columns or df["kickoff_time"].isna().all():
        df["days_since_last"] = np.nan
        df["matches_in_7_days"] = 1
        return df

    # Days since last match for each player
    df["days_since_last"] = (
        df.groupby("element_id")["kickoff_time"]
        .transform(lambda x: x.diff().dt.total_seconds() / 86400)
    )
    df["days_since_last"] = df["days_since_last"].fillna(7.0)  # assume rested if first match

    # Fixture congestion: count team matches in surrounding window
    if not fixtures_df.empty and "kickoff_time" in fixtures_df.columns:
        fixtures_df = fixtures_df.copy()
        fixtures_df["kickoff_time"] = pd.to_datetime(fixtures_df["kickoff_time"], errors="coerce")
        # Simplified: just use days_since_last as main congestion signal
        df["congestion_score"] = (7.0 / df["days_since_last"].clip(lower=1)).clip(upper=3)
    else:
        df["congestion_score"] = 1.0

    return df


def _add_market_features(df: pd.DataFrame, players_df: pd.DataFrame) -> pd.DataFrame:
    """Add ownership, transfer momentum, and value features."""
    df = df.copy()

    # Transfer momentum
    if "transfers_in" in df.columns and "transfers_out" in df.columns:
        df["transfer_balance"] = df["transfers_in"] - df["transfers_out"]
        for w in ROLLING_WINDOWS:
            df[f"transfer_momentum_{w}"] = (
                df.groupby("element_id")["transfer_balance"]
                .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            )

    # Points per million (value efficiency)
    if "value" in df.columns:
        safe_value = df["value"].clip(lower=1)
        for w in ROLLING_WINDOWS:
            pts_col = f"roll_pts_{w}"
            if pts_col in df.columns:
                df[f"pts_per_million_{w}"] = df[pts_col] / safe_value * 10

    # Current ownership and selection %
    if "selected" in df.columns:
        df["ownership_pct"] = pd.to_numeric(df["selected"], errors="coerce").fillna(0)

    return df


def _add_form_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Add the Efficiency metric: mean / CV — penalizes volatility."""
    df = df.copy()

    for w in [5, 10]:
        mean_col = f"roll_pts_{w}"
        if mean_col not in df.columns:
            continue

        # Rolling standard deviation (shifted)
        df[f"_pts_std_{w}"] = (
            df.groupby("element_id")["total_points"]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=2).std())
        )
        # Coefficient of variation
        safe_mean = df[mean_col].clip(lower=0.1)
        df[f"pts_cv_{w}"] = df[f"_pts_std_{w}"] / safe_mean

        # Efficiency = mean / CV (higher = more consistent value)
        safe_cv = df[f"pts_cv_{w}"].clip(lower=0.01)
        df[f"efficiency_{w}"] = df[mean_col] / safe_cv

        df.drop(columns=[f"_pts_std_{w}"], inplace=True, errors="ignore")

    return df


# ── Feature column lists (used by models) ────────────────────────────────────

def get_minutes_feature_cols() -> list[str]:
    """Feature columns for the minutes/availability model."""
    cols = []
    for w in ROLLING_WINDOWS:
        cols.extend([
            f"p_start_{w}", f"p_sub_{w}", f"p_any_{w}", f"p60_{w}", f"e_mins_{w}",
        ])
    cols.extend([
        "was_home",
        "days_since_last", "congestion_score",
        "opp_strength_overall", "opp_strength_attack", "opp_strength_defence",
    ])
    # Status encoding
    cols.extend(["status_available", "chance_next_round_num"])
    return cols


def get_points_feature_cols() -> list[str]:
    """Feature columns for the expected-points model."""
    cols = []

    # Rolling performance stats
    for stat in ["pts", "xG", "xA", "xGI", "xGC", "goals", "assists",
                 "cs", "bonus", "bps", "infl", "crea", "threat", "ict"]:
        for w in ROLLING_WINDOWS:
            cols.append(f"roll_{stat}_{w}")

    # Per-90 rates
    for stat in ["xG", "xA", "xGI", "goals", "assists", "bps"]:
        for w in ROLLING_WINDOWS:
            cols.append(f"per90_{stat}_{w}")

    # Availability (key: minutes model feeds into points model)
    for w in ROLLING_WINDOWS:
        cols.extend([f"p_start_{w}", f"p60_{w}", f"e_mins_{w}"])

    # Opponent
    cols.extend([
        "was_home",
        "opp_strength_overall", "opp_strength_attack", "opp_strength_defence",
    ])

    # Market and form
    for w in ROLLING_WINDOWS:
        cols.append(f"pts_per_million_{w}")
    for w in [5, 10]:
        cols.extend([f"pts_cv_{w}", f"efficiency_{w}"])

    # Congestion
    cols.extend(["days_since_last", "congestion_score"])

    return cols


def prepare_model_input(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Select feature columns and fill any remaining NaN."""
    available = [c for c in feature_cols if c in df.columns]
    result = df[available].copy()

    # Add status encoding if needed
    if "status_available" in feature_cols and "status_available" not in result.columns:
        if "status" in df.columns:
            result["status_available"] = (df["status"] == "a").astype(int)
        else:
            result["status_available"] = 1

    if "chance_next_round_num" in feature_cols and "chance_next_round_num" not in result.columns:
        if "chance_next_round" in df.columns:
            result["chance_next_round_num"] = pd.to_numeric(
                df["chance_next_round"], errors="coerce"
            ).fillna(100) / 100.0
        else:
            result["chance_next_round_num"] = 1.0

    # Fill remaining NaN with 0
    result = result.fillna(0.0)
    return result
