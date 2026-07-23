"""Upcoming-fixture prediction frames — fixes the train/serve skew.

The bug this module exists to fix
─────────────────────────────────
`FPLEngine._get_latest_features()` built its prediction rows by taking each
player's **most recent played** row out of the feature matrix. That row's
`was_home`, `opp_strength_*`, `opp_goals_conceded_roll5`, `days_since_last` and
`congestion_score` all describe the fixture that has already been played.

Training, meanwhile, pairs each row's features with that same row's outcome — so
at training time those columns correctly describe the fixture being predicted.
The result is a textbook train/serve skew: the model learns "hard opponent →
fewer points", then at inference is handed *last week's* opponent. A player who
just visited the league leaders is penalised for a home tie against the bottom
club, and vice versa. Fixture difficulty is one of the largest legitimate edges
in FPL, and the previous code fed it in backwards.

There is a second, quieter version of the same problem. Rolling features are
built with `.shift(1)`, so the last played row's `roll_pts_5` deliberately
excludes that match. Predicting from it therefore throws away the most recent
gameweek of form.

How this module fixes both
──────────────────────────
1. Append one synthetic future row per player to the history, then run the
   **existing** feature pipeline over the combined frame. Because the pipeline
   shifts by one, the synthetic row's rolling window covers every real match up
   to and including the latest one. No feature logic is duplicated, so the
   carried-forward columns cannot drift from the training definitions.

2. Strip the fixture-dependent columns off that carry frame and recompute them
   per *upcoming* fixture, from the real fixture list.

Doubles and blanks fall out of the design for free: the frame is keyed on
(player, upcoming fixture), so a double gameweek yields two rows for that
player and a blank yields none.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import build_player_features

# Columns whose value depends on *which* fixture is being played. They must be
# recomputed for the upcoming fixture rather than carried forward.
FIXTURE_DEPENDENT_COLS = [
    "was_home",
    "opp_strength_overall",
    "opp_strength_attack",
    "opp_strength_defence",
    "opp_goals_conceded_roll5",
    "days_since_last",
    "congestion_score",
    "opponent_team",
    "fixture",
    "round",
    "kickoff_time",
]

# FPL availability statuses that mean "will not play". 'd' (doubtful) is
# deliberately excluded — doubtful players do often play, and the minutes model
# is the right place to price that risk.
UNAVAILABLE_STATUSES = {"i", "s", "u", "n"}


def build_upcoming_frame(
    history_df: pd.DataFrame,
    players_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    target_gws: list[int],
) -> pd.DataFrame:
    """Build a prediction frame keyed on (player, upcoming fixture).

    Args:
        history_df: Per-fixture player history (from `get_all_player_histories`).
        players_df: Player master data.
        fixtures_df: Full fixture list, including unplayed fixtures.
        teams_df: Team metadata with strength ratings.
        target_gws: Gameweeks to build prediction rows for, e.g. `[15, 16, 17]`.

    Returns:
        One row per (player, upcoming fixture) carrying every feature column the
        models were trained on, with fixture-dependent columns describing the
        upcoming fixture. Includes `target_gw`, `fixture_id`, `is_home` and
        `n_fixtures_in_gw` for downstream aggregation.

        Empty if no fixtures are scheduled for `target_gws`.
    """
    if not target_gws:
        return pd.DataFrame()

    carry = _build_carry_frame(history_df, players_df, fixtures_df, teams_df)
    if carry.empty:
        return pd.DataFrame()

    upcoming_fixtures = _upcoming_fixtures(fixtures_df, target_gws)
    if upcoming_fixtures.empty:
        return pd.DataFrame()

    conceded = _team_latest_goals_conceded(fixtures_df)
    strength = _team_strength_lookup(teams_df)

    rows = carry.merge(upcoming_fixtures, on="team_id", how="inner")
    if rows.empty:
        return pd.DataFrame()

    # ── Recompute fixture-dependent features for the upcoming fixture ────
    rows["was_home"] = rows["is_home"].astype(int)
    rows["opponent_team"] = rows["opponent_team_id"]

    for key in ("strength_overall", "strength_attack", "strength_defence"):
        rows[f"opp_{key}"] = rows["opponent_team_id"].map(
            lambda tid: strength.get(tid, {}).get(key, 0.0)
        )

    rows["opp_goals_conceded_roll5"] = (
        rows["opponent_team_id"].map(conceded).astype(float)
    )

    # Rest days: upcoming kickoff minus the player's most recent kickoff.
    rows["days_since_last"] = (
        rows["kickoff_time"] - rows["last_kickoff"]
    ).dt.total_seconds() / 86400.0
    # No parseable last kickoff (new signings, season openers) → assume rested.
    rows["days_since_last"] = rows["days_since_last"].fillna(7.0).clip(lower=0.0)
    rows["congestion_score"] = (
        7.0 / rows["days_since_last"].clip(lower=1.0)
    ).clip(upper=3.0)

    # Fixture load within the target gameweek: 2 in a double, 1 normally.
    rows["n_fixtures_in_gw"] = rows.groupby(
        ["element_id", "target_gw"]
    )["fixture_id"].transform("size")

    rows["round"] = rows["target_gw"]
    rows["is_future"] = True

    rows = _apply_availability_overrides(rows)

    return rows.sort_values(["target_gw", "element_id"]).reset_index(drop=True)


# ── Internals ────────────────────────────────────────────────────────────────


def _build_carry_frame(
    history_df: pd.DataFrame,
    players_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    teams_df: pd.DataFrame,
) -> pd.DataFrame:
    """One row per player holding form/availability features valid *now*.

    Appends a synthetic future row per player and runs the real feature
    pipeline, so the shifted rolling windows resolve to "every match played so
    far" — including the most recent one, which `_get_latest_features()` used
    to discard.
    """
    if history_df.empty:
        return pd.DataFrame()

    hist = history_df.copy()
    if "round" in hist.columns:
        hist["round"] = pd.to_numeric(hist["round"], errors="coerce")
    if "kickoff_time" in hist.columns:
        hist["kickoff_time"] = pd.to_datetime(
            hist["kickoff_time"], errors="coerce", utc=True
        )

    # Placeholder row: real identity, no outcomes. Sorted last so the shifted
    # rolling windows see the entire real history.
    placeholder = pd.DataFrame({"element_id": hist["element_id"].unique()})
    placeholder["round"] = (hist["round"].max() or 0) + 1
    placeholder["is_future"] = True
    # `_prepare_history` casts this to int unconditionally, so it cannot be
    # null. The value is meaningless — every fixture-dependent column is
    # recomputed from the real upcoming fixture downstream.
    placeholder["was_home"] = 0

    # Every numeric outcome column must exist but be empty, so the placeholder
    # contributes nothing to any aggregate it is not supposed to influence.
    for col in hist.columns:
        if col in placeholder.columns:
            continue
        placeholder[col] = np.nan

    hist["is_future"] = False
    combined = pd.concat([hist, placeholder], ignore_index=True)
    combined = combined.sort_values(["element_id", "round"]).reset_index(drop=True)

    features = build_player_features(
        history_df=combined,
        players_df=players_df,
        fixtures_df=fixtures_df,
        teams_df=teams_df,
    )

    carry = features[features["is_future"] == True].copy()  # noqa: E712
    carry = carry.drop_duplicates(subset="element_id", keep="last")

    # Fixture-dependent columns are recomputed downstream, so drop them here to
    # guarantee no placeholder value leaks into a prediction.
    carry = carry.drop(columns=[c for c in FIXTURE_DEPENDENT_COLS if c in carry.columns])

    # Most recent real kickoff per player, for the rest-days calculation.
    played = hist[hist["minutes"].notna()] if "minutes" in hist.columns else hist
    last_kickoff = (
        played.groupby("element_id")["kickoff_time"].max().rename("last_kickoff")
        if "kickoff_time" in played.columns
        else pd.Series(dtype="datetime64[ns, UTC]", name="last_kickoff")
    )
    carry = carry.merge(last_kickoff, on="element_id", how="left")

    # Refresh identity/price/status from the live players table — history rows
    # carry the price at the time each match was played.
    live_cols = [
        c
        for c in ("element_id", "team_id", "position", "name", "full_name",
                  "price", "status", "chance_next_round", "selected_pct", "form")
        if c in players_df.columns
    ]
    carry = carry.drop(
        columns=[c for c in live_cols if c != "element_id" and c in carry.columns]
    )
    carry = carry.merge(players_df[live_cols], on="element_id", how="inner")

    return carry


def _upcoming_fixtures(fixtures_df: pd.DataFrame, target_gws: list[int]) -> pd.DataFrame:
    """Explode the fixture list into one row per (team, fixture) for target GWs."""
    if fixtures_df.empty:
        return pd.DataFrame()

    fx = fixtures_df.copy()
    fx["gameweek"] = pd.to_numeric(fx["gameweek"], errors="coerce")
    fx = fx[fx["gameweek"].isin(target_gws)]
    # Only fixtures that have not been played.
    if "finished" in fx.columns:
        fx = fx[~fx["finished"].fillna(False).astype(bool)]
    if fx.empty:
        return pd.DataFrame()

    fx["kickoff_time"] = pd.to_datetime(fx["kickoff_time"], errors="coerce", utc=True)

    home = pd.DataFrame({
        "team_id": fx["home_team_id"],
        "opponent_team_id": fx["away_team_id"],
        "is_home": True,
        "target_gw": fx["gameweek"].astype(int),
        "fixture_id": fx["fixture_id"],
        "kickoff_time": fx["kickoff_time"],
        "fdr": fx.get("home_difficulty"),
    })
    away = pd.DataFrame({
        "team_id": fx["away_team_id"],
        "opponent_team_id": fx["home_team_id"],
        "is_home": False,
        "target_gw": fx["gameweek"].astype(int),
        "fixture_id": fx["fixture_id"],
        "kickoff_time": fx["kickoff_time"],
        "fdr": fx.get("away_difficulty"),
    })

    return pd.concat([home, away], ignore_index=True)


def _team_strength_lookup(teams_df: pd.DataFrame) -> dict[int, dict[str, float]]:
    """Team strength ratings, keyed by team id.

    Matches `features._add_opponent_features` exactly — home and away ratings
    summed, venue handled separately by `was_home`. Any change here must be
    mirrored there or the skew this module fixes reappears.
    """
    lookup: dict[int, dict[str, float]] = {}
    for _, t in teams_df.iterrows():
        lookup[t["id"]] = {
            "strength_overall": t.get("strength_overall_home", 0) + t.get("strength_overall_away", 0),
            "strength_attack": t.get("strength_attack_home", 0) + t.get("strength_attack_away", 0),
            "strength_defence": t.get("strength_defence_home", 0) + t.get("strength_defence_away", 0),
        }
    return lookup


def _team_latest_goals_conceded(fixtures_df: pd.DataFrame) -> dict[int, float]:
    """Most recent rolling goals-conceded-per-game for each team.

    The training-time version joins on (opponent, round); an unplayed round has
    no entry, so a naive join would silently fill 0 and tell the model every
    upcoming opponent is a watertight defence. Carrying the latest value
    forward is the correct extrapolation.
    """
    if fixtures_df.empty or "finished" not in fixtures_df.columns:
        return {}

    finished = fixtures_df[fixtures_df["finished"].fillna(False).astype(bool)].copy()
    if finished.empty:
        return {}

    records = []
    for _, f in finished.iterrows():
        gw = f.get("gameweek")
        if gw is None or pd.isna(gw):
            continue
        records.append({"team_id": f["home_team_id"], "gameweek": gw,
                        "goals_conceded": f["away_score"] or 0})
        records.append({"team_id": f["away_team_id"], "gameweek": gw,
                        "goals_conceded": f["home_score"] or 0})

    if not records:
        return {}

    tdf = pd.DataFrame(records).sort_values(["team_id", "gameweek"])
    rolled = (
        tdf.groupby("team_id")["goals_conceded"]
        .apply(lambda x: x.rolling(5, min_periods=1).mean().iloc[-1])
    )
    return rolled.to_dict()


def _apply_availability_overrides(rows: pd.DataFrame) -> pd.DataFrame:
    """Encode hard availability facts the model should not have to infer.

    `prepare_model_input` fills a null `chance_next_round` with 100 — which is
    right for a fit player with no news and badly wrong for a player flagged
    injured or suspended, where FPL commonly leaves the field null. Resolve it
    here from `status`, which is never null.
    """
    rows = rows.copy()

    if "status" not in rows.columns:
        rows["unavailable"] = False
        return rows

    status = rows["status"].fillna("a").astype(str)
    rows["unavailable"] = status.isin(UNAVAILABLE_STATUSES)

    if "chance_next_round" in rows.columns:
        chance = pd.to_numeric(rows["chance_next_round"], errors="coerce")
        # Null + flagged → 0. Null + available → 100. Explicit values respected.
        chance = chance.where(~(chance.isna() & rows["unavailable"]), 0.0)
        chance = chance.where(~(chance.isna() & ~rows["unavailable"]), 100.0)
        rows["chance_next_round"] = chance

    return rows
