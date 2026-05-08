"""Rival tracking and counter-optimization for mini-leagues.

Fetches rival squads from the FPL API and adjusts the optimizer objective
to maximize rank gain against specific opponents rather than raw points.

Key insight: in a mini-league, your edge comes from differential picks —
players rivals don't own who then score big. This module quantifies that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .client import FPLClient
from .optimizer import GameState


@dataclass
class RivalSquad:
    """A rival manager's squad for a given gameweek."""
    manager_id: int
    manager_name: str
    gameweek: int
    picks: pd.DataFrame
    total_points: int = 0
    rank: int = 0


@dataclass
class DifferentialOpportunity:
    element_id: int
    name: str
    position: str
    xp: float
    rival_ownership_pct: float
    differential_value: float
    captain_differential: bool = False


@dataclass
class RiskPlayer:
    element_id: int
    name: str
    position: str
    xp: float
    rival_ownership_pct: float
    points_at_risk: float


@dataclass
class RivalTracker:
    """Fetches and analyzes mini-league rival squads."""

    client: FPLClient
    league_id: int
    _standings_cache: pd.DataFrame = field(default_factory=pd.DataFrame, init=False)
    _squad_cache: dict[str, pd.DataFrame] = field(default_factory=dict, init=False)

    # ── League data ──────────────────────────────────────────────────────

    def fetch_league_standings(self) -> pd.DataFrame:
        """Fetch classic league standings.

        Returns:
            DataFrame: manager_id, name, team_name, total_points, rank,
                       gw_points, transfers_made
        """
        data = self.client._get(
            f"/leagues-classic/{self.league_id}/standings/",
            cache_key=f"league_{self.league_id}_standings",
        )
        results = data.get("standings", {}).get("results", [])
        rows = []
        for r in results:
            rows.append({
                "manager_id": r["entry"],
                "name": r["player_name"],
                "team_name": r["entry_name"],
                "total_points": r["total"],
                "rank": r["rank"],
                "gw_points": r.get("event_total", 0),
                "transfers_made": r.get("transfers_made", 0),
            })
        df = pd.DataFrame(rows).sort_values("rank").reset_index(drop=True)
        self._standings_cache = df
        return df

    def fetch_rival_squad(self, manager_id: int, gameweek: int) -> pd.DataFrame:
        """Fetch a rival's squad for a specific gameweek.

        Returns:
            DataFrame: element_id, position_order, is_captain,
                       is_vice_captain, multiplier
        """
        cache_key = f"rival_{manager_id}_gw{gameweek}"
        if cache_key in self._squad_cache:
            return self._squad_cache[cache_key]

        try:
            data = self.client.manager_picks(manager_id, gameweek)
        except Exception:
            return pd.DataFrame()

        picks = data.get("picks", [])
        rows = []
        for p in picks:
            rows.append({
                "element_id": p["element"],
                "position_order": p["position"],
                "is_captain": p["is_captain"],
                "is_vice_captain": p["is_vice_captain"],
                "multiplier": p["multiplier"],
            })
        df = pd.DataFrame(rows)
        self._squad_cache[cache_key] = df
        return df

    def fetch_top_rivals(
        self,
        gameweek: int,
        n: int = 5,
        exclude_manager_id: int | None = None,
    ) -> list[dict]:
        """Fetch squads for the top N rivals in the league.

        Args:
            gameweek: Current gameweek.
            n: Number of rivals to fetch.
            exclude_manager_id: Your own manager ID (skip it).

        Returns:
            List of dicts: [{manager_id, name, total_points, squad_df}, ...]
        """
        standings = (
            self._standings_cache
            if not self._standings_cache.empty
            else self.fetch_league_standings()
        )

        rivals = []
        for _, row in standings.iterrows():
            mid = int(row["manager_id"])
            if mid == exclude_manager_id:
                continue
            squad_df = self.fetch_rival_squad(mid, gameweek)
            rivals.append({
                "manager_id": mid,
                "name": row["name"],
                "team_name": row["team_name"],
                "total_points": row["total_points"],
                "rank": row["rank"],
                "squad_df": squad_df,
            })
            if len(rivals) >= n:
                break

        return rivals

    # ── Analysis ─────────────────────────────────────────────────────────

    def compute_rival_template(
        self,
        rival_squads: list[dict],
    ) -> pd.DataFrame:
        """What % of tracked rivals own each player?

        Args:
            rival_squads: Output of fetch_top_rivals().

        Returns:
            DataFrame: element_id, rival_ownership_pct, rival_captain_pct
        """
        n_rivals = len(rival_squads)
        if n_rivals == 0:
            return pd.DataFrame(columns=["element_id", "rival_ownership_pct", "rival_captain_pct"])

        ownership: dict[int, int] = {}
        captaincy: dict[int, int] = {}

        for rival in rival_squads:
            # Accept both RivalSquad dataclass and legacy dict
            if isinstance(rival, RivalSquad):
                squad_df = rival.picks
            elif isinstance(rival, dict):
                squad_df = rival.get("squad_df", pd.DataFrame())
            else:
                squad_df = pd.DataFrame()
            if squad_df is None or (hasattr(squad_df, "empty") and squad_df.empty):
                continue
            for _, row in squad_df.iterrows():
                pid = int(row["element_id"])
                ownership[pid] = ownership.get(pid, 0) + 1
                if row.get("is_captain", False):
                    captaincy[pid] = captaincy.get(pid, 0) + 1

        rows = []
        for pid, count in ownership.items():
            rows.append({
                "element_id": pid,
                "rival_ownership_pct": round(count / n_rivals * 100, 1),
                "rival_captain_pct": round(captaincy.get(pid, 0) / n_rivals * 100, 1),
            })

        return pd.DataFrame(rows).sort_values("rival_ownership_pct", ascending=False)

    def compute_differential_opportunities(
        self,
        my_squad_ids: list[int],
        rival_template: pd.DataFrame,
        predictions_df: pd.DataFrame,
        min_xp: float = 4.0,
        max_rival_ownership: float = 40.0,
    ) -> pd.DataFrame:
        """High-xP players that rivals don't own (differentials).

        Args:
            my_squad_ids: Your current squad element_ids.
            rival_template: Output of compute_rival_template().
            predictions_df: Must have element_id, xp, name, position, price.
            min_xp: Minimum xP to consider.
            max_rival_ownership: Max rival ownership % to count as differential.

        Returns:
            DataFrame of differential opportunities sorted by xP desc.
        """
        df = predictions_df.merge(rival_template, on="element_id", how="left")
        df["rival_ownership_pct"] = df["rival_ownership_pct"].fillna(0.0)
        df["in_my_squad"] = df["element_id"].isin(my_squad_ids)

        differentials = df[
            (~df["in_my_squad"])
            & (df["xp"] >= min_xp)
            & (df["rival_ownership_pct"] <= max_rival_ownership)
        ].copy()

        # Differential score: xP weighted by (1 - rival_ownership)
        differentials["differential_score"] = (
            differentials["xp"] * (1 - differentials["rival_ownership_pct"] / 100)
        )
        return differentials.sort_values("differential_score", ascending=False).head(20)

    def compute_risk_players(
        self,
        my_squad_ids: list[int],
        rival_template: pd.DataFrame,
        predictions_df: pd.DataFrame,
        min_rival_ownership: float = 60.0,
        min_xp: float = 3.0,
    ) -> pd.DataFrame:
        """High-xP players most rivals own but I DON'T.

        Not owning these means falling behind if they score big.

        Returns:
            DataFrame of risk players sorted by rival_ownership_pct desc.
        """
        df = predictions_df.merge(rival_template, on="element_id", how="left")
        df["rival_ownership_pct"] = df["rival_ownership_pct"].fillna(0.0)
        df["in_my_squad"] = df["element_id"].isin(my_squad_ids)

        risks = df[
            (~df["in_my_squad"])
            & (df["rival_ownership_pct"] >= min_rival_ownership)
            & (df["xp"] >= min_xp)
        ].copy()

        risks["risk_score"] = risks["rival_ownership_pct"] * risks["xp"] / 100
        return risks.sort_values("risk_score", ascending=False).head(10)

    # ── Counter-optimization ─────────────────────────────────────────────

    def get_rival_adjusted_objective(
        self,
        predictions_df: pd.DataFrame,
        rival_template: pd.DataFrame,
        gamestate: GameState,
    ) -> pd.DataFrame:
        """Compute rival-adjusted xP for the optimizer.

        For MINI_LEAGUE: strongly boost differentials, penalize template picks.
        For CHASING: moderate differential bonus.
        For LEADING: mild template bonus (protect lead).

        Returns:
            predictions_df with added 'adj_xp_rival' column.
        """
        df = predictions_df.merge(rival_template, on="element_id", how="left")
        df["rival_ownership_pct"] = df["rival_ownership_pct"].fillna(5.0)
        df["rival_eo"] = df["rival_ownership_pct"] / 100.0

        if gamestate == GameState.MINI_LEAGUE:
            # Huge differential bonus: owning what rivals don't = rank gain
            df["adj_xp_rival"] = (
                df["xp"]
                + 0.5 * (1 - df["rival_eo"]) * df["xp"]
                - 0.1 * df["rival_eo"] * df["xp"]
            )
        elif gamestate == GameState.CHASING:
            df["adj_xp_rival"] = (
                df["xp"] + 0.25 * (1 - df["rival_eo"]) * df["xp"]
            )
        elif gamestate == GameState.LEADING:
            # Slight template bonus — don't fall behind on template players
            df["adj_xp_rival"] = (
                df["xp"] + 0.1 * df["rival_eo"] * df["xp"]
            )
        else:
            df["adj_xp_rival"] = df["xp"].copy()

        return df

    def get_captain_differential(
        self,
        predictions_df: pd.DataFrame,
        rival_template: pd.DataFrame,
        min_xp: float = 5.0,
        max_rival_captain_pct: float = 30.0,
    ) -> pd.DataFrame:
        """Find high-xP captain options that rivals won't captain.

        Returns:
            DataFrame sorted by captain differential score.
        """
        df = predictions_df.merge(rival_template, on="element_id", how="left")
        df["rival_captain_pct"] = df["rival_captain_pct"].fillna(0.0)

        candidates = df[
            (df["xp"] >= min_xp)
            & (df["rival_captain_pct"] <= max_rival_captain_pct)
        ].copy()

        # Captain differential value: xP × (1 - rival_captain_pct) × 2 (doubling)
        candidates["captain_diff_score"] = (
            candidates["xp"] * (1 - candidates["rival_captain_pct"] / 100) * 2
        )
        return candidates.sort_values("captain_diff_score", ascending=False).head(5)

    # ── Points gap ───────────────────────────────────────────────────────

    def compute_points_gap(
        self,
        my_points: int,
        rival_points_list: list[int],
        gws_remaining: int,
    ) -> dict:
        """Analyze points gap and recommend strategy.

        Args:
            my_points: Your total points.
            rival_points_list: List of rival total points.
            gws_remaining: Gameweeks left in season.

        Returns:
            Dict with gap, strategy, suggested_gamestate, details.
        """
        if not rival_points_list:
            return {
                "gap": 0,
                "gap_to_leader": 0,
                "avg_gap": 0.0,
                "strategy": "neutral",
                "suggested_gamestate": GameState.NEUTRAL,
            }

        max_rival = max(rival_points_list)
        avg_rival = sum(rival_points_list) / len(rival_points_list)
        gap_to_leader = my_points - max_rival  # negative = behind
        avg_gap = my_points - avg_rival

        # Typical points per GW is ~50
        catchable = abs(gap_to_leader) <= gws_remaining * 50

        if gap_to_leader > 20:
            strategy = "safe"
            gamestate = GameState.LEADING
        elif gap_to_leader >= -10:
            strategy = "competitive"
            gamestate = GameState.NEUTRAL
        elif catchable:
            strategy = "need_differentials"
            gamestate = GameState.CHASING
        else:
            strategy = "desperate"
            gamestate = GameState.MINI_LEAGUE

        gaps = [my_points - rp for rp in rival_points_list]
        return {
            "gap_to_leader": gap_to_leader,
            "avg_gap": round(avg_gap, 1),
            "gaps": gaps,
            "strategy": strategy,
            "suggested_gamestate": gamestate,
            "gws_remaining": gws_remaining,
            "catchable": catchable,
            "leading_rivals": sum(1 for g in gaps if g > 0),
            "trailing_rivals": sum(1 for g in gaps if g < 0),
        }

    # ── Report ───────────────────────────────────────────────────────────

    def generate_report(
        self,
        my_squad_ids: list[int],
        rival_squads: list[dict],
        predictions_df: pd.DataFrame,
        my_points: int,
        rival_points: list[int],
        gws_remaining: int = 10,
    ) -> str:
        """Generate a formatted rival intelligence report.

        Returns:
            Multi-line string suitable for printing or display.
        """
        rival_template = self.compute_rival_template(rival_squads)
        differentials = self.compute_differential_opportunities(
            my_squad_ids, rival_template, predictions_df
        )
        risks = self.compute_risk_players(my_squad_ids, rival_template, predictions_df)
        gap_analysis = self.compute_points_gap(my_points, rival_points, gws_remaining)

        lines = [
            "╔═══════════════════════════════════════════════════╗",
            "║       RIVAL INTELLIGENCE REPORT                  ║",
            "╚═══════════════════════════════════════════════════╝",
            "",
            f"  My Points    : {my_points}",
            f"  Gap to Leader: {gap_analysis['gap_to_leader']:+d} pts",
            f"  Avg Gap      : {gap_analysis['avg_gap']:+.1f} pts",
            f"  GWs Left     : {gws_remaining}",
            f"  Strategy     : {gap_analysis['strategy'].upper()}",
            f"  Gamestate    : {gap_analysis['suggested_gamestate'].value}",
            "",
        ]

        if not rival_squads:
            lines.append("  (No rival data available)")
        else:
            lines.append(f"  Tracking {len(rival_squads)} rivals:")
            for r in rival_squads[:5]:
                lines.append(f"    {r['name']:<20} {r['total_points']:>5} pts  (rank {r['rank']})")

        if not differentials.empty:
            lines.extend([
                "",
                "  ── DIFFERENTIAL OPPORTUNITIES ──",
                f"  {'Player':<18} {'Pos':<5} {'xP':<6} {'Rival Own%':<12} {'Diff Score':<10}",
                f"  {'─'*18} {'─'*5} {'─'*6} {'─'*12} {'─'*10}",
            ])
            for _, row in differentials.head(5).iterrows():
                lines.append(
                    f"  {str(row.get('name', row['element_id'])):<18} "
                    f"{str(row.get('position', '')):<5} "
                    f"{row.get('xp', 0):<6.2f} "
                    f"{row['rival_ownership_pct']:<12.1f} "
                    f"{row['differential_score']:<10.2f}"
                )

        if not risks.empty:
            lines.extend([
                "",
                "  ── RISK PLAYERS (rivals own, you don't) ──",
            ])
            for _, row in risks.head(5).iterrows():
                lines.append(
                    f"  ⚠  {str(row.get('name', row['element_id'])):<18} "
                    f"{row['rival_ownership_pct']:.0f}% rival own  "
                    f"xP: {row.get('xp', 0):.2f}"
                )

        return "\n".join(lines)
