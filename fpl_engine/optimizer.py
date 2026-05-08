"""Ownership-Aware MILP Optimizer — the meta-game engine.

This is NOT a standard "pick highest xP" optimizer. It maximizes
EXPECTED RANK GAIN by incorporating:

1. Effective ownership (EO) — picking the same players as everyone else
   gives no edge. Differentials are weighted by (1 - EO) × xP.

2. Gamestate awareness — when leading, minimize variance (template).
   When chasing, maximize differential upside.

3. Multi-week planning — sequences transfers over a horizon with
   penalty accounting for extra transfers (-4 per hit).

4. Chip strategy — Wildcard, Free Hit, Bench Boost, Triple Captain
   as strategic options to be placed at maximum-value gameweeks.

Solver: PuLP with HiGHS (or CBC fallback).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

try:
    import pulp
    HAS_PULP = True
except ImportError:
    HAS_PULP = False

from .config import (
    DIFFERENTIAL_LAMBDA,
    FORMATION_CONSTRAINTS,
    MAX_PER_TEAM,
    POSITION_CONSTRAINTS,
    RISK_PENALTY_MU,
    SQUAD_SIZE,
    STARTING_XI,
    TOTAL_BUDGET,
    TRANSFER_HIT_COST,
)


class GameState(Enum):
    """Strategic posture based on current rank vs target."""
    LEADING = "leading"           # Play safe, match template
    CHASING = "chasing"           # Go differential, accept variance
    NEUTRAL = "neutral"           # Balanced approach
    MINI_LEAGUE = "mini_league"   # Target specific rival weaknesses


class Chip(Enum):
    NONE = "none"
    WILDCARD = "wildcard"
    FREE_HIT = "free_hit"
    BENCH_BOOST = "bench_boost"
    TRIPLE_CAPTAIN = "triple_captain"


@dataclass
class SquadConstraints:
    """Constraints for the optimization problem."""
    budget: int = TOTAL_BUDGET
    squad_size: int = SQUAD_SIZE
    max_per_team: int = MAX_PER_TEAM
    position_constraints: dict = field(default_factory=lambda: dict(POSITION_CONSTRAINTS))
    formation_constraints: dict = field(default_factory=lambda: dict(FORMATION_CONSTRAINTS))
    # Locked players (must include)
    must_include: list[int] = field(default_factory=list)
    # Banned players (must exclude)
    must_exclude: list[int] = field(default_factory=list)


@dataclass
class OptimizationResult:
    """Result of the optimization."""
    squad: pd.DataFrame           # 15-player squad with xP and role
    starting_xi: pd.DataFrame     # 11 starters
    bench: pd.DataFrame           # 4 bench players (ordered)
    captain_id: int               # Captain element_id
    vice_captain_id: int          # Vice captain element_id
    total_xp: float               # Total expected points (starting XI + captain bonus)
    differential_score: float     # Ownership-weighted score
    transfers_suggested: list[dict] = field(default_factory=list)
    chip_recommendation: Chip = Chip.NONE


class FPLOptimizer:
    """Ownership-aware MILP squad optimizer."""

    def __init__(
        self,
        gamestate: GameState = GameState.NEUTRAL,
        differential_lambda: float = DIFFERENTIAL_LAMBDA,
        risk_mu: float = RISK_PENALTY_MU,
    ):
        if not HAS_PULP:
            raise ImportError("pulp required. pip install pulp")
        self.gamestate = gamestate
        self.lam = differential_lambda
        self.mu = risk_mu

    def optimize_squad(
        self,
        players: pd.DataFrame,
        constraints: SquadConstraints | None = None,
        chip: Chip = Chip.NONE,
        n_gameweeks: int = 1,
    ) -> OptimizationResult:
        """Find the optimal 15-player squad.

        Args:
            players: DataFrame with columns:
                element_id, position, team_id, price, xp (expected points),
                ownership_pct (0-100), name
            constraints: Squad constraints (budget, position limits, etc.)
            chip: Active chip for this gameweek.
            n_gameweeks: Number of gameweeks to optimize over.

        Returns:
            OptimizationResult with squad, XI, captain, and scores.
        """
        constraints = constraints or SquadConstraints()
        players = players.copy()

        # ── Compute ownership-adjusted xP ────────────────────────────────
        players["eo"] = players["ownership_pct"].clip(0, 100) / 100.0
        players = self._compute_adjusted_xp(players)

        # Scale for multi-GW optimization
        if n_gameweeks > 1:
            players["adj_xp"] = players["adj_xp"] * n_gameweeks

        # ── MILP: Squad selection ────────────────────────────────────────
        squad_ids = self._solve_squad_milp(players, constraints, chip)
        squad = players[players["element_id"].isin(squad_ids)].copy()

        # ── MILP: Starting XI + formation ────────────────────────────────
        xi_ids = self._solve_xi_milp(squad, constraints)
        squad["in_xi"] = squad["element_id"].isin(xi_ids)

        starting_xi = squad[squad["in_xi"]].sort_values("xp", ascending=False)
        bench = squad[~squad["in_xi"]].copy()
        bench = self._order_bench(bench)

        # ── Captain selection ────────────────────────────────────────────
        captain_id, vice_id = self._select_captain(starting_xi)

        # ── Compute total xP ────────────────────────────────────────────
        captain_bonus = starting_xi.loc[
            starting_xi["element_id"] == captain_id, "xp"
        ].iloc[0]

        if chip == Chip.TRIPLE_CAPTAIN:
            total_xp = starting_xi["xp"].sum() + captain_bonus * 2
        elif chip == Chip.BENCH_BOOST:
            total_xp = squad["xp"].sum() + captain_bonus
        else:
            total_xp = starting_xi["xp"].sum() + captain_bonus

        differential_score = squad["adj_xp"].sum()

        return OptimizationResult(
            squad=squad.sort_values(["in_xi", "xp"], ascending=[False, False]),
            starting_xi=starting_xi,
            bench=bench,
            captain_id=captain_id,
            vice_captain_id=vice_id,
            total_xp=total_xp,
            differential_score=differential_score,
            chip_recommendation=chip,
        )

    def optimize_transfers(
        self,
        current_squad: pd.DataFrame,
        all_players: pd.DataFrame,
        free_transfers: int = 1,
        bank: int = 0,
        horizon: int = 3,
    ) -> list[dict]:
        """Find optimal transfers over a multi-week horizon.

        Args:
            current_squad: Current 15-player squad with element_id, price, sell_price.
            all_players: All available players with xp predictions for next N GWs.
            free_transfers: Number of free transfers available.
            bank: Money in the bank (0.1m units).
            horizon: Number of GWs to plan ahead.

        Returns:
            List of transfer suggestions: [{out: id, in: id, gain: xp_delta}, ...]
        """
        current_ids = set(current_squad["element_id"].tolist())
        budget = current_squad["price"].sum() + bank

        # Compute marginal value of each potential swap
        transfers = []

        for _, player_out in current_squad.iterrows():
            # Find valid replacements
            same_pos = all_players[
                (all_players["position"] == player_out["position"])
                & (~all_players["element_id"].isin(current_ids))
            ]

            for _, player_in in same_pos.iterrows():
                # Budget check: can we afford this swap?
                new_budget_used = budget - player_out["price"] + player_in["price"]
                if new_budget_used > budget + bank:
                    continue

                # Team constraint check
                team_count = current_squad[
                    (current_squad["team_id"] == player_in["team_id"])
                    & (current_squad["element_id"] != player_out["element_id"])
                ].shape[0]
                if team_count >= MAX_PER_TEAM:
                    continue

                xp_gain = (player_in.get("xp", 0) - player_out.get("xp", 0)) * horizon
                transfers.append({
                    "out_id": player_out["element_id"],
                    "out_name": player_out.get("name", ""),
                    "in_id": player_in["element_id"],
                    "in_name": player_in.get("name", ""),
                    "xp_gain": xp_gain,
                    "cost_delta": player_in["price"] - player_out["price"],
                })

        # Sort by xP gain and select top transfers
        transfers.sort(key=lambda t: t["xp_gain"], reverse=True)

        # Greedy selection respecting free transfer limit
        selected = []
        used_outs = set()
        used_ins = set()
        hits = 0

        for t in transfers:
            if t["out_id"] in used_outs or t["in_id"] in used_ins:
                continue
            if len(selected) >= free_transfers:
                # Additional transfer costs -4
                if t["xp_gain"] <= TRANSFER_HIT_COST:
                    break
                hits += 1
                t["hit"] = True
            else:
                t["hit"] = False

            selected.append(t)
            used_outs.add(t["out_id"])
            used_ins.add(t["in_id"])

            if len(selected) >= free_transfers + 2:  # max 2 hits
                break

        return selected

    def suggest_chip(
        self,
        squad: pd.DataFrame,
        all_players: pd.DataFrame,
        upcoming_fixtures: pd.DataFrame,
        available_chips: list[Chip],
    ) -> dict[Chip, float]:
        """Score each available chip for the upcoming gameweek.

        Returns:
            Dict of {Chip: estimated_value} — play the highest-value chip.
        """
        scores = {}

        if Chip.BENCH_BOOST in available_chips:
            # Value = sum of bench xP (higher when bench is strong)
            bench = squad.nsmallest(4, "xp")
            scores[Chip.BENCH_BOOST] = bench["xp"].sum()

        if Chip.TRIPLE_CAPTAIN in available_chips:
            # Value = captain's xP (play in DGW when captain has 2 fixtures)
            captain_xp = squad["xp"].max()
            scores[Chip.TRIPLE_CAPTAIN] = captain_xp

        if Chip.FREE_HIT in available_chips:
            # Value = optimal unconstrained squad xP - current squad xP
            optimal = self.optimize_squad(all_players)
            current_xp = squad.nlargest(11, "xp")["xp"].sum()
            scores[Chip.FREE_HIT] = optimal.total_xp - current_xp

        if Chip.WILDCARD in available_chips:
            # Value is complex — approximate as sum of transfer gains over 6 GWs
            transfers = self.optimize_transfers(
                squad, all_players, free_transfers=15, horizon=6
            )
            scores[Chip.WILDCARD] = sum(t["xp_gain"] for t in transfers[:15])

        scores[Chip.NONE] = 0.0
        return scores

    # ── Internal methods ─────────────────────────────────────────────────

    def _compute_adjusted_xp(self, players: pd.DataFrame) -> pd.DataFrame:
        """Compute ownership-adjusted expected points based on gamestate."""
        players = players.copy()

        if self.gamestate == GameState.CHASING:
            # Maximize differential upside
            players["adj_xp"] = (
                players["xp"]
                + self.lam * 2.0 * (1 - players["eo"]) * players["xp"]
            )
        elif self.gamestate == GameState.LEADING:
            # Match template — slightly prefer high-ownership players
            players["adj_xp"] = (
                players["xp"]
                + self.lam * 0.3 * players["eo"] * players["xp"]
                - self.mu * (1 - players["eo"]) * players["xp"]  # penalize differentials
            )
        elif self.gamestate == GameState.MINI_LEAGUE:
            # Strong differential weighting
            players["adj_xp"] = (
                players["xp"]
                + self.lam * 3.0 * (1 - players["eo"]) * players["xp"]
            )
        else:  # NEUTRAL
            players["adj_xp"] = (
                players["xp"]
                + self.lam * (1 - players["eo"]) * players["xp"]
            )

        return players

    def _solve_squad_milp(
        self,
        players: pd.DataFrame,
        constraints: SquadConstraints,
        chip: Chip,
    ) -> list[int]:
        """Solve the squad selection MILP."""
        prob = pulp.LpProblem("FPL_Squad", pulp.LpMaximize)

        # Decision variables: x_i ∈ {0, 1} for each player
        player_ids = players["element_id"].tolist()
        x = pulp.LpVariable.dicts("x", player_ids, cat="Binary")

        # Objective: maximize adjusted xP
        player_xp = dict(zip(players["element_id"], players["adj_xp"]))
        prob += pulp.lpSum(player_xp[i] * x[i] for i in player_ids)

        # Budget constraint
        player_price = dict(zip(players["element_id"], players["price"]))
        prob += pulp.lpSum(player_price[i] * x[i] for i in player_ids) <= constraints.budget

        # Squad size
        prob += pulp.lpSum(x[i] for i in player_ids) == constraints.squad_size

        # Position constraints
        player_pos = dict(zip(players["element_id"], players["position"]))
        for pos, (min_count, max_count) in constraints.position_constraints.items():
            pos_players = [i for i in player_ids if player_pos[i] == pos]
            prob += pulp.lpSum(x[i] for i in pos_players) >= min_count
            prob += pulp.lpSum(x[i] for i in pos_players) <= max_count

        # Max per team
        player_team = dict(zip(players["element_id"], players["team_id"]))
        team_ids = players["team_id"].unique()
        for tid in team_ids:
            team_players = [i for i in player_ids if player_team[i] == tid]
            prob += pulp.lpSum(x[i] for i in team_players) <= constraints.max_per_team

        # Must include/exclude
        for pid in constraints.must_include:
            if pid in x:
                prob += x[pid] == 1
        for pid in constraints.must_exclude:
            if pid in x:
                prob += x[pid] == 0

        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        if prob.status != 1:
            raise RuntimeError(f"Optimization failed with status: {pulp.LpStatus[prob.status]}")

        return [i for i in player_ids if x[i].varValue > 0.5]

    def _solve_xi_milp(
        self,
        squad: pd.DataFrame,
        constraints: SquadConstraints,
    ) -> list[int]:
        """Select starting XI from the 15-player squad with formation constraints."""
        prob = pulp.LpProblem("FPL_XI", pulp.LpMaximize)

        squad_ids = squad["element_id"].tolist()
        x = pulp.LpVariable.dicts("xi", squad_ids, cat="Binary")

        # Objective: maximize xP for starting XI
        player_xp = dict(zip(squad["element_id"], squad["xp"]))
        prob += pulp.lpSum(player_xp[i] * x[i] for i in squad_ids)

        # Exactly 11 starters
        prob += pulp.lpSum(x[i] for i in squad_ids) == STARTING_XI

        # Formation constraints
        player_pos = dict(zip(squad["element_id"], squad["position"]))
        for pos, (min_count, max_count) in constraints.formation_constraints.items():
            pos_players = [i for i in squad_ids if player_pos[i] == pos]
            if pos_players:
                prob += pulp.lpSum(x[i] for i in pos_players) >= min_count
                prob += pulp.lpSum(x[i] for i in pos_players) <= max_count

        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        if prob.status != 1:
            # Fallback: just pick top 11 by xP respecting 1 GKP minimum
            gkps = squad[squad["position"] == "GKP"].nlargest(1, "xp")["element_id"].tolist()
            outfield = squad[squad["position"] != "GKP"].nlargest(10, "xp")["element_id"].tolist()
            return gkps + outfield

        return [i for i in squad_ids if x[i].varValue > 0.5]

    def _select_captain(self, starting_xi: pd.DataFrame) -> tuple[int, int]:
        """Select captain and vice-captain.

        Captain: highest adj_xp in XI.
        In CHASING gamestate, prefer lower-ownership high-xP players.
        """
        xi = starting_xi.copy()

        if self.gamestate == GameState.CHASING:
            # Weight by differential potential
            xi["captain_score"] = xi["xp"] * (1 + 0.5 * (1 - xi.get("eo", 0.5)))
        elif self.gamestate == GameState.LEADING:
            # Weight by ownership (safe pick)
            xi["captain_score"] = xi["xp"] * (1 + 0.3 * xi.get("eo", 0.5))
        else:
            xi["captain_score"] = xi["xp"]

        xi_sorted = xi.sort_values("captain_score", ascending=False)
        captain_id = int(xi_sorted.iloc[0]["element_id"])
        vice_id = int(xi_sorted.iloc[1]["element_id"])

        return captain_id, vice_id

    def _order_bench(self, bench: pd.DataFrame) -> pd.DataFrame:
        """Order bench players: GKP first, then by xP descending."""
        gkps = bench[bench["position"] == "GKP"].sort_values("xp", ascending=False)
        outfield = bench[bench["position"] != "GKP"].sort_values("xp", ascending=False)
        return pd.concat([gkps, outfield]).reset_index(drop=True)
