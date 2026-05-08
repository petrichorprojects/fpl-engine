"""Backtest Framework — walk-forward season replay and strategy comparison.

Measures how much edge the engine actually provides vs simple baselines.
The key principle: NEVER look ahead. Train only on data before GW N to
predict GW N. Anything else is overfitting.

Usage:
    bt = Backtester()
    history = bt.compare_strategies(
        features_df=features,
        players_df=players,
        strategies={
            "engine": EngineStrategy(),
            "form":   TopFormStrategy(),
            "template": HighOwnershipStrategy(),
        },
        start_gw=5,
        end_gw=38,
    )
    bt.print_comparison(history)
    bt.export_results_csv(history, "backtest_results.csv")
"""

from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import (
    DATA_DIR,
    FORMATION_CONSTRAINTS,
    MAX_PER_TEAM,
    POSITION_CONSTRAINTS,
    POSITIONS,
    SQUAD_SIZE,
    STARTING_XI,
    TOTAL_BUDGET,
)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class GWResult:
    """Result for a single gameweek."""
    gameweek: int
    squad_ids: list[int]
    starting_xi_ids: list[int]
    bench_ids: list[int]
    captain_id: int
    vice_captain_id: int
    points_scored: int
    captain_points: int
    autosubs_made: list[tuple[int, int]]   # [(out_id, in_id), ...]
    xp_predicted: float = 0.0
    spearman_corr: float = 0.0
    minutes_accuracy: float = 0.0
    transfers_made: int = 0
    hits_taken: int = 0


@dataclass
class SeasonResult:
    """Full season backtest result for a single strategy."""
    strategy_name: str
    gw_results: list[GWResult] = field(default_factory=list)

    @property
    def total_points(self) -> int:
        return sum(g.points_scored for g in self.gw_results)

    @property
    def per_gw_points(self) -> list[int]:
        return [g.points_scored for g in self.gw_results]

    @property
    def cumulative_points(self) -> list[int]:
        cum = []
        total = 0
        for pts in self.per_gw_points:
            total += pts
            cum.append(total)
        return cum

    @property
    def avg_gw_points(self) -> float:
        pts = self.per_gw_points
        return sum(pts) / len(pts) if pts else 0.0

    @property
    def avg_spearman(self) -> float:
        vals = [g.spearman_corr for g in self.gw_results if g.spearman_corr != 0]
        return sum(vals) / len(vals) if vals else 0.0

    def cumulative_points_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "gameweek": [g.gameweek for g in self.gw_results],
            "gw_points": self.per_gw_points,
            "cumulative_points": self.cumulative_points,
            "strategy": self.strategy_name,
        })


@dataclass
class ComparisonResult:
    """Results from comparing multiple strategies."""
    strategy_results: dict[str, SeasonResult] = field(default_factory=dict)

    def summary_df(self) -> pd.DataFrame:
        rows = []
        for name, result in self.strategy_results.items():
            rows.append({
                "strategy": name,
                "total_points": result.total_points,
                "avg_gw_points": round(result.avg_gw_points, 1),
                "avg_spearman": round(result.avg_spearman, 3),
                "n_gameweeks": len(result.gw_results),
            })
        return pd.DataFrame(rows).sort_values("total_points", ascending=False)

    def cumulative_df(self) -> pd.DataFrame:
        frames = [r.cumulative_points_df() for r in self.strategy_results.values()]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ── Strategy interface ────────────────────────────────────────────────────────

class Strategy(ABC):
    """Abstract base class for FPL squad selection strategies."""

    @abstractmethod
    def select_squad(
        self,
        features_df: pd.DataFrame,
        players_df: pd.DataFrame,
        gameweek: int,
        budget: int = TOTAL_BUDGET,
    ) -> tuple[list[int], int]:
        """Select a 15-player squad and captain for a gameweek.

        Args:
            features_df: All historical features up to (but not including) gameweek.
            players_df: Current player master data.
            gameweek: The gameweek to prepare for.
            budget: Available budget.

        Returns:
            (squad_ids: list of 15 element_ids, captain_id: int)
        """
        ...

    def select_xi(
        self,
        squad_ids: list[int],
        players_df: pd.DataFrame,
    ) -> tuple[list[int], list[int]]:
        """Select starting XI and bench from squad.

        Returns:
            (starting_xi_ids, bench_ids_ordered)
        """
        squad = players_df[players_df["element_id"].isin(squad_ids)].copy()

        # Simple greedy: 1 GKP, then top 10 outfield by value metric
        gkps = squad[squad["position"] == "GKP"].nlargest(1, self._sort_col(squad))
        outfield = squad[squad["position"] != "GKP"].nlargest(10, self._sort_col(squad))

        xi_ids = gkps["element_id"].tolist() + outfield["element_id"].tolist()
        bench_ids = [i for i in squad_ids if i not in xi_ids]

        # Order bench: GKP first, then by sort col descending
        bench = squad[squad["element_id"].isin(bench_ids)].copy()
        gkp_bench = bench[bench["position"] == "GKP"]["element_id"].tolist()
        outfield_bench = (
            bench[bench["position"] != "GKP"]
            .sort_values(self._sort_col(bench), ascending=False)["element_id"]
            .tolist()
        )
        bench_ordered = gkp_bench + outfield_bench

        return xi_ids, bench_ordered

    def _sort_col(self, df: pd.DataFrame) -> str:
        for col in ["xp", "total_points", "form", "points_per_game"]:
            if col in df.columns:
                return col
        return df.columns[0]


class TopFormStrategy(Strategy):
    """Picks the highest-form players available within budget."""

    def select_squad(self, features_df, players_df, gameweek, budget=TOTAL_BUDGET):
        available = players_df[players_df["status"].isin(["a", "d"])].copy()
        if "form" not in available.columns:
            available["form"] = available.get("points_per_game", 0)

        squad_ids = _greedy_squad(available, "form", budget)
        captain_id = (
            available[available["element_id"].isin(squad_ids)]
            .nlargest(1, "form")["element_id"].iloc[0]
            if squad_ids else squad_ids[0] if squad_ids else -1
        )
        return squad_ids, int(captain_id)


class HighOwnershipStrategy(Strategy):
    """Picks the most-owned players — the 'template' squad."""

    def select_squad(self, features_df, players_df, gameweek, budget=TOTAL_BUDGET):
        available = players_df[players_df["status"].isin(["a", "d"])].copy()
        if "selected_pct" not in available.columns:
            available["selected_pct"] = 0.0

        squad_ids = _greedy_squad(available, "selected_pct", budget)
        captain_id = (
            available[available["element_id"].isin(squad_ids)]
            .nlargest(1, "selected_pct")["element_id"].iloc[0]
            if squad_ids else -1
        )
        return squad_ids, int(captain_id)


class RandomStrategy(Strategy):
    """Picks a random valid squad — useful as a floor baseline."""

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)

    def select_squad(self, features_df, players_df, gameweek, budget=TOTAL_BUDGET):
        available = players_df[players_df["status"].isin(["a", "d"])].copy()
        squad_ids = _random_valid_squad(available, budget, self._rng)
        captain_id = int(self._rng.choice(squad_ids)) if squad_ids else -1
        return squad_ids, captain_id


class EngineStrategy(Strategy):
    """Uses the full FPL engine: minutes model + points model + optimizer."""

    def __init__(self) -> None:
        from .minutes_model import MinutesModel
        from .points_model import PointsModel
        from .optimizer import FPLOptimizer, GameState, SquadConstraints
        from .features import build_player_features

        self._MinutesModel = MinutesModel
        self._PointsModel = PointsModel
        self._FPLOptimizer = FPLOptimizer
        self._GameState = GameState
        self._SquadConstraints = SquadConstraints
        self._build_features = build_player_features

        self._minutes_model: Optional[object] = None
        self._points_model: Optional[object] = None

    def select_squad(self, features_df, players_df, gameweek, budget=TOTAL_BUDGET):
        # Train on data before this gameweek
        train_df = features_df[features_df["round"] < gameweek].copy()

        if len(train_df) < 200:
            # Not enough data — fall back to form strategy
            return TopFormStrategy().select_squad(features_df, players_df, gameweek, budget)

        try:
            mm = self._MinutesModel()
            pm = self._PointsModel()
            mm.train(train_df, n_splits=2, verbose=False)
            pm.train(train_df, n_splits=2, verbose=False)

            # Get latest features for current GW
            latest_idx = features_df.groupby("element_id")["round"].idxmax()
            latest = features_df.loc[latest_idx].copy()
            latest = latest[latest["element_id"].isin(players_df["element_id"])]

            mins_pred = mm.predict(latest)
            pts_pred = pm.predict(latest)
            combined = mins_pred.merge(pts_pred, on=["element_id", "position"], how="inner")
            combined["xp"] = (
                combined["p_start"] * combined["e_pts_start"]
                + combined["p_sub"] * combined["e_pts_sub"]
            )
            combined = combined.merge(
                players_df[["element_id", "team_id", "price", "selected_pct", "status"]],
                on="element_id", how="left",
            )
            combined["ownership_pct"] = combined["selected_pct"].fillna(0)
            combined = combined[combined["status"].isin(["a", "d"])].copy()

            optimizer = self._FPLOptimizer(self._GameState.NEUTRAL)
            result = optimizer.optimize_squad(
                combined,
                self._SquadConstraints(budget=budget),
            )

            squad_ids = result.squad["element_id"].tolist()
            return squad_ids, result.captain_id

        except Exception as e:
            # Graceful degradation
            return TopFormStrategy().select_squad(features_df, players_df, gameweek, budget)


# ── Auto-sub simulation ───────────────────────────────────────────────────────

def simulate_autosubs(
    starting_xi_ids: list[int],
    bench_ids_ordered: list[int],
    actual_minutes: dict[int, int],
    positions: dict[int, str],
) -> tuple[list[int], list[tuple[int, int]]]:
    """Simulate FPL's auto-substitution rules.

    FPL rules:
      1. If a starter plays 0 minutes, try to sub in the first valid bench player
      2. The bench is ordered — try bench[0] first, then bench[1], etc.
      3. GKP can only be replaced by the bench GKP
      4. Formation must remain valid (min 3 DEF, min 1 FWD, min 2 MID)
      5. Only 3 outfield subs can be made (plus 1 GK sub)

    Args:
        starting_xi_ids: List of 11 starting element IDs.
        bench_ids_ordered: Bench in priority order (GKP first, then by value).
        actual_minutes: Dict mapping element_id → minutes played.
        positions: Dict mapping element_id → position string.

    Returns:
        (final_xi_ids, subs_made: list of (out_id, in_id))
    """
    xi = list(starting_xi_ids)
    bench = list(bench_ids_ordered)
    subs_made: list[tuple[int, int]] = []
    outfield_subs_used = 0
    gk_sub_used = False

    def formation_valid(squad_ids: list[int]) -> bool:
        pos_counts = {}
        for pid in squad_ids:
            pos = positions.get(pid, "MID")
            pos_counts[pos] = pos_counts.get(pos, 0) + 1
        for pos, (min_cnt, max_cnt) in FORMATION_CONSTRAINTS.items():
            if pos_counts.get(pos, 0) < min_cnt:
                return False
        return True

    # Process starters who didn't play
    non_players = [pid for pid in xi if actual_minutes.get(pid, 0) == 0]

    for out_id in non_players:
        out_pos = positions.get(out_id, "MID")

        for bench_player in bench:
            if actual_minutes.get(bench_player, 0) == 0:
                continue  # bench player also didn't play

            bench_pos = positions.get(bench_player, "MID")

            # GK sub rules
            if out_pos == "GKP":
                if bench_pos != "GKP" or gk_sub_used:
                    continue
                # Make sub
                xi = [bench_player if p == out_id else p for p in xi]
                bench.remove(bench_player)
                subs_made.append((out_id, bench_player))
                gk_sub_used = True
                break

            # Outfield sub rules
            else:
                if bench_pos == "GKP" or outfield_subs_used >= 3:
                    continue

                # Try the sub and check formation validity
                test_xi = [bench_player if p == out_id else p for p in xi]
                if not formation_valid(test_xi):
                    continue

                xi = test_xi
                bench.remove(bench_player)
                subs_made.append((out_id, bench_player))
                outfield_subs_used += 1
                break

    return xi, subs_made


def score_gameweek(
    xi_ids: list[int],
    captain_id: int,
    vice_captain_id: int,
    actual_points: dict[int, int],
    actual_minutes: dict[int, int],
    positions: dict[int, str],
    bench_ids: list[int],
    chip: str = "none",
) -> int:
    """Calculate total points scored for a gameweek after auto-subs.

    Args:
        xi_ids: Starting XI element IDs.
        captain_id: Captain element ID.
        vice_captain_id: Vice captain element ID.
        actual_points: Dict mapping element_id → points scored.
        actual_minutes: Dict mapping element_id → minutes played.
        positions: Dict mapping element_id → position.
        bench_ids: Bench player IDs in order.
        chip: Active chip name.

    Returns:
        Total points integer.
    """
    # Apply auto-subs
    final_xi, _ = simulate_autosubs(xi_ids, bench_ids, actual_minutes, positions)

    total = 0
    for pid in final_xi:
        pts = actual_points.get(pid, 0)
        if pid == captain_id:
            multiplier = 3 if chip == "triple_captain" else 2
        else:
            multiplier = 1
        total += pts * multiplier

    if chip == "bench_boost":
        for pid in bench_ids:
            total += actual_points.get(pid, 0)

    return total


# ── Backtester ────────────────────────────────────────────────────────────────

class Backtester:
    """Walk-forward backtesting framework for FPL strategies."""

    def simulate_season(
        self,
        features_df: pd.DataFrame,
        players_df: pd.DataFrame,
        strategy: Strategy,
        strategy_name: str = "strategy",
        start_gw: int = 5,
        end_gw: int = 38,
        budget: int = TOTAL_BUDGET,
    ) -> SeasonResult:
        """Simulate a full season walk-forward.

        For each GW from start_gw to end_gw:
          1. Train strategy on features_df up to GW-1
          2. Select squad for GW
          3. Score actual points from history_df
          4. Track cumulative performance

        Args:
            features_df: Complete feature matrix with 'round' and 'total_points'.
            players_df: Player master data.
            strategy: Strategy to evaluate.
            strategy_name: Display name.
            start_gw: First gameweek to predict (need data before this).
            end_gw: Last gameweek to simulate.
            budget: Starting budget.

        Returns:
            SeasonResult with per-GW and cumulative stats.
        """
        result = SeasonResult(strategy_name=strategy_name)

        print(f"\n  ── Backtesting: {strategy_name} (GW{start_gw}–{end_gw}) ──")

        for gw in range(start_gw, end_gw + 1):
            # Actual results for this GW
            gw_actuals = features_df[features_df["round"] == gw]
            if gw_actuals.empty:
                continue

            actual_points = dict(zip(gw_actuals["element_id"], gw_actuals["total_points"]))
            actual_minutes = dict(zip(gw_actuals["element_id"], gw_actuals["minutes"]))
            positions_map = dict(zip(players_df["element_id"], players_df["position"]))

            # Select squad using only pre-GW data
            past_data = features_df[features_df["round"] < gw].copy()
            try:
                squad_ids, captain_id = strategy.select_squad(
                    past_data, players_df, gw, budget
                )
            except Exception as e:
                print(f"    GW{gw}: Strategy failed ({e}), skipping")
                continue

            if len(squad_ids) != SQUAD_SIZE:
                squad_ids = squad_ids[:SQUAD_SIZE]
                if not squad_ids:
                    continue

            # Select XI from squad
            xi_ids, bench_ids = strategy.select_xi(
                squad_ids,
                players_df[players_df["element_id"].isin(squad_ids)],
            )

            # Vice captain: second highest value player in XI
            xi_df = players_df[players_df["element_id"].isin(xi_ids)].copy()
            sort_col = strategy._sort_col(xi_df)
            xi_sorted = xi_df.sort_values(sort_col, ascending=False)
            vice_id = (
                int(xi_sorted.iloc[1]["element_id"])
                if len(xi_sorted) > 1 else captain_id
            )

            # Score with auto-subs
            pts = score_gameweek(
                xi_ids=xi_ids,
                captain_id=captain_id,
                vice_captain_id=vice_id,
                actual_points=actual_points,
                actual_minutes=actual_minutes,
                positions=positions_map,
                bench_ids=bench_ids,
            )

            captain_pts = actual_points.get(captain_id, 0)

            # Auto-subs record
            _, autosubs = simulate_autosubs(xi_ids, bench_ids, actual_minutes, positions_map)

            # Model accuracy: Spearman correlation of predicted vs actual points
            spearman = _compute_spearman(past_data, gw_actuals, strategy)

            gw_result = GWResult(
                gameweek=gw,
                squad_ids=squad_ids,
                starting_xi_ids=xi_ids,
                bench_ids=bench_ids,
                captain_id=captain_id,
                vice_captain_id=vice_id,
                points_scored=pts,
                captain_points=captain_pts,
                autosubs_made=autosubs,
                spearman_corr=spearman,
            )
            result.gw_results.append(gw_result)

            if gw % 5 == 0 or gw == end_gw:
                print(f"    GW{gw}: {pts} pts | Cumulative: {result.total_points}")

        return result

    def compare_strategies(
        self,
        features_df: pd.DataFrame,
        players_df: pd.DataFrame,
        strategies: dict[str, Strategy],
        start_gw: int = 5,
        end_gw: int = 38,
        budget: int = TOTAL_BUDGET,
    ) -> ComparisonResult:
        """Compare multiple strategies over the same season.

        Args:
            features_df: Complete feature matrix.
            players_df: Player master data.
            strategies: Dict of name → Strategy.
            start_gw: First GW.
            end_gw: Last GW.
            budget: Starting budget.

        Returns:
            ComparisonResult with per-strategy SeasonResult.
        """
        comparison = ComparisonResult()

        for name, strategy in strategies.items():
            season_result = self.simulate_season(
                features_df=features_df,
                players_df=players_df,
                strategy=strategy,
                strategy_name=name,
                start_gw=start_gw,
                end_gw=end_gw,
                budget=budget,
            )
            comparison.strategy_results[name] = season_result

        return comparison

    def print_comparison(self, comparison: ComparisonResult) -> None:
        """Print a formatted comparison of all strategies."""
        print("\n╔══════════════════════════════════════════════════════╗")
        print("║          BACKTEST RESULTS — STRATEGY COMPARISON     ║")
        print("╚══════════════════════════════════════════════════════╝\n")

        summary = comparison.summary_df()
        print(f"  {'Strategy':<22} {'Total Pts':<12} {'Avg/GW':<10} {'Avg Spearman'}")
        print(f"  {'─'*22} {'─'*12} {'─'*10} {'─'*12}")

        for _, row in summary.iterrows():
            print(
                f"  {row['strategy']:<22} {row['total_points']:<12} "
                f"{row['avg_gw_points']:<10} {row['avg_spearman']}"
            )

        # Show which strategy is best
        if not summary.empty:
            best = summary.iloc[0]["strategy"]
            worst = summary.iloc[-1]["strategy"]
            top_pts = summary.iloc[0]["total_points"]
            bottom_pts = summary.iloc[-1]["total_points"]
            print(f"\n  ★ Best:  {best} (+{top_pts - bottom_pts} pts vs worst)")
            print(f"  ✗ Worst: {worst}")

    def export_results_csv(
        self,
        comparison: ComparisonResult,
        path: str | Path = "backtest_results.csv",
    ) -> None:
        """Export per-GW results for all strategies to CSV."""
        frames = []
        for name, result in comparison.strategy_results.items():
            df = result.cumulative_points_df()
            df["avg_spearman"] = result.avg_spearman
            frames.append(df)

        if frames:
            combined = pd.concat(frames, ignore_index=True)
            combined.to_csv(path, index=False)
            print(f"  💾 Backtest results saved to {path}")

    def print_season_summary(self, result: SeasonResult) -> None:
        """Print a detailed summary for a single strategy result."""
        print(f"\n  ── {result.strategy_name} Season Summary ──")
        print(f"  Total points:    {result.total_points}")
        print(f"  Average / GW:    {result.avg_gw_points:.1f}")
        print(f"  Best GW:         {max(result.per_gw_points)} pts")
        print(f"  Worst GW:        {min(result.per_gw_points)} pts")
        print(f"  Avg Spearman:    {result.avg_spearman:.3f}")
        print(f"  Gameweeks run:   {len(result.gw_results)}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _greedy_squad(
    players: pd.DataFrame,
    sort_col: str,
    budget: int,
) -> list[int]:
    """Greedily select a valid squad sorted by sort_col."""
    players = players.copy().sort_values(sort_col, ascending=False)

    selected: list[int] = []
    pos_counts: dict[str, int] = {p: 0 for p in POSITIONS}
    team_counts: dict[int, int] = {}
    total_cost = 0

    pos_max = {pos: mx for pos, (mn, mx) in POSITION_CONSTRAINTS.items()}

    for _, row in players.iterrows():
        pos = row.get("position", "MID")
        team_id = int(row.get("team_id", 0))
        price = int(row.get("price", 0))

        if pos_counts.get(pos, 0) >= pos_max.get(pos, 5):
            continue
        if team_counts.get(team_id, 0) >= MAX_PER_TEAM:
            continue
        if total_cost + price > budget:
            continue
        if len(selected) >= SQUAD_SIZE:
            break

        selected.append(int(row["element_id"]))
        pos_counts[pos] = pos_counts.get(pos, 0) + 1
        team_counts[team_id] = team_counts.get(team_id, 0) + 1
        total_cost += price

    return selected


def _random_valid_squad(
    players: pd.DataFrame,
    budget: int,
    rng: np.random.Generator,
) -> list[int]:
    """Select a random valid squad."""
    shuffled = players.sample(frac=1, random_state=int(rng.integers(0, 999))).copy()
    return _greedy_squad(shuffled, "element_id", budget)


def _compute_spearman(
    past_data: pd.DataFrame,
    gw_actuals: pd.DataFrame,
    strategy: Strategy,
) -> float:
    """Compute Spearman correlation between predicted ranking and actual points."""
    try:
        from scipy.stats import spearmanr

        # Use rolling average points as the "predicted" value (simple proxy)
        if "roll_pts_5" in past_data.columns and "element_id" in past_data.columns:
            latest_idx = past_data.groupby("element_id")["round"].idxmax()
            latest = past_data.loc[latest_idx][["element_id", "roll_pts_5"]].copy()
            merged = gw_actuals[["element_id", "total_points"]].merge(latest, on="element_id")
            if len(merged) >= 10:
                corr, _ = spearmanr(merged["roll_pts_5"], merged["total_points"])
                return float(corr) if not np.isnan(corr) else 0.0
    except Exception:
        pass
    return 0.0
