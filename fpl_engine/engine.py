"""Unified FPL Engine — orchestrates data, models, and optimization.

This is the top-level interface. Usage:

    from fpl_engine import FPLEngine

    engine = FPLEngine()
    engine.fetch_data()          # Pull from FPL API
    engine.build_features()      # Engineer features
    engine.train()               # Train minutes + points models
    engine.predict()             # Generate xP for all players
    result = engine.optimize()   # Pick optimal squad
    engine.report()              # Print recommendations

Or for a returning user with trained models:

    engine = FPLEngine()
    engine.load_models()
    engine.fetch_data()
    engine.build_features()
    engine.predict()
    result = engine.optimize()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .client import FPLClient
from .config import POSITIONS, TOTAL_BUDGET
from .features import build_player_features
from .minutes_model import MinutesModel
from .optimizer import (
    Chip,
    FPLOptimizer,
    GameState,
    OptimizationResult,
    SquadConstraints,
)
from .points_model import PointsModel


@dataclass
class FPLEngine:
    """Unified FPL analytics engine.

    Wires together: data ingestion → feature engineering → minutes prediction
    → points prediction → ownership-aware optimization.
    """

    # Components
    client: FPLClient = field(default_factory=FPLClient)
    minutes_model: MinutesModel = field(default_factory=MinutesModel)
    points_model: PointsModel = field(default_factory=PointsModel)
    optimizer: FPLOptimizer = field(default_factory=lambda: FPLOptimizer(GameState.NEUTRAL))

    # Data state
    players_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    fixtures_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    teams_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    history_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    features_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Prediction state
    predictions_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    current_gw: int = 0

    def fetch_data(self, fetch_histories: bool = True, verbose: bool = True) -> None:
        """Fetch all data from the FPL API.

        Args:
            fetch_histories: If True, fetch per-fixture history for all players
                (~5 min with rate limiting). Set False to use cached data.
            verbose: Print progress.
        """
        if verbose:
            print("═══════════════════════════════════════════════════════")
            print("  FPL ENGINE — Data Fetch")
            print("═══════════════════════════════════════════════════════")

        self.players_df = self.client.get_players_df()
        if verbose:
            print(f"  ✓ Players: {len(self.players_df)}")

        self.fixtures_df = self.client.get_fixtures_df()
        if verbose:
            print(f"  ✓ Fixtures: {len(self.fixtures_df)}")

        self.teams_df = self.client.get_teams_df()
        if verbose:
            print(f"  ✓ Teams: {len(self.teams_df)}")

        self.current_gw = self.client.current_gameweek()
        if verbose:
            print(f"  ✓ Current GW: {self.current_gw}")

        if fetch_histories:
            if verbose:
                print("  ⏳ Fetching player histories (this takes ~5 min)...")
            self.history_df = self.client.get_all_player_histories(progress=verbose)
            if verbose:
                print(f"  ✓ History rows: {len(self.history_df)}")
        elif not self.history_df.empty:
            if verbose:
                print("  ℹ Using existing history data")

    def build_features(self, verbose: bool = True) -> None:
        """Build the unified feature matrix."""
        if self.history_df.empty:
            raise RuntimeError("No history data. Call fetch_data(fetch_histories=True) first.")

        if verbose:
            print("\n═══════════════════════════════════════════════════════")
            print("  FPL ENGINE — Feature Engineering")
            print("═══════════════════════════════════════════════════════")

        self.features_df = build_player_features(
            history_df=self.history_df,
            players_df=self.players_df,
            fixtures_df=self.fixtures_df,
            teams_df=self.teams_df,
        )

        if verbose:
            print(f"  ✓ Feature matrix: {self.features_df.shape[0]} rows × "
                  f"{self.features_df.shape[1]} columns")
            print(f"  ✓ Players with features: "
                  f"{self.features_df['element_id'].nunique()}")

    def train(self, verbose: bool = True) -> dict:
        """Train both the minutes model and points model."""
        if self.features_df.empty:
            raise RuntimeError("No features. Call build_features() first.")

        if verbose:
            print("\n═══════════════════════════════════════════════════════")
            print("  FPL ENGINE — Training Models")
            print("═══════════════════════════════════════════════════════")

        # Train minutes model
        if verbose:
            print("\n  ── Minutes Model ──")
        minutes_metrics = self.minutes_model.train(self.features_df, verbose=verbose)

        # Train points model
        if verbose:
            print("\n  ── Points Model ──")
        points_metrics = self.points_model.train(self.features_df, verbose=verbose)

        return {"minutes": minutes_metrics, "points": points_metrics}

    def predict(self, verbose: bool = True) -> pd.DataFrame:
        """Generate xP predictions for all players.

        Combines: P(start|sub|bench) from minutes model × E[pts|start/sub]
        from points model to produce final xP per player.
        """
        if not self.minutes_model.is_trained or not self.points_model.is_trained:
            raise RuntimeError("Models not trained. Call train() first.")

        if verbose:
            print("\n═══════════════════════════════════════════════════════")
            print("  FPL ENGINE — Predictions")
            print("═══════════════════════════════════════════════════════")

        # Use the latest gameweek features for prediction
        latest = self._get_latest_features()

        if verbose:
            print(f"  Predicting for {len(latest)} players...")

        # Minutes predictions
        mins_pred = self.minutes_model.predict(latest)
        if verbose:
            print(f"  ✓ Minutes predictions: {len(mins_pred)} players")

        # Points predictions (conditional on playing)
        pts_pred = self.points_model.predict(latest)
        if verbose:
            print(f"  ✓ Points predictions: {len(pts_pred)} players")

        # Combine: xP = P(start) × E[pts|start] + P(sub) × E[pts|sub]
        combined = mins_pred.merge(pts_pred, on=["element_id", "position"], how="inner")
        combined["xp"] = (
            combined["p_start"] * combined["e_pts_start"]
            + combined["p_sub"] * combined["e_pts_sub"]
        )

        # Merge with player metadata
        combined = combined.merge(
            self.players_df[["element_id", "name", "full_name", "team_id",
                             "team_name", "price", "selected_pct",
                             "status", "chance_next_round", "form"]],
            on="element_id",
            how="left",
        )
        combined["ownership_pct"] = combined["selected_pct"]

        # Sort by xP descending
        combined = combined.sort_values("xp", ascending=False).reset_index(drop=True)
        self.predictions_df = combined

        if verbose:
            print(f"\n  Top 10 by xP:")
            print(f"  {'Rank':<5} {'Player':<20} {'Pos':<5} {'Team':<6} "
                  f"{'Price':<7} {'P(Start)':<10} {'xP':<8}")
            print(f"  {'─'*5} {'─'*20} {'─'*5} {'─'*6} {'─'*7} {'─'*10} {'─'*8}")
            for i, row in combined.head(10).iterrows():
                print(f"  {i+1:<5} {row['name']:<20} {row['position']:<5} "
                      f"{row.get('team_name', '')[:5]:<6} "
                      f"£{row['price']/10:.1f}m  "
                      f"{row['p_start']:.2f}      "
                      f"{row['xp']:.2f}")

        return combined

    def optimize(
        self,
        budget: int | None = None,
        gamestate: GameState = GameState.NEUTRAL,
        chip: Chip = Chip.NONE,
        must_include: list[int] | None = None,
        must_exclude: list[int] | None = None,
        verbose: bool = True,
    ) -> OptimizationResult:
        """Run the ownership-aware optimizer.

        Args:
            budget: Total budget in 0.1m units (default: 1000 = £100.0m).
            gamestate: Strategic posture (LEADING/CHASING/NEUTRAL/MINI_LEAGUE).
            chip: Active chip for this GW.
            must_include: Player IDs that must be in the squad.
            must_exclude: Player IDs that must NOT be in the squad.
            verbose: Print results.

        Returns:
            OptimizationResult with optimal squad, XI, captain, etc.
        """
        if self.predictions_df.empty:
            raise RuntimeError("No predictions. Call predict() first.")

        if verbose:
            print("\n═══════════════════════════════════════════════════════")
            print(f"  FPL ENGINE — Optimizer (Gamestate: {gamestate.value})")
            print("═══════════════════════════════════════════════════════")

        self.optimizer.gamestate = gamestate

        constraints = SquadConstraints(
            budget=budget or TOTAL_BUDGET,
            must_include=must_include or [],
            must_exclude=must_exclude or [],
        )

        # Filter to available players
        available = self.predictions_df[
            self.predictions_df["status"].isin(["a", "d", None, ""])
            | self.predictions_df["status"].isna()
        ].copy()

        if verbose:
            print(f"  Available players: {len(available)}")

        result = self.optimizer.optimize_squad(available, constraints, chip)

        if verbose:
            self._print_result(result, chip)

        return result

    def optimize_transfers(
        self,
        current_squad_ids: list[int],
        free_transfers: int = 1,
        bank: int = 0,
        horizon: int = 3,
        verbose: bool = True,
    ) -> list[dict]:
        """Suggest optimal transfers for an existing squad.

        Args:
            current_squad_ids: List of 15 element_ids in current squad.
            free_transfers: Number of free transfers.
            bank: Money in bank (0.1m units).
            horizon: GWs to plan ahead.
            verbose: Print suggestions.

        Returns:
            List of transfer suggestions.
        """
        if self.predictions_df.empty:
            raise RuntimeError("No predictions. Call predict() first.")

        current = self.predictions_df[
            self.predictions_df["element_id"].isin(current_squad_ids)
        ].copy()

        if len(current) < len(current_squad_ids):
            print(f"  ⚠ Found {len(current)}/{len(current_squad_ids)} players in predictions")

        transfers = self.optimizer.optimize_transfers(
            current_squad=current,
            all_players=self.predictions_df,
            free_transfers=free_transfers,
            bank=bank,
            horizon=horizon,
        )

        if verbose and transfers:
            print(f"\n  ── Transfer Suggestions (horizon: {horizon} GWs) ──")
            for i, t in enumerate(transfers):
                hit_str = " [HIT -4]" if t.get("hit") else " [FREE]"
                print(f"  {i+1}. OUT: {t['out_name']:<18} → IN: {t['in_name']:<18} "
                      f"| xP gain: {t['xp_gain']:+.2f}{hit_str}")

        return transfers

    def report(self, result: OptimizationResult | None = None) -> str:
        """Generate a formatted text report of recommendations."""
        if result is None:
            result = self.optimize(verbose=False)

        lines = [
            "╔═══════════════════════════════════════════════════════╗",
            "║           FPL ENGINE — GAMEWEEK REPORT               ║",
            f"║           Gameweek {self.current_gw:>2}                              ║",
            "╚═══════════════════════════════════════════════════════╝",
            "",
            f"  Total xP: {result.total_xp:.1f}",
            f"  Differential Score: {result.differential_score:.1f}",
            f"  Strategy: {self.optimizer.gamestate.value}",
            "",
            "  ── STARTING XI ──",
            f"  {'Player':<20} {'Pos':<5} {'Team':<6} {'Price':<8} {'xP':<8} {'Own%':<6} {'Role':<8}",
            f"  {'─'*20} {'─'*5} {'─'*6} {'─'*8} {'─'*8} {'─'*6} {'─'*8}",
        ]

        for _, p in result.starting_xi.iterrows():
            role = ""
            if p["element_id"] == result.captain_id:
                role = "★ CAPT"
            elif p["element_id"] == result.vice_captain_id:
                role = "VC"
            lines.append(
                f"  {p['name']:<20} {p['position']:<5} "
                f"{str(p.get('team_name', ''))[:5]:<6} "
                f"£{p['price']/10:.1f}m   "
                f"{p['xp']:.2f}    "
                f"{p.get('ownership_pct', 0):.0f}%    "
                f"{role}"
            )

        lines.extend([
            "",
            "  ── BENCH (in order) ──",
        ])
        for _, p in result.bench.iterrows():
            lines.append(
                f"  {p['name']:<20} {p['position']:<5} "
                f"{str(p.get('team_name', ''))[:5]:<6} "
                f"£{p['price']/10:.1f}m   "
                f"{p['xp']:.2f}"
            )

        budget_used = result.squad["price"].sum()
        lines.extend([
            "",
            f"  Budget: £{budget_used/10:.1f}m / £{TOTAL_BUDGET/10:.1f}m "
            f"(£{(TOTAL_BUDGET - budget_used)/10:.1f}m ITB)",
        ])

        report_text = "\n".join(lines)
        return report_text

    # ── Persistence ──────────────────────────────────────────────────────

    def save_models(self) -> None:
        """Save trained models to disk."""
        self.minutes_model.save()
        self.points_model.save()

    def load_models(self) -> None:
        """Load previously trained models from disk."""
        self.minutes_model.load()
        self.points_model.load()

    # ── Internal ─────────────────────────────────────────────────────────

    def _get_latest_features(self) -> pd.DataFrame:
        """Get the most recent feature row per player for prediction."""
        df = self.features_df.copy()

        # Get the latest gameweek per player
        if "round" in df.columns:
            latest_idx = df.groupby("element_id")["round"].idxmax()
            latest = df.loc[latest_idx].copy()
        else:
            latest = df.drop_duplicates(subset="element_id", keep="last").copy()

        return latest

    def _print_result(self, result: OptimizationResult, chip: Chip) -> None:
        """Pretty-print optimization result."""
        print(self.report(result))
        if chip != Chip.NONE:
            print(f"\n  🎯 Active Chip: {chip.value}")
