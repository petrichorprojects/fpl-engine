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
from .config import POSITIONS, SQUAD_SIZE, TOTAL_BUDGET
from .deadlines import DeadlineTracker
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
from .upcoming import build_upcoming_frame


def _attach_predictions(
    scored: pd.DataFrame,
    mins_pred: pd.DataFrame,
    pts_pred: pd.DataFrame,
) -> pd.DataFrame:
    """Join model outputs onto the fixture frame by row index.

    Not by `element_id`: a player with a double gameweek occupies two rows, and
    merging on the player id would produce a cross product that double-counts
    them. Both models preserve the input index for exactly this reason.

    Rows a model skipped (a position with too little training data) keep
    neutral values rather than dropping out of the squad pool entirely.
    """
    out = scored.copy()

    for col, default in (("p_bench", 1.0), ("p_sub", 0.0), ("p_start", 0.0)):
        out[col] = (
            mins_pred[col].reindex(out.index) if col in mins_pred.columns
            else pd.Series(default, index=out.index)
        )
        out[col] = out[col].fillna(default)

    for col, default in (("e_pts_start", 2.0), ("e_pts_sub", 1.0)):
        out[col] = (
            pts_pred[col].reindex(out.index) if col in pts_pred.columns
            else pd.Series(default, index=out.index)
        )
        out[col] = out[col].fillna(default)

    return out


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
    fixture_predictions_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    current_gw: int = 0
    target_gw: int = 0          # the gameweek whose deadline has not yet passed
    horizon_gws: list[int] = field(default_factory=list)
    deadlines: DeadlineTracker | None = None

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

        # The gameweek we are actually picking a team for is the one whose
        # deadline has not yet passed — which is not always `current_gw`, since
        # FPL keeps a gameweek "current" until its last match finishes.
        self.deadlines = DeadlineTracker.from_events(self.client.bootstrap()["events"])
        nxt = self.deadlines.next_deadline()
        self.target_gw = nxt.gameweek if nxt else self.current_gw

        if verbose:
            print(f"  ✓ Current GW: {self.current_gw}")
            print(f"  ✓ Target GW:  {self.target_gw}")
            if nxt:
                print(f"  ⏰ Deadline in {nxt.human_countdown()} "
                      f"({nxt.local().strftime('%a %d %b %H:%M %Z')})")

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

    def predict(self, horizon: int = 1, verbose: bool = True) -> pd.DataFrame:
        """Generate xP predictions for the upcoming gameweek(s).

        Predictions are built against each player's **upcoming** fixtures — the
        opponent, venue and rest days they are about to face — not the fixture
        they last played. See `fpl_engine.upcoming` for why that distinction is
        the difference between fixture signal and fixture noise.

        Args:
            horizon: How many gameweeks to predict, starting at `target_gw`.
                `horizon=1` scores only the upcoming deadline; larger values
                also produce `xp_horizon`, which transfer planning needs.
            verbose: Print progress and a leaderboard.

        Returns:
            One row per player with `xp` (the target gameweek, summed across
            fixtures so a double counts twice), `xp_gw{n}` per gameweek in the
            horizon, `xp_horizon` (the sum), and `n_fixtures` (0 for a blank).
        """
        if not self.minutes_model.is_trained or not self.points_model.is_trained:
            raise RuntimeError("Models not trained. Call train() first.")
        if self.history_df.empty:
            raise RuntimeError("No history data. Call fetch_data() first.")

        if verbose:
            print("\n═══════════════════════════════════════════════════════")
            print("  FPL ENGINE — Predictions")
            print("═══════════════════════════════════════════════════════")

        target = self.target_gw or self.current_gw
        self.horizon_gws = list(range(target, target + max(1, horizon)))

        # Build one row per (player, upcoming fixture).
        upcoming = build_upcoming_frame(
            history_df=self.history_df,
            players_df=self.players_df,
            fixtures_df=self.fixtures_df,
            teams_df=self.teams_df,
            target_gws=self.horizon_gws,
        )

        if upcoming.empty:
            raise RuntimeError(
                f"No scheduled fixtures found for GW {self.horizon_gws}. "
                "The fixture list may be stale — try client.clear_cache()."
            )

        if verbose:
            n_players = upcoming["element_id"].nunique()
            print(f"  Target GW: {target} (horizon: {self.horizon_gws})")
            print(f"  Fixture rows: {len(upcoming)} across {n_players} players")

        # ── Score every fixture row ──────────────────────────────────────
        mins_pred = self.minutes_model.predict(upcoming)
        pts_pred = self.points_model.predict(upcoming)

        # The models return one row per input row in input order, so align by
        # position within position-group rather than merging on element_id —
        # a player with a double gameweek appears twice and a merge would
        # produce a cross product.
        scored = upcoming.copy()
        scored = _attach_predictions(scored, mins_pred, pts_pred)

        scored["xp"] = (
            scored["p_start"] * scored["e_pts_start"]
            + scored["p_sub"] * scored["e_pts_sub"]
        )

        # Hard availability: a suspended or long-term-injured player scores 0,
        # regardless of what their rolling form suggests.
        if "unavailable" in scored.columns:
            scored.loc[scored["unavailable"], ["xp", "p_start", "p_sub"]] = 0.0

        self.fixture_predictions_df = scored

        # ── Aggregate to one row per player ──────────────────────────────
        combined = self._aggregate_fixture_predictions(scored, target)
        self.predictions_df = combined

        if verbose:
            self._print_prediction_leaderboard(combined, target)

        return combined

    def _aggregate_fixture_predictions(
        self, scored: pd.DataFrame, target: int
    ) -> pd.DataFrame:
        """Collapse per-fixture rows into one row per player."""
        # Per-gameweek totals: a double gameweek sums both fixtures.
        per_gw = (
            scored.groupby(["element_id", "target_gw"])
            .agg(
                gw_xp=("xp", "sum"),
                gw_fixtures=("fixture_id", "size"),
                gw_p_start=("p_start", "max"),
                gw_p_sub=("p_sub", "max"),
            )
            .reset_index()
        )

        wide = per_gw.pivot(index="element_id", columns="target_gw", values="gw_xp")
        wide = wide.rename(columns=lambda gw: f"xp_gw{int(gw)}")
        # A blank gameweek means no fixture row at all, which is genuinely 0 xP.
        wide = wide.fillna(0.0)
        wide["xp_horizon"] = wide.sum(axis=1)
        wide = wide.reset_index()

        target_rows = per_gw[per_gw["target_gw"] == target].set_index("element_id")

        base_cols = [
            c for c in ("element_id", "position", "team_id", "name", "full_name",
                        "price", "selected_pct", "status", "chance_next_round",
                        "form", "unavailable")
            if c in scored.columns
        ]
        base = scored[base_cols].drop_duplicates(subset="element_id")

        combined = base.merge(wide, on="element_id", how="left")
        combined["xp"] = combined["element_id"].map(target_rows["gw_xp"]).fillna(0.0)
        combined["n_fixtures"] = (
            combined["element_id"].map(target_rows["gw_fixtures"]).fillna(0).astype(int)
        )
        combined["p_start"] = combined["element_id"].map(target_rows["gw_p_start"]).fillna(0.0)
        combined["p_sub"] = combined["element_id"].map(target_rows["gw_p_sub"]).fillna(0.0)
        combined["xp_horizon"] = combined["xp_horizon"].fillna(0.0)

        # Team names for reporting.
        if not self.players_df.empty and "team_name" in self.players_df.columns:
            combined = combined.merge(
                self.players_df[["element_id", "team_name"]],
                on="element_id", how="left",
            )

        # The optimizer's effective-ownership term expects a percentage.
        combined["ownership_pct"] = combined.get("selected_pct", 0.0)

        return combined.sort_values("xp", ascending=False).reset_index(drop=True)

    def _print_prediction_leaderboard(self, combined: pd.DataFrame, target: int) -> None:
        blanks = int((combined["n_fixtures"] == 0).sum())
        doubles = int((combined["n_fixtures"] > 1).sum())
        print(f"  ✓ {len(combined)} players scored "
              f"({doubles} with a double, {blanks} blanking)")
        print(f"\n  Top 10 by xP (GW {target}):")
        print(f"  {'Rank':<5} {'Player':<20} {'Pos':<5} {'Team':<6} "
              f"{'Price':<7} {'P(Start)':<10} {'GWs':<5} {'xP':<8}")
        print(f"  {'─'*5} {'─'*20} {'─'*5} {'─'*6} {'─'*7} {'─'*10} {'─'*5} {'─'*8}")
        for i, row in combined.head(10).iterrows():
            print(f"  {i+1:<5} {str(row['name']):<20} {row['position']:<5} "
                  f"{str(row.get('team_name', ''))[:5]:<6} "
                  f"£{row['price']/10:.1f}m  "
                  f"{row['p_start']:.2f}      "
                  f"{row['n_fixtures']:<5} "
                  f"{row['xp']:.2f}")

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

        # Filter to selectable players. `unavailable` is resolved from FPL
        # status in the prediction frame; doubtful players stay in the pool
        # because the minutes model already discounts their xP.
        available = self.predictions_df.copy()
        if "unavailable" in available.columns:
            available = available[~available["unavailable"].fillna(False)]
        else:
            available = available[
                available["status"].isin(["a", "d", None, ""])
                | available["status"].isna()
            ]

        # A player with no fixture cannot score. Leaving blanks in the pool lets
        # the optimizer spend budget on a guaranteed zero.
        if "n_fixtures" in available.columns and chip != Chip.FREE_HIT:
            has_fixture = available[available["n_fixtures"] > 0]
            # Only apply if enough players remain to fill a legal squad.
            if len(has_fixture) >= SQUAD_SIZE * 2:
                available = has_fixture

        available = available.copy()

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

        if "xp_horizon" not in self.predictions_df.columns or len(self.horizon_gws) < 2:
            print(f"  ⚠ Predictions cover {len(self.horizon_gws) or 1} GW but the "
                  f"transfer horizon is {horizon}. Call "
                  f"predict(horizon={horizon}) for fixture-aware planning.")

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

        gw = self.target_gw or self.current_gw
        lines = [
            "╔═══════════════════════════════════════════════════════╗",
            "║           FPL ENGINE — GAMEWEEK REPORT                ║",
            f"║           Gameweek {gw:>2}                                 ║",
            "╚═══════════════════════════════════════════════════════╝",
            "",
        ]

        nxt = self.deadlines.next_deadline() if self.deadlines else None
        if nxt:
            lines += [
                f"  ⏰ DEADLINE: {nxt.local().strftime('%a %d %b, %H:%M %Z')} "
                f"— {nxt.human_countdown()} left",
                "",
            ]

        lines += [
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

    def _print_result(self, result: OptimizationResult, chip: Chip) -> None:
        """Pretty-print optimization result."""
        print(self.report(result))
        if chip != Chip.NONE:
            print(f"\n  🎯 Active Chip: {chip.value}")
