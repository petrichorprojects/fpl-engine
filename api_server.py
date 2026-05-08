"""FastAPI server — bridges the Next.js frontend to the Python FPL engine.

Run with:
    uv run uvicorn api_server:app --reload --port 8000

Or in Docker:
    docker compose up

Endpoints:
    GET  /predictions         → top player xP predictions
    POST /optimize            → squad optimization
    POST /optimize/transfers  → transfer suggestions
    POST /backtest            → strategy backtest
    POST /rivals              → rival intelligence
    GET  /calendar            → DGW/BGW calendar
    GET  /health              → health check
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any, Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from fpl_engine.client import FPLClient
from fpl_engine.engine import FPLEngine
from fpl_engine.optimizer import Chip, GameState
from fpl_engine.calendar import FixtureCalendar
from fpl_engine.rivals import RivalTracker


# ── Global engine instance (lazy-loaded) ─────────────────────────────────────

_engine: FPLEngine | None = None


def get_engine() -> FPLEngine:
    global _engine
    if _engine is None:
        _engine = FPLEngine()
        try:
            _engine.load_models()
            print("  ✓ Loaded saved models")
        except Exception:
            print("  ℹ No saved models found — will train on first request")
    return _engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: fast boot — no data fetching. Engine loads lazily on first request."""
    print("🚀 FPL Engine API starting up...")
    print("  ℹ Engine will load data on first request")
    yield
    print("FPL Engine API shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="FPL Analytics Engine API",
    description="World-class Fantasy Premier League analytics.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ───────────────────────────────────────────────────────────

class OptimizeRequest(BaseModel):
    budget: int = 1000
    gamestate: str = "neutral"
    chip: str = "none"
    must_include: list[int] = []
    must_exclude: list[int] = []


class TransferRequest(BaseModel):
    squad_ids: list[int]
    free_transfers: int = 1
    bank: int = 0
    horizon: int = 3


class BacktestRequest(BaseModel):
    strategy: str = "top_form"
    start_gw: int = 5
    end_gw: int = 38


class RivalsRequest(BaseModel):
    manager_id: int
    league_id: int
    gameweek: int


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_predictions() -> None:
    engine = get_engine()
    if engine.predictions_df.empty:
        if not engine.minutes_model.is_trained:
            if engine.features_df.empty:
                engine.fetch_data(fetch_histories=True, verbose=False)
                engine.build_features(verbose=False)
            engine.train(verbose=False)
        engine.predict(verbose=False)


def _df_to_records(df: pd.DataFrame) -> list[dict]:
    return df.fillna(0).to_dict(orient="records")


def _parse_gamestate(gs: str) -> GameState:
    try:
        return GameState(gs)
    except ValueError:
        return GameState.NEUTRAL


def _parse_chip(c: str) -> Chip:
    try:
        return Chip(c)
    except ValueError:
        return Chip.NONE


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    engine = get_engine()
    return {
        "status": "ok",
        "models_trained": engine.minutes_model.is_trained,
        "predictions_available": not engine.predictions_df.empty,
        "current_gw": engine.current_gw,
        "n_players": len(engine.players_df),
    }


@app.get("/predictions")
async def predictions(limit: int = Query(50, ge=1, le=700)) -> list[dict]:
    """Top xP predictions for the current gameweek."""
    _ensure_predictions()
    engine = get_engine()
    cols = [
        "element_id", "name", "position", "team_name", "price",
        "xp", "p_start", "p_sub", "p_bench",
        "e_pts_start", "e_pts_sub", "ownership_pct", "status",
    ]
    available_cols = [c for c in cols if c in engine.predictions_df.columns]
    return _df_to_records(engine.predictions_df[available_cols].head(limit))


@app.post("/optimize")
async def optimize(req: OptimizeRequest) -> dict:
    """Optimize a fresh squad."""
    _ensure_predictions()
    engine = get_engine()
    gamestate = _parse_gamestate(req.gamestate)
    chip = _parse_chip(req.chip)
    engine.optimizer.gamestate = gamestate

    try:
        result = engine.optimize(
            budget=req.budget,
            gamestate=gamestate,
            chip=chip,
            must_include=req.must_include,
            must_exclude=req.must_exclude,
            verbose=False,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Build response
    squad_records = _df_to_records(result.squad)
    xi_ids = set(result.starting_xi["element_id"].tolist())
    for rec in squad_records:
        rec["in_xi"] = rec["element_id"] in xi_ids
        rec["is_captain"] = rec["element_id"] == result.captain_id
        rec["is_vice_captain"] = rec["element_id"] == result.vice_captain_id

    bench_df = result.bench.reset_index(drop=True)
    bench_df["bench_order"] = bench_df.index

    return {
        "squad": squad_records,
        "starting_xi": _df_to_records(result.starting_xi),
        "bench": _df_to_records(bench_df),
        "captain_id": result.captain_id,
        "vice_captain_id": result.vice_captain_id,
        "total_xp": round(result.total_xp, 2),
        "differential_score": round(result.differential_score, 2),
        "budget_used": int(result.squad["price"].sum()),
        "bank": req.budget - int(result.squad["price"].sum()),
    }


@app.post("/optimize/transfers")
async def transfers(req: TransferRequest) -> list[dict]:
    """Suggest optimal transfers for an existing squad."""
    _ensure_predictions()
    engine = get_engine()

    try:
        suggestions = engine.optimize_transfers(
            current_squad_ids=req.squad_ids,
            free_transfers=req.free_transfers,
            bank=req.bank,
            horizon=req.horizon,
            verbose=False,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    return suggestions


@app.post("/backtest")
async def backtest(req: BacktestRequest) -> dict:
    """Run a walk-forward backtest for a given strategy."""
    engine = get_engine()
    if engine.history_df.empty:
        raise HTTPException(status_code=503, detail="No history data loaded")

    from fpl_engine.backtest import (
        Backtester,
        TopFormStrategy,
        HighOwnershipStrategy,
        RandomStrategy,
        EngineStrategy,
    )

    strategy_map = {
        "engine": EngineStrategy(),
        "top_form": TopFormStrategy(),
        "high_ownership": HighOwnershipStrategy(),
        "random": RandomStrategy(seed=42),
    }
    strategy = strategy_map.get(req.strategy, TopFormStrategy())
    strategy.name = req.strategy

    bt = Backtester()
    try:
        result = bt.simulate_season(
            engine.history_df,
            engine.players_df,
            engine.fixtures_df,
            engine.teams_df,
            strategy,
            start_gw=req.start_gw,
            end_gw=req.end_gw,
            verbose=False,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    gw_rows = []
    for i, r in enumerate(result.gw_results):
        gw_rows.append({
            "gameweek": r.gameweek,
            "points_scored": r.points_scored,
            "cumulative_points": result.cumulative_points_series[i],
            "spearman_r": round(r.spearman_r, 3) if r.spearman_r == r.spearman_r else None,
            "minutes_accuracy": round(r.minutes_accuracy, 3) if r.minutes_accuracy == r.minutes_accuracy else None,
            "autosubs_count": len(r.autosubs_made),
            "captain_id": r.captain_id,
            "captain_points": r.captain_points,
        })

    return {
        "strategy": result.strategy_name,
        "total_points": result.total_points,
        "avg_gw_points": round(result.avg_gw_points, 2),
        "avg_spearman": round(result.avg_spearman, 3) if result.avg_spearman == result.avg_spearman else None,
        "gw_results": gw_rows,
    }


@app.post("/rivals")
async def rivals(req: RivalsRequest) -> dict:
    """Get rival intelligence report."""
    _ensure_predictions()
    engine = get_engine()

    tracker = RivalTracker(client=engine.client, league_id=req.league_id)

    try:
        standings = tracker.fetch_league_standings()
        rival_squads = tracker.fetch_top_rivals(
            gameweek=req.gameweek,
            n=5,
            exclude_manager_id=req.manager_id,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Rival fetch failed: {e}")

    rival_template = tracker.compute_rival_template(rival_squads)

    my_pts_row = standings[standings["manager_id"] == req.manager_id]
    my_pts = int(my_pts_row.iloc[0]["total_points"]) if not my_pts_row.empty else 0

    rival_ids = [r["manager_id"] for r in rival_squads]
    rival_pts = standings[standings["manager_id"].isin(rival_ids)]["total_points"].tolist()
    gws_left = max(0, 38 - req.gameweek)

    gap = tracker.compute_points_gap(my_pts, rival_pts, gws_left)
    differentials = tracker.compute_differential_opportunities(
        [], rival_template, engine.predictions_df
    )
    risks = tracker.compute_risk_players([], rival_template, engine.predictions_df)

    return {
        "my_points": my_pts,
        "gap_to_leader": gap["gap_to_leader"],
        "avg_gap": gap["avg_gap"],
        "strategy": gap["strategy"],
        "suggested_gamestate": gap["suggested_gamestate"].value,
        "gws_remaining": gws_left,
        "rivals": [
            {"name": r["name"], "total_points": r["total_points"], "rank": r["rank"]}
            for r in rival_squads
        ],
        "differentials": _df_to_records(differentials.head(5)),
        "risk_players": _df_to_records(risks.head(5)),
    }


@app.get("/calendar")
async def calendar() -> dict:
    """DGW/BGW calendar and chip timing scores."""
    engine = get_engine()
    if engine.fixtures_df.empty:
        raise HTTPException(status_code=503, detail="No fixture data loaded")

    cal = FixtureCalendar(engine.fixtures_df, engine.teams_df)
    doubles = {str(k): v for k, v in cal.get_doubles().items()}
    blanks = {str(k): v for k, v in cal.get_blanks().items()}

    chip_timing = cal.suggest_chip_timing(
        available_chips=["bench_boost", "free_hit", "triple_captain", "wildcard"],
        current_gw=engine.current_gw or 1,
        remaining_gameweeks=38 - (engine.current_gw or 1),
    )
    # Serialise keys
    chip_timing_str = {
        chip: {str(gw): score for gw, score in scores.items()}
        for chip, scores in chip_timing.items()
    }

    future_events = cal.predict_future_dgw_bgw(engine.current_gw or 1)
    return {
        "doubles": doubles,
        "blanks": blanks,
        "chip_timing": chip_timing_str,
        "predicted_events": [
            {
                "gameweek": e.gameweek,
                "event_type": e.event_type,
                "teams": e.teams,
                "confidence": round(e.confidence, 2),
                "reason": e.reason,
            }
            for e in future_events[:10]
        ],
    }


if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
