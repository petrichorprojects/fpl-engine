# FPL Analytics Engine

A Fantasy Premier League analytics engine with a minutes prediction model, ownership-aware optimizer, deadline tracking, and full web/desktop UI.

**77/77 tests passing** · **~8,500 lines** across Python engine, Next.js frontend, and deployment configs.

---

## Never miss a deadline

Missing a deadline costs 20-30 points. No modelling improvement in this repo is
worth a fraction of that, so start here.

```bash
# Write every remaining gameweek deadline to a calendar file, with
# alarms 24h and 2h before each one. Import it once; done for the season.
uv run python -m fpl_engine.deadlines --fresh --ics fpl-deadlines.ics
```

Deadlines are **not** weekly-on-Saturday. They are 90 minutes before the first
kickoff of the round, which moves with TV scheduling, midweek rounds, and
international breaks. The `.ics` tracks the real times from the FPL API.

For the deadline plus a full recommendation in one place:

```bash
uv run python scripts/gameweek_brief.py --check-only    # deadline only, instant
uv run python scripts/gameweek_brief.py --horizon 4     # full brief (~5 min fetch)
```

`gameweek_brief.py` always renders the deadline block, even when the model
cannot run — the reminder must never depend on the modelling path succeeding.
Exit code `2` means "deadline delivered, model unavailable".

### The nag loop (Slack)

A reminder you can ignore is a reminder that fails. `scripts/lineup_nag.py`
posts to Slack on a cadence that tightens as the deadline nears — a 6h heads-up
far out, every 20 minutes inside the final two hours — and the **only** thing
that silences it is proof of action: a screenshot of your lineup posted back
into the channel. No screenshot, it keeps going.

- **Brain:** `fpl_engine/nag.py` — escalation ladder, state machine, screenshot
  detection. Pure and network-free (22 tests).
- **Slack arm:** `fpl_engine/slack_gateway.py` → n8n relay "FPL Lineup Nag
  Relay" (`/webhook/fpl-nag-relay`). The Slack credential lives on the n8n node;
  this repo never holds a token. Same pattern as the morning-briefing bot.

Preview the cadence and tone without posting:

```bash
uv run python scripts/lineup_nag.py --dry-run --simulate-hours 1.5
```

Deploy (Mac Mini, once): fill `deploy/fpl-nag.env` from the example, then load
`deploy/com.philipp.fpl-nag.plist` — it runs the check every 15 minutes. Details
in `deploy/`.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   DATA LAYER                             │
│  FPL API · Understat · Press Conference NLP ·            │
│  Cup Tracker · Fixture Calendar                          │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                 MODEL STACK                               │
│  ① Minutes Model → P(start), P(sub), P(bench)           │
│  ② Points Model  → E[pts|start], E[pts|sub]             │
│  ③ xP = P(start)×E[pts|start] + P(sub)×E[pts|sub]      │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│            OWNERSHIP-AWARE OPTIMIZER                     │
│  MILP with gamestate (LEADING/CHASING/MINI_LEAGUE)      │
│  + Rival counter-optimization                            │
│  + Chip timing + DGW/BGW planning                        │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│  Web UI (Next.js on Vercel) or Desktop (Electron)        │
│  FastAPI backend ← → React dashboard                     │
└─────────────────────────────────────────────────────────┘
```

## Modules

| Module | Lines | What it does |
|---|---|---|
| `client.py` | 209 | FPL API client with caching + rate limiting |
| `features.py` | 415 | 100+ rolling features: availability, form, opponent, market |
| `minutes_model.py` | 240 | Per-position XGBoost → P(start/sub/bench) |
| `points_model.py` | 279 | Per-position XGBoost → E[pts\|start], E[pts\|sub] |
| `optimizer.py` | 456 | Ownership-aware MILP with 4 gamestates |
| `understat.py` | 640 | Scrapes npxG, xGChain, ppda from Understat |
| `pressers.py` | 412 | NLP extraction from manager press conferences |
| `calendar.py` | 844 | DGW/BGW detection + prediction + chip timing |
| `rivals.py` | 474 | Mini-league rival tracking + counter-optimization |
| `backtest.py` | 713 | Walk-forward season replay + strategy comparison |
| `deadlines.py` | 280 | Gameweek deadlines, countdowns, `.ics` export |
| `upcoming.py` | 330 | Prediction frames keyed on the **upcoming** fixture |
| `engine.py` | 520 | Unified orchestrator |
| `api_server.py` | 398 | FastAPI REST backend |

## Quick Start

### Python Engine (CLI)

```bash
# Install
git clone <repo> && cd fpl-engine
uv sync   # or: pip install -e .

# Full pipeline
uv run python run_engine.py

# With options
uv run python run_engine.py --gamestate chasing --chip bench_boost --save-models

# Tests
uv run --with pytest pytest tests/ -v
```

### Web App (Vercel)

```bash
cd web
npm install
npm run dev           # http://localhost:3000

# Deploy
vercel deploy
```

### Desktop App (Electron)

```bash
cd web && npm install && npm run build
cd ../electron && npm install
npx electron .
```

### Docker

```bash
docker compose up     # API on :8000, web on :3000
```

### FastAPI Backend (standalone)

```bash
uv run --with fastapi --with uvicorn uvicorn api_server:app --reload --port 8000
# Swagger docs at http://localhost:8000/docs
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Engine status |
| `GET` | `/predictions` | Player xP predictions (filterable) |
| `POST` | `/optimize` | Squad optimization with gamestate/budget/chip |
| `POST` | `/optimize/transfers` | Multi-week transfer suggestions |
| `POST` | `/backtest` | Strategy comparison backtest |
| `POST` | `/rivals` | Rival intelligence report |
| `GET` | `/calendar` | DGW/BGW calendar + chip timing |

## Operating notes

**The models need in-season data.** Training uses per-fixture history from the
current season only (`element-summary`). Before roughly GW4 there are not enough
rounds for the rolling windows to carry signal, and `gameweek_brief.py` will say
so rather than hand you a confident squad fitted to three matches. The deadline
half works year-round.

**Predictions are keyed on the upcoming fixture.** `engine.predict(horizon=N)`
builds one row per (player, upcoming fixture) via `fpl_engine.upcoming`, so a
double gameweek counts twice and a blank counts zero. Use `xp` for this week's
lineup and `xp_horizon` for transfer planning — they answer different questions.

## Key Design Decisions

1. **Minutes model is the edge** — separate per-position XGBoost classifiers predict P(start/sub/bench). Most FPL bots skip this entirely.

2. **No raw goals in features** — use xG only. Scoring 5 goals from 0.5 xG is luck, not skill.

3. **Ownership-aware optimization** — four gamestates (NEUTRAL, LEADING, CHASING, MINI_LEAGUE) adjust the MILP objective to weight differentials vs template players.

4. **Press conference NLP** — regex/keyword extraction from manager pressers feeds injury/rotation signals into the minutes model.

5. **Chip timing as strategic options** — Bench Boost best in DGWs, Free Hit best in BGWs, scored per-GW by the calendar module.

## License

MIT
