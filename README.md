# FPL Analytics Engine

A world-class Fantasy Premier League analytics engine with a minutes prediction model, ownership-aware optimizer, and full web/desktop UI.

**34/34 tests passing** · **~7,800 lines** across Python engine, Next.js frontend, and deployment configs.

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
| `engine.py` | 410 | Unified orchestrator |
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

## Key Design Decisions

1. **Minutes model is the edge** — separate per-position XGBoost classifiers predict P(start/sub/bench). Most FPL bots skip this entirely.

2. **No raw goals in features** — use xG only. Scoring 5 goals from 0.5 xG is luck, not skill.

3. **Ownership-aware optimization** — four gamestates (NEUTRAL, LEADING, CHASING, MINI_LEAGUE) adjust the MILP objective to weight differentials vs template players.

4. **Press conference NLP** — regex/keyword extraction from manager pressers feeds injury/rotation signals into the minutes model.

5. **Chip timing as strategic options** — Bench Boost best in DGWs, Free Hit best in BGWs, scored per-GW by the calendar module.

## License

MIT
