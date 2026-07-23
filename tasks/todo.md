# FPL Engine — Fix Pack: Deadlines + Fixture Skew

**Branch:** `fix/fixture-skew-and-deadlines`
**Opened:** 2026-07-23

## Goal

Two problems, one repo:

1. **Phil misses lineup deadlines.** FPL deadlines move every week (Sat 10:30 GMT is
   only the common case — midweek rounds, TV picks, and international breaks shift
   them). There is no reminder anywhere in this system. `deadline_time` is present in
   the FPL bootstrap payload and is currently read by *nothing*.

2. **The engine has never been run and, if run, would predict the wrong fixture.**
   `data/cache/` is empty, `models/` is empty, there is no `predictions.csv`. And
   `FPLEngine._get_latest_features()` takes each player's **last played** row as the
   prediction row — so opponent strength, home/away, and rest days all describe the
   match that already happened, not the one being predicted. Training uses the true
   contemporaneous fixture. That is a train/serve skew that inverts fixture signal.

## Acceptance criteria

- [x] AC1 — `fpl_engine/deadlines.py` exposes next deadline, all deadlines, and
      time-remaining, sourced from `bootstrap["events"]`.
- [x] AC2 — A generated `.ics` file contains every remaining GW deadline with -24h
      and -2h alarms, importable once into any calendar.
- [x] AC3 — Prediction rows carry the **upcoming** fixture's opponent, venue, and
      rest days — not the last played fixture's. Verified by a test that asserts the
      prediction frame's `opponent_team` matches the fixture list for the target GW.
- [x] AC4 — Double gameweeks produce one prediction row per fixture; blank gameweeks
      produce zero rows for that team's players.
- [x] AC5 — GKP log-transform shift is persisted at train time and reused at predict
      time (currently recomputed from a different population → biased GKP xP).
- [x] AC6 — Players with FPL status `i`/`s`/`u`/`n` get xP forced to 0 rather than
      relying on the model to infer it, and `chance_next_round` nulls no longer fill
      to 100 for injured players.
- [x] AC7 — Multi-GW transfer horizon sums real per-GW xP instead of multiplying a
      single GW's xP by the horizon.
- [x] AC8 — `scripts/gameweek_brief.py` produces a single deadline-aware brief:
      countdown, XI, captain, bench order, transfers, chip note.
- [x] AC9 — Existing test suite still passes; new tests cover AC3, AC4, AC5, AC7.

## Working memory

- Repo is on an exFAT volume (`/Volumes/T9`). `core.fileMode=false` set; AppleDouble
  `._*` files added to `.gitignore`.
- `understat.py` and `pressers.py` are **orphaned** — imported by no other module and
  by neither `engine.py` nor `api_server.py`. ~1,050 lines of dead code. Out of scope
  for this pack; logged as follow-up.
- `calendar.py` (DGW/BGW) *is* imported by `api_server.py` but not by `engine.py`, so
  the CLI path has no DGW awareness at all. WS2 fixes this at the prediction layer.
- `ownership_pct` means two different things: raw manager count in `features.py`
  (`selected`), percentage in `engine.py` (`selected_pct`). The optimizer's EO term
  consumes the percentage version, so the live path is correct — but the feature-level
  column is wrong and unused. Logged as follow-up.

## Results

**Tests: 34 → 77 passing.** `uv run python -m pytest tests/ -q`

### What changed

| File | Change |
|---|---|
| `fpl_engine/deadlines.py` | New. Deadline parsing, countdowns, urgency window, RFC 5545 `.ics` export with folded lines. CLI: `python -m fpl_engine.deadlines --ics out.ics`. |
| `fpl_engine/upcoming.py` | New. `build_upcoming_frame()` — prediction rows keyed on (player, upcoming fixture) instead of last-played fixture. |
| `fpl_engine/engine.py` | `predict(horizon=N)` rewritten onto the upcoming frame. Added `target_gw` (deadline-derived, not `is_current`), per-GW `xp_gw{n}` columns, `xp_horizon`, `n_fixtures`. Deleted `_get_latest_features()`. `optimize()` now drops unavailable and blanking players. `report()` leads with the deadline. |
| `fpl_engine/points_model.py` | GKP log-transform offset persisted at train time (`log_shifts`), saved/loaded, reused at predict. |
| `fpl_engine/minutes_model.py`, `points_model.py` | `predict()` preserves the input row index so double-gameweek rows can be aligned without a cross-product merge. |
| `fpl_engine/optimizer.py` | `optimize_transfers()` uses `xp_horizon` when present instead of `xp × horizon`. |
| `scripts/gameweek_brief.py` | New. Deadline-first brief; renders the deadline even when the model fails. |
| `tests/test_deadlines.py`, `test_upcoming.py`, `test_engine_integration.py` | New. 43 tests. |

### Verification

- `pytest tests/ -q` → **77 passed** (was 34; all 34 originals still pass).
- Live FPL API: `python -m fpl_engine.deadlines --fresh` returns GW1 = Fri 21 Aug
  13:30 EDT, 38 deadlines exported, longest `.ics` line 74 octets (limit 75).
- `scripts/gameweek_brief.py --check-only` renders against the live API.
- End-to-end model path verified on synthetic data in `test_engine_integration.py`
  — a live run is **not possible yet**: the current season has 0 rows of
  per-fixture history (checked against the API, 555 players, 0 history rows).

### Known limitation surfaced during this work

The engine trains on current-season per-fixture history only. Pre-season and
through roughly GW3 there is nothing to train on, so the model half of this repo
is dormant until the 2026/27 season is a few rounds old. `history_past` (5 prior
seasons, season-aggregate) is available from the API and unused — a cold-start
prior built from it would close the gap. Not attempted here.

## Follow-ups (out of scope, logged not fixed)

- `understat.py` + `pressers.py` are dead code — wire in or delete.
- Effective ownership ignores captaincy. True EO = ownership% + captaincy%. The
  differential maths is systematically wrong for premium captains.
- Captain selection maximises the mean. Captaincy doubles the draw, so it should
  maximise the ceiling (variance is an asset on a doubled pick).
- Starting-XI MILP ignores autosub value — bench players have option value the
  objective never prices.
