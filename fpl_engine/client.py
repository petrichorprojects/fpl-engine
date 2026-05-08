"""FPL API client — fetches and caches data from the official endpoints."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from .config import DATA_DIR, FPL_BASE_URL, POSITION_MAP, REQUEST_DELAY


@dataclass
class FPLClient:
    """Thin async-capable client for the FPL REST API."""

    base_url: str = FPL_BASE_URL
    cache_dir: Path = field(default_factory=lambda: DATA_DIR / "cache")
    use_cache: bool = True
    _http: httpx.Client = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._http = httpx.Client(
            base_url=self.base_url,
            timeout=30.0,
            headers={"User-Agent": "FPL-Engine/0.1"},
        )

    # ── Low-level ────────────────────────────────────────────────────────
    def _get(self, path: str, cache_key: str | None = None) -> dict | list:
        if self.use_cache and cache_key:
            cache_file = self.cache_dir / f"{cache_key}.json"
            if cache_file.exists():
                return json.loads(cache_file.read_text())

        time.sleep(REQUEST_DELAY)
        resp = self._http.get(path)
        resp.raise_for_status()
        data = resp.json()

        if cache_key:
            cache_file = self.cache_dir / f"{cache_key}.json"
            cache_file.write_text(json.dumps(data))

        return data

    # ── High-level endpoints ─────────────────────────────────────────────
    def bootstrap(self) -> dict[str, Any]:
        """Master data: players, teams, gameweeks, settings."""
        return self._get("/bootstrap-static/", cache_key="bootstrap")

    def fixtures(self) -> list[dict]:
        """All fixtures for the current season."""
        return self._get("/fixtures/", cache_key="fixtures")

    def player_history(self, player_id: int) -> dict:
        """Detailed per-fixture history for a single player."""
        return self._get(
            f"/element-summary/{player_id}/",
            cache_key=f"player_{player_id}",
        )

    def live_gameweek(self, gw: int) -> dict:
        """Live stats for a given gameweek."""
        return self._get(f"/event/{gw}/live/", cache_key=f"live_gw_{gw}")

    def manager_picks(self, manager_id: int, gw: int) -> dict:
        """A manager's squad for a given gameweek."""
        return self._get(
            f"/entry/{manager_id}/event/{gw}/picks/",
            cache_key=f"manager_{manager_id}_gw_{gw}",
        )

    # ── DataFrame builders ───────────────────────────────────────────────
    def get_players_df(self) -> pd.DataFrame:
        """All players as a DataFrame with clean column names."""
        boot = self.bootstrap()
        teams = {t["id"]: t for t in boot["teams"]}
        elements = boot["elements"]

        rows = []
        for e in elements:
            team = teams[e["team"]]
            rows.append({
                "element_id": e["id"],
                "fpl_code": e["code"],
                "name": e["web_name"],
                "full_name": f"{e['first_name']} {e['second_name']}",
                "position": POSITION_MAP.get(e["element_type"], "UNK"),
                "team_id": e["team"],
                "team_name": team["name"],
                "team_short": team["short_name"],
                "price": e["now_cost"],  # in 0.1m units
                "total_points": e["total_points"],
                "minutes": e["minutes"],
                "goals_scored": e["goals_scored"],
                "assists": e["assists"],
                "clean_sheets": e["clean_sheets"],
                "goals_conceded": e["goals_conceded"],
                "saves": e.get("saves", 0),
                "bonus": e["bonus"],
                "bps": e["bps"],
                "form": float(e["form"]) if e["form"] else 0.0,
                "points_per_game": float(e["points_per_game"]) if e["points_per_game"] else 0.0,
                "selected_pct": float(e["selected_by_percent"]) if e["selected_by_percent"] else 0.0,
                "transfers_in_event": e["transfers_in_event"],
                "transfers_out_event": e["transfers_out_event"],
                "ict_index": float(e["ict_index"]) if e["ict_index"] else 0.0,
                "influence": float(e["influence"]) if e["influence"] else 0.0,
                "creativity": float(e["creativity"]) if e["creativity"] else 0.0,
                "threat": float(e["threat"]) if e["threat"] else 0.0,
                "expected_goals": float(e.get("expected_goals", 0) or 0),
                "expected_assists": float(e.get("expected_assists", 0) or 0),
                "expected_goal_involvements": float(e.get("expected_goal_involvements", 0) or 0),
                "expected_goals_conceded": float(e.get("expected_goals_conceded", 0) or 0),
                "status": e["status"],  # 'a', 'd', 'i', 'n', 's', 'u'
                "chance_next_round": e.get("chance_of_playing_next_round"),
                "news": e.get("news", ""),
                "news_added": e.get("news_added", ""),
                "starts": e.get("starts", 0),
            })
        return pd.DataFrame(rows)

    def get_fixtures_df(self) -> pd.DataFrame:
        """All fixtures as a DataFrame."""
        fixtures = self.fixtures()
        rows = []
        for f in fixtures:
            rows.append({
                "fixture_id": f["id"],
                "gameweek": f.get("event"),
                "home_team_id": f["team_h"],
                "away_team_id": f["team_a"],
                "home_score": f.get("team_h_score"),
                "away_score": f.get("team_a_score"),
                "finished": f["finished"],
                "kickoff_time": f.get("kickoff_time"),
                "home_difficulty": f.get("team_h_difficulty"),
                "away_difficulty": f.get("team_a_difficulty"),
            })
        return pd.DataFrame(rows)

    def get_player_fixture_history(self, player_id: int) -> pd.DataFrame:
        """Per-fixture stats for a player (current season)."""
        data = self.player_history(player_id)
        if not data.get("history"):
            return pd.DataFrame()
        df = pd.DataFrame(data["history"])
        df["element_id"] = player_id
        return df

    def get_all_player_histories(self, player_ids: list[int] | None = None,
                                  progress: bool = True) -> pd.DataFrame:
        """Fetch fixture-level history for all (or specified) players.

        WARNING: ~800 API calls with rate limiting. Takes ~5 minutes.
        """
        if player_ids is None:
            players = self.get_players_df()
            player_ids = players["element_id"].tolist()

        frames = []
        total = len(player_ids)
        for i, pid in enumerate(player_ids):
            if progress and (i + 1) % 50 == 0:
                print(f"  Fetched {i + 1}/{total} player histories...")
            try:
                df = self.get_player_fixture_history(pid)
                if not df.empty:
                    frames.append(df)
            except Exception as e:
                print(f"  Warning: failed for player {pid}: {e}")

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def get_gameweeks_df(self) -> pd.DataFrame:
        """Gameweek metadata."""
        boot = self.bootstrap()
        return pd.DataFrame(boot["events"])

    def get_teams_df(self) -> pd.DataFrame:
        """Team metadata with strength ratings."""
        boot = self.bootstrap()
        return pd.DataFrame(boot["teams"])

    def current_gameweek(self) -> int:
        """Return the current (or next upcoming) gameweek number."""
        gws = self.get_gameweeks_df()
        current = gws[gws["is_current"] == True]
        if not current.empty:
            return int(current.iloc[0]["id"])
        upcoming = gws[gws["is_next"] == True]
        if not upcoming.empty:
            return int(upcoming.iloc[0]["id"])
        finished = gws[gws["finished"] == True]
        return int(finished.iloc[-1]["id"]) + 1 if not finished.empty else 1

    def clear_cache(self) -> None:
        """Remove all cached JSON files."""
        for f in self.cache_dir.glob("*.json"):
            f.unlink()
        print("Cache cleared.")
