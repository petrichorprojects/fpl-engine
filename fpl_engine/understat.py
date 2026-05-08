"""Understat scraping client, FPL↔Understat name mapper, and history enrichment.

Understat embeds JSON data inside HTML script tags as:
    var <varName> = JSON.parse('...')
We extract the raw string with a regex and decode unicode escapes with codecs.
"""

from __future__ import annotations

import codecs
import json
import re
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from .config import DATA_DIR

# ── Constants ─────────────────────────────────────────────────────────────────

UNDERSTAT_BASE = "https://understat.com"

# Regex for each variable Understat embeds in HTML
_JS_VAR_RE = re.compile(
    r"var\s+(\w+)\s*=\s*JSON\.parse\('((?:[^'\\]|\\.)*)'\)",
    re.DOTALL,
)

# FPL team name → Understat URL team name (used by get_team_matches helper)
FPL_TO_UNDERSTAT_TEAM: dict[str, str] = {
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston_Villa",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brighton": "Brighton",
    "Chelsea": "Chelsea",
    "Crystal Palace": "Crystal_Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Ipswich": "Ipswich",
    "Leicester": "Leicester",
    "Liverpool": "Liverpool",
    "Man City": "Manchester_City",
    "Man Utd": "Manchester_United",
    "Newcastle": "Newcastle_United",
    "Nott'm Forest": "Nottingham_Forest",
    "Southampton": "Southampton",
    "Spurs": "Tottenham",
    "West Ham": "West_Ham",
    "Wolves": "Wolverhampton_Wanderers",
}

# Column names added by enrich_with_understat
_ENRICH_COLS = [
    "us_npxG",
    "us_xA",
    "us_xGChain",
    "us_xGBuildup",
    "us_shots",
    "us_key_passes",
    "us_minutes",
    "us_npxG_per90",
    "us_xA_per90",
]


# ── Parsing helpers ───────────────────────────────────────────────────────────

def _extract_var(html: str, var_name: str) -> Any:
    """Extract and decode a ``JSON.parse('...')`` JS variable from Understat HTML.

    Understat escapes all strings with unicode escapes (``\\uXXXX``).  We pull
    the raw escaped string out of the HTML with a regex and then round-trip it
    through ``codecs`` to produce a proper Python str before handing it to
    ``json.loads``.
    """
    # Build a targeted pattern so we only search for the variable we want
    pattern = re.compile(
        rf"var\s+{re.escape(var_name)}\s*=\s*JSON\.parse\('((?:[^'\\]|\\.)*)'\)",
        re.DOTALL,
    )
    m = pattern.search(html)
    if not m:
        raise ValueError(
            f"Variable '{var_name}' not found in Understat HTML "
            f"(page length {len(html)} chars)"
        )
    raw: str = m.group(1)
    # Decode unicode escapes: encode to raw bytes first so codecs can work
    decoded: str = codecs.decode(raw.encode("raw_unicode_escape"), "unicode_escape")
    return json.loads(decoded)


def _float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _int(val: Any, default: int = 0) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


# ── UnderstatClient ───────────────────────────────────────────────────────────

@dataclass
class UnderstatClient:
    """Thin scraping client for understat.com.

    Attributes:
        cache_dir:      Directory for cached HTML responses.
        use_cache:      Read from / write to the cache when True.
        request_delay:  Seconds to sleep before each live request.
    """

    cache_dir: Path = field(
        default_factory=lambda: DATA_DIR / "cache" / "understat"
    )
    use_cache: bool = True
    request_delay: float = 1.0
    _http: httpx.Client = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._http = httpx.Client(
            timeout=30.0,
            follow_redirects=True,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "en-GB,en;q=0.9",
            },
        )

    # ── Low-level ────────────────────────────────────────────────────────

    def _get_html(self, url: str, cache_key: str) -> str:
        """Return the HTML for *url*, hitting the disk cache first if enabled."""
        if self.use_cache:
            cache_file = self.cache_dir / f"{cache_key}.html"
            if cache_file.exists():
                return cache_file.read_text(encoding="utf-8")

        time.sleep(self.request_delay)
        resp = self._http.get(url)
        resp.raise_for_status()
        html = resp.text

        if self.use_cache:
            cache_file = self.cache_dir / f"{cache_key}.html"
            cache_file.write_text(html, encoding="utf-8")

        return html

    def invalidate(self, cache_key: str) -> None:
        """Remove a single cached HTML file."""
        target = self.cache_dir / f"{cache_key}.html"
        if target.exists():
            target.unlink()

    # ── Public API ───────────────────────────────────────────────────────

    def get_league_players(self, season: str = "2024") -> pd.DataFrame:
        """All player season totals for EPL in *season*.

        Returns
        -------
        DataFrame with columns:
            understat_id, player_name, team, npg, npxG, xA, xGChain,
            xGBuildup, shots, key_passes, minutes
        """
        url = f"{UNDERSTAT_BASE}/league/EPL/{season}"
        cache_key = f"league_EPL_{season}"

        try:
            html = self._get_html(url, cache_key)
            raw: list[dict] = _extract_var(html, "playersData")
        except Exception as exc:
            print(f"[UnderstatClient] get_league_players failed: {exc}")
            return pd.DataFrame(
                columns=[
                    "understat_id", "player_name", "team", "npg", "npxG",
                    "xA", "xGChain", "xGBuildup", "shots", "key_passes",
                    "minutes",
                ]
            )

        rows = []
        for p in raw:
            rows.append(
                {
                    "understat_id": _int(p.get("id")),
                    "player_name": str(p.get("player_name", "")),
                    "team": str(p.get("team_title", "")),
                    "npg": _float(p.get("npg")),
                    "npxG": _float(p.get("npxG")),
                    "xA": _float(p.get("xA")),
                    "xGChain": _float(p.get("xGChain")),
                    "xGBuildup": _float(p.get("xGBuildup")),
                    "shots": _int(p.get("shots")),
                    "key_passes": _int(p.get("key_passes")),
                    # Understat calls playing-time "time"
                    "minutes": _int(p.get("time")),
                }
            )

        return pd.DataFrame(rows)

    def get_player_matches(self, understat_id: int) -> pd.DataFrame:
        """Match-by-match stats for a single player.

        Returns
        -------
        DataFrame with columns:
            date, h_team, a_team, npxG, xA, xGChain, xGBuildup,
            shots, key_passes, minutes, position
        """
        url = f"{UNDERSTAT_BASE}/player/{understat_id}"
        cache_key = f"player_{understat_id}"

        try:
            html = self._get_html(url, cache_key)
            raw: list[dict] = _extract_var(html, "matchesData")
        except Exception as exc:
            print(
                f"[UnderstatClient] get_player_matches({understat_id}) "
                f"failed: {exc}"
            )
            return pd.DataFrame(
                columns=[
                    "date", "h_team", "a_team", "npxG", "xA", "xGChain",
                    "xGBuildup", "shots", "key_passes", "minutes", "position",
                ]
            )

        rows = []
        for m in raw:
            # Understat stores xA as the field "a" in matchesData
            xa = _float(m.get("xA", m.get("a")))
            rows.append(
                {
                    "date": str(m.get("date", "")),
                    "h_team": str(m.get("h_team", "")),
                    "a_team": str(m.get("a_team", "")),
                    "npxG": _float(m.get("npxG", m.get("xG"))),
                    "xA": xa,
                    "xGChain": _float(m.get("xGChain")),
                    "xGBuildup": _float(m.get("xGBuildup")),
                    "shots": _int(m.get("shots")),
                    "key_passes": _int(m.get("key_passes")),
                    # Understat calls minutes "time"
                    "minutes": _int(m.get("time")),
                    "position": str(m.get("position", "")),
                }
            )

        df = pd.DataFrame(rows)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

    def get_team_matches(
        self, team_name: str, season: str = "2024"
    ) -> pd.DataFrame:
        """Match-level stats for a team.

        Parameters
        ----------
        team_name:
            Understat URL format, e.g. ``'Manchester_United'``.
            Use ``FPL_TO_UNDERSTAT_TEAM`` to convert FPL team names.
        season:
            Season year, e.g. ``'2024'`` for 2024/25.

        Returns
        -------
        DataFrame with columns:
            date, h_team, a_team, xG, xGA, npxGD,
            ppda_att, ppda_def, deep, deep_allowed
        """
        url = f"{UNDERSTAT_BASE}/team/{team_name}/{season}"
        cache_key = f"team_{team_name}_{season}"

        try:
            html = self._get_html(url, cache_key)
            raw: list[dict] = _extract_var(html, "datesData")
        except Exception as exc:
            print(
                f"[UnderstatClient] get_team_matches({team_name}, {season}) "
                f"failed: {exc}"
            )
            return pd.DataFrame(
                columns=[
                    "date", "h_team", "a_team", "xG", "xGA", "npxGD",
                    "ppda_att", "ppda_def", "deep", "deep_allowed",
                ]
            )

        rows = []
        for m in raw:
            h_info = m.get("h", {}) or {}
            a_info = m.get("a", {}) or {}

            # xG is nested: {"h": "1.23", "a": "0.78"}
            xg_obj = m.get("xG", {}) or {}
            npxg_obj = m.get("npxG", {}) or {}

            h_xg = _float(xg_obj.get("h", 0))
            a_xg = _float(xg_obj.get("a", 0))
            h_npxg = _float(npxg_obj.get("h", 0))
            a_npxg = _float(npxg_obj.get("a", 0))

            # PPDA (pressing intensity) and deep completions
            ppda = m.get("ppda", {}) or {}
            ppda_allowed = m.get("ppda_allowed", {}) or {}

            rows.append(
                {
                    "date": str(m.get("datetime", m.get("date", ""))),
                    "h_team": str(h_info.get("title", "")),
                    "a_team": str(a_info.get("title", "")),
                    "xG": h_xg,
                    "xGA": a_xg,
                    "npxGD": h_npxg - a_npxg,
                    "ppda_att": _float(ppda.get("att")),
                    "ppda_def": _float(ppda.get("def")),
                    "deep": _int(m.get("deep")),
                    "deep_allowed": _int(m.get("deep_allowed")),
                }
            )

        df = pd.DataFrame(rows)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df


# ── FPLUnderstatMapper ────────────────────────────────────────────────────────

@dataclass
class FPLUnderstatMapper:
    """Bidirectional mapping between FPL player codes and Understat IDs.

    Persists to ``DATA_DIR/mappings/fpl_understat_map.json`` as
    ``{ "<fpl_code>": <understat_id>, ... }``.
    """

    mappings_dir: Path = field(
        default_factory=lambda: DATA_DIR / "mappings"
    )
    _map: dict[int, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.mappings_dir.mkdir(parents=True, exist_ok=True)
        self._load()

    @property
    def _map_file(self) -> Path:
        return self.mappings_dir / "fpl_understat_map.json"

    # ── Persistence ──────────────────────────────────────────────────────

    def _load(self) -> None:
        """Populate ``_map`` from disk (no-op if file absent)."""
        if self._map_file.exists():
            raw: dict[str, int] = json.loads(
                self._map_file.read_text(encoding="utf-8")
            )
            self._map = {int(k): int(v) for k, v in raw.items()}

    def _save(self) -> None:
        """Write ``_map`` to disk as JSON."""
        self._map_file.write_text(
            json.dumps({str(k): v for k, v in sorted(self._map.items())}, indent=2),
            encoding="utf-8",
        )

    # ── Fuzzy matching ───────────────────────────────────────────────────

    @staticmethod
    def _ratio(a: str, b: str) -> float:
        return SequenceMatcher(
            None, a.lower().strip(), b.lower().strip()
        ).ratio()

    @staticmethod
    def _normalise_team(name: str) -> str:
        """Lower-case + strip common suffixes so FPL 'Man Utd' ~ 'Manchester United'."""
        replacements = {
            "manchester united": "man utd",
            "manchester city": "man city",
            "tottenham hotspur": "spurs",
            "nottingham forest": "nott'm forest",
            "newcastle united": "newcastle",
            "wolverhampton wanderers": "wolves",
            "west bromwich albion": "west brom",
            "aston villa": "aston villa",
            "crystal palace": "crystal palace",
            "sheffield united": "sheffield utd",
            "queens park rangers": "qpr",
        }
        n = name.lower().strip().replace("_", " ")
        return replacements.get(n, n)

    def build_mapping(
        self,
        fpl_players_df: pd.DataFrame,
        understat_players_df: pd.DataFrame,
        name_threshold: float = 0.72,
        team_threshold: float = 0.50,
    ) -> dict[int, int]:
        """Fuzzy-match FPL players to Understat players and persist.

        Scoring: ``0.7 * name_similarity + 0.3 * team_similarity``.
        A candidate is accepted only if *both* individual thresholds are met.

        Parameters
        ----------
        fpl_players_df:
            From :meth:`FPLClient.get_players_df` — requires columns
            ``fpl_code``, ``full_name`` (or ``name``), ``team_name``.
        understat_players_df:
            From :meth:`UnderstatClient.get_league_players` — requires
            columns ``understat_id``, ``player_name``, ``team``.
        name_threshold:
            Minimum name ratio (default 0.72).
        team_threshold:
            Minimum team ratio (default 0.50).

        Returns
        -------
        The full ``{fpl_code: understat_id}`` mapping dict.
        """
        us_records = understat_players_df.to_dict("records")
        # Pre-normalise Understat side once
        us_records_norm = [
            {
                "understat_id": int(r["understat_id"]),
                "player_name": str(r.get("player_name", "")).lower().strip(),
                "team": self._normalise_team(str(r.get("team", ""))),
            }
            for r in us_records
        ]

        for _, fpl_row in fpl_players_df.iterrows():
            fpl_code = int(fpl_row["fpl_code"])
            # Use full_name when available, fall back to web_name
            fpl_name = str(
                fpl_row.get("full_name") or fpl_row.get("name", "")
            ).lower().strip()
            fpl_team = self._normalise_team(str(fpl_row.get("team_name", "")))

            best_score = -1.0
            best_id: int | None = None

            for us in us_records_norm:
                name_score = self._ratio(fpl_name, us["player_name"])
                if name_score < name_threshold:
                    continue  # short-circuit – most players won't match

                team_score = self._ratio(fpl_team, us["team"])
                if team_score < team_threshold:
                    continue

                combined = name_score * 0.7 + team_score * 0.3
                if combined > best_score:
                    best_score = combined
                    best_id = us["understat_id"]

            if best_id is not None:
                self._map[fpl_code] = best_id

        self._save()
        return dict(self._map)

    # ── Lookup & manual overrides ────────────────────────────────────────

    def get_understat_id(self, fpl_code: int) -> int | None:
        """Return the Understat player ID for *fpl_code*, or ``None``."""
        return self._map.get(int(fpl_code))

    def add_mapping(self, fpl_code: int, understat_id: int) -> None:
        """Manually add or override a single mapping and persist."""
        self._map[int(fpl_code)] = int(understat_id)
        self._save()

    def remove_mapping(self, fpl_code: int) -> None:
        """Delete a mapping entry and persist."""
        self._map.pop(int(fpl_code), None)
        self._save()

    def __len__(self) -> int:  # noqa: D105
        return len(self._map)

    def __repr__(self) -> str:  # noqa: D105
        return f"FPLUnderstatMapper({len(self)} mappings, file={self._map_file})"


# ── enrich_with_understat ─────────────────────────────────────────────────────

def enrich_with_understat(
    history_df: pd.DataFrame,
    players_df: pd.DataFrame,
    client: UnderstatClient | None = None,
    mapper: FPLUnderstatMapper | None = None,
) -> pd.DataFrame:
    """Attach per-fixture Understat xG metrics to an FPL history DataFrame.

    For each (player, fixture) row the function looks up the player's
    Understat match by calendar date and merges the xG stats.  Rows with no
    Understat counterpart are left as ``NaN`` — the function never raises.

    Added columns
    -------------
    us_npxG, us_xA, us_xGChain, us_xGBuildup, us_shots, us_key_passes,
    us_minutes      — raw per-fixture values
    us_npxG_per90   — npxG × 90 / max(minutes, 1)
    us_xA_per90     — xA  × 90 / max(minutes, 1)

    Parameters
    ----------
    history_df:
        Per-fixture FPL history from :meth:`FPLClient.get_all_player_histories`.
        Must contain ``element_id`` and ``kickoff_time``.
    players_df:
        Player master table from :meth:`FPLClient.get_players_df`.
        Must contain ``element_id`` and ``fpl_code``.
    client:
        :class:`UnderstatClient` instance; created with defaults when ``None``.
    mapper:
        :class:`FPLUnderstatMapper` instance; created with defaults when ``None``.

    Returns
    -------
    Copy of *history_df* with the nine Understat columns appended.
    """
    if client is None:
        client = UnderstatClient()
    if mapper is None:
        mapper = FPLUnderstatMapper()

    df = history_df.copy()

    # Initialise all output columns with NaN
    for col in _ENRICH_COLS:
        df[col] = float("nan")

    # ── Guard: need kickoff_time to match on date ─────────────────────────
    if "kickoff_time" not in df.columns:
        return df

    df["_ko_date"] = pd.to_datetime(
        df["kickoff_time"], errors="coerce", utc=True
    ).dt.date

    # ── Attach fpl_code if not already present ────────────────────────────
    if "fpl_code" not in df.columns:
        if "element_id" in df.columns and "fpl_code" in players_df.columns:
            code_lookup: dict[int, int] = (
                players_df.set_index("element_id")["fpl_code"]
                .dropna()
                .astype(int)
                .to_dict()
            )
            df["fpl_code"] = df["element_id"].map(code_lookup)
        else:
            # Cannot resolve codes → return blank enrichment
            df.drop(columns=["_ko_date"], errors="ignore", inplace=True)
            return df

    # ── Player-level loop ─────────────────────────────────────────────────
    #    Group once so we iterate only over distinct codes, not every row.
    grouped = df.groupby("fpl_code", sort=False)

    for fpl_code, player_idx in grouped.groups.items():
        try:
            us_id = mapper.get_understat_id(int(fpl_code))
        except (TypeError, ValueError):
            continue
        if us_id is None:
            continue

        # Fetch (cached) Understat match history
        try:
            us_matches = client.get_player_matches(us_id)
        except Exception:
            continue

        if us_matches.empty or "date" not in us_matches.columns:
            continue

        us_matches = us_matches.copy()
        us_matches["_date"] = pd.to_datetime(
            us_matches["date"], errors="coerce", utc=True
        ).dt.date

        # Build date → row index for O(1) lookup
        # If duplicate dates exist (DGW / cup) keep last occurrence
        us_by_date: dict[Any, pd.Series] = {
            row["_date"]: row
            for _, row in us_matches.dropna(subset=["_date"]).iterrows()
        }

        # Iterate over this player's FPL fixture rows
        for idx in player_idx:
            ko_date = df.at[idx, "_ko_date"]
            if pd.isna(ko_date) or ko_date not in us_by_date:
                continue

            us_row = us_by_date[ko_date]
            mins = _float(us_row.get("minutes", 0))
            safe_mins = max(mins, 1.0)

            npxg = _float(us_row.get("npxG", 0))
            xa   = _float(us_row.get("xA",   0))

            df.at[idx, "us_npxG"]       = npxg
            df.at[idx, "us_xA"]         = xa
            df.at[idx, "us_xGChain"]    = _float(us_row.get("xGChain",   0))
            df.at[idx, "us_xGBuildup"]  = _float(us_row.get("xGBuildup", 0))
            df.at[idx, "us_shots"]      = _float(us_row.get("shots",      0))
            df.at[idx, "us_key_passes"] = _float(us_row.get("key_passes", 0))
            df.at[idx, "us_minutes"]    = mins
            df.at[idx, "us_npxG_per90"] = npxg / safe_mins * 90.0
            df.at[idx, "us_xA_per90"]   = xa   / safe_mins * 90.0

    df.drop(columns=["_ko_date"], errors="ignore", inplace=True)
    return df
