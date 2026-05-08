"""
FPL Calendar analytics
======================
Blank / double GW detection, cup tracking, event prediction and chip timing.

Depends on:
    config.DATA_DIR
    FPLClient.get_fixtures_df()   → fixtures_df
    FPLClient.get_teams_df()      → teams_df
    FPLClient.get_players_df()    → players_df
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .config import DATA_DIR

# ─────────────────────────────────────────────────────────────────────────────
# Calendar knowledge constants  (approximate – updated as fixtures are released)
# ─────────────────────────────────────────────────────────────────────────────

# GWs where a UCL/UEL midweek match is typically scheduled.
# Managers rotate → rotation risk for players at European clubs.
_EUROPEAN_GWS: frozenset[int] = frozenset({
    2, 3, 4, 5, 6, 7, 9, 10, 11, 16, 17, 24, 25, 26, 28, 29,
})

# GWs overlapping with FA Cup rounds (3rd–semi-final range, varies yearly).
_FA_CUP_GWS: frozenset[int] = frozenset({22, 23, 26, 27, 31, 32, 35})

# EFL (League) Cup GWs.
_LEAGUE_CUP_GWS: frozenset[int] = frozenset({3, 4, 8, 15, 22, 26})

# Historical DGW / BGW peak zones.
_DGW_PEAK_GWS: range = range(28, 38)   # GW28-37
_BGW_PEAK_GWS: range = range(25, 31)   # GW25-30

# Default difficulty when fixture data is missing.
_DEFAULT_DIFF: float = 3.0

# Shared path for cup-tracker persistence.
_CUP_TRACKER_PATH: Path = DATA_DIR / "cache" / "cup_tracker.json"

# Competition name → CupTracker attribute name
_COMP_TO_ATTR: dict[str, str] = {
    "fa_cup":     "teams_in_fa_cup",
    "league_cup": "teams_in_league_cup",
    "ucl":        "teams_in_ucl",
    "uel":        "teams_in_uel",
}


# =============================================================================
# 1.  PredictedEvent
# =============================================================================

@dataclass
class PredictedEvent:
    """
    A predicted blank or double gameweek for a set of teams.

    Attributes
    ----------
    gameweek  : int
        The FPL gameweek number this event is predicted to fall in.
    teams     : list[int]
        Team IDs expected to be affected (empty list = unknown yet).
    event_type: 'DGW' | 'BGW'
        Double gameweek or blank gameweek.
    confidence: float in [0, 1]
        1.0 = confirmed from fixture list; <1.0 = rule-based estimate.
    notes     : str
        Human-readable explanation of why this event was flagged.
    """

    gameweek:   int
    teams:      list[int]
    event_type: Literal["DGW", "BGW"]
    confidence: float
    notes:      str = ""

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence must be in [0, 1]; got {self.confidence!r}"
            )
        if self.event_type not in ("DGW", "BGW"):
            raise ValueError(
                f"event_type must be 'DGW' or 'BGW'; got {self.event_type!r}"
            )


# =============================================================================
# 2.  FixtureCalendar
# =============================================================================

class FixtureCalendar:
    """
    Analyses the FPL fixture schedule to surface blanks, doubles and
    per-team difficulty profiles.

    Parameters
    ----------
    fixtures_df : pd.DataFrame
        From ``FPLClient.get_fixtures_df()``.
        Required columns: gameweek, home_team_id, away_team_id,
                          home_difficulty, away_difficulty.
    teams_df : pd.DataFrame
        From ``FPLClient.get_teams_df()``.
        Required column: id.
    """

    def __init__(
        self,
        fixtures_df: pd.DataFrame,
        teams_df: pd.DataFrame,
    ) -> None:
        # ── Store clean copies ────────────────────────────────────────────
        self._fixtures: pd.DataFrame = (
            fixtures_df
            .dropna(subset=["gameweek"])
            .copy()
        )
        self._fixtures["gameweek"] = self._fixtures["gameweek"].astype(int)

        self._teams: pd.DataFrame = teams_df.copy()
        self._team_ids: list[int] = sorted(self._teams["id"].astype(int).tolist())
        self._all_gws: list[int] = sorted(
            self._fixtures["gameweek"].unique().tolist()
        )

        # ── Build long-form team↔fixture table once ───────────────────────
        self._long: pd.DataFrame = self._build_long()

    # ── Private helpers ───────────────────────────────────────────────────

    def _build_long(self) -> pd.DataFrame:
        """
        Explode the fixtures table into a per-team, per-fixture long table.

        Columns: gameweek, team_id, opponent_id, is_home, difficulty
        """
        home = self._fixtures[[
            "gameweek", "home_team_id", "away_team_id", "home_difficulty"
        ]].rename(columns={
            "home_team_id": "team_id",
            "away_team_id": "opponent_id",
            "home_difficulty": "difficulty",
        }).copy()
        home["is_home"] = True

        away = self._fixtures[[
            "gameweek", "away_team_id", "home_team_id", "away_difficulty"
        ]].rename(columns={
            "away_team_id": "team_id",
            "home_team_id": "opponent_id",
            "away_difficulty": "difficulty",
        }).copy()
        away["is_home"] = False

        long = (
            pd.concat([home, away], ignore_index=True)
            .assign(
                team_id=lambda df: df["team_id"].astype(int),
                opponent_id=lambda df: df["opponent_id"].astype(int),
                difficulty=lambda df: pd.to_numeric(
                    df["difficulty"], errors="coerce"
                ).fillna(_DEFAULT_DIFF),
            )
            .sort_values(["team_id", "gameweek"])
            .reset_index(drop=True)
        )
        return long

    def _fixture_counts(self) -> pd.DataFrame:
        """Return ``(gameweek, team_id, n_fixtures)`` aggregation."""
        return (
            self._long
            .groupby(["gameweek", "team_id"], as_index=False)
            .size()
            .rename(columns={"size": "n_fixtures"})
        )

    # ── Public API ────────────────────────────────────────────────────────

    def get_blanks(self) -> dict[int, list[int]]:
        """
        Return ``{gameweek: [team_ids]}`` for every team that has **zero**
        fixtures scheduled in that gameweek.

        Only gameweeks where at least one other team *does* have a fixture
        are included (i.e. pre-season / international-break GWs are skipped).
        """
        counts = self._fixture_counts()
        blanks: dict[int, list[int]] = {}

        for gw in self._all_gws:
            active_teams: set[int] = set(
                counts.loc[counts["gameweek"] == gw, "team_id"].tolist()
            )
            blanking = sorted(t for t in self._team_ids if t not in active_teams)
            if blanking:
                blanks[gw] = blanking

        return blanks

    def get_doubles(self) -> dict[int, list[int]]:
        """
        Return ``{gameweek: [team_ids]}`` for every team that has **2 or more**
        fixtures in that gameweek.
        """
        counts = self._fixture_counts()
        doubles: dict[int, list[int]] = {}

        dgw_rows = counts[counts["n_fixtures"] >= 2]
        for gw, grp in dgw_rows.groupby("gameweek"):
            doubles[int(gw)] = sorted(grp["team_id"].tolist())

        return doubles

    def get_team_calendar(self, team_id: int) -> pd.DataFrame:
        """
        Full gameweek-by-gameweek schedule for *team_id*.

        Columns
        -------
        gameweek    int    FPL GW number
        opponent_id int    Opponent's team ID (0 if blank)
        is_home     bool   True if home fixture (False for blanks)
        difficulty  float  FPL fixture difficulty rating (3.0 for blanks)
        n_fixtures  int    0, 1 or 2+
        is_blank    bool   True when n_fixtures == 0
        is_double   bool   True when n_fixtures >= 2

        For a double gameweek the row reflects the *easier* fixture
        (lower difficulty); the harder fixture is accessible via
        ``calendar._long``.
        """
        team_fx = self._long[self._long["team_id"] == team_id]

        rows: list[dict] = []
        for gw in self._all_gws:
            gw_fx = team_fx[team_fx["gameweek"] == gw]
            n = len(gw_fx)

            if n == 0:
                rows.append({
                    "gameweek":    gw,
                    "opponent_id": 0,
                    "is_home":     False,
                    "difficulty":  _DEFAULT_DIFF,
                    "n_fixtures":  0,
                    "is_blank":    True,
                    "is_double":   False,
                })
            else:
                # Primary row = easiest fixture (lower difficulty = better draw)
                primary = gw_fx.sort_values("difficulty").iloc[0]
                rows.append({
                    "gameweek":    gw,
                    "opponent_id": int(primary["opponent_id"]),
                    "is_home":     bool(primary["is_home"]),
                    "difficulty":  float(primary["difficulty"]),
                    "n_fixtures":  n,
                    "is_blank":    False,
                    "is_double":   n >= 2,
                })

        return pd.DataFrame(rows)

    def get_fixture_difficulty(
        self,
        team_id: int,
        from_gw: int,
        n_gameweeks: int,
    ) -> float:
        """
        Weighted average fixture difficulty for *team_id* over
        ``from_gw … from_gw + n_gameweeks - 1``.

        Each fixture contributes its difficulty rating.
        Blank gameweeks contribute nothing (weight 0).
        Returns ``3.0`` (FPL mid-difficulty) when no fixtures are found.

        Parameters
        ----------
        team_id      : int
        from_gw      : int   First gameweek to include.
        n_gameweeks  : int   Window length.
        """
        target_gws = set(range(from_gw, from_gw + n_gameweeks))
        subset = self._long[
            (self._long["team_id"] == team_id)
            & (self._long["gameweek"].isin(target_gws))
        ]

        if subset.empty:
            return _DEFAULT_DIFF

        difficulties = subset["difficulty"].values.astype(float)
        weights      = np.ones(len(difficulties))          # weight = 1 per fixture
        return float(np.average(difficulties, weights=weights))

    def score_dgw_value(
        self,
        player_ids: list[int],
        players_df: pd.DataFrame,
        dgw_teams: list[int],
        predictions_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Assign a DGW value multiplier to each player.

        Rules
        -----
        * Player's team **not** in ``dgw_teams``               → 1.00
        * Player's team in ``dgw_teams``, **rotation risk**   → 1.30
        * Player's team in ``dgw_teams``, **regular starter** → 1.65

        "Rotation risk" heuristic
        ~~~~~~~~~~~~~~~~~~~~~~~~~
        1. If ``predictions_df`` contains an ``xP`` column, a player whose
           predicted xP is more than 30 % below the median for their position
           is flagged as rotation risk.
        2. Otherwise, ``players_df`` columns ``starts`` and ``minutes`` are
           used: a player averaging < 65 minutes per start is considered
           rotation risk.  A player with zero starts is always rotation risk.

        Parameters
        ----------
        player_ids     : list[int]   FPL element IDs to score.
        players_df     : pd.DataFrame   From ``FPLClient.get_players_df()``.
        dgw_teams      : list[int]   Team IDs that have a double GW.
        predictions_df : pd.DataFrame   May contain ``element_id`` + ``xP``.

        Returns
        -------
        pd.DataFrame with columns: ``element_id``, ``dgw_multiplier``
        """
        dgw_teams_set = set(dgw_teams)

        # ── Build player lookup ───────────────────────────────────────────
        keep = [c for c in ("element_id", "team_id", "minutes", "starts", "position")
                if c in players_df.columns]
        plookup = (
            players_df[players_df["element_id"].isin(player_ids)][keep]
            .set_index("element_id")
        )

        # ── Build xP lookup (optional) ────────────────────────────────────
        xp_by_id: dict[int, float] = {}
        xp_median_by_pos: dict[str, float] = {}
        xp_overall_median: float = 0.0
        has_xp = (
            not predictions_df.empty
            and "xP" in predictions_df.columns
            and "element_id" in predictions_df.columns
        )
        if has_xp:
            xp_by_id = dict(zip(
                predictions_df["element_id"].astype(int),
                predictions_df["xP"].astype(float),
            ))
            # Overall median across all predicted players (position-agnostic baseline)
            xp_overall_median = float(predictions_df["xP"].median())

            # Per-position median (used when position data is available)
            if "position" in players_df.columns and "element_id" in players_df.columns:
                pred_merged = predictions_df.merge(
                    players_df[["element_id", "position"]],
                    on="element_id",
                    how="left",
                )
                if "position" in pred_merged.columns:
                    for pos, grp in pred_merged.groupby("position"):
                        pos_vals = grp["xP"].dropna()
                        if len(pos_vals) >= 3:          # need enough peers for a meaningful median
                            xp_median_by_pos[str(pos)] = float(pos_vals.median())

        # ── Assign multipliers ────────────────────────────────────────────
        results: list[dict] = []
        for pid in player_ids:
            if pid not in plookup.index:
                results.append({"element_id": pid, "dgw_multiplier": 1.0})
                continue

            prow = plookup.loc[pid]
            team_id = int(prow["team_id"])

            if team_id not in dgw_teams_set:
                results.append({"element_id": pid, "dgw_multiplier": 1.0})
                continue

            # ── Determine rotation risk ───────────────────────────────────
            rotation_risk: bool

            if has_xp and pid in xp_by_id:
                player_xp = xp_by_id[pid]
                pos = str(prow.get("position", "UNK"))
                # Prefer per-position median when we have enough peers;
                # fall back to the overall median across all predicted players.
                median_xp = xp_median_by_pos.get(pos, xp_overall_median) or xp_overall_median
                # More than 30 % below the reference median → rotation risk
                rotation_risk = bool(player_xp < median_xp * 0.70)

            else:
                starts  = float(prow.get("starts", 0) or 0)
                minutes = float(prow.get("minutes", 0) or 0)

                if starts == 0:
                    rotation_risk = True
                else:
                    # Average minutes per start; < 65 → often subbed, rotation risk
                    avg_mins_per_start = minutes / starts
                    rotation_risk = avg_mins_per_start < 65.0

            multiplier = 1.30 if rotation_risk else 1.65
            results.append({"element_id": pid, "dgw_multiplier": multiplier})

        return pd.DataFrame(results)


# =============================================================================
# 3.  CupTracker
# =============================================================================

@dataclass
class CupTracker:
    """
    Tracks which PL clubs are still alive in domestic and European cups.

    Persistence
    -----------
    Use ``CupTracker.load()`` to restore state from the JSON cache and
    ``tracker.save()`` to write it back.  The default path is
    ``DATA_DIR/cache/cup_tracker.json``.

    Competition keys
    ----------------
    ``"fa_cup"`` | ``"league_cup"`` | ``"ucl"`` | ``"uel"``
    """

    teams_in_fa_cup:     list[str] = field(default_factory=list)
    teams_in_league_cup: list[str] = field(default_factory=list)
    teams_in_ucl:        list[str] = field(default_factory=list)
    teams_in_uel:        list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        _CUP_TRACKER_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ── Persistence ───────────────────────────────────────────────────────

    @classmethod
    def load(cls, path: Path | None = None) -> "CupTracker":
        """
        Load from *path* (default: ``DATA_DIR/cache/cup_tracker.json``).
        Returns a blank tracker if the file does not exist.
        """
        p = path or _CUP_TRACKER_PATH
        if p.exists():
            try:
                data: dict = json.loads(p.read_text())
                return cls(
                    teams_in_fa_cup=data.get("teams_in_fa_cup", []),
                    teams_in_league_cup=data.get("teams_in_league_cup", []),
                    teams_in_ucl=data.get("teams_in_ucl", []),
                    teams_in_uel=data.get("teams_in_uel", []),
                )
            except (json.JSONDecodeError, KeyError):
                pass  # fall through → return blank tracker
        return cls()

    def save(self, path: Path | None = None) -> None:
        """Persist current state to JSON (creates directories as needed)."""
        p = path or _CUP_TRACKER_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {
                    "teams_in_fa_cup":     self.teams_in_fa_cup,
                    "teams_in_league_cup": self.teams_in_league_cup,
                    "teams_in_ucl":        self.teams_in_ucl,
                    "teams_in_uel":        self.teams_in_uel,
                    "_saved_at":           datetime.utcnow().isoformat(),
                },
                indent=2,
            )
        )

    # ── Mutation ──────────────────────────────────────────────────────────

    def eliminate_team(self, team: str, competition: str) -> None:
        """
        Remove *team* from *competition* when they are knocked out.

        Parameters
        ----------
        team        : str   Short or full club name exactly as stored.
        competition : str   ``"fa_cup"``, ``"league_cup"``, ``"ucl"``
                            or ``"uel"``.

        Raises
        ------
        ValueError  If *competition* is not one of the four valid keys.
        """
        attr = _COMP_TO_ATTR.get(competition)
        if attr is None:
            raise ValueError(
                f"Unknown competition {competition!r}. "
                f"Valid options: {sorted(_COMP_TO_ATTR.keys())}"
            )
        team_list: list[str] = getattr(self, attr)
        if team in team_list:
            team_list.remove(team)

    # ── Queries ───────────────────────────────────────────────────────────

    def is_in_european_competition(self, team: str) -> bool:
        """Return ``True`` if *team* is still active in UCL **or** UEL."""
        return team in self.teams_in_ucl or team in self.teams_in_uel

    def all_teams_in_cups(self) -> set[str]:
        """Union of all teams still alive in any tracked competition."""
        return (
            set(self.teams_in_fa_cup)
            | set(self.teams_in_league_cup)
            | set(self.teams_in_ucl)
            | set(self.teams_in_uel)
        )

    def european_teams(self) -> set[str]:
        """Teams still alive in UCL or UEL."""
        return set(self.teams_in_ucl) | set(self.teams_in_uel)


# =============================================================================
# 4.  predict_future_events
# =============================================================================

def predict_future_events(
    calendar: FixtureCalendar,
    cup_tracker: CupTracker,
    current_gw: int,
    lookahead: int = 10,
) -> list[PredictedEvent]:
    """
    Rule-based prediction of blank and double gameweeks in the lookahead window.

    Rules (applied in priority order)
    -----------------------------------
    1. **Confirmed DGWs** – fixture data already shows 2+ games for a team
       (confidence 1.0).
    2. **Confirmed BGWs** – fixture data shows teams with zero fixtures;
       confidence 1.0 if 8+ teams blank, 0.85 otherwise.
    3. **FA Cup potential blanks** – GWs overlapping with known FA Cup round
       dates *and* PL teams still in the competition may blank
       (confidence 0.6).
    4. **Historical DGW tendency** – GW28–37 historically peak for doubles;
       unconfirmed GWs in this range get confidence 0.45.
    5. **Historical BGW tendency** – GW25–30 historically peak for blanks;
       unconfirmed GWs in this range get confidence 0.40.

    Note: European rotation risk (UCL/UEL in ``_EUROPEAN_GWS``) is **not**
    modelled as a BGW/DGW event – it is a player-level concern surfaced
    separately via ``FixtureCalendar.score_dgw_value``.

    Returns
    -------
    list[PredictedEvent]
        Sorted by confidence (descending), then by gameweek (ascending).
    """
    target_gws = list(range(current_gw, current_gw + lookahead))

    blanks_map:  dict[int, list[int]] = calendar.get_blanks()
    doubles_map: dict[int, list[int]] = calendar.get_doubles()

    events: list[PredictedEvent] = []
    seen: set[tuple[int, str]] = set()   # (gw, event_type) de-duplication

    # ── Rule 1: Confirmed double gameweeks ────────────────────────────────
    for gw in target_gws:
        if gw in doubles_map and doubles_map[gw]:
            key = (gw, "DGW")
            if key not in seen:
                events.append(PredictedEvent(
                    gameweek=gw,
                    teams=doubles_map[gw],
                    event_type="DGW",
                    confidence=1.0,
                    notes=(
                        f"{len(doubles_map[gw])} team(s) have 2+ fixtures "
                        "in this GW (confirmed from fixture list)."
                    ),
                ))
                seen.add(key)

    # ── Rule 2: Confirmed blank gameweeks ─────────────────────────────────
    for gw in target_gws:
        if gw in blanks_map and blanks_map[gw]:
            n_blank = len(blanks_map[gw])
            # Large-scale blanks (e.g. FA Cup 5th round) are near-certain
            confidence = 1.0 if n_blank >= 8 else 0.85
            key = (gw, "BGW")
            if key not in seen:
                events.append(PredictedEvent(
                    gameweek=gw,
                    teams=blanks_map[gw],
                    event_type="BGW",
                    confidence=confidence,
                    notes=(
                        f"{n_blank} team(s) have no fixture in GW{gw} "
                        "(confirmed from fixture list)."
                    ),
                ))
                seen.add(key)

    # ── Rule 3: FA Cup potential blanks ───────────────────────────────────
    n_fa_teams = len(cup_tracker.teams_in_fa_cup)
    if n_fa_teams > 0:
        for gw in target_gws:
            if gw in _FA_CUP_GWS and (gw, "BGW") not in seen:
                events.append(PredictedEvent(
                    gameweek=gw,
                    teams=[],   # team IDs not yet confirmed
                    event_type="BGW",
                    confidence=0.60,
                    notes=(
                        f"FA Cup round expected around GW{gw}; "
                        f"{n_fa_teams} PL team(s) still in the competition "
                        "and may have their PL fixture postponed."
                    ),
                ))
                seen.add((gw, "BGW"))

    # ── Rule 4: Historical DGW tendency (peak zone GW28-37) ───────────────
    for gw in target_gws:
        if gw in _DGW_PEAK_GWS and (gw, "DGW") not in seen:
            events.append(PredictedEvent(
                gameweek=gw,
                teams=[],
                event_type="DGW",
                confidence=0.45,
                notes=(
                    f"GW{gw} falls in the historical DGW peak zone (GW28–37). "
                    "No fixture data yet to confirm specific teams."
                ),
            ))
            seen.add((gw, "DGW"))

    # ── Rule 5: Historical BGW tendency (peak zone GW25-30) ───────────────
    for gw in target_gws:
        if gw in _BGW_PEAK_GWS and (gw, "BGW") not in seen:
            events.append(PredictedEvent(
                gameweek=gw,
                teams=[],
                event_type="BGW",
                confidence=0.40,
                notes=(
                    f"GW{gw} falls in the historical BGW peak zone (GW25–30). "
                    "No fixture data yet to confirm specific teams."
                ),
            ))
            seen.add((gw, "BGW"))

    # Sort: confidence desc → gameweek asc
    events.sort(key=lambda e: (-e.confidence, e.gameweek))
    return events


# =============================================================================
# 5.  suggest_chip_timing
# =============================================================================

def suggest_chip_timing(
    available_chips: list[str],
    calendar: FixtureCalendar,
    cup_tracker: CupTracker,
    current_gw: int,
    n_remaining: int,
    predictions_by_gw: dict[int, pd.DataFrame] | None = None,
) -> dict[str, dict[int, float]]:
    """
    Score each chip for every gameweek in ``current_gw … current_gw + n_remaining - 1``.

    Scoring formulae
    ----------------
    bench_boost
        ``Σ(avg_squad_xP × fixture_multiplier)`` across the squad (15 players).
        ``fixture_multiplier = 1 + n_double_teams / 20``.
        Best in DGWs.  Falls back to heuristic when xP data absent.

    free_hit
        ``(n_blanking_teams × 3.0) + base_quality_score``.
        Best in heavy BGWs where the existing squad is decimated.

    triple_captain
        ``top_player_xP × (1 + n_double_teams / 20)``.
        Best in DGWs with a premium captaincy target.

    wildcard
        Sum of ``(4 − mean_difficulty)`` over the next 6 GWs from each
        candidate activation point.  Higher = easier fixtures ahead.
        A BGW bonus (0.5 per blanking team) rewards activating in a week
        where the squad needs major surgery.

    Parameters
    ----------
    available_chips     : list[str]
        Subset of ``["bench_boost", "free_hit", "triple_captain", "wildcard"]``
        still available to the manager.
    calendar            : FixtureCalendar
    cup_tracker         : CupTracker
    current_gw          : int
    n_remaining         : int
        Number of gameweeks to evaluate.
    predictions_by_gw   : dict[int, pd.DataFrame] | None
        Optional per-GW prediction tables with ``element_id`` and ``xP``
        columns.

    Returns
    -------
    dict[str, dict[int, float]]
        ``{chip_name: {gameweek: value_score}}``.
        Only chips present in *available_chips* are included.
    """
    gw_range   = list(range(current_gw, current_gw + n_remaining))
    blanks_map = calendar.get_blanks()
    doubles_map = calendar.get_doubles()
    n_teams     = len(calendar._team_ids) or 20   # Premier League = 20 teams

    # ── Pre-compute per-GW statistics ────────────────────────────────────
    gw_stats: dict[int, dict] = {}
    for gw in gw_range:
        blank_teams  = blanks_map.get(gw, [])
        double_teams = doubles_map.get(gw, [])

        gw_fx    = calendar._long[calendar._long["gameweek"] == gw]
        mean_diff = float(gw_fx["difficulty"].mean()) if not gw_fx.empty else _DEFAULT_DIFF

        # xP data (optional)
        avg_squad_xp  = 0.0
        top_player_xp = 0.0
        if predictions_by_gw and gw in predictions_by_gw:
            pred = predictions_by_gw[gw]
            if not pred.empty and "xP" in pred.columns:
                avg_squad_xp  = float(pred["xP"].mean())
                top_player_xp = float(pred["xP"].max())

        gw_stats[gw] = {
            "n_blank":        len(blank_teams),
            "n_double":       len(double_teams),
            "blank_teams":    blank_teams,
            "double_teams":   double_teams,
            "mean_diff":      mean_diff,
            "avg_squad_xp":   avg_squad_xp,
            "top_player_xp":  top_player_xp,
        }

    result: dict[str, dict[int, float]] = {}

    # ── bench_boost ───────────────────────────────────────────────────────
    if "bench_boost" in available_chips:
        bb: dict[int, float] = {}
        for gw in gw_range:
            s = gw_stats[gw]
            # fixture_multiplier: doubles mean bench players also get 2 games
            fixture_mult = 1.0 + (s["n_double"] / n_teams)

            if s["avg_squad_xp"] > 0:
                # Full squad = 15 players; boost unlocks all bench players
                score = s["avg_squad_xp"] * fixture_mult * 15.0
            else:
                # Heuristic: base value + DGW bonus - BGW penalty
                score = (
                    5.0
                    + s["n_double"] * 0.80
                    - s["n_blank"]  * 0.30
                    + (1.0 - s["mean_diff"] / 5.0) * 2.0
                )
            bb[gw] = round(max(0.0, score), 3)
        result["bench_boost"] = bb

    # ── free_hit ─────────────────────────────────────────────────────────
    if "free_hit" in available_chips:
        fh: dict[int, float] = {}
        for gw in gw_range:
            s = gw_stats[gw]
            # Easier fixtures = better free-hit squad quality
            base = 5.0 - (s["mean_diff"] - _DEFAULT_DIFF) * 0.5
            # Main driver: blanking teams force squad holes the free-hit can fix
            blank_bonus  = s["n_blank"]  * 3.0
            double_bonus = s["n_double"] * 0.5   # can also grab DGW players
            score = base + blank_bonus + double_bonus
            fh[gw] = round(max(0.0, score), 3)
        result["free_hit"] = fh

    # ── triple_captain ────────────────────────────────────────────────────
    if "triple_captain" in available_chips:
        tc: dict[int, float] = {}
        for gw in gw_range:
            s = gw_stats[gw]
            fixture_mult = 1.0 + (s["n_double"] / n_teams)

            if s["top_player_xp"] > 0:
                score = s["top_player_xp"] * fixture_mult
            else:
                # Heuristic: DGW adds big captain upside; high difficulty reduces it
                score = (
                    6.0
                    + s["n_double"] * 1.20
                    - (s["mean_diff"] - _DEFAULT_DIFF) * 0.80
                )
            tc[gw] = round(max(0.0, score), 3)
        result["triple_captain"] = tc

    # ── wildcard ─────────────────────────────────────────────────────────
    if "wildcard" in available_chips:
        wc: dict[int, float] = {}
        for gw in gw_range:
            # Measure fixture quality swing over the *next* 6 GWs from this point
            swing = 0.0
            for lookahead_gw in range(gw, gw + 6):
                lx = calendar._long[calendar._long["gameweek"] == lookahead_gw]
                if lx.empty:
                    continue
                gw_mean_diff = float(lx["difficulty"].mean())
                # Inverted: easier fixtures (low diff) = higher wildcard value
                # Scaled on a 1-5 FPL difficulty range; 4 is the reference pivot
                swing += max(0.0, 4.0 - gw_mean_diff)

            # BGW bonus: wildcarding in a blank week helps immediately
            blank_bonus = len(blanks_map.get(gw, [])) * 0.50
            wc[gw] = round(swing + blank_bonus, 3)
        result["wildcard"] = wc

    return result
