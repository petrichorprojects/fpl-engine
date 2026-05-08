"""Press conference NLP — extract injury/rotation signals from manager transcripts.

Uses regex + keyword patterns (no ML dependencies) to identify:
  - Players ruled out (INJURY_OUT)
  - Players doubtful (INJURY_DOUBT)
  - Rotation risk (ROTATION_RISK)
  - Likely rotation (ROTATION_LIKELY)
  - Confirmed fit (CONFIRMED_FIT)
  - Returning from injury (RETURNING)

These signals feed into the minutes model as presser_adjustment features
(-1.0 = definitely out, +1.0 = definitely starting).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import httpx
import pandas as pd

from .config import DATA_DIR

CACHE_DIR = DATA_DIR / "cache"
ROTATION_CACHE = CACHE_DIR / "rotation_patterns.json"

SignalType = Literal[
    "INJURY_OUT", "INJURY_DOUBT", "ROTATION_RISK",
    "ROTATION_LIKELY", "CONFIRMED_FIT", "RETURNING",
]

# ── Confidence weights per signal type ──────────────────────────────────────
SIGNAL_CONFIDENCE: dict[SignalType, float] = {
    "INJURY_OUT": 0.95,
    "INJURY_DOUBT": 0.65,
    "ROTATION_RISK": 0.50,
    "ROTATION_LIKELY": 0.70,
    "CONFIRMED_FIT": 0.90,
    "RETURNING": 0.80,
}

# presser_adjustment: how much to shift P(start) for this signal
SIGNAL_ADJUSTMENT: dict[SignalType, float] = {
    "INJURY_OUT": -1.0,
    "INJURY_DOUBT": -0.5,
    "ROTATION_RISK": -0.25,
    "ROTATION_LIKELY": -0.15,
    "CONFIRMED_FIT": +0.3,
    "RETURNING": +0.2,
}

# ── Regex patterns ───────────────────────────────────────────────────────────
INJURY_OUT_PATTERNS = re.compile(
    r"(?:ruled?\s*out|out\s+for|not\s+available|unavailable|sidelined|"
    r"long.?term|surgery|operation|no\s+chance|definitely\s+out|"
    r"won.t\s+(?:play|feature|be\s+involved))",
    re.IGNORECASE,
)
INJURY_DOUBT_PATTERNS = re.compile(
    r"(?:doubt(?:ful)?|knock|injury|injured|slight(?:ly)?\s+(?:injured|hurt)|"
    r"scan|not\s+100|touch\s+and\s+go|we.ll\s+see|assess\s+him|"
    r"50.50|unlikely|probably\s+(?:won.t|not)|may\s+miss|might\s+miss)",
    re.IGNORECASE,
)
ROTATION_PATTERNS = re.compile(
    r"(?:rest(?:ed|ing)?|rotat(?:e|ion|ing)|fresh\s+legs?|give\s+(?:him|them)\s+a\s+break|"
    r"squad\s+rotat|changes?\s+(?:expected|likely|coming)|opportunity|"
    r"chance\s+to\s+(?:play|feature|get\s+minutes)|look\s+at\s+(?:the\s+)?squad|"
    r"might\s+(?:rest|rotate)|could\s+(?:rest|change))",
    re.IGNORECASE,
)
CONFIRMED_PATTERNS = re.compile(
    r"(?:fit\s+and\s+(?:ready|available)|(?:fully\s+)?fit|available|"
    r"in\s+contention|trained\s+(?:fully|normally|well|with\s+(?:the\s+)?(?:group|squad))|"
    r"no\s+(?:concerns?|issues?|worries?|problems?)|back\s+to\s+(?:full\s+)?fitness|"
    r"(?:fully\s+)?recovered|ready\s+to\s+(?:play|go|feature))",
    re.IGNORECASE,
)
RETURNING_PATTERNS = re.compile(
    r"(?:back\s+in\s+training|returned?\s+to\s+training|stepped?\s+up\s+(?:his\s+)?fitness|"
    r"rejoined?\s+(?:the\s+)?(?:group|squad|team)|coming\s+back|"
    r"(?:close|close\s+to)\s+(?:a\s+)?return|making\s+(?:good\s+)?progress)",
    re.IGNORECASE,
)

# Name extraction: words starting with uppercase (rough heuristic)
NAME_PATTERN = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b")


@dataclass
class PresserSignal:
    """A single injury/rotation signal extracted from a press conference."""
    player_name: str
    team: str
    signal_type: SignalType
    confidence: float
    raw_quote: str
    gameweek: int

    @property
    def adjustment(self) -> float:
        """Minutes model adjustment value (-1.0 to +1.0)."""
        return SIGNAL_ADJUSTMENT[self.signal_type] * self.confidence


def _extract_names_near(text: str, match_start: int, match_end: int,
                        window: int = 80) -> list[str]:
    """Find player names (proper nouns) within window chars of a pattern match."""
    region_start = max(0, match_start - window)
    region_end = min(len(text), match_end + window)
    region = text[region_start:region_end]
    names = NAME_PATTERN.findall(region)
    # Filter out common non-names
    STOP = {
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
        "Saturday", "Sunday", "Premier", "League", "Cup", "January",
        "February", "March", "April", "May", "June", "July",
        "August", "September", "October", "November", "December",
        "We", "He", "She", "They", "The", "Our", "His", "Her",
        "It", "That", "This", "There", "Their", "Just",
    }
    return [n for n in names if n.split()[0] not in STOP and len(n) > 4]


def _classify_signal(text: str) -> tuple[SignalType | None, re.Match | None]:
    """Classify the primary signal type in a text fragment."""
    if m := INJURY_OUT_PATTERNS.search(text):
        return "INJURY_OUT", m
    if m := INJURY_DOUBT_PATTERNS.search(text):
        return "INJURY_DOUBT", m
    if m := RETURNING_PATTERNS.search(text):
        return "RETURNING", m
    if m := CONFIRMED_PATTERNS.search(text):
        return "CONFIRMED_FIT", m
    if m := ROTATION_PATTERNS.search(text):
        return "ROTATION_RISK", m
    return None, None


@dataclass
class PresserAnalyzer:
    """Analyzes press conference transcripts for FPL-relevant signals."""

    _transcripts: list[dict] = field(default_factory=list, init=False)

    def add_transcript(
        self,
        manager: str,
        team: str,
        text: str,
        gameweek: int,
    ) -> list[PresserSignal]:
        """Add a raw transcript and immediately extract signals.

        Args:
            manager: Manager name (for logging).
            team: Team name (used in PresserSignal.team).
            text: Full transcript text.
            gameweek: FPL gameweek this presser relates to.

        Returns:
            List of PresserSignal extracted from this transcript.
        """
        signals = self.analyze_transcript(text, team, gameweek)
        self._transcripts.append({
            "manager": manager,
            "team": team,
            "text": text,
            "gameweek": gameweek,
            "signals": [vars(s) for s in signals],
        })
        return signals

    def analyze_transcript(
        self,
        text: str,
        team: str,
        gameweek: int,
    ) -> list[PresserSignal]:
        """Extract PresserSignals from a raw press conference transcript.

        Strategy:
        1. Split into sentences.
        2. For each sentence, test for signal patterns.
        3. Extract player names near matching patterns.
        4. Emit a PresserSignal per (player, signal_type) pair.
        """
        signals: list[PresserSignal] = []
        seen: set[tuple[str, str]] = set()

        # Split by sentence
        sentences = re.split(r"(?<=[.!?])\s+", text)

        for sentence in sentences:
            sig_type, match = _classify_signal(sentence)
            if sig_type is None or match is None:
                continue

            names = _extract_names_near(sentence, match.start(), match.end())
            if not names:
                # Try the whole sentence for a name
                names = NAME_PATTERN.findall(sentence)
                names = [n for n in names if len(n.split()) >= 2 and len(n) > 4]

            for name in names:
                key = (name, sig_type)
                if key in seen:
                    continue
                seen.add(key)

                # Get a short quote around the match
                quote_start = max(0, match.start() - 30)
                quote_end = min(len(sentence), match.end() + 30)
                raw_quote = sentence[quote_start:quote_end].strip()

                conf = SIGNAL_CONFIDENCE[sig_type]
                signals.append(PresserSignal(
                    player_name=name,
                    team=team,
                    signal_type=sig_type,
                    confidence=conf,
                    raw_quote=raw_quote,
                    gameweek=gameweek,
                ))

        return signals

    def get_all_signals(self, gameweek: int | None = None) -> list[PresserSignal]:
        """Return all signals, optionally filtered by gameweek."""
        out: list[PresserSignal] = []
        for t in self._transcripts:
            if gameweek is not None and t["gameweek"] != gameweek:
                continue
            for s in t["signals"]:
                out.append(PresserSignal(**s))
        return out

    def fetch_transcripts(self, gameweek: int) -> list[PresserSignal]:
        """Attempt to fetch pre-match press conference summaries.

        Currently tries the BBC Sport team pages. Falls back gracefully.
        """
        signals: list[PresserSignal] = []
        try:
            url = "https://www.bbc.co.uk/sport/football/premier-league"
            resp = httpx.get(url, timeout=10, follow_redirects=True,
                             headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code == 200:
                # Very rough: extract text from <p> tags near "press conference"
                text = re.sub(r"<[^>]+>", " ", resp.text)
                snippet = ""
                idx = text.lower().find("press conference")
                if idx > 0:
                    snippet = text[max(0, idx - 200): idx + 500]
                if snippet:
                    signals = self.analyze_transcript(snippet, "unknown", gameweek)
        except Exception:
            pass
        return signals


def get_presser_adjustments(
    signals: list[PresserSignal],
    players_df: pd.DataFrame,
) -> pd.DataFrame:
    """Map PresserSignals onto FPL player IDs.

    Fuzzy-matches player_name in signals to players_df['name'] / ['full_name'].
    Returns DataFrame with element_id, presser_adjustment (-1.0 to +1.0).
    """
    import difflib

    adjustments: dict[int, float] = {}

    # Build name lookup
    name_to_id: dict[str, int] = {}
    for _, row in players_df.iterrows():
        pid = int(row["element_id"])
        for col in ["name", "full_name"]:
            raw = str(row.get(col, ""))
            if raw:
                name_to_id[raw.lower()] = pid

    all_names = list(name_to_id.keys())

    for sig in signals:
        lookup_name = sig.player_name.lower()
        matches = difflib.get_close_matches(lookup_name, all_names, n=1, cutoff=0.75)
        if not matches:
            continue
        pid = name_to_id[matches[0]]
        adj = sig.adjustment
        # Combine multiple signals: take strongest (most negative = worst injury)
        if pid not in adjustments or abs(adj) > abs(adjustments[pid]):
            adjustments[pid] = adj

    if not adjustments:
        return pd.DataFrame(columns=["element_id", "presser_adjustment"])

    return pd.DataFrame([
        {"element_id": pid, "presser_adjustment": adj}
        for pid, adj in adjustments.items()
    ])


@dataclass
@dataclass
class RotationContext:
    """Context that drives rotation probability estimates."""
    had_european_match: bool = False
    had_cup_match: bool = False
    days_since_last_match: int = 7
    games_in_7_days: int = 1
    player_recent_starts: int = 3
    season_minutes: int = 0


class ManagerRotationTracker:
    """Tracks per-manager rotation tendency to predict rotation risk.

    Learns from historical patterns:
    - How often does this manager rotate after European games?
    - How often after League Cup / FA Cup?
    - At what fixture congestion level does rotation kick in?
    """

    cache_file: Path = field(default_factory=lambda: ROTATION_CACHE)
    _patterns: dict = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self) -> None:
        if self.cache_file.exists():
            try:
                self._patterns = json.loads(self.cache_file.read_text())
            except Exception:
                self._patterns = {}

    def _save(self) -> None:
        self.cache_file.write_text(json.dumps(self._patterns, indent=2))

    def record_rotation(
        self,
        manager: str,
        team: str,
        context: str,  # 'european', 'cup', 'congestion', 'normal'
        rotated: bool,
    ) -> None:
        """Record whether a manager rotated in a given context."""
        key = f"{manager}:{context}"
        if key not in self._patterns:
            self._patterns[key] = {"rotated": 0, "total": 0}
        self._patterns[key]["total"] += 1
        if rotated:
            self._patterns[key]["rotated"] += 1
        self._save()

    def get_rotation_probability(
        self,
        team: str,
        player_name: str = "",
        context: "RotationContext | str" = "normal",
    ) -> float:
        """Estimate P(rotation) for a manager in a given context.

        Args:
            manager: Manager identifier string.
            context: One of 'european', 'cup', 'congestion', 'normal'.

        Returns:
            Float probability 0.0–1.0. Defaults by context if no history.
        """
        DEFAULTS = {
            "european": 0.55,
            "cup": 0.40,
            "congestion": 0.35,
            "normal": 0.15,
        }
        if isinstance(context, RotationContext):
            base = 0.15
            if context.had_european_match and context.days_since_last_match < 4:
                base += 0.25
            if context.had_cup_match:
                base += 0.15
            if context.games_in_7_days >= 3:
                base += 0.20
            elif context.games_in_7_days >= 2 and context.days_since_last_match < 4:
                base += 0.10
            if context.season_minutes > 2500:
                base += 0.05
            return min(1.0, max(0.0, base))
        DEFAULTS = {
            "european": 0.55, "cup": 0.40,
            "congestion": 0.35, "normal": 0.15,
        }
        key = f"{team}:{context}"
        if key in self._patterns:
            rec = self._patterns[key]
            if rec.get("total", 0) >= 3:
                return rec["rotated"] / rec["total"]
        return DEFAULTS.get(str(context), 0.20)

    def get_all_patterns(self) -> dict:
        """Return all stored rotation patterns."""
        return dict(self._patterns)
