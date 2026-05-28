"""
Build historical WTA training rows for the match predictor.

Scope:
- 2010-2025 by default
- WTA 500 and above, approximated from historical tourney_level values
- main-draw singles rows from Jeff Sackmann yearly WTA match CSVs

Every row is created before updating player state for that match, so features
only use information available before the match was played.
"""

from __future__ import annotations

import argparse
import csv
import io
import random
import sys
import urllib.request
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


MATCH_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_{year}.csv"
USER_AGENT = "Mozilla/5.0 (compatible; WTA-Match-Predictor/0.1; +https://localhost)"

INITIAL_ELO = 1500.0
ELO_K = 24.0
RECENT_PRIOR_WINS = 2
RECENT_PRIOR_LOSSES = 2
MISSING_RANK = 999
NEUTRAL_AGE = 26.0
RECENT_MATCHES = 10

FEATURES = [
    "rankAdvantage",
    "eloAdvantage",
    "surfaceEloAdvantage",
    "surfaceWinRateAdvantage",
    "surfaceMatchSampleAdvantage",
    "levelWinRateAdvantage",
    "levelMatchSampleAdvantage",
    "recentFormAdvantage",
    "recentSampleAdvantage",
    "h2hAdvantage",
    "h2hSampleSize",
    "ageAdvantage",
    "experienceAdvantage",
    "rankMissingAdvantage",
    "ageMissingAdvantage",
    "tournamentLevel",
]

LEVEL_WEIGHTS = {
    "500": 0.18,
    "1000": 0.24,
    "grand_slam": 0.32,
    "finals": 0.28,
}


@dataclass
class PlayerState:
    elo: float = INITIAL_ELO
    surface_elo: dict[str, float] = field(default_factory=lambda: defaultdict(lambda: INITIAL_ELO))
    surface_records: dict[str, list[int]] = field(default_factory=lambda: defaultdict(lambda: [0, 0]))
    level_records: dict[str, list[int]] = field(default_factory=lambda: defaultdict(lambda: [0, 0]))
    recent: deque[int] = field(default_factory=lambda: deque(maxlen=RECENT_MATCHES))
    high_level_matches: int = 0


def fetch_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=45) as response:
        return response.read().decode("utf-8", "replace")


def read_year(year: int) -> list[dict]:
    text = fetch_text(MATCH_URL.format(year=year))
    return list(csv.DictReader(io.StringIO(text)))


def parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y%m%d")


def normalize_surface(value: str) -> str | None:
    surface = (value or "").strip().lower()
    if surface in {"hard", "clay", "grass"}:
        return surface
    return None


def level_from_row(row: dict) -> str | None:
    raw = (row.get("tourney_level") or "").strip()
    name = (row.get("tourney_name") or "").lower()

    if raw == "G":
        return "grand_slam"
    if raw == "F":
        return "finals"
    if raw == "PM":
        return "1000"
    if raw == "P":
        if any(token in name for token in ["indian wells", "miami", "madrid", "rome", "beijing", "wuhan", "doha", "dubai", "canada", "toronto", "montreal", "cincinnati"]):
            return "1000"
        return "500"
    return None


def clean_name(value: str) -> str:
    return " ".join((value or "").strip().split())


def rank_for(row: dict, side: str) -> tuple[int, int]:
    value = row.get(f"{side}_rank")
    try:
        return int(float(value)), 0
    except (TypeError, ValueError):
        return MISSING_RANK, 1


def age_for(row: dict, side: str) -> tuple[float, int]:
    value = row.get(f"{side}_age")
    try:
        return float(value), 0
    except (TypeError, ValueError):
        return NEUTRAL_AGE, 1


def recent_rate(state: PlayerState) -> tuple[float, int]:
    wins = sum(state.recent)
    total = len(state.recent)
    rate = (wins + RECENT_PRIOR_WINS) / (total + RECENT_PRIOR_WINS + RECENT_PRIOR_LOSSES)
    return rate, total


def surface_rate(state: PlayerState, surface: str) -> tuple[float, int]:
    wins, losses = state.surface_records[surface]
    total = wins + losses
    rate = (wins + RECENT_PRIOR_WINS) / (total + RECENT_PRIOR_WINS + RECENT_PRIOR_LOSSES)
    return rate, total


def level_rate(state: PlayerState, level: str) -> tuple[float, int]:
    wins, losses = state.level_records[level]
    total = wins + losses
    rate = (wins + RECENT_PRIOR_WINS) / (total + RECENT_PRIOR_WINS + RECENT_PRIOR_LOSSES)
    return rate, total


def h2h_record(h2h: dict[tuple[str, str], list[int]], player_a: str, player_b: str) -> tuple[int, int, int]:
    wins_a, wins_b = h2h[(player_a, player_b)]
    total = wins_a + wins_b
    return wins_a, wins_b, total


def elo_expected(rating_a: float, rating_b: float) -> float:
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def update_elo(winner_state: PlayerState, loser_state: PlayerState, surface: str) -> None:
    expected = elo_expected(winner_state.elo, loser_state.elo)
    change = ELO_K * (1 - expected)
    winner_state.elo += change
    loser_state.elo -= change

    expected_surface = elo_expected(winner_state.surface_elo[surface], loser_state.surface_elo[surface])
    surface_change = ELO_K * (1 - expected_surface)
    winner_state.surface_elo[surface] += surface_change
    loser_state.surface_elo[surface] -= surface_change


def feature_row(row: dict, player_a: str, player_b: str, result: int, states: dict[str, PlayerState], h2h: dict[tuple[str, str], list[int]], surface: str, level: str, a_side: str, b_side: str) -> dict:
    state_a = states[player_a]
    state_b = states[player_b]
    rank_a, rank_missing_a = rank_for(row, a_side)
    rank_b, rank_missing_b = rank_for(row, b_side)
    age_a, age_missing_a = age_for(row, a_side)
    age_b, age_missing_b = age_for(row, b_side)
    recent_a, recent_n_a = recent_rate(state_a)
    recent_b, recent_n_b = recent_rate(state_b)
    surface_rate_a, surface_n_a = surface_rate(state_a, surface)
    surface_rate_b, surface_n_b = surface_rate(state_b, surface)
    level_rate_a, level_n_a = level_rate(state_a, level)
    level_rate_b, level_n_b = level_rate(state_b, level)
    h2h_wins_a, h2h_wins_b, h2h_total = h2h_record(h2h, player_a, player_b)

    values = {
        "rankAdvantage": rank_b - rank_a,
        "eloAdvantage": state_a.elo - state_b.elo,
        "surfaceEloAdvantage": state_a.surface_elo[surface] - state_b.surface_elo[surface],
        "surfaceWinRateAdvantage": surface_rate_a - surface_rate_b,
        "surfaceMatchSampleAdvantage": surface_n_a - surface_n_b,
        "levelWinRateAdvantage": level_rate_a - level_rate_b,
        "levelMatchSampleAdvantage": level_n_a - level_n_b,
        "recentFormAdvantage": recent_a - recent_b,
        "recentSampleAdvantage": recent_n_a - recent_n_b,
        "h2hAdvantage": ((h2h_wins_a - h2h_wins_b) / h2h_total) if h2h_total else 0,
        "h2hSampleSize": h2h_total,
        "ageAdvantage": abs(age_b - NEUTRAL_AGE) - abs(age_a - NEUTRAL_AGE),
        "experienceAdvantage": state_a.high_level_matches - state_b.high_level_matches,
        "rankMissingAdvantage": rank_missing_b - rank_missing_a,
        "ageMissingAdvantage": age_missing_b - age_missing_a,
        "tournamentLevel": LEVEL_WEIGHTS[level],
        "result": result,
        "matchDate": row["tourney_date"],
        "tournament": row.get("tourney_name", ""),
        "surface": surface,
        "level": level,
        "playerA": player_a,
        "playerB": player_b,
    }
    return values


def build_rows(start_year: int, end_year: int) -> list[dict]:
    states: dict[str, PlayerState] = defaultdict(PlayerState)
    h2h: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    rows_out: list[dict] = []
    raw_rows: list[dict] = []

    for year in range(start_year, end_year + 1):
        try:
            raw_rows.extend(read_year(year))
        except Exception as exc:
            print(f"Warning: could not fetch {year}: {exc}", file=sys.stderr)

    raw_rows.sort(key=lambda row: (row.get("tourney_date", ""), row.get("match_num", "")))
    random.seed(7)

    for row in raw_rows:
        level = level_from_row(row)
        surface = normalize_surface(row.get("surface", ""))
        if not level or not surface:
            continue

        winner = clean_name(row.get("winner_name", ""))
        loser = clean_name(row.get("loser_name", ""))
        if not winner or not loser:
            continue

        flip = random.random() < 0.5
        if flip:
            rows_out.append(feature_row(row, loser, winner, 0, states, h2h, surface, level, "loser", "winner"))
        else:
            rows_out.append(feature_row(row, winner, loser, 1, states, h2h, surface, level, "winner", "loser"))

        winner_state = states[winner]
        loser_state = states[loser]
        update_elo(winner_state, loser_state, surface)
        winner_state.surface_records[surface][0] += 1
        loser_state.surface_records[surface][1] += 1
        winner_state.level_records[level][0] += 1
        loser_state.level_records[level][1] += 1
        winner_state.recent.append(1)
        loser_state.recent.append(0)
        winner_state.high_level_matches += 1
        loser_state.high_level_matches += 1
        h2h[(winner, loser)][0] += 1
        h2h[(loser, winner)][1] += 1

    return rows_out


def write_rows(path: Path, rows: list[dict]) -> None:
    fieldnames = FEATURES + [
        "result",
        "matchDate",
        "tournament",
        "surface",
        "level",
        "playerA",
        "playerB",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--output", default="data/training_rows.csv")
    args = parser.parse_args()

    rows = build_rows(args.start_year, args.end_year)
    write_rows(Path(args.output), rows)
    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
