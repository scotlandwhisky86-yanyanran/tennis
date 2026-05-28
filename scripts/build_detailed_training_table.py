"""
Build a detailed historical WTA feature table.

Output rows:
- 2010-2025 by default
- WTA 500 and above only, matching the product's prediction scope

State updates:
- the same 2010-2025 WTA 500+ main-draw singles rows used for output

Every output row is created before updating state for that match.
"""

from __future__ import annotations

import argparse
import csv
import io
import random
import re
import sys
import urllib.request
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


MATCH_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_{year}.csv"
USER_AGENT = "Mozilla/5.0 (compatible; WTA-Match-Predictor/0.2; +https://localhost)"

INITIAL_ELO = 1500.0
ELO_K = 24.0
MISSING_RANK = 999
NEUTRAL_AGE = 26.0
NEUTRAL_HEIGHT = 170.0
RECENT_PRIOR_WINS = 2
RECENT_PRIOR_LOSSES = 2

SURFACES = ["hard", "clay", "grass"]
STYLE_PRIORS = {
    "aceRate": (0.04, 50),
    "doubleFaultRate": (0.06, 50),
    "firstServeInRate": (0.62, 50),
    "firstServeWonRate": (0.62, 50),
    "secondServeWonRate": (0.45, 50),
    "servicePointWonRate": (0.57, 80),
    "returnPointWonRate": (0.43, 80),
    "breakPointSaveRate": (0.52, 12),
    "breakPointConvertRate": (0.48, 12),
    "tiebreakWinRate": (0.50, 4),
}

LEVEL_WEIGHTS = {
    "250": 0.10,
    "500": 0.18,
    "1000": 0.24,
    "grand_slam": 0.32,
    "finals": 0.28,
}


FEATURES = [
    "rankAdvantage",
    "rankPointsAdvantage",
    "overallEloAdvantage",
    "hardEloAdvantage",
    "clayEloAdvantage",
    "grassEloAdvantage",
    "matchSurfaceEloAdvantage",
    "ageAdvantage",
    "heightAdvantage",
    "leftyAdvantage",
    "sameHandedness",
    "overallWinRateAdvantage",
    "overallSampleAdvantage",
    "surfaceWinRateAdvantage",
    "surfaceSampleAdvantage",
    "recent5WinRateAdvantage",
    "recent10WinRateAdvantage",
    "recent20WinRateAdvantage",
    "recentSurface10WinRateAdvantage",
    "matchesLast30Advantage",
    "matchesLast60Advantage",
    "matchesLast90Advantage",
    "winRateLast30Advantage",
    "winRateLast60Advantage",
    "winRateLast90Advantage",
    "h2hAdvantage",
    "h2hSampleSize",
    "surfaceH2hAdvantage",
    "surfaceH2hSampleSize",
    "highLevelExperienceAdvantage",
    "top20WinRateAdvantage",
    "top20SampleAdvantage",
    "aceRateAdvantage",
    "doubleFaultRateAdvantage",
    "firstServeInRateAdvantage",
    "firstServeWonRateAdvantage",
    "secondServeWonRateAdvantage",
    "servicePointWonRateAdvantage",
    "returnPointWonRateAdvantage",
    "breakPointSaveRateAdvantage",
    "breakPointConvertRateAdvantage",
    "tiebreakWinRateAdvantage",
    "styleServeSampleAdvantage",
    "styleReturnSampleAdvantage",
    "tiebreakSampleAdvantage",
    "rankMissingAdvantage",
    "rankPointsMissingAdvantage",
    "ageMissingAdvantage",
    "heightMissingAdvantage",
    "surfaceHard",
    "surfaceClay",
    "surfaceGrass",
    "drawSize",
    "tournamentLevel",
]


@dataclass
class PlayerState:
    elo: float = INITIAL_ELO
    surface_elo: dict[str, float] = field(default_factory=lambda: defaultdict(lambda: INITIAL_ELO))
    overall_wins: int = 0
    overall_losses: int = 0
    surface_wins: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    surface_losses: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    high_level_matches: int = 0
    recent: deque[int] = field(default_factory=lambda: deque(maxlen=20))
    recent_by_surface: dict[str, deque[int]] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=20)))
    dated_results: deque[tuple[datetime, int]] = field(default_factory=deque)
    top20_wins: int = 0
    top20_losses: int = 0
    aces: float = 0.0
    double_faults: float = 0.0
    service_points: float = 0.0
    first_in: float = 0.0
    first_won: float = 0.0
    second_serves: float = 0.0
    second_won: float = 0.0
    service_points_won: float = 0.0
    break_points_saved: float = 0.0
    break_points_faced: float = 0.0
    return_points: float = 0.0
    return_points_won: float = 0.0
    break_points_return: float = 0.0
    break_points_converted: float = 0.0
    tiebreak_wins: int = 0
    tiebreak_losses: int = 0


def fetch_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=45) as response:
        return response.read().decode("utf-8", "replace")


def read_year(year: int) -> list[dict]:
    text = fetch_text(MATCH_URL.format(year=year))
    return list(csv.DictReader(io.StringIO(text)))


def clean_name(value: str) -> str:
    return " ".join((value or "").strip().split())


def parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y%m%d")


def normalize_surface(value: str) -> str | None:
    surface = (value or "").strip().lower()
    if surface in SURFACES:
        return surface
    if surface == "carpet":
        return "hard"
    return None


def is_played_main_draw(row: dict) -> bool:
    score = (row.get("score") or "").lower()
    round_name = (row.get("round") or "").upper()
    if "w/o" in score or "walkover" in score:
        return False
    if round_name in {"Q1", "Q2", "Q3", "Q4", "Q"}:
        return False
    return True


def raw_level(row: dict) -> str:
    return (row.get("tourney_level") or "").strip()


def state_level_from_row(row: dict) -> str | None:
    raw = raw_level(row)
    if raw == "I":
        return "250"
    if raw == "G":
        return "grand_slam"
    if raw == "F":
        return "finals"
    if raw == "PM":
        return "1000"
    if raw == "P":
        return "1000" if is_premier_1000_name(row.get("tourney_name", "")) else "500"
    return None


def target_level_from_row(row: dict) -> str | None:
    level = state_level_from_row(row)
    if level in {"500", "1000", "grand_slam", "finals"}:
        return level
    return None


def is_premier_1000_name(name: str) -> bool:
    normalized = name.lower()
    return any(
        token in normalized
        for token in [
            "indian wells",
            "miami",
            "madrid",
            "rome",
            "beijing",
            "wuhan",
            "doha",
            "dubai",
            "canada",
            "toronto",
            "montreal",
            "cincinnati",
        ]
    )


def to_float(value: str | None, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def value_with_missing(value: str | None, fallback: float) -> tuple[float, int]:
    if value in {None, ""}:
        return fallback, 1
    try:
        return float(value), 0
    except ValueError:
        return fallback, 1


def rank_for(row: dict, side: str) -> tuple[float, int]:
    return value_with_missing(row.get(f"{side}_rank"), MISSING_RANK)


def rank_points_for(row: dict, side: str) -> tuple[float, int]:
    return value_with_missing(row.get(f"{side}_rank_points"), 0.0)


def age_for(row: dict, side: str) -> tuple[float, int]:
    return value_with_missing(row.get(f"{side}_age"), NEUTRAL_AGE)


def height_for(row: dict, side: str) -> tuple[float, int]:
    return value_with_missing(row.get(f"{side}_ht"), NEUTRAL_HEIGHT)


def hand_left_for(row: dict, side: str) -> int:
    return 1 if (row.get(f"{side}_hand") or "").strip().upper() == "L" else 0


def smoothed_win_rate(wins: int, losses: int) -> float:
    return (wins + RECENT_PRIOR_WINS) / (wins + losses + RECENT_PRIOR_WINS + RECENT_PRIOR_LOSSES)


def deque_win_rate(results: deque[int], length: int) -> float:
    sample = list(results)[-length:]
    wins = sum(sample)
    losses = len(sample) - wins
    return smoothed_win_rate(wins, losses)


def deque_sample(results: deque[int], length: int) -> int:
    return min(len(results), length)


def dated_window(state: PlayerState, match_date: datetime, days: int) -> tuple[float, int]:
    sample = [result for date_value, result in state.dated_results if 0 < (match_date - date_value).days <= days]
    wins = sum(sample)
    losses = len(sample) - wins
    return smoothed_win_rate(wins, losses), len(sample)


def rate(numerator: float, denominator: float, name: str) -> float:
    prior_rate, prior_weight = STYLE_PRIORS[name]
    return (numerator + prior_rate * prior_weight) / (denominator + prior_weight)


def style_rates(state: PlayerState) -> dict[str, float]:
    return {
        "aceRate": rate(state.aces, state.service_points, "aceRate"),
        "doubleFaultRate": rate(state.double_faults, state.service_points, "doubleFaultRate"),
        "firstServeInRate": rate(state.first_in, state.service_points, "firstServeInRate"),
        "firstServeWonRate": rate(state.first_won, state.first_in, "firstServeWonRate"),
        "secondServeWonRate": rate(state.second_won, state.second_serves, "secondServeWonRate"),
        "servicePointWonRate": rate(state.service_points_won, state.service_points, "servicePointWonRate"),
        "returnPointWonRate": rate(state.return_points_won, state.return_points, "returnPointWonRate"),
        "breakPointSaveRate": rate(state.break_points_saved, state.break_points_faced, "breakPointSaveRate"),
        "breakPointConvertRate": rate(state.break_points_converted, state.break_points_return, "breakPointConvertRate"),
        "tiebreakWinRate": rate(state.tiebreak_wins, state.tiebreak_wins + state.tiebreak_losses, "tiebreakWinRate"),
    }


def h2h_record(h2h: dict[tuple[str, str], list[int]], player_a: str, player_b: str) -> tuple[int, int, int]:
    wins_a, wins_b = h2h[(player_a, player_b)]
    return wins_a, wins_b, wins_a + wins_b


def surface_h2h_record(surface_h2h: dict[tuple[str, str, str], list[int]], surface: str, player_a: str, player_b: str) -> tuple[int, int, int]:
    wins_a, wins_b = surface_h2h[(surface, player_a, player_b)]
    return wins_a, wins_b, wins_a + wins_b


def h2h_advantage(wins_a: int, wins_b: int, total: int) -> float:
    return ((wins_a - wins_b) / total) if total else 0.0


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


def surface_flags(surface: str) -> dict[str, int]:
    return {
        "surfaceHard": 1 if surface == "hard" else 0,
        "surfaceClay": 1 if surface == "clay" else 0,
        "surfaceGrass": 1 if surface == "grass" else 0,
    }


def tiebreak_counts_from_score(score: str) -> tuple[int, int]:
    winner_tbs = 0
    loser_tbs = 0
    for token in score.split():
        match = re.match(r"^(\d+)-(\d+)", token)
        if not match:
            continue
        winner_games = int(match.group(1))
        loser_games = int(match.group(2))
        if winner_games == 7 and loser_games == 6:
            winner_tbs += 1
        elif winner_games == 6 and loser_games == 7:
            loser_tbs += 1
    return winner_tbs, loser_tbs


def player_stats_from_row(row: dict, prefix: str, opponent_prefix: str) -> dict[str, float]:
    svpt = to_float(row.get(f"{prefix}_svpt"))
    first_in = to_float(row.get(f"{prefix}_1stIn"))
    first_won = to_float(row.get(f"{prefix}_1stWon"))
    second_won = to_float(row.get(f"{prefix}_2ndWon"))
    second_serves = max(svpt - first_in, 0)
    service_points_won = first_won + second_won
    bp_saved = to_float(row.get(f"{prefix}_bpSaved"))
    bp_faced = to_float(row.get(f"{prefix}_bpFaced"))

    opp_svpt = to_float(row.get(f"{opponent_prefix}_svpt"))
    opp_first_won = to_float(row.get(f"{opponent_prefix}_1stWon"))
    opp_second_won = to_float(row.get(f"{opponent_prefix}_2ndWon"))
    opp_service_points_won = opp_first_won + opp_second_won
    opp_bp_saved = to_float(row.get(f"{opponent_prefix}_bpSaved"))
    opp_bp_faced = to_float(row.get(f"{opponent_prefix}_bpFaced"))

    return {
        "aces": to_float(row.get(f"{prefix}_ace")),
        "double_faults": to_float(row.get(f"{prefix}_df")),
        "service_points": svpt,
        "first_in": first_in,
        "first_won": first_won,
        "second_serves": second_serves,
        "second_won": second_won,
        "service_points_won": service_points_won,
        "break_points_saved": bp_saved,
        "break_points_faced": bp_faced,
        "return_points": opp_svpt,
        "return_points_won": max(opp_svpt - opp_service_points_won, 0),
        "break_points_return": opp_bp_faced,
        "break_points_converted": max(opp_bp_faced - opp_bp_saved, 0),
    }


def apply_stats(state: PlayerState, stats: dict[str, float]) -> None:
    for name, value in stats.items():
        setattr(state, name, getattr(state, name) + value)


def feature_row(
    row: dict,
    player_a: str,
    player_b: str,
    result: int,
    states: dict[str, PlayerState],
    h2h: dict[tuple[str, str], list[int]],
    surface_h2h: dict[tuple[str, str, str], list[int]],
    surface: str,
    level: str,
    a_side: str,
    b_side: str,
) -> dict:
    state_a = states[player_a]
    state_b = states[player_b]
    match_date = parse_date(row["tourney_date"])

    rank_a, rank_missing_a = rank_for(row, a_side)
    rank_b, rank_missing_b = rank_for(row, b_side)
    rank_points_a, rank_points_missing_a = rank_points_for(row, a_side)
    rank_points_b, rank_points_missing_b = rank_points_for(row, b_side)
    age_a, age_missing_a = age_for(row, a_side)
    age_b, age_missing_b = age_for(row, b_side)
    height_a, height_missing_a = height_for(row, a_side)
    height_b, height_missing_b = height_for(row, b_side)
    left_a = hand_left_for(row, a_side)
    left_b = hand_left_for(row, b_side)

    overall_rate_a = smoothed_win_rate(state_a.overall_wins, state_a.overall_losses)
    overall_rate_b = smoothed_win_rate(state_b.overall_wins, state_b.overall_losses)
    surface_rate_a = smoothed_win_rate(state_a.surface_wins[surface], state_a.surface_losses[surface])
    surface_rate_b = smoothed_win_rate(state_b.surface_wins[surface], state_b.surface_losses[surface])

    last30_a, last30_n_a = dated_window(state_a, match_date, 30)
    last30_b, last30_n_b = dated_window(state_b, match_date, 30)
    last60_a, last60_n_a = dated_window(state_a, match_date, 60)
    last60_b, last60_n_b = dated_window(state_b, match_date, 60)
    last90_a, last90_n_a = dated_window(state_a, match_date, 90)
    last90_b, last90_n_b = dated_window(state_b, match_date, 90)

    h2h_wins_a, h2h_wins_b, h2h_total = h2h_record(h2h, player_a, player_b)
    surface_h2h_wins_a, surface_h2h_wins_b, surface_h2h_total = surface_h2h_record(surface_h2h, surface, player_a, player_b)
    style_a = style_rates(state_a)
    style_b = style_rates(state_b)

    top20_rate_a = smoothed_win_rate(state_a.top20_wins, state_a.top20_losses)
    top20_rate_b = smoothed_win_rate(state_b.top20_wins, state_b.top20_losses)

    row_out = {
        "rankAdvantage": rank_b - rank_a,
        "rankPointsAdvantage": rank_points_a - rank_points_b,
        "overallEloAdvantage": state_a.elo - state_b.elo,
        "hardEloAdvantage": state_a.surface_elo["hard"] - state_b.surface_elo["hard"],
        "clayEloAdvantage": state_a.surface_elo["clay"] - state_b.surface_elo["clay"],
        "grassEloAdvantage": state_a.surface_elo["grass"] - state_b.surface_elo["grass"],
        "matchSurfaceEloAdvantage": state_a.surface_elo[surface] - state_b.surface_elo[surface],
        "ageAdvantage": abs(age_b - NEUTRAL_AGE) - abs(age_a - NEUTRAL_AGE),
        "heightAdvantage": height_a - height_b,
        "leftyAdvantage": left_a - left_b,
        "sameHandedness": 1 if left_a == left_b else 0,
        "overallWinRateAdvantage": overall_rate_a - overall_rate_b,
        "overallSampleAdvantage": (state_a.overall_wins + state_a.overall_losses) - (state_b.overall_wins + state_b.overall_losses),
        "surfaceWinRateAdvantage": surface_rate_a - surface_rate_b,
        "surfaceSampleAdvantage": (state_a.surface_wins[surface] + state_a.surface_losses[surface]) - (state_b.surface_wins[surface] + state_b.surface_losses[surface]),
        "recent5WinRateAdvantage": deque_win_rate(state_a.recent, 5) - deque_win_rate(state_b.recent, 5),
        "recent10WinRateAdvantage": deque_win_rate(state_a.recent, 10) - deque_win_rate(state_b.recent, 10),
        "recent20WinRateAdvantage": deque_win_rate(state_a.recent, 20) - deque_win_rate(state_b.recent, 20),
        "recentSurface10WinRateAdvantage": deque_win_rate(state_a.recent_by_surface[surface], 10) - deque_win_rate(state_b.recent_by_surface[surface], 10),
        "matchesLast30Advantage": last30_n_a - last30_n_b,
        "matchesLast60Advantage": last60_n_a - last60_n_b,
        "matchesLast90Advantage": last90_n_a - last90_n_b,
        "winRateLast30Advantage": last30_a - last30_b,
        "winRateLast60Advantage": last60_a - last60_b,
        "winRateLast90Advantage": last90_a - last90_b,
        "h2hAdvantage": h2h_advantage(h2h_wins_a, h2h_wins_b, h2h_total),
        "h2hSampleSize": h2h_total,
        "surfaceH2hAdvantage": h2h_advantage(surface_h2h_wins_a, surface_h2h_wins_b, surface_h2h_total),
        "surfaceH2hSampleSize": surface_h2h_total,
        "highLevelExperienceAdvantage": state_a.high_level_matches - state_b.high_level_matches,
        "top20WinRateAdvantage": top20_rate_a - top20_rate_b,
        "top20SampleAdvantage": (state_a.top20_wins + state_a.top20_losses) - (state_b.top20_wins + state_b.top20_losses),
        "aceRateAdvantage": style_a["aceRate"] - style_b["aceRate"],
        "doubleFaultRateAdvantage": style_a["doubleFaultRate"] - style_b["doubleFaultRate"],
        "firstServeInRateAdvantage": style_a["firstServeInRate"] - style_b["firstServeInRate"],
        "firstServeWonRateAdvantage": style_a["firstServeWonRate"] - style_b["firstServeWonRate"],
        "secondServeWonRateAdvantage": style_a["secondServeWonRate"] - style_b["secondServeWonRate"],
        "servicePointWonRateAdvantage": style_a["servicePointWonRate"] - style_b["servicePointWonRate"],
        "returnPointWonRateAdvantage": style_a["returnPointWonRate"] - style_b["returnPointWonRate"],
        "breakPointSaveRateAdvantage": style_a["breakPointSaveRate"] - style_b["breakPointSaveRate"],
        "breakPointConvertRateAdvantage": style_a["breakPointConvertRate"] - style_b["breakPointConvertRate"],
        "tiebreakWinRateAdvantage": style_a["tiebreakWinRate"] - style_b["tiebreakWinRate"],
        "styleServeSampleAdvantage": state_a.service_points - state_b.service_points,
        "styleReturnSampleAdvantage": state_a.return_points - state_b.return_points,
        "tiebreakSampleAdvantage": (state_a.tiebreak_wins + state_a.tiebreak_losses) - (state_b.tiebreak_wins + state_b.tiebreak_losses),
        "rankMissingAdvantage": rank_missing_b - rank_missing_a,
        "rankPointsMissingAdvantage": rank_points_missing_b - rank_points_missing_a,
        "ageMissingAdvantage": age_missing_b - age_missing_a,
        "heightMissingAdvantage": height_missing_b - height_missing_a,
        "drawSize": to_float(row.get("draw_size")),
        "tournamentLevel": LEVEL_WEIGHTS[level],
        "result": result,
        "matchDate": row["tourney_date"],
        "tournament": row.get("tourney_name", ""),
        "surface": surface,
        "level": level,
        "round": row.get("round", ""),
        "playerA": player_a,
        "playerB": player_b,
        "rankA": rank_a,
        "rankB": rank_b,
        "rankPointsA": rank_points_a,
        "rankPointsB": rank_points_b,
        "ageA": age_a,
        "ageB": age_b,
        "heightA": height_a,
        "heightB": height_b,
        "overallEloA": state_a.elo,
        "overallEloB": state_b.elo,
        "matchSurfaceEloA": state_a.surface_elo[surface],
        "matchSurfaceEloB": state_b.surface_elo[surface],
        "h2hWinsA": h2h_wins_a,
        "h2hWinsB": h2h_wins_b,
        "surfaceH2hWinsA": surface_h2h_wins_a,
        "surfaceH2hWinsB": surface_h2h_wins_b,
    }
    row_out.update(surface_flags(surface))
    return row_out


def update_player_state(
    row: dict,
    states: dict[str, PlayerState],
    h2h: dict[tuple[str, str], list[int]],
    surface_h2h: dict[tuple[str, str, str], list[int]],
    surface: str,
    level: str,
) -> None:
    winner = clean_name(row.get("winner_name", ""))
    loser = clean_name(row.get("loser_name", ""))
    winner_state = states[winner]
    loser_state = states[loser]
    match_date = parse_date(row["tourney_date"])
    loser_rank, _ = rank_for(row, "loser")
    winner_rank, _ = rank_for(row, "winner")

    update_elo(winner_state, loser_state, surface)
    winner_state.overall_wins += 1
    loser_state.overall_losses += 1
    winner_state.surface_wins[surface] += 1
    loser_state.surface_losses[surface] += 1
    winner_state.recent.append(1)
    loser_state.recent.append(0)
    winner_state.recent_by_surface[surface].append(1)
    loser_state.recent_by_surface[surface].append(0)
    winner_state.dated_results.append((match_date, 1))
    loser_state.dated_results.append((match_date, 0))
    if level in {"500", "1000", "grand_slam", "finals"}:
        winner_state.high_level_matches += 1
        loser_state.high_level_matches += 1
    if loser_rank <= 20:
        winner_state.top20_wins += 1
    if winner_rank <= 20:
        loser_state.top20_losses += 1

    h2h[(winner, loser)][0] += 1
    h2h[(loser, winner)][1] += 1
    surface_h2h[(surface, winner, loser)][0] += 1
    surface_h2h[(surface, loser, winner)][1] += 1

    winner_stats = player_stats_from_row(row, "w", "l")
    loser_stats = player_stats_from_row(row, "l", "w")
    if winner_stats["service_points"] > 0:
        apply_stats(winner_state, winner_stats)
    if loser_stats["service_points"] > 0:
        apply_stats(loser_state, loser_stats)

    winner_tbs, loser_tbs = tiebreak_counts_from_score(row.get("score", ""))
    winner_state.tiebreak_wins += winner_tbs
    winner_state.tiebreak_losses += loser_tbs
    loser_state.tiebreak_wins += loser_tbs
    loser_state.tiebreak_losses += winner_tbs


def build_rows(state_start_year: int, output_start_year: int, output_end_year: int) -> list[dict]:
    states: dict[str, PlayerState] = defaultdict(PlayerState)
    h2h: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    surface_h2h: dict[tuple[str, str, str], list[int]] = defaultdict(lambda: [0, 0])
    raw_rows: list[dict] = []
    output_rows: list[dict] = []

    for year in range(state_start_year, output_end_year + 1):
        try:
            raw_rows.extend(read_year(year))
        except Exception as exc:
            print(f"Warning: could not fetch {year}: {exc}", file=sys.stderr)

    raw_rows.sort(key=lambda row: (row.get("tourney_date", ""), row.get("match_num", "")))
    random.seed(7)

    for row in raw_rows:
        surface = normalize_surface(row.get("surface", ""))
        target_level = target_level_from_row(row)
        if not surface or not target_level or not is_played_main_draw(row):
            continue

        winner = clean_name(row.get("winner_name", ""))
        loser = clean_name(row.get("loser_name", ""))
        if not winner or not loser:
            continue

        match_year = int(row["tourney_date"][:4])
        if output_start_year <= match_year <= output_end_year:
            flip = random.random() < 0.5
            if flip:
                output_rows.append(
                    feature_row(row, loser, winner, 0, states, h2h, surface_h2h, surface, target_level, "loser", "winner")
                )
            else:
                output_rows.append(
                    feature_row(row, winner, loser, 1, states, h2h, surface_h2h, surface, target_level, "winner", "loser")
                )

        update_player_state(row, states, h2h, surface_h2h, surface, target_level)

    return output_rows


def write_rows(path: Path, rows: list[dict]) -> None:
    metadata = [
        "result",
        "matchDate",
        "tournament",
        "surface",
        "level",
        "round",
        "playerA",
        "playerB",
        "rankA",
        "rankB",
        "rankPointsA",
        "rankPointsB",
        "ageA",
        "ageB",
        "heightA",
        "heightB",
        "overallEloA",
        "overallEloB",
        "matchSurfaceEloA",
        "matchSurfaceEloB",
        "h2hWinsA",
        "h2hWinsB",
        "surfaceH2hWinsA",
        "surfaceH2hWinsB",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FEATURES + metadata)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-start-year", type=int, default=2010)
    parser.add_argument("--output-start-year", type=int, default=2010)
    parser.add_argument("--output-end-year", type=int, default=2025)
    parser.add_argument("--output", default="data/training_rows_detailed.csv")
    args = parser.parse_args()

    rows = build_rows(args.state_start_year, args.output_start_year, args.output_end_year)
    write_rows(Path(args.output), rows)
    print(f"Wrote {len(rows)} detailed rows to {args.output}")
    print(f"State range: {args.state_start_year}-{args.output_end_year}")
    print(f"Output: {args.output_start_year}-{args.output_end_year}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
