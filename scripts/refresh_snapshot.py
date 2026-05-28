"""
Refresh data/snapshot.json from public tennis data sources.

Sources:
- Tennis Abstract WTA Elo report for current ranking, age, Elo, and surface Elo.
- Jeff Sackmann WTA match CSVs for recent form and head-to-head.

This script intentionally uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import urllib.request
from collections import defaultdict
from datetime import date, datetime, timedelta
from html import unescape
from pathlib import Path


ELO_URL = "https://www.tennisabstract.com/reports/wta_elo_ratings.html"
MATCH_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_{year}.csv"
USER_AGENT = "Mozilla/5.0 (compatible; WTA-Match-Predictor/0.1; +https://localhost)"


def fetch_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=45) as response:
        return response.read().decode("utf-8", "replace")


def cell_text(value: str) -> str:
    text = re.sub(r"<[^>]+>", "", value)
    return unescape(text).replace("\xa0", " ").strip()


def to_float(value: str, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def to_int(value: str, fallback: int = 9999) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return fallback


def parse_elo_page(html: str, player_limit: int) -> tuple[date, list[dict]]:
    update_match = re.search(r"Last update:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})", html)
    if not update_match:
        raise ValueError("Could not find Tennis Abstract last-update date.")

    updated_at = datetime.strptime(update_match.group(1), "%Y-%m-%d").date()
    players: list[dict] = []

    for row_html in re.findall(r"<tr>(.*?)</tr>", html, flags=re.DOTALL | re.IGNORECASE):
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row_html, flags=re.DOTALL | re.IGNORECASE)
        if len(cells) < 16:
            continue

        values = [cell_text(cell) for cell in cells]
        name = values[1]
        if not name:
            continue

        player = {
            "name": name,
            "rank": to_int(values[15], to_int(values[0])),
            "age": to_float(values[2]),
            "elo": round(to_float(values[3]), 1),
            "hardElo": round(to_float(values[6]), 1),
            "clayElo": round(to_float(values[8]), 1),
            "grassElo": round(to_float(values[10]), 1),
            "recent": {"wins": 0, "losses": 0},
            "h2h": {},
        }
        players.append(player)

    if not players:
        raise ValueError("Could not parse any players from Tennis Abstract Elo page.")

    ranked_players = [player for player in players if player["rank"] < 9999]
    ranked_players.sort(key=lambda player: (player["rank"], player["name"]))
    return updated_at, ranked_players[:player_limit]


def normalize_name(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().casefold())


def parse_match_date(value: str) -> date | None:
    try:
        return datetime.strptime(value, "%Y%m%d").date()
    except (TypeError, ValueError):
        return None


def high_level_from_row(row: dict) -> bool:
    raw = (row.get("tourney_level") or "").strip()
    return raw in {"G", "F", "P", "PM"}


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
    if raw == "I":
        return "250"
    return None


def normalize_surface(value: str) -> str | None:
    surface = (value or "").strip().lower()
    if surface in {"hard", "clay", "grass"}:
        return surface
    return None


def read_match_rows(year: int) -> list[dict]:
    text = fetch_text(MATCH_URL.format(year=year))
    reader = csv.DictReader(io.StringIO(text))
    return list(reader)


def add_match_signals(
    players: list[dict],
    as_of: date,
    recent_window_days: int,
    h2h_start_year: int,
) -> None:
    by_name = {normalize_name(player["name"]): player for player in players}
    recent_since = as_of - timedelta(days=recent_window_days)
    recent = defaultdict(lambda: {"wins": 0, "losses": 0})
    high_level_matches = defaultdict(int)
    surface_records = defaultdict(lambda: defaultdict(lambda: {"wins": 0, "losses": 0}))
    level_records = defaultdict(lambda: defaultdict(lambda: {"wins": 0, "losses": 0}))
    h2h = defaultdict(lambda: defaultdict(lambda: {"wins": 0, "losses": 0, "total": 0}))

    for year in range(h2h_start_year, as_of.year + 1):
        try:
            rows = read_match_rows(year)
        except Exception as exc:
            print(f"Warning: could not fetch {year} matches: {exc}", file=sys.stderr)
            continue

        for row in rows:
            match_date = parse_match_date(row.get("tourney_date", ""))
            if not match_date or match_date > as_of:
                continue

            winner_key = normalize_name(row.get("winner_name", ""))
            loser_key = normalize_name(row.get("loser_name", ""))
            if winner_key not in by_name and loser_key not in by_name:
                continue

            if match_date >= recent_since:
                if winner_key in by_name:
                    recent[winner_key]["wins"] += 1
                if loser_key in by_name:
                    recent[loser_key]["losses"] += 1

            if high_level_from_row(row):
                if winner_key in by_name:
                    high_level_matches[winner_key] += 1
                if loser_key in by_name:
                    high_level_matches[loser_key] += 1

                surface = normalize_surface(row.get("surface", ""))
                if surface:
                    if winner_key in by_name:
                        surface_records[winner_key][surface]["wins"] += 1
                    if loser_key in by_name:
                        surface_records[loser_key][surface]["losses"] += 1

            level = level_from_row(row)
            if level:
                if winner_key in by_name:
                    level_records[winner_key][level]["wins"] += 1
                if loser_key in by_name:
                    level_records[loser_key][level]["losses"] += 1

            if winner_key in by_name and loser_key in by_name:
                winner_name = by_name[winner_key]["name"]
                loser_name = by_name[loser_key]["name"]
                h2h[winner_name][loser_name]["wins"] += 1
                h2h[winner_name][loser_name]["total"] += 1
                h2h[loser_name][winner_name]["losses"] += 1
                h2h[loser_name][winner_name]["total"] += 1

    for player in players:
        key = normalize_name(player["name"])
        player["recent"] = dict(recent[key])
        player["highLevelMatches"] = high_level_matches[key]
        player["surfaceRecords"] = {
            surface: dict(surface_records[key][surface])
            for surface in ["hard", "clay", "grass"]
        }
        player["levelRecords"] = {
            level: dict(level_records[key][level])
            for level in ["250", "500", "1000", "grand_slam", "finals"]
        }
        player["rankMissing"] = 0 if player["rank"] < 9999 else 1
        player["ageMissing"] = 0 if player["age"] else 1
        player["h2h"] = {
            opponent: record
            for opponent, record in sorted(h2h[player["name"]].items())
            if record["total"] > 0
        }


def build_snapshot(args: argparse.Namespace) -> dict:
    html = fetch_text(ELO_URL)
    updated_at, players = parse_elo_page(html, args.player_limit)
    generated_at = date.today()
    add_match_signals(players, updated_at, args.recent_window_days, args.h2h_start_year)

    return {
        "meta": {
            "label": "Public data snapshot",
            "updatedAt": updated_at.isoformat(),
            "generatedAt": generated_at.isoformat(),
            "maxAgeDays": 15,
            "isDemo": False,
            "recentWindowDays": args.recent_window_days,
            "refreshCadence": "twice weekly via GitHub Actions",
            "sources": [
                {
                    "name": "Tennis Abstract WTA Elo ratings",
                    "url": ELO_URL,
                },
                {
                    "name": "Jeff Sackmann yearly WTA match CSVs",
                    "urlTemplate": MATCH_URL,
                    "years": f"{args.h2h_start_year}-{updated_at.year}",
                },
            ],
            "licenseNote": "Jeff Sackmann tennis_wta data is licensed for non-commercial use with attribution.",
        },
        "players": players,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/snapshot.json")
    parser.add_argument("--player-limit", type=int, default=250)
    parser.add_argument("--recent-window-days", type=int, default=120)
    parser.add_argument("--h2h-start-year", type=int, default=2000)
    args = parser.parse_args()

    snapshot = build_snapshot(args)
    output = Path(args.output)
    output.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    print(f"Wrote {output} with {len(snapshot['players'])} players.")
    print(f"Snapshot date: {snapshot['meta']['updatedAt']} (max age: 15 days)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
