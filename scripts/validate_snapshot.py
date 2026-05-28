"""
Validate data/snapshot.json before using it for predictions.

This catches the two failures that would hurt trust most:
1. missing player fields
2. stale current-data snapshots
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path


REQUIRED_PLAYER_FIELDS = {
    "name",
    "rank",
    "age",
    "elo",
    "hardElo",
    "clayElo",
    "grassElo",
    "recent",
    "h2h",
}

MAX_ALLOWED_AGE_DAYS = 15


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/snapshot.json")
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []

    meta = snapshot.get("meta", {})
    try:
        updated_at = parse_date(meta.get("updatedAt", ""))
        max_age_days = int(meta.get("maxAgeDays", MAX_ALLOWED_AGE_DAYS))
        if max_age_days > MAX_ALLOWED_AGE_DAYS:
            errors.append(f"meta.maxAgeDays cannot exceed {MAX_ALLOWED_AGE_DAYS}.")
        age_days = (date.today() - updated_at).days
        if age_days > max_age_days:
            errors.append(f"Snapshot is stale: {age_days} days old, max is {max_age_days}.")
    except ValueError:
        errors.append("meta.updatedAt must use YYYY-MM-DD.")

    players = snapshot.get("players")
    if not isinstance(players, list) or len(players) < 2:
        errors.append("snapshot.players must contain at least two players.")
    else:
        seen_names: set[str] = set()
        for index, player in enumerate(players, start=1):
            missing = REQUIRED_PLAYER_FIELDS - set(player)
            if missing:
                errors.append(f"Player #{index} is missing: {', '.join(sorted(missing))}.")

            name = str(player.get("name", "")).strip()
            if not name:
                errors.append(f"Player #{index} needs a name.")
            elif name.lower() in seen_names:
                errors.append(f"Duplicate player name: {name}.")
            seen_names.add(name.lower())

            recent = player.get("recent", {})
            if not {"wins", "losses"} <= set(recent):
                errors.append(f"{name or f'Player #{index}'} needs recent wins/losses.")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print(f"Snapshot OK: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
