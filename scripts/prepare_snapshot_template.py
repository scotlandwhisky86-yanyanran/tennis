"""
Create a blank player snapshot template.

This is intentionally not a scraper. For V1 we keep the public-data refresh step
explicit: collect current ranking/Elo/recent-form/H2H from chosen public sources,
normalize the names, then write data/snapshot.json.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path


def main() -> None:
    snapshot = {
        "meta": {
            "label": "Manual public-data snapshot",
            "updatedAt": date.today().isoformat(),
            "isDemo": False,
            "sources": [],
        },
        "players": [
            {
                "name": "",
                "rank": 0,
                "age": 0,
                "elo": 0,
                "hardElo": 0,
                "clayElo": 0,
                "grassElo": 0,
                "recent": {"wins": 0, "losses": 0},
                "h2h": {},
            }
        ],
    }
    Path("data/snapshot.template.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    print("Wrote data/snapshot.template.json")


if __name__ == "__main__":
    main()
