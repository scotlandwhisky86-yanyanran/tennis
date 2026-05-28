from __future__ import annotations

import argparse
import json
import re
import urllib.parse
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ELO_URL = "https://www.tennisabstract.com/reports/wta_elo_ratings.html"
USER_AGENT = "Mozilla/5.0 (compatible; WTA-Match-Predictor/0.1; +https://localhost)"


def cell_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", value).replace("&nbsp;", " ")).strip()


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


def normalize_search(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def aliases_for(name: str) -> list[str]:
    parts = name.split()
    aliases = [name]
    if len(parts) == 2:
        aliases.append(f"{parts[1]} {parts[0]}")
    if len(parts) > 2:
        aliases.append(f"{parts[-1]} {' '.join(parts[:-1])}")
        aliases.append(f"{' '.join(parts[-2:])} {' '.join(parts[:-2])}")
    return aliases


def levenshtein(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    current = [0] * (len(b) + 1)
    for i, char_a in enumerate(a, start=1):
        current[0] = i
        for j, char_b in enumerate(b, start=1):
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (0 if char_a == char_b else 1),
            )
        previous, current = current[:], previous
    return previous[len(b)]


def similarity(a: str, b: str) -> float:
    return 1 - levenshtein(a, b) / max(len(a), len(b), 1)


def fetch_elo_html() -> str:
    request = urllib.request.Request(ELO_URL, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=45) as response:
        return response.read().decode("utf-8", "replace")


def parse_players(html: str) -> list[dict]:
    players: list[dict] = []
    for row_match in re.finditer(r"<tr>(.*?)</tr>", html, flags=re.DOTALL | re.IGNORECASE):
        cells = [
            cell_text(cell.group(1))
            for cell in re.finditer(r"<td[^>]*>(.*?)</td>", row_match.group(1), flags=re.DOTALL | re.IGNORECASE)
        ]
        if len(cells) < 16 or not cells[1]:
            continue
        players.append(
            {
                "name": cells[1],
                "rank": to_int(cells[15], to_int(cells[0])),
                "age": round(to_float(cells[2]), 1),
                "elo": round(to_float(cells[3]), 1),
                "hardElo": round(to_float(cells[6]), 1),
                "clayElo": round(to_float(cells[8]), 1),
                "grassElo": round(to_float(cells[10]), 1),
            }
        )
    return players


def find_player(players: list[dict], query: str) -> dict | None:
    key = normalize_search(query)
    best: tuple[dict, float] | None = None
    second_score = 0.0
    for player in players:
        for alias in aliases_for(player["name"]):
            alias_key = normalize_search(alias)
            score = 1.0 if alias_key == key else similarity(key, alias_key)
            if best is None or score > best[1]:
                second_score = best[1] if best else 0.0
                best = (player, score)
            elif score > second_score:
                second_score = score

    if best and best[1] >= 0.84 and best[1] - second_score >= 0.025:
        return best[0]
    return None


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/player-lookup":
            self.handle_player_lookup(parsed)
            return
        super().do_GET()

    def handle_player_lookup(self, parsed: urllib.parse.ParseResult) -> None:
        params = urllib.parse.parse_qs(parsed.query)
        query = (params.get("name") or [""])[0].strip()
        if not query:
            self.send_json({"error": "Missing name."}, 400)
            return

        try:
            player = find_player(parse_players(fetch_elo_html()), query)
        except Exception as exc:  # noqa: BLE001
            self.send_json({"error": str(exc)}, 502)
            return

        if not player:
            self.send_json({"error": "Player not found."}, 404)
            return
        self.send_json({"player": player}, 200)

    def send_json(self, payload: dict, status: int) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5173)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving {ROOT} at http://{args.host}:{args.port}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
