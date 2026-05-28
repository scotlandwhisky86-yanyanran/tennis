const ELO_URL = "https://www.tennisabstract.com/reports/wta_elo_ratings.html";

function cellText(value) {
  return String(value || "")
    .replace(/<[^>]+>/g, "")
    .replace(/&nbsp;/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function toNumber(value, fallback = 0) {
  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function toInt(value, fallback = 9999) {
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function normalizeSearch(value) {
  return String(value || "")
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]/g, "");
}

function aliasesFor(name) {
  const parts = name.split(/\s+/).filter(Boolean);
  const aliases = [name];
  if (parts.length === 2) {
    aliases.push(`${parts[1]} ${parts[0]}`);
  }
  if (parts.length > 2) {
    aliases.push(`${parts.at(-1)} ${parts.slice(0, -1).join(" ")}`);
    aliases.push(`${parts.slice(-2).join(" ")} ${parts.slice(0, -2).join(" ")}`);
  }
  return aliases;
}

function levenshtein(a, b) {
  const previous = Array.from({ length: b.length + 1 }, (_, index) => index);
  const current = Array(b.length + 1).fill(0);

  for (let i = 1; i <= a.length; i += 1) {
    current[0] = i;
    for (let j = 1; j <= b.length; j += 1) {
      current[j] = Math.min(
        previous[j] + 1,
        current[j - 1] + 1,
        previous[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1),
      );
    }
    previous.splice(0, previous.length, ...current);
  }

  return previous[b.length];
}

function similarity(a, b) {
  return 1 - levenshtein(a, b) / Math.max(a.length, b.length, 1);
}

function parsePlayers(html) {
  const players = [];
  for (const row of html.matchAll(/<tr>([\s\S]*?)<\/tr>/gi)) {
    const cells = [...row[1].matchAll(/<td[^>]*>([\s\S]*?)<\/td>/gi)].map((match) => cellText(match[1]));
    if (cells.length < 16 || !cells[1]) continue;

    players.push({
      name: cells[1],
      rank: toInt(cells[15], toInt(cells[0])),
      age: toNumber(cells[2]),
      elo: Math.round(toNumber(cells[3]) * 10) / 10,
      hardElo: Math.round(toNumber(cells[6]) * 10) / 10,
      clayElo: Math.round(toNumber(cells[8]) * 10) / 10,
      grassElo: Math.round(toNumber(cells[10]) * 10) / 10,
    });
  }
  return players;
}

function findPlayer(players, query) {
  const key = normalizeSearch(query);
  let best = null;
  let secondScore = 0;

  for (const player of players) {
    for (const alias of aliasesFor(player.name)) {
      const aliasKey = normalizeSearch(alias);
      if (!aliasKey) continue;
      const score = aliasKey === key ? 1 : similarity(key, aliasKey);
      if (!best || score > best.score) {
        secondScore = best?.score || 0;
        best = { player, score };
      } else if (score > secondScore) {
        secondScore = score;
      }
    }
  }

  if (best && best.score >= 0.84 && best.score - secondScore >= 0.025) {
    return best.player;
  }
  return null;
}

module.exports = async function handler(req, res) {
  const url = new URL(req.url, `http://${req.headers.host || "localhost"}`);
  const query = url.searchParams.get("name") || "";

  if (!query.trim()) {
    res.statusCode = 400;
    res.setHeader("content-type", "application/json");
    res.end(JSON.stringify({ error: "Missing name." }));
    return;
  }

  try {
    const response = await fetch(ELO_URL, {
      headers: { "user-agent": "Mozilla/5.0 (compatible; WTA-Match-Predictor/0.1)" },
    });
    if (!response.ok) throw new Error(`Tennis Abstract returned ${response.status}`);

    const html = await response.text();
    const player = findPlayer(parsePlayers(html), query);
    res.setHeader("cache-control", "s-maxage=86400, stale-while-revalidate=604800");
    res.setHeader("content-type", "application/json");

    if (!player) {
      res.statusCode = 404;
      res.end(JSON.stringify({ error: "Player not found." }));
      return;
    }

    res.end(JSON.stringify({ player }));
  } catch (error) {
    res.statusCode = 502;
    res.setHeader("content-type", "application/json");
    res.end(JSON.stringify({ error: error.message }));
  }
};
