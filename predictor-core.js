export async function fetchJson(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`${path} returned ${response.status}`);
  }
  return response.json();
}

export function sortPlayersByName(players) {
  return [...players].sort((a, b) => a.name.localeCompare(b.name, "en"));
}

export function findPlayer(players, name) {
  const normalized = normalizeName(name);
  return players.find((player) => normalizeName(player.name) === normalized);
}

export function predictMatch(model, playerA, playerB, tournament) {
  const features = buildFeatures(model, playerA, playerB, tournament);
  const reverseFeatures = buildFeatures(model, playerB, playerA, tournament);
  const forwardProbability = rawProbability(model, features);
  const reverseProbability = rawProbability(model, reverseFeatures);

  const probabilityA = clamp((forwardProbability + (1 - reverseProbability)) / 2, 0.03, 0.97);
  const probabilityB = 1 - probabilityA;
  const favorite = probabilityA >= probabilityB ? playerA : playerB;

  return {
    playerA,
    playerB,
    tournament,
    features,
    probabilityA,
    probabilityB,
    favorite,
  };
}

function rawProbability(model, features) {
  const score = Object.entries(features).reduce((total, [name, value]) => {
    return total + (model.coefficients[name] || 0) * normalizeFeature(model, name, value);
  }, model.intercept || 0);

  return sigmoid(score);
}

export function buildFeatures(model, playerA, playerB, tournament) {
  const surfaceKey = `${tournament.modelSurface}Elo`;
  const surfaceRecordA = surfaceRecord(playerA, tournament.modelSurface);
  const surfaceRecordB = surfaceRecord(playerB, tournament.modelSurface);
  const levelRecordA = levelRecord(playerA, tournament.level);
  const levelRecordB = levelRecord(playerB, tournament.level);
  const h2h = getH2h(playerA, playerB);
  const recentA = recentRate(playerA);
  const recentB = recentRate(playerB);
  const levelWeight = model.levelWeights[tournament.level] || 0;

  return {
    rankAdvantage: safeNumber(playerB.rank - playerA.rank),
    eloAdvantage: safeNumber(playerA.elo - playerB.elo),
    surfaceEloAdvantage: safeNumber((playerA[surfaceKey] || playerA.elo) - (playerB[surfaceKey] || playerB.elo)),
    surfaceWinRateAdvantage: safeNumber(smoothedWinPct(surfaceRecordA) - smoothedWinPct(surfaceRecordB)),
    surfaceMatchSampleAdvantage: safeNumber(recordSample(surfaceRecordA) - recordSample(surfaceRecordB)),
    levelWinRateAdvantage: safeNumber(smoothedWinPct(levelRecordA) - smoothedWinPct(levelRecordB)),
    levelMatchSampleAdvantage: safeNumber(recordSample(levelRecordA) - recordSample(levelRecordB)),
    recentFormAdvantage: safeNumber(recentA - recentB),
    recentSampleAdvantage: safeNumber(recentSampleForModel(playerA) - recentSampleForModel(playerB)),
    h2hAdvantage: safeNumber(h2h.total ? (h2h.wins - h2h.losses) / h2h.total : 0),
    h2hSampleSize: safeNumber(h2h.total),
    ageAdvantage: safeNumber(Math.abs(playerB.age - 27) - Math.abs(playerA.age - 27)),
    experienceAdvantage: safeNumber((playerA.highLevelMatches || 0) - (playerB.highLevelMatches || 0)),
    rankMissingAdvantage: safeNumber((playerB.rankMissing || 0) - (playerA.rankMissing || 0)),
    ageMissingAdvantage: safeNumber((playerB.ageMissing || 0) - (playerA.ageMissing || 0)),
    tournamentLevel: levelWeight,
  };
}

function surfaceRecord(player, surface) {
  return player.surfaceRecords?.[surface] || { wins: 0, losses: 0 };
}

function levelRecord(player, level) {
  return player.levelRecords?.[level] || { wins: 0, losses: 0 };
}

export function getH2h(playerA, playerB) {
  const direct = playerA.h2h?.[playerB.name];
  if (direct) return direct;

  const reverse = playerB.h2h?.[playerA.name];
  if (reverse) {
    return {
      wins: reverse.losses,
      losses: reverse.wins,
      total: reverse.total,
    };
  }

  return { wins: 0, losses: 0, total: 0 };
}

function smoothedWinPct(record) {
  const wins = record?.wins || 0;
  const losses = record?.losses || 0;
  return (wins + 2) / (wins + losses + 4);
}

function recordSample(record) {
  return (record?.wins || 0) + (record?.losses || 0);
}

export function normalizeName(name) {
  return String(name || "")
    .trim()
    .toLowerCase()
    .replace(/\s+/g, " ");
}

export function getFreshness(snapshot) {
  const maxAgeDays = snapshot.meta?.maxAgeDays ?? 15;
  const updatedAt = Date.parse(snapshot.meta?.updatedAt || "");

  if (!Number.isFinite(updatedAt)) {
    return { isStale: true, ageDays: "unknown" };
  }

  const ageMs = Date.now() - updatedAt;
  const ageDays = Math.max(0, Math.floor(ageMs / 86400000));
  return {
    ageDays,
    isStale: ageDays > maxAgeDays,
  };
}

export function recordText(record) {
  return `${record?.wins || 0}-${record?.losses || 0}`;
}

export function signed(value) {
  return value > 0 ? `+${Math.round(value)}` : `${Math.round(value)}`;
}

export function formatLevel(level) {
  const names = {
    grand_slam: "Grand Slam",
    "1000": "WTA 1000",
    "500": "WTA 500",
    "250": "WTA 250",
    finals: "WTA Finals",
  };
  return names[level] || level;
}

export function capitalize(value) {
  return String(value || "").charAt(0).toUpperCase() + String(value || "").slice(1);
}

export function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function normalizeFeature(model, name, value) {
  const stats = model.featureStats?.[name];
  if (!stats) return value;
  return (value - stats.mean) / (stats.std || 1);
}

function recentSample(player) {
  const record = player.recent || {};
  return (record.wins || 0) + (record.losses || 0);
}

function recentSampleForModel(player) {
  return Math.min(recentSample(player), 10);
}

function recentRate(player) {
  const record = player.recent || {};
  const total = recentSample(player);
  if (!total) return 0.5;

  const modelSample = recentSampleForModel(player);
  const winRate = (record.wins || 0) / total;
  const modelWins = winRate * modelSample;
  return (modelWins + 2) / (modelSample + 4);
}

function safeNumber(value) {
  return Number.isFinite(value) ? value : 0;
}

function sigmoid(value) {
  return 1 / (1 + Math.exp(-value));
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}
