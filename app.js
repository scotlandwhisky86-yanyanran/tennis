import {
  capitalize,
  escapeHtml,
  fetchJson,
  findPlayer,
  formatLevel,
  getFreshness,
  getH2h,
  normalizeName,
  predictMatch,
  recordText,
  sortPlayersByName,
} from "./predictor-core.js";

const state = {
  snapshot: null,
  model: null,
  tournaments: [],
  players: [],
};

const elements = {
  form: document.querySelector("#predict-form"),
  playerA: document.querySelector("#player-a"),
  playerB: document.querySelector("#player-b"),
  surfaceFilter: document.querySelector("#surface-filter"),
  levelFilter: document.querySelector("#level-filter"),
  tournament: document.querySelector("#tournament"),
  result: document.querySelector("#result-panel"),
  status: document.querySelector("#snapshot-status"),
  playerPoolStatus: document.querySelector("#player-pool-status"),
  datalist: document.querySelector("#players-list"),
  playerAMenu: document.querySelector("#player-a-menu"),
  playerBMenu: document.querySelector("#player-b-menu"),
};

async function boot() {
  try {
    const [snapshot, model, tournamentsData] = await Promise.all([
      fetchJson("./data/snapshot.json"),
      fetchJson("./data/model.json"),
      fetchJson("./data/tournaments.json"),
    ]);

    state.snapshot = snapshot;
    state.model = model;
    state.tournaments = tournamentsData.tournaments || [];

    hydratePlayers(snapshot.players || []);
    hydrateTournamentFilters();
    updateStatus(snapshot);

    elements.playerA.value = "Iga Swiatek";
    elements.playerB.value = "Aryna Sabalenka";
  } catch (error) {
    elements.status.textContent = "Data unavailable";
    elements.result.innerHTML = `<p class="warning">Could not load model data. ${escapeHtml(error.message)}</p>`;
  }
}

function hydratePlayers(players) {
  state.players = sortPlayersByName(players);
  elements.playerPoolStatus.textContent = `${state.players.length} active top players loaded - click a field or type to search`;
  if (elements.datalist) {
    elements.datalist.innerHTML = state.players
      .map((player) => `<option value="${escapeHtml(player.name)}"></option>`)
      .join("");
  }
  setupPlayerPicker(elements.playerA, elements.playerAMenu);
  setupPlayerPicker(elements.playerB, elements.playerBMenu);
}

function setupPlayerPicker(input, menu) {
  if (!input || !menu) return;

  input.addEventListener("focus", () => openPlayerMenu(input, menu));
  input.addEventListener("click", () => openPlayerMenu(input, menu));
  input.addEventListener("input", () => openPlayerMenu(input, menu));
  input.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closePlayerMenu(input, menu);
  });
}

document.addEventListener("click", (event) => {
  if (!event.target.closest(".player-combobox")) {
    closePlayerMenu(elements.playerA, elements.playerAMenu);
    closePlayerMenu(elements.playerB, elements.playerBMenu);
  }
});

function openPlayerMenu(input, menu) {
  if (!input || !menu) return;

  renderPlayerMenu(input, menu);
  menu.hidden = false;
  input.setAttribute("aria-expanded", "true");
}

function closePlayerMenu(input, menu) {
  if (!input || !menu) return;

  menu.hidden = true;
  input.setAttribute("aria-expanded", "false");
}

function renderPlayerMenu(input, menu) {
  if (!input || !menu) return;

  const query = normalizeName(input.value);
  const matches = state.players
    .filter((player) => !query || normalizeName(player.name).includes(query))
    .slice(0, 200);

  if (!matches.length) {
    menu.innerHTML = `<div class="player-empty">No players found</div>`;
    return;
  }

  menu.innerHTML = matches
    .map((player) => `
      <button class="player-option" type="button" role="option" data-player="${escapeHtml(player.name)}">
        <span>${escapeHtml(player.name)}</span>
        <span class="player-rank">#${escapeHtml(player.rank)}</span>
      </button>
    `)
    .join("");

  menu.querySelectorAll(".player-option").forEach((option) => {
    option.addEventListener("mousedown", (event) => {
      event.preventDefault();
      input.value = option.dataset.player;
      closePlayerMenu(input, menu);
    });
  });
}

function hydrateTournamentFilters() {
  renderTournamentOptions();
  elements.surfaceFilter.addEventListener("change", renderTournamentOptions);
  elements.levelFilter.addEventListener("change", renderTournamentOptions);
}

function renderTournamentOptions() {
  const surface = elements.surfaceFilter.value;
  const level = elements.levelFilter.value;
  const tournaments = filteredTournaments(surface, level);

  elements.tournament.innerHTML = tournaments.length
    ? tournaments.map(tournamentOption).join("")
    : `<option value="">No matching tournaments</option>`;
}

function filteredTournaments(surface, level) {
  return state.tournaments
    .filter((tournament) => surface === "all" || tournament.modelSurface === surface)
    .filter((tournament) => level === "all" || tournament.level === level)
    .sort((a, b) => a.startDate.localeCompare(b.startDate));
}

function tournamentOption(tournament) {
  return `<option value="${escapeHtml(tournament.id)}">${escapeHtml(tournament.name)}</option>`;
}

function selectedTournament() {
  return state.tournaments.find((tournament) => tournament.id === elements.tournament.value);
}

function updateStatus(snapshot) {
  const label = snapshot.meta?.label || "Snapshot";
  const date = snapshot.meta?.updatedAt || "unknown date";
  const freshness = getFreshness(snapshot);
  elements.status.textContent = `${label} - ${date}${freshness.isStale ? " - stale" : ""}`;
  elements.status.classList.toggle("is-stale", freshness.isStale);
}

elements.form.addEventListener("submit", (event) => {
  event.preventDefault();
  if (!state.snapshot || !state.model) return;

  const tournament = selectedTournament();
  if (!tournament) {
    renderMessage("Choose a WTA tournament first.", true);
    return;
  }

  const playerA = findPlayer(state.players, elements.playerA.value);
  const playerB = findPlayer(state.players, elements.playerB.value);

  if (!playerA || !playerB) {
    renderMessage("Both players must exist in the active snapshot.", true);
    return;
  }

  if (playerA.name === playerB.name) {
    renderMessage("Choose two different players.", true);
    return;
  }

  const prediction = predictMatch(state.model, playerA, playerB, tournament);
  renderPrediction(prediction);
});

function renderPrediction(prediction) {
  const aPct = Math.round(prediction.probabilityA * 100);
  const bPct = 100 - aPct;
  const favPct = prediction.favorite.name === prediction.playerA.name ? aPct : bPct;
  const h2h = getH2h(prediction.playerA, prediction.playerB);
  const surfaceKey = `${prediction.tournament.modelSurface}Elo`;
  const surface = prediction.tournament.modelSurface;
  const level = prediction.tournament.level;

  elements.result.innerHTML = `
    <div class="winner-line">
      <div>
        <h2 class="winner-name">${escapeHtml(prediction.favorite.name)}</h2>
        <p class="winner-meta">${escapeHtml(prediction.tournament.name)} - ${capitalize(prediction.tournament.modelSurface)} - ${formatLevel(prediction.tournament.level)}</p>
      </div>
      <div class="probability">${favPct}%</div>
    </div>

    <div class="bars">
      ${renderBar(prediction.playerA.name, aPct, false)}
      ${renderBar(prediction.playerB.name, bPct, true)}
    </div>

    <div class="explain-grid">
      ${metric("Ranking", `#${prediction.playerA.rank} vs #${prediction.playerB.rank}`)}
      ${metric("Overall Elo", `${prediction.playerA.elo} vs ${prediction.playerB.elo}`)}
      ${metric(`${capitalize(prediction.tournament.modelSurface)} Elo`, `${prediction.playerA[surfaceKey]} vs ${prediction.playerB[surfaceKey]}`)}
      ${metric("Recent form", `${prediction.playerA.name}: ${recordText(prediction.playerA.recent)} / ${prediction.playerB.name}: ${recordText(prediction.playerB.recent)}`)}
      ${metric("Head-to-head", h2h.total ? `${h2h.wins}-${h2h.losses}` : "No matches in snapshot")}
      ${metric(`${capitalize(surface)} record`, `${recordText(prediction.playerA.surfaceRecords?.[surface])} vs ${recordText(prediction.playerB.surfaceRecords?.[surface])}`)}
      ${metric(`${formatLevel(level)} record`, `${recordText(prediction.playerA.levelRecords?.[level])} vs ${recordText(prediction.playerB.levelRecords?.[level])}`)}
    </div>

    ${renderSnapshotWarnings()}
  `;
}

function renderSnapshotWarnings() {
  const warnings = [];
  const freshness = getFreshness(state.snapshot);

  if (state.snapshot.meta?.isDemo) {
    warnings.push("This is a demo player snapshot. Run scripts/refresh_snapshot.py before treating the output as current.");
  }

  if (freshness.isStale) {
    warnings.push(`The active snapshot is ${freshness.ageDays} days old. Refresh public tennis data before relying on this prediction.`);
  }

  return warnings.map((warning) => `<p class="warning">${escapeHtml(warning)}</p>`).join("");
}

function renderBar(name, percent, alt) {
  return `
    <div class="bar-row">
      <strong>${escapeHtml(name)}</strong>
      <div class="bar-track"><div class="bar-fill ${alt ? "alt" : ""}" style="width: ${percent}%"></div></div>
      <span>${percent}%</span>
    </div>
  `;
}

function metric(label, value) {
  return `
    <div class="metric">
      <div class="metric-label">${escapeHtml(label)}</div>
      <p class="metric-value">${escapeHtml(value)}</p>
    </div>
  `;
}

function renderMessage(message, isError = false) {
  elements.result.innerHTML = `<p class="${isError ? "warning" : ""}">${escapeHtml(message)}</p>`;
}

boot();
