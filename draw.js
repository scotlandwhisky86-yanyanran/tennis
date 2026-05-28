import {
  capitalize,
  escapeHtml,
  fetchJson,
  formatLevel,
  getFreshness,
  normalizeName,
  predictMatch,
  sortPlayersByName,
} from "./predictor-core.js";

const SECTION_SIZE = 16;
const ROW_STEP = 46;
const BASE_Y = 18;
const SECTION_COLUMNS = [42, 250, 456, 662, 868];
const FINAL_COLUMNS = [72, 306, 540, 774];
const QUALIFIER_OPTION = "__qualifier__";
const TESSERACT_SOURCES = [
  {
    script: "https://cdn.jsdelivr.net/npm/tesseract.js@5/dist/tesseract.min.js",
    workerPath: "https://cdn.jsdelivr.net/npm/tesseract.js@5/dist/worker.min.js",
    corePath: "https://cdn.jsdelivr.net/npm/tesseract.js-core@5/tesseract-core.wasm.js",
  },
  {
    script: "https://unpkg.com/tesseract.js@5/dist/tesseract.min.js",
    workerPath: "https://unpkg.com/tesseract.js@5/dist/worker.min.js",
    corePath: "https://unpkg.com/tesseract.js-core@5/tesseract-core.wasm.js",
  },
];
const TESSERACT_LANG_PATH = "https://tessdata.projectnaptha.com/4.0.0";

const state = {
  snapshot: null,
  model: null,
  tournaments: [],
  players: [],
  playerByName: new Map(),
  drawSize: 128,
  slots: Array(128).fill(null),
  activeTab: 0,
  activeSlot: null,
  projection: null,
  titleOdds: [],
  lastTournament: null,
  importImageFile: null,
  importRows: [],
  ocrPreviewTimer: null,
  tesseractOptions: null,
};

const elements = {
  form: document.querySelector("#draw-form"),
  importButton: document.querySelector("#import-image"),
  sampleButton: document.querySelector("#sample-draw"),
  clearButton: document.querySelector("#clear-draw"),
  drawSizeDisplay: document.querySelector("#draw-size-display"),
  surfaceFilter: document.querySelector("#surface-filter"),
  levelFilter: document.querySelector("#level-filter"),
  tournament: document.querySelector("#tournament"),
  board: document.querySelector("#bracket-board"),
  tabs: document.querySelector("#section-tabs"),
  status: document.querySelector("#snapshot-status"),
  poolStatus: document.querySelector("#draw-pool-status"),
  note: document.querySelector("#draw-note"),
  summary: document.querySelector("#draw-summary"),
  picker: document.querySelector("#slot-picker"),
  pickerTitle: document.querySelector("#slot-picker-title"),
  pickerClose: document.querySelector("#close-picker"),
  pickerSearch: document.querySelector("#slot-search"),
  pickerOptions: document.querySelector("#slot-options"),
  importModal: document.querySelector("#import-modal"),
  importClose: document.querySelector("#close-import"),
  importDropzone: document.querySelector("#import-dropzone"),
  imageFile: document.querySelector("#draw-image-file"),
  chooseImage: document.querySelector("#choose-image"),
  importPreview: document.querySelector("#import-preview"),
  recognizeImage: document.querySelector("#recognize-image"),
  matchOcrText: document.querySelector("#match-ocr-text"),
  ocrStatus: document.querySelector("#ocr-status"),
  ocrText: document.querySelector("#ocr-text"),
  ocrStartSlot: document.querySelector("#ocr-start-slot"),
  applyOcrImport: document.querySelector("#apply-ocr-import"),
  ocrMatches: document.querySelector("#ocr-matches"),
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
    state.players = sortPlayersByName(snapshot.players || []);
    state.playerByName = new Map(state.players.map((player) => [normalizeName(player.name), player]));

    elements.poolStatus.textContent = `${state.players.length} active top players loaded - choose a tournament and build the first column`;
    hydrateTournamentFilters();
    updateStatus(snapshot);
  } catch (error) {
    elements.status.textContent = "Data unavailable";
    elements.board.innerHTML = `<p class="warning">Could not load draw data. ${escapeHtml(error.message)}</p>`;
  }
}

function hydrateTournamentFilters() {
  renderTournamentOptions();
  syncDrawSizeFromTournament("Tournament selected. The draw size was set automatically.");
  elements.surfaceFilter.addEventListener("change", () => {
    renderTournamentOptions();
    syncDrawSizeFromTournament("Tournament list changed. The draw size was set automatically.");
  });
  elements.levelFilter.addEventListener("change", () => {
    renderTournamentOptions();
    syncDrawSizeFromTournament("Tournament list changed. The draw size was set automatically.");
  });
  elements.tournament.addEventListener("change", () => {
    syncDrawSizeFromTournament("Tournament changed. The draw size was set automatically.");
  });
}

function renderTournamentOptions() {
  const surface = elements.surfaceFilter.value;
  const level = elements.levelFilter.value;
  const tournaments = state.tournaments
    .filter((tournament) => surface === "all" || tournament.modelSurface === surface)
    .filter((tournament) => level === "all" || tournament.level === level)
    .sort((a, b) => a.startDate.localeCompare(b.startDate));

  elements.tournament.innerHTML = tournaments.length
    ? tournaments.map(tournamentOption).join("")
    : `<option value="">No matching tournaments</option>`;
}

function tournamentOption(tournament) {
  return `<option value="${escapeHtml(tournament.id)}">${escapeHtml(tournament.name)}</option>`;
}

function selectedTournament() {
  return state.tournaments.find((tournament) => tournament.id === elements.tournament.value);
}

function syncDrawSizeFromTournament(message) {
  const tournament = selectedTournament();
  if (!tournament) {
    elements.drawSizeDisplay.textContent = "No tournament";
    return;
  }

  const nextSize = drawSizeForTournament(tournament);
  const sizeChanged = nextSize !== state.drawSize;
  state.drawSize = nextSize;
  state.slots = Array.from({ length: nextSize }, (_, index) => state.slots[index] || null);

  const maxTab = hasFinalTab() ? finalTabIndex() : 0;
  if (state.activeTab > maxTab) {
    state.activeTab = 0;
  }

  elements.drawSizeDisplay.textContent = `${nextSize} slots`;
  closePicker();
  clearProjection(message);
  if (sizeChanged) {
    state.activeTab = 0;
  }
  renderSectionTabs();
  renderBoard();
}

function drawSizeForTournament(tournament) {
  if (Number.isFinite(tournament?.drawSize)) {
    return tournament.drawSize;
  }
  if (tournament?.level === "grand_slam") return 128;
  if (tournament?.level === "finals") return 8;
  return 32;
}

function updateStatus(snapshot) {
  const label = snapshot.meta?.label || "Snapshot";
  const date = snapshot.meta?.updatedAt || "unknown date";
  const freshness = getFreshness(snapshot);
  elements.status.textContent = `${label} - ${date}${freshness.isStale ? " - stale" : ""}`;
  elements.status.classList.toggle("is-stale", freshness.isStale);
}

function sectionCount() {
  return state.drawSize <= SECTION_SIZE ? 1 : state.drawSize / SECTION_SIZE;
}

function sectionSize(sectionIndex = state.activeTab) {
  const startSlot = sectionIndex * SECTION_SIZE;
  return Math.max(0, Math.min(SECTION_SIZE, state.drawSize - startSlot));
}

function hasFinalTab() {
  return sectionCount() > 1;
}

function finalTabIndex() {
  return sectionCount();
}

function finalTabLabel() {
  const count = sectionCount();
  if (count === 2) return "Final";
  return `Final ${count}`;
}

function renderSectionTabs() {
  const count = sectionCount();
  const tabs = Array.from({ length: count }, (_, index) => ({
    index,
    label: count === 1 ? "Main Draw" : `${index + 1}/${count} Section`,
  }));

  if (hasFinalTab()) {
    tabs.push({ index: finalTabIndex(), label: finalTabLabel() });
  }

  elements.tabs.innerHTML = tabs
    .map((tab) => `
      <button class="section-tab ${state.activeTab === tab.index ? "is-active" : ""}" type="button" data-tab="${tab.index}">
        ${escapeHtml(tab.label)}
      </button>
    `)
    .join("");
}

elements.tabs.addEventListener("click", (event) => {
  const button = event.target.closest("[data-tab]");
  if (!button) return;

  state.activeTab = Number(button.dataset.tab);
  closePicker();
  renderSectionTabs();
  renderBoard();
});

elements.board.addEventListener("click", (event) => {
  const button = event.target.closest("[data-slot-index]");
  if (!button) return;

  openPicker(Number(button.dataset.slotIndex), button);
});

elements.board.addEventListener("paste", (event) => {
  const button = event.target.closest("[data-slot-index]");
  if (!button) return;

  const text = event.clipboardData?.getData("text") || "";
  if (!shouldTreatAsBulkText(text)) return;

  event.preventDefault();
  fillSlotsFromText(Number(button.dataset.slotIndex), text);
});

elements.pickerSearch.addEventListener("input", () => {
  renderPickerOptions(elements.pickerSearch.value);
});

elements.pickerSearch.addEventListener("paste", (event) => {
  const text = event.clipboardData?.getData("text") || "";
  if (state.activeSlot === null || !shouldTreatAsBulkText(text)) return;

  event.preventDefault();
  fillSlotsFromText(state.activeSlot, text);
  closePicker();
});

elements.pickerClose.addEventListener("click", closePicker);

elements.pickerOptions.addEventListener("click", (event) => {
  const option = event.target.closest("[data-player-option]");
  if (!option || state.activeSlot === null) return;

  const playerName = option.dataset.playerOption;
  const player = playerName === QUALIFIER_OPTION
    ? createQualifierPlayer(state.activeSlot)
    : playerName ? state.playerByName.get(normalizeName(playerName)) : null;
  if (player) registerPlayer(player, { includeInPicker: false });
  state.slots[state.activeSlot] = player || null;
  clearProjection();
  closePicker();
  renderBoard();
});

document.addEventListener("click", (event) => {
  if (!elements.picker.hidden && !event.target.closest(".slot-picker") && !event.target.closest("[data-slot-index]")) {
    closePicker();
  }
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closePicker();
  if (event.key === "Escape") closeImportModal();
});

elements.importButton.addEventListener("click", openImportModal);
elements.importClose.addEventListener("click", closeImportModal);
elements.importModal.addEventListener("click", (event) => {
  if (event.target === elements.importModal) closeImportModal();
});
elements.chooseImage.addEventListener("click", () => elements.imageFile.click());

elements.imageFile.addEventListener("change", (event) => {
  const file = event.target.files?.[0];
  if (file) setImportImage(file);
});

elements.importDropzone.addEventListener("dragover", (event) => {
  event.preventDefault();
  elements.importDropzone.classList.add("is-dragging");
});

elements.importDropzone.addEventListener("dragleave", () => {
  elements.importDropzone.classList.remove("is-dragging");
});

elements.importDropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  elements.importDropzone.classList.remove("is-dragging");
  const file = [...(event.dataTransfer?.files || [])].find((item) => item.type.startsWith("image/"));
  if (file) setImportImage(file);
});

document.addEventListener("paste", (event) => {
  if (elements.importModal.hidden) return;
  const file = [...(event.clipboardData?.files || [])].find((item) => item.type.startsWith("image/"));
  if (!file) return;
  event.preventDefault();
  setImportImage(file);
});

elements.recognizeImage.addEventListener("click", recognizeImportImage);
elements.matchOcrText.addEventListener("click", () => matchOcrText(true));
elements.ocrText.addEventListener("input", scheduleOcrPreview);
elements.applyOcrImport.addEventListener("click", applyOcrImport);

elements.sampleButton.addEventListener("click", () => {
  const samplePlayers = [...state.players]
    .sort((a, b) => (a.rank || 9999) - (b.rank || 9999))
    .slice(0, state.drawSize);

  state.slots = Array.from({ length: state.drawSize }, (_, index) => samplePlayers[index] || null);
  clearProjection();
  closePicker();
  renderBoard();
});

elements.clearButton.addEventListener("click", () => {
  state.slots = Array(state.drawSize).fill(null);
  clearProjection("Draw cleared. Click a first-column slot or paste a copied list to start again.");
  closePicker();
  renderBoard();
});

elements.form.addEventListener("submit", (event) => {
  event.preventDefault();
  if (!state.snapshot || !state.model) return;

  const tournament = selectedTournament();
  if (!tournament) {
    renderNote("Choose a WTA tournament first.", true);
    return;
  }

  const validation = validateDraw(tournament);
  if (validation.error) {
    renderNote(validation.error, true);
    return;
  }

  const slots = state.slots.map((player, index) => ({
    index,
    player,
    label: playerLabel(player) || "BYE",
    isBye: !player,
  }));

  state.projection = projectBracket(slots, tournament);
  state.titleOdds = calculateTitleOdds(slots, tournament);
  state.lastTournament = tournament;

  renderNote("Prediction submitted. Later-round cells now show projected winners.", false);
  closePicker();
  renderBoard();
  renderSummary();
});

function validateDraw(tournament) {
  const players = state.slots.filter(Boolean);
  if (players.length < 2) {
    return { error: "Choose at least two players before submitting the draw." };
  }

  const seen = new Set();
  const duplicate = players.find((player) => {
    const key = normalizeName(player.name);
    if (seen.has(key)) return true;
    seen.add(key);
    return false;
  });

  if (duplicate) {
    return { error: `${playerLabel(duplicate)} appears more than once in the draw.` };
  }

  const emptySlots = state.slots
    .map((player, index) => (player ? null : index + 1))
    .filter(Boolean);

  if (requiresFullDraw(tournament) && emptySlots.length) {
    return {
      error: `${tournament.name} is a ${state.drawSize}-slot no-BYE draw. Fill every first-column slot before submitting. Empty slots: ${emptySlots.slice(0, 8).join(", ")}${emptySlots.length > 8 ? "..." : ""}.`,
    };
  }

  const emptyFirstRoundMatches = [];
  for (let index = 0; index < state.slots.length; index += 2) {
    if (!state.slots[index] && !state.slots[index + 1]) {
      emptyFirstRoundMatches.push(`${index + 1}-${index + 2}`);
    }
  }

  if (emptyFirstRoundMatches.length) {
    return {
      error: `Some first-round matches are fully empty, which would create BYEs after round one. Fill at least one player in slots ${emptyFirstRoundMatches.slice(0, 6).join(", ")}${emptyFirstRoundMatches.length > 6 ? "..." : ""}.`,
    };
  }

  return {};
}

function requiresFullDraw(tournament) {
  return tournament?.level === "grand_slam" || tournament?.level === "finals";
}

function clearProjection(message = "Draw changed. Submit again to refresh projected winners.") {
  state.projection = null;
  state.titleOdds = [];
  state.lastTournament = null;
  elements.summary.hidden = true;
  elements.summary.innerHTML = "";
  renderNote(message, false);
}

function renderNote(message, isError = false) {
  elements.note.textContent = message;
  elements.note.classList.toggle("is-error", isError);
}

function renderBoard() {
  if (hasFinalTab() && state.activeTab === finalTabIndex()) {
    renderFinalStage();
    return;
  }

  renderSection(state.activeTab);
}

function renderSection(sectionIndex) {
  const startSlot = sectionIndex * SECTION_SIZE;
  const visibleRows = sectionSize(sectionIndex);
  const cells = [];

  for (let row = 0; row < visibleRows; row += 1) {
    const slotIndex = startSlot + row;
    cells.push(renderSlotNumber(slotIndex, row));
    cells.push(renderEntryCell(slotIndex, row));
  }

  renderProjectedColumn(cells, 0, sectionIndex, visibleRows, 2, SECTION_COLUMNS[1]);
  renderProjectedColumn(cells, 1, sectionIndex, visibleRows, 4, SECTION_COLUMNS[2]);
  renderProjectedColumn(cells, 2, sectionIndex, visibleRows, 8, SECTION_COLUMNS[3]);
  renderProjectedColumn(cells, 3, sectionIndex, visibleRows, 16, SECTION_COLUMNS[4]);

  elements.board.className = "bracket-board section-board";
  elements.board.style.height = `${BASE_Y * 2 + visibleRows * ROW_STEP}px`;
  elements.board.innerHTML = cells.join("");
}

function renderProjectedColumn(cells, roundIndex, sectionIndex, visibleRows, groupSize, x) {
  if (groupSize > visibleRows) return;

  const matchCount = visibleRows / groupSize;
  const matchStart = sectionIndex * matchCount;
  const matches = state.projection?.rounds[roundIndex]?.matches || [];

  for (let index = 0; index < matchCount; index += 1) {
    const match = matches[matchStart + index];
    const row = index * groupSize + (groupSize - 1) / 2;
    cells.push(renderProjectedCell(match, x, yForRow(row)));
  }
}

function renderFinalStage() {
  const cells = [];
  const count = sectionCount();
  const entrants = state.projection?.rounds[3]?.matches.map((match) => match.winner) || Array(count).fill(null);

  entrants.forEach((player, row) => {
    cells.push(renderFinalSeed(row, player));
  });

  for (let groupSize = 2, columnIndex = 1; groupSize <= count; groupSize *= 2, columnIndex += 1) {
    const roundIndex = 3 + Math.log2(groupSize);
    renderFinalProjectedColumn(cells, roundIndex, groupSize, FINAL_COLUMNS[columnIndex]);
  }

  elements.board.className = "bracket-board final-board";
  elements.board.style.height = `${BASE_Y * 2 + count * 70}px`;
  elements.board.innerHTML = cells.join("");
}

function renderFinalProjectedColumn(cells, roundIndex, groupSize, x) {
  const count = sectionCount();
  const matchCount = count / groupSize;
  const matches = state.projection?.rounds[roundIndex]?.matches || [];

  for (let index = 0; index < matchCount; index += 1) {
    const match = matches[index];
    const row = index * groupSize + (groupSize - 1) / 2;
    cells.push(renderProjectedCell(match, x, finalYForRow(row)));
  }
}

function renderSlotNumber(slotIndex, row) {
  return `
    <div class="slot-number" style="--x: 0px; --y: ${yForRow(row)}px;">
      ${slotIndex + 1}
    </div>
  `;
}

function renderEntryCell(slotIndex, row) {
  const player = state.slots[slotIndex];
  return `
    <button
      class="bracket-cell entry-cell ${player ? "has-player" : ""}"
      type="button"
      data-slot-index="${slotIndex}"
      style="--x: ${SECTION_COLUMNS[0]}px; --y: ${yForRow(row)}px;"
      aria-label="Slot ${slotIndex + 1}${player ? ` ${player.name}` : " empty"}"
    >
      <span>${player ? escapeHtml(playerLabel(player)) : ""}</span>
      ${player?.rank && player.rank < 9999 ? `<em>#${escapeHtml(player.rank)}</em>` : ""}
    </button>
  `;
}

function renderFinalSeed(row, player) {
  return `
    <div class="slot-number final-seed-number" style="--x: 0px; --y: ${finalYForRow(row)}px;">
      ${row + 1}
    </div>
    <div
      class="bracket-cell projected-cell ${player ? "has-winner" : ""}"
      style="--x: ${FINAL_COLUMNS[0]}px; --y: ${finalYForRow(row)}px;"
    >
      <span>${escapeHtml(playerLabel(player) || "")}</span>
    </div>
  `;
}

function renderProjectedCell(match, x, y) {
  const winner = match?.winner || null;
  const probability = winner ? winnerProbability(match, winner) : null;
  return `
    <div
      class="bracket-cell projected-cell ${winner ? "has-winner" : ""}"
      style="--x: ${x}px; --y: ${y}px;"
    >
      <span>${escapeHtml(playerLabel(winner) || "")}</span>
      ${probability !== null ? `<em>${Math.round(probability * 100)}%</em>` : ""}
    </div>
  `;
}

function yForRow(row) {
  return BASE_Y + row * ROW_STEP;
}

function finalYForRow(row) {
  return BASE_Y + row * 70;
}

function openPicker(slotIndex, anchor) {
  state.activeSlot = slotIndex;
  elements.pickerTitle.textContent = `Slot ${slotIndex + 1}`;
  elements.pickerSearch.value = "";
  renderPickerOptions("");
  positionPicker(anchor);
  elements.picker.hidden = false;
  elements.pickerSearch.focus();
}

function closePicker() {
  elements.picker.hidden = true;
  state.activeSlot = null;
}

function openImportModal() {
  closePicker();
  elements.importModal.hidden = false;
  elements.ocrStartSlot.max = state.drawSize;
  elements.ocrStartSlot.value = state.activeTab < finalTabIndex() ? state.activeTab * SECTION_SIZE + 1 : 1;
  elements.ocrText.focus();
}

function closeImportModal() {
  elements.importModal.hidden = true;
}

function setImportImage(file) {
  state.importImageFile = file;
  state.importRows = [];
  elements.ocrMatches.innerHTML = "";
  elements.ocrStatus.textContent = `${file.name || "Pasted image"} selected.`;

  if (elements.importPreview.src) {
    URL.revokeObjectURL(elements.importPreview.src);
  }
  elements.importPreview.src = URL.createObjectURL(file);
  elements.importPreview.hidden = false;
}

async function recognizeImportImage() {
  if (!state.importImageFile) {
    elements.ocrStatus.textContent = "Choose or paste an image first.";
    return;
  }

  try {
    elements.recognizeImage.disabled = true;
    elements.ocrStatus.textContent = "Loading English OCR engine...";
    await loadTesseract();

    elements.ocrStatus.textContent = "Recognizing English text...";
    const result = await window.Tesseract.recognize(state.importImageFile, "eng", {
      ...(state.tesseractOptions || {}),
      logger: (message) => {
        if (message.status === "recognizing text" && Number.isFinite(message.progress)) {
          elements.ocrStatus.textContent = `Recognizing English text - ${Math.round(message.progress * 100)}%`;
        }
      },
    });

    elements.ocrText.value = result.data?.text || "";
    elements.ocrStatus.textContent = "Text recognized. Review it, then match names.";
    await matchOcrText(false);
  } catch (error) {
    elements.ocrStatus.textContent = `OCR unavailable. ${error.message}`;
  } finally {
    elements.recognizeImage.disabled = false;
  }
}

function loadTesseract() {
  if (window.Tesseract) return Promise.resolve();
  return loadScriptWithFallback(TESSERACT_SOURCES);
}

function loadScriptWithFallback(sources, index = 0) {
  return new Promise((resolve, reject) => {
    const source = sources[index];
    if (!source) {
      reject(new Error("Could not load Tesseract.js from CDN."));
      return;
    }

    const script = document.createElement("script");
    script.src = source.script;
    script.async = true;
    script.onload = () => {
      state.tesseractOptions = {
        workerPath: source.workerPath,
        corePath: source.corePath,
        langPath: TESSERACT_LANG_PATH,
      };
      resolve();
    };
    script.onerror = () => {
      script.remove();
      loadScriptWithFallback(sources, index + 1).then(resolve).catch(reject);
    };
    document.head.append(script);
  });
}

async function matchOcrText(showStatus) {
  clearTimeout(state.ocrPreviewTimer);
  const entries = parseOcrEntries(elements.ocrText.value);
  state.importRows = entries.map((entry) => resolveDrawEntry(entry, 0));
  renderOcrMatches();

  const missingRows = state.importRows.filter((row) => row.status === "missing");
  if (missingRows.length) {
    elements.ocrStatus.textContent = `Looking up ${missingRows.length} player${missingRows.length === 1 ? "" : "s"} outside the current list...`;
    await lookupMissingRows(missingRows);
    renderOcrMatches();
  }

  const matched = state.importRows.filter((row) => row.player).length;
  const missing = state.importRows.filter((row) => row.status === "missing").length;
  if (showStatus || !missingRows.length) {
    elements.ocrStatus.textContent = `Matched ${matched} slot${matched === 1 ? "" : "s"}${missing ? `, ${missing} need review` : ""}.`;
  }
  return state.importRows;
}

function scheduleOcrPreview() {
  clearTimeout(state.ocrPreviewTimer);
  state.ocrPreviewTimer = setTimeout(renderOcrPreviewFromText, 180);
}

function renderOcrPreviewFromText() {
  const entries = parseOcrEntries(elements.ocrText.value);
  state.importRows = entries.map((entry) => resolveDrawEntry(entry, 0));
  renderOcrMatches();

  const matched = state.importRows.filter((row) => row.player).length;
  const missing = state.importRows.filter((row) => row.status === "missing").length;
  elements.ocrStatus.textContent = state.importRows.length
    ? `Preview updated from edited text: ${matched} matched${missing ? `, ${missing} need lookup` : ""}.`
    : "No player names found in the edited text.";
}

function renderOcrMatches() {
  if (!state.importRows.length) {
    elements.ocrMatches.innerHTML = `<div class="player-empty">No player names matched yet</div>`;
    return;
  }

  elements.ocrMatches.innerHTML = state.importRows
    .map((row, index) => `
      <div class="ocr-match-row ${row.status === "missing" ? "is-missing" : ""} ${row.status === "qualifier" ? "is-qualifier" : ""} ${row.status === "external" ? "is-external" : ""}">
        <span>${index + 1}</span>
        <em>${escapeHtml(row.source)}</em>
        <strong>${escapeHtml(playerLabel(row.player) || "Not matched")}</strong>
        <span>${escapeHtml(row.method)}</span>
      </div>
    `)
    .join("");
}

async function applyOcrImport() {
  const rows = await matchOcrText(false);
  if (!rows.length) {
    elements.ocrStatus.textContent = "No matched rows to apply.";
    return;
  }

  const startSlot = Math.max(0, Math.min(state.drawSize - 1, Number(elements.ocrStartSlot.value || 1) - 1));
  fillSlotsFromResolvedRows(startSlot, rows);
  closeImportModal();
}

function positionPicker(anchor) {
  const rect = anchor.getBoundingClientRect();
  const pickerWidth = 330;
  const left = Math.max(12, Math.min(window.innerWidth - pickerWidth - 12, rect.left));
  const top = Math.min(window.innerHeight - 420, rect.bottom + 8);
  elements.picker.style.left = `${left}px`;
  elements.picker.style.top = `${Math.max(12, top)}px`;
}

function renderPickerOptions(query) {
  const normalizedQuery = normalizeName(query);
  const used = usedPlayerNames(state.activeSlot);
  const matches = state.players
    .filter((player) => !used.has(normalizeName(player.name)))
    .filter((player) => !normalizedQuery || normalizeName(player.name).includes(normalizedQuery))
    .slice(0, 250);

  elements.pickerOptions.innerHTML = `
    <button class="slot-option is-empty-option" type="button" data-player-option="">
      <span>Blank / BYE</span>
      <em>empty slot</em>
    </button>
    <button class="slot-option is-empty-option" type="button" data-player-option="${QUALIFIER_OPTION}">
      <span>Qualifier</span>
      <em>Q slot</em>
    </button>
    ${matches.length ? matches.map(renderPlayerOption).join("") : `<div class="player-empty">No available players found</div>`}
  `;
}

function renderPlayerOption(player) {
  return `
    <button class="slot-option" type="button" data-player-option="${escapeHtml(player.name)}">
      <span>${escapeHtml(playerLabel(player))}</span>
      <em>${player.rank && player.rank < 9999 ? `#${escapeHtml(player.rank)}` : "external"}</em>
    </button>
  `;
}

function usedPlayerNames(excludeSlot) {
  return new Set(
    state.slots
      .map((player, index) => (index === excludeSlot ? null : player))
      .filter(Boolean)
      .map((player) => normalizeName(player.name)),
  );
}

function shouldTreatAsBulkText(text) {
  return text.includes("\n") || text.includes("\r") || text.includes("\t");
}

async function fillSlotsFromText(startSlot, text) {
  const entries = parsePastedEntries(text);
  if (!entries.length) {
    renderNote("No player names found in the pasted text.", true);
    return;
  }

  const rows = entries.map((entry) => resolveDrawEntry(entry, 0));
  const missingRows = rows.filter((row) => row.status === "missing");
  if (missingRows.length) {
    renderNote(`Looking up ${missingRows.length} player${missingRows.length === 1 ? "" : "s"} outside the current list...`, false);
    await lookupMissingRows(missingRows);
  }
  fillSlotsFromResolvedRows(startSlot, rows);
}

function fillSlotsFromResolvedRows(startSlot, rows) {
  const capacity = state.drawSize - startSlot;
  const clipped = rows.slice(0, capacity);
  const nextSlots = [...state.slots];

  for (let offset = 0; offset < clipped.length; offset += 1) {
    nextSlots[startSlot + offset] = null;
  }

  const used = new Set(nextSlots.filter(Boolean).map((player) => normalizeName(player.name)));
  const missing = [];
  const duplicates = [];
  let filled = 0;

  clipped.forEach((entry, offset) => {
    const slotIndex = startSlot + offset;
    if (entry.status === "bye") {
      nextSlots[slotIndex] = null;
      return;
    }

    const player = entry.status === "qualifier" ? createQualifierPlayer(slotIndex) : entry.player;
    if (!player) {
      missing.push(entry.source);
      return;
    }

    registerPlayer(player, { includeInPicker: false });
    const key = normalizeName(player.name);
    if (used.has(key)) {
      duplicates.push(playerLabel(player));
      return;
    }

    nextSlots[slotIndex] = player;
    used.add(key);
    filled += 1;
  });

  state.slots = nextSlots;
  clearProjection();
  renderBoard();

  const warnings = [];
  if (rows.length > capacity) warnings.push(`${rows.length - capacity} extra names did not fit`);
  if (missing.length) warnings.push(`not found: ${missing.slice(0, 5).join(", ")}${missing.length > 5 ? "..." : ""}`);
  if (duplicates.length) warnings.push(`duplicates skipped: ${duplicates.slice(0, 5).join(", ")}${duplicates.length > 5 ? "..." : ""}`);

  const message = warnings.length
    ? `Imported ${filled} player${filled === 1 ? "" : "s"}. ${warnings.join("; ")}.`
    : `Imported ${filled} player${filled === 1 ? "" : "s"} from slot ${startSlot + 1}. Submit to predict.`;
  renderNote(message, warnings.length > 0);
}

function parsePastedEntries(text) {
  const lines = text.replace(/\r/g, "").split("\n");
  while (lines.length && !cleanEntryName(lines[lines.length - 1].split("\t")[0])) {
    lines.pop();
  }
  return lines.map((line) => cleanEntryName(line.split("\t")[0]) || "BYE");
}

function parseOcrEntries(text) {
  return String(text || "")
    .replace(/\r/g, "\n")
    .split(/\n+/)
    .map(cleanOcrLine)
    .filter(Boolean)
    .filter((line) => !isOcrNoise(line));
}

function resolveDrawEntry(entry, slotIndex) {
  if (isBye(entry)) {
    return { source: entry, player: null, status: "bye", method: "BYE" };
  }
  if (isQualifier(entry)) {
    return { source: entry, player: createQualifierPlayer(slotIndex), status: "qualifier", method: "Qualifier" };
  }

  const player = findPlayerByImportedName(entry);
  if (player) {
    return { source: entry, player, status: "matched", method: "Matched" };
  }

  return { source: entry, player: null, status: "missing", method: "Lookup" };
}

function cleanOcrLine(line) {
  let value = String(line || "")
    .replace(/[|]/g, " ")
    .replace(/[«»<>]/g, " ")
    .replace(/\s+/g, " ")
    .trim();

  value = value
    .replace(/^\d{1,3}\s+/, "")
    .replace(/^[a-z][.)]\s+/i, "")
    .replace(/^(?:Q|W|WC|LL|PR|SR|SE)\s+/i, "")
    .replace(/\(\d{1,3}\)/g, "")
    .replace(/\s+\d{1,3}\s*$/, "")
    .trim();

  value = stripTrailingCountryCode(value);

  if (normalizeSearch(value).includes("qualif")) {
    return "QUALIFIER";
  }

  return cleanEntryName(value);
}

function stripTrailingCountryCode(value) {
  const parts = String(value || "").trim().split(/\s+/).filter(Boolean);
  if (parts.length < 3) return value;

  const last = parts.at(-1);
  if (/^[A-Za-z]{3}$/.test(last)) {
    return parts.slice(0, -1).join(" ");
  }
  return value;
}

function isOcrNoise(line) {
  const compact = String(line || "").toLowerCase().replace(/\s+/g, "");
  if (/^[a-z]{0,2}\d{1,2}\/\d{1,2}$/.test(compact)) return true;

  const value = normalizeSearch(line);
  if (!value) return true;
  if (/^(draw|main|round|section|court|match|winner|seed|ranking)$/.test(value)) return true;
  return /^\d+$/.test(value);
}

function findPlayerByImportedName(entry) {
  const cleaned = cleanEntryName(entry);
  const exact = state.playerByName.get(normalizeName(cleaned));
  if (exact) return exact;

  const key = normalizeSearch(cleaned);
  if (!key) return null;

  let best = null;
  let secondScore = 0;
  for (const player of state.players) {
    for (const alias of playerSearchAliases(player)) {
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

function playerSearchAliases(player) {
  const parts = player.name.split(/\s+/).filter(Boolean);
  const aliases = [player.name];
  if (parts.length === 2) {
    aliases.push(`${parts[1]} ${parts[0]}`);
  }
  if (parts.length > 2) {
    aliases.push(`${parts.at(-1)} ${parts.slice(0, -1).join(" ")}`);
    aliases.push(`${parts.slice(-2).join(" ")} ${parts.slice(0, -2).join(" ")}`);
  }
  return aliases;
}

function cleanEntryName(line) {
  return String(line || "")
    .replace(/^\s*(\(\d{1,3}\)|\[\d{1,3}\]|\d{1,3}[.)])\s*/, "")
    .replace(/\s+\(\d{1,3}\)\s*$/, "")
    .replace(/\s+\[\d{1,3}\]\s*$/, "")
    .replace(/\s+\[[A-Z]{1,3}\]\s*$/, "")
    .trim();
}

function isBye(value) {
  return /^(bye|\(bye\)|-|--|empty)$/i.test(value);
}

function isQualifier(value) {
  return /^(qualifiee|qualifier|qualif[i1]er|qualif[i1]ee|q)$/i.test(String(value || "").trim());
}

function playerLabel(player) {
  return player?.displayName || player?.name || "";
}

function registerPlayer(player, options = {}) {
  if (!player?.name) return;
  state.playerByName.set(normalizeName(player.name), player);

  if (options.includeInPicker) {
    const existing = state.players.some((item) => normalizeName(item.name) === normalizeName(player.name));
    if (!existing) {
      state.players = sortPlayersByName([...state.players, player]);
    }
  }
}

function createQualifierPlayer(slotIndex) {
  return createGenericPlayer({
    name: `Qualifier ${slotIndex + 1}`,
    displayName: "Qualifier",
    rank: 150,
    elo: 1650,
    hardElo: 1650,
    clayElo: 1650,
    grassElo: 1600,
    isQualifier: true,
  });
}

function createExternalPlayer(name, data = {}) {
  const rank = Number.isFinite(data.rank) && data.rank > 0 ? data.rank : 9999;
  return createGenericPlayer({
    name: data.name || titleCaseName(name),
    displayName: data.name || titleCaseName(name),
    rank,
    age: Number.isFinite(data.age) && data.age > 0 ? data.age : 26,
    elo: Number.isFinite(data.elo) ? data.elo : 1550,
    hardElo: Number.isFinite(data.hardElo) ? data.hardElo : Number.isFinite(data.elo) ? data.elo : 1550,
    clayElo: Number.isFinite(data.clayElo) ? data.clayElo : Number.isFinite(data.elo) ? data.elo : 1550,
    grassElo: Number.isFinite(data.grassElo) ? data.grassElo : Number.isFinite(data.elo) ? data.elo : 1500,
    rankMissing: rank >= 9999 ? 1 : 0,
    ageMissing: Number.isFinite(data.age) && data.age > 0 ? 0 : 1,
    isExternal: true,
  });
}

function createGenericPlayer(values) {
  return {
    age: 26,
    rank: 9999,
    elo: 1550,
    hardElo: 1550,
    clayElo: 1550,
    grassElo: 1500,
    recent: { wins: 5, losses: 5 },
    h2h: {},
    highLevelMatches: 0,
    surfaceRecords: {
      hard: { wins: 0, losses: 0 },
      clay: { wins: 0, losses: 0 },
      grass: { wins: 0, losses: 0 },
    },
    levelRecords: {
      "250": { wins: 0, losses: 0 },
      "500": { wins: 0, losses: 0 },
      "1000": { wins: 0, losses: 0 },
      grand_slam: { wins: 0, losses: 0 },
      finals: { wins: 0, losses: 0 },
    },
    rankMissing: 1,
    ageMissing: 1,
    ...values,
  };
}

async function lookupMissingRows(rows) {
  await Promise.all(rows.map(async (row) => {
    try {
      const response = await fetch(`./api/player-lookup?name=${encodeURIComponent(row.source)}`);
      if (!response.ok) throw new Error(`lookup returned ${response.status}`);
      const data = await response.json();
      if (!data?.player) throw new Error("no player found");

      const player = createExternalPlayer(row.source, data.player);
      registerPlayer(player, { includeInPicker: true });
      row.player = player;
      row.status = "external";
      row.method = "Fetched";
    } catch {
      const player = createExternalPlayer(row.source);
      registerPlayer(player, { includeInPicker: false });
      row.player = player;
      row.status = "external";
      row.method = "Neutral";
    }
  }));
}

function titleCaseName(value) {
  return String(value || "")
    .toLowerCase()
    .split(/\s+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function normalizeSearch(value) {
  return String(value || "")
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]/g, "");
}

function similarity(a, b) {
  const distance = levenshtein(a, b);
  return 1 - distance / Math.max(a.length, b.length, 1);
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

function projectBracket(slots, tournament) {
  let current = slots.map((slot) => slot.player);
  const rounds = [];

  while (current.length > 1) {
    const matches = [];
    const next = [];

    for (let index = 0; index < current.length; index += 2) {
      const match = resolveProjectedMatch(current[index], current[index + 1], tournament);
      matches.push(match);
      next.push(match.winner);
    }

    rounds.push({
      name: roundName(current.length),
      matches,
    });
    current = next;
  }

  return {
    rounds,
    champion: current[0],
  };
}

function resolveProjectedMatch(left, right, tournament) {
  if (left && !right) {
    return { left, right, winner: left, probabilityA: 1, probabilityB: 0, isBye: true };
  }
  if (!left && right) {
    return { left, right, winner: right, probabilityA: 0, probabilityB: 1, isBye: true };
  }
  if (!left && !right) {
    return { left, right, winner: null, probabilityA: 0, probabilityB: 0, isBye: true };
  }

  const prediction = predictMatch(state.model, left, right, tournament);
  return {
    ...prediction,
    left,
    right,
    winner: prediction.favorite,
  };
}

function calculateTitleOdds(slots, tournament) {
  let current = slots.map((slot) => {
    if (!slot.player) return new Map();
    return new Map([[slot.player.name, 1]]);
  });

  while (current.length > 1) {
    const next = [];
    for (let index = 0; index < current.length; index += 2) {
      next.push(combineDistributions(current[index], current[index + 1], tournament));
    }
    current = next;
  }

  return [...current[0].entries()]
    .map(([name, probability]) => ({
      player: state.playerByName.get(normalizeName(name)),
      probability,
    }))
    .filter((row) => row.player)
    .sort((a, b) => b.probability - a.probability);
}

function combineDistributions(leftDistribution, rightDistribution, tournament) {
  if (!leftDistribution.size) return new Map(rightDistribution);
  if (!rightDistribution.size) return new Map(leftDistribution);

  const combined = new Map();
  for (const [leftName, leftPathProbability] of leftDistribution.entries()) {
    for (const [rightName, rightPathProbability] of rightDistribution.entries()) {
      const leftPlayer = state.playerByName.get(normalizeName(leftName));
      const rightPlayer = state.playerByName.get(normalizeName(rightName));
      const prediction = predictMatch(state.model, leftPlayer, rightPlayer, tournament);
      const matchupProbability = leftPathProbability * rightPathProbability;

      addProbability(combined, leftName, matchupProbability * prediction.probabilityA);
      addProbability(combined, rightName, matchupProbability * prediction.probabilityB);
    }
  }
  return combined;
}

function addProbability(map, key, value) {
  map.set(key, (map.get(key) || 0) + value);
}

function winnerProbability(match, winner) {
  if (!match || !winner) return 0;
  return match.left?.name === winner.name ? match.probabilityA : match.probabilityB;
}

function renderSummary() {
  const champion = state.projection?.champion || state.titleOdds[0]?.player;
  const championOdds = state.titleOdds.find((row) => row.player.name === champion?.name)?.probability || 0;

  elements.summary.hidden = false;
  elements.summary.innerHTML = `
    <div class="winner-line">
      <div>
        <h2 class="winner-name">${escapeHtml(playerLabel(champion) || "No champion")}</h2>
        <p class="winner-meta">${escapeHtml(state.lastTournament.name)} - ${capitalize(state.lastTournament.modelSurface)} - ${formatLevel(state.lastTournament.level)} - ${state.drawSize}-slot draw</p>
      </div>
      <div class="probability">${Math.round(championOdds * 100)}%</div>
    </div>

    <div class="summary-grid">
      ${renderChampionPath(champion, state.projection.rounds)}
      ${renderTitleOdds(state.titleOdds)}
    </div>
    ${renderSnapshotWarnings()}
  `;
}

function renderChampionPath(champion, rounds) {
  if (!champion) return "";

  const path = rounds
    .map((round, roundIndex) => {
      const match = round.matches.find((item) => item.winner?.name === champion.name);
      if (!match) return null;
      const opponent = match.left?.name === champion.name ? match.right : match.left;
      const probability = match.left?.name === champion.name ? match.probabilityA : match.probabilityB;
      return {
        round: round.name,
        opponent: playerLabel(opponent) || (roundIndex === 0 ? "BYE" : "Unfilled section"),
        probability,
      };
    })
    .filter(Boolean);

  return `
    <section class="draw-section">
      <h2>Projected Champion Path</h2>
      <div class="path-strip">
        ${path.map((step) => `
          <div class="path-step">
            <span>${escapeHtml(step.round)}</span>
            <strong>${escapeHtml(step.opponent)}</strong>
            <em>${Math.round(step.probability * 100)}%</em>
          </div>
        `).join("")}
      </div>
    </section>
  `;
}

function renderTitleOdds(titleOdds) {
  return `
    <section class="draw-section">
      <h2>Title Odds</h2>
      <div class="odds-table">
        ${titleOdds.slice(0, 10).map((row, index) => `
          <div class="odds-row">
            <span>${index + 1}</span>
            <strong>${escapeHtml(playerLabel(row.player))}</strong>
            <div class="mini-track"><div class="mini-fill" style="width: ${Math.max(3, row.probability * 100)}%"></div></div>
            <em>${Math.round(row.probability * 100)}%</em>
          </div>
        `).join("")}
      </div>
    </section>
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

function roundName(playersLeft) {
  const names = {
    2: "Final",
    4: "Semifinals",
    8: "Quarterfinals",
    16: "Round of 16",
    32: "Round of 32",
    64: "Round of 64",
    128: "Round of 128",
  };
  return names[playersLeft] || `${playersLeft}-player round`;
}

boot();
