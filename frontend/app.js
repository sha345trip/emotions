/**
 * app.js — Emotional Weight Frontend Logic
 * =========================================
 * Phase 1: Stubs only. Full implementation in Phase 4.
 *
 * Responsibility breakdown:
 *   - BACKEND_URL : configurable constant — change for local vs. deployed
 *   - analyse()   : POST to /analyse/batch, render highlights (Phase 4)
 *   - updateCounts() : live word + sentence count in footer
 *   - tooltip     : click → show region details popover (Phase 4)
 */

// ── Configuration ──────────────────────────────────────────────────────────

/**
 * Backend URL — change this to switch between local dev and deployed Space.
 * Local dev   : "http://localhost:8000"
 * HF Spaces   : "https://<your-space>.hf.space"  (set in Phase 5)
 */
const BACKEND_URL = "http://localhost:8000";

// ── Region metadata mirrored from data/roi_map.py ─────────────────────────

const REGION_META = {
  TPJ: {
    label: "TPJ / Angular Gyrus",
    cls: "region-tpj",
    explanation:
      "The TPJ activates when we attribute beliefs, feelings, and intentions " +
      "to others — the bedrock of narrative empathy. Sentences that recruit " +
      "this region feel emotionally charged or deeply character-driven.",
  },
  MTG: {
    label: "MTG / Superior Temporal",
    cls: "region-tpj",   // grouped visually with TPJ (amber)
    explanation:
      "TE1a sits at the boundary between auditory processing and lexical " +
      "knowledge. It activates for words with rich sensory or emotional valence.",
  },
  Broca: {
    label: "Broca Area / IFG",
    cls: "region-broca",
    explanation:
      "Broca's area is the engine of grammatical processing. Sentences that " +
      "light it up have nested clauses, unusual word order, or dense " +
      "propositional content — writing that makes the reader work syntactically.",
  },
  STS: {
    label: "STS / Auditory Language",
    cls: "region-sts",
    explanation:
      "The STS is sensitive to the sound of language even during silent reading. " +
      "Sentences with natural prosody, dialogue, or a strong internal voice " +
      "engage this region most.",
  },
  DMN: {
    label: "Default Mode / Prefrontal",
    cls: "region-dmn",
    explanation:
      "The DMN underpins our sense of a continuous story-world. It activates " +
      "when readers integrate new information with prior knowledge, making " +
      "abstract ideas feel personally meaningful.",
  },
  Neutral: {
    label: "Neutral",
    cls: "",
    explanation:
      "No single region dominates — this sentence produces balanced or low " +
      "cortical activation across all tracked areas.",
  },
};

// ── DOM references ─────────────────────────────────────────────────────────

const editorEl      = document.getElementById("editor");
const editorWrapper = document.getElementById("editor-wrapper");
const outputWrapper = document.getElementById("output-wrapper");
const outputEl      = document.getElementById("output");
const loadingEl     = document.getElementById("loading-overlay");
const btnAnalyse    = document.getElementById("btn-analyse");
const statusDot     = document.getElementById("status-indicator");
const tooltipEl     = document.getElementById("sentence-tooltip");
const tooltipRegion = document.getElementById("tooltip-region-name");
const tooltipConf   = document.getElementById("tooltip-confidence");
const tooltipExpl   = document.getElementById("tooltip-explanation");
const wordCountEl   = document.getElementById("word-count");
const sentCountEl   = document.getElementById("sentence-count");
const regionSumEl   = document.getElementById("region-summary");

// ── Live word / sentence counter ──────────────────────────────────────────

function updateCounts() {
  const text = editorEl.value.trim();
  if (!text) {
    wordCountEl.textContent = "0 words";
    sentCountEl.textContent = "0 sentences";
    return;
  }
  const words     = text.split(/\s+/).filter(Boolean).length;
  const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0).length;
  wordCountEl.textContent = `${words} word${words !== 1 ? "s" : ""}`;
  sentCountEl.textContent = `${sentences} sentence${sentences !== 1 ? "s" : ""}`;
}

editorEl.addEventListener("input", updateCounts);

// ── Health-check on load ──────────────────────────────────────────────────

async function checkHealth() {
  try {
    const res = await fetch(`${BACKEND_URL}/health`, { signal: AbortSignal.timeout(5000) });
    if (res.ok) {
      const data = await res.json();
      statusDot.className = data.model_loaded
        ? "status-dot status-ready"
        : "status-dot status-loading";
      statusDot.title = data.model_loaded ? "Backend ready" : "Backend up, model loading…";
    }
  } catch {
    statusDot.className = "status-dot status-offline";
    statusDot.title = "Backend offline — start FastAPI server";
  }
}

// ── Analyse (Phase 4 — full implementation) ────────────────────────────────

async function analyse() {
  const text = editorEl.value.trim();
  if (!text) return;

  // Show loading overlay
  loadingEl.classList.remove("hidden");
  btnAnalyse.disabled = true;

  try {
    // Phase 4: real API call
    const res = await fetch(`${BACKEND_URL}/analyse/batch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sentences: splitSentences(text) }),
    });

    if (!res.ok) throw new Error(`API error ${res.status}`);
    const results = await res.json(); // [{sentence, region, confidence}, …]

    renderHighlights(results);
  } catch (err) {
    console.error("Analysis failed:", err);
    alert(`Analysis failed: ${err.message}\n\nMake sure the backend is running at:\n${BACKEND_URL}`);
  } finally {
    loadingEl.classList.add("hidden");
    btnAnalyse.disabled = false;
  }
}

// ── Sentence splitting ────────────────────────────────────────────────────

function splitSentences(text) {
  // Simple sentence splitter — Phase 2 uses nltk server-side as authoritative
  return text
    .split(/(?<=[.!?])\s+/)
    .map(s => s.trim())
    .filter(Boolean);
}

// ── Render highlights (Phase 4) ───────────────────────────────────────────

function renderHighlights(results) {
  outputEl.innerHTML = "";
  const regionCounts = {};

  results.forEach((item, idx) => {
    const meta  = REGION_META[item.region] || REGION_META.Neutral;
    const span  = document.createElement("span");
    span.className   = `sentence-span ${meta.cls}`.trim();
    span.textContent = item.sentence + " ";
    span.dataset.region     = item.region || "Neutral";
    span.dataset.confidence = (item.confidence * 100).toFixed(1);
    span.dataset.idx        = idx;
    span.setAttribute("role", "button");
    span.setAttribute("tabindex", "0");
    span.setAttribute("aria-label",
      `${item.sentence} — ${meta.label}, ${(item.confidence * 100).toFixed(0)}% confidence`);

    span.addEventListener("click", (e) => {
      e.stopPropagation();
      showTooltip(span, meta, item.confidence);
    });
    span.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") showTooltip(span, meta, item.confidence);
    });

    outputEl.appendChild(span);

    // Tally for footer region summary
    regionCounts[item.region] = (regionCounts[item.region] || 0) + 1;
  });

  // Footer region summary
  const summary = Object.entries(regionCounts)
    .sort((a, b) => b[1] - a[1])
    .map(([r, n]) => `${n} ${r}`)
    .join(" · ");
  regionSumEl.textContent = summary;

  // Update sentence count
  sentCountEl.textContent = `${results.length} sentence${results.length !== 1 ? "s" : ""}`;

  // Switch to output view
  editorWrapper.classList.add("hidden");
  outputWrapper.classList.remove("hidden");
}

// ── Tooltip ───────────────────────────────────────────────────────────────

function showTooltip(spanEl, meta, confidence) {
  tooltipRegion.textContent = meta.label;
  tooltipConf.textContent   = `${(confidence * 100).toFixed(1)}% confidence`;
  tooltipExpl.textContent   = meta.explanation;

  // Position near the clicked span
  const rect   = spanEl.getBoundingClientRect();
  const scrollY = window.scrollY;
  tooltipEl.style.top  = `${rect.bottom + scrollY + 8}px`;
  tooltipEl.style.left = `${Math.min(rect.left, window.innerWidth - 340)}px`;

  tooltipEl.classList.remove("hidden");
}

function closeTooltip() {
  tooltipEl.classList.add("hidden");
}

// Close tooltip when clicking elsewhere
document.addEventListener("click", (e) => {
  if (!tooltipEl.contains(e.target)) closeTooltip();
});

// ── Editor controls ───────────────────────────────────────────────────────

function returnToEditor() {
  outputWrapper.classList.add("hidden");
  editorWrapper.classList.remove("hidden");
  closeTooltip();
}

function clearEditor() {
  editorEl.value = "";
  outputEl.innerHTML = "";
  returnToEditor();
  updateCounts();
  regionSumEl.textContent = "";
}

// ── Init ──────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  checkHealth();
  updateCounts();
});
