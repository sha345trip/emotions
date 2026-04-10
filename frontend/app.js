/**
 * app.js — Emotional Weight Frontend Logic (Phase 4)
 * ====================================================
 * Full implementation: analyse, highlight, tooltip, counters.
 *
 * Responsibility breakdown:
 *   BACKEND_URL       – configurable constant, change for local vs deployed
 *   analyse()         – POST to /analyse/batch, render highlights
 *   renderHighlights()– stagger-animated sentence spans with region colours
 *   showTooltip()     – smart-positioned popover with neuro explanation
 *   updateCounts()    – live word + sentence count in footer
 *   checkHealth()     – polls /health and updates status indicator
 */

// ── Configuration ──────────────────────────────────────────────────────────

/**
 * Backend URL — change to switch between local dev and deployed HF Space.
 * Local dev  : "http://localhost:8000"
 * HF Spaces  : "https://<your-space>.hf.space"  ← set in Phase 5
 */
const BACKEND_URL = "https://sha345trip--emotional-weight-fastapi-app.modal.run";

// ── Region metadata (mirrors data/roi_map.py REGION_META) ─────────────────

const REGION_META = {
  TPJ: {
    label: "TPJ / Angular Gyrus",
    cls: "region-tpj",
    explanation:
      "The temporo-parietal junction activates when we attribute beliefs, feelings, " +
      "and intentions to others — the bedrock of narrative empathy. " +
      "Sentences that recruit this region feel emotionally charged or deeply character-driven.",
  },
  MTG: {
    label: "MTG / Superior Temporal",
    cls: "region-tpj",   // grouped visually with TPJ (amber)
    explanation:
      "TE1a sits at the boundary between auditory processing and lexical knowledge. " +
      "It activates strongly for words with rich sensory or emotional valence — " +
      "names, faces, vivid materials, or evocative imagery.",
  },
  Broca: {
    label: "Broca Area / IFG",
    cls: "region-broca",
    explanation:
      "Broca's area is the engine of grammatical processing. " +
      "Sentences that light it up tend to have nested clauses, unusual word order, " +
      "or dense propositional content — writing that makes the reader work syntactically.",
  },
  STS: {
    label: "STS / Auditory Language",
    cls: "region-sts",
    explanation:
      "The superior temporal sulcus is sensitive to the sound of language even during silent reading. " +
      "Sentences with natural prosody, dialogue, onomatopoeia, or a strong internal voice " +
      "engage this region most.",
  },
  DMN: {
    label: "Default Mode / Prefrontal",
    cls: "region-dmn",
    explanation:
      "The default mode network underpins our sense of a continuous story-world. " +
      "It activates when readers integrate new information with prior knowledge and personal experience, " +
      "making abstract ideas feel personally meaningful or situationally grounded.",
  },
  Neutral: {
    label: "Neutral",
    cls: "",
    explanation:
      "No single region dominates — this sentence produces balanced or low cortical " +
      "activation across all five tracked areas. Common in transitional or functional prose.",
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
const errorBanner   = document.getElementById("error-banner");
const errorMsg      = document.getElementById("error-message");

// ── Error banner ───────────────────────────────────────────────────────────

function showError(message) {
  errorMsg.textContent = message;
  errorBanner.classList.remove("hidden");
  // Auto-dismiss after 8 s
  setTimeout(() => errorBanner.classList.add("hidden"), 8000);
}

function dismissError() {
  errorBanner.classList.add("hidden");
}

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
      const isReady = data.status === "ok" || data.model_loaded;
      statusDot.className = isReady
        ? "status-dot status-ready"
        : "status-dot status-loading";
      
      statusDot.title = isReady
        ? `Backend ready · GPU Mode (Modal) · max ${data.max_sentences_per_request ?? MAX_SENTENCES} sentences`
        : "Backend up — TRIBE v2 model loading…";
      
      if (data.max_sentences_per_request) {
        MAX_SENTENCES = data.max_sentences_per_request;
      } else if (data.mode === "Modal-GPU-T4") {
        MAX_SENTENCES = 25; // GPU can handle much more
      }
    }
  } catch {
    statusDot.className = "status-dot status-offline";
    statusDot.title = "Backend offline — start uvicorn";
  }
}

// ── Sentence splitting ─────────────────────────────────────────────────────

function splitSentences(text) {
  // Client-side splitter — server uses nltk.sent_tokenize as authoritative
  return text
    .split(/(?<=[.!?])\s+/)
    .map(s => s.trim())
    .filter(Boolean);
}

// ── Sentence cap (read from /health at startup) ───────────────────────────
let MAX_SENTENCES = 5;  // default; overridden by /health response

// ── Analyse ────────────────────────────────────────────────────────────────

async function analyse() {
  const text = editorEl.value.trim();
  if (!text) return;

  // Client-side sentence count check before hitting the API
  const clientSentences = splitSentences(text);
  if (clientSentences.length > MAX_SENTENCES) {
    const modeText = statusDot.title.includes("GPU") ? "GPU" : "CPU";
    showError(
      `This text has ${clientSentences.length} sentences. ` +
      `The current ${modeText} backend is limited to ${MAX_SENTENCES} sentences per request. ` +
      `Please shorten your text or analyse a passage at a time.`
    );
    return;
  }

  errorBanner.classList.add("hidden");
  loadingEl.classList.remove("hidden");
  btnAnalyse.disabled = true;

  try {
    const res = await fetch(`${BACKEND_URL}/analyse/batch`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ sentences: splitSentences(text) }),
    });

    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      // Sentence cap error — surface it clearly
      throw new Error(detail.detail || `HTTP ${res.status}`);
    }

    const results = await res.json();  // [{sentence, region, confidence}, …]
    renderHighlights(results);

  } catch (err) {
    console.error("Analysis failed:", err);
    showError(`Analysis failed: ${err.message}. Make sure the backend is running at ${BACKEND_URL}`);
  } finally {
    loadingEl.classList.add("hidden");
    btnAnalyse.disabled = false;
  }
}

// ── Render highlights ──────────────────────────────────────────────────────

function renderHighlights(results) {
  outputEl.innerHTML = "";
  const regionCounts = {};

  results.forEach((item, idx) => {
    const meta = REGION_META[item.region] || REGION_META.Neutral;
    const span = document.createElement("span");

    span.className = ["sentence-span", meta.cls].filter(Boolean).join(" ");
    span.textContent = item.sentence + " ";

    // Data attributes
    span.dataset.region     = item.region || "Neutral";
    span.dataset.confidence = (item.confidence * 100).toFixed(1);
    span.dataset.idx        = idx;

    // Accessibility
    span.setAttribute("role", "button");
    span.setAttribute("tabindex", "0");
    span.setAttribute("aria-label",
      `${item.sentence} — ${meta.label}, ${(item.confidence * 100).toFixed(0)}% confidence`
    );

    // Stagger animation
    span.style.animationDelay = `${idx * 40}ms`;
    span.classList.add("sentence-enter");

    // Interaction
    span.addEventListener("click", e => {
      e.stopPropagation();
      showTooltip(span, meta, item.confidence);
    });
    span.addEventListener("keydown", e => {
      if (e.key === "Enter" || e.key === " ") showTooltip(span, meta, item.confidence);
    });

    outputEl.appendChild(span);
    regionCounts[item.region] = (regionCounts[item.region] || 0) + 1;
  });

  // Footer region summary (sorted by count)
  const summary = Object.entries(regionCounts)
    .sort((a, b) => b[1] - a[1])
    .map(([r, n]) => `${n} ${r}`)
    .join(" · ");
  regionSumEl.textContent = summary;

  // Update counts (word count from saved text, sentence count from results)
  const words = editorEl.value.trim().split(/\s+/).filter(Boolean).length;
  wordCountEl.textContent = `${words} word${words !== 1 ? "s" : ""}`;
  sentCountEl.textContent = `${results.length} sentence${results.length !== 1 ? "s" : ""}`;

  // Switch to output view
  editorWrapper.classList.add("hidden");
  outputWrapper.classList.remove("hidden");
  closeTooltip();
}

// ── Tooltip / Popover ──────────────────────────────────────────────────────

function showTooltip(spanEl, meta, confidence) {
  tooltipRegion.textContent = meta.label;
  tooltipConf.textContent   = `${(confidence * 100).toFixed(1)}% confidence`;
  tooltipExpl.textContent   = meta.explanation;

  // Set colour accent on tooltip region label based on region
  const colorMap = {
    "region-tpj":   "#BA7517",
    "region-broca": "#7F77DD",
    "region-sts":   "#1D9E75",
    "region-dmn":   "#378ADD",
  };
  tooltipRegion.style.color = colorMap[meta.cls] || "hsl(250, 50%, 35%)";

  // Smart positioning — keep within viewport
  tooltipEl.classList.remove("hidden");

  const rect      = spanEl.getBoundingClientRect();
  const tipWidth  = tooltipEl.offsetWidth  || 320;
  const tipHeight = tooltipEl.offsetHeight || 120;
  const scrollY   = window.scrollY;
  const scrollX   = window.scrollX;
  const vw        = window.innerWidth;
  const vh        = window.innerHeight;

  let top  = rect.bottom + scrollY + 10;
  let left = rect.left   + scrollX;

  // Flip upward if tooltip would clip below viewport
  if (rect.bottom + tipHeight + 10 > vh) {
    top = rect.top + scrollY - tipHeight - 10;
  }

  // Clip horizontal to keep within viewport
  if (left + tipWidth > scrollX + vw - 16) {
    left = scrollX + vw - tipWidth - 16;
  }
  if (left < scrollX + 16) left = scrollX + 16;

  tooltipEl.style.top  = `${top}px`;
  tooltipEl.style.left = `${left}px`;
}

function closeTooltip() {
  tooltipEl.classList.add("hidden");
}

// Close tooltip when clicking elsewhere
document.addEventListener("click", e => {
  if (!tooltipEl.contains(e.target)) closeTooltip();
});

// ── Editor controls ────────────────────────────────────────────────────────

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

// ── Init ───────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  checkHealth();
  updateCounts();
});
