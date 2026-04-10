"""
Emotional Weight — FastAPI Backend (Phase 3)
=============================================
Full inference + ROI scoring pipeline.

Pipeline per sentence:
    text  →  TTS via model.get_events_dataframe()
          →  model.predict()  →  (n_timesteps, 20484) float32
          →  mean across timesteps  →  (20484,) activation vector
          →  roi_scorer.score_regions()  →  {region: softmax_prob}
          →  roi_scorer.classify()       →  (winning_region, confidence)
          →  {sentence, region, confidence}

License: CC-BY-NC-4.0 (non-commercial use only)
Model:   facebook/tribev2  —  d'Ascoli et al., Meta FAIR, 2026
"""

import os
import sys
import logging
import tempfile
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import nltk
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Ensure project root is on the path so `data/` is importable ──────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data.roi_map import REGION_VERTICES          # {region: [vertex_indices]}
from backend.roi_scorer import score_regions, classify  # Phase 3 scoring layer

# ── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
log = logging.getLogger("emotional_weight")

# ── NLTK punkt tokenizer ─────────────────────────────────────────────────
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── Global model handle (populated in lifespan) ───────────────────────────
_model: Any = None


# ── Lifespan — model loaded once, kept in memory ─────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load TRIBE v2 at startup; release on shutdown.

    TribeModel.from_pretrained() downloads / reads from cache_folder.
    At ~7 GB this takes 60–120 s on first run; subsequent starts use cache.

    Set HF_TOKEN env var if the HuggingFace repo is gated:
        export HF_TOKEN=hf_...
    """
    global _model
    log.info("🧠 Loading TRIBE v2 model…  (this may take a while on first run)")
    try:
        from tribev2 import TribeModel  # pip install -e git+…/tribev2
        _model = TribeModel.from_pretrained(
            "facebook/tribev2",
            cache_folder=os.path.join(_ROOT, "cache"),
        )
        log.info("✅ TRIBE v2 model ready.")
    except ImportError:
        log.warning(
            "tribev2 is not installed — inference will return placeholder data. "
            "Install with: pip install -e git+https://github.com/facebookresearch/tribev2.git#egg=tribev2"
        )
        _model = None
    except Exception as exc:
        log.error(f"Failed to load TRIBE v2: {exc}")
        _model = None

    yield  # ←── app runs here

    log.info("🛑 Shutting down. Releasing model.")
    _model = None


# ── App ───────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Emotional Weight API",
    description=(
        "Brain-aware text analysis powered by TRIBE v2. "
        "Predicts cortical activation patterns from text and maps them "
        "to named brain regions (HCP Glasser parcellation)."
    ),
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten to specific origins in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Request / response schemas ────────────────────────────────────────────

class AnalyseRequest(BaseModel):
    text: str


class BatchAnalyseRequest(BaseModel):
    sentences: list[str]


class SentenceResult(BaseModel):
    sentence:   str
    region:     str
    confidence: float   # 0.0 – 1.0  (normalised score for the winning region)


# ── Core inference helpers ────────────────────────────────────────────────

def run_tribe_on_text(sentence: str) -> np.ndarray:
    """
    Run TRIBE v2 on a single sentence and return the mean vertex activation.

    Steps
    -----
    1.  Write sentence to a temporary .txt file (required by TRIBE's TTS pipeline).
    2.  Call model.get_events_dataframe(text_path=…) to produce word-level
        timing events; TRIBE v2 handles TTS internally.
    3.  Call model.predict(events=df) → ndarray of shape (n_timesteps, n_vertices).
        n_vertices = 20,484  (fsaverage5, LH 0–10241, RH 10242–20483)
    4.  Average across the time dimension → shape (n_vertices,).
    5.  Return that 1-D activation vector.

    Returns
    -------
    np.ndarray, shape (20484,), dtype float32
        Mean cortical activation per fsaverage5 vertex.
        Falls back to a zero vector if the model is not loaded.
    """
    N_VERTICES = 20_484

    if _model is None:
        log.warning("Model not loaded — returning zero activation vector.")
        return np.zeros(N_VERTICES, dtype=np.float32)

    # Write sentence to a named temp file (TRIBE expects a file path)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".txt",
        prefix="tmp_sentence_",
        delete=False,
        encoding="utf-8",
    ) as f:
        f.write(sentence)
        tmp_path = f.name

    try:
        # Step 2 — word-level timing events via built-in TTS
        events_df = _model.get_events_dataframe(text_path=tmp_path)

        # Step 3 — cortical prediction: (n_timesteps, n_vertices)
        preds = _model.predict(events=events_df)   # numpy array

        # Validate shape
        if preds.ndim != 2 or preds.shape[1] != N_VERTICES:
            raise ValueError(
                f"Unexpected prediction shape from TRIBE v2: {preds.shape}. "
                f"Expected (n_timesteps, {N_VERTICES})."
            )

        # Step 4 — collapse time dimension
        mean_activation: np.ndarray = preds.mean(axis=0).astype(np.float32)
        return mean_activation   # shape (20484,)

    finally:
        # Always clean up the temp file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# score_regions() and classify() are now imported from backend.roi_scorer (Phase 3).
# They replace the preliminary min-max scoring used in Phase 2.


def _process_sentences(sentences: list[str]) -> list[SentenceResult]:
    """
    Shared logic: run TRIBE v2 inference on each sentence, score via ROI
    scorer (Phase 3), return structured results.
    """
    results = []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        activations          = run_tribe_on_text(sent)         # (20484,) float32
        region, confidence   = classify(activations)           # roi_scorer.classify
        all_scores           = score_regions(activations)      # full probability dict
        results.append(SentenceResult(
            sentence=sent,
            region=region,
            confidence=round(confidence, 4),
        ))
        log.debug(f"  → {region} ({confidence:.3f}) | {all_scores}")
    return results


# ── Routes ────────────────────────────────────────────────────────────────

@app.get("/health", tags=["status"])
async def health():
    """
    Health-check endpoint.

    Returns model_loaded: true once TRIBE v2 is ready to serve requests.
    The frontend polls this on startup to show the correct status indicator.
    """
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "version": "0.2.0",
    }


@app.post("/analyse", response_model=list[SentenceResult], tags=["inference"])
async def analyse(req: AnalyseRequest):
    """
    Analyse a block of text.

    Splits the input into sentences using nltk.sent_tokenize, runs
    TRIBE v2 on each, scores the resulting vertex activation against
    the Glasser ROI lookup, and returns one result per sentence.

    Request body
    ------------
    {
        "text": "Your paragraph here. As many sentences as you like."
    }

    Response
    --------
    [
        {"sentence": "…", "region": "TPJ", "confidence": 0.84},
        …
    ]
    """
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="text must not be empty.")

    sentences = nltk.sent_tokenize(text)
    if not sentences:
        raise HTTPException(status_code=422, detail="Could not extract sentences from input.")

    log.info(f"/analyse — {len(sentences)} sentence(s) received.")
    return _process_sentences(sentences)


@app.post("/analyse/batch", response_model=list[SentenceResult], tags=["inference"])
async def analyse_batch(req: BatchAnalyseRequest):
    """
    Analyse a pre-split list of sentences.

    Accepts sentences already tokenised by the frontend, avoiding
    redundant tokenisation overhead. Useful when the frontend needs
    precise control over sentence boundaries (e.g. for preserving
    paragraph breaks in the highlighted output).

    Request body
    ------------
    {
        "sentences": ["First sentence.", "Second sentence.", …]
    }

    Response
    --------
    Same schema as /analyse.
    """
    sentences = [s.strip() for s in req.sentences if s.strip()]
    if not sentences:
        raise HTTPException(status_code=422, detail="sentences list must not be empty.")

    log.info(f"/analyse/batch — {len(sentences)} sentence(s) received.")
    return _process_sentences(sentences)
