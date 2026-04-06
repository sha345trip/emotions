"""
Emotional Weight — FastAPI Backend
===================================
Brain-aware text analysis using TRIBE v2 (d'Ascoli et al., Meta FAIR, 2026).
Accepts text input, runs cortical surface fMRI predictions, maps to named
brain regions (HCP Glasser parcellation), returns per-sentence region labels.

License: CC-BY-NC-4.0 (non-commercial use only)
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


# ---------------------------------------------------------------------------
# Lifespan – model loading (Phase 2 will populate this)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load TRIBE v2 once at startup, keep in memory for the process lifetime.
    Phase 2 will add: model = TribeModel.from_pretrained(...)
    """
    # Phase 2: model loading goes here
    print("🧠 Emotional Weight backend starting…")
    yield
    # Cleanup on shutdown (if needed)
    print("🛑 Backend shutting down.")


# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Emotional Weight API",
    description=(
        "Brain-aware text analysis powered by TRIBE v2. "
        "Predicts cortical activation patterns from text and maps them "
        "to named brain regions (HCP Glasser parcellation)."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# Allow requests from GitHub Pages and local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tightened to specific origins in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["status"])
async def health():
    """
    Health-check endpoint.
    Returns 200 OK so the frontend (and HF Spaces) can confirm the backend
    is alive before sending analysis requests.
    """
    return {"status": "ok", "model_loaded": False}  # Phase 2: True when model ready


@app.post("/analyse", tags=["inference"])
async def analyse(payload: dict):
    """
    Phase 2 placeholder.

    Expected request body:
        {"text": "Your paragraph here."}

    Expected response:
        [{"sentence": "…", "region": "TPJ", "confidence": 0.72}, …]
    """
    # Phase 2: split into sentences, run TRIBE v2, call roi_scorer
    return {"detail": "Inference not yet implemented — Phase 2"}


@app.post("/analyse/batch", tags=["inference"])
async def analyse_batch(payload: dict):
    """
    Phase 3 placeholder.

    Expected request body:
        {"sentences": ["Sentence one.", "Sentence two.", …]}

    Returns same schema as /analyse but accepts pre-split sentences,
    avoiding redundant tokenisation overhead on the frontend.
    """
    # Phase 3: batch inference goes here
    return {"detail": "Batch inference not yet implemented — Phase 3"}
