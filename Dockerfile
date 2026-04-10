# ──────────────────────────────────────────────────────────────────────────
# Emotional Weight — HuggingFace Spaces Dockerfile (CPU tier)
# ──────────────────────────────────────────────────────────────────────────
# SDK:      Docker
# Hardware: Free CPU Basic (2 vCPU, 16 GB RAM)
# Port:     7860 (HF Spaces default)
#
# ⚠️  CPU inference note:
#   TRIBE v2 is a large transformer model. On the free CPU tier, predicting
#   cortical activation for a single sentence takes ~60–120 s.
#   The /analyse/batch endpoint caps requests at 3 sentences to stay within
#   reasonable response times (~3–6 min max).
#   For GPU-speed inference, upgrade to a T4 GPU Space.
#
# Build notes:
#   - python:3.11-slim keeps the image lean; TRIBE v2 requires Python ≥ 3.11
#   - git is needed to pip-install tribev2 directly from GitHub
#   - Model weights (~7 GB) are downloaded at first call and cached in /app/cache
#   - HF_TOKEN must be set as a Space secret (required for facebook/tribev2)
#   - PyTorch is fetched as CPU-only to avoid pulling the 2+ GB CUDA wheel
#
# License: CC-BY-NC-4.0  (non-commercial use only)
# ──────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System deps: git (pip install from GitHub) + build tools for numpy/scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install CPU-only PyTorch FIRST ───────────────────────────────────────
# Pulling the CPU wheel separately prevents pip from fetching the 2+ GB CUDA
# variant when tribev2 declares torch as a dependency.
RUN pip install --no-cache-dir \
    torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu

# ── Copy project files ───────────────────────────────────────────────────
COPY backend/   ./backend/
COPY data/      ./data/

# ── Python dependencies ───────────────────────────────────────────────────
RUN pip install --no-cache-dir -r backend/requirements.txt

# ── TRIBE v2 from Meta FAIR (CC-BY-NC-4.0) ───────────────────────────────
RUN pip install --no-cache-dir \
    "git+https://github.com/facebookresearch/tribev2.git#egg=tribev2"

# ── Environment ───────────────────────────────────────────────────────────
ENV HF_TOKEN=""
ENV HF_HOME="/app/cache"
ENV PYTHONPATH="/app"
# Use all available vCPUs for PyTorch intra-op parallelism on free CPU tier
ENV OMP_NUM_THREADS="2"
ENV MKL_NUM_THREADS="2"

# Pre-download NLTK data at build time (avoids cold-start delay)
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

# ── Port ──────────────────────────────────────────────────────────────────
EXPOSE 7860

# ── Run ───────────────────────────────────────────────────────────────────
CMD ["uvicorn", "backend.app:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1", \
     "--timeout-keep-alive", "600", \
     "--log-level", "info"]
