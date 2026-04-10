# ──────────────────────────────────────────────────────────────────────────
# Emotional Weight — HuggingFace Spaces Dockerfile
# ──────────────────────────────────────────────────────────────────────────
# SDK:  Docker
# GPU:  T4 (free tier)
# Port: 7860 (HF Spaces default)
#
# Build notes:
#   - python:3.10-slim keeps the image lean; TRIBE v2 requires Python ≥ 3.10
#   - git is needed to pip-install tribev2 directly from GitHub
#   - Model weights are downloaded at first request, cached in /app/cache
#   - HF_TOKEN must be set as a Space secret in the HF Spaces settings
#     (required to authenticate with the gated facebook/tribev2 repo)
#
# License: CC-BY-NC-4.0  (non-commercial use only)
# ──────────────────────────────────────────────────────────────────────────

FROM python:3.10-slim

# System deps: git (for pip install from GitHub) + build tools for numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Copy project files ───────────────────────────────────────────────────
# Copy only what the backend needs; frontend is served via GitHub Pages
COPY backend/   ./backend/
COPY data/      ./data/

# ── Python dependencies ───────────────────────────────────────────────────
# Step 1: core requirements (fastapi, uvicorn, nltk, numpy, nibabel, etc.)
RUN pip install --no-cache-dir -r backend/requirements.txt

# Step 2: TRIBE v2 from Meta FAIR (CC-BY-NC-4.0)
# Install as editable so tribev2's internal imports resolve correctly
RUN pip install --no-cache-dir \
    "git+https://github.com/facebookresearch/tribev2.git#egg=tribev2"

# ── Environment ───────────────────────────────────────────────────────────
# HF_TOKEN is injected as a Space secret — do NOT hardcode here
ENV HF_TOKEN=""
# Tell huggingface_hub where to cache model weights
ENV HF_HOME="/app/cache"
# Ensure the project root is on PYTHONPATH so `data.roi_map` imports cleanly
ENV PYTHONPATH="/app"

# Pre-download NLTK punkt tokeniser data at build time to avoid cold-start delay
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

# ── Port ──────────────────────────────────────────────────────────────────
EXPOSE 7860

# ── Run ───────────────────────────────────────────────────────────────────
# Single worker — TRIBE v2 holds the GPU; multiple workers would conflict
CMD ["uvicorn", "backend.app:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1", \
     "--log-level", "info"]
