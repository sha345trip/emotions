You are helping me build a browser-based web app called "Emotional Weight" — a brain-aware text editor for creative writers that uses the REAL TRIBE v2 model (d'Ascoli et al., Meta FAIR, 2026) to predict brain region activation from text input.

---

WHAT THE APP DOES:
As a user types in a text editor, individual sentences are sent to a FastAPI backend. The backend runs TRIBE v2 in text-only mode, gets vertex-wise fMRI predictions across the fsaverage5 cortical surface (~20k vertices), then maps those predictions onto named brain regions using a pre-defined ROI lookup. The app returns a label and confidence per sentence, which the frontend uses to highlight sentences in different colors.

BRAIN REGION → COLOR MAPPING (from the paper's findings):
- TPJ / MTG (emotional processing) → amber highlight
- Broca area / IFG (syntactic complexity) → purple highlight  
- STS / A5 (speech-like, auditory language) → teal highlight
- Default mode / prefrontal (semantic, narrative) → blue highlight
- Neutral / no dominant region → no highlight

ROI VERTEX INDICES come from the HCP parcellation (Glasser et al., 2016) which TRIBE v2 uses. Specifically:
- TPJ → parcel "PGi" (label index in fsaverage5 atlas)
- Broca → parcels "44", "45", "IFSp"
- STS → parcels "STSva", "STSvp"
- MTG → parcel "TE1a"
- DMN / prefrontal → parcels "d32", "10pp", "10d"

For each sentence, the "winning" region is the one whose mean predicted activation (across its vertices) is highest relative to baseline.

---

STACK:
- Backend: Python 3.10+, FastAPI, uvicorn
- Model: TRIBE v2 installed from github.com/facebookresearch/tribev2 via pip install -e .
- Model weights: facebook/tribev2 from HuggingFace (loaded once at startup, cached)
- Text-to-speech for word timings: use tribev2's built-in pipeline (model.get_events_dataframe accepts text_path)
- Frontend: Plain HTML + CSS + vanilla JavaScript — no frameworks
- Deployment: HuggingFace Spaces (free GPU tier, use a Space with T4 GPU) for backend, GitHub Pages for frontend
- License note: TRIBE v2 is CC-BY-NC-4.0 — non-commercial use only, acknowledge this in README

---

BUILD IN 5 PHASES. After each phase, tell me the exact commit message and stop — wait for me to confirm before proceeding to the next phase.

---

PHASE 1 — Project scaffold + ROI atlas setup
Tasks:
- Create folder structure: /backend, /frontend, /data
- In /backend: create app.py (FastAPI skeleton with /health endpoint), requirements.txt
- In /data: create roi_map.py — a Python dict mapping each HCP parcel name to its fsaverage5 vertex indices. Use nilearn or a hardcoded lookup based on the Glasser 360-parcel atlas. Include only the 10-15 parcels relevant to our 5 brain regions above.
- In /frontend: index.html with a textarea, an "Analyse" button, and placeholder colored spans
- README.md with: project description, the neuroscience motivation from the paper, architecture diagram (ASCII is fine), setup instructions, and a note on CC-BY-NC-4.0 license
- .gitignore: ignore .env, __pycache__, *.pyc, model cache folders, .wav and .txt temp files
- Commit message: "init: project scaffold, ROI atlas map, and static frontend shell"

PHASE 2 — TRIBE v2 local inference pipeline
Tasks:
- In backend/app.py, add model loading at startup:
    from tribev2 import TribeModel
    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")
  Wrap this in a lifespan context manager so it loads once and stays in memory.
- Add a helper function run_tribe_on_text(sentence: str) -> dict that:
    1. Writes the sentence to a temp .txt file
    2. Calls model.get_events_dataframe(text_path=...) to get word-level timings with auto TTS
    3. Calls model.predict(events=df) to get preds of shape (n_timesteps, n_vertices)
    4. Averages predictions across timesteps → shape (n_vertices,)
    5. Returns the mean activation per vertex as a numpy array
- Add a /analyse endpoint (POST, JSON body: {"text": "..."}) that:
    1. Splits input into sentences (use nltk.sent_tokenize)
    2. Calls run_tribe_on_text for each sentence
    3. Returns list of {sentence, region, confidence} objects
- Test this locally with a hardcoded example before moving on
- Commit message: "feat: TRIBE v2 model loading and text inference pipeline"

PHASE 3 — ROI scoring + region classification
Tasks:
- In backend/roi_scorer.py, implement score_regions(vertex_activations: np.ndarray) -> dict:
    1. Import the ROI vertex index dict from /data/roi_map.py
    2. For each region (TPJ, Broca, STS, MTG, DMN), compute mean activation across its vertices
    3. Normalise scores so they sum to 1 (softmax or simple normalisation)
    4. Return {region_name: score} dict
- In the /analyse endpoint, call score_regions on each sentence's vertex activations
- Return the top region + its score as "confidence" for each sentence
- Add a /analyse/batch endpoint that accepts multiple sentences in one call to avoid repeated model loading overhead
- Write a simple test script tests/test_roi_scorer.py that runs with random activations to verify the scoring logic
- Commit message: "feat: ROI scoring layer maps vertex activations to brain region labels"

PHASE 4 — Frontend editor with live sentence highlighting
Tasks:
- In frontend/index.html, build a clean distraction-free writing interface:
    · Full-width textarea with comfortable line height (not a code editor — a writing tool)
    · "Analyse text" button that POSTs to /analyse/batch
    · On response, replace textarea content with a styled 
 where each sentence is wrapped in a  with background color based on region
    · Show a loading indicator while waiting
- Color mapping in CSS:
    · TPJ/MTG → background: #FAEEDA (amber-50), border-left: 3px solid #BA7517
    · Broca → background: #EEEDFE (purple-50), border-left: 3px solid #7F77DD
    · STS → background: #E1F5EE (teal-50), border-left: 3px solid #1D9E75
    · DMN → background: #E6F1FB (blue-50), border-left: 3px solid #378ADD
    · Neutral → no highlight
- Add a fixed legend panel on the right showing each color, region name, and a 1-line plain English description of what that region does
- On click of a highlighted sentence, show a small tooltip/popover with: region name, confidence score (%), and a 2-sentence neuroscience explanation
- Add word count and sentence count in the footer
- Make the backend URL configurable via a JS const at the top of the file (BACKEND_URL) so it's easy to switch between local and deployed
- Commit message: "feat: sentence highlighting editor with region legend and click tooltips"

PHASE 5 — HuggingFace Spaces deployment
Tasks:
- Create a HuggingFace Space using the "Docker" SDK (not Gradio — we want full FastAPI control)
- In the repo root, create:
    · Dockerfile: base python:3.10-slim, install tribev2 and dependencies, expose port 7860, run uvicorn
    · README.md for the Space (HF Spaces uses the top of README for the Space card — add title, description, license CC-BY-NC-4.0)
    · .env.example with: HF_TOKEN (needed to download gated LLaMA 3.2 weights at Space startup)
- In the Space settings, set HF_TOKEN as a secret environment variable
- Update frontend/index.html: set BACKEND_URL to the deployed HF Space URL
- Push frontend/ to GitHub Pages (use the /docs folder approach or gh-pages branch)
- Update README.md in the main repo with:
    · Live demo link (HF Space URL)
    · GitHub Pages frontend link
    · Architecture diagram showing: User → GitHub Pages frontend → HF Space FastAPI → TRIBE v2 → ROI scorer → response
    · How to run locally
    · Citation for the TRIBE v2 paper
- Commit message: "deploy: HuggingFace Spaces Docker config and GitHub Pages frontend"

---

HARD CONSTRAINTS:
- Use real TRIBE v2 — no mocks, no substitutes in the final version
- Plain HTML/CSS/JS frontend only — no React, no bundlers
- Everything free — HF Spaces free GPU tier, GitHub Pages
- Every phase produces independently runnable code
- Add inline comments throughout — this is a public portfolio project
- Acknowledge CC-BY-NC-4.0 license and cite the paper in README
- After finishing each phase, say exactly: "Commit now → [commit message]" and wait for me to say 'next' before continuing

Create an implementation plan now.