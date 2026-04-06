# Emotional Weight 🧠

> A brain-aware text editor for creative writers — powered by **TRIBE v2** (d'Ascoli et al., Meta FAIR, 2026).

As you write, each sentence is scored against predicted fMRI activation across 20,484 fsaverage5 cortical surface vertices. The dominant brain region for each sentence is highlighted with a meaningful colour, giving writers real-time intuition about the cognitive texture of their prose.

---

## Neuroscience Motivation

TRIBE v2 is a multimodal brain-encoding model trained on fMRI data from subjects listening to naturalistic speech. Given a sentence (converted to audio via TTS), it predicts vertex-wise BOLD activation across the entire cortical surface using the HCP MMP Glasser parcellation (360 parcels, both hemispheres).

This project uses TRIBE v2 in **text-only mode**: sentences are fed through the model's built-in TTS pipeline, producing word-level timing events, which are then used to drive cortical predictions. The resulting activation map is summarised per brain region using a hand-curated ROI lookup built on the Glasser atlas.

### Why does region matter for writing?

| Region | Glasser Parcels | What lights up |
|--------|-----------------|----------------|
| **TPJ / Angular Gyrus** | PGi, TE1a | Emotional salience, Theory of Mind, character attribution |
| **Broca Area / IFG** | 44, 45, IFSp | Syntactic complexity, nested clauses, working memory for grammar |
| **STS / Auditory** | STSva, STSvp | Speech prosody, voice imagery in silent reading |
| **Default Mode / PFC** | d32, 10pp, 10d | Semantic integration, narrative continuity, self-referential meaning |
| **Neutral** | — | Balanced or low activation across all tracked regions |

A sentence dominated by **TPJ** activation reads as emotionally charged. One that lights up **Broca** tends to be syntactically dense. **STS** activation signals speakable, voice-like prose. **DMN** activation marks sentences that feel meaningful or world-building in a narrative sense.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER BROWSER                                │
│   GitHub Pages frontend (HTML + CSS + Vanilla JS)                   │
│   ┌──────────────────────────┐  ┌──────────────────────────────┐   │
│   │  Writing textarea        │  │  Brain Region Legend Panel   │   │
│   │  Highlighted output      │  │  (TPJ / Broca / STS / DMN)  │   │
│   │  Sentence tooltip popover│  │                              │   │
│   └─────────────┬────────────┘  └──────────────────────────────┘   │
└─────────────────┼───────────────────────────────────────────────────┘
                  │  POST /analyse/batch
                  │  {"sentences": [...]}
                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│         HuggingFace Spaces (Docker, Free T4 GPU)                    │
│         FastAPI / uvicorn — backend/app.py                          │
│                                                                     │
│  ① nltk.sent_tokenize(text)                                         │
│       │                                                             │
│  ② run_tribe_on_text(sentence)  ──► TRIBE v2 model                 │
│       │   • write sentence to temp .txt                             │
│       │   • model.get_events_dataframe(text_path=...)  (TTS)       │
│       │   • model.predict(events=df)                                │
│       │     → (n_timesteps, 20484) float array                     │
│       │   • mean across timesteps → (20484,) activation vector     │
│       │                                                             │
│  ③ score_regions(vertex_activations)  ──► roi_scorer.py            │
│       │   • load REGION_VERTICES from data/roi_map.py              │
│       │   • mean activation per region                             │
│       │   • softmax normalise → {region: score}                    │
│       │                                                             │
│  ④ return [{sentence, region, confidence}, ...]                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
emotions/
├── backend/
│   ├── app.py           # FastAPI app — model loading, /analyse endpoints
│   ├── roi_scorer.py    # Vertex activation → region label (Phase 3)
│   └── requirements.txt
├── data/
│   ├── roi_map.py           # Hardcoded Glasser parcel → vertex index lookup
│   └── fetch_roi_indices.py # One-time setup: re-parse atlas & update roi_map.py
├── frontend/
│   ├── index.html       # Full writing UI shell
│   ├── style.css        # Design system & region highlight styles
│   └── app.js           # BACKEND_URL config + all frontend logic
├── tests/
│   └── test_roi_scorer.py   # Phase 3: pytest unit tests
├── Dockerfile               # Phase 5: HF Spaces Docker config
├── .env.example
├── .gitignore
└── README.md
```

---

## Setup — Local Development

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/emotional-weight.git
cd emotional-weight
```

### 2. Install TRIBE v2

```bash
pip install -e git+https://github.com/facebookresearch/tribev2.git#egg=tribev2
```

### 3. Install backend dependencies

```bash
pip install -r backend/requirements.txt
```

### 4. (One-time) Regenerate ROI vertex indices from atlas

If you want to refresh `data/roi_map.py` from the Glasser fsaverage5 annotation:

```bash
pip install templateflow   # or: pip install neuromaps
python data/fetch_roi_indices.py
```

Copy the printed index lists into `data/roi_map.py` under `_PARCEL_VERTICES`.

### 5. Download TRIBE v2 model weights

The model weights are hosted at `facebook/tribev2` on HuggingFace Hub.
Set your token if the repo is gated:

```bash
export HF_TOKEN=hf_...   # Windows: set HF_TOKEN=hf_...
```

### 6. Run the backend

```bash
uvicorn backend.app:app --reload --port 8000
```

Health-check: [http://localhost:8000/health](http://localhost:8000/health)

### 7. Open the frontend

Open `frontend/index.html` in your browser directly, or serve it:

```bash
npx serve frontend/
```

Make sure `BACKEND_URL` at the top of `frontend/app.js` is set to `http://localhost:8000`.

---

## Setup — Deployed

| Component | Platform | Config |
|-----------|----------|--------|
| Backend   | HuggingFace Spaces (Docker, T4 GPU) | Set `HF_TOKEN` as a Space secret |
| Frontend  | GitHub Pages (`/docs` folder or `gh-pages` branch) | Set `BACKEND_URL` in `app.js` to your HF Space URL |

See Phase 5 for the full deployment walkthrough.

---

## Running Tests (Phase 3+)

```bash
pytest tests/
```

---

## License

This project uses **TRIBE v2**, which is licensed under **CC-BY-NC-4.0** (Creative Commons Attribution–NonCommercial 4.0 International). Non-commercial use only.

See: [creativecommons.org/licenses/by-nc/4.0](https://creativecommons.org/licenses/by-nc/4.0/)

---

## Citation

If you use TRIBE v2 in your work, please cite:

```bibtex
@article{dascoli2026tribev2,
  title   = {TRIBE v2: Scaling Text-Driven Brain Encoding to the Full Cortical Surface},
  author  = {d'Ascoli, St\'{e}phane and others},
  journal = {Meta FAIR Technical Report},
  year    = {2026},
  url     = {https://github.com/facebookresearch/tribev2}
}
```

And the HCP Glasser parcellation:

```bibtex
@article{glasser2016multimodal,
  title   = {A multi-modal parcellation of human cerebral cortex},
  author  = {Glasser, Matthew F and others},
  journal = {Nature},
  volume  = {536},
  pages   = {171--178},
  year    = {2016},
  doi     = {10.1038/nature18933}
}
```

---

*Emotional Weight is a portfolio project demonstrating real-time neuroscientific text analysis. It is not a clinical tool.*
