---
title: Emotional Weight
emoji: 🧠
colorFrom: yellow
colorTo: purple
sdk: docker
pinned: false
license: cc-by-nc-4.0
short_description: Brain-aware text editor highlighting sentences via TRIBE v2.
---

# Emotional Weight 🧠

> A brain-aware text editor for creative writers — powered by **TRIBE v2** (d'Ascoli et al., Meta FAIR, 2026).

As you write, each sentence is scored against predicted fMRI activation across 20,484 fsaverage5 cortical surface vertices. The dominant brain region for each sentence is highlighted with a meaningful colour, giving writers real-time intuition about the cognitive texture of their prose.

**Live demo:** [shanky1230.github.io/emotions](https://shanky1230.github.io/emotions) ← GitHub Pages frontend  
**API backend:** [shanky1230-emotional-weight.hf.space](https://shanky1230-emotional-weight.hf.space) ← HuggingFace Spaces (Docker, T4 GPU)

---

## Neuroscience Motivation

TRIBE v2 is a multimodal brain-encoding model trained on fMRI data from subjects listening to naturalistic speech. Given a sentence (converted to audio via TTS), it predicts vertex-wise BOLD activation across the entire cortical surface using the HCP MMP Glasser parcellation (360 parcels, both hemispheres).

This project uses TRIBE v2 in **text-only mode**: sentences are fed through the model's built-in TTS pipeline, producing word-level timing events, which are then used to drive cortical predictions. The resulting activation map is summarised per brain region using a hand-curated ROI lookup built on the Glasser atlas.

### Why does region matter for writing?

| Region | Glasser Parcels | Hemisphere | What activates |
|--------|-----------------|------------|----------------|
| **TPJ / Angular Gyrus** | PGi, TE1a | Bilateral | Emotional salience, Theory of Mind, character attribution |
| **Broca Area / IFG** | 44, 45, IFSp | Left only | Syntactic complexity, nested clauses, working memory for grammar |
| **STS / Auditory** | STSva, STSvp | Left only | Speech prosody, voice imagery in silent reading |
| **Default Mode / PFC** | d32, 10pp, 10d | Bilateral | Semantic integration, narrative continuity, self-referential meaning |
| **Neutral** | — | — | Balanced or low activation across all tracked regions |

A sentence dominated by **TPJ** activation reads as emotionally charged. One that lights up **Broca** tends to be syntactically dense. **STS** activation signals speakable, voice-like prose. **DMN** activation marks sentences that feel meaningful or world-building in a narrative sense.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│           WRITER'S BROWSER                                          │
│   GitHub Pages  →  shanky1230.github.io/emotions                   │
│                                                                     │
│   ┌──────────────────────────┐  ┌──────────────────────────────┐   │
│   │  Distraction-free        │  │  Brain Region Legend Panel   │   │
│   │  prose editor (Lora)     │  │  ┌─────┐ TPJ / MTG          │   │
│   │                          │  │  │yell.│ Emotional salience  │   │
│   │  "She looked out the     │  │  ├─────┤                    │   │
│   │   window and felt the    │  │  │purp.│ Broca / IFG        │   │
│   │   cold press of grief…"  │  │  ├─────┤                    │   │
│   │                          │  │  │teal │ STS / Auditory     │   │
│   │  [ Analyse text ]        │  │  ├─────┤                    │   │
│   └─────────────┬────────────┘  │  │blue │ Default Mode / PFC │   │
│                 │               └──────────────────────────────┘   │
│      Word count · Sentence count · Region summary footer           │
└─────────────────┼───────────────────────────────────────────────────┘
                  │  POST /analyse/batch
                  │  {"sentences": ["She looked out…", …]}
                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│   HuggingFace Spaces (Docker SDK, Free T4 GPU)                     │
│   shanky1230-emotional-weight.hf.space                             │
│   FastAPI / uvicorn — backend/app.py                               │
│                                                                     │
│  ① nltk.sent_tokenize(text) — authoritative sentence splitting     │
│       │                                                             │
│  ② run_tribe_on_text(sentence)  ──► TRIBE v2 (facebook/tribev2)   │
│       │   • write sentence → temp .txt                             │
│       │   • model.get_events_dataframe(text_path)  ← TTS pipeline  │
│       │   • model.predict(events=df)                               │
│       │     → ndarray (n_timesteps, 20 484) float32               │
│       │   • mean over time axis  →  (20 484,) activation vector   │
│       │                                                             │
│  ③ roi_scorer.score_regions(activations)  ──► backend/roi_scorer  │
│       │   • load REGION_VERTICES from data/roi_map.py             │
│       │   • mean activation per region over its vertex indices     │
│       │   • softmax normalise  →  {region: probability}           │
│       │                                                             │
│  ④ roi_scorer.classify(activations)                                │
│       │   • winning region + confidence                            │
│       │   • Neutral if below uniform baseline + margin             │
│       │                                                             │
│  ⑤ return [{sentence, region, confidence}, …]                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
emotions/
├── backend/
│   ├── app.py           # FastAPI app — model loading, inference, /analyse endpoints
│   ├── roi_scorer.py    # Softmax ROI scoring layer
│   ├── requirements.txt
│   └── __init__.py
├── data/
│   ├── roi_map.py           # Hardcoded Glasser parcel → fsaverage5 vertex lookup
│   └── fetch_roi_indices.py # One-time setup: re-parse atlas & regenerate roi_map.py
├── frontend/            # Local development frontend (BACKEND_URL = localhost)
│   ├── index.html
│   ├── style.css
│   └── app.js
├── docs/                # GitHub Pages deployment (BACKEND_URL = HF Space URL)
│   ├── index.html
│   ├── style.css
│   └── app.js
├── tests/
│   └── test_roi_scorer.py   # 23 pytest unit tests
├── Dockerfile               # HF Spaces Docker config
├── .env.example             # Required environment variables
├── .gitignore
└── README.md
```

---

## Setup — Local Development

### 1. Clone the repo

```bash
git clone https://github.com/shanky1230/emotions.git
cd emotions
```

### 2. Create and activate the virtual environment

```bash
python -m venv backend/venv

# Windows
backend\venv\Scripts\activate

# Linux / macOS
source backend/venv/bin/activate
```

### 3. Install backend dependencies

```bash
# Core requirements
pip install -r backend/requirements.txt

# TRIBE v2 from Meta FAIR (CC-BY-NC-4.0)
pip install -e git+https://github.com/facebookresearch/tribev2.git#egg=tribev2
```

### 4. Set your HuggingFace token

```bash
# Windows
set HF_TOKEN=hf_...

# Linux / macOS
export HF_TOKEN=hf_...
```

### 5. (One-time) Regenerate ROI vertex indices

To refresh `data/roi_map.py` from the Glasser fsaverage5 annotation:

```bash
pip install templateflow   # or: pip install neuromaps
python data/fetch_roi_indices.py
```

### 6. Run the backend

```bash
uvicorn backend.app:app --reload --port 8000
```

Health-check: [http://localhost:8000/health](http://localhost:8000/health)  
API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### 7. Open the frontend

Open `frontend/index.html` in your browser (or `npx serve frontend/`).  
`BACKEND_URL` in `frontend/app.js` is already set to `http://localhost:8000`.

---

## Setup — HuggingFace Spaces Deployment

1. **Create a new Space** at [huggingface.co/new-space](https://huggingface.co/new-space)
   - Space name: `emotional-weight`
   - SDK: **Docker**
   - Hardware: **T4 GPU** (free tier)

2. **Push this repo** as the Space repo:
   ```bash
   git remote add space https://huggingface.co/spaces/shanky1230/emotional-weight
   git push space main
   ```

3. **Set secrets** in Space → Settings → Repository secrets:
   - `HF_TOKEN` = your HuggingFace read token

4. The Space will build the Docker image, download TRIBE v2 weights at first startup, and expose the FastAPI server at `https://shanky1230-emotional-weight.hf.space`.

---

## Setup — GitHub Pages Deployment

The `docs/` folder is a production build of the frontend with `BACKEND_URL` pointing to the HF Space.

1. Push to GitHub
2. Go to repo **Settings → Pages**
3. Source: **Deploy from a branch** → branch: `main`, folder: `/docs`
4. GitHub Pages will serve the frontend at `https://shanky1230.github.io/emotions`

---

## Running Tests

```bash
cd emotions
.\backend\venv\Scripts\python.exe -m pytest tests/ -v
# 23 passed in ~0.5s
```

---

## License

This project uses **TRIBE v2**, licensed under **CC-BY-NC-4.0** (Creative Commons Attribution–NonCommercial 4.0 International). **Non-commercial use only.**

[creativecommons.org/licenses/by-nc/4.0](https://creativecommons.org/licenses/by-nc/4.0/)

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
