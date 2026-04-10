"""
Emotional Weight — Modal GPU Backend
====================================
High-performance inference using NVIDIA T4 GPUs.
"""

import os
import modal
from typing import List

# 1. Define the Container Image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "torch>=2.5.1",
        "transformers",
        "pandas",
        "numpy",
        "nltk",
        "scipy",
        "fastapi[standard]",
        "uv",
        "git+https://github.com/facebookresearch/tribev2.git#egg=tribev2"
    )
    .run_commands("pip install soundfile librosa nibabel matplotlib")
    .env({"HF_HOME": "/cache", "OMP_NUM_THREADS": "4"})
)

# 2. Define Persistent Storage (Volume) for 10GB+ of models
cache_vol = modal.Volume.from_name("emotional-weight-cache", create_if_missing=True)

app = modal.App("emotional-weight")

@app.cls(
    image=image,
    gpu="T4",
    volumes={"/cache": cache_vol},
    secrets=[modal.Secret.from_name("hf-token")],
    scaledown_window=300,
)
class EmotionalWeightModel:
    @modal.enter()
    def setup(self):
        """Pre-load models into GPU VRAM once at startup."""
        import torch
        import nltk
        from tribev2.main import TribeModel
        
        nltk.download("punkt",     quiet=True)
        nltk.download("punkt_tab", quiet=True)
        
        print("🧠 Loading TRIBE v2 and Llama 3.2 into T4 GPU...")
        self.model = TribeModel.from_pretrained("facebook/tribev2")
        print("✅ Models loaded and ready.")

    @modal.method()
    def predict_sentence(self, sentence: str):
        """Run inference on a single sentence using the GPU."""
        import pandas as pd
        import numpy as np
        
        events_df = self.model.get_events_dataframe(sentence)
        preds = self.model.predict(events=events_df)
        mean_activation = np.mean(preds, axis=0)
        return mean_activation.tolist()

@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import numpy as np
    
    # Imports for scoring (Now working because the file is in the project root)
    from data.roi_map import REGION_VERTICES
    from backend.roi_scorer import score_regions, classify

    web_app = FastAPI(title="Emotional Weight GPU API")
    web_app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    class AnalysisRequest(BaseModel):
        sentences: List[str]

    @web_app.post("/analyse/batch")
    async def analyse(request: AnalysisRequest):
        model = EmotionalWeightModel()
        results = []
        
        for sent in request.sentences:
            try:
                activation_vec = np.array(model.predict_sentence.remote(sent))
                scores = score_regions(activation_vec, REGION_VERTICES)
                top_region, confidence = classify(scores)
                
                results.append({
                    "sentence": sent,
                    "region": top_region,
                    "confidence": float(confidence),
                    "all_scores": {k: float(v) for k, v in scores.items()}
                })
            except Exception as e:
                print(f"Error processing sentence: {e}")
                results.append({"sentence": sent, "error": str(e)})
        
        return {"results": results}

    @web_app.get("/health")
    async def health():
        return {"status": "ok", "mode": "Modal-GPU-T4"}

    return web_app
