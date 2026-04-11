"""
Emotional Weight — Modal GPU Backend
====================================
High-performance inference using NVIDIA T4 GPUs.
"""

import os
import modal
from typing import List

# 1. Define the Container Image (Using Modal 1.0 API)
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
    # Modal 1.0 way to include local folders
    .add_local_dir("data", remote_path="/root/data")
    .add_local_dir("backend", remote_path="/root/backend")
)

# 2. Define Persistent Storage (Volume)
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
        import os
        import torch
        import nltk
        from tribev2 import TribeModel
        
        # Authentication injection
        if os.environ.get("HF_TOKEN"):
             from huggingface_hub import login
             login(token=os.environ["HF_TOKEN"])
        
        nltk.download("punkt",     quiet=True)
        nltk.download("punkt_tab", quiet=True)
        
        print("🧠 Loading TRIBE v2 and Llama 3.2 into T4 GPU...")
        self.model = TribeModel.from_pretrained("facebook/tribev2")
        print("✅ Models loaded and ready.")

    @modal.method()
    def predict_sentence(self, sentence: str):
        """Run inference on a single sentence and return mean activation."""
        import numpy as np
        
        # TRIBE v2 handles string input directly in get_events_dataframe
        events_df = self.model.get_events_dataframe(sentence)
        preds = self.model.predict(events=events_df)
        
        if preds is None or preds.shape[0] == 0:
            return np.zeros(20484, dtype=np.float32).tolist()
            
        mean_activation = preds.mean(axis=0).astype(np.float32)
        return mean_activation.tolist()

@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import numpy as np
    
    # Precise imports from regional project folders
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
                # 1. GPU Prediction
                activation_vec = np.array(model.predict_sentence.remote(sent))
                
                # 2. ROI Scoring (Glasser mapping)
                scores = score_regions(activation_vec, REGION_VERTICES)
                
                # 3. Winning Region Discovery
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
