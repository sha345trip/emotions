"""
Emotional Weight — Modal GPU Backend
====================================
High-performance inference using NVIDIA T4 GPUs.
"""

import os
import modal
from typing import List

# 1. Define the Container Image (Using Modal 1.0 API - v1.0.1)
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

app = modal.App("emotional-weight-gpu")

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
    def predict_batch(self, sentences: list[str]):
        """Run inference on a batch of sentences and return a list of mean activations."""
        import numpy as np
        import tempfile
        import os
        
        results = []
        for sentence in sentences:
            # TRIBE v2 requires a .txt file path
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
                f.write(sentence)
                tmp_path = f.name
            f.close()

            try:
                events_df = self.model.get_events_dataframe(text_path=tmp_path)
                preds = self.model.predict(events=events_df)
                
                # [FIX] Handle tuple return type (activations, states) sometimes returned by TRIBE v2
                if isinstance(preds, tuple):
                    preds = preds[0]
                
                if preds is None or (not hasattr(preds, 'shape')) or preds.shape[0] == 0:
                    results.append(np.zeros(20484, dtype=np.float32).tolist())
                else:
                    mean_activation = preds.mean(axis=0).astype(np.float32)
                    results.append(mean_activation.tolist())
            except Exception as e:
                import traceback
                traceback.print_exc()
                results.append(None)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                    
        return results

@app.function(image=image, timeout=600)
@modal.asgi_app()
def fastapi_app():
    import asyncio
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import numpy as np
    
    # Precise imports from regional project folders
    from data.roi_map import REGION_VERTICES
    from backend.roi_scorer import score_regions, classify

    web_app = FastAPI(title="Emotional Weight GPU Backend")
    
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://sha345trip.github.io", "http://localhost:8000", "http://127.0.0.1:8000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @web_app.get("/")
    async def index():
        return {"message": "Emotional Weight GPU API is online. Query /analyse/batch via POST."}

    @web_app.get("/health")
    async def health():
        return {"status": "ok", "mode": "Modal-GPU-T4"}

    class AnalysisRequest(BaseModel):
        sentences: List[str]

    @web_app.post("/analyse")
    @web_app.post("/analyse/batch")
    async def analyse(request: AnalysisRequest):
        model = EmotionalWeightModel()
        try:
            sentences_clean = [s.strip() for s in request.sentences if s.strip()]
            if not sentences_clean:
                return []
                
            # Send entire batch to a single GPU worker sequentially 
            # (avoids blowing up concurrency / hitting modal timeouts)
            activation_lists = await model.predict_batch.remote.aio(sentences_clean)
            
            results = []
            for sent, activation_list in zip(sentences_clean, activation_lists):
                if activation_list is None:
                    continue
                
                activation_vec  = np.array(activation_list, dtype=np.float32)
                scores = score_regions(activation_vec)
                top_region, confidence = classify(activation_vec)
                
                results.append({
                    "sentence": sent,
                    "region": top_region,
                    "confidence": float(confidence),
                    "all_scores": {k: float(v) for k, v in scores.items()}
                })
            
            return results
        except Exception as e:
            import traceback
            traceback.print_exc()
            from fastapi import HTTPException
            raise HTTPException(status_code=500, detail=str(e))

    return web_app
