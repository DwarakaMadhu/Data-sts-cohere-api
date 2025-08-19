
"""
api.py
------
FastAPI app exposing the required endpoint:
POST /predict
Request body:
{ "text1": "...", "text2": "..." }
Response body:
{ "similarity score": 0.123 }
Run locally:
    pip install -r requirements.txt
    uvicorn api:app --host 0.0.0.0 --port 8000
Environment:
- COHERE_API_KEY: set to call Cohere (or USE_LOCAL_BASELINE=1 for offline TF-IDF)
"""
import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from similarity import similarity_score

class PredictRequest(BaseModel):
    text1: str
    text2: str

app = FastAPI(title="Semantic Textual Similarity API", version="1.0.0")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    text1 = (req.text1 or "").strip()
    text2 = (req.text2 or "").strip()
    if not text1 or not text2:
        raise HTTPException(status_code=400, detail="text1 and text2 must be non-empty strings.")
    score = similarity_score(text1, text2)
    return {"similarity score": float(score)}
