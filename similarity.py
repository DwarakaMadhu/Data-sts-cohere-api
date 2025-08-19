
"""
similarity.py
-------------
Core similarity functions.
- Primary approach: Cohere embeddings ("embed-english-v3.0") + cosine similarity -> [0,1]
- Fallback (no internet / no API key): Local TF-IDF cosine similarity -> [0,1]
Environment variables:
- COHERE_API_KEY: your Cohere API key (required for cloud inference)
- USE_LOCAL_BASELINE: set to "1" to force local TF-IDF baseline
"""
import os
import math
from typing import Tuple, List

import numpy as np

# Optional imports deferred for speed
_COHERE = None
def _lazy_import_cohere():
    global _COHERE
    if _COHERE is None:
        import cohere
        _COHERE = cohere
    return _COHERE

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def _to_unit_interval(x: float) -> float:
    # Map cosine in [-1, 1] to [0, 1]
    return max(0.0, min(1.0, (x + 1.0) / 2.0))

def cohere_embed_pair(text1: str, text2: str, model: str = "embed-english-v3.0") -> Tuple[np.ndarray, np.ndarray]:
    COHERE_API_KEY = os.getenv("COHERE_API_KEY", "").strip()
    if not COHERE_API_KEY:
        raise RuntimeError("COHERE_API_KEY not set")
    cohere = _lazy_import_cohere()
    client = cohere.Client(COHERE_API_KEY)
    # Batch both texts in a single request
    resp = client.embed(texts=[text1, text2], model=model, input_type="search_document")
    # Cohere returns a list of embeddings; convert to numpy arrays
    e1 = np.array(resp.embeddings[0], dtype="float32")
    e2 = np.array(resp.embeddings[1], dtype="float32")
    return e1, e2

def tfidf_similarity(text1: str, text2: str) -> float:
    """Local fallback: TF-IDF cosine on just the two texts."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    vect = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_df=1.0)
    X = vect.fit_transform([text1, text2])
    sim = cosine_similarity(X[0], X[1])[0,0]
    # cosine_similarity output is already in [0,1] for TF-IDF, but clamp just in case
    return float(max(0.0, min(1.0, sim)))

def cohere_similarity(text1: str, text2: str) -> float:
    e1, e2 = cohere_embed_pair(text1, text2)
    cos = _cosine(e1, e2)
    return _to_unit_interval(cos)

def similarity_score(text1: str, text2: str) -> float:
    """Public entry point used by the API. Respects USE_LOCAL_BASELINE env var."""
    if os.getenv("USE_LOCAL_BASELINE", "").strip() == "1":
        return tfidf_similarity(text1, text2)
    # Try Cohere, fall back to local if anything goes wrong
    try:
        return cohere_similarity(text1, text2)
    except Exception as e:
        # Fallback path
        return tfidf_similarity(text1, text2)

def batch_predict(text_pairs: List[Tuple[str, str]]) -> List[float]:
    """Convenience batch function (uses the same logic as similarity_score)."""
    scores = []
    for a, b in text_pairs:
        scores.append(similarity_score(a, b))
    return scores
