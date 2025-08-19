
# Short Report: Semantic Textual Similarity (STS) – Cohere + FastAPI

**Candidate:** Madhura Dwaraka Mayee 
**Contact: 9900947943• Email_  dwarakamayee.884@gmail.com 
**Date:** 19-08-2025

---

## Part A – Modeling Approach (Unsupervised)

**Objective.** Given two paragraphs, output a score in **[0,1]** where **1 = highly similar** and **0 = highly dissimilar**.

**Method.**
1. **Embeddings:** Use **Cohere `embed-english-v3.0`** to convert each paragraph into a dense vector.
2. **Similarity:** Compute **cosine similarity** between the two vectors, then map from **[-1,1] → [0,1]** using `(cos + 1) / 2`.
3. **Why this is appropriate without labels:** Semantic embedding models are trained on large corpora and capture meaning; cosine distance naturally measures semantic closeness without any task‑specific labels.
4. **Text preprocessing:** Minimal (trim spaces); modern embeddings are robust to case and punctuation.
5. **Fallback (no API key/internet):** A **TF‑IDF** cosine similarity baseline ensures the pipeline remains runnable for testing and demos.

**Complexity & Efficiency.** Embedding computation is O(n·d) per text; cosine is O(d). Batch embedding reduces network overhead.

**Optional calibration.** If needed, score distribution can be calibrated using min‑max scaling from the training CSV to spread scores more evenly in [0,1].

---

## Part B – Deployment as an API

**Framework:** **FastAPI** + **Uvicorn**.  
**Endpoint:** `POST /predict`  
**Request body:**
```json
{ "text1": "....", "text2": "...." }
```
**Response body (exact key as required):**
```json
{ "similarity score": 0.123 }
```

**Environment:**
- `COHERE_API_KEY` (preferred, uses Cohere embeddings)
- or `USE_LOCAL_BASELINE=1` (runs offline TF‑IDF baseline)

**Cloud (example – Render free tier):**
1. Push the repo (`api.py`, `similarity.py`, `requirements.txt`) to GitHub.
2. Create a **Web Service** on Render, set start command `uvicorn api:app --host 0.0.0.0 --port 10000`.
3. Add `COHERE_API_KEY` in Environment. Deploy.
4. Share the live URL as your **Live API endpoint**.

**Validation.** Test with `curl` or Postman using the prescribed request/response format.

---

## Assumptions & Justifications

- **No labels provided.** An **unsupervised** approach using pretrained embeddings is appropriate.
- **Generalization.** Using large, general-purpose sentence embeddings ensures transfer to the test set drawn from the same distribution.
- **Constraints.** Only Python used; solution is lightweight and deployable on free tiers.

---

## Files Delivered

- **Part A:** `similarity.py`, `offline_evaluate.py`
- **Part B:** `api.py`, `requirements.txt`
- **Report:** this document
- **(User to attach) Updated resume with contact number**
