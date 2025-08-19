
# STS (Semantic Textual Similarity) – Cohere + FastAPI (Free Plan)

This project solves **Part A** (compute semantic similarity 0–1 for two paragraphs) and **Part B** (deploy a server API endpoint) using **free APIs** and a **local fallback**.

---

## Part A: Modeling Approach (Unsupervised)

**Core idea:** use **Cohere embeddings** (`embed-english-v3.0`) to encode each paragraph, then compute **cosine similarity** and map it from **[-1, 1] to [0, 1]** via `(cos + 1) / 2`.

Why this works:
- Embedding models trained on large corpora capture **semantic meaning**.
- Cosine similarity reflects how close two meanings are in high-dimensional space.
- No labels are required for training — this is **unsupervised**, perfect for the provided dataset.

### Local fallback (no API key / no internet)
If `COHERE_API_KEY` is not set or API fails, we gracefully fall back to a **TF‑IDF** cosine similarity that runs fully offline. It’s weaker than embeddings but keeps the pipeline operational for testing and demos.

---

##  Quickstart (Local)

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **(Recommended) Set your Cohere key**
   ```bash
   export COHERE_API_KEY=your_key_here   # macOS/Linux
   set COHERE_API_KEY=your_key_here      # Windows (CMD)
   ```

   > No key? Use the offline baseline:
   > ```bash
   > export USE_LOCAL_BASELINE=1
   > ```

3. **Run the API**
   ```bash
   uvicorn api:app --host 0.0.0.0 --port 8000
   ```

4. **Test the API**
   ```bash
   curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text1":"nuclear body seeks new tech","text2":"terror suspects face arrest"}'
   ```
   Response:
   ```json
   { "similarity score": 0.2 }
   ```

---

## Using the Attached Dataset (Part A demo)

To compute scores for the whole CSV (must have columns `text1,text2`):

```bash
python offline_evaluate.py --data /path/to/DataNeuron_Text_Similarity.csv --out scores.csv
```

This writes `scores.csv` with an extra column named **`similarity score`**.

---

##  Part B: Deploy as a Cloud API (Free)

Below is a zero-cost path using **Render** free tier (alternatives: Railway, Fly.io, Cloud Run free).

1. **Create a GitHub repo** and push these files:
   - `api.py`, `similarity.py`, `requirements.txt`
2. On **Render**:
   - Create **New + Web Service** → Connect your repo
   - **Runtime**: Python 3.10+
   - **Start command**: `uvicorn api:app --host 0.0.0.0 --port 10000`
   - **Port**: `10000` (or the default Render port env var)
   - **Environment** → Add `COHERE_API_KEY` (or `USE_LOCAL_BASELINE=1` for offline fallback)
   - Deploy
3. **Your live endpoint** will look like:
   ```
   https://your-app.onrender.com/predict
   ```

**Request body (required format):**
```json
{ "text1": "......", "text2": "......" }
```

**Response body (required key name):**
```json
{ "similarity score": 0.42 }
```

---

##  Notes & Tips

- Preprocessing is intentionally light — modern embeddings handle casing/punctuation well.
- If you want score calibration (optional), compute mean/variance of cosine scores on the training CSV and apply **min‑max** or **z‑score** calibration to better spread scores in [0,1].
- Throughput: batch requests to Cohere (`client.embed(texts=[...])`) for speed in offline evaluation scripts.

---

##  Project Structure

```
.
├── api.py                 # FastAPI server (Part B)
├── similarity.py          # Cohere embeddings + local TF-IDF fallback (Part A)
├── offline_evaluate.py    # Runs scores over the CSV
├── requirements.txt
└── README.md
```
