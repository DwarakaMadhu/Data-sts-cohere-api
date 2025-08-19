
"""
offline_evaluate.py
-------------------
Compute similarity scores for the attached dataset CSV and save as a new CSV.
By default, uses Cohere if COHERE_API_KEY is set; otherwise falls back to local TF-IDF.
Usage:
    python offline_evaluate.py --data /path/to/DataNeuron_Text_Similarity.csv --out scores.csv
"""
import argparse, os
import pandas as pd
from similarity import similarity_score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to input CSV with columns: text1,text2")
    ap.add_argument("--out", required=True, help="Path to write CSV with similarity scores")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    if not {"text1","text2"}.issubset(df.columns):
        raise ValueError("CSV must contain columns: text1, text2")
    scores = []
    for i, row in df.iterrows():
        s = similarity_score(str(row["text1"]), str(row["text2"]))
        scores.append(s)
        if (i+1) % 200 == 0:
            print(f"Processed {i+1} rows...")
    out_df = df.copy()
    out_df["similarity score"] = scores
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {len(out_df)} rows to {args.out}")

if __name__ == "__main__":
    main()
