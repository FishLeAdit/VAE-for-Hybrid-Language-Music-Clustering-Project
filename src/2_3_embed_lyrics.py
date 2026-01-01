#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

IN_MANIFEST = "data/manifest_english_only.csv"
OUT_EMB = "data/lyrics_tfidf.npy"
OUT_IDS = "data/lyrics_tfidf_track_ids.npy"

MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)
MIN_DF = 2

def main():
    df = pd.read_csv(IN_MANIFEST)
    if "lyrics" not in df.columns:
        raise ValueError("Expected 'lyrics' column in manifest_english_only.csv")

    lyrics = df["lyrics"].fillna("").astype(str).tolist()
    ids = df["track_id"].astype(int).to_numpy()

    print("Building TF-IDF embeddings...")
    vec = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        stop_words="english",
        lowercase=True
    )
    X = vec.fit_transform(lyrics)  # sparse

    # Save as dense float32 for easy fusion
    X_dense = X.toarray().astype(np.float32)

    np.save(OUT_EMB, X_dense)
    np.save(OUT_IDS, ids.astype(np.int32))

    print("Saved:", OUT_EMB, X_dense.shape)
    print("Saved:", OUT_IDS, ids.shape)

if __name__ == "__main__":
    main()
