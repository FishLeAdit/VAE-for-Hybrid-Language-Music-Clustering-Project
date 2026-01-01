#!/usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

MANIFEST = "data/manifest_english_only.csv"

# audio latents (ConvVAE)
AUDIO_Z = "data/audio_latents_convvae.npy"
AUDIO_IDS = "data/audio_track_ids_convvae.npy"

# lyrics embeddings
LYR_X = "data/lyrics_tfidf.npy"
LYR_IDS = "data/lyrics_tfidf_track_ids.npy"

# outputs
OUT_FUSED = "data/fused_audio_lyrics.npy"
OUT_FUSED_IDS = "data/fused_track_ids.npy"

LYR_PCA_DIM = 128  # reduce TF-IDF to something manageable

def main():
    df = pd.read_csv(MANIFEST)
    manifest_ids = set(df["track_id"].astype(int).tolist())

    Z = np.load(AUDIO_Z)
    z_ids = np.load(AUDIO_IDS).astype(int)

    Xl = np.load(LYR_X)
    l_ids = np.load(LYR_IDS).astype(int)

    # build dict for alignment
    z_map = {tid: Z[i] for i, tid in enumerate(z_ids) if tid in manifest_ids}
    l_map = {tid: Xl[i] for i, tid in enumerate(l_ids) if tid in manifest_ids}

    common = sorted(set(z_map.keys()) & set(l_map.keys()))
    if not common:
        raise RuntimeError("No overlapping track_ids between audio latents and lyric embeddings.")

    Zc = np.vstack([z_map[i] for i in common]).astype(np.float32)
    Lc = np.vstack([l_map[i] for i in common]).astype(np.float32)

    # scale both spaces
    Zc = StandardScaler().fit_transform(Zc).astype(np.float32)

    # reduce lyrics dimension then scale
    Lp = PCA(n_components=min(LYR_PCA_DIM, Lc.shape[1]), random_state=42).fit_transform(Lc)
    Lp = StandardScaler().fit_transform(Lp).astype(np.float32)

    fused = np.hstack([Zc, Lp]).astype(np.float32)

    np.save(OUT_FUSED, fused)
    np.save(OUT_FUSED_IDS, np.array(common, dtype=np.int32))

    print("Fused shape:", fused.shape, "(audio_dim + lyrics_dim)")
    print("Saved:", OUT_FUSED)
    print("Saved:", OUT_FUSED_IDS)

if __name__ == "__main__":
    main()
