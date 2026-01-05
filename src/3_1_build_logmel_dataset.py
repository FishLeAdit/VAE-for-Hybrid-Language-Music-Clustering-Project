
from __future__ import annotations
import os, sys, warnings
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.spectrogram_utils import audio_to_logmel

MANIFEST = "data/fma_manifest_5k_5genres_lyrics_whisper_dropped_removed.csv"
AUDIO_ROOT = Path("data/fma_small/fma_small")

OUT_X = "data/logmel_X.npy"
OUT_IDS = "data/logmel_ids.npy"
OUT_META = "data/logmel_meta.csv"

@contextmanager
def quiet_stderr():
    old = sys.stderr
    try:
        with open(os.devnull, "w") as devnull:
            sys.stderr = devnull
            yield
    finally:
        sys.stderr = old

def track_id_to_path(track_id: int) -> Path:
    fname = f"{track_id:06d}.mp3"
    return AUDIO_ROOT / fname[:3] / fname

def main():
    df = pd.read_csv(MANIFEST)
    if "track_id" not in df.columns:
        raise SystemExit("Manifest must have track_id")

    track_ids = df["track_id"].astype(int).tolist()

    specs = []
    kept_rows = []
    kept_ids = []
    missing = 0
    failed = 0

    print(f"Building log-mel dataset from {len(track_ids)} manifest rows...")

    for tid in tqdm(track_ids, desc="log-mel", unit="track"):
        p = track_id_to_path(tid)
        if not p.exists():
            missing += 1
            continue
        try:
            with quiet_stderr():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    S = audio_to_logmel(str(p))  # should be (64, 215)
            if not np.isfinite(S).all():
                failed += 1
                continue
            specs.append(S)
            kept_ids.append(tid)
            kept_rows.append(df.loc[df["track_id"] == tid].iloc[0].to_dict())
        except Exception:
            failed += 1
            continue

    if not specs:
        raise RuntimeError("No log-mels extracted. Check paths/decoding.")

    X = np.stack(specs).astype(np.float32)            # (N, 64, 215)
    X = X[:, None, :, :]                              # (N, 1, 64, 215)
    ids = np.array(kept_ids, dtype=np.int32)
    meta = pd.DataFrame(kept_rows)

    np.save(OUT_X, X)
    np.save(OUT_IDS, ids)
    meta.to_csv(OUT_META, index=False)

    print("\n--- log-mel dataset built ---")
    print("rows in manifest:", len(df))
    print("saved X:", OUT_X, X.shape)
    print("saved ids:", OUT_IDS, ids.shape)
    print("saved meta:", OUT_META, meta.shape)
    print("missing files:", missing, "| decode/quality failed:", failed)
    print("----------------------------\n")

if __name__ == "__main__":
    main()
