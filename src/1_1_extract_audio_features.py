#!/usr/bin/env python3
import os
import sys
import csv
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

try:
    import audioread
    HAS_AUDIOREAD = True
except Exception:
    HAS_AUDIOREAD = False

MANIFEST = "data/fma_manifest_5k_5genres_lyrics_whisper_dropped_removed.csv"


FMA_ROOT = Path("data/fma_small/fma_small")
OUT_X = "data/audio_features_keptwhisper.npy"
OUT_IDS = "data/audio_track_ids_keptwhisper.npy"
OUT_LOG = "data/audio_extract_log_keptwhisper.csv"


# FEATURE SETTINGS
SR = 22050
DURATION = 30.0
N_MFCC = 20



# Helpers
@contextmanager
def suppress_stderr():
    old = sys.stderr
    try:
        with open(os.devnull, "w") as devnull:
            sys.stderr = devnull
            yield
    finally:
        sys.stderr = old


def track_id_to_path(track_id: int) -> Path:
    fname = f"{track_id:06d}.mp3"
    return FMA_ROOT / fname[:3] / fname


def decode_with_audioread(path: str) -> np.ndarray:
    samples = []
    with audioread.audio_open(path) as f:
        orig_sr = int(getattr(f, "samplerate", SR))
        ch = int(getattr(f, "channels", 1))
        for buf in f:
            a = np.frombuffer(buf, dtype=np.int16)
            samples.append(a)

    if not samples:
        raise ValueError("No decoded frames")

    pcm = np.concatenate(samples)
    if ch > 1:
        pcm = pcm.reshape(-1, ch).mean(axis=1)

    y = pcm.astype(np.float32) / 32768.0

    if orig_sr != SR:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=SR)

    return y[: int(DURATION * SR)]


def decode_with_librosa(path: str) -> np.ndarray:
    with suppress_stderr():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, _ = librosa.load(path, sr=SR, mono=True, duration=DURATION)
    return y.astype(np.float32)


def extract_mfcc(y: np.ndarray) -> np.ndarray:
    if y is None or len(y) == 0:
        raise ValueError("Empty audio")
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)]).astype(np.float32)


# Main
def main():
    if not Path(MANIFEST).exists():
        raise SystemExit(f"Manifest not found: {MANIFEST}")

    df = pd.read_csv(MANIFEST)

    # We EXPECT track_id to exist in the filtered manifest
    id_col = "track_id" if "track_id" in df.columns else None
    if id_col is None:
        raise ValueError(f"Manifest must contain 'track_id'. Found columns: {df.columns.tolist()}")

    feats, ids = [], []
    missing = bad = kept = 0

    # Some filtered manifests may have fewer rows; show that clearly
    print(f"\nStarting feature extraction on FILTERED tracks...")
    print(f"Manifest: {MANIFEST}")
    print(f"Rows in manifest: {len(df)}")
    print(f"Audioread available: {HAS_AUDIOREAD}\n")

    with open(OUT_LOG, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["track_id", "audio_path", "status", "decoder", "reason"]
        )
        writer.writeheader()

        for tid in tqdm(df[id_col].astype(int), desc="Extracting audio", unit="track"):
            p = track_id_to_path(tid)

            if not p.exists():
                missing += 1
                writer.writerow({"track_id": tid, "audio_path": str(p), "status": "missing", "decoder": "", "reason": ""})
                continue

            try:
                if HAS_AUDIOREAD:
                    try:
                        y = decode_with_audioread(str(p))
                        decoder = "audioread"
                    except Exception as e_ar:
                        y = decode_with_librosa(str(p))
                        decoder = f"librosa_fallback({type(e_ar).__name__})"
                else:
                    y = decode_with_librosa(str(p))
                    decoder = "librosa"

                x = extract_mfcc(y)
                feats.append(x)
                ids.append(tid)
                kept += 1

                writer.writerow({"track_id": tid, "audio_path": str(p), "status": "ok", "decoder": decoder, "reason": ""})

            except Exception as e:
                bad += 1
                writer.writerow({"track_id": tid, "audio_path": str(p), "status": "skip", "decoder": "", "reason": f"{type(e).__name__}: {e}"})

            # periodic live feedback
            if (kept + bad + missing) % 200 == 0:
                tqdm.write(f"[progress] kept={kept}  skipped={bad}  missing={missing}")

    X = np.vstack(feats) if feats else np.zeros((0, 2 * N_MFCC), dtype=np.float32)
    np.save(OUT_X, X)
    np.save(OUT_IDS, np.array(ids, dtype=np.int32))

    print("\n====== EXTRACTION COMPLETE (FILTERED) ======")
    print(f"Manifest used    : {MANIFEST}")
    print(f"Total requested  : {len(df)}")
    print(f"Kept features    : {kept}")
    print(f"Decode errors    : {bad}")
    print(f"Missing files    : {missing}")
    print(f"Feature shape    : {X.shape}")
    print(f"Saved features   : {OUT_X}")
    print(f"Saved track ids  : {OUT_IDS}")
    print(f"Log file         : {OUT_LOG}")
    print("===========================================\n")


if __name__ == "__main__":
    main()
