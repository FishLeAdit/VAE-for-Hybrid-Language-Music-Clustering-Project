#!/usr/bin/env python3
"""
Build an English-only manifest by reading lyrics from txt files.

- Lyrics txt folder: data/whisper_kept/
  Example filename:
    "1-800-Band - Diver Blue 142128.txt"
  Track ID is the trailing integer before ".txt"

- Joins those lyrics into the main manifest via track_id
- Filters for English-ish lyrics
- Outputs: data/manifest_english_only.csv
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

# -----------------------------
# Hardcoded paths
# -----------------------------
IN_MANIFEST = Path("data/fma_manifest_5k_5genres_lyrics_whisper_dropped_removed.csv")
LYRICS_DIR = Path("data/whisper_kept")
OUT_MANIFEST = Path("data/manifest_english_only.csv")
OUT_AUDIT = Path("data/english_subset_audit.csv")

# -----------------------------
# English filter settings
# -----------------------------
MIN_ENGLISH_WORDS = 20
MAX_NON_ASCII_RATIO = 0.30

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
TID_IN_NAME_RE = re.compile(r"(\d+)\.txt$", re.IGNORECASE)


def english_word_count(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


def non_ascii_ratio(text: str) -> float:
    if not text:
        return 1.0
    n = len(text)
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return non_ascii / max(n, 1)


def extract_track_id_from_filename(p: Path) -> int | None:
    """
    Pull trailing integer from filename like:
      "... 142128.txt" -> 142128
    """
    m = TID_IN_NAME_RE.search(p.name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def load_lyrics_map(folder: Path) -> Tuple[Dict[int, str], Dict[int, str]]:
    """
    Returns:
      lyrics_by_id: track_id -> lyrics text
      path_by_id: track_id -> filename (for audit/debug)
    If duplicates exist, keeps the longest lyrics text.
    """
    lyrics_by_id: Dict[int, str] = {}
    path_by_id: Dict[int, str] = {}

    txts = list(folder.glob("*.txt"))
    if not txts:
        raise FileNotFoundError(f"No .txt files found in {folder}")

    for p in txts:
        tid = extract_track_id_from_filename(p)
        if tid is None:
            continue

        try:
            text = p.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            continue

        if not text:
            continue

        # handle duplicates by taking the longer text
        if tid not in lyrics_by_id or len(text) > len(lyrics_by_id[tid]):
            lyrics_by_id[tid] = text
            path_by_id[tid] = p.name

    return lyrics_by_id, path_by_id


def main():
    if not IN_MANIFEST.exists():
        raise SystemExit(f"Manifest not found: {IN_MANIFEST}")
    if not LYRICS_DIR.exists():
        raise SystemExit(f"Lyrics folder not found: {LYRICS_DIR}")

    df = pd.read_csv(IN_MANIFEST)
    if "track_id" not in df.columns:
        raise SystemExit("Manifest must contain a 'track_id' column.")

    df["track_id"] = df["track_id"].astype(int)

    print(f"Loading lyrics from: {LYRICS_DIR}")
    lyrics_by_id, path_by_id = load_lyrics_map(LYRICS_DIR)
    print(f"Found lyrics files mapped to track_ids: {len(lyrics_by_id)}")

    # attach lyrics into df (only for those that exist in the folder)
    df["lyrics"] = df["track_id"].map(lyrics_by_id).fillna("")
    df["lyrics_file"] = df["track_id"].map(path_by_id).fillna("")

    # compute english signals
    df["eng_words"] = df["lyrics"].map(english_word_count)
    df["non_ascii_ratio"] = df["lyrics"].map(non_ascii_ratio)

    # keep only tracks that have lyrics file AND pass english filter
    has_lyrics = df["lyrics"].str.len() > 0
    english_like = (df["eng_words"] >= MIN_ENGLISH_WORDS) & (df["non_ascii_ratio"] <= MAX_NON_ASCII_RATIO)

    df["keep_english"] = has_lyrics & english_like

    kept = df[df["keep_english"]].copy()
    dropped = df[~df["keep_english"]].copy()

    kept.to_csv(OUT_MANIFEST, index=False)
    dropped.to_csv(OUT_AUDIT, index=False)

    print("\n====== ENGLISH SUBSET BUILDER ======")
    print(f"Manifest rows                  : {len(df)}")
    print(f"Tracks with lyrics txt found   : {int(has_lyrics.sum())}")
    print(f"Kept (English-only)            : {len(kept)}")
    print(f"Dropped                        : {len(dropped)}")
    print("-----------------------------------")
    print(f"Saved English manifest         : {OUT_MANIFEST}")
    print(f"Saved audit (dropped+reasons)  : {OUT_AUDIT}")
    print("===================================\n")

    # show a few examples
    cols = [c for c in ["track_id", "genre", "lyrics_source", "lyrics_file", "eng_words", "non_ascii_ratio"] if c in kept.columns]
    print("Sample kept rows:")
    print(kept[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
