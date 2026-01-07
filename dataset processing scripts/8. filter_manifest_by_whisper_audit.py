#!/usr/bin/env python3
import os
import re
from pathlib import Path

import pandas as pd

# =========================
# HARDCODED INPUTS
# =========================
MANIFEST_IN = Path("data/fma_manifest_5k_5genres_lyrics_whisper.csv")
WHISPER_AUDIT = Path("data/whisper_audit.csv")

# =========================
# HARDCODED OUTPUTS
# =========================
OUT_KEPT_ONLY = Path("data/fma_manifest_5k_5genres_lyrics_whisper_kept_whisper.csv")
OUT_DROPPED_REMOVED = Path("data/fma_manifest_5k_5genres_lyrics_whisper_dropped_removed.csv")
OUT_KEPT_AND_GENIUS = Path("data/fma_manifest_5k_5genres_lyrics_whisper_kept_whisper_and_genius.csv")


def extract_track_id_from_audit_path(fp: str) -> int | None:
    """
    Whisper audit 'file' looks like a filename with numbers embedded.
    We want the true track_id at the end (common patterns):
      ... _11773_.txt
      ... 11773.txt
      ... 011773.txt
    Strategy:
      1) prefer _<digits>_.txt
      2) else last whitespace + <digits>.txt
      3) else last <digits>.txt
      4) else last 4-6 digit run in name
    """
    base = os.path.basename(str(fp))

    m = re.search(r"_(\d{1,6})_\.txt$", base)
    if m:
        return int(m.group(1))

    m = re.search(r"\s(\d{1,6})\.txt$", base)
    if m:
        return int(m.group(1))

    m = re.search(r"(\d{1,6})\.txt$", base)
    if m:
        return int(m.group(1))

    ms = re.findall(r"(\d{4,6})", base)
    if ms:
        return int(ms[-1])

    return None


def main():
    if not MANIFEST_IN.exists():
        raise SystemExit(f"Missing manifest: {MANIFEST_IN}")
    if not WHISPER_AUDIT.exists():
        raise SystemExit(f"Missing whisper audit: {WHISPER_AUDIT}")

    df = pd.read_csv(MANIFEST_IN)
    audit = pd.read_csv(WHISPER_AUDIT)

    if "track_id" not in df.columns:
        raise SystemExit(f"Manifest must contain 'track_id' column. Found: {df.columns.tolist()}")

    # Parse track_id from audit filenames
    audit["track_id"] = audit["file"].apply(extract_track_id_from_audit_path)

    # Build sets
    kept_ids = set(audit.loc[audit["status"].astype(str).str.lower() == "kept", "track_id"].dropna().astype(int))
    dropped_ids = set(audit.loc[audit["status"].astype(str).str.lower() == "dropped", "track_id"].dropna().astype(int))

    # A) Keep only tracks that have a kept whisper transcript
    df_kept_only = df[df["track_id"].astype(int).isin(kept_ids)].copy()

    # B) Remove tracks that have a dropped whisper transcript (keep unknowns)
    df_dropped_removed = df[~df["track_id"].astype(int).isin(dropped_ids)].copy()

    # C) Kept whisper AND genius lyrics_source
    if "lyrics_source" in df.columns:
        df_kept_and_genius = df_kept_only[df_kept_only["lyrics_source"].astype(str).str.lower() == "genius"].copy()
    else:
        df_kept_and_genius = df_kept_only.iloc[0:0].copy()

    # Write outputs
    OUT_KEPT_ONLY.parent.mkdir(parents=True, exist_ok=True)
    df_kept_only.to_csv(OUT_KEPT_ONLY, index=False)
    df_dropped_removed.to_csv(OUT_DROPPED_REMOVED, index=False)
    df_kept_and_genius.to_csv(OUT_KEPT_AND_GENIUS, index=False)

    # Report
    print("\n====== MANIFEST FILTER REPORT ======")
    print(f"Input manifest rows                : {len(df)}")
    print(f"Whisper audit rows                 : {len(audit)}")
    print("-----------------------------------")
    print(f"Unique kept whisper track_ids      : {len(kept_ids)}")
    print(f"Unique dropped whisper track_ids   : {len(dropped_ids)}")
    print("-----------------------------------")
    print(f"Output A (kept whisper only) rows  : {len(df_kept_only)}")
    print(f"Saved -> {OUT_KEPT_ONLY}")
    print("-----------------------------------")
    print(f"Output B (dropped removed) rows    : {len(df_dropped_removed)}")
    print(f"Saved -> {OUT_DROPPED_REMOVED}")
    print("-----------------------------------")
    print(f"Output C (kept whisper + genius)   : {len(df_kept_and_genius)}")
    print(f"Saved -> {OUT_KEPT_AND_GENIUS}")
    if len(df_kept_and_genius) == 0:
        print("NOTE: This being 0 usually means your genius-lyrics tracks do not overlap with whisper-kept tracks.")
    print("===================================\n")


if __name__ == "__main__":
    main()
