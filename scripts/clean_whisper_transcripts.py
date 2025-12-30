#!/usr/bin/env python3
"""
Whisper .txt audit + optional split into KEPT / DROPPED (hardcoded).

What it does:
1) Scans ALL .txt files in:       data/whisper_transcriptions/
2) Cleans common Whisper junk: timestamps, [music], (applause), etc.
3) Scores transcript quality + language signal (more robust than "non-ascii only")
4) Prints a report + breakdown
5) Prompts: Proceed? [Y/N]
6) If Y:
     - Writes CLEAN kept files to:   data/whisper_kept/
     - Copies dropped originals to:  data/whisper_dropped/
     - Writes a CSV audit:           data/whisper_audit.csv

Original input files are NOT modified.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from collections import Counter
from pathlib import Path
import shutil
import math


# =============================
# HARD-CODED PATHS
# =============================
INPUT_DIR = Path("data/whisper_transcriptions")
KEPT_DIR = Path("data/whisper_kept")
DROPPED_DIR = Path("data/whisper_dropped")
AUDIT_CSV = Path("data/whisper_audit.csv")


# =============================
# HARD-CODED RULES / THRESHOLDS
# =============================
MIN_WORDS = 6                   # more realistic for "lyrics-like" than 4
MIN_UNIQUE_TOKEN_RATIO = 0.35   # diversity (unique tokens / total tokens)
MIN_ALPHA_RATIO = 0.60          # letters / total chars
MAX_SYMBOL_RATIO = 0.35         # non-alnum-non-space / total chars
MAX_DIGIT_RATIO = 0.20          # digits / total chars (timestamps, IDs, etc.)
MAX_REPEAT_TOKEN_RATIO = 0.35   # most common token frequency / total tokens
MIN_AVG_TOKEN_LEN = 3.0         # prevents lots of 1-2 char garbage tokens

LANG_MODE = "strict"            # "off" | "fast" | "strict"
# fast: mostly non-ascii check only
# strict: langdetect if installed, else falls back to "fast"


# =============================
# Cleaning logic
# =============================
_TS_PATTERNS = [
    re.compile(r"\[\s*\d{1,2}:\d{2}(?::\d{2})?\s*\]"),                 # [00:12] [00:00:12]
    re.compile(r"\b\d{1,2}:\d{2}:\d{2}\s*-->\s*\d{1,2}:\d{2}:\d{2}\b"), # SRT
]
_BRACKET_TAGS = re.compile(r"(\[[^\]]*\]|\([^\)]*\))", re.IGNORECASE)
_SPEAKER_TAG = re.compile(r"^\s*(speaker\s*\d+|spk\s*\d+|host|narrator)\s*:\s*", re.IGNORECASE)
_JUNK_LINE = re.compile(
    r"^\s*(music|applause|laughter|laughs|silence|noise|background noise|"
    r"inaudible|unintelligible)\s*$",
    re.IGNORECASE,
)
_MULTI_SPACE = re.compile(r"\s+")

# Tokens/words
_TOKEN_RE = re.compile(r"[A-Za-z]{2,}(?:'[A-Za-z]+)?")  # at least 2 letters
_ALPHA_RE = re.compile(r"[A-Za-z]")
_SYMBOL_RE = re.compile(r"[^A-Za-z0-9\s]")
_DIGIT_RE = re.compile(r"\d")


# Common boilerplate / non-lyric phrases (expand as needed)
BOILERPLATE_SUBSTRINGS = [
    "for more information visit",
    "visit www",
    "subscribe",
    "thanks for watching",
    "like and subscribe",
    "follow me on",
    "follow us on",
    "all rights reserved",
    "copyright",
    "welcome to the podcast",
    "this is an instrumental",
    "instrumental",
]


def normalize_text(t: str) -> str:
    if t is None:
        return ""
    t = unicodedata.normalize("NFKC", str(t))

    # Remove timestamps
    for pat in _TS_PATTERNS:
        t = pat.sub(" ", t)

    # Remove speaker tags at line start
    t = "\n".join(_SPEAKER_TAG.sub("", line) for line in t.splitlines())

    # Remove bracketed tags like [Music], (Applause)
    t = _BRACKET_TAGS.sub(" ", t)

    # Drop junk-only lines
    kept_lines = []
    for line in t.splitlines():
        s = line.strip()
        if not s:
            continue
        if _JUNK_LINE.match(s):
            continue
        kept_lines.append(s)

    t = " ".join(kept_lines)
    t = _MULTI_SPACE.sub(" ", t).strip()
    return t


# =============================
# Metrics
# =============================
def tokens(t: str) -> list[str]:
    return _TOKEN_RE.findall(t.lower()) if t else []


def alpha_ratio(t: str) -> float:
    if not t:
        return 0.0
    return len(_ALPHA_RE.findall(t)) / len(t)


def symbol_ratio(t: str) -> float:
    if not t:
        return 1.0
    return len(_SYMBOL_RE.findall(t)) / len(t)


def digit_ratio(t: str) -> float:
    if not t:
        return 0.0
    return len(_DIGIT_RE.findall(t)) / len(t)


def unique_token_ratio(tok: list[str]) -> float:
    if not tok:
        return 0.0
    return len(set(tok)) / len(tok)


def repeat_token_ratio(tok: list[str]) -> float:
    if not tok:
        return 1.0
    c = Counter(tok)
    return c.most_common(1)[0][1] / len(tok)


def avg_token_len(tok: list[str]) -> float:
    if not tok:
        return 0.0
    return sum(len(x) for x in tok) / len(tok)


def contains_boilerplate(t: str) -> bool:
    tl = (t or "").lower()
    return any(s in tl for s in BOILERPLATE_SUBSTRINGS)


def is_mostly_non_ascii(t: str, threshold: float = 0.45) -> bool:
    if not t:
        return True
    non_ascii = sum(1 for ch in t if ord(ch) > 127)
    return (non_ascii / len(t)) > threshold


def detect_lang_strict(t: str) -> str:
    """
    Returns a language code via langdetect if installed, else 'unknown'.
    """
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0
        if len(t) < 30:
            return "unknown"
        return detect(t)
    except Exception:
        return "unknown"


def is_english_strict(t: str) -> bool:
    lang = detect_lang_strict(t)
    if lang == "unknown":
        # fallback to fast heuristic (non-ascii)
        return not is_mostly_non_ascii(t)
    return lang == "en"


# =============================
# Drop decision
# =============================
def classify_drop_reason(cleaned: str) -> tuple[str | None, dict]:
    """
    Returns (reason_or_None, metrics_dict)
    """
    tok = tokens(cleaned)
    wc = len(tok)

    m = {
        "word_count": wc,
        "alpha_ratio": round(alpha_ratio(cleaned), 4),
        "symbol_ratio": round(symbol_ratio(cleaned), 4),
        "digit_ratio": round(digit_ratio(cleaned), 4),
        "unique_token_ratio": round(unique_token_ratio(tok), 4),
        "repeat_token_ratio": round(repeat_token_ratio(tok), 4),
        "avg_token_len": round(avg_token_len(tok), 4),
        "strict_lang": detect_lang_strict(cleaned),
    }

    if not cleaned:
        return "empty_after_clean", m

    if contains_boilerplate(cleaned):
        return "boilerplate_or_non_lyrics", m

    if wc < MIN_WORDS:
        return f"too_short(<{MIN_WORDS}_words)", m

    if m["alpha_ratio"] < MIN_ALPHA_RATIO:
        return "low_alpha_ratio_gibberish", m

    if m["symbol_ratio"] > MAX_SYMBOL_RATIO:
        return "too_many_symbols", m

    if m["digit_ratio"] > MAX_DIGIT_RATIO:
        return "too_many_digits", m

    if m["unique_token_ratio"] < MIN_UNIQUE_TOKEN_RATIO:
        return "low_token_diversity", m

    if m["repeat_token_ratio"] > MAX_REPEAT_TOKEN_RATIO:
        return "too_repetitive", m

    if m["avg_token_len"] < MIN_AVG_TOKEN_LEN:
        return "tokens_too_short_gibberish", m

    # Language filtering
    if LANG_MODE == "fast":
        if is_mostly_non_ascii(cleaned):
            return "non_english_or_low_signal_fast", m
    elif LANG_MODE == "strict":
        if not is_english_strict(cleaned):
            return "non_english_strict", m
    elif LANG_MODE == "off":
        pass
    else:
        return "bad_lang_mode_config", m

    return None, m


# =============================
# Main
# =============================
def main():
    if not INPUT_DIR.exists():
        raise SystemExit(f"Input directory not found: {INPUT_DIR}")

    txt_files = sorted(INPUT_DIR.rglob("*.txt"))
    total = len(txt_files)
    if total == 0:
        raise SystemExit(f"No .txt files found under {INPUT_DIR}")

    decisions = {}  # file -> (cleaned, reason, metrics)
    reasons = Counter()
    errors = 0

    for f in txt_files:
        try:
            raw = f.read_text(encoding="utf-8", errors="ignore")
            cleaned = normalize_text(raw)
            reason, metrics = classify_drop_reason(cleaned)
            decisions[f] = (cleaned, reason, metrics)
            if reason is not None:
                reasons[reason] += 1
        except Exception as e:
            errors += 1
            reasons["read_error"] += 1
            decisions[f] = ("", "read_error", {"error": f"{type(e).__name__}: {e}"})

    dropped = sum(reasons.values())
    kept = total - dropped

    # -------- REPORT --------
    print("\n====== WHISPER TXT AUDIT (ROBUST) ======")
    print(f"Directory scanned : {INPUT_DIR}")
    print(f"Total .txt files  : {total}")
    print(f"Would KEEP        : {kept}")
    print(f"Would DROP        : {dropped}")
    print("--------------------------------------")
    if reasons:
        print("Drop breakdown:")
        for r, c in reasons.most_common():
            pct = (c / total) * 100
            print(f"  {r:<30} {c:>5}  ({pct:5.2f}%)")
    print("======================================")

    # Prompt
    choice = input("\nProceed to write KEPT/DROPPED folders + audit CSV? [Y/N]: ").strip().lower()
    if choice != "y":
        print("Aborted. No output folders were written.")
        return

    # Prepare dirs
    KEPT_DIR.mkdir(parents=True, exist_ok=True)
    DROPPED_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Write audit CSV + files
    with AUDIT_CSV.open("w", newline="", encoding="utf-8") as fp:
        fieldnames = [
            "file",
            "status",
            "reason",
            "word_count",
            "alpha_ratio",
            "symbol_ratio",
            "digit_ratio",
            "unique_token_ratio",
            "repeat_token_ratio",
            "avg_token_len",
            "strict_lang",
            "preview",
        ]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()

        for f, (cleaned, reason, m) in decisions.items():
            rel = f.relative_to(INPUT_DIR)

            if reason is None:
                out_path = KEPT_DIR / rel
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(cleaned, encoding="utf-8")
                status = "kept"
                r = ""
            else:
                out_path = DROPPED_DIR / rel
                out_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, out_path)  # keep dropped original for inspection
                status = "dropped"
                r = reason

            writer.writerow({
                "file": str(f),
                "status": status,
                "reason": r,
                "word_count": m.get("word_count", ""),
                "alpha_ratio": m.get("alpha_ratio", ""),
                "symbol_ratio": m.get("symbol_ratio", ""),
                "digit_ratio": m.get("digit_ratio", ""),
                "unique_token_ratio": m.get("unique_token_ratio", ""),
                "repeat_token_ratio": m.get("repeat_token_ratio", ""),
                "avg_token_len": m.get("avg_token_len", ""),
                "strict_lang": m.get("strict_lang", m.get("error", "")),
                "preview": cleaned[:250] if cleaned else "",
            })

    print("\nCompleted.")
    print(f"Kept folder   : {KEPT_DIR}")
    print(f"Dropped folder: {DROPPED_DIR}")
    print(f"Audit CSV     : {AUDIT_CSV}\n")


if __name__ == "__main__":
    main()
