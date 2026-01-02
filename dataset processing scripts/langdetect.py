#!/usr/bin/env python3
"""
Hardcoded report-only script for Whisper .txt quality.

Reads:
  data/whisper_transcriptions/**/*.txt

Writes:
  data/whisper_report.csv

Prints:
  - counts
  - word-count distribution
  - how many would drop under rules
  - FAST vs STRICT language detection breakdown (STRICT only if langdetect installed)

Does NOT modify/move/delete any .txt files.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from pathlib import Path
from collections import Counter

# =============================
# HARD-CODED PATHS
# =============================
INPUT_DIR  = Path("data/whisper_transcriptions")
REPORT_CSV = Path("data/whisper_report.csv")

# =============================
# HARD-CODED RULES
# =============================
MIN_WORDS = 4
SYMBOL_THRESHOLD = 0.35
NON_ASCII_THRESHOLD = 0.45

# Language mode for "would drop" counts:
# - "fast": your current approach (mostly-non-ascii)
# - "strict": use langdetect if installed; otherwise falls back to fast
DROP_LANG_MODE = "strict"   # change to "strict" after installing langdetect


# =============================
# Cleaning logic
# =============================
_TS_PATTERNS = [
    re.compile(r"\[\s*\d{1,2}:\d{2}(?::\d{2})?\s*\]"),
    re.compile(r"\b\d{1,2}:\d{2}:\d{2}\s*-->\s*\d{1,2}:\d{2}:\d{2}\b"),
]
_BRACKET_TAGS = re.compile(r"(\[[^\]]*\]|\([^\)]*\))", re.IGNORECASE)
_JUNK_LINE = re.compile(
    r"^\s*(music|applause|laughter|laughs|silence|noise|background noise|"
    r"inaudible|unintelligible)\s*$",
    re.IGNORECASE
)
_MULTI_SPACE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)

    for pat in _TS_PATTERNS:
        text = pat.sub(" ", text)

    text = _BRACKET_TAGS.sub(" ", text)

    kept_lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if _JUNK_LINE.match(s):
            continue
        kept_lines.append(s)

    text = " ".join(kept_lines)
    text = _MULTI_SPACE.sub(" ", text).strip()
    return text


# =============================
# Metrics / filters
# =============================
def english_word_count(t: str) -> int:
    return len(_WORD_RE.findall(t))


def has_too_many_symbols(t: str) -> bool:
    if not t:
        return True
    symbols = sum(1 for c in t if not c.isalnum() and not c.isspace())
    return (symbols / len(t)) > SYMBOL_THRESHOLD


def is_mostly_non_ascii(t: str) -> bool:
    if not t:
        return True
    non_ascii = sum(1 for c in t if ord(c) > 127)
    return (non_ascii / len(t)) > NON_ASCII_THRESHOLD


def detect_lang_strict(t: str) -> str:
    """
    Returns language code (e.g., 'en') if langdetect available; else returns 'unknown'.
    """
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0
        if len(t) < 20:
            return "unknown"
        return detect(t)
    except Exception:
        return "unknown"


def is_probably_english_fast(t: str) -> bool:
    # FAST heuristic: mostly ASCII + enough "English-ish" words
    if not t:
        return False
    if is_mostly_non_ascii(t):
        return False
    return english_word_count(t) >= MIN_WORDS


def is_probably_english_strict(t: str) -> bool:
    lang = detect_lang_strict(t)
    if lang == "unknown":
        return is_probably_english_fast(t)
    return lang == "en"


def classify_drop_reason(cleaned: str, lang_mode: str) -> str | None:
    if not cleaned:
        return "empty_after_clean"

    wc = english_word_count(cleaned)
    if wc < MIN_WORDS:
        return f"too_short(<{MIN_WORDS}_words)"

    if has_too_many_symbols(cleaned):
        return "too_many_symbols"

    if lang_mode == "fast":
        # Only flags if mostly non-ascii
        if is_mostly_non_ascii(cleaned):
            return "non_english_or_low_signal_fast"
    elif lang_mode == "strict":
        # Uses langdetect if available; otherwise falls back to fast
        if not is_probably_english_strict(cleaned):
            return "non_english_strict"
    else:
        raise ValueError("lang_mode must be 'fast' or 'strict'")

    return None


def percentile(sorted_vals: list[int], p: float) -> int:
    if not sorted_vals:
        return 0
    k = int(round((len(sorted_vals) - 1) * p))
    return sorted_vals[k]


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

    # Counters
    drop_reasons = Counter()
    fast_lang = Counter()    # english-ish or not
    strict_lang = Counter()  # detected lang buckets

    word_counts = []
    rows = []

    for f in txt_files:
        try:
            raw = f.read_text(encoding="utf-8", errors="ignore")
            cleaned = normalize_text(raw)
            wc = english_word_count(cleaned)
            sym_flag = has_too_many_symbols(cleaned)
            non_ascii_flag = is_mostly_non_ascii(cleaned)

            # FAST english-ish label
            fast_is_en = is_probably_english_fast(cleaned)
            fast_lang["fast_en"] += int(fast_is_en)
            fast_lang["fast_not_en"] += int(not fast_is_en)

            # STRICT language label (if detector exists)
            lang = detect_lang_strict(cleaned)
            strict_lang[lang] += 1

            # Would-drop under configured drop language mode
            reason = classify_drop_reason(cleaned, DROP_LANG_MODE)
            if reason is not None:
                drop_reasons[reason] += 1

            word_counts.append(wc)

            rows.append({
                "file": str(f),
                "cleaned_word_count": wc,
                "has_too_many_symbols": int(sym_flag),
                "mostly_non_ascii": int(non_ascii_flag),
                "fast_englishish": int(fast_is_en),
                "strict_langdetect": lang,
                "would_drop_reason": reason or "",
                "preview": cleaned[:250],
            })
        except Exception as e:
            drop_reasons["read_error"] += 1
            rows.append({
                "file": str(f),
                "cleaned_word_count": "",
                "has_too_many_symbols": "",
                "mostly_non_ascii": "",
                "fast_englishish": "",
                "strict_langdetect": "error",
                "would_drop_reason": f"error:{type(e).__name__}",
                "preview": "",
            })

    # Write CSV report
    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_CSV.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Print summary
    word_counts_sorted = sorted([w for w in word_counts if isinstance(w, int)])
    p10 = percentile(word_counts_sorted, 0.10)
    p50 = percentile(word_counts_sorted, 0.50)
    p90 = percentile(word_counts_sorted, 0.90)
    min_wc = word_counts_sorted[0] if word_counts_sorted else 0
    max_wc = word_counts_sorted[-1] if word_counts_sorted else 0

    would_drop = sum(drop_reasons.values())
    would_keep = total - would_drop

    print("\n====== WHISPER TRANSCRIPT REPORT ======")
    print(f"Scanned files             : {total}")
    print(f"Would KEEP (rules={DROP_LANG_MODE})    : {would_keep}")
    print(f"Would DROP (rules={DROP_LANG_MODE})    : {would_drop}")
    print("--------------------------------------")
    print("Word-count distribution (cleaned):")
    print(f"  min={min_wc}  p10={p10}  median={p50}  p90={p90}  max={max_wc}")
    print("--------------------------------------")
    print("FAST English-ish breakdown:")
    print(f"  fast_en      : {fast_lang['fast_en']}")
    print(f"  fast_not_en  : {fast_lang['fast_not_en']}")
    print("--------------------------------------")
    print("Top STRICT langdetect results (if installed):")
    for k, v in strict_lang.most_common(8):
        print(f"  {k:<10} {v}")
    print("--------------------------------------")
    if drop_reasons:
        print("Would-drop reasons:")
        for r, c in drop_reasons.most_common():
            print(f"  {r:<30} {c}")
    print("--------------------------------------")
    print(f"CSV report written to      : {REPORT_CSV}")
    print("======================================\n")


if __name__ == "__main__":
    main()
