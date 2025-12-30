from urllib.request import urlopen
from bs4 import BeautifulSoup
from time import sleep
import csv
import time
import requests
import pandas as pd
import re
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

# -------------------- CONFIG --------------------
MANIFEST_IN = Path("data/fma_manifest_5k_5genres_lyrics_test.csv")
MANIFEST_OUT = Path("data/fma_manifest_5k_5genres_lyrics_scraped.csv")

LYRICS_DIR = Path("data/lyrics_scraped")

SLEEP_SECONDS = 0.8  # Slow down to reduce blocking
MIN_CHARS = 80       # Ignore very short text as garbage
OVERWRITE_EXISTING = False
MAX_TO_PROCESS = None  # Process all rows

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
# ------------------------------------------------

def safe_filename(s: str) -> str:
    s = re.sub(r"[^\w\-_\. ]", "_", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return (s[:150] if s else "unknown") + ".txt"

def normalize_query(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()

def fetch_page(url: str) -> str | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.ok:
            return r.text
    except Exception:
        pass
    return None

# --- AZLyrics Scraper ---
def scrape_azlyrics(artist: str, title: str) -> str | None:
    artist_clean = re.sub(r"\s+", "", normalize_query(artist))
    title_clean = re.sub(r"\s+", "", normalize_query(title))
    url = f"https://www.azlyrics.com/lyrics/{artist_clean}/{title_clean}.html"
    html = fetch_page(url)
    if not html:
        return None

    soup = BeautifulSoup(html, "html.parser")
    # AZLyrics puts lyrics between comments
    divs = soup.find_all("div")
    for div in divs:
        text = div.get_text("\n").strip()
        if "\n" in text and len(text) > MIN_CHARS:
            return text
    return None

# --- Lyrics.com Scraper ---
def scrape_lyricsdotcom(artist: str, title: str) -> str | None:
    query = requests.utils.quote(f"{artist} {title}")
    search_url = f"https://www.lyrics.com/serp.php?st={query}&qtype=2"
    html = fetch_page(search_url)
    if not html:
        return None
    soup = BeautifulSoup(html, "html.parser")
    link_tag = soup.select_one("td.tal.qx a")
    if not link_tag:
        return None
    lyric_page = fetch_page("https://www.lyrics.com" + link_tag["href"])
    if not lyric_page:
        return None
    lyric_soup = BeautifulSoup(lyric_page, "html.parser")
    lyric_div = lyric_soup.select_one(".lyric-body")
    if lyric_div:
        text = lyric_div.get_text("\n").strip()
        if len(text) > MIN_CHARS:
            return text
    return None

def main() -> None:
    if not MANIFEST_IN.exists():
        raise FileNotFoundError(f"Missing manifest: {MANIFEST_IN}")

    df = pd.read_csv(MANIFEST_IN)

    if "lyrics_path" not in df.columns:
        df["lyrics_path"] = ""
    if "lyrics_source" not in df.columns:
        df["lyrics_source"] = ""

    LYRICS_DIR.mkdir(parents=True, exist_ok=True)
    df_iter = df.head(MAX_TO_PROCESS) if isinstance(MAX_TO_PROCESS, int) else df

    found_count = 0

    for idx, row in tqdm(df_iter.iterrows(), total=len(df_iter), desc="Scraping lyrics"):
        artist = normalize_query(row.get("artist", ""))
        title  = normalize_query(row.get("title", ""))

        # Skip if already from Genius
        src = str(row.get("lyrics_source") or "")
        if "genius" in src.lower():
            print(f"SKIP (genius exists): {artist} - {title}")
            continue

        existing_path = str(row.get("lyrics_path") or "").strip()
        if existing_path and Path(existing_path).exists() and not OVERWRITE_EXISTING:
            print(f"SKIP (file exists): {artist} - {title}")
            continue

        lyrics_text = None
        used_source = ""

        # Try AZLyrics
        print(f"TRY AZLyrics: {artist} - {title}")
        lyrics_text = scrape_azlyrics(artist, title)
        if lyrics_text:
            used_source = "azlyrics"
            print(f"FOUND (azlyrics): {artist} - {title}")

        # Try Lyrics.com
        if not lyrics_text:
            print(f"TRY Lyrics.com: {artist} - {title}")
            lyrics_text = scrape_lyricsdotcom(artist, title)
            if lyrics_text:
                used_source = "lyricsdotcom"
                print(f"FOUND (lyrics.com): {artist} - {title}")

        if lyrics_text:
            fname = safe_filename(f"{artist} - {title} ({row['track_id']})")
            out_path = LYRICS_DIR / fname
            out_path.write_text(lyrics_text, encoding="utf-8")

            df.at[idx, "lyrics_path"] = str(out_path)
            df.at[idx, "lyrics_source"] = used_source
            found_count += 1
        else:
            print(f"NOT FOUND: {artist} - {title}")

        time.sleep(SLEEP_SECONDS)

    MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MANIFEST_OUT, index=False)

    print("\n📌 Done scraping.")
    print(f"Lyrics found (this run): {found_count}/{len(df)}")
    print(f"Updated manifest saved to: {MANIFEST_OUT}")

if __name__ == "__main__":
    main()
