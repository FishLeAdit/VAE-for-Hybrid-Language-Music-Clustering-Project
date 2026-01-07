import os
import whisper
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time
import torch



# --- CONFIGURATION ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")    
model = whisper.load_model("small") 
MANIFEST_IN = Path("data/fma_manifest_5k_5genres_lyrics.csv")
MANIFEST_OUT = Path("data/fma_manifest_5k_5genres_lyrics_whisper.csv")
AUDIO_DIR = Path("data/fma_small/fma_small") 
TRANSCRIPTIONS_DIR = Path("data/whisper_transcriptions")

TRANSCRIPTIONS_DIR.mkdir(parents=True, exist_ok=True)

# --- STEP 1: LOAD DATA ---
print("Loading manifest...")
df = pd.read_csv(MANIFEST_IN)

# Filter out songs that already have lyrics from Genius
df["lyrics_source"] = df["lyrics_source"].fillna("")
df_filtered = df[df["lyrics_source"].str.lower() != "genius"]

# --- STEP 2: SCAN DISK FOR ACTUAL MP3 FILES ---
# This fixes the WinError 2 by finding exactly where files are located
print(f"Scanning {AUDIO_DIR} for mp3 files...")
mp3_map = {} # Dictionary to map { track_id (int) : full_file_path (Path) }

for root, dirs, files in os.walk(AUDIO_DIR):
    for file in files:
        if file.endswith(".mp3"):
            try:
                # Filename is usually "000123.mp3", extract "123"
                track_id_from_file = int(file.split('.')[0])
                full_path = Path(root) / file
                mp3_map[track_id_from_file] = full_path
            except ValueError:
                continue

print(f"Found {len(mp3_map)} audio files on disk.")

# --- STEP 3: PROCESS AUDIO ---
processed_songs = []

# Function to transcribe
def transcribe_audio(audio_path: Path) -> str:
    # Convert Path object to string for Whisper
    audio_path_str = str(audio_path.resolve())
    
    # Load and transcribe
    audio = whisper.load_audio(audio_path_str)
    audio = whisper.pad_or_trim(audio)
    result = model.transcribe(audio)
    return result["text"]

# Iterate through the filtered songs
print("Starting transcription...")
for idx, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="Transcribing"):
    artist = row["artist"]
    title = row["title"]
    track_id = int(row["track_id"])

    # Construct the transcription filename using artist, title, and track_id
    transcription_filename = f"{artist} - {title} _{track_id}_.txt"
    
    # Construct the path to the mp3 file using subdirectories
    track_id_str = f"{track_id:06d}"
    audio_file = mp3_map.get(track_id, None)

    # Print the file path for debugging
    print(f"Checking file: {audio_file}")

    # CHECK IF WE ACTUALLY HAVE THE FILE
    if audio_file:
        # Check if already processed in this run
        if track_id not in processed_songs:
            try:
                # Transcribe audio using Whisper
                lyrics_text = transcribe_audio(audio_file)

                # Save transcription with the custom filename
                transcription_file = TRANSCRIPTIONS_DIR / transcription_filename
                with open(transcription_file, "w", encoding="utf-8") as f:
                    f.write(lyrics_text)

                # Update DataFrame with the new transcription file path and source
                df.at[idx, "lyrics_path"] = str(transcription_file)
                df.at[idx, "lyrics_source"] = "whisper"
                processed_songs.append(track_id)

                # Print confirmation
                print(f"Transcription for {artist} - {title} saved!")

            except Exception as e:
                print(f"Error processing {artist} - {title}: {str(e)}")

        else:
            print(f"Skipping {artist} - {title} (Already processed)")

    else:
        print(f"Audio file not found for {artist} - {title} at {audio_file}")

    # Add delay to avoid hitting rate limits (optional)
    time.sleep(1)

# --- STEP 4: SAVE ---
df.to_csv(MANIFEST_OUT, index=False)
print(f"\n✅ Done! Processed {len(processed_songs)} songs.")
print(f"Updated manifest saved to: {MANIFEST_OUT}")
