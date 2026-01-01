import librosa
import numpy as np

SR = 22050
N_MELS = 64
MAX_LEN = 5.0
HOP_LENGTH = 512
FIXED_FRAMES = int((SR * MAX_LEN) / HOP_LENGTH)

def audio_to_logmel(path):
    y, sr = librosa.load(path, sr=SR, mono=True)
    
    # Trim or pad waveform
    max_samples = int(SR * MAX_LEN)
    if len(y) < max_samples:
        y = np.pad(y, (0, max_samples - len(y)))
    else:
        y = y[:max_samples]

    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH
    )
    # Avoid log(0) -> -inf
    S = np.maximum(S, 1e-10)

    logS = librosa.power_to_db(S, ref=np.max)

    # Replace any inf/nan just in case
    logS = np.nan_to_num(logS, nan=0.0, posinf=0.0, neginf=0.0)


    # Fix time dimension
    if logS.shape[1] < FIXED_FRAMES:
        pad_width = FIXED_FRAMES - logS.shape[1]
        logS = np.pad(logS, ((0, 0), (0, pad_width)))
    else:
        logS = logS[:, :FIXED_FRAMES]

    return logS.astype(np.float32)
