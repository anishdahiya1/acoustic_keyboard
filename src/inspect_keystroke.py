# single-step: load, verify, visualize one keystroke
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

# paths (adjust if your files are elsewhere)
tensors_path = "artifacts/zoom_keystrokes_tensors.pt"
meta_path    = "artifacts/zoom_keystrokes_meta.csv"

# 1) load
wave_tensors = torch.load(tensors_path)        # list of 1-D tensors (or tensors with shape [1, L])
meta = pd.read_csv(meta_path)

# 2) basic checks / print
print("Num tensors:", len(wave_tensors))
print("Num meta rows:", len(meta))
print("Unique labels:", sorted(meta['Key'].unique()))
print("Label counts:")
print(meta['Key'].value_counts().sort_index())

# 3) pick an index to inspect (0 .. len-1). Change idx to view different examples.
idx = 0
wav_t = wave_tensors[idx]
# if tensor shape is [1, L], squeeze to 1D
if wav_t.ndim == 2 and wav_t.shape[0] == 1:
    wav = wav_t.squeeze(0).numpy()
else:
    wav = wav_t.numpy()

# if you know sample_rate used in extraction; typical is 48000 or librosa.load default
# we loaded original files with librosa.load(sr=None) so preserve original sample rate
# if you don't know, try 48000 first
sample_rate = 48000

print(f"\nShowing example idx={idx}, label={meta['Key'].iloc[idx]}, waveform length={len(wav)} samples")

# 4) plot waveform
plt.figure(figsize=(10, 2.2))
librosa.display.waveshow(wav, sr=sample_rate)
plt.title(f"Waveform — idx={idx}, label={meta['Key'].iloc[idx]}")
plt.tight_layout()
plt.show()

# 5) plot spectrogram (log-magnitude)
D = np.abs(librosa.stft(wav, n_fft=512, hop_length=128))
DB = librosa.amplitude_to_db(D, ref=np.max)
plt.figure(figsize=(10, 3))
librosa.display.specshow(DB, sr=sample_rate, hop_length=128, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title(f"Spectrogram (log) — idx={idx}")
plt.tight_layout()
plt.show()
