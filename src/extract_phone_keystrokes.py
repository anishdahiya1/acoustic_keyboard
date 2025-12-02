# src/inspect_phone.py
import torch
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os

tensors_path = "artifacts/phone_keystrokes_tensors.pt"
meta_path    = "artifacts/phone_keystrokes_meta.csv"

if not os.path.exists(tensors_path) or not os.path.exists(meta_path):
    print("Required files not found. Check artifacts/phone_keystrokes_*.")
    raise SystemExit

wave_tensors = torch.load(tensors_path)
meta = pd.read_csv(meta_path)

print("Num tensors:", len(wave_tensors))
print("Num meta rows:", len(meta))
print("Unique labels (ints):", sorted(meta['Key'].unique()))
print("Label counts:")
print(meta['Key'].value_counts().sort_index())

# mapping used earlier
keys_s = '1234567890qwertyuiopasdfghjklzxcvbnm'
mapping = {i: ch for i, ch in enumerate(keys_s)}
print("Mapping (0->char):", mapping)

# show one example (idx 0). Change idx variable to view others.
idx = 0
t = wave_tensors[idx]
print("raw tensor shape:", tuple(t.shape))
# squeeze if needed
if t.ndim == 2 and t.shape[0] == 1:
    arr = t.squeeze(0).numpy()
else:
    arr = t.numpy()
print(f"Example idx={idx} label={meta['Key'].iloc[idx]} -> char='{mapping[int(meta['Key'].iloc[idx])]}' len_samples={arr.shape[0]}")

# guess sample rate commonly used (or check original file if needed)
sample_rate = 48000
# waveform plot
plt.figure(figsize=(10,2))
librosa.display.waveshow(arr, sr=sample_rate)
plt.title(f"Phone example idx={idx} label={mapping[int(meta['Key'].iloc[idx])]}")
plt.tight_layout()
out = f"artifacts/phone_example_idx{idx}.png"
plt.savefig(out)
plt.close()
print("Saved waveform plot to", out)

# spectrogram
D = np.abs(librosa.stft(arr, n_fft=1024, hop_length=128))
DB = librosa.amplitude_to_db(D, ref=np.max)
plt.figure(figsize=(10,3))
librosa.display.specshow(DB, sr=sample_rate, hop_length=128, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title("Spectrogram (phone) idx="+str(idx))
plt.tight_layout()
out2 = f"artifacts/phone_spec_idx{idx}.png"
plt.savefig(out2)
plt.close()
print("Saved spectrogram to", out2)
