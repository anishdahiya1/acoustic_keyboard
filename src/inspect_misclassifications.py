# src/inspect_misclassifications.py
import os
import json
import numpy as np
import torch
import librosa
import librosa.display
import matplotlib.pyplot as plt

# config (must match train)
# --- CONFIG (paste this, overwriting previous top config block) ---
SAMPLE_RATE = 48000
N_MFCC = 40
N_FFT = 1024
HOP_LENGTH = 128    # must match training
MAX_FRAMES = 120    # must match training (the checkpoint used 120)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# paths (ensure these match what's on disk)
NPZ_PATH = "artifacts/zoom_dataset_npz.npz"
CKPT_PATH = "artifacts/cnn_keystroke_resopt.pt"
OUT_DIR = "artifacts/misclassified_plots"
os.makedirs(OUT_DIR, exist_ok=True)

# label mapping (same as before)
keys_s = '1234567890qwertyuiopasdfghjklzxcvbnm'
mapping = {i: ch for i, ch in enumerate(keys_s)}

# helper: compute mfcc + delta + delta2 (must match how model was trained)
import numpy as np
def compute_mfcc_seq(signal, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH, max_frames=MAX_FRAMES):
    mf = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    d1 = librosa.feature.delta(mf, order=1)
    d2 = librosa.feature.delta(mf, order=2)
    def pad_trunc(arr):
        if arr.shape[1] < max_frames:
            pad_width = max_frames - arr.shape[1]
            return np.pad(arr, ((0,0),(0,pad_width)), mode='constant', constant_values=0.0)
        else:
            return arr[:, :max_frames]
    mf = pad_trunc(mf)
    d1 = pad_trunc(d1)
    d2 = pad_trunc(d2)
    stacked = np.vstack([mf, d1, d2])   # shape (3*n_mfcc, max_frames)
    stacked = (stacked - stacked.mean(axis=1, keepdims=True)) / (stacked.std(axis=1, keepdims=True) + 1e-9)
    return stacked.astype(np.float32)
# -------------------------------------------------------------------


# load data
data = np.load(NPZ_PATH)
X_test = data["X_test"]    # shape (N_test, L)
y_test = data["y_test"]

# load checkpoint & build model (same structure as train)
from train_cnn import SimpleCNN  # re-use model class from train script
n_classes = int(y_test.max()) + 1
# Build initial model with the local MAX_FRAMES, but be prepared to adjust if the
# checkpoint was trained with a different frame-length (this causes fc1 size mismatches).
model = SimpleCNN(n_mfcc=N_MFCC, n_frames=MAX_FRAMES, n_classes=n_classes).to(DEVICE)
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
st = ckpt.get("model_state", ckpt)
try:
    model.load_state_dict(st)
    trained_n_mfcc = N_MFCC
    trained_n_frames = MAX_FRAMES
except RuntimeError as e:
    print("Warning: failed to load checkpoint into model (size mismatch). Trying to infer trained shape:", e)
    # infer fc1 in_features -> trained n_frames (two poolings reduce by 4)
    fc1_keys = [k for k in st.keys() if k.endswith("fc1.weight")]
    conv1_keys = [k for k in st.keys() if k.endswith("conv1.weight")]
    if len(fc1_keys) == 0 or len(conv1_keys) == 0:
        raise
    fc1_k = fc1_keys[0]
    conv1_k = conv1_keys[0]
    in_features = st[fc1_k].shape[1]
    trained_out_t = in_features // 256
    trained_n_frames = int(trained_out_t * 4)
    # conv1.weight shape: (out_channels, in_channels, kernel_size)
    conv_in_channels = int(st[conv1_k].shape[1])
    # train_cnn stacks MFCC + delta + delta2 => channels = 3 * n_mfcc
    trained_n_mfcc = int(conv_in_channels // 3)
    print(f"Inferred checkpoint shapes: n_mfcc={trained_n_mfcc}, n_frames={trained_n_frames} (fc1 in_features={in_features})")
    # rebuild model with inferred shapes and load checkpoint
    model = SimpleCNN(n_mfcc=trained_n_mfcc, n_frames=trained_n_frames, n_classes=n_classes).to(DEVICE)
    model.load_state_dict(st)

model.eval()

mis = []
with torch.no_grad():
    for i in range(len(X_test)):
        x = X_test[i]
        y_true = int(y_test[i])
        # compute MFCCs using the same n_mfcc and frame length as the trained model
        mf = compute_mfcc_seq(x, n_mfcc=trained_n_mfcc, max_frames=trained_n_frames)  # (n_mfcc, max_frames)
        inp = torch.from_numpy(mf).unsqueeze(0).to(DEVICE)  # (1, n_mfcc, max_frames)
        out = model(inp)
        pred = int(out.argmax(dim=1).cpu().item())
        if pred != y_true:
            mis.append((i, y_true, pred))

print(f"Total test samples: {len(X_test)}, misclassified: {len(mis)}")
# show first up to 8 misclassifications
show_n = min(8, len(mis))
print(f"Showing first {show_n} misclassified examples (index, true_label->char -> pred_label->char):")
for k in range(show_n):
    idx, t, p = mis[k]
    print(f"  idx={idx}  {t} -> '{mapping[t]}'   predicted: {p} -> '{mapping[p]}'")

    # prepare plots
    wav = X_test[idx]
    duration = len(wav) / SAMPLE_RATE
    fig, axes = plt.subplots(2, 1, figsize=(10, 5))
    librosa.display.waveshow(wav, sr=SAMPLE_RATE, ax=axes[0])
    axes[0].set_title(f"idx={idx} true={mapping[t]} pred={mapping[p]}  duration={duration:.3f}s")
    D = np.abs(librosa.stft(wav, n_fft=1024, hop_length=128))
    DB = librosa.amplitude_to_db(D, ref=np.max)
    img = librosa.display.specshow(DB, sr=SAMPLE_RATE, hop_length=128, x_axis='time', y_axis='log', ax=axes[1])
    fig.colorbar(img, ax=axes[1], format='%+2.0f dB')
    plt.tight_layout()
    out_file = os.path.join(OUT_DIR, f"mis_idx{idx}_t{t}_p{p}.png")
    fig.savefig(out_file)
    plt.close(fig)
    print("   Saved plot:", out_file)

if len(mis) == 0:
    print("No misclassifications found — excellent!")
else:
    print("\nCheck the PNGs saved in", OUT_DIR, "and paste the printed lines above (first few).")
