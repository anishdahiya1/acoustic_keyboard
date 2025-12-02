# src/debug_classify_uploaded.py
import sys, os
import numpy as np
import torch
import torch.nn.functional as F
import librosa
import librosa.display
import matplotlib.pyplot as plt

# ---------- Config (must match training) ----------
SAMPLE_RATE = 48000   # used only where we resample; isolator preserves original sr
N_MFCC = 40
N_FFT = 1024
HOP_LENGTH = 128      # MUST match your final training setting
MAX_FRAMES = 120      # MUST match your final training setting
MODEL_PATH = "artifacts/cnn_finetune_hard.pt"   # change if needed
OUT_DIR = "artifacts/debug_uploaded"
KEYS_S = '1234567890qwertyuiopasdfghjklzxcvbnm'
mapping = {i: ch for i, ch in enumerate(KEYS_S)}
# isolator params (same as used earlier)
ISO_SIZE = 48
ISO_SCAN = 24
ISO_BEFORE = 2400
ISO_AFTER = 12000
# initial threshold and tuning step (same logic as your earlier extractor)
ISO_THRESHOLD_START = 0.06
ISO_STEP = 0.005
# --------------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

# --- model class (must match saved checkpoint) ---
import torch.nn as nn
class SimpleCNN(nn.Module):
    def __init__(self, n_mfcc=N_MFCC, n_frames=MAX_FRAMES, n_classes=36):
        super().__init__()
        in_ch = n_mfcc * 3
        self.conv1 = nn.Conv1d(in_channels=in_ch, out_channels=128, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(256)
        self.pool2 = nn.MaxPool1d(2)
        out_t = (n_frames // 2) // 2
        self.fc1 = nn.Linear(256 * out_t, 512)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, n_classes)
    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = torch.relu(x); x = self.pool1(x)
        x = self.conv2(x); x = self.bn2(x); x = torch.relu(x); x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x)); x = self.dropout(x)
        x = self.fc2(x)
        return x

def load_model(path=MODEL_PATH):
    ckpt = torch.load(path, map_location="cpu")
    n_classes = 36
    model = SimpleCNN(n_mfcc=N_MFCC, n_frames=MAX_FRAMES, n_classes=n_classes)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model

# --- isolator (same as your code) ---
def isolator(signal, sample_rate, size, scan, before, after, threshold, show=False):
    strokes = []
    fft = librosa.stft(signal, n_fft=size, hop_length=scan)
    energy = np.abs(np.sum(fft, axis=0)).astype(float)
    threshed = energy > threshold
    peaks = np.where(threshed == True)[0]
    prev_end = int(-0.1 * sample_rate)
    for i in range(len(peaks)):
        this_peak = peaks[i]
        timestamp = (this_peak * scan) + size//2
        if timestamp > prev_end + int(0.1 * sample_rate):
            start = max(0, timestamp - before)
            end = min(len(signal), timestamp + after)
            keystroke = signal[start:end]
            strokes.append(torch.tensor(keystroke)[None, :])
            prev_end = timestamp + after
    return strokes

# --- MFCC stack (mfcc + delta + delta2) same as training ---
import librosa
import numpy as np

TARGET_SR = 48000  # training sample rate

def compute_mfcc_stack(y, sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH, max_frames=MAX_FRAMES):
    # resample to training rate if needed
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    d1 = librosa.feature.delta(mf, order=1)
    d2 = librosa.feature.delta(mf, order=2)
    def pad_trunc(arr):
        if arr.shape[1] < max_frames:
            pad_width = max_frames - arr.shape[1]
            return np.pad(arr, ((0,0),(0,pad_width)), mode='constant', constant_values=0.0)
        else:
            return arr[:, :max_frames]
    mf = pad_trunc(mf); d1 = pad_trunc(d1); d2 = pad_trunc(d2)
    stacked = np.vstack([mf, d1, d2])
    stacked = (stacked - stacked.mean(axis=1, keepdims=True)) / (stacked.std(axis=1, keepdims=True) + 1e-9)
    return stacked.astype(np.float32)


# --- helper to run model on a waveform and print topk ---
def predict_on_wave(model, wav, sr, topk=5):
    x = compute_mfcc_stack(wav, sr)
    inp = torch.from_numpy(x).unsqueeze(0)  # (1, channels, frames)
    with torch.no_grad():
        out = model(inp)
        probs = F.softmax(out, dim=1).cpu().numpy()[0]
    idxs = probs.argsort()[::-1][:topk]
    return [(int(i), mapping[int(i)], float(probs[i])) for i in idxs]

# --- plotting helper ---
def save_plot(wav, sr, out_path, title=None):
    plt.figure(figsize=(8,3))
    plt.subplot(2,1,1)
    librosa.display.waveshow(wav, sr=sr)
    if title:
        plt.title(title)
    plt.subplot(2,1,2)
    D = np.abs(librosa.stft(wav, n_fft=1024, hop_length=128))
    librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), sr=sr, hop_length=128, x_axis='time', y_axis='log')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# ---------------- main ----------------
def main():
    if len(sys.argv) > 1:
        wav_path = sys.argv[1]
    else:
        wav_path = "data/Zoom/1.wav"   # default example - change as needed

    if not os.path.exists(wav_path):
        print("File not found:", wav_path)
        return

    print("Loading WAV:", wav_path)
    samples, sr = librosa.load(wav_path, sr=None)
    if sr != 48000:
     samples = librosa.resample(samples, orig_sr=sr, target_sr=48000)
     sr = 48000
   # preserve original sr
    print(f"Original sample rate: {sr}, total samples: {len(samples)}, duration: {len(samples)/sr:.3f}s")

    # 1) try to isolate strokes with auto-threshold (same logic used earlier)
    strokes = []
    prom = ISO_THRESHOLD_START
    step = ISO_STEP
    attempts = 0
    while attempts < 200 and not (len(strokes) == 25):
        strokes = isolator(samples[int(1*sr):], sr, ISO_SIZE, ISO_SCAN, ISO_BEFORE, ISO_AFTER, prom, show=False)
        if len(strokes) < 25:
            prom -= step
        if len(strokes) > 25:
            prom += step
        if prom <= 0:
            print("-- isolator threshold fell <=0; stopping auto-tune")
            break
        step *= 0.99
        attempts += 1

    print("Isolated strokes found:", len(strokes))
    model = load_model()

    if len(strokes) == 0:
        print("No strokes found. Running inference on the full uploaded audio (not ideal).")
        # normalize
        wav = samples / (np.max(np.abs(samples)) + 1e-9)
        preds = predict_on_wave(model, wav, sr)
        print("Predictions on full audio (top3):", preds[:3])
        save_plot(wav, sr, os.path.join(OUT_DIR, "full_audio.png"), title="Full audio")
        print("Saved full audio plot to:", os.path.join(OUT_DIR, "full_audio.png"))
        return

    # For each stroke, predict and save plot
    for i, t in enumerate(strokes):
        # t is tensor shaped (1, L) — convert to 1D numpy
        if isinstance(t, torch.Tensor):
            if t.ndim == 2 and t.shape[0] == 1:
                wav = t.squeeze(0).numpy()
            else:
                wav = t.numpy()
        else:
            wav = np.asarray(t).squeeze()

        # normalize snippet by peak (same as dataset prep)
        wav = wav / (np.max(np.abs(wav)) + 1e-9)
        duration = len(wav) / sr
        print(f"\nStroke idx {i}: samples={len(wav)}, duration={duration:.3f}s")

        preds = predict_on_wave(model, wav, sr, topk=5)
        for rank, (idx, ch, prob) in enumerate(preds):
            print(f"  top{rank+1}: label={idx} char='{ch}' prob={prob:.4f}")

        out_png = os.path.join(OUT_DIR, f"stroke_{i:02d}_pred_{preds[0][1]}.png")
        save_plot(wav, sr, out_png, title=f"stroke {i} pred {preds[0][1]} ({preds[0][2]:.3f})")
        print("  saved plot:", out_png)

    print("\nDone. Inspect the printed predictions and the PNGs in", OUT_DIR)

if __name__ == "__main__":
    main()
