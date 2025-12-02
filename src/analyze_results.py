# src/analyze_results.py  (overwrite existing)
import os
import numpy as np
import joblib
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

# ---- CONFIG ----
# point to your best model here (use .pt for PyTorch or .joblib for sklearn)
MODEL_PATH = "artifacts/cnn_keystroke_resopt.pt"
FEAT_PATH = "artifacts/zoom_features_npz.npz"
OUT_IMG = "artifacts/confusion_matrix.png"
os.makedirs("artifacts", exist_ok=True)

# ---- helper to load PyTorch model if needed ----
SimpleCNN = None
try:
    # import SimpleCNN and config if train_cnn.py exists in src/
    from train_cnn import SimpleCNN, N_MFCC, MAX_FRAMES
except Exception:
    # if import fails, we'll restrict to sklearn models or raise later for .pt
    SimpleCNN = None

def load_model_auto(path, device="cpu"):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".joblib", ".pkl"]:
        model = joblib.load(path)
        return model, "sklearn"
    elif ext in [".pt", ".pth"]:
        if SimpleCNN is None:
            raise RuntimeError("Cannot load PyTorch model: SimpleCNN not importable from train_cnn.py.")
        ckpt = torch.load(path, map_location=device)
        # infer n_classes if stored, fallback to 36
        n_classes = ckpt.get("model_state") and None
        model = SimpleCNN(n_mfcc=N_MFCC, n_frames=MAX_FRAMES, n_classes=36)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        model.eval()
        return model, "pytorch"
    else:
        raise RuntimeError("Unsupported model extension: " + ext)

# ---- load model ----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model_obj, model_kind = load_model_auto(MODEL_PATH, device=DEVICE)
print("Loaded model:", MODEL_PATH, "as", model_kind)

# ---- load test features ----
data = np.load(FEAT_PATH)
# prefer test split in features; fallback to dataset npz if not present
if "X_test" in data and "y_test" in data:
    X_test = data["X_test"]
    y_test = data["y_test"]
else:
    raise RuntimeError(f"{FEAT_PATH} missing X_test/y_test arrays.")

# ---- predict depending on model type ----
# ---- predict depending on model type ----
if model_kind == "sklearn":
    # sklearn expects 2D arrays shape (N, D)
    y_pred = model_obj.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print("Test acc (recomputed):", test_acc)
else:
    # PyTorch path: compute MFCC+delta+delta2 sequences from raw waveforms, then predict in batches.
    # We'll load the raw waveform dataset npz (contains X_test raw waveforms)
    RAW_NPZ = "artifacts/zoom_dataset_npz.npz"
    if not os.path.exists(RAW_NPZ):
        raise RuntimeError(f"Required raw dataset {RAW_NPZ} not found. This file contains raw waveforms (N, L).")
    raw = np.load(RAW_NPZ)
    X_raw_test = raw["X_test"]   # shape (N, L) raw waveforms
    y_test = raw["y_test"]

    # import/config from train_cnn if available; otherwise set defaults
    try:
        from train_cnn import N_MFCC, N_FFT, HOP_LENGTH, MAX_FRAMES, SAMPLE_RATE
    except Exception:
        # fallback defaults used during training (should match your train_cnn.py)
        N_MFCC = 40
        N_FFT = 1024
        HOP_LENGTH = 128
        MAX_FRAMES = 120
        SAMPLE_RATE = 48000

    import librosa

    def compute_mfcc_stack(signal, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH, max_frames=MAX_FRAMES):
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
        stacked = np.vstack([mf, d1, d2])  # shape (3*n_mfcc, max_frames)
        # per-channel normalization
        stacked = (stacked - stacked.mean(axis=1, keepdims=True)) / (stacked.std(axis=1, keepdims=True) + 1e-9)
        return stacked.astype(np.float32)

    # compute features for all test samples (this may take a short while)
    print("Computing MFCC+delta features for test set (this may take a minute)...")
    X_seq = np.stack([compute_mfcc_stack(x) for x in X_raw_test], axis=0)  # shape (N, C, T)
    # Sanity: channels should match model conv1 in_channels
    try:
        conv_in_channels = model.conv1.in_channels
    except Exception:
        conv_in_channels = None
    if conv_in_channels is not None and X_seq.shape[1] != conv_in_channels:
        raise RuntimeError(f"Model expects {conv_in_channels} channels but computed features have {X_seq.shape[1]} channels. Check N_MFCC/have you stacked deltas?")

    # batch predict on device
    model = model_obj
    model.to(DEVICE)
    model.eval()
    batch_size = 64
    preds = []
    with torch.no_grad():
        for i in range(0, X_seq.shape[0], batch_size):
            batch = X_seq[i:i+batch_size]
            bt = torch.from_numpy(batch).to(DEVICE)  # shape (B, C, T)
            out = model(bt)
            p = out.argmax(dim=1).cpu().numpy()
            preds.append(p)
    y_pred = np.concatenate(preds, axis=0)
    test_acc = accuracy_score(y_test, y_pred)
    print("Test acc (recomputed):", test_acc)
