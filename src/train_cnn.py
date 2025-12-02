# src/train_cnn.py
import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import time
import json
import random

# ----------------------
# Config / hyperparams
# ----------------------
# ---------- replace the config block at the top of train_cnn.py with this ----------
SAMPLE_RATE = 48000
N_MFCC = 40
N_FFT = 1024
HOP_LENGTH = 128   # ← finer time resolution (was 256)
MAX_FRAMES = 120   # ← allow longer MFCC time axis (was 60)
BATCH_SIZE = 32    # smaller to fit GPU memory with larger input
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 25        # a few more epochs to let model adapt
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
NPZ_PATH = "artifacts/zoom_dataset_npz.npz"
OUT_MODEL = "artifacts/cnn_keystroke_resopt.pt"
METRIC_LOG = "artifacts/cnn_train_log_resopt.json"
os.makedirs("artifacts", exist_ok=True)
# -------------------------------------------------------------------------------


torch.manual_seed(SEED)
np.random.seed(SEED)

# ----------------------
# MFCC helper
# ----------------------
def compute_mfcc_seq(signal, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH, max_frames=MAX_FRAMES):
    # base MFCC (n_mfcc, T)
    mf = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    # deltas
    d1 = librosa.feature.delta(mf, order=1)
    d2 = librosa.feature.delta(mf, order=2)
    # pad / truncate each to max_frames
    def pad_trunc(arr):
        if arr.shape[1] < max_frames:
            pad_width = max_frames - arr.shape[1]
            return np.pad(arr, ((0,0),(0,pad_width)), mode='constant', constant_values=0.0)
        else:
            return arr[:, :max_frames]
    mf = pad_trunc(mf)
    d1 = pad_trunc(d1)
    d2 = pad_trunc(d2)
    # normalize per-coeff (stack then normalize per channel)
    stacked = np.vstack([mf, d1, d2])  # shape (3*n_mfcc, max_frames)
    # per-channel zero-mean unit-variance
    stacked = (stacked - stacked.mean(axis=1, keepdims=True)) / (stacked.std(axis=1, keepdims=True) + 1e-9)
    return stacked.astype(np.float32)

# ----------------------
# Dataset
# ----------------------
class KeystrokeDataset(Dataset):
    def __init__(self, X, y, sr=48000, precompute=False, augment=False):
        """
        X: (N, L) raw waveforms (float32, normalized)
        y: (N,)
        precompute: if True compute MFCCs once (keeps augment=False)
        augment: if True, apply on-the-fly simple augmentations (time-shift, noise, gain)
        """
        self.X = X
        self.y = y.astype(np.int64)
        self.sr = sr
        self.precompute = precompute
        self.augment = augment
        if precompute:
            # don't augment if precomputing MFCCs
            self.mfccs = [compute_mfcc_seq(x, sr=self.sr) for x in self.X]
        else:
            self.mfccs = None

    def __len__(self):
        return len(self.X)

    def _augment_wave(self, x):
        # x: 1D numpy array
        # 1) random small time shift (roll)
        max_shift = int(0.01 * self.sr)  # up to +/-10 ms
        shift = random.randint(-max_shift, max_shift)
        if shift != 0:
            x = np.roll(x, shift)
            # zero-fill the rolled area to avoid wrap-around artifacts
            if shift > 0:
                x[:shift] = 0.0
            else:
                x[shift:] = 0.0

        # 2) additive Gaussian noise (low)
        noise_level = random.uniform(0.0, 0.003)  # adjust small
        if noise_level > 0:
            x = x + np.random.normal(0.0, noise_level, size=x.shape).astype(np.float32)

        # 3) random gain (scale)
        gain = random.uniform(0.9, 1.1)
        x = x * gain

        # 4) small random high/low-pass like effect by slight smoothing or bias (optional)
        # keep simple for speed; we skip heavy transforms

        # ensure still float32 and clipped to [-1,1]
        x = np.clip(x, -1.0, 1.0).astype(np.float32)
        return x

    def __getitem__(self, idx):
        if self.precompute:
            mf = self.mfccs[idx]
        else:
            x = self.X[idx]
            if self.augment:
                x = self._augment_wave(x.copy())   # augment raw waveform before MFCC
            mf = compute_mfcc_seq(x, sr=self.sr)  # compute_mfcc_seq from train_cnn.py
        return torch.from_numpy(mf), torch.tensor(self.y[idx], dtype=torch.long)
# ----------------------
# Model: small Conv1D over time (treat MFCC coeffs as channels)
# Input: (batch, n_mfcc, max_frames)
# ----------------------
class SimpleCNN(nn.Module):
    def __init__(self, n_mfcc=N_MFCC, n_frames=MAX_FRAMES, n_classes=36):
        super().__init__()
        # previously: self.conv1 = nn.Conv1d(in_channels=n_mfcc, out_channels=128, kernel_size=5, padding=2)
        self.conv1 = nn.Conv1d(in_channels=n_mfcc * 3, out_channels=128, kernel_size=5, padding=2)

        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(256)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # compute flattened size after pooling
        def out_frames(frames):
            f = frames
            f = f // 2  # pool1
            f = f // 2  # pool2
            return f

        out_t = out_frames(n_frames)
        self.fc1 = nn.Linear(256 * out_t, 512)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, n_classes)

    def forward(self, x):
        # x shape: (batch, n_mfcc, n_frames)
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ----------------------
# Training / utils
# ----------------------
def evaluate(model, loader, device):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            p = out.argmax(dim=1).cpu().numpy()
            preds.append(p)
            trues.append(y.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    return accuracy_score(trues, preds), preds, trues

def train():
    data = np.load(NPZ_PATH)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    n_classes = int(max(y_train.max(), y_val.max(), y_test.max()) + 1)

    # datasets
    train_ds = KeystrokeDataset(X_train, y_train, precompute=False)
    val_ds = KeystrokeDataset(X_val, y_val, precompute=False)
    test_ds = KeystrokeDataset(X_test, y_test, precompute=False)

    # dataloaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=1, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=1, pin_memory=True)

    # model
    model = SimpleCNN(n_mfcc=N_MFCC, n_frames=MAX_FRAMES, n_classes=n_classes).to(DEVICE)
    # data from dataset returns (n_mfcc, max_frames) => we want (batch, n_mfcc, max_frames) -> good
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # compute class weights inversely proportional to class frequency (train set)
    import torch
    unique, counts = np.unique(y_train, return_counts=True)
    freq = dict(zip(unique.astype(int), counts.astype(float)))
    # ensure all classes 0..n_classes-1 are represented in freq (missing -> count=1)
    class_counts = np.array([freq.get(i, 1.0) for i in range(n_classes)], dtype=np.float32)
    class_weights = 1.0 / (class_counts + 1e-9)
    # normalize weights to have mean 1 (keeps learning rate behavior similar)
    class_weights = class_weights / class_weights.mean()
    class_weights = torch.from_numpy(class_weights).float().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print("Using class weights:", class_weights.cpu().numpy())

    # Some torch versions' ReduceLROnPlateau does not accept the `verbose` kwarg.
    # Try creating it with verbose=True and fall back if the signature doesn't accept it.
    try:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3,verbose=True)
    except TypeError:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_val = 0.0
    history = {"train_acc": [], "val_acc": [], "test_acc": [], "epoch_times": []}
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        all_preds = []
        all_trues = []
        for x, y in train_loader:
            # x: (batch, n_mfcc, max_frames) as float32
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            all_preds.append(out.argmax(dim=1).detach().cpu().numpy())
            all_trues.append(y.cpu().numpy())

        train_preds = np.concatenate(all_preds, axis=0)
        train_trues = np.concatenate(all_trues, axis=0)
        train_acc = accuracy_score(train_trues, train_preds)

        val_acc, _, _ = evaluate(model, val_loader, DEVICE)
        test_acc, _, _ = evaluate(model, test_loader, DEVICE)

        epoch_time = time.time() - t0
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["test_acc"].append(test_acc)
        history["epoch_times"].append(epoch_time)

        scheduler.step(val_acc)

        print(f"Epoch {epoch:02d}/{EPOCHS}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  test_acc={test_acc:.4f}  time={epoch_time:.1f}s")

        # save best
        if val_acc > best_val:
            best_val = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "test_acc": test_acc,
            }, OUT_MODEL)
            print(f"  Saved best model (val_acc={val_acc:.4f}) -> {OUT_MODEL}")

    # final evaluation on test using best model
    ckpt = torch.load(OUT_MODEL, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    final_test_acc, _, _ = evaluate(model, test_loader, DEVICE)
    print(f"\nFinal test accuracy (best model): {final_test_acc:.4f}")

    # save history
    with open(METRIC_LOG, "w") as f:
        json.dump(history, f, indent=2)
    print("Saved training history to", METRIC_LOG)

if __name__ == "__main__":
    train()
