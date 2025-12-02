# src/fine_tune_hard.py
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score
from train_cnn import SimpleCNN, KeystrokeDataset, compute_mfcc_seq  # reuse from training script

# config
NPZ_PATH = "artifacts/zoom_dataset_npz.npz"
CKPT_IN = "artifacts/cnn_keystroke_resopt.pt"
CKPT_OUT = "artifacts/cnn_finetune_hard.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FINETUNE_EPOCHS = 6
BATCH_SIZE = 32
LR = 5e-5  # small LR for fine-tuning
HARD_LABELS = [0, 4]  # labels observed misclassified; change if you want different ones
os.makedirs("artifacts", exist_ok=True)

# load data (raw waveforms)
data = np.load(NPZ_PATH)
X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_val"], data["y_val"]
X_test, y_test = data["X_test"], data["y_test"]

# build dataset objects (with lightweight on-the-fly augmentation)
train_ds = KeystrokeDataset(X_train, y_train, precompute=False, augment=True)
val_ds = KeystrokeDataset(X_val, y_val, precompute=False, augment=False)
test_ds = KeystrokeDataset(X_test, y_test, precompute=False, augment=False)

# create sample weights: boost hardness for HARD_LABELS
counts = np.bincount(y_train.astype(int), minlength=int(y_train.max())+1).astype(float)
# base weight = inverse frequency
base_weights = 1.0 / (counts + 1e-9)
# increase weight for hard labels by factor
hard_factor = 4.0
class_weights = base_weights.copy()
for h in HARD_LABELS:
    class_weights[int(h)] *= hard_factor

# per-sample weights
sample_weights = class_weights[y_train.astype(int)]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=1, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=1, pin_memory=True)

# load model
n_classes = int(max(y_train.max(), y_val.max(), y_test.max()) + 1)
model = SimpleCNN(n_mfcc=40, n_frames=120, n_classes=n_classes).to(DEVICE)
ckpt = torch.load(CKPT_IN, map_location=DEVICE)
model.load_state_dict(ckpt["model_state"])  # should match resopt checkpoint
model.train()

optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-6)
criterion = nn.CrossEntropyLoss()

def evaluate(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE); y = y.to(DEVICE)
            out = model(x)
            preds.append(out.argmax(dim=1).cpu().numpy())
            trues.append(y.cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    return accuracy_score(trues, preds), preds, trues

def main():
    best_val = 0.0
    for epoch in range(1, FINETUNE_EPOCHS + 1):
        model.train()
        total = 0
        for x, y in train_loader:
            x = x.to(DEVICE); y = y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total += 1
        val_acc, _, _ = evaluate(model, val_loader)
        test_acc, _, _ = evaluate(model, test_loader)
        print(f"Finetune Epoch {epoch}/{FINETUNE_EPOCHS}  val_acc={val_acc:.4f}  test_acc={test_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "test_acc": test_acc
            }, CKPT_OUT)
            print("  Saved best fine-tuned model ->", CKPT_OUT)

    # final eval
    ckpt2 = torch.load(CKPT_OUT, map_location=DEVICE)
    model.load_state_dict(ckpt2["model_state"])
    final_test_acc, preds, trues = evaluate(model, test_loader)
    print("\nFinal test acc (fine-tuned):", final_test_acc)
    # print remaining misclassifications (if any)
    mis = []
    for i, (t, p) in enumerate(zip(trues, preds)):
        if t != p:
            mis.append((i, int(t), int(p)))
    print("Misclassified count (test):", len(mis))
    print("First few misclassified (idx, true, pred):", mis[:8])


if __name__ == '__main__':
    # On Windows the 'spawn' start method requires this guard when using
    # DataLoader with num_workers>0. This ensures multiprocessing forks safely.
    import multiprocessing as mp
    mp.freeze_support()
    main()
