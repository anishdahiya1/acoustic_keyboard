# src/prepare_dataset.py
import os
import random
import numpy as np
import torch
import pandas as pd

# paths
tensors_path = "artifacts/zoom_keystrokes_tensors.pt"
meta_path = "artifacts/zoom_keystrokes_meta.csv"
out_path = "artifacts/zoom_dataset_npz.npz"

# params
seed = 42
train_frac = 0.8
val_frac = 0.1
test_frac = 0.1
assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

# 1) load
wave_tensors = torch.load(tensors_path)   # list of tensors shaped (1, L)
meta = pd.read_csv(meta_path)

# 2) convert to numpy (squeeze first dim)
X_list = []
for t in wave_tensors:
    if t.ndim == 2 and t.shape[0] == 1:
        arr = t.squeeze(0).numpy()
    else:
        arr = t.numpy()
    X_list.append(arr.astype(np.float32))

y = meta['Key'].astype(np.int64).to_numpy()

# 3) stack -> (N, L)
X = np.stack(X_list, axis=0)   # shape (N, L)
print("X.shape =", X.shape, "y.shape =", y.shape)

# 4) simple normalization (per-sample peak normalization to [-1,1])
# avoids huge amplitude differences between recordings
eps = 1e-9
X = X / (np.max(np.abs(X), axis=1, keepdims=True) + eps)

# 5) shuffle with reproducible seed
rng = np.random.default_rng(seed)
perm = rng.permutation(X.shape[0])
X = X[perm]
y = y[perm]

# 6) split
N = X.shape[0]
n_train = int(train_frac * N)
n_val = int(val_frac * N)
n_test = N - n_train - n_val

X_train = X[:n_train]
y_train = y[:n_train]
X_val = X[n_train:n_train+n_val]
y_val = y[n_train:n_train+n_val]
X_test = X[n_train+n_val:]
y_test = y[n_train+n_val:]

print(f"split -> train: {X_train.shape[0]}, val: {X_val.shape[0]}, test: {X_test.shape[0]}")

# 7) save as compressed npz
os.makedirs("artifacts", exist_ok=True)
np.savez_compressed(out_path,
                    X_train=X_train, y_train=y_train,
                    X_val=X_val, y_val=y_val,
                    X_test=X_test, y_test=y_test)
print("Saved dataset to", out_path)
