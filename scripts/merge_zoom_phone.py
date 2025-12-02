import numpy as np
import torch
import pandas as pd
import os

OUT_PATH = "artifacts/zoom_phone_dataset_npz.npz"
ZOOM_PATH = "artifacts/zoom_dataset_npz.npz"
PHONE_TENS_PATH = "artifacts/phone_keystrokes_tensors.pt"
PHONE_META_PATH = "artifacts/phone_keystrokes_meta.csv"

os.makedirs("artifacts", exist_ok=True)

# load zoom npz
z = np.load(ZOOM_PATH)
Xtr_z, ytr_z = z['X_train'], z['y_train']
Xv_z, yv_z = z['X_val'], z['y_val']
Xt_z, yt_z = z['X_test'], z['y_test']
print('Zoom shapes:', Xtr_z.shape, Xv_z.shape, Xt_z.shape)

# load phone tensors + meta
tensors = torch.load(PHONE_TENS_PATH)
meta = pd.read_csv(PHONE_META_PATH)
print('Phone raw count:', len(tensors), 'meta rows:', len(meta))

# target length: use zoom's per-sample length
target_len = Xtr_z.shape[1]

Xp = []
yp = []
for i, t in enumerate(tensors):
    if isinstance(t, torch.Tensor):
        if t.ndim == 2 and t.shape[0] == 1:
            arr = t.squeeze(0).numpy()
        else:
            arr = t.numpy()
    else:
        arr = np.asarray(t)
    arr = arr.astype(np.float32)
    if arr.size == 0:
        # skip empty
        continue
    # pad or truncate
    if arr.shape[0] < target_len:
        pad = np.zeros(target_len - arr.shape[0], dtype=np.float32)
        arr = np.concatenate([arr, pad])
    else:
        arr = arr[:target_len]
    Xp.append(arr)
    # assume meta lines correspond one-to-one
    yp.append(int(meta['Key'].iloc[i]))

if len(Xp) == 0:
    raise SystemExit('No valid phone snippets found to merge')

X_p = np.stack(Xp, axis=0)
y_p = np.array(yp, dtype=np.int64)
print('Phone cleaned shapes:', X_p.shape, y_p.shape)

# shuffle phone data reproducibly and split
rng = np.random.default_rng(42)
perm = rng.permutation(X_p.shape[0])
X_p = X_p[perm]
y_p = y_p[perm]

train_frac, val_frac = 0.8, 0.1
N = X_p.shape[0]
n_train = int(train_frac * N)
n_val = int(val_frac * N)
n_test = N - n_train - n_val
Xtr_p, ytr_p = X_p[:n_train], y_p[:n_train]
Xv_p, yv_p = X_p[n_train:n_train+n_val], y_p[n_train:n_train+n_val]
Xt_p, yt_p = X_p[n_train+n_val:], y_p[n_train+n_val:]
print('Phone split ->', Xtr_p.shape, Xv_p.shape, Xt_p.shape)

# concat per-split and shuffle combined splits
def concat_and_shuffle(Ax, Ay, Bx, By, seed=42):
    X = np.concatenate([Ax, Bx], axis=0)
    y = np.concatenate([Ay, By], axis=0)
    rng = np.random.default_rng(seed)
    p = rng.permutation(X.shape[0])
    return X[p], y[p]

Xtr_c, ytr_c = concat_and_shuffle(Xtr_z, ytr_z, Xtr_p, ytr_p, seed=123)
Xv_c, yv_c = concat_and_shuffle(Xv_z, yv_z, Xv_p, yv_p, seed=123)
Xt_c, yt_c = concat_and_shuffle(Xt_z, yt_z, Xt_p, yt_p, seed=123)

print('Combined shapes:', Xtr_c.shape, Xv_c.shape, Xt_c.shape)

np.savez_compressed(OUT_PATH,
                    X_train=Xtr_c, y_train=ytr_c,
                    X_val=Xv_c, y_val=yv_c,
                    X_test=Xt_c, y_test=yt_c)
print('Saved combined dataset to', OUT_PATH)
