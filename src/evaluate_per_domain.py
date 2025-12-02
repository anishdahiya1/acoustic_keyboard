import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader

from train_cnn import SimpleCNN, KeystrokeDataset

CKPT = 'artifacts/cnn_finetune_merged_hard.pt'
ZOOM_NPZ = 'artifacts/zoom_dataset_npz.npz'
PHONE_TENS = 'artifacts/phone_keystrokes_tensors.pt'
PHONE_META = 'artifacts/phone_keystrokes_meta.csv'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            out = model(x)
            preds.append(out.argmax(dim=1).cpu().numpy())
            trues.append(y.cpu().numpy())
    if len(preds) == 0:
        return 0.0
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    return (preds == trues).mean()

def load_phone_arrays(target_len=None):
    import torch
    t = torch.load(PHONE_TENS)
    meta = pd.read_csv(PHONE_META)
    X_list = []
    y_list = []
    for i, arr in enumerate(t):
        if isinstance(arr, torch.Tensor):
            if arr.ndim == 2 and arr.shape[0] == 1:
                a = arr.squeeze(0).numpy()
            else:
                a = arr.numpy()
        else:
            a = np.asarray(arr)
        a = a.astype(np.float32)
        if a.size == 0:
            continue
        if target_len is not None:
            if a.shape[0] < target_len:
                pad = np.zeros(target_len - a.shape[0], dtype=np.float32)
                a = np.concatenate([a, pad])
            else:
                a = a[:target_len]
        X_list.append(a)
        y_list.append(int(meta['Key'].iloc[i]))
    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    return X, y

def main():
    if not os.path.exists(CKPT):
        print('Checkpoint not found:', CKPT)
        return
    ck = torch.load(CKPT, map_location=DEVICE)
    # infer n_classes from checkpoint if possible
    # default to 36
    n_classes = 36
    model = SimpleCNN(n_mfcc=40, n_frames=120, n_classes=n_classes).to(DEVICE)
    model.load_state_dict(ck['model_state'])

    # Zoom test
    if os.path.exists(ZOOM_NPZ):
        z = np.load(ZOOM_NPZ)
        X_test = z['X_test']; y_test = z['y_test']
        zoom_ds = KeystrokeDataset(X_test, y_test, precompute=False, augment=False)
        zoom_loader = DataLoader(zoom_ds, batch_size=64, shuffle=False, num_workers=1)
        zoom_acc = evaluate(model, zoom_loader, DEVICE)
        print('Zoom test acc:', zoom_acc)
    else:
        print('Zoom NPZ not found:', ZOOM_NPZ)

    # Phone test
    if os.path.exists(PHONE_TENS) and os.path.exists(PHONE_META):
        # use target_len equal to zoom sample length if available
        target_len = None
        if os.path.exists(ZOOM_NPZ):
            z = np.load(ZOOM_NPZ)
            target_len = z['X_train'].shape[1]
        Xp, yp = load_phone_arrays(target_len=target_len)
        phone_ds = KeystrokeDataset(Xp, yp, precompute=False, augment=False)
        phone_loader = DataLoader(phone_ds, batch_size=64, shuffle=False, num_workers=1)
        phone_acc = evaluate(model, phone_loader, DEVICE)
        print('Phone test acc:', phone_acc)
    else:
        print('Phone tensors/meta not found')

if __name__ == '__main__':
    main()
