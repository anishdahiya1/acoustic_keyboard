#!/usr/bin/env python
# src/fine_tune_merged_hard.py
"""
Fine-tune the CNN on the merged Zoom+Phone dataset.
Usage:
  python src/fine_tune_merged_hard.py [--npz PATH] [--ckpt-out PATH] [--smoke]

--smoke runs a single training batch and exits (fast check).
"""
import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score
from train_cnn import SimpleCNN, KeystrokeDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--npz', default='artifacts/zoom_phone_dataset_npz.npz')
    p.add_argument('--ckpt-out', default='artifacts/cnn_finetune_merged_hard.pt')
    p.add_argument('--init-checkpoint', default=None, help='Path to checkpoint to initialize model from (optional)')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--epochs', type=int, default=6)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--hard-factor', type=float, default=4.0, help='Multiplier for hard labels when computing class/sample weights')
    p.add_argument('--hard-labels', nargs='*', type=int, default=[0,4])
    p.add_argument('--smoke', action='store_true', help='Run a single batch then exit')
    return p.parse_args()


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
        return 0.0, np.array([]), np.array([])
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    return accuracy_score(trues, preds), preds, trues


def main():
    args = parse_args()
    NPZ_PATH = args.npz
    CKPT_OUT = args.ckpt_out
    INIT_CKPT = args.init_checkpoint
    DEVICE = args.device
    FINETUNE_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LR = args.lr
    HARD_LABELS = args.hard_labels

    os.makedirs(os.path.dirname(CKPT_OUT) or '.', exist_ok=True)

    data = np.load(NPZ_PATH)
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']

    print('Loaded merged NPZ:', NPZ_PATH)
    print('Train/Val/Test sizes:', X_train.shape[0], X_val.shape[0], X_test.shape[0])

    train_ds = KeystrokeDataset(X_train, y_train, precompute=False, augment=True)
    val_ds = KeystrokeDataset(X_val, y_val, precompute=False, augment=False)
    test_ds = KeystrokeDataset(X_test, y_test, precompute=False, augment=False)

    # class weights
    n_classes = int(max(y_train.max(), y_val.max(), y_test.max()) + 1)
    counts = np.bincount(y_train.astype(int), minlength=n_classes).astype(float)
    base_weights = 1.0 / (counts + 1e-9)
    hard_factor = float(args.hard_factor)
    class_weights = base_weights.copy()
    for h in HARD_LABELS:
        if 0 <= int(h) < class_weights.shape[0]:
            class_weights[int(h)] *= hard_factor
    sample_weights = class_weights[y_train.astype(int)]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=1, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=1, pin_memory=True)

    model = SimpleCNN(n_mfcc=40, n_frames=120, n_classes=n_classes).to(DEVICE)
    # optionally initialize from a provided checkpoint (e.g., phone-only model)
    if INIT_CKPT and os.path.exists(INIT_CKPT):
        try:
            # Some checkpoints were saved with numpy types that require allowlisting
            # under PyTorch 2.6+ weights_only safety. Use a narrowly scoped allowlist
            # for the common numpy globals we saw and call torch.load with weights_only=False
            import inspect
            params = inspect.signature(torch.load).parameters
            kwargs = {'map_location': DEVICE}
            if 'weights_only' in params:
                kwargs['weights_only'] = False
            try:
                # use the safe_globals context manager to temporarily allowlist numpy globals
                allowlist = [np.dtype]
                try:
                    scalar_obj = np._core.multiarray.scalar
                except Exception:
                    scalar_obj = None
                if scalar_obj is not None:
                    allowlist.append(scalar_obj)
                with torch.serialization.safe_globals(allowlist):
                    ck = torch.load(INIT_CKPT, **kwargs)
            except Exception:
                # fallback to direct load (may raise same error)
                ck = torch.load(INIT_CKPT, **kwargs)

            # ck may contain different key names depending on how it was saved
            if isinstance(ck, dict):
                if 'model_state' in ck:
                    model.load_state_dict(ck['model_state'])
                elif 'model_state_dict' in ck:
                    model.load_state_dict(ck['model_state_dict'])
                elif 'model' in ck:
                    try:
                        model.load_state_dict(ck['model'])
                    except Exception:
                        model.load_state_dict(ck)
                else:
                    # maybe ck is already a state_dict
                    try:
                        model.load_state_dict(ck)
                    except Exception:
                        print('Loaded checkpoint dict but could not map keys directly; skipping init')
            else:
                # ck is likely a raw state_dict
                try:
                    model.load_state_dict(ck)
                except Exception:
                    print('Checkpoint loaded but failed to apply to model; skipping init')
            print('Initialized from', INIT_CKPT)
        except Exception as e:
            print('Could not initialize from', INIT_CKPT, 'error:', e)

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-6)
    criterion = nn.CrossEntropyLoss()

    best_val = 0.0
    for epoch in range(1, FINETUNE_EPOCHS + 1):
        model.train()
        for batch_i, (x, y) in enumerate(train_loader):
            x = x.to(DEVICE); y = y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            if args.smoke and batch_i >= 0:
                break

        val_acc, _, _ = evaluate(model, val_loader, DEVICE)
        test_acc, _, _ = evaluate(model, test_loader, DEVICE)
        print(f'Finetune Epoch {epoch}/{FINETUNE_EPOCHS}  val_acc={val_acc:.4f}  test_acc={test_acc:.4f}')

        if val_acc > best_val:
            best_val = val_acc
            torch.save({
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'test_acc': test_acc
            }, CKPT_OUT)
            print('  Saved best fine-tuned model ->', CKPT_OUT)

        if args.smoke:
            print('Smoke run enabled — exiting after one batch/epoch')
            break

    # final evaluation using best ckpt
    if os.path.exists(CKPT_OUT):
        ckpt2 = torch.load(CKPT_OUT, map_location=DEVICE)
        model.load_state_dict(ckpt2['model_state'])
        final_test_acc, preds, trues = evaluate(model, test_loader, DEVICE)
        print('\nFinal test acc (fine-tuned merged):', final_test_acc)


if __name__ == '__main__':
    main()
