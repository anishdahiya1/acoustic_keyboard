# src/tta_inference.py
import os
import numpy as np
import torch
import torch.nn.functional as F
from train_cnn import SimpleCNN, compute_mfcc_seq  # reuse model+mfcc code
from train_cnn import KeystrokeDataset  # not used for TTA but available
NPZ_PATH = "artifacts/zoom_dataset_npz.npz"
CKPT_PATH = "artifacts/cnn_finetune_hard.pt"   # adjust if you used another ckpt
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_TTA = 9   # number of augmented variants per sample (odd is fine)
soft = True  # average probabilities; if False average logits

# small augmentations used at test time (must be small)
def tta_variants(wav, sr=48000):
    # return list of numpy arrays including original
    variants = []
    variants.append(wav.copy())
    L = len(wav)
    max_shift = int(0.005 * sr)  # up to +/-5 ms
    for i in range(N_TTA-1):
        x = wav.copy()
        # small random shift
        shift = np.random.randint(-max_shift, max_shift+1)
        if shift != 0:
            x = np.roll(x, shift)
            if shift > 0:
                x[:shift] = 0.0
            else:
                x[shift:] = 0.0
        # small noise
        noise_level = np.random.uniform(0.0, 0.002)
        if noise_level > 0:
            x = x + np.random.normal(0.0, noise_level, size=x.shape).astype(np.float32)
        # small gain
        gain = np.random.uniform(0.95, 1.05)
        x = np.clip(x * gain, -1.0, 1.0).astype(np.float32)
        variants.append(x)
    return variants

def load_model():
    data = np.load(NPZ_PATH)
    y_test = data["y_test"]
    n_classes = int(y_test.max()) + 1
    model = SimpleCNN(n_mfcc=40, n_frames=120, n_classes=n_classes).to(DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model

def main():
    data = np.load(NPZ_PATH)
    X_test = data["X_test"]
    y_test = data["y_test"].astype(int)
    model = load_model()

    # baseline (no TTA)
    preds = []
    with torch.no_grad():
        for x in X_test:
            mf = compute_mfcc_seq(x)         # (C, T)
            inp = torch.from_numpy(mf).unsqueeze(0).to(DEVICE)
            out = model(inp)                # logits
            preds.append(int(out.argmax(dim=1).cpu().item()))
    preds = np.array(preds)
    acc_base = (preds == y_test).mean()
    print("Baseline test acc:", acc_base)

    # TTA inference
    preds_tta = []
    with torch.no_grad():
        for idx, x in enumerate(X_test):
            variants = tta_variants(x)
            probs = []
            for v in variants:
                mf = compute_mfcc_seq(v)
                inp = torch.from_numpy(mf).unsqueeze(0).to(DEVICE)
                out = model(inp)
                if soft:
                    p = F.softmax(out, dim=1).cpu().numpy()[0]
                else:
                    p = out.cpu().numpy()[0]   # logits
                probs.append(p)
            avg = np.mean(probs, axis=0)
            if soft:
                pred = int(np.argmax(avg))
            else:
                pred = int(np.argmax(avg))  # logits averaged -> same argmax step
            preds_tta.append(pred)
    preds_tta = np.array(preds_tta)
    acc_tta = (preds_tta == y_test).mean()
    print("TTA test acc :", acc_tta)

    # show which samples changed (and remaining misclassifications)
    changed = np.where(preds != preds_tta)[0]
    print("Num predictions changed by TTA:", len(changed))
    mis_idx = np.where(preds_tta != y_test)[0]
    print("Remaining misclassified count:", len(mis_idx))
    print("First few remaining (idx true pred):")
    for i in mis_idx[:8]:
        print(f"  idx={i}  {y_test[i]} -> {preds_tta[i]}")

if __name__ == '__main__':
    main()
