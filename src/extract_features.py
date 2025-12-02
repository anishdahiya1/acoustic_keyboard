# src/extract_features.py
import os
import numpy as np
import librosa

IN_PATH = "artifacts/zoom_dataset_npz.npz"
OUT_PATH = "artifacts/zoom_features_npz.npz"
SAMPLE_RATE = 48000  # change if your recordings use different SR
N_MFCC = 40

def mfcc_stats(signal, sr, n_mfcc=N_MFCC):
    # compute MFCCs (shape: n_mfcc x T)
    mf = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=1024, hop_length=256)
    # stats per coefficient across time: mean and std -> 2*n_mfcc features
    means = mf.mean(axis=1)
    stds = mf.std(axis=1)
    feat = np.concatenate([means, stds], axis=0)
    return feat.astype(np.float32)

def process_split(X, sr=SAMPLE_RATE):
    feats = []
    for i, x in enumerate(X):
        # x should be 1D numpy float32
        feat = mfcc_stats(x, sr)
        feats.append(feat)
    return np.stack(feats, axis=0)

def main():
    os.makedirs("artifacts", exist_ok=True)
    data = np.load(IN_PATH)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    print("Computing MFCC features (this may take a short while)...")
    Xtr_f = process_split(X_train)
    Xv_f  = process_split(X_val)
    Xt_f  = process_split(X_test)

    print("Feature shapes:", Xtr_f.shape, Xv_f.shape, Xt_f.shape)
    np.savez_compressed(OUT_PATH,
                        X_train=Xtr_f, y_train=y_train,
                        X_val=Xv_f, y_val=y_val,
                        X_test=Xt_f, y_test=y_test)
    print("Saved features to", OUT_PATH)

if __name__ == "__main__":
    main()
