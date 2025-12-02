import os
import numpy as np
import torch
import librosa
import matplotlib.pyplot as plt

from train_cnn import SimpleCNN

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'artifacts')
ARTIFACTS_DIR = os.path.abspath(ARTIFACTS_DIR)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
OUT_DIR = os.path.join(ARTIFACTS_DIR, 'phone_failure_analysis')
os.makedirs(OUT_DIR, exist_ok=True)

PHONE_TENSORS = os.path.join(ARTIFACTS_DIR, 'phone_keystrokes_tensors.pt')
PHONE_META = os.path.join(ARTIFACTS_DIR, 'phone_keystrokes_meta.csv')
CHECKPOINT = os.path.join(ARTIFACTS_DIR, 'cnn_finetune_merged_hard.pt')


def load_phone_tensors():
    if not os.path.exists(PHONE_TENSORS):
        raise FileNotFoundError(PHONE_TENSORS)
    data = torch.load(PHONE_TENSORS)
    # assume shape (N, L) float tensors
    X = data.numpy()
    return X


def compute_simple_snr(x, sr=48000, ref_ms=50):
    # energy in signal center vs edges as proxy
    N = x.shape[0]
    ref = int(sr * ref_ms / 1000)
    if N < ref * 2:
        return float('nan')
    head = x[:ref]
    tail = x[-ref:]
    noise_energy = float((head ** 2).mean() + (tail ** 2).mean()) / 2
    sig_energy = float((x ** 2).mean())
    if noise_energy <= 0:
        return float('inf')
    return 10 * np.log10(sig_energy / noise_energy + 1e-12)


def mfcc_stack(x, sr=48000, n_mfcc=40, n_fft=1024, hop_length=128):
    mf = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    d1 = librosa.feature.delta(mf)
    d2 = librosa.feature.delta(mf, order=2)
    stacked = np.concatenate([mf, d1, d2], axis=0)
    return stacked


def plot_example(x, sr, stacked_mfcc, outpath, title=''):
    plt.figure(figsize=(8, 4))
    ax1 = plt.subplot(2, 1, 1)
    t = np.arange(len(x)) / sr
    ax1.plot(t, x)
    ax1.set_xlabel('Time (s)')
    ax1.set_title(title)
    ax2 = plt.subplot(2, 1, 2)
    librosa.display.specshow(stacked_mfcc[:40], x_axis='time', sr=sr, hop_length=128)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def main():
    print('Loading phone tensors...')
    X = load_phone_tensors()
    print('Loaded', X.shape)

    device = torch.device('cpu')
    print('Loading checkpoint:', CHECKPOINT)
    ckpt = torch.load(CHECKPOINT, map_location=device)
    # try to infer model params
    try:
        model = SimpleCNN(n_mfcc=40, max_frames=120, n_classes=36)
        model.load_state_dict(ckpt['model_state_dict'])
    except Exception:
        # fallback: load weights where possible
        model = SimpleCNN(n_mfcc=40, max_frames=120, n_classes=36)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()

    sr = 48000
    preds = []
    trues = []
    metrics = []

    # For phone meta labels we don't have direct labels; assume test labels are in phone meta CSV if available
    # For diagnostics we'll just run model and look at confidence and SNR
    for i in range(X.shape[0]):
        x = X[i].astype(np.float32)
        # normalize
        if np.abs(x).max() > 0:
            x = x / (np.abs(x).max())
        stacked = mfcc_stack(x, sr=sr)
        # center/truncate/pad to model expected frames
        max_frames = model.max_frames if hasattr(model, 'max_frames') else 120
        if stacked.shape[1] < max_frames:
            pad = np.zeros((stacked.shape[0], max_frames - stacked.shape[1]), dtype=stacked.dtype)
            stacked2 = np.concatenate([stacked, pad], axis=1)
        else:
            stacked2 = stacked[:, :max_frames]
        inp = torch.from_numpy(stacked2).unsqueeze(0).float()
        with torch.no_grad():
            out = model(inp)
            prob = torch.nn.functional.softmax(out, dim=1).numpy()[0]
            pred = int(prob.argmax())
            conf = float(prob.max())
        preds.append(pred)
        trues.append(-1)
        snr = compute_simple_snr(x, sr=sr)
        energy = float((x ** 2).mean())
        metrics.append({'idx': i, 'snr': snr, 'energy': energy, 'conf': conf})

    # sort by confidence ascending (low-confidence = likely failures)
    metrics_sorted = sorted(metrics, key=lambda m: m['conf'])
    print('Total phone samples:', len(metrics))
    low_conf = metrics_sorted[:6]
    high_conf = metrics_sorted[-6:]
    print('Low confidence examples (idx, conf, snr, energy):')
    for m in low_conf:
        print(m['idx'], round(m['conf'], 3), round(m['snr'], 2), round(m['energy'], 6))

    # save plots
    print('Saving example plots to', OUT_DIR)
    for rank, m in enumerate(low_conf + high_conf):
        i = m['idx']
        x = X[i].astype(np.float32)
        if np.abs(x).max() > 0:
            x = x / (np.abs(x).max())
        stacked = mfcc_stack(x, sr=sr)
        outp = os.path.join(OUT_DIR, f'phone_example_{rank}_{i}.png')
        title = f'idx={i} conf={m["conf"]:.3f} snr={m["snr"]:.2f}'
        try:
            plot_example(x, sr, stacked, outp, title=title)
        except Exception as e:
            print('Plot failed for', i, e)

    print('Done.')


if __name__ == '__main__':
    main()
