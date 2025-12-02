import sys
sys.path.insert(0, 'src')
import os
import argparse
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import find_peaks
import torch
from train_cnn import SimpleCNN, compute_mfcc_seq


def compute_short_time_energy(y, sr, frame_ms=10, hop=None):
    frame_len = int(sr * frame_ms / 1000)
    if frame_len < 1:
        frame_len = 1
    if hop is None:
        hop = max(1, frame_len // 2)
    # pad y so framing works
    if len(y) < frame_len:
        pad = frame_len - len(y)
        y = np.pad(y, (0, pad))
    frames = librosa.util.frame(y, frame_length=frame_len, hop_length=hop)
    energy = (frames ** 2).mean(axis=0)
    # smooth
    energy = np.convolve(energy, np.ones(5) / 5, mode='same')
    return energy, frame_len, hop


def estimate_isolation_params(y, sr):
    duration_s = len(y) / sr
    # compute a short-time energy with a small frame for fine resolution
    energy, frame_len, hop = compute_short_time_energy(y, sr, frame_ms=10)

    # quick statistics
    e_max = float(energy.max())
    e_median = float(np.median(energy) + 1e-12)
    ratio = e_max / e_median

    # choose peak percentile based on dynamic range
    if ratio > 30:
        peak_percentile = 98
    elif ratio > 15:
        peak_percentile = 95
    elif ratio > 7:
        peak_percentile = 92
    else:
        peak_percentile = 90

    # rough estimate of inter-onset spacing: find loose peaks with low threshold
    loose_thresh = np.percentile(energy, 70)
    loose_peaks, _ = find_peaks(energy, height=loose_thresh, distance=max(1, int(0.02 * sr / hop)))
    if len(loose_peaks) >= 2:
        # median spacing in frames -> convert to ms
        spacings = np.diff(loose_peaks) * hop / sr * 1000.0
        median_spacing_ms = float(np.median(spacings))
    else:
        # fallback: assume typical typing speed with 100ms between strokes
        median_spacing_ms = 100.0 if duration_s > 1 else 60.0

    # set min_peak_distance a bit smaller than median spacing to allow close strokes
    min_peak_distance_ms = max(20.0, median_spacing_ms * 0.6)

    # adjust frame length: if audio is long prefer slightly larger frame to be robust
    if duration_s > 20:
        frame_ms = 12
    elif duration_s < 1.0:
        frame_ms = 5
    else:
        frame_ms = 10

    return {
        'frame_ms': int(frame_ms),
        'peak_percentile': int(peak_percentile),
        'min_peak_distance_ms': int(min_peak_distance_ms),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('wav', help='input wav file')
    p.add_argument('--model', default='artifacts/cnn_finetune_merged_hard.pt')
    p.add_argument('--out-dir', default='outputs')
    p.add_argument('--frame-ms', type=int, default=None, help='override estimated frame length in ms')
    p.add_argument('--peak-percentile', type=int, default=None, help='override peak energy percentile')
    p.add_argument('--min-peak-distance-ms', type=int, default=None, help='override min peak distance in ms')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs('artifacts/isolated_snippets', exist_ok=True)

    print('Loading', args.wav)
    y, sr = librosa.load(args.wav, sr=48000)
    print('Loaded samples', len(y), 'sr', sr, 'duration_s', len(y)/sr)

    params = estimate_isolation_params(y, sr)
    # apply overrides if provided
    if args.frame_ms is not None:
        params['frame_ms'] = int(args.frame_ms)
    if args.peak_percentile is not None:
        params['peak_percentile'] = int(args.peak_percentile)
    if args.min_peak_distance_ms is not None:
        params['min_peak_distance_ms'] = int(args.min_peak_distance_ms)
    print('Isolation params used:', params)

    # compute energy with chosen frame/hop
    energy, frame_len, hop = compute_short_time_energy(y, sr, frame_ms=params['frame_ms'])
    peak_thresh = np.percentile(energy, params['peak_percentile'])
    min_dist_frames = max(1, int(params['min_peak_distance_ms'] * sr / hop / 1000.0))
    peaks, props = find_peaks(energy, height=peak_thresh, distance=min_dist_frames)
    print('Found peaks (snippets):', len(peaks))

    # load model
    m = SimpleCNN(n_mfcc=40, n_frames=120, n_classes=36)
    ck = torch.load(args.model, map_location='cpu')
    m.load_state_dict(ck['model_state'])
    m.eval()

    KEYS_S = '1234567890qwertyuiopasdfghjklzxcvbnm'
    mapping = {i: ch for i, ch in enumerate(KEYS_S)}

    import csv
    out_csv = os.path.join(args.out_dir, os.path.basename(args.wav).rsplit('.', 1)[0] + '_inference.csv')
    with open(out_csv, 'w', newline='', encoding='utf-8') as cf:
        writer = csv.DictWriter(cf, fieldnames=['idx', 'start_sample', 'end_sample', 'out_wav', 'top1_label', 'top1_prob', 'top5'])
        writer.writeheader()

        for i, p in enumerate(peaks[:200]):
            center = p * hop
            start = max(0, int(center - int(0.02 * sr)))
            end = min(len(y), int(center + int(0.05 * sr)))
            snippet = y[start:end]
            if snippet.size == 0:
                continue
            out_w = f'artifacts/isolated_snippets/snippet_{i}.wav'
            sf.write(out_w, snippet, sr)

            x = compute_mfcc_seq(snippet, sr)
            inp = torch.from_numpy(x).unsqueeze(0)
            with torch.no_grad():
                out = m(inp)
                import torch.nn.functional as F
                probs = F.softmax(out, dim=1).cpu().numpy()[0]
            topk = probs.argsort()[::-1][:5]
            top = [(int(k), mapping[int(k)], float(probs[int(k)])) for k in topk]
            top1 = top[0]
            print(i, 'samples', snippet.shape[0], 'top1', top1)
            writer.writerow({'idx': i, 'start_sample': start, 'end_sample': end, 'out_wav': out_w,
                             'top1_label': top1[1], 'top1_prob': top1[2], 'top5': str(top)})

    print('Wrote CSV to', out_csv)


if __name__ == '__main__':
    main()
