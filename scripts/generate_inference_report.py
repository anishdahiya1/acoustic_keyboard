import os
import csv
import ast
import argparse
import soundfile as sf
import librosa
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save_spectrogram(wav_path, out_png, sr=48000, n_fft=1024, hop_length=256):
    y, _ = librosa.load(wav_path, sr=sr)
    if len(y) == 0:
        return False
    S = librosa.feature.melspectrogram(y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(3, 2))
    librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(out_png, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()
    return True


def smooth_labels(labels, window=3):
    # simple majority smoothing over sliding window
    n = len(labels)
    out = labels.copy()
    for i in range(n):
        lo = max(0, i - window // 2)
        hi = min(n, i + window // 2 + 1)
        window_items = [l for l in labels[lo:hi] if l is not None]
        if not window_items:
            continue
        # pick majority
        vals, counts = np.unique(window_items, return_counts=True)
        out[i] = vals[np.argmax(counts)]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('csv', help='inference CSV produced by run_inference_auto_isolate')
    p.add_argument('--out-dir', default='outputs')
    p.add_argument('--prob-threshold', type=float, default=0.28, help='min top1 prob to accept label')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    with open(args.csv, 'r', encoding='utf-8') as cf:
        reader = csv.DictReader(cf)
        for r in reader:
            # parse top5 stored as Python list
            try:
                top5 = ast.literal_eval(r.get('top5', '[]'))
            except Exception:
                top5 = []
            row = {
                'idx': int(r['idx']),
                'start_sample': int(r['start_sample']),
                'end_sample': int(r['end_sample']),
                'out_wav': r['out_wav'],
                'top1_label': r.get('top1_label', ''),
                'top1_prob': float(r.get('top1_prob', 0.0)),
                'top5': top5,
            }
            rows.append(row)

    rows.sort(key=lambda x: x['idx'])

    labels = []
    probs = []
    for r in rows:
        if r['top1_prob'] >= args.prob_threshold:
            labels.append(r['top1_label'])
        else:
            labels.append(None)
        probs.append(r['top1_prob'])

    # apply smoothing to fill short unknown gaps
    smoothed = smooth_labels(labels, window=3)

    base = os.path.basename(args.csv).rsplit('.', 1)[0]
    html_path = os.path.join(args.out_dir, base + '_report.html')
    transcript_path = os.path.join(args.out_dir, base + '_transcript.txt')

    with open(html_path, 'w', encoding='utf-8') as hf:
        hf.write('<html><head><meta charset="utf-8"><title>Inference Report</title></head><body>')
        hf.write(f'<h2>Report for {base}</h2>')
        hf.write('<table border=1 cellpadding=4 cellspacing=0>')
        hf.write('<tr><th>idx</th><th>snippet</th><th>top1 (prob)</th><th>top5</th><th>accepted</th></tr>')
        for i, r in enumerate(rows):
            png = os.path.join(args.out_dir, f'{base}_snippet_{r["idx"]}.png')
            # create spectrogram
            try:
                ok = save_spectrogram(r['out_wav'], png)
            except Exception as e:
                ok = False
            top5_html = ''
            try:
                top5_html = '<br>'.join([f'{t[1]} ({t[2]:.2f})' for t in r['top5']])
            except Exception:
                top5_html = str(r['top5'])
            accepted = smoothed[i] if smoothed[i] is not None else 'UNK'
            hf.write('<tr>')
            hf.write(f'<td>{r["idx"]}</td>')
            if ok:
                hf.write(f'<td><img src="{os.path.basename(png)}" width=200></td>')
            else:
                hf.write(f'<td>(no image)</td>')
            hf.write(f'<td>{r["top1_label"]} ({r["top1_prob"]:.2f})</td>')
            hf.write(f'<td>{top5_html}</td>')
            hf.write(f'<td>{accepted}</td>')
            hf.write('</tr>')
        hf.write('</table></body></html>')

    # write transcript: join accepted labels, replacing None with '_' and compress contiguous '_'s
    seq = [s if s is not None else '_' for s in smoothed]
    # compress underscores
    comp = []
    prev = None
    for s in seq:
        if s == '_' and prev == '_':
            continue
        comp.append(s)
        prev = s
    with open(transcript_path, 'w', encoding='utf-8') as tf:
        tf.write(''.join([c if c != '_' else ' ' for c in comp]).strip())

    # copy images into same output dir (they were saved there already)
    # write summary to stdout
    print('Wrote HTML report to', html_path)
    print('Wrote transcript to', transcript_path)


if __name__ == '__main__':
    main()
