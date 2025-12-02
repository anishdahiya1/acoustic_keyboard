import sys
sys.path.insert(0, 'src')
from train_cnn import SimpleCNN, compute_mfcc_seq
import librosa, numpy as np, torch, os
from scipy.signal import find_peaks
import soundfile as sf

os.makedirs('artifacts/isolated_snippets', exist_ok=True)
fn='data/Zoom/0.wav'
y, sr = librosa.load(fn, sr=48000)
print('Loaded', fn, 'len', len(y), 'sr', sr)

# compute short-time energy over 10ms frames
frame_ms = 10
frame_len = int(sr * frame_ms / 1000)
hop = frame_len // 2
frames = librosa.util.frame(y, frame_length=frame_len, hop_length=hop)
energy = (frames ** 2).mean(axis=0)
# smooth energy
energy = np.convolve(energy, np.ones(5)/5, mode='same')
peaks, _ = find_peaks(energy, height=np.percentile(energy, 95), distance=int(0.05*sr/hop))
print('Found peaks:', len(peaks))

m = SimpleCNN(n_mfcc=40, n_frames=120, n_classes=36)
ck = torch.load('artifacts/cnn_finetune_merged_hard.pt', map_location='cpu')
m.load_state_dict(ck['model_state'])
m.eval()

KEYS_S = '1234567890qwertyuiopasdfghjklzxcvbnm'
mapping = {i:ch for i,ch in enumerate(KEYS_S)}

results = []
for i,p in enumerate(peaks[:30]):
    center = p * hop
    start = max(0, center - int(0.02*sr))
    end = min(len(y), center + int(0.05*sr))
    snippet = y[start:end]
    # pad to target length for wav save or inference
    if snippet.size == 0:
        continue
    # save snippet WAV
    out_w = f'artifacts/isolated_snippets/snippet_{i}.wav'
    sf.write(out_w, snippet, sr)
    # compute mfcc stack
    x = compute_mfcc_seq(snippet, sr)
    inp = torch.from_numpy(x).unsqueeze(0)
    with torch.no_grad():
        out = m(inp)
        import torch.nn.functional as F
        probs = F.softmax(out, dim=1).cpu().numpy()[0]
    topk = probs.argsort()[::-1][:5]
    top = [(int(k), mapping[int(k)], float(probs[int(k)])) for k in topk]
    print(i, 'snippet samples', snippet.shape[0], 'top1', top[0])
    results.append({'idx': i, 'file': out_w, 'top': top})

print('\nTotal snippets inferred:', len(results))
for r in results[:10]:
    print(r)
