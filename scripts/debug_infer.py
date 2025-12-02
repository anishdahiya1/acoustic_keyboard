import sys
sys.path.insert(0, 'src')
from train_cnn import SimpleCNN, compute_mfcc_seq
import librosa, numpy as np, torch

fn = 'data/Zoom/0.wav'
print('Loading', fn)
y, sr = librosa.load(fn, sr=48000)
print('Loaded len samples', len(y), 'duration s', len(y)/sr)

x = compute_mfcc_seq(y, sr)
print('MFCC stacked shape', x.shape, 'min/max', x.min(), x.max())

m = SimpleCNN(n_mfcc=40, n_frames=120, n_classes=36)
ck = torch.load('artifacts/cnn_finetune_merged_hard.pt', map_location='cpu')
m.load_state_dict(ck['model_state'])
m.eval()

inp = torch.from_numpy(x).unsqueeze(0)
with torch.no_grad():
    out = m(inp)
    import torch.nn.functional as F
    probs = F.softmax(out, dim=1).cpu().numpy()[0]
    topk = probs.argsort()[::-1][:10]

KEYS_S = '1234567890qwertyuiopasdfghjklzxcvbnm'
mapping = {i:ch for i,ch in enumerate(KEYS_S)}
print('\nTop-10 predictions:')
for rank,k in enumerate(topk):
    print(rank+1, k, mapping[k], probs[k])
print('\nArgmax:', mapping[int(probs.argmax())], probs.max())
