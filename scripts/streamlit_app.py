import sys
sys.path.insert(0, 'src')
import streamlit as st
import tempfile
import os
import numpy as np
import librosa
import soundfile as sf
import torch
from train_cnn import SimpleCNN, compute_mfcc_seq
from scipy.signal import find_peaks

MODEL_PATH = 'artifacts/cnn_finetune_merged_hard.pt'


def isolate_snippets(y, sr, frame_ms=12, peak_percentile=90, min_peak_distance_ms=27):
    frame_len = int(sr * frame_ms / 1000)
    hop = max(1, frame_len // 2)
    if len(y) < frame_len:
        y = np.pad(y, (0, frame_len - len(y)))
    frames = librosa.util.frame(y, frame_length=frame_len, hop_length=hop)
    energy = (frames ** 2).mean(axis=0)
    energy = np.convolve(energy, np.ones(5)/5, mode='same')
    thresh = np.percentile(energy, peak_percentile)
    min_dist_frames = max(1, int(min_peak_distance_ms * sr / hop / 1000.0))
    peaks, _ = find_peaks(energy, height=thresh, distance=min_dist_frames)
    centers = peaks * hop
    snippets = []
    for c in centers:
        start = max(0, int(c - 0.02 * sr))
        end = min(len(y), int(c + 0.05 * sr))
        snippets.append((start, end, y[start:end]))
    return snippets


@st.cache_resource
def load_model(path=MODEL_PATH):
    m = SimpleCNN(n_mfcc=40, n_frames=120, n_classes=36)
    ck = torch.load(path, map_location='cpu')
    m.load_state_dict(ck['model_state'])
    m.eval()
    return m


def infer_snippet(model, snippet, sr):
    x = compute_mfcc_seq(snippet, sr)
    inp = torch.from_numpy(x).unsqueeze(0)
    with torch.no_grad():
        out = model(inp)
        import torch.nn.functional as F
        probs = F.softmax(out, dim=1).cpu().numpy()[0]
    return probs


def main():
    st.title('Keystroke classifier demo')
    st.write('Upload a WAV or choose an example. The app isolates short snippets and runs the CNN model.')

    uploaded = st.file_uploader('Upload WAV', type=['wav'])
    col1, col2 = st.columns(2)
    with col1:
        peak_percentile = st.slider('Peak percentile', 80, 99, 90)
    with col2:
        prob_thresh = st.slider('Accept prob threshold', 0.0, 1.0, 0.20)

    sample_files = [f for f in os.listdir('data/Zoom') if f.lower().endswith('.wav')]
    example = st.selectbox('Or pick example', ['--none--'] + sample_files)

    if uploaded is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tf:
            tf.write(uploaded.read())
            wav_path = tf.name
    elif example and example != '--none--':
        wav_path = os.path.join('data/Zoom', example)
    else:
        wav_path = None

    if wav_path:
        y, sr = librosa.load(wav_path, sr=48000)
        st.audio(wav_path)
        st.write('Loaded', os.path.basename(wav_path), 'samples', len(y), 'sr', sr)

        snippets = isolate_snippets(y, sr, frame_ms=12, peak_percentile=peak_percentile, min_peak_distance_ms=27)
        st.write('Found', len(snippets), 'snippets')

        model = load_model()
        KEYS_S = '1234567890qwertyuiopasdfghjklzxcvbnm'
        mapping = {i: ch for i, ch in enumerate(KEYS_S)}

        for i, (s, e, snip) in enumerate(snippets[:100]):
            probs = infer_snippet(model, snip, sr)
            topk = probs.argsort()[::-1][:5]
            top = [(mapping[int(k)], float(probs[int(k)])) for k in topk]
            top1_label, top1_prob = top[0]
            if top1_prob >= prob_thresh:
                accept = True
            else:
                accept = False
            st.markdown(f'**Snippet {i}** — samples {s}..{e} — top1: {top1_label} ({top1_prob:.2f}) — accepted: {accept}')
            st.bar_chart([p for (_, p) in top])


if __name__ == '__main__':
    main()
