# app.py
import streamlit as st
import numpy as np
import torch
import torch.nn.functional as F
import librosa
import librosa.display
import matplotlib.pyplot as plt
from io import BytesIO
import os

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap');
    :root {
        --bg-shell: #050913;
        --bg-panel: rgba(14, 20, 36, 0.96);
        --bg-card: rgba(21, 28, 46, 0.92);
        --accent-primary: #4ea0f6;
        --accent-secondary: #8b7cf6;
        --accent-tertiary: #6ef2c6;
        --stroke-subtle: rgba(255, 255, 255, 0.08);
        --text-strong: #f4f7ff;
        --text-muted: #9aa7c4;
    }
    body, [data-testid="stAppViewContainer"] {
        font-family: 'Space Grotesk', 'Inter', sans-serif;
        color: var(--text-strong);
        background-color: var(--bg-shell);
    }
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at 20% 20%, rgba(78,160,246,0.22), transparent 55%),
                    radial-gradient(circle at 80% 10%, rgba(110,242,198,0.18), transparent 45%),
                    linear-gradient(135deg, #04060e 0%, #070b16 55%, #05070f 100%);
        min-height: 100vh;
        position: relative;
    }
    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: fixed;
        inset: 0;
        background-image:
            linear-gradient(120deg, rgba(255,255,255,0.02) 1px, transparent 1px),
            linear-gradient(300deg, rgba(255,255,255,0.02) 1px, transparent 1px);
        background-size: 220px 220px, 180px 180px;
        opacity: 0.45;
        pointer-events: none;
    }
    [data-testid="stSidebar"] > div:first-child {
        background: var(--bg-panel);
        border-left: 1px solid var(--stroke-subtle);
        box-shadow: -12px 0 32px rgba(0,0,0,0.45);
    }
    .hero-card {
        position: relative;
        border-radius: 30px;
        padding: 48px 52px;
        background: linear-gradient(135deg, rgba(15,22,38,0.97), rgba(10,14,24,0.95));
        border: 1px solid var(--stroke-subtle);
        box-shadow: 0 40px 80px rgba(2,5,12,0.75);
        overflow: hidden;
        margin-bottom: 2.5rem;
    }
    .hero-card::before,
    .hero-card::after {
        content: "";
        position: absolute;
        width: 280px;
        height: 280px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(78,160,246,0.18), transparent 65%);
        filter: blur(8px);
        opacity: 0.7;
        pointer-events: none;
    }
    .hero-card::before { inset: -25% auto auto 50%; }
    .hero-card::after {
        background: radial-gradient(circle, rgba(139,124,246,0.22), transparent 60%);
        inset: auto 40% -30% auto;
    }
    .hero-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 6px 18px;
        border-radius: 999px;
        background: rgba(78,160,246,0.12);
        border: 1px solid rgba(78,160,246,0.4);
        color: var(--accent-primary);
        font-size: 0.85rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .hero-card h2 {
        margin: 1.5rem 0 0.75rem;
        font-size: 2.7rem;
        font-weight: 600;
        color: var(--text-strong);
        line-height: 1.25;
    }
    .gradient-text {
        background: linear-gradient(120deg, var(--accent-primary), var(--accent-secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-card p,
    .hero-card .micro-copy {
        color: var(--text-muted);
        max-width: 680px;
    }
    .hero-actions {
        margin-top: 1.6rem;
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        align-items: center;
    }
    .glow-button {
        padding: 0.85rem 2rem;
        border-radius: 999px;
        text-decoration: none;
        font-weight: 600;
        color: #07101f;
        background: linear-gradient(120deg, #f8fbff 0%, #d8e6ff 60%, #f4ecff 100%);
        border: 1px solid rgba(255,255,255,0.35);
        box-shadow: 0 18px 40px rgba(8,13,24,0.45);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    .glow-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 30px 50px rgba(8,13,24,0.55);
    }
    .hero-link {
        font-weight: 600;
        color: var(--accent-primary);
        text-decoration: none;
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
    }
    .hero-link::after {
        content: '→';
        font-size: 0.9rem;
    }
    .accent-divider {
        position: relative;
        width: 100%;
        margin: 2.5rem 0 1.8rem;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(78,160,246,0.7), transparent);
    }
    .badge-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 0.9rem;
        margin-bottom: 2rem;
    }
    .badge-card {
        padding: 0.65rem 1rem;
        border-radius: 16px;
        border: 1px solid var(--stroke-subtle);
        background: rgba(8,12,22,0.85);
        display: flex;
        flex-direction: column;
        gap: 0.3rem;
        min-height: 96px;
    }
    .badge-card strong {
        color: var(--accent-primary);
        font-size: 0.95rem;
    }
    .badge-card span {
        color: var(--text-muted);
        font-size: 0.8rem;
    }
    .metrics-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.75rem;
        margin-top: 1.2rem;
    }
    .metric-pill {
        border-radius: 999px;
        padding: 0.5rem 1.4rem;
        border: 1px solid var(--stroke-subtle);
        background: rgba(255,255,255,0.02);
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        font-weight: 600;
        color: var(--text-strong);
    }
    .metric-pill small {
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-muted);
        font-size: 0.65rem;
    }
    .highlight-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        gap: 1.2rem;
        margin-bottom: 2.5rem;
    }
    .highlight-card {
        position: relative;
        background: var(--bg-card);
        border-radius: 22px;
        padding: 24px;
        border: 1px solid var(--stroke-subtle);
        box-shadow: 0 18px 50px rgba(0,0,0,0.35);
        transition: transform 0.25s ease, border 0.25s ease;
        min-height: 180px;
    }
    .highlight-card::after {
        content: "";
        position: absolute;
        inset: 12px;
        border-radius: 18px;
        border: 1px dashed rgba(255,255,255,0.05);
        pointer-events: none;
    }
    .highlight-card:hover {
        transform: translateY(-6px);
        border-color: rgba(255,255,255,0.18);
    }
    .card-icon {
        font-size: 1.8rem;
        margin-bottom: 0.8rem;
        color: var(--accent-primary);
    }
    .stat-card {
        background: rgba(10,16,30,0.9);
        border-radius: 18px;
        padding: 18px 22px;
        border: 1px solid var(--stroke-subtle);
        box-shadow: 0 16px 42px rgba(0,0,0,0.45);
    }
    .stat-label {
        font-size: 0.75rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 0.35rem;
    }
    .stat-value {
        font-size: 1.55rem;
        font-weight: 600;
        color: var(--accent-primary);
    }
    .tip-list {
        list-style: none;
        padding-left: 0;
        margin: 0;
    }
    .tip-list li {
        margin-bottom: 0.6rem;
        display: flex;
        gap: 0.5rem;
        align-items: flex-start;
    }
    .tip-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: var(--accent-secondary);
        margin-top: 0.45rem;
    }
    .summary-card {
        background: var(--bg-card);
        border: 1px solid var(--stroke-subtle);
        border-radius: 22px;
        padding: 22px 26px;
        box-shadow: 0 24px 60px rgba(0,0,0,0.45);
        margin-bottom: 1.2rem;
    }
    .summary-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 6px 16px;
        border-radius: 999px;
        background: rgba(78,160,246,0.15);
        color: var(--accent-primary);
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .matrix-panel {
        position: relative;
        border-radius: 28px;
        padding: 32px 36px;
        background: linear-gradient(135deg, rgba(9,13,24,0.96), rgba(16,22,38,0.95));
        border: 1px solid var(--stroke-subtle);
        box-shadow: 0 30px 70px rgba(0,0,0,0.5);
        margin-bottom: 2.4rem;
        overflow: hidden;
    }
    .matrix-panel h3 {
        margin: 0;
        font-size: 1.35rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .matrix-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 1rem;
        margin-top: 1.3rem;
    }
    .matrix-card {
        background: rgba(6,10,20,0.85);
        border-radius: 18px;
        border: 1px solid var(--stroke-subtle);
        padding: 1rem;
        min-height: 130px;
        display: flex;
        flex-direction: column;
        gap: 0.4rem;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.02);
    }
    .matrix-card span {
        font-size: 0.78rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--text-muted);
    }
    .matrix-card strong {
        font-size: 1.2rem;
        color: var(--accent-primary);
    }
    .matrix-card small {
        color: var(--text-muted);
        line-height: 1.4;
    }
    [data-testid="stTabs"] > div:first-child {
        border-bottom: none;
        gap: 0.85rem;
    }
    [data-testid="stTabs"] button {
        border-radius: 16px;
        background: rgba(255,255,255,0.02);
        border: 1px solid transparent;
        color: var(--text-muted);
        font-weight: 600;
        padding: 0.45rem 1.25rem;
        transition: color 0.2s ease, border 0.2s ease, background 0.2s ease;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--text-strong);
        border-color: rgba(255,255,255,0.12);
        background: rgba(255,255,255,0.04);
        box-shadow: 0 12px 32px rgba(0,0,0,0.25);
    }
    .status-panel {
        display: flex;
        flex-wrap: wrap;
        gap: 0.75rem 1rem;
        align-items: center;
        justify-content: space-between;
        padding: 1rem 1.4rem;
        border-radius: 22px;
        background: rgba(7,11,20,0.85);
        border: 1px solid var(--stroke-subtle);
        box-shadow: 0 20px 50px rgba(0,0,0,0.4);
        margin-bottom: 1.5rem;
    }
    .status-message {
        font-size: 0.95rem;
        color: var(--text-muted);
    }
    .telemetry-badges {
        display: flex;
        gap: 0.75rem;
        flex-wrap: wrap;
    }
    .telemetry-chip {
        display: inline-flex;
        flex-direction: column;
        padding: 0.5rem 1rem;
        border-radius: 14px;
        border: 1px solid var(--stroke-subtle);
        background: rgba(4,8,16,0.85);
        min-width: 130px;
    }
    .telemetry-chip span {
        font-size: 0.65rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--text-muted);
    }
    .telemetry-chip strong {
        font-size: 1rem;
        color: var(--accent-primary);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- streamlit-webrtc imports ---
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import av
import queue
from typing import Optional

# ---------------- CONFIG (match training) ----------------
SAMPLE_RATE = 48000
N_MFCC = 40
N_FFT = 1024
HOP_LENGTH = 128
MAX_FRAMES = 120
MODEL_PATH = "artifacts/cnn_finetune_merged_hard.pt"   # use balanced merged checkpoint by default
KEYS_S = '1234567890qwertyuiopasdfghjklzxcvbnm'
mapping = {i: ch for i, ch in enumerate(KEYS_S)}
DEVICE = "cpu"
# --------------------------------------------------------

st.markdown(
    """
    <div class="hero-card">
        <span class="hero-pill">Live acoustic AI</span>
        <h2>Hear every keystroke. <span class="gradient-text">Visualize every pattern.</span></h2>
        <p>
            Stream real-time microphone data or upload raw WAVs, then isolate keystrokes with studio-grade controls.
            A fine-tuned CNN delivers top-5 predictions with optional TTA smoothing.
        </p>
        <div class="hero-actions">
            <a class="glow-button" href="#upload-section">Upload audio</a>
            <span class="micro-copy">Best results with mono 48 kHz recordings</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="accent-divider"></div>
    <div class="badge-grid">
        <div class="badge-card">
            <span>Latency tuned</span>
            <strong>&lt; 25 ms audio hop</strong>
        </div>
        <div class="badge-card">
            <span>Model checkpoint</span>
            <strong>Merged-hard CNN</strong>
        </div>
        <div class="badge-card">
            <span>Visualization</span>
            <strong>Waveform + LogSpec</strong>
        </div>
        <div class="badge-card">
            <span>Isolation control</span>
            <strong>Pre/post configurable</strong>
        </div>
    </div>
    <div class="metrics-row">
        <div class="metric-pill"><small>Inference</small>Top-5 logits</div>
        <div class="metric-pill"><small>TTA</small>1–15 passes</div>
        <div class="metric-pill"><small>Buffer</small>400 ms slices</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="highlight-grid">
        <div class="highlight-card">
            <span class="card-icon">🎧</span>
            <h4>Live capture lab</h4>
            <p>Calibrate microphone gain, preview buffered audio, and classify the last 400 ms with one tap.</p>
        </div>
        <div class="highlight-card">
            <span class="card-icon">📊</span>
            <h4>Explainable snippets</h4>
            <p>Overlay spectrograms, waveforms, and aggregated votes for every detected keystroke peak.</p>
        </div>
        <div class="highlight-card">
            <span class="card-icon">⚙️</span>
            <h4>Studio isolation</h4>
            <p>Adjust frame windows, thresholds, and pre/post roll to mirror the 300 ms training snippets.</p>
        </div>
        <div class="highlight-card">
            <span class="card-icon">🚀</span>
            <h4>TTA-ready</h4>
            <p>Blend up to 15 stochastic passes for robust predictions across laptops, phones, or Zoom audio.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="matrix-panel">
        <h3>Keystroke intelligence stack</h3>
        <p style="color: var(--text-muted); max-width: 720px; margin-top: 0.4rem;">
            Signal acquisition, isolation heuristics, and neural inference work together. Each tier can be tuned live to
            keep the CNN calibrated across laptops, phones, and conferencing apps.
        </p>
        <div class="matrix-grid">
            <div class="matrix-card">
                <span>Capture</span>
                <strong>48 kHz mono</strong>
                <small>Buffered microphone stream with WebRTC, auto-normalized gain, and audible preview.</small>
            </div>
            <div class="matrix-card">
                <span>Isolation</span>
                <strong>Adaptive frames</strong>
                <small>Pre/Post roll sliders mirror training windows; percentile thresholds reduce cross-talk.</small>
            </div>
            <div class="matrix-card">
                <span>Inference</span>
                <strong>Top-5 logits</strong>
                <small>Fine-tuned CNN (merged-hard checkpoint) with optional 15-pass TTA smoothing.</small>
            </div>
            <div class="matrix-card">
                <span>Feedback</span>
                <strong>Visual forensics</strong>
                <small>Spectrogram + waveform overlays expose ambiguous peaks and support rapid retuning.</small>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

DEFAULT_STATUS = "Model loads on first inference. Upload mono 48 kHz WAVs or use the mic tab to capture live audio."
if "telemetry" not in st.session_state:
    st.session_state["telemetry"] = {"latency_ms": None, "buffer_samples": None, "buffer_seconds": None}
if "status_message" not in st.session_state:
    st.session_state["status_message"] = DEFAULT_STATUS

status_placeholder = st.empty()

def render_status(message: Optional[str] = None):
    if message is not None:
        st.session_state["status_message"] = message
    telem = st.session_state.get("telemetry", {})
    latency_ms = telem.get("latency_ms")
    buffer_samples = telem.get("buffer_samples")
    buffer_seconds = telem.get("buffer_seconds")
    latency_display = f"{latency_ms:.0f} ms" if latency_ms is not None else "--"
    buffer_display = "--"
    if buffer_samples is not None and buffer_seconds is not None:
        buffer_display = f"{buffer_samples} smp / {buffer_seconds:.2f} s"
    status_placeholder.markdown(
        f"""
        <div class="status-panel">
            <div class="status-message">{st.session_state['status_message']}</div>
            <div class="telemetry-badges">
                <div class="telemetry-chip"><span>Latency</span><strong>{latency_display}</strong></div>
                <div class="telemetry-chip"><span>Buffer window</span><strong>{buffer_display}</strong></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

render_status()

@st.cache_data(show_spinner=False)
def load_dataset_overview(npz_path="artifacts/zoom_phone_dataset_npz.npz"):
    try:
        if not os.path.exists(npz_path):
            return None
        data = np.load(npz_path)
        train = data["X_train"].shape[0]
        val = data["X_val"].shape[0]
        test = data["X_test"].shape[0]
        sample_len = data["X_train"].shape[1]
        return {
            "train": int(train),
            "val": int(val),
            "test": int(test),
            "sample_len": int(sample_len),
        }
    except Exception:
        return None

stats = load_dataset_overview()
stat_cols = st.columns(4)
stat_cols[0].markdown('<div class="stat-card"><div class="stat-label">Classes</div><div class="stat-value">36</div></div>', unsafe_allow_html=True)
if stats is not None:
    stat_cols[1].markdown(f'<div class="stat-card"><div class="stat-label">Train clips</div><div class="stat-value">{stats["train"]}</div></div>', unsafe_allow_html=True)
    stat_cols[2].markdown(f'<div class="stat-card"><div class="stat-label">Validation clips</div><div class="stat-value">{stats["val"]}</div></div>', unsafe_allow_html=True)
    sample_ms = int((stats["sample_len"] / SAMPLE_RATE) * 1000)
    stat_cols[3].markdown(f'<div class="stat-card"><div class="stat-label">Snippet length</div><div class="stat-value">{sample_ms} ms</div></div>', unsafe_allow_html=True)
else:
    stat_cols[1].markdown('<div class="stat-card"><div class="stat-label">Dataset</div><div class="stat-value">Not found</div></div>', unsafe_allow_html=True)
    stat_cols[2].empty()
    stat_cols[3].empty()
import streamlit.components.v1 as components

# ---------------- Model definition (must match saved checkpoint) ----------------
import torch.nn as nn
class SimpleCNN(nn.Module):
    def __init__(self, n_mfcc=N_MFCC, n_frames=MAX_FRAMES, n_classes=36):
        super().__init__()
        in_ch = n_mfcc * 3
        self.conv1 = nn.Conv1d(in_channels=in_ch, out_channels=128, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(256)
        self.pool2 = nn.MaxPool1d(2)
        out_t = (n_frames // 2) // 2
        self.fc1 = nn.Linear(256 * out_t, 512)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, n_classes)
    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = torch.relu(x); x = self.pool1(x)
        x = self.conv2(x); x = self.bn2(x); x = torch.relu(x); x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x)); x = self.dropout(x)
        x = self.fc2(x)
        return x

@st.cache_resource
def load_model(path=MODEL_PATH, device=DEVICE):
    ckpt = torch.load(path, map_location=device)
    model = SimpleCNN(n_mfcc=N_MFCC, n_frames=MAX_FRAMES, n_classes=36)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model

# --------------- MFCC + delta stack helper (same as training) ----------------
import librosa
import numpy as np

TARGET_SR = 48000  # training sample rate

def compute_mfcc_stack(y, sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH, max_frames=MAX_FRAMES):
    # resample to training rate if needed
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    d1 = librosa.feature.delta(mf, order=1)
    d2 = librosa.feature.delta(mf, order=2)
    def pad_trunc(arr):
        if arr.shape[1] < max_frames:
            pad_width = max_frames - arr.shape[1]
            return np.pad(arr, ((0,0),(0,pad_width)), mode='constant', constant_values=0.0)
        else:
            return arr[:, :max_frames]
    mf = pad_trunc(mf); d1 = pad_trunc(d1); d2 = pad_trunc(d2)
    stacked = np.vstack([mf, d1, d2])
    stacked = (stacked - stacked.mean(axis=1, keepdims=True)) / (stacked.std(axis=1, keepdims=True) + 1e-9)
    return stacked.astype(np.float32)


# ---------------- Inference function ----------------
model = None
def infer(y, sr=SAMPLE_RATE, use_tta=False, n_tta=7):
    global model
    if model is None:
        model = load_model()
    if not use_tta:
        x = compute_mfcc_stack(y, sr)
        inp = torch.from_numpy(x).unsqueeze(0)  # (1, channels, frames)
        with torch.no_grad():
            logits = model(inp)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        topk = probs.argsort()[::-1][:5]
        return probs, topk
    else:
        # simple TTA: small shifts + noise + average probs
        probs_acc = None
        for i in range(n_tta):
            wav = y.copy()
            # small random shift
            shift = np.random.randint(-int(0.004*sr), int(0.004*sr)+1)
            if shift != 0:
                wav = np.roll(wav, shift)
                if shift > 0:
                    wav[:shift] = 0.0
                else:
                    wav[shift:] = 0.0
            # small noise
            wav = np.clip(wav + np.random.normal(0, np.random.uniform(0.0, 0.002), size=wav.shape), -1.0, 1.0)
            x = compute_mfcc_stack(wav, sr)
            inp = torch.from_numpy(x).unsqueeze(0)
            with torch.no_grad():
                logits = model(inp)
                p = F.softmax(logits, dim=1).cpu().numpy()[0]
            if probs_acc is None:
                probs_acc = p
            else:
                probs_acc += p
        probs = probs_acc / float(n_tta)
        topk = probs.argsort()[::-1][:5]
        return probs, topk

# ---------------- streamlit UI layout ----------------
with st.sidebar:
    st.markdown("## 🎛️ Inference controls")
    tta_enabled = st.toggle("Test-time augmentation", value=True, help="Average multiple perturbed passes for stabler logits")
    tta_passes = st.slider("TTA passes", min_value=1, max_value=15, value=7)
    st.caption("Tip: keep passes under 10 for real-time mic usage.")
    st.markdown("---")
    st.markdown("## 💡 Workflow tips")
    st.markdown(
        """
        <ul class="tip-list">
            <li><span class="tip-dot"></span><span>Record mono 48 kHz audio to stay aligned with the training corpus.</span></li>
            <li><span class="tip-dot"></span><span>Lower the peak percentile if the detector misses softer taps.</span></li>
            <li><span class="tip-dot"></span><span>Expand the post window when working with resonant laptop chassis.</span></li>
        </ul>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Checkpoint: {os.path.basename(MODEL_PATH)}")

st.markdown('<span id="upload-section"></span>', unsafe_allow_html=True)
uploaded = None
upload_tab, mic_tab = st.tabs(["Upload & isolate", "Live microphone"])

with upload_tab:
    st.subheader("Upload studio")
    st.caption("Drop a mono WAV with one or more keystrokes to auto-slice, visualize, and classify each tap.")
    uploaded = st.file_uploader("Upload .wav", type=["wav"], help="48 kHz mono recommended")

    if uploaded is not None:
        try:
            data_bytes = uploaded.read()
            y, sr = librosa.load(BytesIO(data_bytes), sr=SAMPLE_RATE)
            render_status(f"Loaded audio — {len(y)} samples, sr={sr}")

            with st.expander("Isolation studio", expanded=False):
                colp, colq, colr = st.columns(3)
                with colp:
                    frame_ms = st.slider('Frame length (ms)', min_value=5, max_value=40, value=12)
                with colq:
                    peak_pct = st.slider('Peak percentile', min_value=80, max_value=99, value=94)
                with colr:
                    min_peak_dist_ms = st.slider('Min peak distance (ms)', min_value=20, max_value=300, value=50)

                colpre, colpost = st.columns(2)
                with colpre:
                    pre_ms = st.slider('Pre window (ms)', min_value=10, max_value=120, value=50,
                                       help='How many ms before the detected peak to include (training used ~50 ms).')
                with colpost:
                    post_ms = st.slider('Post window (ms)', min_value=80, max_value=350, value=250,
                                        help='How many ms after the peak to include (training used ~250 ms).')

            def isolate_snippets(y, sr, frame_ms=12, peak_pct=94, min_peak_dist_ms=50, max_snippets=40, pre_ms=50, post_ms=250):
                frame_len = max(1, int(sr * frame_ms / 1000))
                hop = max(1, frame_len // 2)
                if len(y) < frame_len:
                    return [(0, len(y), y)]
                frames = librosa.util.frame(y, frame_length=frame_len, hop_length=hop)
                energy = (frames ** 2).mean(axis=0)
                energy = np.convolve(energy, np.ones(7) / 7, mode='same')
                from scipy.signal import find_peaks
                thresh = np.percentile(energy, peak_pct)
                min_dist = max(1, int((min_peak_dist_ms / 1000.0) * sr / hop))
                peaks, _ = find_peaks(energy, height=thresh, distance=min_dist)
                snippets = []
                for p in peaks[:max_snippets]:
                    center = int(p * hop)
                    start = max(0, center - int(pre_ms * sr / 1000))
                    end = min(len(y), center + int(post_ms * sr / 1000))
                    snippet = y[start:end]
                    if snippet.size > 0:
                        snippets.append((start, end, snippet))
                return snippets

            snippets = isolate_snippets(y, sr, frame_ms=frame_ms, peak_pct=peak_pct,
                                        min_peak_dist_ms=min_peak_dist_ms, pre_ms=pre_ms, post_ms=post_ms)
            if len(snippets) == 0:
                st.info('No clear keystroke peaks found — running whole-file inference (may be noisy).')
                probs, topk = infer(y, SAMPLE_RATE, use_tta=tta_enabled, n_tta=tta_passes)
                st.subheader(f"Prediction: {mapping[int(topk[0])]}  —  confidence {probs[topk[0]]:.3f}")
                st.write([ (mapping[int(k)], float(probs[k])) for k in topk[:5] ])
            else:
                snippet_results = []
                weighted_votes = {}
                for i, (s, e, snip) in enumerate(snippets):
                    probs, topk = infer(snip, sr, use_tta=tta_enabled, n_tta=tta_passes)
                    top1 = int(topk[0])
                    conf = float(probs[top1])
                    weighted_votes[top1] = weighted_votes.get(top1, 0.0) + conf
                    snippet_results.append({
                        "idx": i,
                        "start": s,
                        "end": e,
                        "snip": snip,
                        "probs": probs,
                        "topk": topk,
                        "top1": top1,
                        "conf": conf,
                        "sr": sr
                    })

                from collections import Counter
                top1_list = [res["top1"] for res in snippet_results]
                maj = Counter(top1_list).most_common(1)[0]
                weighted_best = max(weighted_votes.items(), key=lambda kv: kv[1])

                st.markdown(
                    f"""
                    <div class="summary-card">
                        <div class="summary-pill">Batch summary</div>
                        <h4 style="margin-top:0.6rem;margin-bottom:0.2rem;">Top majority vote: {mapping[maj[0]]}</h4>
                        <p style="margin:0;color:#475569;">Count: {maj[1]} | Confidence-weighted leader: {mapping[weighted_best[0]]} ({weighted_best[1]:.3f})</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.subheader(f'Found {len(snippet_results)} snippet(s); drill into each prediction')
                for res in snippet_results:
                    cols = st.columns([1, 2, 2])
                    with cols[0]:
                        st.audio(res["snip"], sample_rate=int(res["sr"]))
                    with cols[1]:
                        st.write(f"Snippet {res['idx']}: samples {res['start']}..{res['end']} — top: {mapping[res['top1']]} ({res['conf']:.3f})")
                        st.write([ (mapping[int(k)], float(res["probs"][int(k)])) for k in res['topk'][:5] ])
                    with cols[2]:
                        fig, ax = plt.subplots(1, 1, figsize=(4, 2))
                        D = np.abs(librosa.stft(res["snip"], n_fft=1024, hop_length=128))
                        librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), sr=res["sr"], hop_length=128, x_axis='time', y_axis='log', ax=ax)
                        ax.set_title('Spectrogram')
                        st.pyplot(fig)
                        plt.close(fig)

        except Exception as e:
            st.error("Failed to load audio: " + str(e))

# ---------------- streamlit-webrtc mic capture ----------------
# Microphone processor: collects chunks and buffers last buffer_seconds of audio
# Replace your MicrophoneProcessor with this version

class MicrophoneProcessor(AudioProcessorBase):
    def __init__(self, buffer_seconds: float = 0.4, frame_stride: int = 3, sr: int = SAMPLE_RATE):
        """
        buffer_seconds: how many seconds of audio to keep for classification
        frame_stride: only enqueue every `frame_stride`-th incoming frame to avoid overload
        """
        self._q = queue.Queue()
        self.buffer_seconds = buffer_seconds
        self.sr = sr
        self._parts = []       # local small buffer parts (list of numpy arrays)
        self._total_samples = 0
        self._frame_counter = 0
        self.frame_stride = max(1, int(frame_stride))

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        """
        Called very frequently by the webrtc component.
        We *do minimal work* here: convert the frame to a numpy array,
        but only push it into our internal queue every `frame_stride` frames.
        """
        # Convert frame to numpy array (channels, samples) or (samples,)
        arr = frame.to_ndarray().astype(np.float32)
        if arr.ndim == 2:
            arr = arr.mean(axis=0)
        # increment and maybe enqueue (throttle)
        self._frame_counter += 1
        if (self._frame_counter % self.frame_stride) == 0:
            try:
                self._q.put(arr, block=False)
            except queue.Full:
                # if the queue is full, just drop this part silently
                pass
        # Return frame unchanged (we're not modifying audio sent back to the client)
        return frame

    def get_buffered_audio(self):
        """
        Drain queued parts and return the last `buffer_seconds` seconds as 1D numpy array.
        """
        # collect everything currently in queue
        parts = []
        while not self._q.empty():
            try:
                p = self._q.get_nowait()
                parts.append(p)
            except queue.Empty:
                break
        if parts:
            # append to our rolling list and update total count
            for p in parts:
                self._parts.append(p)
                self._total_samples += p.shape[0]
            # trim if too long (keep last buffer_seconds)
            target = int(self.buffer_seconds * self.sr)
            # flatten tail until length <= target
            while self._total_samples > target and self._parts:
                # remove from leftmost parts
                first = self._parts.pop(0)
                self._total_samples -= first.shape[0]
        # concatenate to one array (or return None if empty)
        if self._parts:
            audio = np.concatenate(self._parts)
            # ensure exact length (pad if shorter)
            target = int(self.buffer_seconds * self.sr)
            if audio.shape[0] >= target:
                audio = audio[-target:]
            else:
                pad = np.zeros(target - audio.shape[0], dtype=np.float32)
                audio = np.concatenate([pad, audio])
            return audio
        else:
            return None

with mic_tab:
    st.subheader("Live microphone lab")
    st.caption("Grant browser access, monitor buffered frames, and classify the freshest 400 ms slice in real time.")
    if st.button("Request browser microphone access", key="mic-permission"):
        components.html(
            """
            <script>
            async function askMic(){
              try {
                const p = await navigator.mediaDevices.getUserMedia({ audio: true });
                alert("Microphone access granted. Return to the Streamlit tab to start capturing.");
                p.getTracks().forEach(t => t.stop());
              } catch(e) {
                alert("Microphone access failed or blocked: " + e.name + " - " + e.message);
              }
            }
            askMic();
            </script>
            """,
            height=100,
        )

    try:
        webrtc_ctx = webrtc_streamer(
            key="keystroke-mic",
            mode=WebRtcMode.SENDRECV,
            audio_processor_factory=MicrophoneProcessor,
            async_processing=True,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"audio": True, "video": False},
        )
    except Exception as e:
        try:
            from streamlit_webrtc.session_info import NoSessionError
            if isinstance(e, NoSessionError):
                st.warning('webrtc streamer not created (no Streamlit session). Mic capture disabled.')
            else:
                st.warning(f'webrtc streamer not created: {e}')
        except Exception:
            st.warning(f'webrtc streamer not created: {e}')
        webrtc_ctx = None

    if webrtc_ctx is None:
        st.warning("WebRTC context unavailable — launch via `streamlit run app.py` and open the browser tab.")
    else:
        st.write("**WebRTC state:**", webrtc_ctx.state)
        proc = getattr(webrtc_ctx, "audio_processor", None)
        if proc is None:
            st.info("Waiting for audio frames from the browser...")
        else:
            buf = proc.get_buffered_audio()
            if buf is None:
                st.info("Buffer empty — press keys near the mic to capture a snippet.")
                st.session_state["telemetry"] = {"latency_ms": None, "buffer_samples": None, "buffer_seconds": None}
                render_status()
            else:
                buffered_seconds = buf.shape[0] / SAMPLE_RATE
                st.markdown(f"**Buffered window:** {buf.shape[0]} samples (~{buffered_seconds:.2f} s)")
                st.session_state["telemetry"] = {
                    "latency_ms": buffered_seconds * 1000.0,
                    "buffer_samples": int(buf.shape[0]),
                    "buffer_seconds": float(buffered_seconds),
                }
                render_status()
                st.audio(buf, sample_rate=SAMPLE_RATE)

            if st.button("Classify last 400 ms", key="mic-classify"):
                audio = proc.get_buffered_audio()
                if audio is None:
                    st.warning("No audio captured yet — allow mic access and press keys.")
                else:
                    audio = audio / (np.max(np.abs(audio)) + 1e-9)
                    probs, topk = infer(audio, SAMPLE_RATE, use_tta=tta_enabled, n_tta=tta_passes)
                    st.subheader(f"Prediction: {mapping[int(topk[0])]}  —  confidence {probs[topk[0]]:.3f}")
                    st.write([ (mapping[int(k)], float(probs[k])) for k in topk[:5] ])
                    fig, ax = plt.subplots(2,1,figsize=(8,4))
                    librosa.display.waveshow(audio, sr=SAMPLE_RATE, ax=ax[0])
                    ax[0].set_title("Microphone waveform (last 400 ms)")
                    D = np.abs(librosa.stft(audio, n_fft=1024, hop_length=128))
                    librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), sr=SAMPLE_RATE, hop_length=128, x_axis='time', y_axis='log', ax=ax[1])
                    ax[1].set_title("Spectrogram")
                    st.pyplot(fig)
                    plt.close(fig)

# (File upload workflow now lives inside the Upload tab above)

st.markdown("---")
st.caption("Note: web-based mic capture uses your browser's microphone permission. Hugging Face Spaces is a convenient place to deploy this Streamlit app; Spaces may be CPU-only.")
