# Streamlit demo guide

This guide explains how to run the Streamlit demo locally, deploy it, and recommended copy and assets for a LinkedIn/GitHub showcase.

## Run locally

1. Create and activate your virtualenv (optional but recommended):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Start the demo:

```powershell
streamlit run scripts/streamlit_app.py
```

Open the URL Streamlit prints (usually http://localhost:8501).

## What the demo does
- Upload a WAV or pick a sample from `data/Zoom/`.
- The app isolates short snippets using short-time energy peaks.
- Each snippet is run through the `cnn_finetune_merged_hard.pt` model and top-5 predictions are shown.

## Deployment options
- Streamlit Community Cloud: free for public repos. Push this repo to GitHub, add a `packages.txt` or `requirements.txt` (we already added `requirements.txt`), and connect the repo in Streamlit Cloud.
- Heroku / Render / Railway: create a small web service; install dependencies and run `streamlit run` as the web command.
- Docker: create a minimal Dockerfile that installs Python, required libs, copies the repo, and runs the app.

## Suggested GitHub README section
Include a short demo GIF (3–8s) showing: upload WAV → snippets and predictions show up → final transcript. Provide a one-liner install and `streamlit run` commands.

## Suggested LinkedIn post (short)
Title: "Real-time keystroke detection from audio — demo & model"

Body:
I trained a lightweight CNN to classify isolated keystroke audio snippets (0-9, a-z). It handles noisy Zoom/phone recordings and isolates keystroke onsets automatically. Try the live demo (Streamlit): <link-to-demo>

Highlights:
- Lightweight PyTorch model (checkpoint in `artifacts/`)
- Isolation + inference pipeline that extracts snippets and runs per-snippet classification
- Exportable report with spectrograms and per-snippet predictions

If you’re curious, the repo has the full training notebook, model checkpoints, and scripts to reproduce results: https://github.com/<your-account>/<repo>

## Suggested longer LinkedIn thread (3 posts)
1. Short description + demo GIF + link to Streamlit demo
2. Quick explanation of the isolation method and why short-time energy worked well; include an example spectrogram and snippet audio.
3. Notes on failure modes and what improved results (fine-tuning on matched-device audio, augmentations)

## Tips for the demo GIF
- Keep it short (5–8s). Show upload, a few snippets highlighting model confidences, and the generated transcript.
- Export as MP4 or GIF under 10MB for LinkedIn variations.

## A/B testing ideas for the demo
- Add a 'Model variant' dropdown (baseline vs fine-tuned) so viewers can see differences.
- Add isolation presets so users can compare precision/recall trade-offs.

---
If you want, I can:
- Create a small Dockerfile and GitHub Action to auto-deploy to Streamlit Cloud or Render.
- Generate a short demo GIF using the outputs we already created (I can script a headless capture of the Streamlit UI). 
