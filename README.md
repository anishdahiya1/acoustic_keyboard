🎧 Acoustic Keyboard Sound Detection

Detect keyboard typing patterns using audio + Deep Learning (MFCC + 1D-CNN)
🔗 Live App: https://acoustickeyboard.streamlit.app/

🚀 Overview

This project predicts keyboard activity from short audio segments using a 1D Convolutional Neural Network (1D-CNN) trained on MFCC features.
It demonstrates how sound can be used as a biometric signal for identifying typing patterns — a unique intersection of audio signal processing, ML, and cybersecurity research.

🧠 Key Features

🎙 Audio Recording or Upload Support

🎼 MFCC Extraction using Librosa

🧩 1D-CNN Deep Learning Model

📊 Real-time Prediction via Streamlit

☁️ Deployed Online — no setup needed

🛠 Tech Stack
📌 Backend / ML

Python

NumPy, Pandas

Librosa → MFCC extraction

TensorFlow / Keras → 1D-CNN architecture

Scikit-learn → preprocessing & metrics

📌 Frontend / Deployment

Streamlit → Interactive UI

Hosted on Streamlit Cloud

🎯 Problem Statement

Can we identify keyboard typing patterns just from the sound it produces?
This project builds a deep learning pipeline that listens to audio and classifies whether the sound corresponds to keyboard keystrokes.

🔍 How It Works
1️⃣ Audio Processing

Raw audio is loaded

Converted to mono

MFCCs are computed (typically 40 coefficients)

Normalized for model input

2️⃣ Model Architecture (1D-CNN)

Conv1D → ReLU

MaxPooling

Dropout

Dense layers for classification

3️⃣ Streamlit Interface

Users can:

Upload audio files

Record mic input

View predictions instantly

📂 Project Structure
├── app.py                 # Streamlit UI
├── model.h5               # Trained 1D-CNN model
├── mfcc_extractor.py      # Audio feature extraction
├── utils.py               # Helper functions
├── requirements.txt
└── README.md

▶️ Run Locally
pip install -r requirements.txt
streamlit run app.py

📊 Results

The 1D-CNN model showed strong performance in identifying keyboard click sounds from background noise, validating the power of MFCCs + CNNs in acoustic classification tasks.

🌐 Live Demo

Try the deployed app instantly:
👉 https://acoustickeyboard.streamlit.app/

💡 Future Improvements

Multi-class prediction for different keyboard types

Detect keystrokes vs. non-keystrokes in continuous audio

Add spectrogram-based CNN model

Mobile app version

🧑‍💻 Author

Anish Dahiya
Data Scientist | AI/ML Engineer
🔗 GitHub: https://github.com/anishdahiya1

🔗 LinkedIn: https://www.linkedin.com/in/anishdahiya7
