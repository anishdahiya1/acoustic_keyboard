# src/extract_keystrokes.py
import os
import torch
import librosa
import numpy as np
import pandas as pd

def isolator(signal, sample_rate, size, scan, before, after, threshold, show=False):
    strokes = []
    fft = librosa.stft(signal, n_fft=size, hop_length=scan)
    energy = np.abs(np.sum(fft, axis=0)).astype(float)
    threshed = energy > threshold
    peaks = np.where(threshed == True)[0]
    prev_end = int(-0.1 * sample_rate)
    for i in range(len(peaks)):
        this_peak = peaks[i]
        timestamp = (this_peak * scan) + size//2
        if timestamp > prev_end + int(0.1 * sample_rate):
            keystroke = signal[timestamp-before: timestamp+after]
            strokes.append(torch.tensor(keystroke)[None, :])
            prev_end = timestamp + after
    return strokes

def main():
    AUDIO_FOLDER = "data/Zoom/"   # adjust if needed
    keys_s = '1234567890qwertyuiopasdfghjklzxcvbnm'
    labels = list(keys_s)
    keys = [k + '.wav' for k in labels]

    data_dict = {'Key': [], 'File': []}

    for i, File in enumerate(keys):
        loc = os.path.join(AUDIO_FOLDER, File)
        if not os.path.exists(loc):
            print("Missing:", loc)
            continue
        samples, sample_rate = librosa.load(loc, sr=None)
        strokes = []
        prom = 0.06
        step = 0.005
        # try to find exactly 25 strokes by adjusting threshold
        while not len(strokes) == 25:
            strokes = isolator(samples[int(1*sample_rate):], sample_rate,
                                size=48, scan=24, before=2400, after=12000,
                                threshold=prom, show=False)
            if len(strokes) < 25:
                prom -= step
            if len(strokes) > 25:
                prom += step
            if prom <= 0:
                print('-- not possible for:', File)
                break
            step = step * 0.99

        label = [labels[i]] * len(strokes)
        data_dict['Key'] += label
        data_dict['File'] += strokes

    df = pd.DataFrame(data_dict)
    # map labels to integers
    mapper = {}
    counter = 0
    for l in df['Key']:
        if l not in mapper:
            mapper[l] = counter
            counter += 1
    df.replace({'Key': mapper}, inplace=True)

    # save dataframe (torch tensors can't be saved directly in csv) — save as a torch file + csv metadata
    os.makedirs("artifacts", exist_ok=True)
    # save waveform tensors as a single torch file
    torch.save(df['File'].tolist(), "artifacts/zoom_keystrokes_tensors.pt")
    # save metadata (keys) as csv
    df_meta = pd.DataFrame({'Key': df['Key'].tolist()})
    df_meta.to_csv("artifacts/zoom_keystrokes_meta.csv", index=False)

    print("Done. Saved artifacts/zoom_keystrokes_tensors.pt and artifacts/zoom_keystrokes_meta.csv")

if __name__ == "__main__":
    main()
