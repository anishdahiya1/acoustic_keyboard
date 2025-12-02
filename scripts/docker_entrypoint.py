import os
import sys
import urllib.request
import subprocess

def download_if_missing(url, dest):
    if os.path.exists(dest):
        print('Model already present at', dest)
        return True
    print('Downloading model from', url)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest)
        print('Downloaded model to', dest)
        return True
    except Exception as e:
        print('Failed to download model:', e)
        return False


def main():
    model_url = os.environ.get('MODEL_URL')
    model_path = os.environ.get('MODEL_PATH', 'artifacts/cnn_finetune_merged_hard.pt')
    if model_url and not os.path.exists(model_path):
        ok = download_if_missing(model_url, model_path)
        if not ok:
            print('Continuing without model; demo may fail until model is available.')

    # Execute passed command
    if len(sys.argv) > 1:
        cmd = sys.argv[1:]
    else:
        # default command comes from Dockerfile CMD
        cmd = [
            'streamlit', 'run', 'scripts/streamlit_app.py', '--server.port', '8501', '--server.address', '0.0.0.0'
        ]
    print('Running command:', ' '.join(cmd))
    subprocess.run(cmd)


if __name__ == '__main__':
    main()
