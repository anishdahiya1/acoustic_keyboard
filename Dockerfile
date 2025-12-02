FROM python:3.11-slim

# Install system deps for audio processing and ffmpeg (librosa may need ffmpeg/av)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the repo (small) and artifacts needed for demo
COPY . /app

# Expose Streamlit port
EXPOSE 8501

ENV STREAMLIT_SERVER_HEADLESS=true
ENV PYTHONUNBUFFERED=1

COPY scripts/docker_entrypoint.py /app/scripts/docker_entrypoint.py

ENTRYPOINT ["python", "scripts/docker_entrypoint.py"]
CMD ["streamlit", "run", "scripts/streamlit_app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
