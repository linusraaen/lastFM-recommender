# Serves the FastAPI /recommend endpoint + Streamlit demo together in one
# container (start.sh), deployed as a Hugging Face Space (Docker SDK --
# see the YAML config block at the top of README.md).
# Model artifacts are NOT baked into the image -- start.sh downloads them at
# container startup from a Hugging Face Hub model repo (set HF_MODEL_REPO),
# via src/serve/download_artifacts.py. This keeps the image small and
# decoupled from any one trained model -- retrain and re-upload without
# rebuilding the image. For local testing without HF_MODEL_REPO, mount
# data/ and models/ instead (see `make docker-run`).
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY app/ app/
COPY start.sh start.sh
RUN chmod +x start.sh

ENV PYTHONUNBUFFERED=1
EXPOSE 8000 7860

CMD ["./start.sh"]
