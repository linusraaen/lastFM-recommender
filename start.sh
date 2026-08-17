#!/bin/sh
# Container entrypoint for hosting FastAPI + Streamlit together in one
# process, on any single-port Docker host (Hugging Face Spaces, Render,
# Fly.io, ...). The externally-exposed process is Streamlit; it binds to
# $PORT if the platform injects one (Render, Railway, ...), else 7860 (HF
# Spaces' fixed app_port, and the local/HF-Space default).
set -e

if [ -z "$HF_MODEL_REPO" ]; then
  echo "HF_MODEL_REPO is not set -- assuming data/ and models/ are already populated locally." >&2
else
  echo "Downloading model artifacts from $HF_MODEL_REPO..."
  python -m src.serve.download_artifacts
fi

uvicorn src.serve.api:app --host 0.0.0.0 --port 8000 &

echo "Waiting for the recommender API to finish loading..."
until curl -sf http://localhost:8000/health > /dev/null; do
  sleep 1
done
echo "API ready."

exec streamlit run app/streamlit_app.py \
  --server.port "${PORT:-7860}" \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false
