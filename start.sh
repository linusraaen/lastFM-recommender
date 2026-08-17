#!/bin/sh
# Container entrypoint for hosting FastAPI + Streamlit together in one
# process (deployed on Render, see render.yaml). The externally-exposed
# process is Streamlit; it binds to $PORT (Render injects this), falling
# back to 7860 for local runs where nothing sets it.
set -e

if [ -n "$HF_MODEL_REPO" ]; then
  echo "Downloading model artifacts from $HF_MODEL_REPO..."
  python -m src.serve.download_artifacts
elif [ -f "models/user_tower.pt" ]; then
  echo "HF_MODEL_REPO is not set -- using local data/ and models/ artifacts."
else
  echo "ERROR: HF_MODEL_REPO is not set, and no local artifacts were found at models/user_tower.pt." >&2
  echo "Set HF_MODEL_REPO in this environment (Render's dashboard -> Environment tab if it wasn't" >&2
  echo "deployed via the render.yaml Blueprint), or mount data/ and models/ locally." >&2
  exit 1
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
