# syntax=docker/dockerfile:1

FROM node:22-alpine AS frontend-build
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/index.html frontend/vite.config.js ./
COPY frontend/src ./src
RUN npm run build

FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/huggingface
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
# The default PyPI wheel pulls CUDA libraries even though retrieval runs on CPU.
RUN python -m pip install --no-cache-dir 'torch==2.15.0+cpu' \
    --index-url https://download.pytorch.org/whl/cpu
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY document_processing/harrier/retrieve_harrier.py document_processing/harrier/
COPY document_processing/harrier/harrier_embedding_manifest.json document_processing/harrier/

# Download the exact model revision recorded in the saved index while building.
# A missing or incompatible model fails the build instead of failing at startup.
RUN python -c 'import json; from pathlib import Path; from document_processing.harrier.retrieve_harrier import load_model; manifest = json.loads(Path("document_processing/harrier/harrier_embedding_manifest.json").read_text()); load_model(manifest, manifest["embedding_dim"])'

ENV HF_HUB_OFFLINE=1
COPY api.py rag_answer.py ./
COPY document_processing/harrier/harrier_chunks.jsonl document_processing/harrier/
COPY document_processing/harrier/harrier_embeddings.npy document_processing/harrier/
COPY --from=frontend-build /build/frontend/dist /app/frontend/dist

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
