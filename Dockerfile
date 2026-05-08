# ── FPL Engine API — production Docker image for Railway ─────────────────────
# Two-stage build: install deps first (cached), then copy app code.
#
#   docker build -t fpl-engine .
#   docker run -p 8000:8000 fpl-engine

FROM python:3.11-slim AS base

# System deps for XGBoost (libomp) and general build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Stage 1: Install Python deps (layer cached unless requirements change) ───
FROM base AS deps

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir \
    uvicorn[standard] \
    fastapi \
    httpx \
    pandas \
    numpy \
    scikit-learn \
    xgboost \
    pulp \
    scipy

# ── Stage 2: App code ───────────────────────────────────────────────────────
FROM deps AS app

# Copy engine code
COPY fpl_engine/ ./fpl_engine/
COPY api_server.py ./
COPY run_engine.py ./
COPY scripts/ ./scripts/

# Create data/models dirs
RUN mkdir -p /app/data/cache /app/models

ENV PORT=8000

CMD ["sh", "-c", "echo 'Starting FPL Engine on port $PORT...' && python -c 'from api_server import app; print(\"Import OK\")' && exec uvicorn api_server:app --host 0.0.0.0 --port $PORT"]
