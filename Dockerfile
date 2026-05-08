# FPL Engine — slim CPU-only build for Railway
FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps — pin CPU-only xgboost to avoid 300MB CUDA pull
RUN pip install --no-cache-dir \
    "uvicorn[standard]" fastapi httpx \
    pandas numpy scikit-learn pulp scipy \
    "xgboost>=2,<3"

# Block CUDA packages from ever being pulled
ENV NVIDIA_VISIBLE_DEVICES=""

COPY fpl_engine/ ./fpl_engine/
COPY api_server.py run_engine.py ./
COPY scripts/ ./scripts/
RUN mkdir -p /app/data/cache /app/models

ENV PORT=8000
CMD ["sh", "-c", "exec uvicorn api_server:app --host 0.0.0.0 --port $PORT"]
