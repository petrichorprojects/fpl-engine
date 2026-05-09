# FPL Engine — minimal build for Railway
FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    "uvicorn[standard]" fastapi httpx \
    pandas numpy scikit-learn pulp scipy \
    "xgboost>=2,<3"

COPY fpl_engine/ ./fpl_engine/
COPY api_server.py run_engine.py ./
COPY scripts/ ./scripts/
RUN mkdir -p /app/data/cache /app/models

# Railway REQUIRES listening on $PORT — no default fallback
CMD ["python", "-c", "import os; port = os.environ.get('PORT', '8000'); print(f'Starting on port {port}'); import uvicorn; uvicorn.run('api_server:app', host='0.0.0.0', port=int(port))"]
