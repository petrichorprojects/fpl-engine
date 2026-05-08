#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  Local dev — runs API + Web in parallel (no Docker needed)
#
#  Usage:
#    bash scripts/dev.sh
#
#  Requires: uv, node
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$(dirname "$0")/.."

echo "╔═══════════════════════════════════════════════════╗"
echo "║         FPL Engine — Local Dev                    ║"
echo "╚═══════════════════════════════════════════════════╝"
echo ""
echo "  API:  http://localhost:8000  (Swagger: /docs)"
echo "  Web:  http://localhost:3000"
echo ""

# Trap to kill both processes on Ctrl+C
trap 'kill 0' EXIT

# Start API
echo "▶ Starting Python API..."
NEXT_PUBLIC_API_URL=http://localhost:8000 \
uv run --with fastapi --with uvicorn \
  uvicorn api_server:app --reload --host 0.0.0.0 --port 8000 &

# Wait for API to be ready
echo "  Waiting for API..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "  ✓ API ready"
        break
    fi
    sleep 1
done

# Start Web
echo "▶ Starting Next.js..."
cd web
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev &
cd ..

echo ""
echo "✓ Both services running. Press Ctrl+C to stop."
wait
