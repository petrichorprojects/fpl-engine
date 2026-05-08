#!/usr/bin/env bash
# Deploy to Vercel (requires: npm i -g vercel)

set -e
cd "$(dirname "$0")/.."

echo "Building Next.js..."
(cd web && npm run build)

echo "Deploying to Vercel..."
vercel --prod

echo ""
echo "✓ Deployed! The API routes (/api/*) are served as Python serverless functions."
echo "  Make sure to set PYTHON_VERSION=3.11 in your Vercel project settings."
