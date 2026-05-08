#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  FPL Engine — Full deploy to Railway (API) + Vercel (Web)
#
#  Prerequisites:
#    1. npm i -g railway    (Railway CLI)
#    2. npm i -g vercel     (Vercel CLI)
#    3. railway login       (authenticate)
#    4. vercel login        (authenticate)
#
#  First-time setup:
#    railway init           (creates project)
#    railway volume add     (persistent storage for models/data)
#
#  Usage:
#    bash scripts/deploy.sh           # deploy both
#    bash scripts/deploy.sh api       # deploy API only
#    bash scripts/deploy.sh web       # deploy web only
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$(dirname "$0")/.."

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

TARGET="${1:-both}"

echo -e "${BOLD}╔═══════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║         FPL Engine — Deploy                       ║${NC}"
echo -e "${BOLD}╚═══════════════════════════════════════════════════╝${NC}"
echo ""

# ── Deploy API to Railway ────────────────────────────────────────────────────
if [[ "$TARGET" == "both" || "$TARGET" == "api" ]]; then
    echo -e "${CYAN}▶ Deploying API to Railway...${NC}"

    if ! command -v railway &> /dev/null; then
        echo -e "${RED}✗ Railway CLI not found. Install: npm i -g @railway/cli${NC}"
        exit 1
    fi

    # Check railway is linked to a project
    if ! railway status &> /dev/null; then
        echo -e "${RED}✗ Railway not linked. Run: railway init${NC}"
        exit 1
    fi

    # Deploy
    railway up --detach

    # Get the public URL
    RAILWAY_URL=$(railway domain 2>/dev/null || echo "")
    if [[ -n "$RAILWAY_URL" ]]; then
        echo -e "${GREEN}✓ API deployed to: https://${RAILWAY_URL}${NC}"
    else
        echo -e "${GREEN}✓ API deployed! Run 'railway domain' to get the URL.${NC}"
        echo -e "  Then add a public domain: railway domain add"
    fi

    # Set up cron job for data refresh (Thursday + Friday at 10:00 UTC)
    echo ""
    echo -e "${CYAN}  ℹ  To set up automatic data refresh:${NC}"
    echo -e "     1. Go to your Railway project dashboard"
    echo -e "     2. Add a new service → Cron Job"
    echo -e "     3. Use the same Docker image"
    echo -e "     4. Command: python scripts/refresh_predictions.py"
    echo -e "     5. Schedule: 0 10 * * 4,5  (10:00 UTC Thu+Fri)"
    echo ""
fi

# ── Deploy Web to Vercel ─────────────────────────────────────────────────────
if [[ "$TARGET" == "both" || "$TARGET" == "web" ]]; then
    echo -e "${CYAN}▶ Deploying Web to Vercel...${NC}"

    if ! command -v vercel &> /dev/null; then
        echo -e "${RED}✗ Vercel CLI not found. Install: npm i -g vercel${NC}"
        exit 1
    fi

    cd web

    # Build
    echo "  Building Next.js..."
    npm run build 2>&1 | tail -5

    # Deploy
    vercel --prod

    cd ..

    echo ""
    echo -e "${GREEN}✓ Web deployed to Vercel${NC}"
    echo ""
    echo -e "${CYAN}  ⚠  Don't forget to set the env var in Vercel:${NC}"
    echo -e "     NEXT_PUBLIC_API_URL = https://<your-railway-domain>"
    echo -e ""
    echo -e "     Vercel dashboard → Project → Settings → Environment Variables"
fi

echo ""
echo -e "${BOLD}═══════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Deploy complete!${NC}"
echo ""
echo -e "  ${BOLD}Architecture:${NC}"
echo -e "    Vercel (free)   →  Next.js frontend"
echo -e "    Railway (\$5/mo) →  FastAPI + Python engine"
echo -e ""
echo -e "  ${BOLD}Data refresh:${NC}"
echo -e "    Railway cron    →  Thu+Fri 10:00 UTC"
echo -e "    Manual:         →  curl -X POST https://<railway>/optimize"
echo -e "${BOLD}═══════════════════════════════════════════════════${NC}"
