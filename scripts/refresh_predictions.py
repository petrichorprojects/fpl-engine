#!/usr/bin/env python3
"""Cron job: refresh FPL data and predictions.

Designed to run on Railway's cron scheduler:
  Schedule: 0 10 * * 4,5   (10:00 UTC Thursday + Friday, before GW deadline)

What it does:
  1. Clears API cache (gets fresh FPL data)
  2. Fetches all player histories (~5 min)
  3. Builds features
  4. Trains models on latest data
  5. Generates predictions
  6. Saves models to disk (persisted via Railway volume)

The API server (api_server.py) picks up the new models/predictions on next request.
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fpl_engine.engine import FPLEngine


def main():
    start = time.time()
    print("=" * 60)
    print("  FPL ENGINE — Scheduled Data Refresh")
    print("=" * 60)

    engine = FPLEngine()

    # Clear cache to get fresh data
    engine.client.clear_cache()

    # Full pipeline
    engine.fetch_data(fetch_histories=True, verbose=True)
    engine.build_features(verbose=True)
    metrics = engine.train(verbose=True)
    predictions = engine.predict(verbose=True)

    # Save everything
    engine.save_models()
    predictions.to_csv("data/predictions.csv", index=False)

    elapsed = time.time() - start
    n_players = len(predictions)
    print(f"\n{'=' * 60}")
    print(f"  ✓ Refresh complete in {elapsed:.0f}s")
    print(f"  ✓ {n_players} player predictions saved")
    print(f"  ✓ Models saved to models/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
