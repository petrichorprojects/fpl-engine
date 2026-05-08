#!/usr/bin/env python3
"""Run the FPL Analytics Engine end-to-end.

Usage:
    python run_engine.py                    # Full pipeline (fetch + train + optimize)
    python run_engine.py --skip-fetch       # Use cached data
    python run_engine.py --predict-only     # Use saved models, just predict + optimize
    python run_engine.py --gamestate chasing # Differential strategy
"""

import argparse
import sys

from fpl_engine.engine import FPLEngine
from fpl_engine.optimizer import GameState, Chip


def main():
    parser = argparse.ArgumentParser(description="FPL Analytics Engine")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Skip API fetch, use cached data")
    parser.add_argument("--predict-only", action="store_true",
                        help="Load saved models, skip training")
    parser.add_argument("--gamestate", default="neutral",
                        choices=["neutral", "leading", "chasing", "mini_league"],
                        help="Strategic posture for optimization")
    parser.add_argument("--budget", type=int, default=1000,
                        help="Budget in 0.1m units (default: 1000 = £100m)")
    parser.add_argument("--chip", default="none",
                        choices=["none", "wildcard", "free_hit", "bench_boost", "triple_captain"],
                        help="Active chip")
    parser.add_argument("--save-models", action="store_true",
                        help="Save trained models to disk")
    args = parser.parse_args()

    gamestate = GameState(args.gamestate)
    chip = Chip(args.chip)

    print()
    print("  ╔═══════════════════════════════════════════════════╗")
    print("  ║    FPL ANALYTICS ENGINE v0.1                     ║")
    print("  ║    Minutes Model + Points Model + Meta Optimizer  ║")
    print("  ╚═══════════════════════════════════════════════════╝")
    print()

    engine = FPLEngine()

    # ── Step 1: Data ─────────────────────────────────────────────
    if args.predict_only:
        engine.load_models()
        engine.fetch_data(fetch_histories=True, verbose=True)
    elif args.skip_fetch:
        engine.fetch_data(fetch_histories=False, verbose=True)
    else:
        engine.fetch_data(fetch_histories=True, verbose=True)

    # ── Step 2: Features ─────────────────────────────────────────
    engine.build_features(verbose=True)

    # ── Step 3: Train (unless predict-only) ──────────────────────
    if not args.predict_only:
        metrics = engine.train(verbose=True)
        if args.save_models:
            engine.save_models()

    # ── Step 4: Predict ──────────────────────────────────────────
    predictions = engine.predict(verbose=True)

    # ── Step 5: Optimize ─────────────────────────────────────────
    result = engine.optimize(
        budget=args.budget,
        gamestate=gamestate,
        chip=chip,
        verbose=True,
    )

    # ── Save predictions ─────────────────────────────────────────
    output_path = "data/predictions.csv"
    predictions.to_csv(output_path, index=False)
    print(f"\n  💾 Predictions saved to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
