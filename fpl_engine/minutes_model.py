"""Minutes Model — predicts P(start), P(sub), P(benched) per player per fixture.

This is the high-alpha component. Most FPL bots skip this entirely and assume
all players play. Getting rotation prediction right is worth more than marginal
xG improvements.

Architecture:
  - Per-position XGBoost classifiers (GKP/DEF/MID/FWD)
  - Three-class output: START (60+ min), SUB (1-59 min), BENCH (0 min)
  - Features: rolling availability, fixture congestion, manager rotation patterns,
    opponent strength, player status, days since last match

The minutes prediction feeds directly into the points model:
  xP = P(start) × E[pts|start] + P(sub) × E[pts|sub] + P(bench) × 0
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.model_selection import GroupKFold

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from .config import MODELS_DIR, POSITIONS
from .features import get_minutes_feature_cols, prepare_model_input


# ── Label encoding ───────────────────────────────────────────────────────────
# 0 = BENCH (0 min), 1 = SUB (1-59 min), 2 = START (60+ min)
LABEL_MAP = {"BENCH": 0, "SUB": 1, "START": 2}
LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}


def assign_minutes_label(minutes: pd.Series) -> pd.Series:
    """Convert raw minutes to categorical label."""
    return pd.cut(
        minutes,
        bins=[-1, 0, 59, 200],
        labels=[LABEL_MAP["BENCH"], LABEL_MAP["SUB"], LABEL_MAP["START"]],
    ).astype(int)


@dataclass
class MinutesModel:
    """Predicts playing time category per player per fixture.

    Trains one XGBoost classifier per position. Outputs calibrated
    probabilities for START/SUB/BENCH.
    """

    models: dict[str, xgb.XGBClassifier] = field(default_factory=dict)
    feature_cols: list[str] = field(default_factory=get_minutes_feature_cols)
    is_trained: bool = False

    def train(
        self,
        df: pd.DataFrame,
        n_splits: int = 3,
        verbose: bool = True,
    ) -> dict[str, dict]:
        """Train per-position minutes classifiers.

        Args:
            df: Feature matrix with 'position', 'minutes', 'element_id' columns.
            n_splits: Number of cross-validation folds (grouped by player).
            verbose: Print training metrics.

        Returns:
            Dictionary of per-position evaluation metrics.
        """
        if not HAS_XGB:
            raise ImportError("xgboost is required. Install with: pip install xgboost")

        df = df.copy()
        df["minutes_label"] = assign_minutes_label(df["minutes"])
        metrics = {}

        for pos in POSITIONS:
            pos_df = df[df["position"] == pos].copy()
            if len(pos_df) < 100:
                if verbose:
                    print(f"  [{pos}] Skipping — only {len(pos_df)} samples")
                continue

            X = prepare_model_input(pos_df, self.feature_cols)
            y = pos_df["minutes_label"].values
            groups = pos_df["element_id"].values

            # XGBoost with class-weight balancing
            # BENCH is most common → weight minorities up
            class_counts = np.bincount(y, minlength=3)
            total = class_counts.sum()
            sample_weights = np.array([total / (3 * max(class_counts[yi], 1)) for yi in y])

            model = xgb.XGBClassifier(
                objective="multi:softprob",
                num_class=3,
                n_estimators=200,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbosity=0,
            )

            # Group K-Fold: ensure same player not in train and test
            gkf = GroupKFold(n_splits=min(n_splits, len(set(groups))))
            fold_metrics = []

            for fold_i, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                w_train = sample_weights[train_idx]

                model.fit(X_train, y_train, sample_weight=w_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)

                fold_metrics.append({
                    "accuracy": accuracy_score(y_test, y_pred),
                    "log_loss": log_loss(y_test, y_prob, labels=[0, 1, 2]),
                })

            # Train final model on all data
            model.fit(X, y, sample_weight=sample_weights)
            self.models[pos] = model

            avg_acc = np.mean([m["accuracy"] for m in fold_metrics])
            avg_ll = np.mean([m["log_loss"] for m in fold_metrics])
            metrics[pos] = {"accuracy": avg_acc, "log_loss": avg_ll, "n_samples": len(pos_df)}

            if verbose:
                print(f"  [{pos}] Accuracy: {avg_acc:.3f} | Log Loss: {avg_ll:.3f} | N: {len(pos_df)}")

        self.is_trained = True
        return metrics

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict minutes probabilities for each player.

        Returns:
            DataFrame with columns: element_id, p_bench, p_sub, p_start, predicted_label
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call .train() first.")

        results = []
        for pos in POSITIONS:
            if pos not in self.models:
                continue

            pos_df = df[df["position"] == pos].copy()
            if pos_df.empty:
                continue

            X = prepare_model_input(pos_df, self.feature_cols)
            probas = self.models[pos].predict_proba(X)

            pos_results = pd.DataFrame({
                "element_id": pos_df["element_id"].values,
                "position": pos,
                "p_bench": probas[:, 0],
                "p_sub": probas[:, 1],
                "p_start": probas[:, 2],
            })
            pos_results["predicted_label"] = probas.argmax(axis=1)
            pos_results["predicted_label_name"] = pos_results["predicted_label"].map(LABEL_NAMES)

            # Expected minutes: weighted combination
            pos_results["e_minutes_pred"] = (
                pos_results["p_bench"] * 0
                + pos_results["p_sub"] * 25    # average sub gets ~25 min
                + pos_results["p_start"] * 82  # average starter gets ~82 min
            )

            results.append(pos_results)

        if not results:
            return pd.DataFrame()
        return pd.concat(results, ignore_index=True)

    def feature_importance(self, pos: str = "MID", top_n: int = 15) -> pd.DataFrame:
        """Return top feature importances for a position model."""
        if pos not in self.models:
            raise ValueError(f"No model for position {pos}")

        model = self.models[pos]
        importance = model.feature_importances_
        feature_names = model.get_booster().feature_names or self.feature_cols

        imp_df = pd.DataFrame({
            "feature": feature_names[:len(importance)],
            "importance": importance,
        }).sort_values("importance", ascending=False)

        return imp_df.head(top_n)

    def save(self, path: Path | None = None) -> None:
        """Save trained models to disk."""
        path = path or MODELS_DIR / "minutes"
        path.mkdir(parents=True, exist_ok=True)
        for pos, model in self.models.items():
            model.save_model(str(path / f"minutes_{pos}.json"))
        # Save feature cols
        (path / "feature_cols.json").write_text(json.dumps(self.feature_cols))
        print(f"Minutes models saved to {path}")

    def load(self, path: Path | None = None) -> None:
        """Load trained models from disk."""
        if not HAS_XGB:
            raise ImportError("xgboost required")
        path = path or MODELS_DIR / "minutes"
        for pos in POSITIONS:
            model_path = path / f"minutes_{pos}.json"
            if model_path.exists():
                model = xgb.XGBClassifier()
                model.load_model(str(model_path))
                self.models[pos] = model

        fc_path = path / "feature_cols.json"
        if fc_path.exists():
            self.feature_cols = json.loads(fc_path.read_text())

        self.is_trained = bool(self.models)
        print(f"Loaded minutes models for: {list(self.models.keys())}")
