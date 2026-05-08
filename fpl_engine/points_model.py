"""Points Model — predicts E[points | starts] and E[points | sub] per player.

Architecture:
  - Per-position XGBoost regressors (GKP/DEF/MID/FWD)
  - Predicts points CONDITIONAL on playing (the minutes model handles P(play))
  - Two targets per position:
    * E[pts | 60+ min]  — full-game expectation
    * E[pts | 1-59 min] — sub expectation (lower, dominated by appearance points)

Combined xP = P(start) × E[pts|start] + P(sub) × E[pts|sub]

Key design decisions:
  - Excludes raw goals_scored from features (use xG only — avoids rewarding luck)
  - GKP model uses separate feature set (saves, clean sheets matter more)
  - Log-transform target for GKPs (high variance, skewed distribution)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from .config import MODELS_DIR, POSITIONS
from .features import get_points_feature_cols, prepare_model_input


@dataclass
class PointsModel:
    """Predicts expected FPL points conditional on playing.

    Trains separate models for starters (60+ min) and subs (1-59 min)
    per position.
    """

    starter_models: dict[str, xgb.XGBRegressor] = field(default_factory=dict)
    sub_models: dict[str, xgb.XGBRegressor] = field(default_factory=dict)
    feature_cols: list[str] = field(default_factory=get_points_feature_cols)
    is_trained: bool = False

    def train(
        self,
        df: pd.DataFrame,
        n_splits: int = 3,
        verbose: bool = True,
    ) -> dict[str, dict]:
        """Train per-position points regressors.

        Args:
            df: Feature matrix with position, minutes, total_points, element_id.
            n_splits: Cross-validation folds.
            verbose: Print metrics.

        Returns:
            Per-position evaluation metrics.
        """
        if not HAS_XGB:
            raise ImportError("xgboost required. pip install xgboost")

        metrics = {}

        for pos in POSITIONS:
            pos_df = df[df["position"] == pos].copy()

            # Split into starters and subs
            starters = pos_df[pos_df["minutes"] >= 60].copy()
            subs = pos_df[(pos_df["minutes"] > 0) & (pos_df["minutes"] < 60)].copy()

            if verbose:
                print(f"  [{pos}] Starters: {len(starters)} | Subs: {len(subs)}")

            pos_metrics = {}

            # ── Train starter model ──────────────────────────────────────
            if len(starters) >= 50:
                target = starters["total_points"].values.astype(float)

                # Log-transform for GKPs (high variance, right-skewed)
                use_log = pos == "GKP"
                if use_log:
                    # Shift to handle negative points (own goals, yellow cards)
                    shift = abs(target.min()) + 1
                    target = np.log1p(target + shift)

                X = prepare_model_input(starters, self.feature_cols)

                model = xgb.XGBRegressor(
                    objective="reg:squarederror",
                    n_estimators=250,
                    max_depth=5,
                    learning_rate=0.07,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=5,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=42,
                    verbosity=0,
                )

                groups = starters["element_id"].values
                gkf = GroupKFold(n_splits=min(n_splits, len(set(groups))))
                fold_maes, fold_rmses = [], []

                for train_idx, test_idx in gkf.split(X, target, groups):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = target[train_idx], target[test_idx]
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    if use_log:
                        y_pred_orig = np.expm1(y_pred) - shift
                        y_test_orig = np.expm1(y_test) - shift
                    else:
                        y_pred_orig, y_test_orig = y_pred, y_test

                    fold_maes.append(mean_absolute_error(y_test_orig, y_pred_orig))
                    fold_rmses.append(np.sqrt(mean_squared_error(y_test_orig, y_pred_orig)))

                # Final model on all data
                model.fit(X, target)
                self.starter_models[pos] = model

                pos_metrics["starter_mae"] = np.mean(fold_maes)
                pos_metrics["starter_rmse"] = np.mean(fold_rmses)

                if verbose:
                    print(f"         Starter MAE: {pos_metrics['starter_mae']:.3f} | "
                          f"RMSE: {pos_metrics['starter_rmse']:.3f}")

            # ── Train sub model ──────────────────────────────────────────
            if len(subs) >= 30:
                target_sub = subs["total_points"].values.astype(float)
                X_sub = prepare_model_input(subs, self.feature_cols)

                model_sub = xgb.XGBRegressor(
                    objective="reg:squarederror",
                    n_estimators=150,
                    max_depth=4,
                    learning_rate=0.08,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=5,
                    random_state=42,
                    verbosity=0,
                )

                groups_sub = subs["element_id"].values
                n_sub_splits = min(n_splits, len(set(groups_sub)))
                if n_sub_splits >= 2:
                    gkf_sub = GroupKFold(n_splits=n_sub_splits)
                    sub_maes = []
                    for train_idx, test_idx in gkf_sub.split(X_sub, target_sub, groups_sub):
                        model_sub.fit(X_sub.iloc[train_idx], target_sub[train_idx])
                        y_pred = model_sub.predict(X_sub.iloc[test_idx])
                        sub_maes.append(mean_absolute_error(target_sub[test_idx], y_pred))
                    pos_metrics["sub_mae"] = np.mean(sub_maes)
                    if verbose:
                        print(f"         Sub MAE: {pos_metrics['sub_mae']:.3f}")

                model_sub.fit(X_sub, target_sub)
                self.sub_models[pos] = model_sub

            metrics[pos] = pos_metrics

        self.is_trained = True
        return metrics

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict expected points conditional on playing.

        Returns:
            DataFrame with: element_id, e_pts_start, e_pts_sub, position
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call .train() first.")

        results = []
        for pos in POSITIONS:
            pos_df = df[df["position"] == pos].copy()
            if pos_df.empty:
                continue

            X = prepare_model_input(pos_df, self.feature_cols)

            row = {
                "element_id": pos_df["element_id"].values,
                "position": pos,
            }

            # Starter prediction
            if pos in self.starter_models:
                pred = self.starter_models[pos].predict(X)
                if pos == "GKP":
                    # Reverse log transform
                    starters_data = df[(df["position"] == pos) & (df["minutes"] >= 60)]
                    if not starters_data.empty:
                        shift = abs(starters_data["total_points"].min()) + 1
                    else:
                        shift = 3
                    pred = np.expm1(pred) - shift
                row["e_pts_start"] = pred.clip(min=0)
            else:
                row["e_pts_start"] = np.full(len(pos_df), 2.0)  # safe default

            # Sub prediction
            if pos in self.sub_models:
                row["e_pts_sub"] = self.sub_models[pos].predict(X).clip(min=0)
            else:
                row["e_pts_sub"] = np.full(len(pos_df), 1.0)  # 1 point for appearance

            results.append(pd.DataFrame(row))

        if not results:
            return pd.DataFrame()
        return pd.concat(results, ignore_index=True)

    def feature_importance(self, pos: str = "MID", model_type: str = "starter",
                           top_n: int = 15) -> pd.DataFrame:
        """Return top feature importances."""
        models = self.starter_models if model_type == "starter" else self.sub_models
        if pos not in models:
            raise ValueError(f"No {model_type} model for {pos}")

        model = models[pos]
        importance = model.feature_importances_
        feature_names = model.get_booster().feature_names or self.feature_cols

        return pd.DataFrame({
            "feature": feature_names[:len(importance)],
            "importance": importance,
        }).sort_values("importance", ascending=False).head(top_n)

    def save(self, path: Path | None = None) -> None:
        """Save all models to disk."""
        path = path or MODELS_DIR / "points"
        path.mkdir(parents=True, exist_ok=True)
        for pos, model in self.starter_models.items():
            model.save_model(str(path / f"starter_{pos}.json"))
        for pos, model in self.sub_models.items():
            model.save_model(str(path / f"sub_{pos}.json"))
        (path / "feature_cols.json").write_text(json.dumps(self.feature_cols))
        print(f"Points models saved to {path}")

    def load(self, path: Path | None = None) -> None:
        """Load models from disk."""
        if not HAS_XGB:
            raise ImportError("xgboost required")
        path = path or MODELS_DIR / "points"
        for pos in POSITIONS:
            sp = path / f"starter_{pos}.json"
            if sp.exists():
                m = xgb.XGBRegressor()
                m.load_model(str(sp))
                self.starter_models[pos] = m
            subp = path / f"sub_{pos}.json"
            if subp.exists():
                m = xgb.XGBRegressor()
                m.load_model(str(subp))
                self.sub_models[pos] = m

        fc_path = path / "feature_cols.json"
        if fc_path.exists():
            self.feature_cols = json.loads(fc_path.read_text())

        self.is_trained = bool(self.starter_models)
        print(f"Loaded points models: starters={list(self.starter_models.keys())}, "
              f"subs={list(self.sub_models.keys())}")
