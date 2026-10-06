"""LightGBM quantile forecaster plus the two baselines it has to beat."""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import FEATURES

QUANTILES = (0.1, 0.5, 0.9)


@dataclass
class QuantileForecaster:
    """One LightGBM model per quantile, trained on scaled targets."""

    quantiles: tuple[float, ...] = QUANTILES
    params: dict = field(
        default_factory=lambda: {
            "n_estimators": 250,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 40,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "verbose": -1,
        }
    )
    seed: int = 0

    def fit(self, rows: pd.DataFrame) -> QuantileForecaster:
        rows = rows[rows["target_scaled"].notna()]
        self.models_ = {}
        for q in self.quantiles:
            m = lgb.LGBMRegressor(objective="quantile", alpha=q, random_state=self.seed, **self.params)
            m.fit(rows[FEATURES], rows["target_scaled"])
            self.models_[q] = m
        return self

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        preds = np.column_stack([self.models_[q].predict(rows[FEATURES]) for q in self.quantiles])
        # Quantile models are fit separately, so enforce p10 <= p50 <= p90.
        preds = np.sort(preds, axis=1) * rows["scale"].to_numpy()[:, None]
        preds = np.clip(preds, 0, None)
        names = [f"p{round(q * 100)}" for q in self.quantiles]
        return pd.DataFrame(preds, columns=names, index=rows.index)


def seasonal_naive(rows: pd.DataFrame) -> pd.Series:
    """Same week last year; falls back to the 4-week mean for new SKUs."""
    last_year = rows["season_lag"].fillna(rows["mean_4"])
    return (last_year * rows["scale"]).rename("snaive")


def moving_average(rows: pd.DataFrame, window: int = 8) -> pd.Series:
    """Trailing mean, the default reorder logic in most POS systems."""
    col = f"mean_{window}"
    return (rows[col].fillna(rows["mean_4"]) * rows["scale"]).rename(f"ma{window}")
