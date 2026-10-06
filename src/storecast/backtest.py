"""Rolling-origin backtest and the metrics used to judge it.

At each origin the model only sees rows whose target week has already
happened, so nothing from the future leaks into training.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd

from .model import QuantileForecaster, moving_average, seasonal_naive

ID_COLS = ["store", "sku", "week_idx", "h", "target_idx"]


def rolling_forecasts(
    rows: pd.DataFrame,
    origins: Iterable[int],
    retrain_every: int = 4,
    make_model: Callable[[], QuantileForecaster] = QuantileForecaster,
    log: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Forecast every series at every origin; retrain every ``retrain_every`` origins."""
    origins = sorted(origins)
    out, model = [], None
    for i, origin in enumerate(origins):
        if model is None or i % retrain_every == 0:
            train = rows[rows["target_idx"] <= origin]
            model = make_model().fit(train)
            if log:
                log(f"  trained at origin {origin} on {len(train):,} rows")
        test = rows[rows["week_idx"] == origin]
        fc = test[ID_COLS].copy()
        fc["actual"] = test["target"]
        fc = fc.join(model.predict(test))
        fc["snaive"] = seasonal_naive(test)
        fc["ma8"] = moving_average(test, 8)
        out.append(fc)
    fc = pd.concat(out, ignore_index=True)
    for c in ("store", "sku"):
        fc[c] = fc[c].astype(str)
    return fc


def wape(actual, pred) -> float:
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    return float(np.abs(actual - pred).sum() / actual.sum())


def bias(actual, pred) -> float:
    """Positive = over-forecast, as a share of actual volume."""
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    return float((pred - actual).sum() / actual.sum())


def pinball(actual, pred, q: float) -> float:
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    diff = actual - pred
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def coverage(actual, lo, hi) -> float:
    actual = np.asarray(actual, float)
    return float(np.mean((actual >= np.asarray(lo)) & (actual <= np.asarray(hi))))


METHODS = {"lgbm_p50": "LightGBM (p50)", "snaive": "Seasonal naive", "ma8": "8-week moving average"}


def summarize(fc: pd.DataFrame) -> dict:
    """Headline metrics per method, plus WAPE by horizon."""
    fc = fc[fc["actual"].notna()]
    preds = {"lgbm_p50": fc["p50"], "snaive": fc["snaive"], "ma8": fc["ma8"]}
    methods = {
        k: {"label": METHODS[k], "wape": wape(fc["actual"], p), "bias": bias(fc["actual"], p)} for k, p in preds.items()
    }
    by_h = {
        int(h): {k: wape(g["actual"], g[col]) for k, col in (("lgbm_p50", "p50"), ("snaive", "snaive"), ("ma8", "ma8"))}
        for h, g in fc.groupby("h")
    }
    best_base = min(methods["snaive"]["wape"], methods["ma8"]["wape"])
    return {
        "rows": int(len(fc)),
        "origins": int(fc["week_idx"].nunique()),
        "methods": methods,
        "wape_by_horizon": by_h,
        "wape_vs_best_baseline": methods["lgbm_p50"]["wape"] / best_base - 1,
        "interval_coverage_p10_p90": coverage(fc["actual"], fc["p10"], fc["p90"]),
        "pinball": {f"p{round(q * 100)}": pinball(fc["actual"], fc[f"p{round(q * 100)}"], q) for q in (0.1, 0.5, 0.9)},
    }
