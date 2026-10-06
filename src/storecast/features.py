"""Features for a direct, multi-horizon global model.

One row = one series (store x SKU) at one forecast origin ``t`` for one horizon
``h``. Everything on the row is known at the end of week ``t``: recent sales,
rolling stats, last year's sales for the target week, and the promo calendar
(promotions are planned ahead, so the target week's promo flag is known).

Level features and the target are divided by a per-series scale, so one model
can learn from fast and slow movers together. Predictions are multiplied back.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

KEYS = ["store", "sku"]
LAGS = (0, 1, 2, 3)
WINDOWS = (4, 8, 13, 26)
LEVEL_COLS = [f"lag_{k}" for k in LAGS] + [f"mean_{w}" for w in WINDOWS] + ["std_8", "season_lag"]
CAT_COLS = ["store", "sku", "category"]
NUM_COLS = LEVEL_COLS + [
    "zero_share_13",
    "log_scale",
    "age",
    "h",
    "promo_now",
    "promo_target",
    "woy_sin",
    "woy_cos",
]
FEATURES = CAT_COLS + NUM_COLS


def _rolling(g: pd.core.groupby.SeriesGroupBy, w: int, fn: str) -> pd.Series:
    return g.rolling(w, min_periods=w).agg(fn).reset_index(level=[0, 1], drop=True)


def build_features(sales: pd.DataFrame, horizons: tuple[int, ...] = (1, 2, 3, 4)) -> pd.DataFrame:
    """Return one row per (series, origin, horizon) with features and target."""
    df = sales.sort_values(KEYS + ["week_idx"]).reset_index(drop=True)
    g = df.groupby(KEYS, sort=False)["units"]
    gp = df.groupby(KEYS, sort=False)["promo"]

    base = df[["week_idx", "week", "store", "sku", "category"]].copy()
    for k in LAGS:
        base[f"lag_{k}"] = g.shift(k)
    for w in WINDOWS:
        base[f"mean_{w}"] = _rolling(g, w, "mean")
    base["std_8"] = _rolling(g, 8, "std")
    base["zero_share_13"] = _rolling(df.assign(z=(df["units"] == 0).astype(float)).groupby(KEYS)["z"], 13, "mean")
    sold = df["units"].gt(0).groupby([df["store"], df["sku"]]).cummax()
    base["age"] = sold.astype(int).groupby([df["store"], df["sku"]]).cumsum()
    base["promo_now"] = df["promo"]

    scale = base["mean_26"].fillna(base["mean_13"]) + 0.5
    base["log_scale"] = np.log(scale)

    frames = []
    for h in horizons:
        f = base.copy()
        f["h"] = h
        f["target"] = g.shift(-h)
        f["season_lag"] = g.shift(52 - h)
        f["promo_target"] = gp.shift(-h)
        target_week = f["week"] + pd.to_timedelta(7 * h, unit="D")
        woy = target_week.dt.isocalendar().week.astype(float).to_numpy()
        f["woy_sin"] = np.sin(2 * np.pi * woy / 52.0)
        f["woy_cos"] = np.cos(2 * np.pi * woy / 52.0)
        f["target_idx"] = f["week_idx"] + h
        f["scale"] = scale
        frames.append(f)

    out = pd.concat(frames, ignore_index=True)
    for c in LEVEL_COLS:
        out[c] = out[c] / out["scale"]
    out["target_scaled"] = out["target"] / out["scale"]
    for c in CAT_COLS:
        out[c] = out[c].astype("category")
    # Need at least 13 weeks of history for the level features to mean anything.
    return out[out["mean_13"].notna()].reset_index(drop=True)
