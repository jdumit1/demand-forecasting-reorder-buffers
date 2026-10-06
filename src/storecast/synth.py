"""Synthetic multi-store, multi-SKU weekly sales.

Nothing here comes from a real business. The generator imitates the shape of
apparel retail demand: a few fast movers and a long tail of slow, intermittent
SKUs; bigger and smaller stores; annual seasonality that differs by category;
holiday peaks; chain-wide promotions; and SKUs that launch mid-history.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

CATEGORIES = ("tops", "jeans", "dresses", "accessories", "kids")
# Week of year where each category peaks, and how strong the swing is.
CATEGORY_SEASON = {
    "tops": (24, 0.35),
    "jeans": (40, 0.25),
    "dresses": (20, 0.45),
    "accessories": (50, 0.40),
    "kids": (49, 0.30),
}
HOLIDAY_WEEKS = {47: 1.25, 48: 1.15, 50: 1.20, 51: 1.35}


@dataclass(frozen=True)
class SynthConfig:
    n_stores: int = 8
    n_skus: int = 60
    n_weeks: int = 156
    start: str = "2023-01-02"  # a Monday
    promo_rate: float = 0.06
    dispersion: float = 3.0  # negative binomial k; lower = noisier
    seed: int = 7


@dataclass(frozen=True)
class SynthData:
    sales: pd.DataFrame  # week_idx, week, store, sku, category, promo, units
    catalog: pd.DataFrame  # sku, category, base, unit_cost, unit_price, launch_idx
    stores: pd.DataFrame  # store, size


def _catalog(cfg: SynthConfig, rng: np.random.Generator) -> pd.DataFrame:
    skus = [f"SKU-{i:03d}" for i in range(1, cfg.n_skus + 1)]
    category = rng.choice(CATEGORIES, size=cfg.n_skus)
    # Weekly units in an average store: median ~1.5, a handful of fast movers.
    base = rng.lognormal(mean=0.4, sigma=0.9, size=cfg.n_skus)
    unit_cost = np.round(rng.uniform(4, 40, size=cfg.n_skus), 2)
    launch = np.where(
        rng.random(cfg.n_skus) < 0.2,
        rng.integers(10, max(11, cfg.n_weeks // 2), size=cfg.n_skus),
        0,
    )
    return pd.DataFrame(
        {
            "sku": skus,
            "category": category,
            "base": base,
            "trend": rng.normal(0.0, 0.15, size=cfg.n_skus),
            "unit_cost": unit_cost,
            "unit_price": np.round(unit_cost * 2.2, 2),
            "launch_idx": launch,
        }
    )


def _stores(cfg: SynthConfig, rng: np.random.Generator) -> pd.DataFrame:
    size = np.sort(rng.lognormal(0.0, 0.4, size=cfg.n_stores))[::-1]
    return pd.DataFrame({"store": [f"S{i:02d}" for i in range(1, cfg.n_stores + 1)], "size": size / size.mean()})


def generate(cfg: SynthConfig | None = None) -> SynthData:
    """Generate a reproducible weekly sales history."""
    cfg = cfg or SynthConfig()
    rng = np.random.default_rng(cfg.seed)
    catalog = _catalog(cfg, rng)
    stores = _stores(cfg, rng)

    weeks = pd.date_range(cfg.start, periods=cfg.n_weeks, freq="W-MON")
    t = np.arange(cfg.n_weeks)
    woy = weeks.isocalendar().week.to_numpy().astype(int)
    holiday = np.array([HOLIDAY_WEEKS.get(w, 1.0) for w in woy])

    # Chain-wide promotions: same weeks in every store, per SKU.
    promo = rng.random((cfg.n_skus, cfg.n_weeks)) < cfg.promo_rate
    lift = rng.uniform(1.5, 2.2, size=cfg.n_skus)

    rows = []
    for j, sku in catalog.iterrows():
        peak, amp = CATEGORY_SEASON[sku["category"]]
        season = 1.0 + amp * np.cos(2 * np.pi * (woy - peak) / 52.0)
        trend = np.exp(sku["trend"] * t / 52.0)
        live = (t >= sku["launch_idx"]).astype(float)
        # New SKUs ramp up over ~6 weeks instead of starting at full speed.
        ramp = np.clip((t - sku["launch_idx"] + 1) / 6.0, 0, 1) if sku["launch_idx"] else 1.0
        sku_mu = sku["base"] * season * trend * holiday * live * ramp
        sku_mu = sku_mu * np.where(promo[j], lift[j], 1.0)
        for _, st in stores.iterrows():
            mu = sku_mu * st["size"]
            lam = rng.gamma(cfg.dispersion, np.maximum(mu, 1e-9) / cfg.dispersion)
            units = rng.poisson(lam)
            rows.append(
                pd.DataFrame(
                    {
                        "week_idx": t,
                        "week": weeks,
                        "store": st["store"],
                        "sku": sku["sku"],
                        "category": sku["category"],
                        "promo": promo[j].astype(int),
                        "units": units.astype(int),
                    }
                )
            )
    sales = pd.concat(rows, ignore_index=True)
    sales = sales.sort_values(["store", "sku", "week_idx"], ignore_index=True)
    return SynthData(sales=sales, catalog=catalog, stores=stores)
