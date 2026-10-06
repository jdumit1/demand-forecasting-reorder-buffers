"""Replay the test weeks under three reorder policies and add up the costs.

Each week, at the end of week ``w``, the policy looks at stock and decides an
order that arrives at the start of week ``w + lead_time``. Then week ``w + 1``
happens: arrivals land, demand comes in, unmet demand is lost (stores don't
backorder), and whatever is left on the shelf costs money to hold.

Policies:
  minmax          POS-style min/max from the trailing 8-week average
  ddmrp_classic   DDMRP zones from the trailing average, fixed variability factor
  ddmrp_forecast  DDMRP zones sized from the quantile forecast (this repo)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .buffers import BufferParams, Zones, order_qty, size_zones

POLICIES = {
    "minmax": "Min/max on 8-week average",
    "ddmrp_classic": "DDMRP, history-sized",
    "ddmrp_forecast": "DDMRP, forecast-sized",
}


@dataclass(frozen=True)
class CostParams:
    holding_rate_year: float = 0.25  # carrying cost as a share of unit cost per year
    # Lost sale cost = unit margin (price - cost); taken from the catalog.


def classic_zones(awu: float, params: BufferParams, variability: float = 0.5) -> Zones:
    lt = params.lead_time_weeks
    red_base = awu * lt * params.lead_time_factor
    return Zones(
        red_base=red_base,
        red_safety=red_base * variability,
        yellow=awu * lt,
        green=max(awu * params.order_cycle_weeks, red_base, params.min_order),
    )


# Each policy has one safety knob. These defaults are the usual settings; the
# frontier() below sweeps them to compare policies at equal inventory.
DEFAULT_KNOBS = {"minmax": 1.0, "ddmrp_classic": 0.5, "ddmrp_forecast": 1.0}


@dataclass(frozen=True)
class Plan:
    """What a policy can see at the end of week ``origin``."""

    origin: int
    ma8: float
    p50: np.ndarray
    p90: np.ndarray


def plans_for(fc: pd.DataFrame) -> list[Plan]:
    """Turn one series' forecast rows into a list of weekly plans."""
    out = []
    for w, g in fc.sort_values(["week_idx", "h"]).groupby("week_idx", sort=True):
        out.append(Plan(int(w), float(g["ma8"].iloc[0]), g["p50"].to_numpy(), g["p90"].to_numpy()))
    return out


def _decide(policy: str, position: float, plan: Plan, params: BufferParams, knob: float) -> tuple[int, Zones | None]:
    if policy == "minmax":
        # knob = extra weeks of cover on top of the lead time
        lo = plan.ma8 * (params.lead_time_weeks + knob)
        hi = lo + 2 * plan.ma8
        return (max(0, math.ceil(hi - position)) if position <= lo else 0), None
    if policy == "ddmrp_classic":
        z = classic_zones(plan.ma8, params, variability=knob)  # knob = variability factor
    elif policy == "ddmrp_forecast":
        z = size_zones(plan.p50, plan.p90, params, safety_factor=knob)  # knob = safety multiplier
    else:
        raise ValueError(f"unknown policy {policy!r}")
    return order_qty(position, z), z


def simulate_series(
    demand: dict[int, float],
    plans: list[Plan],
    policy: str,
    params: BufferParams = BufferParams(),
    initial_on_hand: int = 0,
    knob: float | None = None,
) -> pd.DataFrame:
    """Simulate one store x SKU, one week per plan."""
    knob = DEFAULT_KNOBS[policy] if knob is None else knob
    on_hand, pipeline, trace = initial_on_hand, {}, []
    for plan in plans:
        w = plan.origin
        if w + 1 not in demand:
            break
        on_order = sum(q for t, q in pipeline.items() if t > w)
        position = on_hand + on_order
        q, z = _decide(policy, position, plan, params, knob)
        if q > 0:
            arrival = w + params.lead_time_weeks
            pipeline[arrival] = pipeline.get(arrival, 0) + q
        on_hand += pipeline.pop(w + 1, 0)
        d = demand[w + 1]
        sold = min(on_hand, d)
        on_hand -= sold
        trace.append(
            {
                "week_idx": w + 1,
                "demand": d,
                "sold": sold,
                "lost": d - sold,
                "on_hand": on_hand,
                "position": position,
                "ordered": q,
                "tor": z.tor if z else np.nan,
                "toy": z.toy if z else np.nan,
                "tog": z.tog if z else np.nan,
            }
        )
    return pd.DataFrame(trace)


def run(
    sales: pd.DataFrame,
    forecasts: pd.DataFrame,
    catalog: pd.DataFrame,
    params: BufferParams = BufferParams(),
    costs: CostParams = CostParams(),
    knobs: dict[str, float] | None = None,
    keep_traces: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Simulate every series under each policy in ``knobs`` (default: all three).

    Returns per-series totals and, if ``keep_traces``, the week-by-week traces.
    """
    knobs = knobs or DEFAULT_KNOBS
    demand = sales.set_index(["store", "sku", "week_idx"])["units"]
    cat = catalog.set_index("sku")
    totals, traces = [], []
    for (store, sku), fc in forecasts.groupby(["store", "sku"], sort=True):
        plans = plans_for(fc)
        d = demand.loc[(store, sku)].to_dict()
        init = math.ceil(plans[0].ma8 * (params.lead_time_weeks + 1.5))
        unit_cost = float(cat.loc[sku, "unit_cost"])
        margin = float(cat.loc[sku, "unit_price"]) - unit_cost
        hold = unit_cost * costs.holding_rate_year / 52.0
        for policy, knob in knobs.items():
            tr = simulate_series(d, plans, policy, params, init, knob)
            if keep_traces:
                traces.append(tr.assign(store=store, sku=sku, policy=policy))
            totals.append(
                {
                    "store": store,
                    "sku": sku,
                    "policy": policy,
                    "demand": tr["demand"].sum(),
                    "sold": tr["sold"].sum(),
                    "lost": tr["lost"].sum(),
                    "orders": int((tr["ordered"] > 0).sum()),
                    "holding_cost": tr["on_hand"].sum() * hold,
                    "lost_margin": tr["lost"].sum() * margin,
                    "avg_inventory_value": tr["on_hand"].mean() * unit_cost,
                }
            )
    return pd.DataFrame(totals), (pd.concat(traces, ignore_index=True) if keep_traces else None)


FRONTIER_GRID = {
    "minmax": (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0),
    "ddmrp_classic": (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
    "ddmrp_forecast": (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0),
}


def frontier(
    sales: pd.DataFrame,
    forecasts: pd.DataFrame,
    catalog: pd.DataFrame,
    grid: dict[str, tuple[float, ...]] = FRONTIER_GRID,
    **kw,
) -> pd.DataFrame:
    """Sweep each policy's safety knob: one row per (policy, knob) with totals.

    A policy that only looks good because it holds more stock will not look
    good here; this compares policies at the same inventory.
    """
    rows = []
    for policy, ks in grid.items():
        for k in ks:
            tot, _ = run(sales, forecasts, catalog, knobs={policy: k}, keep_traces=False, **kw)
            a = tot.sum(numeric_only=True)
            rows.append(
                {
                    "policy": policy,
                    "knob": k,
                    "fill_rate": a["sold"] / a["demand"],
                    "lost_share": a["lost"] / a["demand"],
                    "avg_inventory_value": a["avg_inventory_value"],
                    "holding_cost": a["holding_cost"],
                    "lost_margin": a["lost_margin"],
                    "total_cost": a["holding_cost"] + a["lost_margin"],
                }
            )
    return pd.DataFrame(rows)


def lost_share_at(front: pd.DataFrame, policy: str, inventory_value: float) -> float:
    """Lost-sales share a policy reaches at a given inventory value (linear interpolation)."""
    f = front[front["policy"] == policy].sort_values("avg_inventory_value")
    return float(np.interp(inventory_value, f["avg_inventory_value"], f["lost_share"]))


def summarize(totals: pd.DataFrame) -> dict:
    agg = totals.groupby("policy").agg(
        demand=("demand", "sum"),
        sold=("sold", "sum"),
        lost=("lost", "sum"),
        orders=("orders", "sum"),
        holding_cost=("holding_cost", "sum"),
        lost_margin=("lost_margin", "sum"),
        inventory_value=("avg_inventory_value", "sum"),
    )
    agg["fill_rate"] = agg["sold"] / agg["demand"]
    agg["total_cost"] = agg["holding_cost"] + agg["lost_margin"]
    base = agg.loc["minmax"]
    out = {}
    for policy, r in agg.iterrows():
        out[policy] = {
            "label": POLICIES[policy],
            "fill_rate": float(r["fill_rate"]),
            "lost_units": int(r["lost"]),
            "orders": int(r["orders"]),
            "holding_cost": float(r["holding_cost"]),
            "lost_margin": float(r["lost_margin"]),
            "total_cost": float(r["total_cost"]),
            "avg_inventory_value": float(r["inventory_value"]),
            "total_cost_vs_minmax": float(r["total_cost"] / base["total_cost"] - 1),
            "inventory_vs_minmax": float(r["inventory_value"] / base["inventory_value"] - 1),
        }
    return {p: out[p] for p in POLICIES}
