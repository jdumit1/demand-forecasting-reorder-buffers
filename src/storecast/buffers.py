"""DDMRP-style buffer zones sized from a quantile forecast.

Classic DDMRP sizes the red safety zone with a fixed variability factor. Here
the forecast's own uncertainty does that job: red safety is the gap between
p90 and p50 demand over the lead time. A SKU the model is unsure about gets a
bigger safety zone; a steady one gets a small one.

All quantities are in units per week. Zones, bottom to top:

    red base   = average weekly usage x lead time x lead-time factor
    red safety = sum over the lead time of (p90 - p50)
    yellow     = average weekly usage x lead time
    green      = max(average weekly usage x order cycle, red base, minimum order)

    top of red    (TOR) = red base + red safety
    top of yellow (TOY) = TOR + yellow
    top of green  (TOG) = TOY + green

Planning rule: when the net flow position (on hand + on order - qualified
demand) is at or below TOY, order up to TOG.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BufferParams:
    lead_time_weeks: int = 2
    lead_time_factor: float = 0.5  # DDMRP's "medium" lead-time category
    order_cycle_weeks: float = 1.0
    min_order: float = 1.0


@dataclass(frozen=True)
class Zones:
    red_base: float
    red_safety: float
    yellow: float
    green: float

    @property
    def tor(self) -> float:
        return self.red_base + self.red_safety

    @property
    def toy(self) -> float:
        return self.tor + self.yellow

    @property
    def tog(self) -> float:
        return self.toy + self.green


def size_zones(p50, p90, params: BufferParams = BufferParams(), safety_factor: float = 1.0) -> Zones:
    """Size zones from forecasts for horizons 1..H (H >= lead time).

    ``safety_factor`` scales red safety; 1.0 means "protect up to p90".
    """
    p50, p90 = np.asarray(p50, float), np.asarray(p90, float)
    lt = params.lead_time_weeks
    if len(p50) < lt:
        raise ValueError("need a forecast at least as long as the lead time")
    awu = float(p50.mean())  # average weekly usage, forward-looking
    red_base = awu * lt * params.lead_time_factor
    red_safety = safety_factor * float(np.clip(p90[:lt] - p50[:lt], 0, None).sum())
    yellow = awu * lt
    green = max(awu * params.order_cycle_weeks, red_base, params.min_order)
    return Zones(red_base, red_safety, yellow, green)


def order_qty(net_flow: float, zones: Zones) -> int:
    """Units to order now, rounded up to whole units."""
    if net_flow > zones.toy:
        return 0
    return max(0, math.ceil(zones.tog - net_flow - 1e-9))


def status(on_hand: float, zones: Zones) -> str:
    """Execution status a planner would see for current stock."""
    if on_hand <= 0:
        return "stockout"
    if on_hand <= zones.tor:
        return "red"
    if on_hand <= zones.toy:
        return "yellow"
    if on_hand <= zones.tog:
        return "green"
    return "over"
