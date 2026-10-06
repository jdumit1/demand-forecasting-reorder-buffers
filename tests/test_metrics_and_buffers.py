import math

import numpy as np
import pytest

from storecast.backtest import bias, coverage, pinball, wape
from storecast.buffers import BufferParams, Zones, order_qty, size_zones, status


def test_wape_and_bias_by_hand():
    actual = [10, 0, 5, 5]
    pred = [8, 1, 5, 7]
    assert wape(actual, pred) == pytest.approx((2 + 1 + 0 + 2) / 20)
    assert bias(actual, pred) == pytest.approx((21 - 20) / 20)


def test_pinball_penalises_the_right_side():
    # Under-forecasting at q=0.9 costs 9x more than over-forecasting by the same amount.
    under = pinball([10], [9], 0.9)
    over = pinball([10], [11], 0.9)
    assert under == pytest.approx(0.9)
    assert over == pytest.approx(0.1)


def test_coverage():
    assert coverage([1, 5, 9], [0, 0, 0], [4, 6, 8]) == pytest.approx(2 / 3)


def test_zones_from_forecast():
    params = BufferParams(lead_time_weeks=2, lead_time_factor=0.5, order_cycle_weeks=1, min_order=1)
    z = size_zones(p50=[10, 10, 10, 10], p90=[14, 16, 20, 20], params=params)
    assert z.red_base == pytest.approx(10)  # 10/wk x 2 wk x 0.5
    assert z.red_safety == pytest.approx(4 + 6)  # p90 - p50 over the 2-week lead time only
    assert z.yellow == pytest.approx(20)
    assert z.green == pytest.approx(10)  # max(10 x 1, red base 10, 1)
    assert (z.tor, z.toy, z.tog) == pytest.approx((20, 40, 50))


def test_more_uncertainty_means_more_safety_stock():
    steady = size_zones([10] * 4, [12] * 4)
    shaky = size_zones([10] * 4, [25] * 4)
    assert shaky.red_safety > steady.red_safety
    assert shaky.yellow == steady.yellow  # the expected-demand zones don't change


def test_safety_factor_scales_only_red_safety():
    a = size_zones([10] * 4, [15] * 4, safety_factor=1.0)
    b = size_zones([10] * 4, [15] * 4, safety_factor=0.5)
    assert b.red_safety == pytest.approx(a.red_safety / 2)
    assert (b.red_base, b.yellow, b.green) == (a.red_base, a.yellow, a.green)


def test_forecast_shorter_than_lead_time_is_rejected():
    with pytest.raises(ValueError):
        size_zones([1], [2], BufferParams(lead_time_weeks=2))


def test_order_rule():
    z = Zones(red_base=10, red_safety=10, yellow=20, green=10)  # TOR 20, TOY 40, TOG 50
    assert order_qty(41, z) == 0  # above yellow: wait
    assert order_qty(40, z) == 10  # at top of yellow: top up to green
    assert order_qty(12.5, z) == math.ceil(50 - 12.5)
    assert order_qty(-5, z) == 55  # backlog still tops up to green


@pytest.mark.parametrize(
    ("on_hand", "expected"),
    [(0, "stockout"), (5, "red"), (20, "red"), (30, "yellow"), (45, "green"), (60, "over")],
)
def test_status(on_hand, expected):
    z = Zones(red_base=10, red_safety=10, yellow=20, green=10)
    assert status(on_hand, z) == expected


def test_zero_forecast_gives_min_order_green():
    z = size_zones(np.zeros(4), np.zeros(4), BufferParams(min_order=3))
    assert z.green == 3
    assert z.tor == 0
