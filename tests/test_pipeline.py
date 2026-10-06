import numpy as np
import pandas as pd
import pytest

from storecast import simulate
from storecast.backtest import rolling_forecasts, summarize
from storecast.buffers import BufferParams
from storecast.features import FEATURES, build_features
from storecast.model import QuantileForecaster
from storecast.synth import SynthConfig, generate

SMALL = SynthConfig(n_stores=2, n_skus=6, n_weeks=90, seed=3)


@pytest.fixture(scope="module")
def data():
    return generate(SMALL)


@pytest.fixture(scope="module")
def rows(data):
    return build_features(data.sales)


@pytest.fixture(scope="module")
def forecasts(rows):
    def small_model():
        m = QuantileForecaster()
        m.params["n_estimators"] = 30
        return m

    return rolling_forecasts(rows, origins=range(80, 86), retrain_every=3, make_model=small_model)


def test_synth_is_reproducible_and_well_formed(data):
    again = generate(SMALL)
    pd.testing.assert_frame_equal(data.sales, again.sales)
    s = data.sales
    assert len(s) == SMALL.n_stores * SMALL.n_skus * SMALL.n_weeks
    assert (s["units"] >= 0).all()
    assert s["units"].dtype.kind == "i"
    assert not s.duplicated(["store", "sku", "week_idx"]).any()


def test_no_sales_before_launch(data):
    s = data.sales.merge(data.catalog[["sku", "launch_idx"]], on="sku")
    assert (s.loc[s["week_idx"] < s["launch_idx"], "units"] == 0).all()


def test_promotions_lift_sales():
    big = generate(SynthConfig(n_stores=4, n_skus=30, n_weeks=104, seed=11)).sales
    by_promo = big.groupby("promo")["units"].mean()
    assert by_promo[1] > 1.3 * by_promo[0]


def test_features_use_only_the_past(data, rows):
    # lag_0 at origin t must equal sales in week t, scaled; target must be week t + h.
    s = data.sales.set_index(["store", "sku", "week_idx"])["units"]
    sample = rows.sample(200, random_state=0)
    for _, r in sample.iterrows():
        key = (str(r["store"]), str(r["sku"]))
        assert r["lag_0"] * r["scale"] == pytest.approx(s[(*key, r["week_idx"])])
        if not np.isnan(r["target"]):
            assert r["target"] == s[(*key, r["week_idx"] + r["h"])]
    assert set(FEATURES) <= set(rows.columns)


def test_training_never_sees_the_target_week(rows):
    seen = []

    class Spy(QuantileForecaster):
        def fit(self, train):
            seen.append(int(train["target_idx"].max()))
            self.params["n_estimators"] = 5
            return super().fit(train)

    rolling_forecasts(rows, origins=[70, 71, 72], retrain_every=1, make_model=Spy)
    assert seen == [70, 71, 72]


def test_quantiles_are_ordered_and_non_negative(forecasts):
    assert (forecasts["p10"] <= forecasts["p50"]).all()
    assert (forecasts["p50"] <= forecasts["p90"]).all()
    assert (forecasts["p10"] >= 0).all()


def test_summary_has_every_method(forecasts):
    out = summarize(forecasts)
    assert set(out["methods"]) == {"lgbm_p50", "snaive", "ma8"}
    assert all(0 <= m["wape"] < 5 for m in out["methods"].values())
    assert 0 <= out["interval_coverage_p10_p90"] <= 1


def test_simulation_conserves_units(data, forecasts):
    totals, traces = simulate.run(data.sales, forecasts, data.catalog)
    assert set(totals["policy"]) == set(simulate.POLICIES)
    assert (traces["sold"] + traces["lost"] == traces["demand"]).all()
    assert (traces["on_hand"] >= 0).all()
    assert (traces["sold"] >= 0).all()


def test_perfect_foresight_buffer_never_stocks_out():
    # Flat demand of 5/week, forecast exactly right, enough starting stock.
    demand = {w: 5 for w in range(0, 40)}
    plans = [simulate.Plan(w, 5.0, np.full(4, 5.0), np.full(4, 5.0)) for w in range(0, 30)]
    tr = simulate.simulate_series(demand, plans, "ddmrp_forecast", BufferParams(lead_time_weeks=2), initial_on_hand=20)
    assert tr["lost"].sum() == 0


def test_frontier_is_monotone_in_inventory(data, forecasts):
    grid = {"ddmrp_forecast": (0.0, 1.0, 2.0)}
    front = simulate.frontier(data.sales, forecasts, data.catalog, grid=grid)
    inv = front.sort_values("knob")["avg_inventory_value"].to_numpy()
    assert (np.diff(inv) >= 0).all()
