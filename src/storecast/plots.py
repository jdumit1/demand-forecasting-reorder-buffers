"""Charts for the README. Static PNGs, light surface, one y-axis each."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, PercentFormatter  # noqa: E402

# Validated reference palette (first three categorical slots pass all-pairs CVD checks).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GOOD, WARNING, CRITICAL = "#0ca30c", "#fab219", "#d03b3b"
POLICY_COLOR = {"ddmrp_forecast": BLUE, "ddmrp_classic": ORANGE, "minmax": AQUA}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "ytick.left": False,
        "legend.frameon": False,
        "legend.labelcolor": INK_2,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 160,
        "axes.axisbelow": True,
    }
)


def _title(fig, title: str, subtitle: str) -> None:
    fig.text(0.012, 0.965, title, fontsize=13, fontweight="bold", color=INK, va="top")
    fig.text(0.012, 0.905, subtitle, fontsize=9.5, color=INK_2, va="top")


def _save(fig, path: Path) -> Path:
    fig.savefig(path, bbox_inches="tight", pad_inches=0.18, metadata={"Software": None})
    plt.close(fig)
    return path


def pick_series(forecasts: pd.DataFrame, target_mean: float = 8.0) -> tuple[str, str]:
    """A mid-volume series from the biggest store, so the charts are readable."""
    h1 = forecasts[(forecasts["h"] == 1) & forecasts["actual"].notna()]
    means = h1.groupby(["store", "sku"])["actual"].mean().reset_index()
    top_store = h1.groupby("store")["actual"].sum().idxmax()
    cand = means[means["store"] == top_store].copy()
    cand["gap"] = (cand["actual"] - target_mean).abs()
    row = cand.sort_values("gap").iloc[0]
    return row["store"], row["sku"]


def forecast_fan(sales: pd.DataFrame, forecasts: pd.DataFrame, store: str, sku: str, path: Path) -> Path:
    hist = sales[(sales["store"] == store) & (sales["sku"] == sku)]
    fc = forecasts[(forecasts["store"] == store) & (forecasts["sku"] == sku) & (forecasts["h"] == 1)]
    fc = fc.sort_values("target_idx")
    start = int(fc["target_idx"].min())
    hist = hist[hist["week_idx"] >= start - 52]

    fig, ax = plt.subplots(figsize=(9, 4.4))
    fig.subplots_adjust(top=0.74)
    ax.fill_between(fc["target_idx"], fc["p10"], fc["p90"], color=ORANGE, alpha=0.16, linewidth=0, step=None)
    ax.plot(hist["week_idx"], hist["units"], color=BLUE, linewidth=1.6)
    ax.plot(fc["target_idx"], fc["p50"], color=ORANGE, linewidth=1.6, linestyle=(0, (4, 2)))
    ax.set_ylim(bottom=0)
    ax.axvline(start - 0.5, color=AXIS, linewidth=1)
    ax.text(start, ax.get_ylim()[1] * 0.97, "test weeks", color=INK_2, fontsize=8.5, va="top")
    ax.set_xlim(hist["week_idx"].min(), fc["target_idx"].max() + 0.5)
    ax.set_xlabel("Week")
    ax.set_ylabel("Units per week")
    handles = [
        Line2D([], [], color=BLUE, linewidth=1.6, label="Actual"),
        Line2D([], [], color=ORANGE, linewidth=1.6, linestyle=(0, (4, 2)), label="Forecast, p50 (1 week ahead)"),
        plt.Rectangle((0, 0), 1, 1, color=ORANGE, alpha=0.16, label="p10 to p90 interval"),
    ]
    ax.legend(handles=handles, loc="lower left", ncol=3, bbox_to_anchor=(0, 1.01), fontsize=9)
    _title(
        fig, f"One store, one SKU ({store} · {sku})", "Synthetic data. The shaded band is the model's own uncertainty."
    )
    return _save(fig, path)


def wape_bars(summary: dict, path: Path) -> Path:
    m = summary["methods"]
    order = sorted(m, key=lambda k: m[k]["wape"], reverse=True)
    labels = [m[k]["label"] for k in order]
    vals = [m[k]["wape"] for k in order]
    colors = [BLUE if k == "lgbm_p50" else AXIS for k in order]

    fig, ax = plt.subplots(figsize=(9, 3.0))
    fig.subplots_adjust(top=0.72, left=0.22)
    bars = ax.barh(labels, vals, color=colors, height=0.56, edgecolor=SURFACE, linewidth=2)
    for b, v in zip(bars, vals, strict=True):
        ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.1%}", va="center", color=INK, fontsize=10)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlim(0, max(vals) * 1.15)
    ax.tick_params(axis="y", length=0, labelsize=10)
    ax.set_xlabel("WAPE, lower is better")
    _title(
        fig,
        "Forecast error by method",
        f"Rolling-origin backtest, {summary['origins']} weekly origins, horizons 1 to 4 weeks, every store x SKU",
    )
    return _save(fig, path)


def frontier_chart(front: pd.DataFrame, labels: dict[str, str], defaults: dict[str, float], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    fig.subplots_adjust(top=0.74)
    # Past about twice the default min/max stock every curve flattens near zero, so crop
    # there. Each line runs to the crop edge (interpolated, no marker) so the end labels
    # share one x and keep the curves' true order.
    base = front[(front["policy"] == "minmax") & (front["knob"] == defaults["minmax"])]
    x_cap = float(base["avg_inventory_value"].iloc[0]) * 2.0 / 1000 if len(base) else None
    ends = []
    for policy in ("minmax", "ddmrp_classic", "ddmrp_forecast"):
        f = front[front["policy"] == policy].sort_values("avg_inventory_value")
        c = POLICY_COLOR[policy]
        x, y = (f["avg_inventory_value"] / 1000).to_numpy(), f["lost_share"].to_numpy()
        keep = x <= x_cap if x_cap else np.ones_like(x, bool)
        lx, ly = x[keep], y[keep]
        if x_cap and x.max() > x_cap:
            lx, ly = np.append(lx, x_cap), np.append(ly, np.interp(x_cap, x, y))
        ax.plot(lx, ly, color=c, linewidth=1.6)
        ax.plot(x[keep], y[keep], "o", color=c, markersize=5.5, markeredgecolor=SURFACE, markeredgewidth=1.5)
        d = f[(f["knob"] == defaults[policy]) & keep]
        ax.plot(
            d["avg_inventory_value"] / 1000,
            d["lost_share"],
            "o",
            markersize=11,
            markerfacecolor="none",
            markeredgecolor=c,
            markeredgewidth=1.6,
        )
        ends.append((ly[-1], lx[-1], policy))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}k"))
    ax.set_ylim(bottom=0)
    xmax = max(e[1] for e in ends)
    ax.set_xlim(right=xmax * 1.3)
    # The curves converge at high inventory, so stagger the end labels and draw leaders.
    ytop = ax.get_ylim()[1]
    for rank, (ey, ex, policy) in enumerate(sorted(ends)):
        label_y = ytop * (0.06 + 0.1 * rank)
        ax.plot([ex, xmax * 1.035], [ey, label_y], color=AXIS, linewidth=0.8)
        ax.text(xmax * 1.045, label_y, labels[policy], color=INK_2, fontsize=9, va="center")
    ax.set_xlabel("Average inventory value across all stores")
    ax.set_ylabel("Lost sales, share of demand")
    handles = [
        Line2D([], [], color=POLICY_COLOR[p], linewidth=1.6, marker="o", markersize=5.5, label=labels[p])
        for p in ("ddmrp_forecast", "ddmrp_classic", "minmax")
    ]
    handles.append(
        Line2D(
            [],
            [],
            color=MUTED,
            marker="o",
            markersize=10,
            markerfacecolor="none",
            linestyle="none",
            label="default setting",
        )
    )
    ax.legend(handles=handles, loc="lower left", ncol=4, bbox_to_anchor=(0, 1.01), fontsize=9)
    _title(
        fig,
        "Lost sales against inventory held",
        "Each point is one safety setting, simulated over the test weeks. Lower and further left is better.",
    )
    return _save(fig, path)


def buffer_trace(traces: pd.DataFrame, store: str, sku: str, path: Path) -> Path:
    tr = traces[(traces["store"] == store) & (traces["sku"] == sku) & (traces["policy"] == "ddmrp_forecast")]
    tr = tr.sort_values("week_idx")
    # Repeat the last row one week later so the final step has width.
    tr = pd.concat([tr, tr.tail(1).assign(week_idx=tr["week_idx"].iloc[-1] + 1)], ignore_index=True)
    x = tr["week_idx"].to_numpy()

    fig, ax = plt.subplots(figsize=(9, 4.6))
    fig.subplots_adjust(top=0.74)
    ax.fill_between(x, 0, tr["tor"], step="post", color=CRITICAL, alpha=0.20, linewidth=0)
    ax.fill_between(x, tr["tor"], tr["toy"], step="post", color=WARNING, alpha=0.26, linewidth=0)
    ax.fill_between(x, tr["toy"], tr["tog"], step="post", color=GOOD, alpha=0.18, linewidth=0)
    ax.step(x, tr["position"], where="post", color=INK_2, linewidth=1.2, linestyle=(0, (3, 2)))
    ax.step(x, tr["on_hand"], where="post", color=BLUE, linewidth=1.8)
    last = tr.iloc[-1]
    for lo, hi, name in (
        (0, last["tor"], "Red zone"),
        (last["tor"], last["toy"], "Yellow zone"),
        (last["toy"], last["tog"], "Green zone"),
    ):
        ax.text(x[-1] + 0.6, (lo + hi) / 2, name, color=INK_2, fontsize=9, va="center")
    ax.set_xlim(x[0], x[-1] + 3.8)
    ax.set_ylim(0, max(tr["tog"].max(), tr["position"].max()) * 1.1)
    ax.set_xlabel("Week")
    ax.set_ylabel("Units")
    handles = [
        Line2D([], [], color=BLUE, linewidth=1.8, label="On hand"),
        Line2D(
            [], [], color=INK_2, linewidth=1.2, linestyle=(0, (3, 2)), label="Net flow position (on hand + on order)"
        ),
    ]
    ax.legend(handles=handles, loc="lower left", ncol=2, bbox_to_anchor=(0, 1.01), fontsize=9)
    _title(
        fig,
        f"Forecast-sized buffer for {store} · {sku}",
        "Zones move with the forecast. When net flow drops into yellow, the policy orders back up to green.",
    )
    return _save(fig, path)
