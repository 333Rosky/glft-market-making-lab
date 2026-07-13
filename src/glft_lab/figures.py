"""Publication-ready figures for the GLFT benchmark and empirical replay.

The rendering layer stays downstream of the research engines: it never recomputes
fills, invents queue information, or changes model outputs.  All figures use a fixed
1600x900 dark canvas so the checked-in PNGs can be reused on the project website.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, PercentFormatter

from .glft import GLFTParameters, inventory_grid, optimal_deltas
from .hazard import normalize_side
from .replay import AccountingModel, Liquidity, OrderStatus, Side
from .strategy import GLFTReplayRun

BACKGROUND = "#050505"
FOREGROUND = "#F2F2F2"
LIGHT_GRAY = "#B8B8B8"
MID_GRAY = "#858585"
DARK_GRAY = "#555555"
GRID = "#242424"
_BOOTSTRAP_WEIGHT_BUDGET_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class QuoteDistanceSeries:
    """Exact GLFT quote distances for one time-to-horizon."""

    time_to_horizon: float
    inventory: np.ndarray
    bid_distance: np.ndarray
    ask_distance: np.ndarray


@dataclass(frozen=True, slots=True)
class CalibrationSeries:
    """One side of an equal-frequency reliability curve."""

    side: str
    predicted: np.ndarray
    observed: np.ndarray
    count: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


def quote_distance_series(
    parameters: GLFTParameters,
    time_to_horizons: tuple[float, ...] = (1.0, 0.5, 0.1),
) -> tuple[QuoteDistanceSeries, ...]:
    """Return exact finite-horizon bid/ask distances over the full inventory grid."""

    if not time_to_horizons or any(
        not math.isfinite(value) or value <= 0.0 for value in time_to_horizons
    ):
        raise ValueError("time_to_horizons must contain finite positive values")
    grid = inventory_grid(parameters)
    series: list[QuoteDistanceSeries] = []
    for time_to_horizon in time_to_horizons:
        bid = np.full(len(grid), np.nan)
        ask = np.full(len(grid), np.nan)
        for index, inventory in enumerate(grid):
            deltas = optimal_deltas(
                parameters,
                int(inventory),
                t=0.0,
                horizon=time_to_horizon,
            )
            if deltas.bid_delta is not None:
                bid[index] = deltas.bid_delta
            if deltas.ask_delta is not None:
                ask[index] = deltas.ask_delta
        series.append(
            QuoteDistanceSeries(
                time_to_horizon=float(time_to_horizon),
                inventory=grid.copy(),
                bid_distance=bid,
                ask_distance=ask,
            )
        )
    return tuple(series)


def quantile_calibration_series(
    event: Any,
    probability: Any,
    *,
    cluster: Any,
    side: str,
    n_bins: int = 10,
    bootstrap_samples: int = 2_000,
    seed: int = 0,
) -> CalibrationSeries:
    """Bin rare events by quantile with passive-order cluster-bootstrap intervals."""

    observed = np.asarray(event, dtype=float).reshape(-1)
    predicted = np.asarray(probability, dtype=float).reshape(-1)
    clusters = np.asarray(cluster).reshape(-1)
    if len(observed) != len(predicted) or len(observed) != len(clusters) or not len(observed):
        raise ValueError("event, probability and cluster must have the same non-zero length")
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    if not np.all((observed == 0.0) | (observed == 1.0)):
        raise ValueError("event must be binary")
    if np.any(~np.isfinite(predicted)) or np.any((predicted < 0.0) | (predicted > 1.0)):
        raise ValueError("probability must be finite and lie in [0, 1]")

    edges = np.unique(np.quantile(predicted, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) == 1:
        bin_index = np.zeros(len(predicted), dtype=int)
        bin_count = 1
    else:
        bin_index = np.searchsorted(edges[1:-1], predicted, side="right")
        bin_count = len(edges) - 1

    active_bins = [index for index in range(bin_count) if np.any(bin_index == index)]
    predicted_mean: list[float] = []
    observed_mean: list[float] = []
    counts: list[int] = []
    for index in active_bins:
        mask = bin_index == index
        sample_count = int(np.sum(mask))
        successes = float(np.sum(observed[mask]))
        predicted_mean.append(float(np.mean(predicted[mask])))
        observed_mean.append(successes / sample_count)
        counts.append(sample_count)

    unique_clusters, cluster_index = np.unique(clusters, return_inverse=True)
    cluster_count = len(unique_clusters)
    cluster_success = np.zeros((cluster_count, len(active_bins)))
    cluster_observations = np.zeros((cluster_count, len(active_bins)))
    for output_index, bin_value in enumerate(active_bins):
        mask = bin_index == bin_value
        cluster_success[:, output_index] = np.bincount(
            cluster_index[mask],
            weights=observed[mask],
            minlength=cluster_count,
        )
        cluster_observations[:, output_index] = np.bincount(
            cluster_index[mask],
            minlength=cluster_count,
        )
    if cluster_count == 1:
        lower = np.asarray(observed_mean)
        upper = np.asarray(observed_mean)
    else:
        bootstrap_rate = _cluster_bootstrap_rates(
            cluster_success,
            cluster_observations,
            samples=bootstrap_samples,
            seed=seed,
        )
        lower = np.nanquantile(bootstrap_rate, 0.025, axis=0)
        upper = np.nanquantile(bootstrap_rate, 0.975, axis=0)

    return CalibrationSeries(
        side=str(side),
        predicted=np.asarray(predicted_mean),
        observed=np.asarray(observed_mean),
        count=np.asarray(counts, dtype=int),
        lower=lower,
        upper=upper,
    )


def active_quote_segments(run: GLFTReplayRun, side: Side) -> np.ndarray:
    """Reconstruct actual exchange-active quote intervals from replay orders."""

    last_fill: dict[str, float] = {}
    for fill in run.replay.fills:
        if fill.liquidity is Liquidity.MAKER:
            last_fill[fill.order_id] = max(last_fill.get(fill.order_id, -math.inf), fill.timestamp)

    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for order in run.replay.orders.values():
        if order.quote_id is None or order.side is not side or order.active_at is None:
            continue
        end = order.canceled_at
        if order.status is OrderStatus.FILLED:
            end = last_fill.get(order.order_id, end)
        if end is None:
            end = run.end_timestamp
        end = min(float(end), run.end_timestamp)
        if end < order.active_at:
            continue
        start_minute = (order.active_at - run.start_timestamp) / 60.0
        end_minute = (end - run.start_timestamp) / 60.0
        segments.append(((start_minute, order.price), (end_minute, order.price)))
    return np.asarray(segments, dtype=float).reshape(-1, 2, 2)


def save_optimal_quote_distance_figure(
    output: str | Path,
    *,
    parameters: GLFTParameters | None = None,
    time_to_horizons: tuple[float, ...] = (1.0, 0.5, 0.1),
    width: int = 1600,
    height: int = 900,
    dpi: int = 100,
) -> Path:
    """Render the exact GLFT inventory skew as synchronized horizon facets."""

    model = parameters or GLFTParameters(
        A=1.0,
        k=1.0,
        gamma=0.1,
        sigma=1.0,
        max_inventory=5,
        mu=0.0,
    )
    data = quote_distance_series(model, time_to_horizons)
    figure = _figure(width, height, dpi)
    axes = figure.subplots(1, len(data), sharex=True, sharey=True)
    axes_array = np.atleast_1d(axes)
    finite_values = np.concatenate(
        [
            values[np.isfinite(values)]
            for item in data
            for values in (item.bid_distance, item.ask_distance)
        ]
    )
    minimum = float(np.min(finite_values))
    maximum = float(np.max(finite_values))
    padding = 0.08 * max(1.0, maximum - minimum)
    lower = min(0.0, minimum - padding)
    upper = max(0.0, maximum + padding)

    for axis, item in zip(axes_array, data, strict=True):
        _style_axis(axis)
        axis.plot(
            item.inventory,
            item.bid_distance,
            color=FOREGROUND,
            linewidth=2.1,
            marker="o",
            markersize=5.5,
            markerfacecolor=BACKGROUND,
            markeredgewidth=1.3,
            label="Bid distance",
        )
        axis.plot(
            item.inventory,
            item.ask_distance,
            color=MID_GRAY,
            linewidth=2.1,
            linestyle=(0, (5, 3)),
            marker="s",
            markersize=5.0,
            markerfacecolor=BACKGROUND,
            markeredgewidth=1.3,
            label="Ask distance",
        )
        axis.set_title(
            rf"$T-t={item.time_to_horizon:g}$",
            color=FOREGROUND,
            fontsize=15,
            fontweight="normal",
            pad=15,
        )
        axis.set_xticks(item.inventory)
        axis.set_xlim(item.inventory[0] - 0.35, item.inventory[-1] + 0.35)
        axis.set_ylim(lower, upper)
        if lower < 0.0:
            axis.axhline(0.0, color=DARK_GRAY, linewidth=1.0)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=6))

    axes_array[0].set_ylabel("Distance from mid · model price units", color=FOREGROUND)
    figure.supxlabel("Inventory q", color=FOREGROUND, fontsize=13, y=0.105)
    figure.suptitle(
        "GLFT Optimal Quote Distance by Inventory",
        color=FOREGROUND,
        fontsize=24,
        fontweight="normal",
        y=0.965,
    )
    figure.text(
        0.5,
        0.915,
        (
            f"Exact finite-horizon solution  ·  A={model.A:g}  k={model.k:g}  "
            f"γ={model.gamma:g}  σ={model.sigma:g}  μ={model.mu:g}  "
            f"Q={model.max_inventory}"
        ),
        color=LIGHT_GRAY,
        fontsize=12,
        ha="center",
    )
    figure.legend(
        handles=(
            Line2D([], [], color=FOREGROUND, marker="o", markerfacecolor=BACKGROUND, lw=2),
            Line2D(
                [],
                [],
                color=MID_GRAY,
                marker="s",
                markerfacecolor=BACKGROUND,
                lw=2,
                linestyle=(0, (5, 3)),
            ),
        ),
        labels=("Bid distance", "Ask distance"),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=2,
        frameon=False,
        labelcolor=FOREGROUND,
        fontsize=11,
        handlelength=3.0,
    )
    figure.text(
        0.985,
        0.028,
        "Risk-increasing side disabled at q = ±Q",
        color=MID_GRAY,
        fontsize=10,
        ha="right",
    )
    figure.subplots_adjust(left=0.075, right=0.975, bottom=0.18, top=0.84, wspace=0.09)
    return _save(figure, output, dpi)


def save_causal_replay_figure(
    run: GLFTReplayRun,
    output: str | Path,
    *,
    symbol: str = "BTCUSD_PERP",
    pnl_unit: str | None = None,
    width: int = 1600,
    height: int = 900,
    dpi: int = 100,
) -> Path:
    """Render synchronized prices, inventory and fee-aware marked-to-mid P&L."""

    points = run.replay.equity_curve
    if not points:
        raise ValueError("replay has no equity points")
    book_points = [point for point in points if point.reason == "book"] or list(points)
    price_time = np.asarray(
        [(point.timestamp - run.start_timestamp) / 60.0 for point in book_points]
    )
    mid_price = np.asarray([point.mid_price for point in book_points])
    event_time = np.asarray([(point.timestamp - run.start_timestamp) / 60.0 for point in points])
    inventory = np.asarray([point.inventory for point in points])
    equity = np.asarray([point.equity for point in points])
    pnl = equity - equity[0]

    accounting = run.replay.accounting_model
    if accounting is AccountingModel.INVERSE:
        scale = 1_000_000.0
        unit = pnl_unit or "base asset"
        pnl_label = f"Cumulative net P&L · μ{unit}"
    else:
        scale = 1.0
        unit = pnl_unit or "quote asset"
        pnl_label = f"Cumulative net P&L · {unit}"

    figure = _figure(width, height, dpi)
    axes = figure.subplots(
        3,
        1,
        sharex=True,
        gridspec_kw={"height_ratios": (2.3, 1.05, 1.05), "hspace": 0.12},
    )
    price_axis, inventory_axis, pnl_axis = axes
    for axis in axes:
        _style_axis(axis)

    price_axis.plot(price_time, mid_price, color=FOREGROUND, linewidth=1.15, label="Mid-price")
    for side, color, linestyle in (
        (Side.BUY, LIGHT_GRAY, "solid"),
        (Side.SELL, DARK_GRAY, (0, (5, 3))),
    ):
        segments = active_quote_segments(run, side)
        if len(segments):
            price_axis.add_collection(
                LineCollection(segments, colors=color, linewidths=1.0, linestyles=linestyle)
            )

    maker_buys = [
        fill
        for fill in run.replay.fills
        if fill.liquidity is Liquidity.MAKER and fill.side is Side.BUY
    ]
    maker_sells = [
        fill
        for fill in run.replay.fills
        if fill.liquidity is Liquidity.MAKER and fill.side is Side.SELL
    ]
    liquidations = [fill for fill in run.replay.fills if fill.is_liquidation]
    _scatter_fills(price_axis, maker_buys, run.start_timestamp, "^", FOREGROUND, "Bid fill")
    _scatter_fills(price_axis, maker_sells, run.start_timestamp, "v", MID_GRAY, "Ask fill")
    if liquidations:
        price_axis.scatter(
            [(fill.timestamp - run.start_timestamp) / 60.0 for fill in liquidations],
            [fill.price for fill in liquidations],
            marker="x",
            s=42,
            linewidths=1.4,
            color=LIGHT_GRAY,
            zorder=6,
            label="Terminal liquidation",
        )
    price_axis.set_ylabel(f"Price · {_quote_unit(symbol)}", color=FOREGROUND)
    legend_handles = [
        Line2D([], [], color=FOREGROUND, lw=1.2, label="Mid-price"),
        Line2D([], [], color=LIGHT_GRAY, lw=1.2, label="Active bid quote"),
        Line2D([], [], color=DARK_GRAY, lw=1.2, linestyle=(0, (5, 3)), label="Active ask quote"),
        Line2D(
            [],
            [],
            color=FOREGROUND,
            marker="^",
            markerfacecolor=BACKGROUND,
            linestyle="None",
            label="Bid fill",
        ),
        Line2D(
            [],
            [],
            color=MID_GRAY,
            marker="v",
            markerfacecolor=BACKGROUND,
            linestyle="None",
            label="Ask fill",
        ),
    ]
    if liquidations:
        legend_handles.append(
            Line2D(
                [],
                [],
                color=LIGHT_GRAY,
                marker="x",
                linestyle="None",
                label="Terminal liquidation",
            )
        )
    price_axis.legend(
        handles=legend_handles,
        loc="upper left",
        ncol=len(legend_handles),
        frameon=False,
        labelcolor=FOREGROUND,
        fontsize=9.5,
        handlelength=2.1,
        columnspacing=1.3,
    )

    inventory_axis.step(event_time, inventory, where="post", color=FOREGROUND, linewidth=1.4)
    inventory_bound = run.parameters.max_inventory * run.inventory_unit
    inventory_axis.axhline(
        inventory_bound,
        color=DARK_GRAY,
        linestyle=(0, (4, 4)),
        linewidth=1,
    )
    inventory_axis.axhline(
        -inventory_bound,
        color=DARK_GRAY,
        linestyle=(0, (4, 4)),
        linewidth=1,
    )
    inventory_limit = max(inventory_bound, float(np.max(np.abs(inventory), initial=0.0)))
    inventory_padding = max(0.75 * run.inventory_unit, 0.05 * inventory_limit)
    inventory_axis.set_ylim(
        -inventory_limit - inventory_padding, inventory_limit + inventory_padding
    )
    inventory_axis.axhline(0.0, color=GRID, linewidth=1)
    inventory_unit_label = (
        "contracts" if accounting is AccountingModel.INVERSE else "inventory units"
    )
    inventory_axis.set_ylabel(f"Inventory · {inventory_unit_label}", color=FOREGROUND)
    inventory_axis.yaxis.set_major_locator(MaxNLocator(integer=True))

    pnl_axis.plot(event_time, pnl * scale, color=FOREGROUND, linewidth=1.5)
    pnl_axis.axhline(0.0, color=DARK_GRAY, linewidth=1)
    pnl_axis.set_ylabel(pnl_label, color=FOREGROUND)
    pnl_axis.text(
        0.01,
        0.9,
        "Marked-to-mid · after configured fees",
        transform=pnl_axis.transAxes,
        color=MID_GRAY,
        fontsize=9,
        ha="left",
        va="top",
    )
    pnl_axis.set_xlabel("Elapsed time · minutes", color=FOREGROUND)
    pnl_axis.set_xlim(0.0, max(0.001, (run.end_timestamp - run.start_timestamp) / 60.0))

    start = datetime.fromtimestamp(run.start_timestamp, tz=timezone.utc)
    end = datetime.fromtimestamp(run.end_timestamp, tz=timezone.utc)
    figure.suptitle(
        "Causal Market Replay",
        color=FOREGROUND,
        fontsize=24,
        fontweight="normal",
        y=0.972,
    )
    figure.text(
        0.5,
        0.928,
        (
            f"{symbol}  ·  {start:%Y-%m-%d %H:%M:%S}–{end:%H:%M:%S} UTC  ·  "
            "research replay, not live performance"
        ),
        color=LIGHT_GRAY,
        fontsize=11.5,
        ha="center",
    )
    figure.text(
        0.5,
        0.895,
        (
            f"placement latency {run.replay_config.placement_latency * 1_000:g} ms  ·  "
            f"cancel latency {run.replay_config.cancel_latency * 1_000:g} ms  ·  "
            "partial fills  ·  BBO queue-ahead approximation  ·  "
            f"{accounting.value} accounting  ·  configured fees "
            f"{run.replay_config.maker_fee_rate * 10_000:g}/"
            f"{run.replay_config.taker_fee_rate * 10_000:g} "
            "bp maker/taker"
        ),
        color=MID_GRAY,
        fontsize=10,
        ha="center",
    )
    figure.subplots_adjust(left=0.09, right=0.975, bottom=0.09, top=0.85)
    return _save(figure, output, dpi)


def save_oos_fill_calibration_figure(
    output: str | Path,
    *,
    probability: Any,
    count: Any,
    side: Any,
    cluster: Any,
    train_label: str,
    test_label: str,
    n_bins: int = 10,
    width: int = 1600,
    height: int = 900,
    dpi: int = 100,
) -> Path:
    """Render bid/ask OOS fill reliability with equal-frequency bins."""

    probabilities = np.asarray(probability, dtype=float).reshape(-1)
    counts = np.asarray(count, dtype=float).reshape(-1)
    clusters = np.asarray(cluster).reshape(-1)
    if len(probabilities) != len(counts) or len(probabilities) != len(clusters):
        raise ValueError("probability, count and cluster must have the same length")
    sides = normalize_side(side, len(counts))
    events = counts > 0.0
    series = tuple(
        quantile_calibration_series(
            events[sides == current_side],
            probabilities[sides == current_side],
            cluster=clusters[sides == current_side],
            side=current_side,
            n_bins=n_bins,
            seed=0 if current_side == "bid" else 1,
        )
        for current_side in ("bid", "ask")
    )

    figure = _figure(width, height, dpi)
    axis = figure.add_axes((0.24, 0.14, 0.52, 0.66))
    _style_axis(axis)
    maximum = max(
        0.01,
        *(float(np.max(values)) for item in series for values in (item.predicted, item.upper)),
    )
    limit = min(1.0, maximum * 1.12)
    axis.plot(
        [0.0, limit],
        [0.0, limit],
        color=DARK_GRAY,
        linewidth=1.2,
        label="Perfect calibration",
    )

    for item, color, marker, linestyle, label in (
        (series[0], FOREGROUND, "o", "solid", "Bid"),
        (series[1], MID_GRAY, "s", (0, (5, 3)), "Ask"),
    ):
        maximum_count = max(1, int(np.max(item.count)))
        marker_size = 5.0 + 3.0 * np.sqrt(item.count / maximum_count)
        error = np.vstack((item.observed - item.lower, item.upper - item.observed))
        axis.errorbar(
            item.predicted,
            item.observed,
            yerr=error,
            color=color,
            linewidth=1.6,
            linestyle=linestyle,
            marker=marker,
            markersize=float(np.mean(marker_size)),
            markerfacecolor=BACKGROUND,
            markeredgewidth=1.2,
            capsize=3,
            label=f"{label} · n={int(np.sum(item.count)):,}",
        )

    axis.set_xlim(0.0, limit)
    axis.set_ylim(0.0, limit)
    axis.set_aspect("equal", adjustable="box")
    axis.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
    axis.set_xlabel("Predicted fill probability", color=FOREGROUND, labelpad=10)
    axis.set_ylabel("Observed fill rate", color=FOREGROUND, labelpad=10)
    axis.legend(
        loc="upper left",
        frameon=False,
        labelcolor=FOREGROUND,
        fontsize=10.5,
        handlelength=2.8,
    )
    figure.suptitle(
        "Out-of-Sample Fill Calibration",
        color=FOREGROUND,
        fontsize=24,
        fontweight="normal",
        y=0.955,
    )
    figure.text(
        0.5,
        0.905,
        f"Train {train_label}  →  Test {test_label}  ·  bid and ask estimated separately",
        color=LIGHT_GRAY,
        fontsize=12,
        ha="center",
    )
    figure.text(
        0.5,
        0.065,
        (
            "Equal-frequency bins by side  ·  episode-cluster bootstrap 95% (descriptive)  ·  "
            "counterfactual BBO episodes with approximate queue, not observed live-order fills"
        ),
        color=MID_GRAY,
        fontsize=10,
        ha="center",
    )
    return _save(figure, output, dpi)


def _scatter_fills(
    axis: Any,
    fills: list[Any],
    start_timestamp: float,
    marker: str,
    color: str,
    label: str,
) -> None:
    if not fills:
        return
    axis.scatter(
        [(fill.timestamp - start_timestamp) / 60.0 for fill in fills],
        [fill.price for fill in fills],
        marker=marker,
        s=38,
        facecolors=BACKGROUND,
        edgecolors=color,
        linewidths=1.2,
        zorder=6,
        label=label,
    )


def _cluster_bootstrap_rates(
    cluster_success: np.ndarray,
    cluster_observations: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> np.ndarray:
    """Draw an exact cluster bootstrap while keeping the weight matrix bounded."""

    cluster_count, bin_count = cluster_success.shape
    probabilities = np.full(cluster_count, 1.0 / cluster_count)
    bytes_per_draw = cluster_count * np.dtype(np.int64).itemsize
    batch_size = max(
        1,
        min(samples, _BOOTSTRAP_WEIGHT_BUDGET_BYTES // max(1, bytes_per_draw)),
    )
    rates = np.empty((samples, bin_count))
    rng = np.random.default_rng(seed)
    for start in range(0, samples, batch_size):
        stop = min(samples, start + batch_size)
        weights = rng.multinomial(
            cluster_count,
            probabilities,
            size=stop - start,
        )
        successes = weights @ cluster_success
        observations = weights @ cluster_observations
        np.divide(
            successes,
            observations,
            out=rates[start:stop],
            where=observations > 0,
        )
        rates[start:stop][observations == 0] = np.nan
    return rates


def _quote_unit(symbol: str) -> str:
    """Infer common quote assets without pretending arbitrary symbols are parseable."""

    normalized = symbol.upper().split("_", maxsplit=1)[0]
    for quote in ("FDUSD", "USDT", "USDC", "BUSD", "USD", "EUR", "GBP"):
        if normalized.endswith(quote):
            return quote
    return "quote asset"


def _figure(width: int, height: int, dpi: int) -> Figure:
    if width < 1 or height < 1 or dpi < 1:
        raise ValueError("width, height and dpi must be positive")
    figure = Figure(
        figsize=(width / dpi, height / dpi),
        dpi=dpi,
        facecolor=BACKGROUND,
    )
    FigureCanvasAgg(figure)
    return figure


def _style_axis(axis: Any) -> None:
    axis.set_facecolor(BACKGROUND)
    axis.grid(True, color=GRID, linewidth=0.7, alpha=0.85)
    axis.set_axisbelow(True)
    axis.tick_params(colors=LIGHT_GRAY, labelsize=10)
    axis.xaxis.label.set_color(FOREGROUND)
    axis.yaxis.label.set_color(FOREGROUND)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(DARK_GRAY)
    axis.spines["bottom"].set_color(DARK_GRAY)


def _save(figure: Figure, output: str | Path, dpi: int) -> Path:
    path = Path(output)
    if path.suffix.lower() != ".png":
        raise ValueError("figure output must use a .png extension")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        dpi=dpi,
        facecolor=BACKGROUND,
        edgecolor=BACKGROUND,
        bbox_inches=None,
        metadata={"Software": "glft-market-making-lab"},
    )
    figure.clear()
    return path


__all__ = [
    "CalibrationSeries",
    "QuoteDistanceSeries",
    "active_quote_segments",
    "quantile_calibration_series",
    "quote_distance_series",
    "save_causal_replay_figure",
    "save_oos_fill_calibration_figure",
    "save_optimal_quote_distance_figure",
]
