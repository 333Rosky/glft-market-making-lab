"""Small, JSON-oriented command line interface for the GLFT research lab."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from .data import (
    AggTradeRow,
    iter_agg_trades,
    iter_book_ticker,
    merge_market_rows,
)
from .episodes import EpisodeConfig, build_hazard_intervals
from .glft import (
    THEORETICAL_BENCHMARK_LABEL,
    GLFTParameters,
    glft_constants,
    optimal_deltas,
    simulate_poisson_benchmark,
)
from .hazard import (
    FEATURE_NAMES,
    StateDependentFillHazard,
    evaluate_fill_predictions,
    hawkes_decision_gate,
    residual_diagnostics,
)
from .replay import (
    AccountingModel,
    Liquidity,
    ReplayConfig,
)
from .strategy import TradeQuantityMode, run_glft_replay
from .walkforward import walk_forward_monthly

BBO_EPISODE_LABEL = (
    "counterfactual passive-order fill episodes using BBO-only queue approximations; "
    "not observed fills or exact FIFO/L2 reconstruction"
)


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("expected a finite positive number")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("expected a finite non-negative number")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("expected a non-negative integer")
    return parsed


def _unit_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0 < parsed <= 1:
        raise argparse.ArgumentTypeError("expected a number in (0, 1]")
    return parsed


def _optional_limit(value: int | None) -> int | None:
    return None if value in {None, 0} else value


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.datetime64):
        return str(value)
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, (Path, Enum)):
        return str(value.value if isinstance(value, Enum) else value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _emit_json(payload: Any) -> None:
    print(json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False))


def _benchmark(args: argparse.Namespace) -> None:
    parameters = GLFTParameters(
        A=args.A,
        k=args.k,
        gamma=args.gamma,
        sigma=args.sigma,
        max_inventory=args.max_inventory,
        mu=args.mu,
    )
    quote = optimal_deltas(parameters, args.inventory, args.quote_time, args.horizon)
    simulation = simulate_poisson_benchmark(
        parameters,
        horizon=args.horizon,
        dt=args.dt,
        initial_mid_price=args.mid_price,
        initial_inventory=args.inventory,
        initial_cash=args.cash,
        seed=args.seed,
    )
    initial_equity = float(simulation.equity[0])
    _emit_json(
        {
            "label": THEORETICAL_BENCHMARK_LABEL,
            "scope": (
                "exact finite-inventory GLFT equations; independent theoretical "
                "Poisson fills, not an empirical execution replay"
            ),
            "parameters": asdict(parameters),
            "constants": asdict(glft_constants(parameters)),
            "quote": {
                "inventory": args.inventory,
                "time": args.quote_time,
                **asdict(quote),
            },
            "simulation": {
                "seed": args.seed,
                "steps": int(len(simulation.step_duration)),
                "bid_fills": int(np.sum(simulation.bid_fills)),
                "ask_fills": int(np.sum(simulation.ask_fills)),
                "final_cash": float(simulation.cash[-1]),
                "final_inventory": int(simulation.inventory[-1]),
                "final_mid_price": float(simulation.mid_price[-1]),
                "final_equity": float(simulation.equity[-1]),
                "net_pnl": float(simulation.equity[-1] - initial_equity),
                "max_absolute_inventory": int(np.max(np.abs(simulation.inventory))),
            },
        }
    )


def _markout_summary(markouts: Iterable[Any]) -> dict[str, Any]:
    grouped: dict[float, list[float]] = defaultdict(list)
    for markout in markouts:
        if markout.signed_bps is not None:
            grouped[markout.horizon].append(markout.signed_bps)
    result: dict[str, Any] = {}
    for horizon, values in sorted(grouped.items()):
        array = np.asarray(values, dtype=float)
        result[str(horizon)] = {
            "count": len(array),
            "mean_signed_bps": float(np.mean(array)),
            "median_signed_bps": float(np.median(array)),
            "adverse_fraction": float(np.mean(array < 0)),
            "p05_signed_bps": float(np.quantile(array, 0.05)),
        }
    return result


def _market_rows(args: argparse.Namespace) -> Iterator[Any]:
    start_ms = _parse_timestamp_ms(args.start)
    end_ms = _parse_timestamp_ms(args.end)
    if start_ms is not None and end_ms is not None and start_ms >= end_ms:
        raise ValueError("start must be strictly before end")
    limit = _optional_limit(args.max_rows)
    return merge_market_rows(
        iter_book_ticker(args.book, start_ms=start_ms, end_ms=end_ms, max_rows=limit),
        iter_agg_trades(args.trades, start_ms=start_ms, end_ms=end_ms, max_rows=limit),
    )


def _replay(args: argparse.Namespace) -> None:
    accounting_model = AccountingModel(args.accounting_model)
    if accounting_model is AccountingModel.INVERSE and args.contract_multiplier is None:
        raise ValueError("--contract-multiplier is required for inverse accounting")
    contract_multiplier = 1.0 if args.contract_multiplier is None else args.contract_multiplier
    parameters = GLFTParameters(
        A=args.A,
        k=args.k,
        gamma=args.gamma,
        sigma=args.sigma,
        max_inventory=args.max_inventory,
        mu=args.mu,
    )
    config = ReplayConfig(
        tick_size=args.tick_size,
        placement_latency=args.placement_latency_ms / 1_000.0,
        cancel_latency=args.cancel_latency_ms / 1_000.0,
        maker_fee_rate=args.maker_fee_bps / 10_000.0,
        taker_fee_rate=args.taker_fee_bps / 10_000.0,
        initial_cash=args.cash,
        initial_inventory=args.inventory,
        liquidate_at_end=not args.no_liquidate,
        markout_horizons=tuple(args.markout_horizons),
        accounting_model=accounting_model,
        contract_multiplier=contract_multiplier,
        initial_entry_price=args.initial_entry_price,
        quantity_step=args.quantity_step,
    )
    run = run_glft_replay(
        _market_rows(args),
        parameters=parameters,
        replay_config=config,
        horizon=args.horizon,
        order_quantity=args.order_size,
        inventory_unit=args.inventory_unit,
        quote_interval=args.quote_interval,
        max_events=args.max_events,
        trade_quantity_mode=TradeQuantityMode(args.trade_quantity_mode),
    )
    result = run.replay
    maker_fills = [fill for fill in result.fills if fill.liquidity is Liquidity.MAKER]
    taker_fills = [fill for fill in result.fills if fill.liquidity is Liquidity.TAKER]
    equity = np.asarray([point.equity for point in result.equity_curve], dtype=float)
    inventory = np.asarray([point.inventory for point in result.equity_curve], dtype=float)
    initial_equity = float(equity[0]) if len(equity) else args.cash
    drawdown = np.maximum.accumulate(equity) - equity if len(equity) else np.asarray([0.0])
    _emit_json(
        {
            "label": "exact GLFT quoting policy on causal Binance BBO event replay",
            "queue_model": result.queue_model,
            "queue_model_is_exact_fifo_l2": False,
            "tie_policy": result.tie_policy,
            "inputs": {
                "book": args.book,
                "trades": args.trades,
                "source_events": run.market_event_count,
                "max_rows_per_file": _optional_limit(args.max_rows),
                "start": args.start,
                "end": args.end,
            },
            "strategy": {
                "description": ("exact finite-horizon GLFT deltas, then passive tick projection"),
                "parameters": parameters,
                "horizon_seconds": run.horizon,
                "quote_interval_seconds": args.quote_interval,
                "order_size": args.order_size,
                "quote_decisions": len(run.decisions),
            },
            "accounting": {
                "model": result.accounting_model,
                "contract_multiplier": result.contract_multiplier,
                "trade_quantity_mode": run.trade_quantity_mode,
                "cash_unit": result.cash_unit,
                "quantity_step": result.quantity_step,
            },
            "performance": {
                "final_cash": result.final_cash,
                "final_inventory": result.final_inventory,
                "final_mid_price": result.final_mid_price,
                "final_equity": result.final_equity,
                "net_pnl": result.final_equity - initial_equity,
                "max_absolute_inventory": (
                    float(np.max(np.abs(inventory))) if len(inventory) else 0.0
                ),
                "max_drawdown": float(np.max(drawdown)),
                "final_average_entry_price": result.final_average_entry_price,
            },
            "fills": {
                "total": len(result.fills),
                "maker": len(maker_fills),
                "final_liquidation": len(taker_fills),
                "maker_quantity": float(sum(fill.quantity for fill in maker_fills)),
                "total_fees": float(sum(fill.fee for fill in result.fills)),
            },
            "realized_markouts_by_horizon_seconds": _markout_summary(result.markouts),
        }
    )


def _episode_rows(args: argparse.Namespace) -> Iterator[dict[str, Any]]:
    config = EpisodeConfig(
        tick_size=args.tick_size,
        order_quantity=args.order_size,
        distance_ticks=tuple(args.distance_ticks),
        decision_interval_ms=args.decision_interval_ms,
        placement_latency_ms=args.placement_latency_ms,
        horizon_ms=args.horizon_ms,
        observation_interval_ms=args.observation_interval_ms,
        adverse_move_ticks=args.adverse_move_ticks,
        state_ewma_alpha=args.volatility_alpha,
        quantity_step=args.quantity_step,
    )
    for interval in build_hazard_intervals(_market_rows(args), config):
        row = asdict(interval)
        row["timestamp"] = interval.timestamp
        yield row


def _episodes(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    fills = 0
    with output.open("w", encoding="utf-8") as handle:
        for row in _episode_rows(args):
            handle.write(json.dumps(_jsonable(row), allow_nan=False) + "\n")
            count += 1
            fills += int(row["count"] > 0)
    _emit_json(
        {
            "label": BBO_EPISODE_LABEL,
            "output": output,
            "episodes": count,
            "fill_opportunities": fills,
            "queue_model_is_exact_fifo_l2": False,
            "quantity_step": args.quantity_step,
        }
    )


def _load_episodes(
    paths: Sequence[str | Path], max_episodes: int | None
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    required = (
        "timestamp",
        "side",
        "distance",
        "spread",
        "imbalance",
        "ofi",
        "volatility",
        "queue_ahead",
        "order_age",
        "count",
        "exposure",
    )
    columns: dict[str, list[Any]] = {name: [] for name in required}
    optional: dict[str, list[Any]] = {
        "markout": [],
        "pnl": [],
        "inventory": [],
    }
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                missing = [name for name in required if name not in row]
                if missing:
                    raise ValueError(f"{path}:{line_number} missing fields: {', '.join(missing)}")
                for name in required:
                    columns[name].append(row[name])
                for name in optional:
                    optional[name].append(row.get(name))
                if max_episodes is not None and len(columns["count"]) >= max_episodes:
                    break
            if max_episodes is not None and len(columns["count"]) >= max_episodes:
                break
    if not columns["count"]:
        raise ValueError("episode input is empty")
    data = {
        name: np.asarray(values)
        for name, values in columns.items()
        if name not in {"count", "exposure"}
    }
    count = np.asarray(columns["count"], dtype=float)
    exposure = np.asarray(columns["exposure"], dtype=float)
    metrics = {
        name: np.asarray([np.nan if value is None else value for value in values], dtype=float)
        for name, values in optional.items()
    }
    return data, count, exposure, metrics


def _clean_diagnostics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    return {name: value for name, value in diagnostics.items() if name != "pearson_residuals"}


def _hazard_fit(args: argparse.Namespace) -> None:
    data, count, exposure, _ = _load_episodes(args.episodes, _optional_limit(args.max_episodes))
    model = StateDependentFillHazard(l2=args.l2).fit(data, count, exposure)
    rate = model.predict_rate(data)
    diagnostics = residual_diagnostics(
        count,
        rate,
        exposure,
        data["side"],
        timestamp=data["timestamp"],
        n_parameters=sum(len(value) for value in model.coefficients_.values()),
        max_lag=args.max_residual_lag,
        out_of_sample=False,
    )
    _emit_json(
        {
            "label": "in-sample state-dependent Poisson fill hazard fit",
            "episode_scope": BBO_EPISODE_LABEL,
            "episodes": len(count),
            "feature_names": FEATURE_NAMES,
            "feature_center": model.feature_builder.center_,
            "feature_scale": model.feature_builder.scale_,
            "coefficients_by_side": model.coefficients_,
            "fit_by_side": model.fit_results_,
            "training_metrics": evaluate_fill_predictions(count, rate, exposure),
            "residual_diagnostics": _clean_diagnostics(diagnostics),
            "hawkes_gate": hawkes_decision_gate(diagnostics),
        }
    )


def _timestamp_months(timestamp: np.ndarray) -> np.ndarray:
    values = np.asarray(timestamp)
    if np.issubdtype(values.dtype, np.number):
        numeric = values.astype(float)
        magnitude = float(np.max(np.abs(numeric), initial=0.0))
        if magnitude >= 1e17:
            nanoseconds = numeric
        elif magnitude >= 1e14:
            nanoseconds = numeric * 1e3
        elif magnitude >= 1e11:
            nanoseconds = numeric * 1e6
        else:
            nanoseconds = numeric * 1e9
        return nanoseconds.astype(np.int64).astype("datetime64[ns]").astype("datetime64[M]")
    return values.astype("datetime64[ns]").astype("datetime64[M]")


def _select_rows(
    data: dict[str, np.ndarray],
    count: np.ndarray,
    exposure: np.ndarray,
    metrics: dict[str, np.ndarray],
    mask: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    return (
        {name: values[mask] for name, values in data.items()},
        count[mask],
        exposure[mask],
        {name: values[mask] for name, values in metrics.items()},
    )


def _walk_forward(args: argparse.Namespace) -> None:
    data, count, exposure, trading = _load_episodes(
        args.episodes, _optional_limit(args.max_episodes)
    )
    explicit_pair = args.train_month is not None or args.test_month is not None
    if explicit_pair:
        if args.train_month is None or args.test_month is None:
            raise ValueError("--train-month and --test-month must be supplied together")
        train_month = np.datetime64(args.train_month, "M")
        test_month = np.datetime64(args.test_month, "M")
        if train_month >= test_month:
            raise ValueError("train month must be strictly before test month")
        months = _timestamp_months(data["timestamp"])
        if not np.any(months == train_month) or not np.any(months == test_month):
            raise ValueError("both requested train and test months must exist in episodes")
        selected = (months == train_month) | (months == test_month)
        data, count, exposure, trading = _select_rows(data, count, exposure, trading, selected)

    result = walk_forward_monthly(
        data,
        count,
        exposure,
        min_train_months=1 if explicit_pair else args.min_train_months,
        mode="expanding" if explicit_pair else args.mode,
        train_window_months=(
            None if explicit_pair or args.mode == "expanding" else args.train_window_months
        ),
        model_factory=lambda: StateDependentFillHazard(l2=args.l2),
        markout=trading["markout"],
        pnl=trading["pnl"],
        inventory=trading["inventory"],
        max_residual_lag=args.max_residual_lag,
    )
    folds = [
        {
            "test_month": fold.test_month,
            "train_months": fold.train_months,
            "train_episodes": len(fold.train_indices),
            "test_episodes": len(fold.test_indices),
            "metrics": fold.metrics,
        }
        for fold in result.folds
    ]
    _emit_json(
        {
            "label": "strict monthly out-of-sample fill-hazard walk-forward",
            "episode_scope": BBO_EPISODE_LABEL,
            "explicit_train_test_pair": explicit_pair,
            "folds": folds,
            "aggregate_oos_metrics": result.metrics,
            "residual_diagnostics": _clean_diagnostics(result.diagnostics),
            "hawkes_gate": result.hawkes_gate,
        }
    )


def _parse_timestamp_ms(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return int(parsed.timestamp() * 1_000)


def _counter_median(counter: Counter[int], count: int) -> float:
    targets = ((count - 1) // 2, count // 2)
    values: list[int] = []
    cumulative = 0
    for value, frequency in sorted(counter.items()):
        previous = cumulative
        cumulative += frequency
        for target in targets[len(values) :]:
            if previous <= target < cumulative:
                values.append(value)
            else:
                break
        if len(values) == 2:
            break
    return 0.5 * (values[0] + values[1])


def _population_fano(total: int, squared_total: int, bins: int) -> float | None:
    if bins < 1 or total == 0:
        return None
    mean = total / bins
    variance = squared_total / bins - mean**2
    return variance / mean


def _arrival_diagnostics(args: argparse.Namespace) -> None:
    start_ms = _parse_timestamp_ms(args.start)
    end_ms = _parse_timestamp_ms(args.end)
    if start_ms is not None and end_ms is not None and start_ms >= end_ms:
        raise ValueError("start must be strictly before end")
    timestamps: Iterator[AggTradeRow] = iter_agg_trades(
        args.trades,
        start_ms=start_ms,
        end_ms=end_ms,
        max_rows=_optional_limit(args.max_rows),
    )
    event_count = 0
    previous: int | None = None
    first_bin: int | None = None
    last_bin: int | None = None
    delta_count = 0
    delta_mean = 0.0
    delta_m2 = 0.0
    delta_histogram: Counter[int] = Counter()
    bin_counts: Counter[int] = Counter()
    for trade in timestamps:
        current = trade.timestamp_ms
        bin_index = current // 100
        bin_counts[bin_index] += 1
        first_bin = bin_index if first_bin is None else min(first_bin, bin_index)
        last_bin = bin_index if last_bin is None else max(last_bin, bin_index)
        if previous is not None:
            delta = current - previous
            delta_histogram[delta] += 1
            delta_count += 1
            difference = delta - delta_mean
            delta_mean += difference / delta_count
            delta_m2 += difference * (delta - delta_mean)
        previous = current
        event_count += 1

    if first_bin is None or last_bin is None:
        if start_ms is None or end_ms is None:
            raise ValueError("no aggTrade events in the requested window")
        first_bin = start_ms // 100
        last_bin = (end_ms - 1) // 100
    else:
        if start_ms is not None:
            first_bin = start_ms // 100
        if end_ms is not None:
            last_bin = (end_ms - 1) // 100
    bins = last_bin - first_bin + 1
    squared_total = sum(value * value for value in bin_counts.values())
    full_fano = _population_fano(event_count, squared_total, bins)
    minute_fanos: list[float] = []
    first_minute = first_bin // 600
    last_minute = last_bin // 600
    counts_by_minute: dict[int, list[int]] = defaultdict(list)
    for bin_index, value in bin_counts.items():
        counts_by_minute[bin_index // 600].append(value)
    for minute in range(first_minute, last_minute + 1):
        lower = max(first_bin, minute * 600)
        upper = min(last_bin, (minute + 1) * 600 - 1)
        minute_bins = upper - lower + 1
        values = counts_by_minute.get(minute, [])
        fano = _population_fano(sum(values), sum(value * value for value in values), minute_bins)
        if fano is not None:
            minute_fanos.append(fano)
    delta_std = math.sqrt(delta_m2 / delta_count) if delta_count else None
    _emit_json(
        {
            "label": (
                "Binance aggTrades arrival diagnostics; rows are aggregate trade "
                "events, not reconstructed child trades"
            ),
            "event_count": event_count,
            "window": {"start_ms": start_ms, "end_ms": end_ms},
            "interarrival_ms": {
                "count": delta_count,
                "mean": delta_mean if delta_count else None,
                "median": (_counter_median(delta_histogram, delta_count) if delta_count else None),
                "coefficient_of_variation": (
                    delta_std / delta_mean if delta_std is not None and delta_mean > 0 else None
                ),
            },
            "counts_100ms": {
                "binning": (
                    "absolute epoch-aligned bins intersecting requested [start, end); "
                    "partial boundary bins are retained, and omitted bounds use the "
                    "first/last occupied bin"
                ),
                "partial_boundary_bins": bool(
                    (start_ms is not None and start_ms % 100 != 0)
                    or (end_ms is not None and end_ms % 100 != 0)
                ),
                "bin_count": bins,
                "fano": full_fano,
                "median_within_minute_fano": (
                    float(np.median(minute_fanos)) if minute_fanos else None
                ),
                "minutes_with_events": len(minute_fanos),
            },
            "homogeneous_poisson_reference": {
                "interarrival_coefficient_of_variation": 1.0,
                "count_fano": 1.0,
            },
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="glft-lab",
        description="GLFT theoretical benchmark and empirical BBO execution research.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    benchmark = commands.add_parser(
        "benchmark", aliases=["glft"], help="run the exact theoretical GLFT benchmark"
    )
    benchmark.add_argument("--A", type=_positive_float, default=1.0)
    benchmark.add_argument("--k", type=_positive_float, default=1.0)
    benchmark.add_argument("--gamma", type=_positive_float, default=0.1)
    benchmark.add_argument("--sigma", type=_non_negative_float, default=1.0)
    benchmark.add_argument("--mu", type=float, default=0.0)
    benchmark.add_argument("--max-inventory", type=_positive_int, default=5)
    benchmark.add_argument("--horizon", type=_positive_float, default=1.0)
    benchmark.add_argument("--dt", type=_positive_float, default=0.01)
    benchmark.add_argument("--mid-price", type=_positive_float, default=100.0)
    benchmark.add_argument("--inventory", type=int, default=0)
    benchmark.add_argument("--cash", type=float, default=0.0)
    benchmark.add_argument("--quote-time", type=_non_negative_float, default=0.0)
    benchmark.add_argument("--seed", type=int, default=0)
    benchmark.set_defaults(handler=_benchmark)

    replay = commands.add_parser(
        "replay", help="replay Binance bookTicker/aggTrades with BBO queue approximation"
    )
    _add_market_paths(replay)
    replay.add_argument("--tick-size", type=_positive_float, required=True)
    replay.add_argument(
        "--accounting-model",
        choices=tuple(model.value for model in AccountingModel),
        required=True,
    )
    replay.add_argument(
        "--contract-multiplier",
        type=_positive_float,
        help="required for inverse accounting; linear defaults to 1",
    )
    replay.add_argument(
        "--trade-quantity-mode",
        choices=tuple(mode.value for mode in TradeQuantityMode),
        required=True,
        help="use as_is for official COIN-M contract counts",
    )
    replay.add_argument("--initial-entry-price", type=_positive_float)
    replay.add_argument(
        "--quantity-step",
        type=_positive_float,
        help="contract-size step; inverse accounting defaults to 1",
    )
    replay.add_argument("--A", type=_positive_float, default=1.0)
    replay.add_argument("--k", type=_positive_float, default=1.0)
    replay.add_argument("--gamma", type=_positive_float, default=0.1)
    replay.add_argument("--sigma", type=_non_negative_float, default=1.0)
    replay.add_argument("--mu", type=float, default=0.0)
    replay.add_argument("--max-inventory", type=_positive_int, default=5)
    replay.add_argument("--horizon", type=_positive_float)
    replay.add_argument("--order-size", type=_positive_float, default=1.0)
    replay.add_argument("--inventory-unit", type=_positive_float)
    replay.add_argument("--quote-interval", type=_non_negative_float, default=1.0)
    replay.add_argument("--max-events", type=_positive_int, default=1_000_000)
    replay.add_argument("--placement-latency-ms", type=_non_negative_float, default=0.0)
    replay.add_argument("--cancel-latency-ms", type=_non_negative_float, default=0.0)
    replay.add_argument("--maker-fee-bps", type=float, default=0.0)
    replay.add_argument("--taker-fee-bps", type=float, default=0.0)
    replay.add_argument("--cash", type=float, default=0.0)
    replay.add_argument("--inventory", type=float, default=0.0)
    replay.add_argument("--no-liquidate", action="store_true")
    replay.add_argument(
        "--markout-horizons",
        type=_comma_separated_non_negative,
        default=(1.0, 5.0, 30.0),
        metavar="SECONDS",
    )
    replay.set_defaults(handler=_replay)

    episodes = commands.add_parser(
        "episodes",
        aliases=["hazard-episodes"],
        help="write hypothetical BBO fill-hazard episodes as JSON Lines",
    )
    _add_market_paths(episodes)
    episodes.add_argument("--output", required=True)
    episodes.add_argument("--tick-size", type=_positive_float, required=True)
    episodes.add_argument("--order-size", type=_positive_float, default=1.0)
    episodes.add_argument(
        "--quantity-step",
        type=_positive_float,
        help="set to 1 for official BTCUSD_PERP COIN-M contract quantities",
    )
    episodes.add_argument(
        "--distance-ticks",
        type=_comma_separated_non_negative_int,
        default=(0, 1, 2),
    )
    episodes.add_argument("--decision-interval-ms", type=_positive_int, default=60_000)
    episodes.add_argument("--placement-latency-ms", type=_non_negative_int, default=10)
    episodes.add_argument("--horizon-ms", type=_positive_int, default=5_000)
    episodes.add_argument("--observation-interval-ms", type=_positive_int, default=250)
    episodes.add_argument("--adverse-move-ticks", type=_positive_float, default=2.0)
    episodes.add_argument("--volatility-alpha", type=_unit_float, default=0.05)
    episodes.set_defaults(handler=_episodes)

    fit = commands.add_parser("hazard-fit", help="fit side-specific Poisson hazards")
    _add_episode_inputs(fit)
    fit.add_argument("--l2", type=_non_negative_float, default=1e-4)
    fit.add_argument("--max-residual-lag", type=_positive_int, default=10)
    fit.set_defaults(handler=_hazard_fit)

    walk = commands.add_parser(
        "walk-forward",
        help="monthly OOS hazard evaluation (for example Sep train / Oct test)",
    )
    _add_episode_inputs(walk)
    walk.add_argument("--train-month", metavar="YYYY-MM")
    walk.add_argument("--test-month", metavar="YYYY-MM")
    walk.add_argument("--mode", choices=("expanding", "rolling"), default="expanding")
    walk.add_argument("--min-train-months", type=_positive_int, default=1)
    walk.add_argument("--train-window-months", type=_positive_int)
    walk.add_argument("--l2", type=_non_negative_float, default=1e-4)
    walk.add_argument("--max-residual-lag", type=_positive_int, default=10)
    walk.set_defaults(handler=_walk_forward)

    arrivals = commands.add_parser(
        "arrival-diagnostics",
        help="stream aggTrades interarrival and 100ms overdispersion diagnostics",
    )
    arrivals.add_argument("--trades", required=True)
    arrivals.add_argument("--start", help="inclusive epoch-ms or ISO-8601 timestamp")
    arrivals.add_argument("--end", help="exclusive epoch-ms or ISO-8601 timestamp")
    arrivals.add_argument("--max-rows", type=_non_negative_int, default=0)
    arrivals.set_defaults(handler=_arrival_diagnostics)
    return parser


def _add_market_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--book", required=True, help="Binance bookTicker CSV")
    parser.add_argument("--trades", required=True, help="Binance aggTrades CSV")
    parser.add_argument("--start", help="inclusive epoch-ms or ISO-8601 timestamp")
    parser.add_argument("--end", help="exclusive epoch-ms or ISO-8601 timestamp")
    parser.add_argument(
        "--max-rows",
        type=_non_negative_int,
        default=100_000,
        help="row cap per file; 0 means unlimited",
    )


def _add_episode_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--episodes", nargs="+", required=True, help="JSONL episode files")
    parser.add_argument(
        "--max-episodes", type=_non_negative_int, default=0, help="0 means unlimited"
    )


def _comma_separated_non_negative(value: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated seconds") from exc
    if not parsed or any(not math.isfinite(item) or item < 0 for item in parsed):
        raise argparse.ArgumentTypeError("markout horizons must be non-negative")
    return parsed


def _comma_separated_non_negative_int(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integer ticks") from exc
    if not parsed or any(item < 0 for item in parsed) or len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("distance ticks must be unique non-negative integers")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
