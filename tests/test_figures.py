from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

import glft_lab.figures as figures
from glft_lab.data import AggTradeRow, BookTickerRow
from glft_lab.figures import (
    active_quote_segments,
    quantile_calibration_series,
    quote_distance_series,
    save_causal_replay_figure,
    save_oos_fill_calibration_figure,
    save_optimal_quote_distance_figure,
)
from glft_lab.glft import GLFTParameters, optimal_deltas
from glft_lab.replay import AccountingModel, ReplayConfig, Side, round_to_tick
from glft_lab.strategy import run_glft_replay


def _dimensions(path: Path) -> tuple[int, int]:
    header = path.read_bytes()[:24]
    assert header[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", header[16:24])


def _parameters() -> GLFTParameters:
    return GLFTParameters(
        A=1.0,
        k=1.0,
        gamma=0.1,
        sigma=1.0,
        max_inventory=5,
        mu=0.0,
    )


def test_quote_distance_series_preserves_glft_symmetry_and_boundaries() -> None:
    series = quote_distance_series(_parameters(), (1.0, 0.5, 0.1))

    assert len(series) == 3
    first = series[0]
    assert np.array_equal(first.inventory, np.arange(-5, 6))
    assert np.isnan(first.ask_distance[0])
    assert np.isnan(first.bid_distance[-1])
    assert first.bid_distance[5] == pytest.approx(1.002016, abs=1e-6)
    assert first.ask_distance[5] == pytest.approx(first.bid_distance[5])
    assert first.bid_distance[:-1] == pytest.approx(first.ask_distance[:0:-1])


def test_quantile_calibration_is_side_specific_and_handles_rare_probabilities() -> None:
    probability = np.asarray([0.001, 0.001, 0.002, 0.003, 0.01, 0.02])
    event = np.asarray([0, 0, 1, 0, 1, 1])

    series = quantile_calibration_series(
        event,
        probability,
        cluster=np.asarray(["a", "a", "b", "b", "c", "c"]),
        side="bid",
        n_bins=3,
        bootstrap_samples=200,
    )

    assert series.side == "bid"
    assert np.sum(series.count) == len(event)
    assert np.all(np.diff(series.predicted) >= 0.0)
    assert np.all(np.isfinite(series.lower))
    assert np.all(np.isfinite(series.upper))
    assert np.all(series.lower <= series.upper)


def test_cluster_bootstrap_batches_large_weight_matrices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(figures, "_BOOTSTRAP_WEIGHT_BUDGET_BYTES", 64)
    success = np.asarray([[1.0], [0.0], [1.0], [0.0]])
    observations = np.ones_like(success)

    rates = figures._cluster_bootstrap_rates(
        success,
        observations,
        samples=9,
        seed=7,
    )

    assert rates.shape == (9, 1)
    assert np.all(np.isfinite(rates))
    assert np.all((rates >= 0.0) & (rates <= 1.0))


def test_static_quote_and_calibration_figures_are_exactly_1600_by_900(
    tmp_path: Path,
) -> None:
    quotes = save_optimal_quote_distance_figure(tmp_path / "quotes.png")
    calibration = save_oos_fill_calibration_figure(
        tmp_path / "calibration.png",
        probability=np.linspace(0.001, 0.03, 40),
        count=np.asarray(([0] * 8 + [1, 0]) * 4),
        side=np.asarray(["bid"] * 20 + ["ask"] * 20),
        cluster=np.asarray([f"episode-{index // 4}" for index in range(40)]),
        train_label="2024-09",
        test_label="2024-10",
        n_bins=4,
    )

    assert _dimensions(quotes) == (1600, 900)
    assert _dimensions(calibration) == (1600, 900)


def test_quote_figure_keeps_negative_deltas_visible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, tuple[float, float]] = {}

    def capture(figure: object, output: str | Path, dpi: int) -> Path:
        captured["ylim"] = figure.axes[0].get_ylim()  # type: ignore[attr-defined]
        return Path(output)

    monkeypatch.setattr(figures, "_save", capture)
    parameters = GLFTParameters(
        A=1.0,
        k=1.0,
        gamma=0.1,
        sigma=1.0,
        max_inventory=5,
        mu=1.0,
    )
    minimum = min(
        np.nanmin(item.bid_distance) for item in quote_distance_series(parameters, (1.0,))
    )

    save_optimal_quote_distance_figure(
        tmp_path / "negative.png",
        parameters=parameters,
        time_to_horizons=(1.0,),
    )

    assert minimum < 0.0
    assert captured["ylim"][0] < minimum


def test_figure_output_requires_png_extension(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=".png extension"):
        save_optimal_quote_distance_figure(tmp_path / "quotes.svg")


def test_replay_figure_uses_exchange_active_quotes_and_inverse_accounting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameters = GLFTParameters(
        A=2.0,
        k=1.0,
        gamma=0.1,
        sigma=0.2,
        max_inventory=2,
    )
    base_ms = 1_725_148_800_000
    delta = optimal_deltas(parameters, 0, 0.0, 2.0)
    bid = round_to_tick(100.0 - delta.bid_delta, Side.BUY, 0.1)
    run = run_glft_replay(
        [
            BookTickerRow(base_ms, 1, 90.0, 0.0, 110.0, 0.0),
            AggTradeRow(base_ms + 1_000, 2, bid, 1.0, True),
            BookTickerRow(base_ms + 2_000, 3, 90.0, 0.0, 110.0, 0.0),
        ],
        parameters=parameters,
        replay_config=ReplayConfig(
            tick_size=0.1,
            placement_latency=0.005,
            cancel_latency=0.005,
            maker_fee_rate=0.0001,
            taker_fee_rate=0.0005,
            accounting_model=AccountingModel.INVERSE,
            contract_multiplier=100.0,
            quantity_step=1.0,
            markout_horizons=(),
        ),
        horizon=2.0,
        order_quantity=1.0,
        inventory_unit=2.0,
    )

    bid_segments = active_quote_segments(run, Side.BUY)
    assert len(bid_segments) > 0
    assert len(active_quote_segments(run, Side.SELL)) > 0
    filled_segment = bid_segments[np.isclose(bid_segments[:, 0, 1], bid)]
    assert np.any(np.isclose(filled_segment[:, 1, 0], 1.0 / 60.0))
    assert run.parameters == parameters
    assert run.inventory_unit == 2.0
    assert run.replay_config.placement_latency == pytest.approx(0.005)
    assert run.replay_config.maker_fee_rate == pytest.approx(0.0001)
    output = save_causal_replay_figure(
        run,
        tmp_path / "replay.png",
        pnl_unit="BTC",
    )
    assert _dimensions(output) == (1600, 900)

    captured: dict[str, tuple[float, float]] = {}

    def capture(figure: object, output: str | Path, dpi: int) -> Path:
        captured["inventory_ylim"] = figure.axes[1].get_ylim()  # type: ignore[attr-defined]
        return Path(output)

    monkeypatch.setattr(figures, "_save", capture)
    save_causal_replay_figure(run, tmp_path / "captured.png", pnl_unit="BTC")
    physical_bound = parameters.max_inventory * run.inventory_unit
    assert captured["inventory_ylim"][1] > physical_bound


@pytest.mark.parametrize(
    ("symbol", "expected"),
    (("BTCUSD_PERP", "USD"), ("ETHUSDT", "USDT"), ("CUSTOM", "quote asset")),
)
def test_quote_unit_is_inferred_without_hard_coding_btcusd(
    symbol: str,
    expected: str,
) -> None:
    assert figures._quote_unit(symbol) == expected
