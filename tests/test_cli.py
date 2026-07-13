from __future__ import annotations

import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from glft_lab.cli import BBO_EPISODE_LABEL, main

BOOK_FIELDS = [
    "update_id",
    "best_bid_price",
    "best_bid_qty",
    "best_ask_price",
    "best_ask_qty",
    "transaction_time",
    "event_time",
]
TRADE_FIELDS = [
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "transact_time",
    "is_buyer_maker",
]


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _market_files(tmp_path: Path) -> tuple[Path, Path]:
    start = 1_725_148_800_000
    book = tmp_path / "book.csv"
    trades = tmp_path / "trades.csv"
    _write_csv(
        book,
        BOOK_FIELDS,
        [
            {
                "update_id": 1,
                "best_bid_price": 100,
                "best_bid_qty": 0,
                "best_ask_price": 101,
                "best_ask_qty": 0,
                "transaction_time": start,
                "event_time": start,
            },
            {
                "update_id": 2,
                "best_bid_price": 100,
                "best_bid_qty": 0,
                "best_ask_price": 101,
                "best_ask_qty": 0,
                "transaction_time": start + 2_000,
                "event_time": start + 2_000,
            },
        ],
    )
    _write_csv(
        trades,
        TRADE_FIELDS,
        [
            {
                "agg_trade_id": 1,
                "price": 90,
                "quantity": 1,
                "first_trade_id": 1,
                "last_trade_id": 1,
                "transact_time": start + 1_000,
                "is_buyer_maker": "true",
            }
        ],
    )
    return book, trades


def _stdout_json(capsys: pytest.CaptureFixture[str]) -> dict[str, object]:
    return json.loads(capsys.readouterr().out)


def test_benchmark_command_is_explicitly_theoretical_and_reproducible(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["benchmark", "--horizon", "0.1", "--dt", "0.05", "--seed", "7"]) == 0
    first = _stdout_json(capsys)
    assert main(["benchmark", "--horizon", "0.1", "--dt", "0.05", "--seed", "7"]) == 0
    second = _stdout_json(capsys)

    assert first == second
    assert "theoretical" in str(first["label"]).lower()
    assert "Poisson" in str(first["scope"])
    assert first["simulation"]["steps"] == 2


def test_plot_quotes_command_writes_a_site_ready_png(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "quotes.png"

    assert main(["plot-quotes", "--output", str(output)]) == 0
    payload = _stdout_json(capsys)

    assert payload["pixels"] == [1600, 900]
    assert payload["time_to_horizons"] == [1.0, 0.5, 0.1]
    assert output.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_replay_reports_bbo_scope_fills_inventory_pnl_and_markouts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    book, trades = _market_files(tmp_path)
    assert (
        main(
            [
                "replay",
                "--book",
                str(book),
                "--trades",
                str(trades),
                "--tick-size",
                "0.1",
                "--accounting-model",
                "linear",
                "--trade-quantity-mode",
                "as_is",
                "--markout-horizons",
                "1",
                "--no-liquidate",
            ]
        )
        == 0
    )
    payload = _stdout_json(capsys)

    assert payload["queue_model_is_exact_fifo_l2"] is False
    assert "bookTicker" in payload["queue_model"]
    assert payload["accounting"] == {
        "cash_unit": "quote_asset",
        "contract_multiplier": 1.0,
        "model": "linear",
        "quantity_step": None,
        "trade_quantity_mode": "as_is",
    }
    assert payload["fills"]["maker"] == 1
    assert payload["fills"]["total_fees"] == 0.0
    assert payload["performance"]["final_inventory"] == 1.0

    figure = tmp_path / "replay.png"
    assert (
        main(
            [
                "plot-replay",
                "--book",
                str(book),
                "--trades",
                str(trades),
                "--symbol",
                "BTCUSD_PERP",
                "--tick-size",
                "0.1",
                "--accounting-model",
                "inverse",
                "--contract-multiplier",
                "100",
                "--trade-quantity-mode",
                "as_is",
                "--quantity-step",
                "1",
                "--placement-latency-ms",
                "5",
                "--cancel-latency-ms",
                "5",
                "--maker-fee-bps",
                "1",
                "--taker-fee-bps",
                "5",
                "--pnl-unit",
                "BTC",
                "--no-liquidate",
                "--output",
                str(figure),
            ]
        )
        == 0
    )
    plotted = _stdout_json(capsys)
    assert plotted["accounting_model"] == "inverse"
    assert figure.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert payload["performance"]["max_absolute_inventory"] == 1.0
    assert payload["performance"]["net_pnl"] > 0
    assert payload["realized_markouts_by_horizon_seconds"]["1.0"]["count"] == 1


def test_replay_exposes_explicit_inverse_coin_m_accounting(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    book, trades = _market_files(tmp_path)
    assert (
        main(
            [
                "replay",
                "--book",
                str(book),
                "--trades",
                str(trades),
                "--tick-size",
                "0.1",
                "--accounting-model",
                "inverse",
                "--contract-multiplier",
                "100",
                "--trade-quantity-mode",
                "as_is",
                "--quantity-step",
                "1",
                "--no-liquidate",
            ]
        )
        == 0
    )
    payload = _stdout_json(capsys)
    assert payload["accounting"] == {
        "cash_unit": "base_asset",
        "contract_multiplier": 100.0,
        "model": "inverse",
        "quantity_step": 1.0,
        "trade_quantity_mode": "as_is",
    }
    assert payload["performance"]["final_inventory"] == 1.0


def test_episode_generation_and_hazard_training_are_json_scriptable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    book, trades = _market_files(tmp_path)
    episodes = tmp_path / "episodes.jsonl"
    assert (
        main(
            [
                "episodes",
                "--book",
                str(book),
                "--trades",
                str(trades),
                "--output",
                str(episodes),
                "--tick-size",
                "0.1",
                "--quantity-step",
                "1",
            ]
        )
        == 0
    )
    generation = _stdout_json(capsys)
    rows = [json.loads(line) for line in episodes.read_text().splitlines()]

    assert generation["label"] == BBO_EPISODE_LABEL
    assert generation["queue_model_is_exact_fifo_l2"] is False
    assert generation["quantity_step"] == 1.0
    assert generation["fill_opportunities"] == 3
    assert {row["side"] for row in rows} == {"bid", "ask"}
    fill = next(row for row in rows if row["risk"] == "fill")
    assert fill["count"] == 1
    assert fill["exposure"] > 0
    assert fill["risk"] == "fill"
    assert fill["queue_is_approximate"] is True

    assert main(["hazard-fit", "--episodes", str(episodes)]) == 0
    fit = _stdout_json(capsys)
    assert fit["episodes"] == len(rows)
    assert set(fit["coefficients_by_side"]) == {"ask", "bid"}
    assert fit["hawkes_gate"]["hawkes_warranted"] is False
    assert "in-sample" in fit["label"]


def test_episode_quantity_step_rejects_fractional_wrong_market_volume(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    book, trades = _market_files(tmp_path)
    start = 1_725_148_800_000
    _write_csv(
        trades,
        TRADE_FIELDS,
        [
            {
                "agg_trade_id": 1,
                "price": 90,
                "quantity": 0.5,
                "first_trade_id": 1,
                "last_trade_id": 1,
                "transact_time": start + 1_000,
                "is_buyer_maker": "true",
            }
        ],
    )
    with pytest.raises(SystemExit):
        main(
            [
                "episodes",
                "--book",
                str(book),
                "--trades",
                str(trades),
                "--output",
                str(tmp_path / "episodes.jsonl"),
                "--tick-size",
                "0.1",
                "--quantity-step",
                "1",
            ]
        )
    assert "multiple of quantity_step=1.0" in capsys.readouterr().err


def _month_ms(month: int, day: int) -> int:
    return int(datetime(2024, month, day, tzinfo=timezone.utc).timestamp() * 1_000)


def test_walk_forward_supports_explicit_september_train_october_test(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "two-months.jsonl"
    rows: list[dict[str, object]] = []
    for month in (9, 10):
        for day in range(1, 7):
            for side in ("bid", "ask"):
                filled = (day + (side == "ask")) % 3 == 0
                rows.append(
                    {
                        "timestamp": _month_ms(month, day),
                        "side": side,
                        "distance": 0.5 + 0.01 * day,
                        "spread": 1.0,
                        "imbalance": -0.1 if side == "bid" else 0.1,
                        "ofi": float(day - 3),
                        "volatility": 0.2,
                        "queue_ahead": float(day),
                        "order_age": float(day - 1),
                        "count": int(filled),
                        "exposure": 1.0,
                        "markout": 0.2 if filled else None,
                        "pnl": 0.2 if filled else 0.0,
                        "inventory": float((day % 3) - 1),
                    }
                )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    assert (
        main(
            [
                "walk-forward",
                "--episodes",
                str(path),
                "--train-month",
                "2024-09",
                "--test-month",
                "2024-10",
                "--max-residual-lag",
                "2",
            ]
        )
        == 0
    )
    payload = _stdout_json(capsys)

    assert payload["explicit_train_test_pair"] is True
    assert len(payload["folds"]) == 1
    assert payload["folds"][0]["train_months"] == ["2024-09"]
    assert payload["folds"][0]["test_month"] == "2024-10"
    assert payload["folds"][0]["train_episodes"] == 12
    assert payload["folds"][0]["test_episodes"] == 12
    assert payload["residual_diagnostics"]["out_of_sample"] is True

    figure = tmp_path / "calibration.png"
    assert (
        main(
            [
                "plot-calibration",
                "--episodes",
                str(path),
                "--train-month",
                "2024-09",
                "--test-month",
                "2024-10",
                "--calibration-bins",
                "3",
                "--output",
                str(figure),
            ]
        )
        == 0
    )
    plotted = _stdout_json(capsys)
    assert plotted["oos_intervals"] == 12
    assert plotted["oos_fill_events"] == 4
    assert figure.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_arrival_diagnostics_streams_exact_interarrival_and_fano_formulas(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "trades.csv"
    start = 1_725_148_800_000
    rows = []
    for identifier, offset in enumerate((0, 0, 100, 200), start=1):
        rows.append(
            {
                "agg_trade_id": identifier,
                "price": 100,
                "quantity": 1,
                "first_trade_id": identifier,
                "last_trade_id": identifier,
                "transact_time": start + offset,
                "is_buyer_maker": "false",
            }
        )
    _write_csv(path, TRADE_FIELDS, rows)

    assert main(["arrival-diagnostics", "--trades", str(path)]) == 0
    payload = _stdout_json(capsys)

    assert payload["event_count"] == 4
    assert payload["interarrival_ms"]["mean"] == pytest.approx(200 / 3)
    assert payload["interarrival_ms"]["median"] == 100.0
    assert payload["interarrival_ms"]["coefficient_of_variation"] == pytest.approx(np.sqrt(2) / 2)
    assert payload["counts_100ms"]["fano"] == pytest.approx(1 / 6)
    assert payload["counts_100ms"]["median_within_minute_fano"] == pytest.approx(1 / 6)

    assert (
        main(
            [
                "arrival-diagnostics",
                "--trades",
                str(path),
                "--start",
                str(start),
                "--end",
                str(start + 1_000),
            ]
        )
        == 0
    )
    explicit = _stdout_json(capsys)
    assert explicit["counts_100ms"]["bin_count"] == 10
    assert explicit["counts_100ms"]["fano"] == pytest.approx(1.1)
    assert explicit["counts_100ms"]["partial_boundary_bins"] is False

    assert (
        main(
            [
                "arrival-diagnostics",
                "--trades",
                str(path),
                "--start",
                str(start + 50),
                "--end",
                str(start + 250),
            ]
        )
        == 0
    )
    partial = _stdout_json(capsys)
    assert partial["counts_100ms"]["bin_count"] == 3
    assert partial["counts_100ms"]["fano"] == pytest.approx(1 / 3)
    assert partial["counts_100ms"]["partial_boundary_bins"] is True
    assert "partial boundary bins" in partial["counts_100ms"]["binning"]


def test_module_entrypoint_displays_help() -> None:
    project = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "-m", "glft_lab", "--help"],
        cwd=project,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "benchmark" in completed.stdout
    assert "arrival-diagnostics" in completed.stdout

    installed = Path(sys.executable).with_name("glft-lab")
    assert installed.is_file()
    installed_help = subprocess.run(
        [str(installed), "--help"],
        cwd=project,
        check=False,
        capture_output=True,
        text=True,
    )
    assert installed_help.returncode == 0
    assert "walk-forward" in installed_help.stdout
