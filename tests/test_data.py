from __future__ import annotations

import csv
from pathlib import Path

import pytest

from glft_lab.data import (
    AggTradeRow,
    BookTickerRow,
    iter_agg_trades,
    iter_book_ticker,
    merge_market_rows,
)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_readers_keep_side_volume_and_book_state(tmp_path: Path) -> None:
    book_path = tmp_path / "book.csv"
    trade_path = tmp_path / "trades.csv"
    _write_csv(
        book_path,
        [
            "update_id",
            "best_bid_price",
            "best_bid_qty",
            "best_ask_price",
            "best_ask_qty",
            "transaction_time",
            "event_time",
        ],
        [
            {
                "update_id": 4,
                "best_bid_price": 99.0,
                "best_bid_qty": 3.0,
                "best_ask_price": 101.0,
                "best_ask_qty": 5.0,
                "transaction_time": 1000,
                "event_time": 1001,
            }
        ],
    )
    _write_csv(
        trade_path,
        [
            "agg_trade_id",
            "price",
            "quantity",
            "first_trade_id",
            "last_trade_id",
            "transact_time",
            "is_buyer_maker",
        ],
        [
            {
                "agg_trade_id": 7,
                "price": 99.0,
                "quantity": 2.5,
                "first_trade_id": 11,
                "last_trade_id": 12,
                "transact_time": 1000,
                "is_buyer_maker": "true",
            }
        ],
    )

    book = next(iter_book_ticker(book_path))
    trade = next(iter_agg_trades(trade_path))
    assert book == BookTickerRow(1000, 4, 99.0, 3.0, 101.0, 5.0, 1001)
    assert trade == AggTradeRow(1000, 7, 99.0, 2.5, True, 11, 12)
    assert trade.aggressor_side == "sell"


def test_merge_processes_trade_before_post_trade_book_on_tie() -> None:
    book = BookTickerRow(1000, 2, 98.0, 1.0, 101.0, 1.0)
    trade = AggTradeRow(1000, 1, 99.0, 1.0, True)
    rows = list(merge_market_rows(iter([book]), iter([trade])))
    assert rows == [trade, book]


def test_reader_rejects_unsorted_input(tmp_path: Path) -> None:
    path = tmp_path / "book.csv"
    fields = [
        "update_id",
        "best_bid_price",
        "best_bid_qty",
        "best_ask_price",
        "best_ask_qty",
        "transaction_time",
        "event_time",
    ]
    row = {
        "best_bid_price": 99,
        "best_bid_qty": 1,
        "best_ask_price": 101,
        "best_ask_qty": 1,
        "event_time": 0,
    }
    _write_csv(
        path,
        fields,
        [
            {**row, "update_id": 2, "transaction_time": 1001},
            {**row, "update_id": 3, "transaction_time": 1000},
        ],
    )
    with pytest.raises(ValueError, match="not sorted"):
        list(iter_book_ticker(path))
