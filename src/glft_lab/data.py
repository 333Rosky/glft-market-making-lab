"""Streaming readers for Binance Futures ``bookTicker`` and ``aggTrades`` files.

The raw files are intentionally kept outside Git under ``data/``.  The two
feeds do not share a sequence number, so equal-millisecond timestamps cannot be
ordered perfectly.  ``merge_market_rows`` uses a documented conservative tie
rule: trades are emitted before book updates at the same transaction time.
That prevents a trade from seeing the post-trade book update.
"""

from __future__ import annotations

import csv
import heapq
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias


@dataclass(frozen=True, slots=True)
class BookTickerRow:
    timestamp_ms: int
    sequence: int
    bid_price: float
    bid_quantity: float
    ask_price: float
    ask_quantity: float
    event_time_ms: int | None = None


@dataclass(frozen=True, slots=True)
class AggTradeRow:
    timestamp_ms: int
    sequence: int
    price: float
    quantity: float
    buyer_is_maker: bool
    first_trade_id: int | None = None
    last_trade_id: int | None = None

    @property
    def aggressor_side(self) -> Literal["buy", "sell"]:
        """Return the liquidity-taking side."""

        return "sell" if self.buyer_is_maker else "buy"


MarketRow: TypeAlias = BookTickerRow | AggTradeRow


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def _optional_int(value: str | None) -> int | None:
    return None if value in {None, ""} else int(value)


def iter_book_ticker(
    path: str | Path,
    *,
    start_ms: int | None = None,
    end_ms: int | None = None,
    max_rows: int | None = None,
) -> Iterator[BookTickerRow]:
    """Stream top-of-book rows without loading a multi-gigabyte file."""

    emitted = 0
    previous_key: tuple[int, int] | None = None
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            timestamp = int(row["transaction_time"])
            if start_ms is not None and timestamp < start_ms:
                continue
            if end_ms is not None and timestamp >= end_ms:
                break
            sequence = int(row["update_id"])
            key = (timestamp, sequence)
            if previous_key is not None and key < previous_key:
                raise ValueError(f"bookTicker is not sorted at {key} after {previous_key}")
            previous_key = key
            yield BookTickerRow(
                timestamp_ms=timestamp,
                sequence=sequence,
                bid_price=float(row["best_bid_price"]),
                bid_quantity=float(row["best_bid_qty"]),
                ask_price=float(row["best_ask_price"]),
                ask_quantity=float(row["best_ask_qty"]),
                event_time_ms=_optional_int(row.get("event_time")),
            )
            emitted += 1
            if max_rows is not None and emitted >= max_rows:
                return


def iter_agg_trades(
    path: str | Path,
    *,
    start_ms: int | None = None,
    end_ms: int | None = None,
    max_rows: int | None = None,
) -> Iterator[AggTradeRow]:
    """Stream aggregate trades and retain taker side and traded quantity."""

    emitted = 0
    previous_key: tuple[int, int] | None = None
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            timestamp = int(row["transact_time"])
            if start_ms is not None and timestamp < start_ms:
                continue
            if end_ms is not None and timestamp >= end_ms:
                break
            sequence = int(row["agg_trade_id"])
            key = (timestamp, sequence)
            if previous_key is not None and key < previous_key:
                raise ValueError(f"aggTrades is not sorted at {key} after {previous_key}")
            previous_key = key
            yield AggTradeRow(
                timestamp_ms=timestamp,
                sequence=sequence,
                price=float(row["price"]),
                quantity=float(row["quantity"]),
                buyer_is_maker=_parse_bool(row["is_buyer_maker"]),
                first_trade_id=_optional_int(row.get("first_trade_id")),
                last_trade_id=_optional_int(row.get("last_trade_id")),
            )
            emitted += 1
            if max_rows is not None and emitted >= max_rows:
                return


def merge_market_rows(
    books: Iterator[BookTickerRow],
    trades: Iterator[AggTradeRow],
) -> Iterator[MarketRow]:
    """Merge two sorted feeds using ``trade before book`` for timestamp ties."""

    def keyed(
        rows: Iterator[MarketRow], priority: int
    ) -> Iterator[tuple[int, int, int, MarketRow]]:
        for row in rows:
            yield row.timestamp_ms, priority, row.sequence, row

    yield from (
        item[3]
        for item in heapq.merge(
            keyed(trades, 0),
            keyed(books, 1),
        )
    )


def month_paths(data_dir: str | Path, symbol: str, month: str) -> tuple[Path, Path]:
    """Return and validate the expected book/trade files for one month."""

    root = Path(data_dir)
    book = root / f"{symbol}-bookTicker-{month}.csv"
    trades = root / f"{symbol}-aggTrades-{month}.csv"
    missing = [str(path) for path in (book, trades) if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing market data: " + ", ".join(missing))
    return book, trades
