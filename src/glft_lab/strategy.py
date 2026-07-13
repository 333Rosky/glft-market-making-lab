"""Closed-loop GLFT quoting policy and Binance-data replay adapter.

This module is the narrow bridge between the components of the lab:

* :mod:`glft_lab.data` provides an already merged causal market stream;
* :func:`glft_lab.glft.optimal_deltas` remains the single implementation of the
  paper's finite-horizon quotes;
* :mod:`glft_lab.replay` applies exchange constraints and empirical fills.

Binance ``bookTicker`` update ids and ``aggTrades`` ids belong to different sequence
domains.  ``market_rows_to_replay_events`` therefore replaces them with a synthetic
within-millisecond sequence that preserves the order produced by ``merge_market_rows``.
It must never compare the two raw ids.  Timestamps stay as Unix seconds (rather than
being shifted to zero), so later hazard features and calendar-month walk-forward splits
retain their time-of-day and month.  GLFT elapsed time is computed separately.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import Enum
from numbers import Integral

from .data import AggTradeRow, BookTickerRow, MarketRow
from .glft import GLFTParameters, QuoteDeltas, optimal_deltas
from .replay import (
    AccountingModel,
    BookEvent,
    EventDrivenReplay,
    QuoteEvent,
    ReplayConfig,
    ReplayResult,
    ReplayStrategyState,
    Side,
    TradeEvent,
    round_to_tick,
)

ReplayMarketEvent = BookEvent | TradeEvent


class TradeQuantityMode(str, Enum):
    """Unit carried by the input ``aggTrades.quantity`` column."""

    AS_IS = "as_is"
    BASE_ASSET = "base_asset"


@dataclass(frozen=True, slots=True)
class GLFTQuoteDecision:
    """Auditable transformation from an exact GLFT delta to exchange quotes."""

    timestamp: float
    elapsed: float
    inventory: float
    inventory_units: float
    grid_inventory: int
    mid_price: float
    deltas: QuoteDeltas | None
    raw_bid_price: float | None
    raw_ask_price: float | None
    quote: QuoteEvent


@dataclass(frozen=True, slots=True)
class GLFTReplayRun:
    """Replay output plus the decisions needed to audit the GLFT adapter."""

    replay: ReplayResult
    decisions: tuple[GLFTQuoteDecision, ...]
    parameters: GLFTParameters
    replay_config: ReplayConfig
    inventory_unit: float
    market_event_count: int
    start_timestamp: float
    end_timestamp: float
    horizon: float
    trade_quantity_mode: TradeQuantityMode = TradeQuantityMode.AS_IS


def market_rows_to_replay_events(
    rows: Iterable[MarketRow],
    *,
    trade_quantity_mode: TradeQuantityMode = TradeQuantityMode.AS_IS,
    contract_multiplier: float | None = None,
    quantity_step: float | None = None,
) -> Iterator[ReplayMarketEvent]:
    """Convert a causally merged Binance stream into replay events.

    ``rows`` must already be ordered, normally by :func:`merge_market_rows`.  The
    encounter order is authoritative for equal-millisecond cross-feed ties.  Unix
    milliseconds are converted to Unix seconds because replay latencies, markout
    horizons, and the GLFT parameter time unit are expressed in seconds.  Official
    COIN-M ``q`` values are contract counts and use ``AS_IS``.  ``BASE_ASSET`` is an
    explicit adapter for normalized third-party files and converts
    ``base_quantity * price / contract_multiplier``; it must never be inferred silently.
    Supplying ``quantity_step`` validates both BBO and trade quantities in contract units.
    """

    try:
        quantity_mode = TradeQuantityMode(trade_quantity_mode)
    except ValueError as exc:
        raise ValueError("invalid trade_quantity_mode") from exc
    if quantity_mode is TradeQuantityMode.BASE_ASSET:
        if contract_multiplier is None or not math.isfinite(contract_multiplier):
            raise ValueError("base_asset quantity mode requires contract_multiplier")
        if contract_multiplier <= 0.0:
            raise ValueError("contract_multiplier must be positive")
    if quantity_step is not None and (not math.isfinite(quantity_step) or quantity_step <= 0.0):
        raise ValueError("quantity_step must be positive and finite")

    previous_timestamp_ms: int | None = None
    within_timestamp_sequence = 0
    for row in rows:
        if not isinstance(row, (BookTickerRow, AggTradeRow)):
            raise TypeError(f"unsupported market row: {type(row).__name__}")
        timestamp_ms = row.timestamp_ms
        if previous_timestamp_ms is not None and timestamp_ms < previous_timestamp_ms:
            raise ValueError("market rows must be merged chronologically before replay conversion")
        if timestamp_ms == previous_timestamp_ms:
            within_timestamp_sequence += 1
        else:
            within_timestamp_sequence = 0
        previous_timestamp_ms = timestamp_ms
        timestamp = timestamp_ms / 1_000.0

        if isinstance(row, BookTickerRow):
            if quantity_step is not None:
                _require_quantity_step_multiple(
                    "book bid quantity", row.bid_quantity, quantity_step
                )
                _require_quantity_step_multiple(
                    "book ask quantity", row.ask_quantity, quantity_step
                )
            yield BookEvent(
                timestamp=timestamp,
                bid_price=row.bid_price,
                bid_quantity=row.bid_quantity,
                ask_price=row.ask_price,
                ask_quantity=row.ask_quantity,
                sequence=within_timestamp_sequence,
            )
        else:
            quantity = row.quantity
            if quantity_mode is TradeQuantityMode.BASE_ASSET:
                if contract_multiplier is None:  # pragma: no cover - validated above
                    raise AssertionError("contract multiplier is unavailable")
                quantity = row.quantity * row.price / contract_multiplier
            if quantity_step is not None:
                _require_quantity_step_multiple("trade quantity", quantity, quantity_step)
            yield TradeEvent(
                timestamp=timestamp,
                price=row.price,
                quantity=quantity,
                aggressor_side=Side(row.aggressor_side),
                sequence=within_timestamp_sequence,
            )


def inventory_to_grid(
    inventory: float,
    *,
    inventory_unit: float,
    max_inventory: int,
) -> int:
    """Map partial empirical inventory to the nearest GLFT state, half away from zero.

    GLFT's finite state is integer-valued while real maker orders can partially fill.
    The mapping is therefore explicit and auditable.  Values outside the theoretical
    bound are clipped to the boundary, which suppresses further risk-increasing quotes.
    """

    if not math.isfinite(inventory):
        raise ValueError("inventory must be finite")
    if not math.isfinite(inventory_unit) or inventory_unit <= 0.0:
        raise ValueError("inventory_unit must be positive and finite")
    if (
        isinstance(max_inventory, bool)
        or not isinstance(max_inventory, Integral)
        or max_inventory < 1
    ):
        raise ValueError("max_inventory must be a positive integer")

    units = inventory / inventory_unit
    rounded_magnitude = math.floor(abs(units) + 0.5)
    rounded = rounded_magnitude if units >= 0.0 else -rounded_magnitude
    return max(-max_inventory, min(max_inventory, rounded))


class GLFTQuotingPolicy:
    """Inventory-aware exact GLFT policy projected onto post-only exchange ticks.

    ``horizon`` and replay timestamps use seconds.  Model parameters must consequently
    use per-second units (``A`` in 1/s, ``sigma`` in price/sqrt(s), ``mu`` in price/s).
    Exact deltas come only from :func:`optimal_deltas`; this adapter then performs the
    unavoidable tick rounding and passive-price projection.  ``quote_interval`` throttles
    the matrix exponential on dense bookTicker streams.  Inventory-grid changes always
    force an immediate risk update.  Pass zero or ``None`` only to intentionally quote on
    every book update.
    """

    def __init__(
        self,
        parameters: GLFTParameters,
        *,
        horizon: float,
        tick_size: float,
        order_quantity: float = 1.0,
        inventory_unit: float | None = None,
        start_timestamp: float | None = None,
        quote_id: str = "glft",
        quote_interval: float | None = 1.0,
    ) -> None:
        if not math.isfinite(horizon) or horizon <= 0.0:
            raise ValueError("horizon must be positive and finite")
        if not math.isfinite(tick_size) or tick_size <= 0.0:
            raise ValueError("tick_size must be positive and finite")
        if not math.isfinite(order_quantity) or order_quantity <= 0.0:
            raise ValueError("order_quantity must be positive and finite")
        resolved_inventory_unit = order_quantity if inventory_unit is None else inventory_unit
        if not math.isfinite(resolved_inventory_unit) or resolved_inventory_unit <= 0.0:
            raise ValueError("inventory_unit must be positive and finite")
        if start_timestamp is not None and not math.isfinite(start_timestamp):
            raise ValueError("start_timestamp must be finite or None")
        if not quote_id:
            raise ValueError("quote_id must not be empty")
        if quote_interval is not None and (
            not math.isfinite(quote_interval) or quote_interval < 0.0
        ):
            raise ValueError("quote_interval must be non-negative and finite or None")

        self.parameters = parameters
        self.horizon = horizon
        self.tick_size = tick_size
        self.order_quantity = order_quantity
        self.inventory_unit = resolved_inventory_unit
        self.start_timestamp = start_timestamp
        self.quote_id = quote_id
        self.quote_interval = quote_interval
        self.decisions: list[GLFTQuoteDecision] = []
        self._last_quote_timestamp: float | None = None
        self._last_grid_inventory: int | None = None

    def on_book(self, state: ReplayStrategyState) -> QuoteEvent | None:
        """Quote from the latest causal book and inventory supplied by replay."""

        if self.start_timestamp is None:
            self.start_timestamp = state.timestamp
        elapsed = state.timestamp - self.start_timestamp
        if elapsed < -1e-12:
            raise ValueError("book timestamp precedes the GLFT policy start")

        inventory_units = state.inventory / self.inventory_unit
        grid_inventory = inventory_to_grid(
            state.inventory,
            inventory_unit=self.inventory_unit,
            max_inventory=self.parameters.max_inventory,
        )
        if not self._should_quote(state.timestamp, grid_inventory, elapsed):
            return None
        mid = (state.book.bid_price + state.book.ask_price) / 2.0

        deltas: QuoteDeltas | None = None
        raw_bid: float | None = None
        raw_ask: float | None = None
        bid: float | None = None
        ask: float | None = None
        if elapsed <= self.horizon + 1e-12:
            model_time = min(max(elapsed, 0.0), self.horizon)
            deltas = optimal_deltas(
                self.parameters,
                grid_inventory,
                model_time,
                self.horizon,
            )
            if deltas.bid_delta is not None:
                raw_bid = mid - deltas.bid_delta
                bid = self._post_only_price(raw_bid, Side.BUY, state.book)
            if deltas.ask_delta is not None:
                raw_ask = mid + deltas.ask_delta
                ask = self._post_only_price(raw_ask, Side.SELL, state.book)

        quote = QuoteEvent(
            timestamp=state.timestamp,
            bid_price=bid,
            bid_quantity=self.order_quantity if bid is not None else 0.0,
            ask_price=ask,
            ask_quantity=self.order_quantity if ask is not None else 0.0,
            quote_id=self.quote_id,
            sequence=state.book.sequence,
        )
        self.decisions.append(
            GLFTQuoteDecision(
                timestamp=state.timestamp,
                elapsed=max(0.0, elapsed),
                inventory=state.inventory,
                inventory_units=inventory_units,
                grid_inventory=grid_inventory,
                mid_price=mid,
                deltas=deltas,
                raw_bid_price=raw_bid,
                raw_ask_price=raw_ask,
                quote=quote,
            )
        )
        self._last_quote_timestamp = state.timestamp
        self._last_grid_inventory = grid_inventory
        return quote

    def _should_quote(
        self,
        timestamp: float,
        grid_inventory: int,
        elapsed: float,
    ) -> bool:
        if self._last_quote_timestamp is None:
            return True
        if elapsed >= self.horizon - 1e-12:
            if self.start_timestamp is None:  # pragma: no cover - set by on_book
                raise AssertionError("policy start is unavailable")
            cutoff = self.start_timestamp + self.horizon
            if elapsed <= self.horizon + 1e-12:
                return self._last_quote_timestamp < cutoff
            return self._last_quote_timestamp <= cutoff
        if grid_inventory != self._last_grid_inventory:
            return True
        if self.quote_interval in {None, 0.0}:
            return True
        return timestamp - self._last_quote_timestamp >= self.quote_interval - 1e-12

    def _post_only_price(
        self,
        raw_price: float,
        side: Side,
        book: BookEvent,
    ) -> float | None:
        if not math.isfinite(raw_price):
            raise FloatingPointError("GLFT produced a non-finite quote price")
        if raw_price <= 0.0:
            return None

        price = round_to_tick(raw_price, side, self.tick_size)
        if side is Side.BUY and price >= book.ask_price:
            passive_limit = book.ask_price - self.tick_size
            if passive_limit <= 0.0:
                return None
            price = round_to_tick(passive_limit, side, self.tick_size)
        elif side is Side.SELL and price <= book.bid_price:
            price = round_to_tick(book.bid_price + self.tick_size, side, self.tick_size)

        if (side is Side.BUY and price >= book.ask_price) or (
            side is Side.SELL and price <= book.bid_price
        ):
            raise AssertionError("passive tick projection crossed the observed book")
        return price


def run_glft_replay(
    rows: Iterable[MarketRow],
    *,
    parameters: GLFTParameters,
    replay_config: ReplayConfig,
    horizon: float | None = None,
    order_quantity: float = 1.0,
    inventory_unit: float | None = None,
    quote_id: str = "glft",
    quote_interval: float | None = 1.0,
    max_events: int = 1_000_000,
    trade_quantity_mode: TradeQuantityMode = TradeQuantityMode.AS_IS,
) -> GLFTReplayRun:
    """Convert a bounded merged stream and run the GLFT policy in closed loop.

    The current replay retains event/fill/equity histories for diagnostics, so this entry
    point intentionally rejects unbounded full-month input.  Select a time window with the
    data readers' ``start_ms``/``end_ms`` or ``max_rows`` arguments.  ``max_events`` is a
    hard guard, not a sampling rule.
    """

    if isinstance(max_events, bool) or not isinstance(max_events, Integral) or max_events < 1:
        raise ValueError("max_events must be a positive integer")
    try:
        quantity_mode = TradeQuantityMode(trade_quantity_mode)
    except ValueError as exc:
        raise ValueError("invalid trade_quantity_mode") from exc
    if replay_config.accounting_model is AccountingModel.INVERSE:
        step = replay_config.quantity_step
        if step is None:  # pragma: no cover - set by ReplayConfig
            raise AssertionError("inverse accounting has no quantity_step")
        resolved_inventory_unit = order_quantity if inventory_unit is None else inventory_unit
        _require_quantity_step_multiple("order_quantity", order_quantity, step)
        _require_quantity_step_multiple("inventory_unit", resolved_inventory_unit, step)
    market_events: list[BookEvent | TradeEvent | QuoteEvent] = []
    for event in market_rows_to_replay_events(
        rows,
        trade_quantity_mode=quantity_mode,
        contract_multiplier=replay_config.contract_multiplier,
        quantity_step=(
            replay_config.quantity_step
            if replay_config.accounting_model is AccountingModel.INVERSE
            else None
        ),
    ):
        if len(market_events) >= max_events:
            raise ValueError("bounded replay exceeded max_events; select a smaller data window")
        market_events.append(event)
    if not market_events:
        raise ValueError("cannot run GLFT replay on an empty market stream")
    first_book = next(
        (event for event in market_events if isinstance(event, BookEvent)),
        None,
    )
    if first_book is None:
        raise ValueError("GLFT replay requires at least one book event")

    market_event_count = len(market_events)
    start = first_book.timestamp
    end = market_events[-1].timestamp
    duration = end - start
    resolved_horizon = duration if horizon is None else horizon
    if not math.isfinite(resolved_horizon) or resolved_horizon <= 0.0:
        raise ValueError("horizon must be supplied when the stream has zero duration")

    policy = GLFTQuotingPolicy(
        parameters,
        horizon=resolved_horizon,
        tick_size=replay_config.tick_size,
        order_quantity=order_quantity,
        inventory_unit=inventory_unit,
        start_timestamp=start,
        quote_id=quote_id,
        quote_interval=quote_interval,
    )
    cutoff = start + resolved_horizon
    if cutoff < end - 1e-12:
        withdrawal = QuoteEvent(
            timestamp=cutoff,
            quote_id=quote_id,
            sequence=2**63 - 1,
        )
        insertion = bisect_right(
            market_events,
            cutoff,
            key=lambda event: event.timestamp,
        )
        market_events.insert(insertion, withdrawal)

    replay = EventDrivenReplay(replay_config).run(
        market_events,
        quote_policy=policy,
        events_are_sorted=True,
    )
    return GLFTReplayRun(
        replay=replay,
        decisions=tuple(policy.decisions),
        parameters=parameters,
        replay_config=replay_config,
        inventory_unit=policy.inventory_unit,
        market_event_count=market_event_count,
        start_timestamp=start,
        end_timestamp=end,
        horizon=resolved_horizon,
        trade_quantity_mode=quantity_mode,
    )


def _require_quantity_step_multiple(name: str, value: float, step: float) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be non-negative and finite")
    ratio = value / step
    if abs(ratio - round(ratio)) > 1e-9:
        raise ValueError(
            f"inverse {name} must be a quantity_step multiple; fractional contract "
            "units indicate a mismatched or normalized source"
        )


__all__ = [
    "GLFTQuoteDecision",
    "GLFTQuotingPolicy",
    "GLFTReplayRun",
    "ReplayMarketEvent",
    "TradeQuantityMode",
    "inventory_to_grid",
    "market_rows_to_replay_events",
    "run_glft_replay",
]
