"""Create exposure-aware hazard observations from chronological market events.

The sampler places independent *counterfactual* passive orders at a fixed
cadence.  It is intended for fill-model research, not for P&L simulation.  Each
order is split into piecewise-constant exposure intervals so order age is known
at the beginning of the interval and never inferred from the eventual fill.

Only top-of-book quantities are available in ``bookTicker``.  Queue ahead is
therefore exact neither at the touch nor deeper in the book: it is initialized
from displayed BBO quantity at the touch, while deeper levels start at zero and
are explicitly flagged as approximate.  Aggressive trade volume is the only
mechanism allowed to deplete queue ahead; book-size drops are not silently
treated as fills.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from numbers import Integral
from typing import Literal

import numpy as np

from .data import AggTradeRow, BookTickerRow, MarketRow

Side = Literal["bid", "ask"]
Risk = Literal["none", "fill", "adverse_move", "cancel"]


def _validate_quantity_multiple(value: float, step: float | None, name: str) -> None:
    if step is None:
        return
    units = value / step
    if not math.isfinite(units):
        raise ValueError(f"{name} cannot be validated against quantity_step={step!r}")
    nearest = round(units)
    tolerance = max(1e-9, 8.0 * math.ulp(units))
    if abs(units - nearest) > tolerance:
        raise ValueError(f"{name} must be a multiple of quantity_step={step!r}")


@dataclass(frozen=True, slots=True)
class EpisodeConfig:
    tick_size: float
    order_quantity: float = 1.0
    distance_ticks: tuple[int, ...] = (0, 1, 2)
    decision_interval_ms: int = 60_000
    placement_latency_ms: int = 10
    horizon_ms: int = 5_000
    observation_interval_ms: int = 250
    adverse_move_ticks: float = 2.0
    state_ewma_alpha: float = 0.05
    quantity_step: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.tick_size) or self.tick_size <= 0:
            raise ValueError("tick_size must be finite and positive")
        if not math.isfinite(self.order_quantity) or self.order_quantity <= 0:
            raise ValueError("order_quantity must be finite and positive")
        if self.quantity_step is not None and (
            isinstance(self.quantity_step, bool)
            or not math.isfinite(self.quantity_step)
            or self.quantity_step <= 0
        ):
            raise ValueError("quantity_step must be None or finite and positive")
        _validate_quantity_multiple(self.order_quantity, self.quantity_step, "order_quantity")
        if not self.distance_ticks or any(
            isinstance(value, bool) or not isinstance(value, Integral) or value < 0
            for value in self.distance_ticks
        ):
            raise ValueError("distance_ticks must contain non-negative integers")
        if len(set(self.distance_ticks)) != len(self.distance_ticks):
            raise ValueError("distance_ticks cannot contain duplicates")
        for name in (
            "decision_interval_ms",
            "horizon_ms",
            "observation_interval_ms",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.placement_latency_ms, bool)
            or not isinstance(self.placement_latency_ms, Integral)
            or self.placement_latency_ms < 0
        ):
            raise ValueError("placement_latency_ms must be a non-negative integer")
        if not math.isfinite(self.adverse_move_ticks) or self.adverse_move_ticks <= 0:
            raise ValueError("adverse_move_ticks must be finite and positive")
        if not math.isfinite(self.state_ewma_alpha) or not 0 < self.state_ewma_alpha <= 1:
            raise ValueError("state_ewma_alpha must lie in (0, 1]")


@dataclass(frozen=True, slots=True)
class HazardInterval:
    """One causal exposure interval whose label is known at ``end_timestamp``.

    ``timestamp`` is a compatibility property for the label-availability time
    (the interval end), which is the safe timestamp for temporal splitting.
    Features describe state at ``start_timestamp``.
    """

    episode_id: int
    start_timestamp: int
    end_timestamp: int
    side: Side
    distance: float
    spread: float
    imbalance: float
    ofi: float
    volatility: float
    queue_ahead: float
    order_age: float
    time_of_day: float
    exposure: float
    count: int
    risk: Risk
    quote_price: float
    requested_distance_ticks: int
    filled_quantity: float
    queue_is_approximate: bool = True

    @property
    def timestamp(self) -> int:
        return self.end_timestamp


@dataclass(slots=True)
class _BookState:
    timestamp_ms: int
    bid_price: float
    bid_quantity: float
    ask_price: float
    ask_quantity: float
    ofi: float
    volatility: float

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid_price + self.ask_price)

    @property
    def spread(self) -> float:
        return self.ask_price - self.bid_price

    @property
    def imbalance(self) -> float:
        total = self.bid_quantity + self.ask_quantity
        return 0.0 if total <= 0 else (self.bid_quantity - self.ask_quantity) / total


@dataclass(frozen=True, slots=True)
class _PendingOrder:
    episode_id: int
    side: Side
    quote_price: float
    distance_ticks: int
    activate_after_ms: int


@dataclass(slots=True)
class _ActiveOrder:
    episode_id: int
    side: Side
    quote_price: float
    distance_ticks: int
    activated_ms: int
    expires_ms: int
    initial_mid: float
    queue_ahead: float
    remaining_quantity: float
    interval_start_ms: int
    snapshot: dict[str, float] = field(default_factory=dict)


def _round_to_tick(price: float, tick: float, side: Side) -> float:
    scaled = price / tick
    units = math.floor(scaled + 1e-10) if side == "bid" else math.ceil(scaled - 1e-10)
    return units * tick


def _state_snapshot(
    state: _BookState,
    order: _ActiveOrder,
    config: EpisodeConfig,
) -> dict[str, float]:
    if order.side == "bid":
        distance = (state.bid_price - order.quote_price) / config.tick_size
    else:
        distance = (order.quote_price - state.ask_price) / config.tick_size
    return {
        # A counterfactual order inside the historical touch would become the
        # new touch, hence distance zero rather than a negative value.
        "distance": max(float(distance), 0.0),
        "spread": state.spread / config.tick_size,
        "imbalance": state.imbalance,
        "ofi": state.ofi,
        "volatility": state.volatility,
        "queue_ahead": max(order.queue_ahead, 0.0),
        "order_age": max(order.interval_start_ms - order.activated_ms, 0) / 1000.0,
        "time_of_day": (order.interval_start_ms // 1000) % 86_400,
    }


def _interval(
    order: _ActiveOrder,
    end_ms: int,
    *,
    count: int,
    risk: Risk,
    filled_quantity: float = 0.0,
) -> HazardInterval:
    exposure_ms = end_ms - order.interval_start_ms
    if exposure_ms <= 0:
        raise ValueError("hazard intervals must have strictly positive exposure")
    return HazardInterval(
        episode_id=order.episode_id,
        start_timestamp=order.interval_start_ms,
        end_timestamp=end_ms,
        side=order.side,
        exposure=exposure_ms / 1000.0,
        count=int(count),
        risk=risk,
        quote_price=order.quote_price,
        requested_distance_ticks=order.distance_ticks,
        filled_quantity=float(filled_quantity),
        **order.snapshot,
    )


def _book_update(
    previous: _BookState | None,
    row: BookTickerRow,
    alpha: float,
) -> _BookState:
    if previous is None:
        return _BookState(
            row.timestamp_ms,
            row.bid_price,
            row.bid_quantity,
            row.ask_price,
            row.ask_quantity,
            0.0,
            0.0,
        )

    raw_ofi = (
        (row.bid_quantity if row.bid_price >= previous.bid_price else 0.0)
        - (previous.bid_quantity if row.bid_price <= previous.bid_price else 0.0)
        - (row.ask_quantity if row.ask_price <= previous.ask_price else 0.0)
        + (previous.ask_quantity if row.ask_price >= previous.ask_price else 0.0)
    )
    depth = max(
        row.bid_quantity + row.ask_quantity + previous.bid_quantity + previous.ask_quantity,
        np.finfo(float).eps,
    )
    normalized_ofi = 2.0 * raw_ofi / depth
    ofi = (1.0 - alpha) * previous.ofi + alpha * normalized_ofi

    elapsed = max((row.timestamp_ms - previous.timestamp_ms) / 1000.0, 1e-3)
    previous_mid = previous.mid
    mid = 0.5 * (row.bid_price + row.ask_price)
    move_bps_per_sqrt_second = (mid - previous_mid) / previous_mid * 10_000.0 / math.sqrt(elapsed)
    variance = (1.0 - alpha) * previous.volatility**2 + alpha * move_bps_per_sqrt_second**2
    return _BookState(
        row.timestamp_ms,
        row.bid_price,
        row.bid_quantity,
        row.ask_price,
        row.ask_quantity,
        ofi,
        math.sqrt(max(variance, 0.0)),
    )


def _is_post_only(order: _PendingOrder, state: _BookState, tick_size: float) -> bool:
    tolerance = tick_size * 1e-6
    if order.side == "bid":
        return order.quote_price < state.ask_price - tolerance
    return order.quote_price > state.bid_price + tolerance


def _activate(
    pending: _PendingOrder,
    timestamp_ms: int,
    state: _BookState,
    config: EpisodeConfig,
) -> _ActiveOrder | None:
    if not _is_post_only(pending, state, config.tick_size):
        return None
    if pending.side == "bid" and math.isclose(
        pending.quote_price,
        state.bid_price,
        rel_tol=0.0,
        abs_tol=config.tick_size * 1e-6,
    ):
        queue = state.bid_quantity
    elif pending.side == "ask" and math.isclose(
        pending.quote_price,
        state.ask_price,
        rel_tol=0.0,
        abs_tol=config.tick_size * 1e-6,
    ):
        queue = state.ask_quantity
    else:
        queue = 0.0
    order = _ActiveOrder(
        episode_id=pending.episode_id,
        side=pending.side,
        quote_price=pending.quote_price,
        distance_ticks=pending.distance_ticks,
        activated_ms=timestamp_ms,
        expires_ms=timestamp_ms + config.horizon_ms,
        initial_mid=state.mid,
        queue_ahead=queue,
        remaining_quantity=config.order_quantity,
        interval_start_ms=timestamp_ms,
    )
    order.snapshot = _state_snapshot(state, order, config)
    return order


def _schedule_orders(
    state: _BookState,
    config: EpisodeConfig,
    decision_ms: int,
    episode_ids: Iterator[int],
) -> list[_PendingOrder]:
    orders: list[_PendingOrder] = []
    for distance in config.distance_ticks:
        orders.append(
            _PendingOrder(
                episode_id=next(episode_ids),
                side="bid",
                quote_price=_round_to_tick(
                    state.bid_price - distance * config.tick_size,
                    config.tick_size,
                    "bid",
                ),
                distance_ticks=distance,
                activate_after_ms=decision_ms + config.placement_latency_ms,
            )
        )
        orders.append(
            _PendingOrder(
                episode_id=next(episode_ids),
                side="ask",
                quote_price=_round_to_tick(
                    state.ask_price + distance * config.tick_size,
                    config.tick_size,
                    "ask",
                ),
                distance_ticks=distance,
                activate_after_ms=decision_ms + config.placement_latency_ms,
            )
        )
    return orders


def _matches(order: _ActiveOrder, trade: AggTradeRow, tick_size: float) -> bool:
    tolerance = tick_size * 1e-6
    if order.side == "bid":
        return trade.buyer_is_maker and trade.price <= order.quote_price + tolerance
    return not trade.buyer_is_maker and trade.price >= order.quote_price - tolerance


def _trades_through(order: _ActiveOrder, trade: AggTradeRow, tick_size: float) -> bool:
    """Whether the tape traded beyond the order's limit price."""

    tolerance = tick_size * 1e-6
    if order.side == "bid":
        return trade.price < order.quote_price - tolerance
    return trade.price > order.quote_price + tolerance


def _rows_by_timestamp(rows: Iterable[MarketRow]) -> Iterator[tuple[int, list[MarketRow]]]:
    previous_timestamp: int | None = None
    for timestamp, group in itertools.groupby(rows, key=lambda row: row.timestamp_ms):
        if previous_timestamp is not None and timestamp < previous_timestamp:
            raise ValueError("market rows must be chronological")
        previous_timestamp = timestamp
        yield timestamp, list(group)


def _validate_market_group(group: Iterable[MarketRow], quantity_step: float | None) -> None:
    """Reject an invalid timestamp group before any queue state can mutate."""

    for row in group:
        if isinstance(row, BookTickerRow):
            values = (row.bid_price, row.bid_quantity, row.ask_price, row.ask_quantity)
            if not all(math.isfinite(value) for value in values):
                raise ValueError("bookTicker contains a non-finite price or quantity")
            if row.bid_price <= 0 or row.ask_price <= row.bid_price:
                raise ValueError("bookTicker contains an invalid or crossed BBO")
            if row.bid_quantity < 0 or row.ask_quantity < 0:
                raise ValueError("bookTicker contains a negative quantity")
            _validate_quantity_multiple(row.bid_quantity, quantity_step, "bookTicker bid_quantity")
            _validate_quantity_multiple(row.ask_quantity, quantity_step, "bookTicker ask_quantity")
        else:
            if (
                not math.isfinite(row.price)
                or not math.isfinite(row.quantity)
                or row.price <= 0
                or row.quantity <= 0
            ):
                raise ValueError("aggTrades contains a non-positive or non-finite value")
            _validate_quantity_multiple(row.quantity, quantity_step, "aggTrades quantity")


def build_hazard_intervals(
    rows: Iterable[MarketRow],
    config: EpisodeConfig,
) -> Iterator[HazardInterval]:
    """Yield leakage-safe, time-to-first-fill observations.

    Placement acknowledgements, observation boundaries, decision times and
    cancellations run on their requested clock even when no market row occurs
    at that exact millisecond.  At equal timestamps, market events are handled
    before cancellation or activation, so a trade at the cancel horizon can
    fill an already-active order while a trade at the acknowledgement time
    cannot fill the newly acknowledged order.

    The first positive own fill terminates an episode; ``filled_quantity`` can
    therefore be smaller than ``order_quantity``.  This is intentional for a
    fill-hazard target and is not a full execution/P&L simulator.
    """

    state: _BookState | None = None
    pending: list[_PendingOrder] = []
    active: list[_ActiveOrder] = []
    next_decision_ms: int | None = None
    last_timestamp: int | None = None
    episode_ids = itertools.count()

    for timestamp, group in _rows_by_timestamp(rows):
        _validate_market_group(group, config.quantity_step)
        last_timestamp = timestamp

        # Advance every non-market timer strictly before this market timestamp.
        # Equality is deliberately excluded because market events win ties.
        while state is not None:
            timer_candidates = [
                order.activate_after_ms for order in pending if order.activate_after_ms < timestamp
            ]
            timer_candidates.extend(
                order.expires_ms for order in active if order.expires_ms < timestamp
            )
            timer_candidates.extend(
                order.interval_start_ms + config.observation_interval_ms
                for order in active
                if order.interval_start_ms + config.observation_interval_ms < timestamp
            )
            if next_decision_ms is not None and next_decision_ms < timestamp:
                timer_candidates.append(next_decision_ms)
            if not timer_candidates:
                break
            timer = min(timer_candidates)

            # Expiration is terminal and therefore wins a tied bin boundary.
            for order in list(active):
                if order.expires_ms == timer:
                    yield _interval(order, timer, count=0, risk="cancel")
                    active.remove(order)

            remaining_pending: list[_PendingOrder] = []
            for order in pending:
                if order.activate_after_ms == timer:
                    activated = _activate(order, timer, state, config)
                    if activated is not None:
                        active.append(activated)
                else:
                    remaining_pending.append(order)
            pending = remaining_pending

            for order in active:
                boundary = order.interval_start_ms + config.observation_interval_ms
                if boundary == timer:
                    yield _interval(order, timer, count=0, risk="none")
                    order.interval_start_ms = timer
                    order.snapshot = _state_snapshot(state, order, config)

            if next_decision_ms == timer:
                new_orders = _schedule_orders(state, config, timer, episode_ids)
                next_decision_ms += config.decision_interval_ms
                if config.placement_latency_ms == 0:
                    for order in new_orders:
                        activated = _activate(order, timer, state, config)
                        if activated is not None:
                            active.append(activated)
                else:
                    pending.extend(new_orders)

        # Trades at a tied timestamp consume the pre-update book and active queue.
        trades = sorted(
            (row for row in group if isinstance(row, AggTradeRow)),
            key=lambda row: row.sequence,
        )
        for trade in trades:
            for order in list(active):
                if not _matches(order, trade, config.tick_size):
                    continue
                # A print beyond our price proves the tape traded through the
                # resting limit; volume printed at that worse level alone would
                # otherwise understate the quantity already consumed at ours.
                available = (
                    order.queue_ahead + order.remaining_quantity
                    if _trades_through(order, trade, config.tick_size)
                    else trade.quantity
                )
                queue_consumed = min(order.queue_ahead, available)
                order.queue_ahead -= queue_consumed
                available -= queue_consumed
                own_fill = min(order.remaining_quantity, available)
                if own_fill <= 0:
                    continue
                order.remaining_quantity -= own_fill
                yield _interval(
                    order,
                    timestamp,
                    count=1,
                    risk="fill",
                    filled_quantity=own_fill,
                )
                active.remove(order)

        # Apply every BBO update, preserving sequence within the feed.
        for book in sorted(
            (row for row in group if isinstance(row, BookTickerRow)),
            key=lambda row: row.sequence,
        ):
            state = _book_update(state, book, config.state_ewma_alpha)

        if state is None:
            continue

        # A price move competes with fill and is evaluated after same-time trades.
        adverse_distance = config.adverse_move_ticks * config.tick_size
        for order in list(active):
            adverse = (
                state.mid <= order.initial_mid - adverse_distance
                if order.side == "bid"
                else state.mid >= order.initial_mid + adverse_distance
            )
            if adverse:
                yield _interval(order, timestamp, count=0, risk="adverse_move")
                active.remove(order)

        # Market events win ties with the requested cancel horizon.  The
        # interval ends at the requested expiry, never at a later feed row.
        for order in list(active):
            if order.expires_ms == timestamp:
                yield _interval(order, timestamp, count=0, risk="cancel")
                active.remove(order)

        # Activation happens after market events at the acknowledgement timestamp.
        remaining_pending: list[_PendingOrder] = []
        for order in pending:
            if order.activate_after_ms == timestamp:
                activated = _activate(order, timestamp, state, config)
                if activated is not None:
                    active.append(activated)
            else:
                remaining_pending.append(order)
        pending = remaining_pending

        # A bin ending at this timestamp receives any same-time terminal event;
        # only surviving orders open a fresh feature snapshot afterwards.
        for order in active:
            boundary = order.interval_start_ms + config.observation_interval_ms
            if boundary == timestamp:
                yield _interval(order, timestamp, count=0, risk="none")
                order.interval_start_ms = timestamp
                order.snapshot = _state_snapshot(state, order, config)

        if next_decision_ms is None:
            next_decision_ms = timestamp
        if next_decision_ms == timestamp:
            new_orders = _schedule_orders(state, config, timestamp, episode_ids)
            next_decision_ms += config.decision_interval_ms
            if config.placement_latency_ms == 0:
                for order in new_orders:
                    activated = _activate(order, timestamp, state, config)
                    if activated is not None:
                        active.append(activated)
            else:
                pending.extend(new_orders)

    # End-of-file is right censoring, not a synthetic fill or cancel.
    if last_timestamp is not None:
        for order in active:
            if order.interval_start_ms < last_timestamp:
                yield _interval(order, last_timestamp, count=0, risk="none")


def intervals_to_columns(intervals: Iterable[HazardInterval]) -> dict[str, np.ndarray]:
    """Convert observations to stable-dtype hazard/walk-forward columns.

    The canonical ``timestamp`` is the interval end, when its outcome becomes
    observable.  Splitting on the feature-snapshot/start timestamp could leak a
    next-month fill or cancellation into the preceding month's training set.
    """

    rows = list(intervals)
    float_names = (
        "distance",
        "spread",
        "imbalance",
        "ofi",
        "volatility",
        "queue_ahead",
        "order_age",
        "time_of_day",
        "exposure",
        "quote_price",
        "filled_quantity",
    )
    result = {
        name: np.asarray([getattr(row, name) for row in rows], dtype=np.float64)
        for name in float_names
    }
    result["start_timestamp"] = np.asarray([row.start_timestamp for row in rows], dtype=np.int64)
    result["end_timestamp"] = np.asarray([row.end_timestamp for row in rows], dtype=np.int64)
    result["timestamp"] = result["end_timestamp"].copy()
    result["episode_id"] = np.asarray([row.episode_id for row in rows], dtype=np.int64)
    result["count"] = np.asarray([row.count for row in rows], dtype=np.int64)
    result["requested_distance_ticks"] = np.asarray(
        [row.requested_distance_ticks for row in rows], dtype=np.int64
    )
    result["side"] = np.asarray([row.side for row in rows], dtype="U3")
    result["risk"] = np.asarray([row.risk for row in rows], dtype="U12")
    result["queue_is_approximate"] = np.asarray(
        [row.queue_is_approximate for row in rows], dtype=np.bool_
    )
    return result


__all__ = [
    "EpisodeConfig",
    "HazardInterval",
    "build_hazard_intervals",
    "intervals_to_columns",
]
