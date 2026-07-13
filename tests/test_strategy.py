from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from glft_lab.data import AggTradeRow, BookTickerRow, merge_market_rows
from glft_lab.glft import GLFTParameters, optimal_deltas
from glft_lab.hazard import HazardFeatureBuilder
from glft_lab.replay import (
    AccountingModel,
    BookEvent,
    EventDrivenReplay,
    QuoteEvent,
    ReplayConfig,
    ReplayStrategyState,
    Side,
    TradeEvent,
    round_to_tick,
)
from glft_lab.strategy import (
    GLFTQuotingPolicy,
    TradeQuantityMode,
    inventory_to_grid,
    market_rows_to_replay_events,
    run_glft_replay,
)
from glft_lab.walkforward import monthly_splits


def parameters(*, max_inventory: int = 2, mu: float = 0.0) -> GLFTParameters:
    return GLFTParameters(
        A=2.0,
        k=1.0,
        gamma=0.1,
        sigma=0.2,
        max_inventory=max_inventory,
        mu=mu,
    )


def replay_state(
    *,
    timestamp: float = 10.0,
    inventory: float = 0.0,
    bid: float = 90.0,
    ask: float = 110.0,
) -> ReplayStrategyState:
    return ReplayStrategyState(
        timestamp=timestamp,
        book=BookEvent(
            timestamp=timestamp,
            bid_price=bid,
            bid_quantity=1.0,
            ask_price=ask,
            ask_quantity=1.0,
        ),
        cash=0.0,
        inventory=inventory,
    )


def test_policy_uses_exact_glft_deltas_then_side_aware_tick_rounding() -> None:
    model = parameters()
    policy = GLFTQuotingPolicy(
        model,
        horizon=2.0,
        tick_size=0.1,
        start_timestamp=10.0,
    )

    quote = policy.on_book(replay_state())
    decision = policy.decisions[-1]
    expected = optimal_deltas(model, inventory=0, t=0.0, horizon=2.0)

    assert decision.deltas == expected
    assert decision.raw_bid_price == pytest.approx(100.0 - expected.bid_delta)
    assert decision.raw_ask_price == pytest.approx(100.0 + expected.ask_delta)
    assert quote.bid_price == round_to_tick(decision.raw_bid_price, Side.BUY, policy.tick_size)
    assert quote.ask_price == round_to_tick(decision.raw_ask_price, Side.SELL, policy.tick_size)
    assert quote.bid_price < 110.0
    assert quote.ask_price > 90.0

    terminal_quote = policy.on_book(replay_state(timestamp=12.0))
    terminal = policy.decisions[-1]
    assert terminal_quote is not None
    assert terminal.deltas == optimal_deltas(model, inventory=0, t=2.0, horizon=2.0)


def test_negative_glft_delta_is_projected_to_a_valid_post_only_tick() -> None:
    model = parameters(mu=1.0)
    policy = GLFTQuotingPolicy(
        model,
        horizon=2.0,
        tick_size=0.1,
        start_timestamp=0.0,
    )

    quote = policy.on_book(replay_state(timestamp=0.0, bid=99.9, ask=100.1))
    decision = policy.decisions[-1]

    assert decision.deltas.bid_delta < 0.0
    assert decision.raw_bid_price > decision.mid_price
    assert quote.bid_price == pytest.approx(100.0)
    assert quote.bid_price < 100.1
    assert quote.ask_price > 99.9
    assert quote.bid_price / 0.1 == pytest.approx(round(quote.bid_price / 0.1))


def test_quote_interval_throttles_books_but_inventory_change_forces_recalculation() -> None:
    policy = GLFTQuotingPolicy(
        parameters(),
        horizon=5.0,
        tick_size=0.1,
        start_timestamp=0.0,
        quote_interval=1.0,
    )

    assert policy.on_book(replay_state(timestamp=0.0)) is not None
    assert policy.on_book(replay_state(timestamp=0.2)) is None
    # A partial fill crossing the explicit nearest-state threshold is a risk update,
    # even though the wall-clock interval has not elapsed.
    assert policy.on_book(replay_state(timestamp=0.3, inventory=0.6)) is not None
    assert policy.decisions[-1].grid_inventory == 1
    assert policy.on_book(replay_state(timestamp=0.9, inventory=0.6)) is None
    assert policy.on_book(replay_state(timestamp=1.3, inventory=0.6)) is not None
    assert [decision.timestamp for decision in policy.decisions] == [0.0, 0.3, 1.3]


@pytest.mark.parametrize(
    ("inventory", "expected"),
    [
        (0.49, 0),
        (0.5, 1),
        (-0.5, -1),
        (9.0, 2),
        (-9.0, -2),
    ],
)
def test_partial_inventory_mapping_is_explicit_and_bounded(inventory: float, expected: int) -> None:
    assert (
        inventory_to_grid(
            inventory,
            inventory_unit=1.0,
            max_inventory=2,
        )
        == expected
    )


def test_adapter_preserves_merged_trade_before_book_with_incomparable_ids() -> None:
    timestamp_ms = 1_725_148_800_000
    trade = AggTradeRow(
        timestamp_ms=timestamp_ms,
        sequence=9_000_000,
        price=100.0,
        quantity=1.0,
        buyer_is_maker=True,
    )
    book = BookTickerRow(
        timestamp_ms=timestamp_ms,
        sequence=1,
        bid_price=99.0,
        bid_quantity=1.0,
        ask_price=101.0,
        ask_quantity=1.0,
    )

    merged = merge_market_rows(iter([book]), iter([trade]))
    converted = list(market_rows_to_replay_events(merged))

    assert [type(event) for event in converted] == [TradeEvent, BookEvent]
    assert [event.sequence for event in converted] == [0, 1]
    assert converted[0].timestamp == timestamp_ms / 1_000.0


def test_base_asset_trade_quantity_conversion_is_explicit() -> None:
    row = AggTradeRow(
        timestamp_ms=1_725_148_800_000,
        sequence=1,
        price=58_941.9,
        quantity=0.224,
        buyer_is_maker=True,
    )

    converted = next(
        market_rows_to_replay_events(
            [row],
            trade_quantity_mode=TradeQuantityMode.BASE_ASSET,
            contract_multiplier=100.0,
        )
    )

    assert isinstance(converted, TradeEvent)
    assert converted.quantity == pytest.approx(0.224 * 58_941.9 / 100.0)


def test_inverse_runner_rejects_fractional_as_is_coin_m_quantity() -> None:
    rows = [
        BookTickerRow(1_000, 1, 100.0, 10.0, 101.0, 10.0),
        AggTradeRow(1_500, 2, 100.0, 0.224, True),
        BookTickerRow(2_000, 3, 100.0, 10.0, 101.0, 10.0),
    ]
    config = ReplayConfig(
        tick_size=1.0,
        accounting_model=AccountingModel.INVERSE,
        contract_multiplier=100.0,
        liquidate_at_end=False,
    )

    with pytest.raises(ValueError, match="trade quantity.*quantity_step"):
        run_glft_replay(
            rows,
            parameters=parameters(),
            replay_config=config,
            trade_quantity_mode=TradeQuantityMode.AS_IS,
        )


def test_inverse_runner_accepts_official_integer_contract_quantity() -> None:
    rows = [
        BookTickerRow(1_000, 1, 90.0, 10.0, 110.0, 10.0),
        AggTradeRow(1_500, 2, 100.0, 3.0, True),
        BookTickerRow(2_000, 3, 90.0, 10.0, 110.0, 10.0),
    ]
    run = run_glft_replay(
        rows,
        parameters=parameters(),
        replay_config=ReplayConfig(
            tick_size=1.0,
            accounting_model=AccountingModel.INVERSE,
            contract_multiplier=100.0,
            liquidate_at_end=False,
        ),
        trade_quantity_mode=TradeQuantityMode.AS_IS,
    )

    assert run.trade_quantity_mode is TradeQuantityMode.AS_IS
    assert run.replay.accounting_model is AccountingModel.INVERSE
    assert run.replay.contract_multiplier == 100.0
    assert run.replay.quantity_step == 1.0


@pytest.mark.parametrize(
    "quantity_kwargs",
    [
        {"order_quantity": 0.5},
        {"inventory_unit": 0.5},
    ],
)
def test_inverse_runner_rejects_fractional_strategy_contract_units(
    quantity_kwargs: dict[str, float],
) -> None:
    rows = [
        BookTickerRow(1_000, 1, 90.0, 10.0, 110.0, 10.0),
        BookTickerRow(2_000, 2, 90.0, 10.0, 110.0, 10.0),
    ]

    with pytest.raises(ValueError, match="quantity_step multiple"):
        run_glft_replay(
            rows,
            parameters=parameters(),
            replay_config=ReplayConfig(
                tick_size=1.0,
                accounting_model=AccountingModel.INVERSE,
                contract_multiplier=100.0,
                liquidate_at_end=False,
            ),
            **quantity_kwargs,
        )


def test_absolute_seconds_remain_compatible_with_calendar_walk_forward() -> None:
    rows = [
        BookTickerRow(1_704_067_200_000, 1, 99.0, 1.0, 101.0, 1.0),
        BookTickerRow(1_706_745_600_000, 2, 99.0, 1.0, 101.0, 1.0),
        BookTickerRow(1_709_251_200_000, 3, 99.0, 1.0, 101.0, 1.0),
    ]
    timestamps = np.array([event.timestamp for event in market_rows_to_replay_events(rows)])

    splits = monthly_splits(timestamps, min_train_months=1)

    assert [str(split.test_month) for split in splits] == ["2024-02", "2024-03"]

    noon = next(
        market_rows_to_replay_events([BookTickerRow(1_704_110_400_000, 4, 99.0, 1.0, 101.0, 1.0)])
    )
    hazard_state = {
        "distance": [1.0],
        "spread": [2.0],
        "imbalance": [0.0],
        "ofi": [0.0],
        "volatility": [0.1],
        "queue_ahead": [1.0],
        "order_age": [0.0],
        "timestamp": [noon.timestamp],
        "side": ["bid"],
    }
    raw = HazardFeatureBuilder.raw_matrix(hazard_state)
    assert raw[0, -2] == pytest.approx(0.0, abs=1e-12)
    assert raw[0, -1] == pytest.approx(-1.0)


def test_closed_loop_replay_quotes_again_from_inventory_after_same_time_fill() -> None:
    model = parameters(max_inventory=1)
    base_ms = 1_725_148_800_000
    horizon = 2.0
    initial_delta = optimal_deltas(model, inventory=0, t=0.0, horizon=horizon)
    initial_bid = round_to_tick(100.0 - initial_delta.bid_delta, Side.BUY, 0.1)

    books = iter(
        [
            BookTickerRow(base_ms - 1_000, 100, 90.0, 1.0, 110.0, 1.0),
            # Its raw update id is lower than the aggTrade id, but cross-feed ids
            # are incomparable: merge order, not numeric id, must win.
            BookTickerRow(base_ms, 101, 90.0, 1.0, 110.0, 1.0),
        ]
    )
    trades = iter(
        [
            AggTradeRow(
                base_ms,
                9_999_999,
                initial_bid,
                1.0,
                True,
            )
        ]
    )

    run = run_glft_replay(
        merge_market_rows(books, trades),
        parameters=model,
        replay_config=ReplayConfig(
            tick_size=0.1,
            liquidate_at_end=False,
            markout_horizons=(),
        ),
        horizon=horizon,
    )

    assert run.market_event_count == 3
    assert len(run.decisions) == 2
    first, second = run.decisions
    assert first.grid_inventory == 0
    assert first.deltas == initial_delta
    assert second.inventory == pytest.approx(1.0)
    assert second.grid_inventory == 1
    assert second.deltas == optimal_deltas(model, 1, 1.0, horizon)
    assert second.quote.bid_price is None  # upper GLFT inventory boundary
    assert run.replay.fills[0].side is Side.BUY
    assert run.replay.fills[0].price == initial_bid
    at_end = [
        event.event_type
        for event in run.replay.processed_events
        if event.timestamp == base_ms / 1_000.0 and event.event_type in {"trade", "book"}
    ]
    assert at_end == ["trade", "book"]


def test_policy_horizon_starts_at_first_book_and_withdraws_before_later_trade() -> None:
    model = parameters(max_inventory=1)
    base_ms = 1_725_148_800_000
    horizon = 0.5
    delta = optimal_deltas(model, inventory=0, t=0.0, horizon=horizon)
    bid = round_to_tick(100.0 - delta.bid_delta, Side.BUY, 0.1)
    rows = [
        # This trade precedes the first observable book and must not consume horizon.
        AggTradeRow(base_ms - 1_000, 1, 100.0, 1.0, True),
        BookTickerRow(base_ms, 2, 90.0, 0.0, 110.0, 0.0),
        # The explicit cutoff at +0.5 s must cancel before this otherwise fillable print.
        AggTradeRow(base_ms + 750, 3, bid, 1.0, True),
        BookTickerRow(base_ms + 1_000, 4, 90.0, 0.0, 110.0, 0.0),
    ]

    run = run_glft_replay(
        rows,
        parameters=model,
        replay_config=ReplayConfig(
            tick_size=0.1,
            liquidate_at_end=False,
            markout_horizons=(),
        ),
        horizon=horizon,
    )

    assert run.start_timestamp == base_ms / 1_000.0
    assert run.decisions[0].elapsed == 0.0
    assert run.decisions[0].deltas == delta
    assert run.replay.fills == ()
    cancel_acks = [
        event for event in run.replay.processed_events if event.event_type == "cancel_ack"
    ]
    assert cancel_acks
    assert all(event.timestamp == base_ms / 1_000.0 + horizon for event in cancel_acks)


def test_bounded_runner_rejects_an_oversized_window() -> None:
    rows = [
        BookTickerRow(1_000, 1, 90.0, 1.0, 110.0, 1.0),
        BookTickerRow(2_000, 2, 90.0, 1.0, 110.0, 1.0),
        BookTickerRow(3_000, 3, 90.0, 1.0, 110.0, 1.0),
    ]
    with pytest.raises(ValueError, match="smaller data window"):
        run_glft_replay(
            rows,
            parameters=parameters(),
            replay_config=ReplayConfig(tick_size=0.1),
            max_events=2,
        )


@dataclass
class RecordingPolicy:
    states: list[ReplayStrategyState] = field(default_factory=list)

    def on_book(self, state: ReplayStrategyState) -> QuoteEvent:
        self.states.append(state)
        return QuoteEvent(
            timestamp=state.timestamp,
            bid_price=100.0,
            bid_quantity=1.0,
            quote_id="recording",
        )


def test_replay_hook_receives_only_current_state_and_rejects_wrong_timestamp() -> None:
    policy = RecordingPolicy()
    replay = EventDrivenReplay(ReplayConfig(tick_size=0.1, liquidate_at_end=False))
    result = replay.run(
        [
            BookEvent(
                timestamp=0.0,
                bid_price=100.0,
                bid_quantity=0.0,
                ask_price=101.0,
                ask_quantity=0.0,
            ),
            TradeEvent(
                timestamp=1.0,
                price=100.0,
                quantity=1.0,
                aggressor_side=Side.SELL,
                sequence=0,
            ),
            BookEvent(
                timestamp=1.0,
                bid_price=99.0,
                bid_quantity=0.0,
                ask_price=101.0,
                ask_quantity=0.0,
                sequence=1,
            ),
        ],
        quote_policy=policy,
    )

    assert [state.inventory for state in policy.states] == [0.0, 1.0]
    assert result.fills[0].timestamp == 1.0

    class WrongTimestampPolicy:
        def on_book(self, state: ReplayStrategyState) -> QuoteEvent:
            return QuoteEvent(timestamp=state.timestamp + 1.0)

    with pytest.raises(ValueError, match="timestamp must equal"):
        replay.run(
            [
                BookEvent(
                    timestamp=0.0,
                    bid_price=100.0,
                    bid_quantity=0.0,
                    ask_price=101.0,
                    ask_quantity=0.0,
                )
            ],
            quote_policy=WrongTimestampPolicy(),
        )


def test_adapter_rejects_unmerged_reverse_chronology() -> None:
    rows = [
        BookTickerRow(2_000, 2, 99.0, 1.0, 101.0, 1.0),
        BookTickerRow(1_000, 1, 99.0, 1.0, 101.0, 1.0),
    ]
    with pytest.raises(ValueError, match="merged chronologically"):
        list(market_rows_to_replay_events(rows))
