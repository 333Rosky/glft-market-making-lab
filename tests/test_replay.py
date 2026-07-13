from __future__ import annotations

import pytest

from glft_lab.replay import (
    QUEUE_MODEL_DESCRIPTION,
    AccountingModel,
    BookEvent,
    CancelOrderEvent,
    EventDrivenReplay,
    Liquidity,
    OrderStatus,
    PlaceOrderEvent,
    QuoteEvent,
    ReplayConfig,
    Side,
    TradeEvent,
    round_to_tick,
)


def book(
    timestamp: float,
    *,
    bid: float = 100.0,
    bid_quantity: float = 0.0,
    ask: float = 101.0,
    ask_quantity: float = 0.0,
    sequence: int = 0,
) -> BookEvent:
    return BookEvent(
        timestamp=timestamp,
        bid_price=bid,
        bid_quantity=bid_quantity,
        ask_price=ask,
        ask_quantity=ask_quantity,
        sequence=sequence,
    )


def place(
    timestamp: float,
    order_id: str,
    side: Side,
    price: float,
    quantity: float = 1.0,
    sequence: int = 0,
) -> PlaceOrderEvent:
    return PlaceOrderEvent(
        timestamp=timestamp,
        order_id=order_id,
        side=side,
        price=price,
        quantity=quantity,
        sequence=sequence,
    )


def trade(
    timestamp: float,
    side: Side,
    *,
    price: float = 100.0,
    quantity: float = 1.0,
    sequence: int = 0,
) -> TradeEvent:
    return TradeEvent(
        timestamp=timestamp,
        price=price,
        quantity=quantity,
        aggressor_side=side,
        sequence=sequence,
    )


def test_chronological_order_latency_tie_policy_and_no_lookahead() -> None:
    replay = EventDrivenReplay(
        ReplayConfig(tick_size=0.1, placement_latency=2.0, liquidate_at_end=False)
    )

    # Deliberately shuffled.  The sell at the exact activation timestamp is processed
    # first and therefore cannot fill; only the causally later trade can.
    result = replay.run(
        [
            trade(3.0, Side.SELL),
            trade(2.0, Side.SELL),
            place(0.0, "bid", Side.BUY, 100.0),
            book(0.0),
        ]
    )

    assert [fill.timestamp for fill in result.fills] == [3.0]
    assert result.order("bid").active_at == 2.0
    assert [event.timestamp for event in result.processed_events] == sorted(
        event.timestamp for event in result.processed_events
    )
    tied = [event.event_type for event in result.processed_events if event.timestamp == 2.0]
    assert tied == ["trade", "placement_ack"]


def test_source_sequence_orders_market_events_before_type_fallback() -> None:
    replay = EventDrivenReplay(ReplayConfig(tick_size=0.1, liquidate_at_end=False))
    result = replay.run(
        [
            # Despite being listed first, sequence 2 comes after the trade at sequence 1.
            book(1.0, bid=99.0, ask=101.0, sequence=2),
            trade(1.0, Side.SELL, price=100.0, sequence=1),
            book(0.0),
            place(0.0, "bid", Side.BUY, 100.0),
        ]
    )

    assert result.order("bid").status is OrderStatus.FILLED
    at_one = [event.event_type for event in result.processed_events if event.timestamp == 1.0]
    assert at_one == ["trade", "book"]


def test_cancel_latency_leaves_order_fillable_until_ack() -> None:
    replay = EventDrivenReplay(
        ReplayConfig(tick_size=0.1, cancel_latency=2.0, liquidate_at_end=False)
    )
    result = replay.run(
        [
            book(0.0),
            place(0.0, "bid", Side.BUY, 100.0, quantity=2.0),
            CancelOrderEvent(timestamp=1.0, order_id="bid"),
            trade(2.0, Side.SELL, quantity=1.0),
            book(4.0),
        ]
    )

    order = result.order("bid")
    assert order.filled_quantity == pytest.approx(1.0)
    assert order.status is OrderStatus.CANCELED
    assert order.canceled_at == 3.0
    assert result.fills[0].timestamp == 2.0


def test_canceling_our_earlier_order_advances_our_later_fifo_order() -> None:
    replay = EventDrivenReplay(ReplayConfig(tick_size=0.1, liquidate_at_end=False))
    result = replay.run(
        [
            book(0.0, bid_quantity=5.0),
            place(0.0, "a", Side.BUY, 100.0, quantity=3.0),
            place(0.0, "b", Side.BUY, 100.0, quantity=2.0),
            CancelOrderEvent(timestamp=1.0, order_id="a"),
            trade(2.0, Side.SELL, quantity=6.0),
        ]
    )

    assert result.order("a").status is OrderStatus.CANCELED
    assert result.order("b").filled_quantity == pytest.approx(1.0)
    assert [(fill.order_id, fill.quantity) for fill in result.fills] == [("b", 1.0)]


def test_fifo_queue_depletion_and_partial_fills() -> None:
    replay = EventDrivenReplay(ReplayConfig(tick_size=0.1, liquidate_at_end=False))
    result = replay.run(
        [
            book(0.0, bid_quantity=5.0),
            place(0.0, "a", Side.BUY, 100.0, quantity=3.0),
            place(0.0, "b", Side.BUY, 100.0, quantity=2.0),
            trade(1.0, Side.SELL, quantity=6.0),
            trade(2.0, Side.SELL, quantity=2.0),
            trade(3.0, Side.SELL, quantity=1.0),
        ]
    )

    assert result.order("a").initial_queue_ahead == pytest.approx(5.0)
    assert result.order("b").initial_queue_ahead == pytest.approx(8.0)
    assert [(fill.order_id, fill.quantity) for fill in result.fills] == [
        ("a", 1.0),
        ("a", 2.0),
        ("b", 1.0),
    ]
    assert result.order("a").status is OrderStatus.FILLED
    assert result.order("b").filled_quantity == pytest.approx(1.0)


def test_book_quantity_reduction_does_not_optimistically_advance_queue() -> None:
    replay = EventDrivenReplay(ReplayConfig(tick_size=0.1, liquidate_at_end=False))
    result = replay.run(
        [
            book(0.0, bid_quantity=5.0),
            place(0.0, "bid", Side.BUY, 100.0),
            # Could be cancellation behind us, so it cannot reduce queue_ahead.
            book(1.0, bid_quantity=0.0),
            trade(2.0, Side.SELL, quantity=4.0),
            trade(3.0, Side.SELL, quantity=2.0),
        ]
    )

    assert [(fill.timestamp, fill.quantity) for fill in result.fills] == [(3.0, 1.0)]
    assert "bookTicker" in result.queue_model
    assert result.queue_model == QUEUE_MODEL_DESCRIPTION


def test_away_order_waits_for_top_of_book_before_fifo_depletion() -> None:
    replay = EventDrivenReplay(ReplayConfig(tick_size=0.1, liquidate_at_end=False))
    result = replay.run(
        [
            book(0.0, bid=100.0, ask=102.0),
            place(0.0, "deep_bid", Side.BUY, 99.0),
            book(1.0, bid=99.0, bid_quantity=7.0, ask=102.0),
            trade(2.0, Side.SELL, price=99.0, quantity=8.0),
        ]
    )

    order = result.order("deep_bid")
    assert order.initial_queue_ahead == pytest.approx(7.0)
    assert order.filled_quantity == pytest.approx(1.0)


def test_wrong_aggressor_side_never_fills_resting_bid() -> None:
    replay = EventDrivenReplay(ReplayConfig(tick_size=0.1, liquidate_at_end=False))
    result = replay.run(
        [
            book(0.0),
            place(0.0, "bid", Side.BUY, 100.0),
            trade(1.0, Side.BUY, price=100.0, quantity=100.0),
        ]
    )

    assert result.fills == ()
    assert result.order("bid").filled_quantity == 0.0


def test_side_aware_tick_rounding_and_post_only_rejection() -> None:
    assert round_to_tick(100.09, Side.BUY, 0.1) == pytest.approx(100.0)
    assert round_to_tick(100.01, Side.SELL, 0.1) == pytest.approx(100.1)

    replay = EventDrivenReplay(ReplayConfig(tick_size=0.1, liquidate_at_end=False))
    result = replay.run(
        [
            book(0.0, bid=99.0, ask=100.0),
            place(0.0, "crossing_bid", Side.BUY, 100.09),
            place(0.0, "crossing_ask", Side.SELL, 98.99),
        ]
    )

    assert result.order("crossing_bid").price == 100.0
    assert result.order("crossing_ask").price == 99.0
    assert result.order("crossing_bid").status is OrderStatus.REJECTED
    assert result.order("crossing_ask").status is OrderStatus.REJECTED
    assert all("post-only" in order.rejection_reason for order in result.orders.values())


def test_activation_without_an_observed_book_rejects_instead_of_using_future_book() -> None:
    replay = EventDrivenReplay(
        ReplayConfig(tick_size=0.1, placement_latency=1.0, liquidate_at_end=False)
    )
    result = replay.run(
        [
            place(0.0, "bid", Side.BUY, 100.0),
            book(2.0),
            trade(3.0, Side.SELL),
        ]
    )

    order = result.order("bid")
    assert order.status is OrderStatus.REJECTED
    assert order.active_at is None
    assert result.fills == ()


def test_maker_fee_supports_negative_rebate_and_equity_is_marked_to_mid() -> None:
    replay = EventDrivenReplay(
        ReplayConfig(
            tick_size=0.1,
            maker_fee_rate=-0.001,
            liquidate_at_end=False,
            markout_horizons=(),
        )
    )
    result = replay.run([book(0.0), place(0.0, "bid", Side.BUY, 100.0), trade(1.0, Side.SELL)])

    fill = result.fills[0]
    assert fill.fee == pytest.approx(-0.1)
    assert fill.cash_after == pytest.approx(-99.9)
    assert fill.inventory_after == pytest.approx(1.0)
    assert result.final_equity == pytest.approx(0.6)
    assert result.equity_curve[-1].reason == "maker_fill"
    assert result.equity_curve[-1].equity == pytest.approx(0.6)


@pytest.mark.parametrize(
    ("maker_side", "aggressor", "maker_price", "liquidation_side", "liquidation_price", "cash"),
    [
        (Side.BUY, Side.SELL, 100.0, Side.SELL, 99.0, -1.99),
        (Side.SELL, Side.BUY, 101.0, Side.BUY, 102.0, -2.02),
    ],
)
def test_final_liquidation_uses_side_aware_touch_and_taker_fee(
    maker_side: Side,
    aggressor: Side,
    maker_price: float,
    liquidation_side: Side,
    liquidation_price: float,
    cash: float,
) -> None:
    replay = EventDrivenReplay(
        ReplayConfig(tick_size=0.1, taker_fee_rate=0.01, markout_horizons=())
    )
    result = replay.run(
        [
            book(0.0, bid=100.0, ask=101.0),
            place(0.0, "maker", maker_side, maker_price),
            trade(1.0, aggressor, price=maker_price),
            book(2.0, bid=99.0, ask=102.0),
        ]
    )

    liquidation = result.fills[-1]
    assert liquidation.is_liquidation
    assert liquidation.liquidity is Liquidity.TAKER
    assert liquidation.side is liquidation_side
    assert liquidation.price == liquidation_price
    assert result.final_inventory == 0.0
    assert result.final_cash == pytest.approx(cash)
    assert result.final_equity == pytest.approx(cash)


def test_quote_refresh_keeps_age_while_replacement_waits_for_cancel() -> None:
    replay = EventDrivenReplay(
        ReplayConfig(
            tick_size=0.1,
            placement_latency=1.0,
            cancel_latency=2.0,
            liquidate_at_end=False,
        )
    )
    result = replay.run(
        [
            book(0.0, bid=100.0, ask=102.0),
            QuoteEvent(timestamp=0.0, quote_id="q", bid_price=100.0, bid_quantity=1.0),
            # Identical refresh: preserve the original FIFO position and active age.
            QuoteEvent(timestamp=2.0, quote_id="q", bid_price=100.0, bid_quantity=1.0),
            QuoteEvent(timestamp=3.0, quote_id="q", bid_price=99.0, bid_quantity=1.0),
            book(5.0, bid=99.0, ask=102.0),
            book(6.0, bid=99.0, ask=102.0),
        ]
    )

    bids = sorted(
        (order for order in result.orders.values() if order.side is Side.BUY),
        key=lambda order: order.requested_at,
    )
    assert len(bids) == 2
    original, replacement = bids
    assert original.active_at == 1.0
    assert original.age_at(4.0) == 3.0
    assert original.status is OrderStatus.CANCELED
    assert original.canceled_at == 5.0
    assert replacement.activation_at == 5.0
    assert replacement.active_at == 5.0


def test_markouts_are_post_trade_diagnostics_not_fill_inputs() -> None:
    replay = EventDrivenReplay(
        ReplayConfig(
            tick_size=0.1,
            liquidate_at_end=False,
            markout_horizons=(2.0, 5.0),
        )
    )
    result = replay.run(
        [
            book(0.0),
            place(0.0, "bid", Side.BUY, 100.0),
            trade(1.0, Side.SELL),
            book(3.0, bid=101.0, ask=103.0),
        ]
    )

    two_seconds, unavailable = result.markouts
    assert two_seconds.observed_at == 3.0
    assert two_seconds.reference_mid == 102.0
    assert two_seconds.signed_pnl_per_unit == 2.0
    assert two_seconds.signed_bps == pytest.approx(200.0)
    assert unavailable.observed_at is None
    assert unavailable.signed_pnl_per_unit is None


def test_inverse_contract_open_position_rebate_mtm_and_markout() -> None:
    replay = EventDrivenReplay(
        ReplayConfig(
            tick_size=1.0,
            accounting_model=AccountingModel.INVERSE,
            contract_multiplier=100.0,
            maker_fee_rate=-0.001,
            liquidate_at_end=False,
            markout_horizons=(1.0,),
        )
    )
    result = replay.run(
        [
            book(0.0, bid=100.0, ask=101.0),
            place(0.0, "inverse_bid", Side.BUY, 100.0),
            trade(1.0, Side.SELL, price=100.0),
            book(2.0, bid=109.0, ask=111.0),
        ]
    )

    fill = result.fills[0]
    expected_unrealized = 100.0 * (1.0 / 100.0 - 1.0 / 110.0)
    assert fill.notional == pytest.approx(100.0)  # USD face value
    assert fill.settlement_notional == pytest.approx(1.0)  # BTC at fill price
    assert fill.fee == pytest.approx(-0.001)
    assert fill.cash_after == pytest.approx(0.001)
    assert fill.average_entry_price_after == pytest.approx(100.0)
    assert result.final_inventory == 1.0
    assert result.final_average_entry_price == pytest.approx(100.0)
    assert result.final_equity == pytest.approx(0.001 + expected_unrealized)
    assert result.cash_unit == "base_asset"
    assert result.accounting_model is AccountingModel.INVERSE
    assert result.quantity_step == 1.0
    assert result.markouts[0].signed_pnl_per_unit == pytest.approx(expected_unrealized)
    assert result.markouts[0].signed_bps == pytest.approx(10_000.0 / 11.0)


def test_inverse_harmonic_cost_basis_and_partial_close_realized_pnl() -> None:
    replay = EventDrivenReplay(
        ReplayConfig(
            tick_size=1.0,
            accounting_model=AccountingModel.INVERSE,
            contract_multiplier=100.0,
            liquidate_at_end=False,
            markout_horizons=(),
        )
    )
    result = replay.run(
        [
            book(0.0, bid=100.0, ask=101.0),
            place(0.0, "first", Side.BUY, 100.0),
            trade(1.0, Side.SELL, price=100.0),
            book(2.0, bid=200.0, ask=201.0),
            place(2.0, "second", Side.BUY, 200.0),
            trade(3.0, Side.SELL, price=200.0),
            book(4.0, bid=149.0, ask=150.0),
            place(4.0, "close_one", Side.SELL, 150.0),
            trade(5.0, Side.BUY, price=150.0),
        ]
    )

    harmonic_entry = 2.0 / (1.0 / 100.0 + 1.0 / 200.0)
    realized = 100.0 * (1.0 / harmonic_entry - 1.0 / 150.0)
    close_fill = next(fill for fill in result.fills if fill.order_id == "close_one")
    assert harmonic_entry == pytest.approx(133.33333333333334)
    assert close_fill.realized_pnl == pytest.approx(realized)
    assert close_fill.average_entry_price_after == pytest.approx(harmonic_entry)
    assert result.final_inventory == 1.0
    assert result.final_cash == pytest.approx(realized)
    assert result.final_average_entry_price == pytest.approx(harmonic_entry)


@pytest.mark.parametrize(
    ("inventory", "entry", "bid", "ask", "side", "price", "expected_cash"),
    [
        (1.0, 100.0, 110.0, 111.0, Side.SELL, 110.0, 0.09),
        (-1.0, 100.0, 89.0, 90.0, Side.BUY, 90.0, 0.11),
    ],
)
def test_inverse_final_liquidation_realizes_at_touch_and_charges_base_fee(
    inventory: float,
    entry: float,
    bid: float,
    ask: float,
    side: Side,
    price: float,
    expected_cash: float,
) -> None:
    replay = EventDrivenReplay(
        ReplayConfig(
            tick_size=1.0,
            accounting_model=AccountingModel.INVERSE,
            contract_multiplier=100.0,
            initial_inventory=inventory,
            initial_entry_price=entry,
            taker_fee_rate=0.001,
            markout_horizons=(),
        )
    )
    result = replay.run([book(0.0, bid=bid, ask=ask)])

    liquidation = result.fills[-1]
    assert liquidation.side is side
    assert liquidation.price == price
    assert liquidation.notional == 100.0
    assert liquidation.settlement_notional == pytest.approx(100.0 / price)
    assert liquidation.fee == pytest.approx(100.0 / price * 0.001)
    assert result.final_cash == pytest.approx(expected_cash)
    assert result.final_equity == pytest.approx(expected_cash)
    assert result.final_inventory == 0.0
    assert result.final_average_entry_price is None


@pytest.mark.parametrize(
    ("events", "label"),
    [
        (
            [book(0.0), trade(1.0, Side.SELL, quantity=0.6)],
            "trade quantity",
        ),
        (
            [book(0.0, bid_quantity=0.5)],
            "book bid quantity",
        ),
        (
            [book(0.0), place(0.0, "fractional", Side.BUY, 100.0, 0.5)],
            "order quantity",
        ),
    ],
)
def test_inverse_contract_quantities_fail_fast_on_step_mismatch(
    events: list[BookEvent | TradeEvent | PlaceOrderEvent],
    label: str,
) -> None:
    replay = EventDrivenReplay(
        ReplayConfig(
            tick_size=1.0,
            accounting_model=AccountingModel.INVERSE,
            contract_multiplier=100.0,
            liquidate_at_end=False,
            markout_horizons=(),
        )
    )

    with pytest.raises(ValueError, match=label):
        replay.run(events)


def test_inverse_initial_inventory_requires_entry_price_and_contract_step() -> None:
    with pytest.raises(ValueError, match="initial_entry_price"):
        ReplayConfig(
            tick_size=1.0,
            accounting_model=AccountingModel.INVERSE,
            initial_inventory=1.0,
        )
    with pytest.raises(ValueError, match="quantity_step"):
        ReplayConfig(
            tick_size=1.0,
            accounting_model=AccountingModel.INVERSE,
            initial_inventory=0.5,
            initial_entry_price=100.0,
        )
