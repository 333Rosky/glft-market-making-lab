from __future__ import annotations

import math

import numpy as np
import pytest

from glft_lab.data import AggTradeRow, BookTickerRow, merge_market_rows
from glft_lab.episodes import EpisodeConfig, build_hazard_intervals, intervals_to_columns


def _book(
    timestamp: int,
    sequence: int,
    bid: float = 99,
    ask: float = 101,
) -> BookTickerRow:
    return BookTickerRow(timestamp, sequence, bid, 2.0, ask, 3.0)


def test_queue_ahead_and_wrong_aggressor_are_handled_without_lookahead() -> None:
    books = iter([_book(0, 1), _book(100, 2), _book(200, 3), _book(300, 4)])
    trades = iter(
        [
            AggTradeRow(100, 1, 99.0, 10.0, False),  # buyer aggressor: cannot hit bid
            AggTradeRow(200, 2, 99.0, 1.5, True),
            AggTradeRow(300, 3, 99.0, 1.0, True),
        ]
    )
    config = EpisodeConfig(
        tick_size=1.0,
        order_quantity=1.0,
        distance_ticks=(0,),
        decision_interval_ms=10_000,
        placement_latency_ms=0,
        horizon_ms=1_000,
        observation_interval_ms=1_000,
        adverse_move_ticks=10,
    )
    intervals = list(build_hazard_intervals(merge_market_rows(books, trades), config))
    bid_fill = [row for row in intervals if row.side == "bid" and row.risk == "fill"]
    assert len(bid_fill) == 1
    assert bid_fill[0].filled_quantity == 0.5
    assert bid_fill[0].queue_ahead == 2.0


def test_latency_prevents_a_trade_before_activation_from_filling() -> None:
    books = iter([_book(0, 1), _book(10, 2), _book(20, 3), _book(50, 4), _book(60, 5)])
    trades = iter(
        [
            AggTradeRow(10, 1, 99.0, 10.0, True),
            AggTradeRow(60, 2, 99.0, 10.0, True),
        ]
    )
    config = EpisodeConfig(
        tick_size=1.0,
        order_quantity=1.0,
        distance_ticks=(0,),
        decision_interval_ms=10_000,
        placement_latency_ms=50,
        horizon_ms=1_000,
        observation_interval_ms=1_000,
        adverse_move_ticks=10,
    )
    rows = list(build_hazard_intervals(merge_market_rows(books, trades), config))
    fills = [row for row in rows if row.side == "bid" and row.risk == "fill"]
    assert len(fills) == 1
    assert fills[0].start_timestamp == 50
    assert fills[0].timestamp == fills[0].end_timestamp == 60


def test_trade_at_acknowledgement_timestamp_precedes_activation() -> None:
    books = iter([_book(0, 1), _book(50, 2), _book(100, 3)])
    trades = iter([AggTradeRow(50, 1, 99.0, 10.0, True)])
    rows = list(
        build_hazard_intervals(
            merge_market_rows(books, trades),
            EpisodeConfig(
                tick_size=1.0,
                distance_ticks=(0,),
                decision_interval_ms=10_000,
                placement_latency_ms=50,
                horizon_ms=50,
                observation_interval_ms=1_000,
                adverse_move_ticks=10,
            ),
        )
    )

    assert not any(row.risk == "fill" for row in rows)
    bid = next(row for row in rows if row.side == "bid")
    assert bid.start_timestamp == 50
    assert bid.end_timestamp == 100
    assert bid.risk == "cancel"


def test_intervals_expose_every_state_feature_for_walk_forward() -> None:
    rows = list(
        build_hazard_intervals(
            iter([_book(1_725_148_800_000, 1), _book(1_725_148_800_300, 2)]),
            EpisodeConfig(
                tick_size=1.0,
                distance_ticks=(0,),
                placement_latency_ms=0,
                horizon_ms=200,
                observation_interval_ms=100,
                adverse_move_ticks=10,
            ),
        )
    )
    columns = intervals_to_columns(rows)
    expected = {
        "timestamp",
        "start_timestamp",
        "end_timestamp",
        "episode_id",
        "side",
        "distance",
        "spread",
        "imbalance",
        "ofi",
        "volatility",
        "queue_ahead",
        "order_age",
        "time_of_day",
        "exposure",
        "count",
        "risk",
        "requested_distance_ticks",
    }
    assert expected <= columns.keys()
    assert len(columns["timestamp"]) == len(rows)


def test_activation_and_expiry_use_timer_clock_between_market_rows() -> None:
    rows = list(
        build_hazard_intervals(
            merge_market_rows(iter([_book(0, 1), _book(1_000, 2)]), iter([])),
            EpisodeConfig(
                tick_size=1.0,
                distance_ticks=(0,),
                decision_interval_ms=10_000,
                placement_latency_ms=50,
                horizon_ms=100,
                observation_interval_ms=1_000,
                adverse_move_ticks=10,
            ),
        )
    )

    assert len(rows) == 2
    assert {row.side for row in rows} == {"bid", "ask"}
    assert all(row.start_timestamp == 50 for row in rows)
    assert all(row.timestamp == row.end_timestamp == 150 for row in rows)
    assert all(row.exposure == pytest.approx(0.1) for row in rows)
    assert all(row.risk == "cancel" for row in rows)


def test_trade_at_expiry_wins_tie_without_fake_one_millisecond_bin() -> None:
    books = iter([_book(0, 1), _book(100, 2)])
    trades = iter([AggTradeRow(100, 1, 99.0, 3.0, True)])
    rows = list(
        build_hazard_intervals(
            merge_market_rows(books, trades),
            EpisodeConfig(
                tick_size=1.0,
                distance_ticks=(0,),
                decision_interval_ms=10_000,
                placement_latency_ms=0,
                horizon_ms=100,
                observation_interval_ms=100,
                adverse_move_ticks=10,
            ),
        )
    )

    bid_rows = [row for row in rows if row.side == "bid"]
    assert len(bid_rows) == 1
    assert bid_rows[0].risk == "fill"
    assert bid_rows[0].exposure == pytest.approx(0.1)
    assert bid_rows[0].filled_quantity == pytest.approx(1.0)
    ask_rows = [row for row in rows if row.side == "ask"]
    assert len(ask_rows) == 1
    assert ask_rows[0].risk == "cancel"
    assert ask_rows[0].exposure == pytest.approx(0.1)


def test_decisions_keep_fixed_cadence_across_sparse_market_rows() -> None:
    rows = list(
        build_hazard_intervals(
            iter([_book(0, 1), _book(350, 2)]),
            EpisodeConfig(
                tick_size=1.0,
                distance_ticks=(0,),
                decision_interval_ms=100,
                placement_latency_ms=0,
                horizon_ms=50,
                observation_interval_ms=1_000,
                adverse_move_ticks=10,
            ),
        )
    )

    assert len(rows) == 8
    assert all(row.risk == "cancel" for row in rows)
    assert len({row.episode_id for row in rows}) == len(rows)
    assert [row.start_timestamp for row in rows] == [
        0,
        0,
        100,
        100,
        200,
        200,
        300,
        300,
    ]
    assert [row.end_timestamp for row in rows] == [
        50,
        50,
        150,
        150,
        250,
        250,
        350,
        350,
    ]
    assert all(row.exposure == pytest.approx(0.05) for row in rows)


def test_sparse_cadence_anchors_latency_to_each_decision_time() -> None:
    rows = list(
        build_hazard_intervals(
            iter([_book(0, 1), _book(250, 2)]),
            EpisodeConfig(
                tick_size=1.0,
                distance_ticks=(0,),
                decision_interval_ms=100,
                placement_latency_ms=20,
                horizon_ms=30,
                observation_interval_ms=1_000,
                adverse_move_ticks=10,
            ),
        )
    )

    assert [row.start_timestamp for row in rows] == [20, 20, 120, 120, 220, 220]
    assert [row.end_timestamp for row in rows] == [50, 50, 150, 150, 250, 250]
    assert all(row.risk == "cancel" for row in rows)


def test_queue_snapshot_changes_only_after_boundary_event() -> None:
    books = iter([_book(0, 1), _book(100, 2), _book(150, 3)])
    trades = iter(
        [
            AggTradeRow(100, 1, 99.0, 1.0, True),
            AggTradeRow(150, 2, 99.0, 1.5, True),
        ]
    )
    rows = list(
        build_hazard_intervals(
            merge_market_rows(books, trades),
            EpisodeConfig(
                tick_size=1.0,
                distance_ticks=(0,),
                decision_interval_ms=10_000,
                placement_latency_ms=0,
                horizon_ms=1_000,
                observation_interval_ms=100,
                adverse_move_ticks=10,
            ),
        )
    )

    bid_rows = [row for row in rows if row.side == "bid"]
    assert len(bid_rows) == 2
    assert bid_rows[0].start_timestamp == 0
    assert bid_rows[0].end_timestamp == 100
    assert bid_rows[0].queue_ahead == pytest.approx(2.0)
    assert bid_rows[0].risk == "none"
    assert bid_rows[1].start_timestamp == 100
    assert bid_rows[1].end_timestamp == 150
    assert bid_rows[1].queue_ahead == pytest.approx(1.0)
    assert bid_rows[1].risk == "fill"
    assert bid_rows[1].filled_quantity == pytest.approx(0.5)
    assert bid_rows[0].episode_id == bid_rows[1].episode_id


def test_latency_snapshot_uses_last_causal_book_and_current_distance() -> None:
    rows = list(
        build_hazard_intervals(
            iter(
                [
                    _book(0, 1),
                    _book(25, 2, bid=100, ask=102),
                    _book(100, 3, bid=100, ask=102),
                ]
            ),
            EpisodeConfig(
                tick_size=1.0,
                distance_ticks=(0,),
                decision_interval_ms=10_000,
                placement_latency_ms=50,
                horizon_ms=200,
                observation_interval_ms=50,
                adverse_move_ticks=10,
            ),
        )
    )

    assert len(rows) == 2
    bid = next(row for row in rows if row.side == "bid")
    ask = next(row for row in rows if row.side == "ask")
    assert bid.start_timestamp == ask.start_timestamp == 50
    assert bid.end_timestamp == ask.end_timestamp == 100
    assert bid.distance == pytest.approx(1.0)
    assert ask.distance == pytest.approx(0.0)
    assert bid.requested_distance_ticks == ask.requested_distance_ticks == 0
    assert bid.queue_ahead == ask.queue_ahead == 0.0


def test_trade_beyond_limit_proves_queue_was_crossed() -> None:
    books = iter([_book(0, 1), _book(100, 2)])
    trades = iter([AggTradeRow(100, 1, 98.0, 0.1, True)])
    rows = list(
        build_hazard_intervals(
            merge_market_rows(books, trades),
            EpisodeConfig(
                tick_size=1.0,
                distance_ticks=(0,),
                decision_interval_ms=10_000,
                placement_latency_ms=0,
                horizon_ms=1_000,
                observation_interval_ms=1_000,
                adverse_move_ticks=10,
            ),
        )
    )

    fill = next(row for row in rows if row.side == "bid" and row.risk == "fill")
    assert fill.filled_quantity == pytest.approx(1.0)


def test_eof_does_not_invent_exposure_for_just_activated_orders() -> None:
    rows = list(
        build_hazard_intervals(
            iter([_book(0, 1)]),
            EpisodeConfig(tick_size=1.0, distance_ticks=(0,), placement_latency_ms=0),
        )
    )
    assert rows == []


def test_column_timestamp_is_label_availability_time_not_feature_time() -> None:
    january_start = int(np.datetime64("2024-01-31T23:59:59.950", "ms").astype(np.int64))
    february_end = january_start + 100
    rows = list(
        build_hazard_intervals(
            iter([_book(january_start, 1), _book(february_end, 2)]),
            EpisodeConfig(
                tick_size=1.0,
                distance_ticks=(0,),
                decision_interval_ms=10_000,
                placement_latency_ms=0,
                horizon_ms=100,
                observation_interval_ms=1_000,
                adverse_move_ticks=10,
            ),
        )
    )
    columns = intervals_to_columns(rows)

    assert np.all(columns["start_timestamp"] == january_start)
    assert np.all(columns["timestamp"] == columns["end_timestamp"])
    months = columns["timestamp"].astype("datetime64[ms]").astype("datetime64[M]")
    assert np.all(months == np.datetime64("2024-02"))


def test_column_dtypes_are_stable_for_empty_input() -> None:
    columns = intervals_to_columns([])

    assert columns["timestamp"].dtype == np.dtype(np.int64)
    assert columns["start_timestamp"].dtype == np.dtype(np.int64)
    assert columns["episode_id"].dtype == np.dtype(np.int64)
    assert columns["count"].dtype == np.dtype(np.int64)
    assert columns["exposure"].dtype == np.dtype(np.float64)
    assert columns["queue_is_approximate"].dtype == np.dtype(np.bool_)


def test_non_finite_book_and_invalid_trade_values_are_rejected() -> None:
    config = EpisodeConfig(tick_size=1.0, distance_ticks=(0,), placement_latency_ms=0)
    invalid_book = BookTickerRow(0, 1, math.nan, 2.0, 101.0, 3.0)
    with pytest.raises(ValueError, match="bookTicker"):
        list(build_hazard_intervals(iter([invalid_book]), config))

    books = iter([_book(0, 1), _book(10, 2)])
    trades = iter([AggTradeRow(10, 1, 99.0, 0.0, True)])
    with pytest.raises(ValueError, match="aggTrades"):
        list(build_hazard_intervals(merge_market_rows(books, trades), config))


def test_quantity_step_validates_order_and_book_quantities() -> None:
    EpisodeConfig(tick_size=1.0, order_quantity=0.3, quantity_step=0.1)

    with pytest.raises(ValueError, match="order_quantity.*quantity_step"):
        EpisodeConfig(tick_size=1.0, order_quantity=0.15, quantity_step=0.1)

    config = EpisodeConfig(tick_size=1.0, quantity_step=1.0)
    fractional_book = BookTickerRow(0, 1, 99.0, 2.5, 101.0, 3.0)
    with pytest.raises(ValueError, match="bookTicker bid_quantity.*quantity_step"):
        list(build_hazard_intervals(iter([fractional_book]), config))


def test_fractional_trade_fails_before_any_queue_depletion() -> None:
    books = iter([_book(0, 1), _book(100, 2)])
    trades = iter(
        [
            AggTradeRow(100, 1, 99.0, 3.0, True),
            AggTradeRow(100, 2, 101.0, 0.5, False),
        ]
    )
    intervals = build_hazard_intervals(
        merge_market_rows(books, trades),
        EpisodeConfig(
            tick_size=1.0,
            order_quantity=1.0,
            quantity_step=1.0,
            distance_ticks=(0,),
            decision_interval_ms=10_000,
            placement_latency_ms=0,
            horizon_ms=1_000,
            observation_interval_ms=1_000,
            adverse_move_ticks=10,
        ),
    )

    # The first integral trade would immediately fill the bid.  Validating the
    # complete timestamp group first ensures the later wrong-market fraction is
    # rejected before that queue can be touched or an interval can be emitted.
    with pytest.raises(ValueError, match="aggTrades quantity.*quantity_step"):
        next(intervals)


@pytest.mark.parametrize(
    "overrides",
    [
        {"decision_interval_ms": 1.5},
        {"placement_latency_ms": True},
        {"adverse_move_ticks": math.nan},
        {"distance_ticks": (0, 0)},
        {"quantity_step": 0.0},
        {"quantity_step": math.nan},
        {"quantity_step": True},
    ],
)
def test_configuration_rejects_non_integral_clocks_and_non_finite_values(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        EpisodeConfig(tick_size=1.0, **overrides)
