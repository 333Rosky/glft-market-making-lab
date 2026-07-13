import math

import numpy as np
import pytest
from scipy.linalg import expm

from glft_lab.glft import (
    THEORETICAL_BENCHMARK_LABEL,
    GLFTParameters,
    finite_horizon_value,
    glft_constants,
    glft_matrix,
    inventory_grid,
    optimal_deltas,
    poisson_fill_intensity,
    simulate_poisson_benchmark,
)


def test_exact_constants_and_hand_computed_matrix() -> None:
    parameters = GLFTParameters(
        A=2.0,
        k=0.5,
        gamma=0.25,
        sigma=3.0,
        mu=0.1,
        max_inventory=1,
    )

    constants = glft_constants(parameters)
    expected_c = 4.0 * math.log(1.5)
    expected_eta = 2.0 * 1.5**-3.0
    expected_alpha = 0.5 * 0.5 * 0.25 * 3.0**2
    expected_beta = 0.5 * 0.1

    assert constants.c == pytest.approx(expected_c)
    assert constants.eta == pytest.approx(expected_eta)
    assert constants.alpha == pytest.approx(expected_alpha)
    assert constants.beta == pytest.approx(expected_beta)
    np.testing.assert_array_equal(inventory_grid(parameters), [-1, 0, 1])

    expected_matrix = np.array(
        [
            [expected_alpha + expected_beta, -expected_eta, 0.0],
            [-expected_eta, 0.0, -expected_eta],
            [0.0, -expected_eta, expected_alpha - expected_beta],
        ]
    )
    np.testing.assert_allclose(glft_matrix(parameters), expected_matrix)


def test_finite_horizon_value_and_exact_quote_ratios() -> None:
    parameters = GLFTParameters(
        A=1.4,
        k=0.8,
        gamma=0.3,
        sigma=1.1,
        mu=0.02,
        max_inventory=2,
    )
    t = 0.35
    horizon = 1.2
    matrix = glft_matrix(parameters)
    expected_value = expm(-matrix * (horizon - t)) @ np.ones(5)

    value = finite_horizon_value(parameters, t, horizon)
    np.testing.assert_allclose(value, expected_value, rtol=1e-13, atol=1e-14)

    quote = optimal_deltas(parameters, inventory=0, t=t, horizon=horizon)
    c = glft_constants(parameters).c
    expected_bid = c + math.log(expected_value[2] / expected_value[3]) / parameters.k
    expected_ask = c + math.log(expected_value[2] / expected_value[1]) / parameters.k
    assert quote.bid_delta == pytest.approx(expected_bid)
    assert quote.ask_delta == pytest.approx(expected_ask)


def test_relative_value_quotes_match_direct_exponential_at_moderate_horizon() -> None:
    parameters = GLFTParameters(
        A=1.7,
        k=0.9,
        gamma=0.2,
        sigma=0.7,
        mu=-0.01,
        max_inventory=3,
    )
    t = 1.25
    horizon = 12.0
    direct = finite_horizon_value(parameters, t, horizon)
    c = glft_constants(parameters).c

    for inventory in range(-parameters.max_inventory, parameters.max_inventory + 1):
        quote = optimal_deltas(parameters, inventory, t, horizon)
        index = inventory + parameters.max_inventory
        if inventory < parameters.max_inventory:
            expected_bid = c + math.log(direct[index] / direct[index + 1]) / parameters.k
            assert quote.bid_delta == pytest.approx(expected_bid, rel=1e-12, abs=1e-12)
        else:
            assert quote.bid_delta is None
        if inventory > -parameters.max_inventory:
            expected_ask = c + math.log(direct[index] / direct[index - 1]) / parameters.k
            assert quote.ask_delta == pytest.approx(expected_ask, rel=1e-12, abs=1e-12)
        else:
            assert quote.ask_delta is None


def test_long_horizon_quotes_and_theoretical_simulation_remain_finite() -> None:
    parameters = GLFTParameters(
        A=1.0,
        k=1.0,
        gamma=0.1,
        sigma=1.0,
        max_inventory=5,
    )
    horizon = 2_000.0

    quote = optimal_deltas(parameters, inventory=0, t=0.0, horizon=horizon)
    assert quote.bid_delta is not None and math.isfinite(quote.bid_delta)
    assert quote.ask_delta is not None and math.isfinite(quote.ask_delta)
    assert quote.bid_delta > 0.0
    assert quote.ask_delta > 0.0

    simulation = simulate_poisson_benchmark(
        parameters,
        horizon=horizon,
        dt=horizon,
        initial_mid_price=100.0,
        seed=7,
    )
    assert np.isfinite(simulation.bid_delta).all()
    assert np.isfinite(simulation.ask_delta).all()
    assert np.isfinite(simulation.cash).all()
    assert np.isfinite(simulation.equity).all()


def test_terminal_quotes_equal_c_and_inventory_bounds_remove_one_side() -> None:
    parameters = GLFTParameters(
        A=3.0,
        k=1.2,
        gamma=0.4,
        sigma=0.8,
        max_inventory=2,
    )
    horizon = 2.0
    c = glft_constants(parameters).c

    np.testing.assert_array_equal(finite_horizon_value(parameters, horizon, horizon), np.ones(5))
    center = optimal_deltas(parameters, 0, horizon, horizon)
    assert center.bid_delta == pytest.approx(c)
    assert center.ask_delta == pytest.approx(c)

    upper = optimal_deltas(parameters, 2, horizon, horizon)
    assert upper.bid_delta is None
    assert upper.ask_delta == pytest.approx(c)

    lower = optimal_deltas(parameters, -2, horizon, horizon)
    assert lower.bid_delta == pytest.approx(c)
    assert lower.ask_delta is None

    with pytest.raises(ValueError, match="inventory"):
        optimal_deltas(parameters, 3, 0.0, horizon)


def test_currency_rescaling_preserves_matrix_and_rescales_deltas() -> None:
    parameters = GLFTParameters(
        A=1.7,
        k=0.8,
        gamma=0.2,
        sigma=1.4,
        mu=0.03,
        max_inventory=2,
    )
    price_scale = 100.0
    scaled = GLFTParameters(
        A=parameters.A,
        k=parameters.k / price_scale,
        gamma=parameters.gamma / price_scale,
        sigma=parameters.sigma * price_scale,
        mu=parameters.mu * price_scale,
        max_inventory=parameters.max_inventory,
    )

    np.testing.assert_allclose(glft_matrix(scaled), glft_matrix(parameters))
    assert glft_constants(scaled).c == pytest.approx(glft_constants(parameters).c * price_scale)

    original_quote = optimal_deltas(parameters, 0, t=0.2, horizon=0.8)
    scaled_quote = optimal_deltas(scaled, 0, t=0.2, horizon=0.8)
    assert scaled_quote.bid_delta == pytest.approx(original_quote.bid_delta * price_scale)
    assert scaled_quote.ask_delta == pytest.approx(original_quote.ask_delta * price_scale)


def test_poisson_benchmark_is_seeded_bounded_and_self_financing() -> None:
    parameters = GLFTParameters(
        A=50.0,
        k=1.0,
        gamma=0.1,
        sigma=0.2,
        mu=0.01,
        max_inventory=3,
    )
    arguments = dict(
        horizon=0.5,
        dt=0.05,
        initial_mid_price=100.0,
        initial_inventory=0,
        initial_cash=1_000.0,
        seed=1234,
    )

    first = simulate_poisson_benchmark(parameters, **arguments)
    second = simulate_poisson_benchmark(parameters, **arguments)
    different_seed = simulate_poisson_benchmark(parameters, **{**arguments, "seed": 1235})

    assert first.label == THEORETICAL_BENCHMARK_LABEL
    assert first.seed == 1234
    for field in (
        "time",
        "step_duration",
        "mid_price",
        "inventory",
        "cash",
        "equity",
        "bid_delta",
        "ask_delta",
        "bid_price",
        "ask_price",
        "bid_fills",
        "ask_fills",
    ):
        np.testing.assert_array_equal(getattr(first, field), getattr(second, field), strict=True)
    assert not np.array_equal(first.mid_price, different_seed.mid_price)

    assert first.time.size == first.inventory.size == first.cash.size
    assert first.step_duration.size == first.time.size - 1
    assert first.bid_fills.sum() + first.ask_fills.sum() > 0
    assert np.all(np.abs(first.inventory) <= parameters.max_inventory)
    np.testing.assert_array_equal(np.diff(first.inventory), first.bid_fills - first.ask_fills)

    expected_cash_change = -np.where(
        first.bid_fills > 0, first.bid_fills * first.bid_price, 0.0
    ) + np.where(first.ask_fills > 0, first.ask_fills * first.ask_price, 0.0)
    np.testing.assert_allclose(np.diff(first.cash), expected_cash_change)
    np.testing.assert_allclose(first.equity, first.cash + first.inventory * first.mid_price)


def test_poisson_intensity_uses_the_paper_exponential_law() -> None:
    parameters = GLFTParameters(
        A=7.0,
        k=1.5,
        gamma=0.2,
        sigma=1.0,
        max_inventory=1,
    )
    delta = 0.4
    assert poisson_fill_intensity(parameters, delta) == pytest.approx(7.0 * math.exp(-1.5 * delta))


def test_poisson_benchmark_has_no_bid_at_upper_inventory_bound() -> None:
    parameters = GLFTParameters(
        A=20.0,
        k=1.0,
        gamma=0.1,
        sigma=0.0,
        max_inventory=1,
    )
    simulation = simulate_poisson_benchmark(
        parameters,
        horizon=0.1,
        dt=0.1,
        initial_mid_price=100.0,
        initial_inventory=1,
        seed=4,
    )

    assert math.isnan(simulation.bid_delta[0])
    assert math.isnan(simulation.bid_price[0])
    assert simulation.bid_fills[0] == 0
    assert np.isfinite(simulation.cash).all()
    assert np.all(np.abs(simulation.inventory) <= parameters.max_inventory)


@pytest.mark.parametrize(
    "overrides",
    [
        {"A": 0.0},
        {"k": 0.0},
        {"gamma": -0.1},
        {"sigma": -1.0},
        {"max_inventory": 0},
    ],
)
def test_invalid_model_parameters_are_rejected(overrides: dict[str, float]) -> None:
    inputs = dict(A=1.0, k=1.0, gamma=0.1, sigma=1.0, max_inventory=2)
    inputs.update(overrides)
    with pytest.raises(ValueError):
        GLFTParameters(**inputs)
