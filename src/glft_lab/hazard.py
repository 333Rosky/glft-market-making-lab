"""State-dependent, exposure-aware fill hazard modelling.

The GLFT closed-form solution assumes an exponential Poisson fill intensity.  This
module keeps a Poisson observation model, but lets its *conditional* intensity
depend on observable order-book state.  It deliberately does not implement a
Hawkes process: residual diagnostics provide an out-of-sample gate for deciding
whether that extra complexity is justified.

All rates are expressed per unit of ``exposure``.  For example, if exposure is in
seconds, ``predict_rate`` returns fills per second and the probability of at least
one fill in an interval is ``1 - exp(-rate * exposure)``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

SIDES = ("bid", "ask")
STATE_FEATURE_NAMES = (
    "distance",
    "spread",
    "imbalance",
    "ofi",
    "volatility",
    "queue_ahead",
    "order_age",
    "tod_sin",
    "tod_cos",
)
FEATURE_NAMES = ("intercept",) + STATE_FEATURE_NAMES
RISK_LABELS = ("none", "fill", "adverse_move", "cancel")


def _column(data: Any, name: str) -> np.ndarray:
    """Return a named one-dimensional column from a mapping/structured array."""

    if isinstance(data, Mapping):
        if name not in data:
            raise KeyError(f"missing required column: {name}")
        value = data[name]
    elif isinstance(data, np.ndarray) and data.dtype.names:
        if name not in data.dtype.names:
            raise KeyError(f"missing required column: {name}")
        value = data[name]
    else:
        try:
            value = data[name]
        except (KeyError, TypeError, IndexError) as exc:
            raise KeyError(f"missing required column: {name}") from exc

    array = np.asarray(value)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1:
        raise ValueError(f"column {name!r} must be one-dimensional")
    return array


def _has_column(data: Any, name: str) -> bool:
    if isinstance(data, Mapping):
        return name in data
    if isinstance(data, np.ndarray) and data.dtype.names:
        return name in data.dtype.names
    try:
        data[name]
    except (KeyError, TypeError, IndexError):
        return False
    return True


def _float_vector(value: Any, name: str, n: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        if n is None:
            array = array.reshape(1)
        else:
            array = np.full(n, float(array))
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if n is not None and len(array) != n:
        raise ValueError(f"{name} has length {len(array)}; expected {n}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def normalize_side(side: Any, n: int | None = None) -> np.ndarray:
    """Normalize common side aliases to an array containing ``bid``/``ask``."""

    raw = np.asarray(side)
    if raw.ndim == 0:
        raw = np.full(1 if n is None else n, raw.item(), dtype=object)
    if raw.ndim != 1:
        raise ValueError("side must be one-dimensional")
    if n is not None and len(raw) != n:
        raise ValueError(f"side has length {len(raw)}; expected {n}")

    aliases = {
        "bid": "bid",
        "buy": "bid",
        "b": "bid",
        "0": "bid",
        "ask": "ask",
        "sell": "ask",
        "a": "ask",
        "1": "ask",
    }
    result = np.empty(len(raw), dtype="U3")
    for i, value in enumerate(raw):
        key = str(value).strip().lower()
        if key not in aliases:
            raise ValueError(f"unknown side {value!r}; expected bid or ask")
        result[i] = aliases[key]
    return result


def _time_fraction(data: Any, n: int) -> np.ndarray:
    """Return UTC time-of-day as a fraction in [0, 1)."""

    if _has_column(data, "time_of_day"):
        values = _float_vector(_column(data, "time_of_day"), "time_of_day", n)
        if np.any(values < 0):
            raise ValueError("time_of_day must be non-negative")
        maximum = float(values.max(initial=0.0))
        if maximum <= 1.0:
            fraction = values
        elif maximum <= 24.0:
            fraction = values / 24.0
        elif maximum <= 86_400.0:
            fraction = values / 86_400.0
        else:
            fraction = np.mod(values, 86_400.0) / 86_400.0
        return np.mod(fraction, 1.0)

    timestamp = _column(data, "timestamp")
    if len(timestamp) != n:
        raise ValueError(f"timestamp has length {len(timestamp)}; expected {n}")

    if np.issubdtype(timestamp.dtype, np.datetime64):
        if np.any(np.isnat(timestamp)):
            raise ValueError("timestamp contains NaT")
        seconds = timestamp.astype("datetime64[ns]").astype(np.int64) / 1e9
    elif np.issubdtype(timestamp.dtype, np.number):
        numeric = _float_vector(timestamp, "timestamp", n)
        magnitude = float(np.max(np.abs(numeric), initial=0.0))
        if magnitude >= 1e17:
            seconds = numeric / 1e9  # nanosecond epoch
        elif magnitude >= 1e14:
            seconds = numeric / 1e6  # microsecond epoch
        elif magnitude >= 1e11:
            seconds = numeric / 1e3  # millisecond epoch
        else:
            seconds = numeric
    else:
        try:
            parsed = timestamp.astype("datetime64[ns]")
        except (TypeError, ValueError) as exc:
            raise ValueError("timestamp must be numeric or datetime-like") from exc
        if np.any(np.isnat(parsed)):
            raise ValueError("timestamp contains invalid or missing values")
        seconds = parsed.astype(np.int64) / 1e9
    return np.mod(seconds, 86_400.0) / 86_400.0


class HazardFeatureBuilder:
    """Build and robustly scale state features without looking at test data.

    The scaler uses the training median and normalized interquartile range.  A
    standard-deviation fallback handles constant or nearly constant features.
    ``fit`` and ``transform`` are intentionally separate so monthly walk-forward
    evaluation cannot accidentally recompute scaling statistics on a test month.
    """

    def __init__(self, clip: float | None = 12.0) -> None:
        if clip is not None and clip <= 0:
            raise ValueError("clip must be positive or None")
        self.clip = clip
        self.center_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    @staticmethod
    def raw_matrix(data: Any) -> np.ndarray:
        distance = _float_vector(_column(data, "distance"), "distance")
        n = len(distance)
        columns = [distance]
        for name in STATE_FEATURE_NAMES[1:7]:
            columns.append(_float_vector(_column(data, name), name, n))

        for index, name in enumerate(
            ("distance", "spread", "volatility", "queue_ahead", "order_age")
        ):
            raw_index = (0, 1, 4, 5, 6)[index]
            if np.any(columns[raw_index] < 0):
                raise ValueError(f"{name} must be non-negative")

        fraction = _time_fraction(data, n)
        angle = 2.0 * np.pi * fraction
        columns.extend((np.sin(angle), np.cos(angle)))
        matrix = np.column_stack(columns)
        if not np.all(np.isfinite(matrix)):
            raise ValueError("state features contain NaN or infinite values")
        return matrix

    def fit(self, data: Any) -> HazardFeatureBuilder:
        raw = self.raw_matrix(data)
        if len(raw) == 0:
            raise ValueError("cannot fit feature builder on an empty sample")
        q25, q75 = np.percentile(raw, (25.0, 75.0), axis=0)
        center = np.median(raw, axis=0)
        scale = (q75 - q25) / 1.349
        standard_deviation = np.std(raw, axis=0)
        scale = np.where(scale > 1e-12, scale, standard_deviation)
        scale = np.where(scale > 1e-12, scale, 1.0)
        self.center_ = center
        self.scale_ = scale
        return self

    def transform(self, data: Any) -> np.ndarray:
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("HazardFeatureBuilder must be fitted before transform")
        raw = self.raw_matrix(data)
        scaled = (raw - self.center_) / self.scale_
        if self.clip is not None:
            scaled = np.clip(scaled, -self.clip, self.clip)
        return np.column_stack((np.ones(len(scaled)), scaled))

    def fit_transform(self, data: Any) -> np.ndarray:
        return self.fit(data).transform(data)


@dataclass(frozen=True)
class SideFitResult:
    side: str
    n_observations: int
    event_count: float
    total_exposure: float
    log_likelihood: float
    converged: bool
    iterations: int


def _validate_observations(
    count: Any, exposure: Any, n: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    count_array = _float_vector(count, "count", n)
    exposure_array = _float_vector(exposure, "exposure", len(count_array))
    if np.any(count_array < 0):
        raise ValueError("count must be non-negative")
    if not np.allclose(count_array, np.round(count_array), atol=1e-10):
        raise ValueError("Poisson count must contain integer-valued observations")
    if np.any(exposure_array <= 0):
        raise ValueError("exposure must be strictly positive")
    return count_array, exposure_array


class StateDependentFillHazard:
    """Separate bid/ask Poisson GLMs with a log exposure offset."""

    def __init__(
        self,
        l2: float = 1e-4,
        max_iter: int = 500,
        feature_builder: HazardFeatureBuilder | None = None,
        max_linear_predictor: float = 30.0,
    ) -> None:
        if l2 < 0:
            raise ValueError("l2 must be non-negative")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if max_linear_predictor <= 0:
            raise ValueError("max_linear_predictor must be positive")
        self.l2 = float(l2)
        self.max_iter = int(max_iter)
        self.feature_builder = feature_builder or HazardFeatureBuilder()
        self.max_linear_predictor = float(max_linear_predictor)
        self.coefficients_: dict[str, np.ndarray] = {}
        self.fit_results_: dict[str, SideFitResult] = {}

    def fit(
        self,
        data: Any,
        count: Any,
        exposure: Any,
        side: Any | None = None,
    ) -> StateDependentFillHazard:
        x = self.feature_builder.fit_transform(data)
        y, duration = _validate_observations(count, exposure, len(x))
        side_array = normalize_side(_column(data, "side") if side is None else side, len(x))
        self.coefficients_.clear()
        self.fit_results_.clear()

        for current_side in SIDES:
            mask = side_array == current_side
            if not np.any(mask):
                continue
            x_side = x[mask]
            y_side = y[mask]
            exposure_side = duration[mask]
            initial = np.zeros(x.shape[1])
            smoothed_rate = (y_side.sum() + 0.5) / (exposure_side.sum() + 0.5)
            initial[0] = np.clip(
                np.log(smoothed_rate),
                -self.max_linear_predictor,
                self.max_linear_predictor,
            )

            penalty = np.ones(x.shape[1])
            penalty[0] = 0.0

            def objective(
                beta: np.ndarray,
                x_side: np.ndarray = x_side,
                y_side: np.ndarray = y_side,
                exposure_side: np.ndarray = exposure_side,
                penalty: np.ndarray = penalty,
            ) -> tuple[float, np.ndarray]:
                eta_raw = x_side @ beta
                eta = np.clip(
                    eta_raw,
                    -self.max_linear_predictor,
                    self.max_linear_predictor,
                )
                mean = exposure_side * np.exp(eta)
                value = np.sum(mean - y_side * eta)
                value += 0.5 * self.l2 * np.dot(penalty * beta, beta)
                active = (eta_raw > -self.max_linear_predictor) & (
                    eta_raw < self.max_linear_predictor
                )
                gradient = x_side.T @ ((mean - y_side) * active)
                gradient += self.l2 * penalty * beta
                return float(value), gradient

            result = minimize(
                objective,
                initial,
                method="L-BFGS-B",
                jac=True,
                bounds=[(-self.max_linear_predictor, self.max_linear_predictor)] * x.shape[1],
                options={"maxiter": self.max_iter, "ftol": 1e-11},
            )
            if not np.all(np.isfinite(result.x)):
                raise RuntimeError(f"non-finite Poisson GLM fit for {current_side}")
            beta = np.asarray(result.x, dtype=float)
            self.coefficients_[current_side] = beta
            rate = np.exp(
                np.clip(
                    x_side @ beta,
                    -self.max_linear_predictor,
                    self.max_linear_predictor,
                )
            )
            self.fit_results_[current_side] = SideFitResult(
                side=current_side,
                n_observations=int(mask.sum()),
                event_count=float(y_side.sum()),
                total_exposure=float(exposure_side.sum()),
                log_likelihood=poisson_log_likelihood(y_side, rate, exposure_side),
                converged=bool(result.success),
                iterations=int(result.nit),
            )
        return self

    def _design_and_side(self, data: Any, side: Any | None = None) -> tuple[np.ndarray, np.ndarray]:
        if not self.coefficients_:
            raise RuntimeError("StateDependentFillHazard must be fitted before prediction")
        x = self.feature_builder.transform(data)
        side_array = normalize_side(_column(data, "side") if side is None else side, len(x))
        unavailable = set(np.unique(side_array)) - self.coefficients_.keys()
        if unavailable:
            names = ", ".join(sorted(unavailable))
            raise ValueError(f"model was not trained for side(s): {names}")
        return x, side_array

    def linear_predictor(self, data: Any, side: Any | None = None) -> np.ndarray:
        x, side_array = self._design_and_side(data, side)
        result = np.empty(len(x), dtype=float)
        for current_side, beta in self.coefficients_.items():
            mask = side_array == current_side
            result[mask] = x[mask] @ beta
        return np.clip(result, -self.max_linear_predictor, self.max_linear_predictor)

    def predict_rate(self, data: Any, side: Any | None = None) -> np.ndarray:
        """Predict the conditional fill intensity per unit exposure."""

        return np.exp(self.linear_predictor(data, side))

    def predict_probability(self, data: Any, exposure: Any, side: Any | None = None) -> np.ndarray:
        rate = self.predict_rate(data, side)
        duration = _float_vector(exposure, "exposure", len(rate))
        if np.any(duration < 0):
            raise ValueError("prediction exposure must be non-negative")
        return fill_probability(rate, duration)


def fill_probability(rate: Any, exposure: Any) -> np.ndarray:
    """Probability of one or more events: ``1 - exp(-rate * exposure)``."""

    rate_array = _float_vector(rate, "rate")
    duration = _float_vector(exposure, "exposure", len(rate_array))
    if np.any(rate_array < 0) or np.any(duration < 0):
        raise ValueError("rate and exposure must be non-negative")
    return -np.expm1(-rate_array * duration)


def poisson_log_likelihood(count: Any, rate: Any, exposure: Any) -> float:
    count_array, duration = _validate_observations(count, exposure)
    rate_array = _float_vector(rate, "rate", len(count_array))
    if np.any(rate_array < 0):
        raise ValueError("rate must be non-negative")
    mean = rate_array * duration
    if np.any((mean == 0) & (count_array > 0)):
        return float("-inf")
    log_mean = np.log(np.maximum(mean, np.finfo(float).tiny))
    terms = count_array * log_mean - mean - gammaln(count_array + 1.0)
    return float(np.sum(terms))


def poisson_deviance(count: Any, rate: Any, exposure: Any) -> float:
    count_array, duration = _validate_observations(count, exposure)
    rate_array = _float_vector(rate, "rate", len(count_array))
    if np.any(rate_array < 0):
        raise ValueError("rate must be non-negative")
    mean = rate_array * duration
    if np.any((mean == 0) & (count_array > 0)):
        return float("inf")
    positive = count_array > 0
    terms = mean - count_array
    terms[positive] += count_array[positive] * np.log(count_array[positive] / mean[positive])
    return float(2.0 * np.sum(terms))


def brier_score(event: Any, probability: Any) -> float:
    observed = np.asarray(event, dtype=float)
    predicted = _float_vector(probability, "probability", observed.size)
    observed = observed.reshape(-1)
    if not np.all((observed == 0) | (observed == 1)):
        raise ValueError("event must be binary")
    if np.any((predicted < 0) | (predicted > 1)):
        raise ValueError("probability must lie in [0, 1]")
    return float(np.mean((observed - predicted) ** 2))


def calibration_curve(event: Any, probability: Any, n_bins: int = 10) -> dict[str, np.ndarray]:
    observed = np.asarray(event, dtype=float).reshape(-1)
    predicted = _float_vector(probability, "probability", len(observed))
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    if not np.all((observed == 0) | (observed == 1)):
        raise ValueError("event must be binary")
    if np.any((predicted < 0) | (predicted > 1)):
        raise ValueError("probability must lie in [0, 1]")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_index = np.minimum(np.searchsorted(edges, predicted, side="right") - 1, n_bins - 1)
    nonempty: list[int] = []
    predicted_mean: list[float] = []
    observed_mean: list[float] = []
    sample_count: list[int] = []
    for index in range(n_bins):
        mask = bin_index == index
        if np.any(mask):
            nonempty.append(index)
            predicted_mean.append(float(np.mean(predicted[mask])))
            observed_mean.append(float(np.mean(observed[mask])))
            sample_count.append(int(mask.sum()))
    return {
        "bin": np.asarray(nonempty, dtype=int),
        "predicted": np.asarray(predicted_mean),
        "observed": np.asarray(observed_mean),
        "count": np.asarray(sample_count, dtype=int),
        "edges": edges,
    }


def evaluate_fill_predictions(
    count: Any,
    rate: Any,
    exposure: Any,
    n_calibration_bins: int = 10,
) -> dict[str, Any]:
    count_array, duration = _validate_observations(count, exposure)
    rate_array = _float_vector(rate, "rate", len(count_array))
    probability = fill_probability(rate_array, duration)
    event = (count_array > 0).astype(float)
    log_likelihood = poisson_log_likelihood(count_array, rate_array, duration)
    return {
        "n_observations": len(count_array),
        "event_count": float(count_array.sum()),
        "total_exposure": float(duration.sum()),
        "poisson_log_likelihood": log_likelihood,
        "mean_poisson_log_likelihood": log_likelihood / len(count_array),
        "poisson_deviance": poisson_deviance(count_array, rate_array, duration),
        "brier_score": brier_score(event, probability),
        "observed_event_frequency": float(event.mean()),
        "predicted_event_frequency": float(probability.mean()),
        "observed_rate": float(count_array.sum() / duration.sum()),
        "predicted_rate": float(np.sum(rate_array * duration) / duration.sum()),
        "calibration": calibration_curve(event, probability, n_calibration_bins),
    }


def competing_risk_labels(
    fill: Any,
    adverse_move: Any,
    cancel: Any,
    *,
    fill_time: Any | None = None,
    adverse_move_time: Any | None = None,
    cancel_time: Any | None = None,
) -> np.ndarray:
    """Label the first observed terminal event for each exposure interval.

    With mutually exclusive indicators no timestamps are needed.  If indicators
    overlap, all three event-time arrays must be supplied and the earliest finite
    event wins.  This prevents arbitrary priority rules from contaminating labels.
    """

    indicators = [
        np.asarray(value, dtype=bool).reshape(-1) for value in (fill, adverse_move, cancel)
    ]
    n = len(indicators[0])
    if any(len(value) != n for value in indicators):
        raise ValueError("competing-risk indicators must have equal length")
    stacked = np.column_stack(indicators)
    labels = np.full(n, "none", dtype="U12")
    overlap = np.sum(stacked, axis=1) > 1

    supplied_times = (fill_time, adverse_move_time, cancel_time)
    if np.any(overlap):
        if any(value is None for value in supplied_times):
            raise ValueError("overlapping risks require all event-time arrays")
        time_columns: list[np.ndarray] = []
        for index, (value, indicator) in enumerate(zip(supplied_times, indicators, strict=True)):
            name = f"{RISK_LABELS[index + 1]}_time"
            event_time = np.asarray(value, dtype=float)
            if event_time.ndim == 0:
                event_time = np.full(n, float(event_time))
            if event_time.ndim != 1 or len(event_time) != n:
                raise ValueError(f"{name} must be one-dimensional with length {n}")
            if np.any(~np.isfinite(event_time[indicator])):
                raise ValueError(f"{name} must be finite when its event occurs")
            if np.any(event_time[indicator] < 0):
                raise ValueError(f"{name} must be non-negative")
            time_columns.append(event_time)
        times = np.column_stack(time_columns)
        times = np.where(stacked, times, np.inf)
        minimum = np.min(times, axis=1)
        tied = overlap & (np.sum(times == minimum[:, None], axis=1) > 1)
        if np.any(tied):
            raise ValueError("simultaneous competing risks cannot be ordered")
        winner = np.argmin(times, axis=1)
        has_event = np.any(stacked, axis=1)
        labels[has_event] = np.asarray(RISK_LABELS[1:])[winner[has_event]]
        return labels

    for label, indicator in zip(RISK_LABELS[1:], indicators, strict=True):
        labels[indicator] = label
    return labels


def performance_metrics(
    *,
    markout: Any | None = None,
    pnl: Any | None = None,
    inventory: Any | None = None,
    tail_probability: float = 0.05,
) -> dict[str, float]:
    """Aggregate markout, net P&L, inventory and left-tail risk statistics."""

    if not 0 < tail_probability < 0.5:
        raise ValueError("tail_probability must lie in (0, 0.5)")
    metrics: dict[str, float] = {}

    if markout is not None:
        values = _float_vector(markout, "markout")
        if len(values):
            metrics.update(
                mean_markout=float(np.mean(values)),
                median_markout=float(np.median(values)),
                adverse_markout_fraction=float(np.mean(values < 0)),
                markout_tail=float(np.quantile(values, tail_probability)),
            )

    if pnl is not None:
        values = _float_vector(pnl, "pnl")
        if len(values):
            tail_cutoff = float(np.quantile(values, tail_probability))
            tail = values[values <= tail_cutoff]
            equity = np.cumsum(values)
            running_maximum = np.maximum.accumulate(np.r_[0.0, equity])
            drawdown = running_maximum[1:] - equity
            metrics.update(
                net_pnl=float(np.sum(values)),
                mean_pnl=float(np.mean(values)),
                pnl_volatility=float(np.std(values)),
                pnl_value_at_risk=float(-tail_cutoff),
                pnl_expected_shortfall=float(-np.mean(tail)),
                max_drawdown=float(np.max(drawdown, initial=0.0)),
            )

    if inventory is not None:
        values = _float_vector(inventory, "inventory")
        if len(values):
            metrics.update(
                inventory_rms=float(np.sqrt(np.mean(values**2))),
                mean_absolute_inventory=float(np.mean(np.abs(values))),
                max_absolute_inventory=float(np.max(np.abs(values))),
                terminal_inventory=float(values[-1]),
            )
    return metrics


def aggregate_oos_metrics(
    count: Any,
    rate: Any,
    exposure: Any,
    *,
    markout: Any | None = None,
    pnl: Any | None = None,
    inventory: Any | None = None,
    n_calibration_bins: int = 10,
) -> dict[str, Any]:
    metrics = evaluate_fill_predictions(count, rate, exposure, n_calibration_bins)
    metrics.update(performance_metrics(markout=markout, pnl=pnl, inventory=inventory))
    return metrics


def _acf(values: np.ndarray, max_lag: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.full(max_lag + 1, np.nan)
    if len(values) == 0:
        return result
    centered = values - np.mean(values)
    denominator = float(np.dot(centered, centered))
    result[0] = 1.0
    if denominator <= np.finfo(float).eps:
        result[1:] = 0.0
        return result
    for lag in range(1, min(max_lag, len(values) - 1) + 1):
        result[lag] = float(np.dot(centered[:-lag], centered[lag:]) / denominator)
    return result


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or len(right) < 2:
        return float("nan")
    if np.std(left) <= np.finfo(float).eps or np.std(right) <= np.finfo(float).eps:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _cross_side_residual_correlation(
    residual: np.ndarray, side: np.ndarray, timestamp: Any | None
) -> tuple[float, int]:
    bid = residual[side == "bid"]
    ask = residual[side == "ask"]
    if timestamp is None:
        n = min(len(bid), len(ask))
        return _safe_correlation(bid[:n], ask[:n]), n

    times = np.asarray(timestamp)
    if times.ndim != 1 or len(times) != len(residual):
        raise ValueError("timestamp must be one-dimensional and match residuals")
    bid_times = times[side == "bid"]
    ask_times = times[side == "ask"]
    common = np.intersect1d(np.unique(bid_times), np.unique(ask_times))
    if not len(common):
        return float("nan"), 0
    bid_aggregate = np.asarray([np.mean(bid[bid_times == value]) for value in common])
    ask_aggregate = np.asarray([np.mean(ask[ask_times == value]) for value in common])
    return _safe_correlation(bid_aggregate, ask_aggregate), len(common)


def residual_diagnostics(
    count: Any,
    rate: Any,
    exposure: Any,
    side: Any,
    *,
    timestamp: Any | None = None,
    n_parameters: int = 0,
    max_lag: int = 10,
    out_of_sample: bool = False,
) -> dict[str, Any]:
    """Pearson dispersion and temporal/cross-side residual diagnostics."""

    if max_lag < 1:
        raise ValueError("max_lag must be at least one")
    count_array, duration = _validate_observations(count, exposure)
    rate_array = _float_vector(rate, "rate", len(count_array))
    if np.any(rate_array < 0):
        raise ValueError("rate must be non-negative")
    side_array = normalize_side(side, len(count_array))
    mean = rate_array * duration
    residual = (count_array - mean) / np.sqrt(np.maximum(mean, np.finfo(float).tiny))
    degrees_of_freedom = max(len(count_array) - int(n_parameters), 1)
    dispersion = float(np.dot(residual, residual) / degrees_of_freedom)
    cross_correlation, paired_count = _cross_side_residual_correlation(
        residual, side_array, timestamp
    )
    result: dict[str, Any] = {
        "out_of_sample": bool(out_of_sample),
        "n_observations": len(count_array),
        "n_parameters": int(n_parameters),
        "dispersion": dispersion,
        "count_acf": _acf(count_array, max_lag),
        "residual_acf": _acf(residual, max_lag),
        "cross_side_residual_correlation": cross_correlation,
        "cross_side_pair_count": paired_count,
        "pearson_residuals": residual,
    }
    for current_side in SIDES:
        mask = side_array == current_side
        result[f"count_acf_{current_side}"] = _acf(count_array[mask], max_lag)
        result[f"residual_acf_{current_side}"] = _acf(residual[mask], max_lag)
    return result


def hawkes_decision_gate(
    diagnostics: Mapping[str, Any],
    *,
    min_abs_correlation: float = 0.10,
    dispersion_threshold: float = 1.50,
    z_score: float = 1.96,
) -> dict[str, Any]:
    """Decide whether residual dependence warrants researching a Hawkes layer.

    The gate can only open on explicitly out-of-sample diagnostics.  Dispersion is
    reported as supporting evidence, but residual auto/cross-correlation is the
    necessary condition; overdispersion alone is more likely a missing state
    variable or misspecified exposure than proof of self-excitation.
    """

    if min_abs_correlation <= 0 or dispersion_threshold <= 0 or z_score <= 0:
        raise ValueError("decision thresholds must be positive")
    n = int(diagnostics.get("n_observations", 0))
    sampling_threshold = z_score / np.sqrt(max(n, 1))
    threshold = max(float(min_abs_correlation), float(sampling_threshold))

    peaks: dict[str, float] = {}
    serial_dependence = False
    for current_side in SIDES:
        acf = np.asarray(diagnostics.get(f"residual_acf_{current_side}", []), dtype=float)
        finite = np.abs(acf[1:][np.isfinite(acf[1:])]) if len(acf) > 1 else np.array([])
        peak = float(np.max(finite)) if len(finite) else 0.0
        peaks[current_side] = peak
        serial_dependence |= peak > threshold

    cross = float(diagnostics.get("cross_side_residual_correlation", np.nan))
    cross_dependence = np.isfinite(cross) and abs(cross) > threshold
    out_of_sample = bool(diagnostics.get("out_of_sample", False))
    warranted = bool(out_of_sample and (serial_dependence or cross_dependence))
    reasons: list[str] = []
    if not out_of_sample:
        reasons.append("diagnostics are not explicitly out of sample")
    if serial_dependence:
        reasons.append("side residual autocorrelation exceeds the sampling threshold")
    if cross_dependence:
        reasons.append("cross-side residual correlation exceeds the sampling threshold")
    dispersion = float(diagnostics.get("dispersion", np.nan))
    if np.isfinite(dispersion) and dispersion > dispersion_threshold:
        reasons.append("residual counts remain overdispersed")
    if out_of_sample and not (serial_dependence or cross_dependence):
        reasons.append("no material out-of-sample residual dependence remains")

    return {
        "hawkes_warranted": warranted,
        "decision": "research_hawkes" if warranted else "keep_state_dependent_poisson",
        "correlation_threshold": threshold,
        "serial_dependence": serial_dependence,
        "cross_side_dependence": cross_dependence,
        "residual_acf_peak": peaks,
        "cross_side_residual_correlation": cross,
        "overdispersed": bool(np.isfinite(dispersion) and dispersion > dispersion_threshold),
        "reasons": tuple(reasons),
    }


__all__ = [
    "FEATURE_NAMES",
    "HazardFeatureBuilder",
    "RISK_LABELS",
    "SIDES",
    "STATE_FEATURE_NAMES",
    "SideFitResult",
    "StateDependentFillHazard",
    "aggregate_oos_metrics",
    "brier_score",
    "calibration_curve",
    "competing_risk_labels",
    "evaluate_fill_predictions",
    "fill_probability",
    "hawkes_decision_gate",
    "normalize_side",
    "performance_metrics",
    "poisson_deviance",
    "poisson_log_likelihood",
    "residual_diagnostics",
]
