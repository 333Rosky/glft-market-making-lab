"""Leakage-safe monthly walk-forward evaluation for fill hazard models."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .hazard import (
    StateDependentFillHazard,
    aggregate_oos_metrics,
    fill_probability,
    hawkes_decision_gate,
    normalize_side,
    residual_diagnostics,
)


@dataclass(frozen=True)
class MonthlySplit:
    """Indices for one test month and its strictly earlier training months."""

    test_month: np.datetime64
    train_months: tuple[np.datetime64, ...]
    train_indices: np.ndarray
    test_indices: np.ndarray


@dataclass
class WalkForwardFold:
    test_month: np.datetime64
    train_months: tuple[np.datetime64, ...]
    train_indices: np.ndarray
    test_indices: np.ndarray
    rate: np.ndarray
    probability: np.ndarray
    metrics: dict[str, Any]
    model: Any


@dataclass
class WalkForwardResult:
    """Predictions remain aligned to original rows; pre-OOS rows contain NaN."""

    folds: tuple[WalkForwardFold, ...]
    rate: np.ndarray
    probability: np.ndarray
    oos_mask: np.ndarray
    metrics: dict[str, Any]
    diagnostics: dict[str, Any]
    hawkes_gate: dict[str, Any]

    @property
    def oos_indices(self) -> np.ndarray:
        return np.flatnonzero(self.oos_mask)


def _as_months(timestamp: Any) -> np.ndarray:
    values = np.asarray(timestamp)
    if values.ndim == 0:
        values = values.reshape(1)
    if values.ndim != 1:
        raise ValueError("timestamp must be one-dimensional")
    if not len(values):
        return np.asarray([], dtype="datetime64[M]")

    if np.issubdtype(values.dtype, np.datetime64):
        parsed = values.astype("datetime64[ns]")
    elif np.issubdtype(values.dtype, np.number):
        numeric = np.asarray(values, dtype=float)
        if not np.all(np.isfinite(numeric)):
            raise ValueError("timestamp contains NaN or infinite values")
        magnitude = float(np.max(np.abs(numeric), initial=0.0))
        if magnitude >= 1e17:
            nanoseconds = numeric.astype(np.int64)
        elif magnitude >= 1e14:
            nanoseconds = (numeric * 1e3).astype(np.int64)  # microseconds
        elif magnitude >= 1e11:
            nanoseconds = (numeric * 1e6).astype(np.int64)  # milliseconds
        else:
            nanoseconds = (numeric * 1e9).astype(np.int64)  # seconds
        parsed = nanoseconds.astype("datetime64[ns]")
    else:
        try:
            parsed = values.astype("datetime64[ns]")
        except (TypeError, ValueError) as exc:
            raise ValueError("timestamp must be numeric or datetime-like") from exc
    if np.any(np.isnat(parsed)):
        raise ValueError("timestamp contains invalid or missing values")
    return parsed.astype("datetime64[M]")


def monthly_splits(
    timestamp: Any,
    *,
    min_train_months: int = 1,
    mode: str = "expanding",
    train_window_months: int | None = None,
) -> tuple[MonthlySplit, ...]:
    """Create chronological calendar-month splits with no train/test overlap.

    ``expanding`` uses every month before the test month.  ``rolling`` uses only
    the most recent ``train_window_months`` months.  Rows may arrive unsorted;
    returned indices refer to their original positions.
    """

    if min_train_months < 1:
        raise ValueError("min_train_months must be at least one")
    if mode not in {"expanding", "rolling"}:
        raise ValueError("mode must be 'expanding' or 'rolling'")
    if mode == "rolling":
        if train_window_months is None or train_window_months < min_train_months:
            raise ValueError("rolling mode requires train_window_months >= min_train_months")
    elif train_window_months is not None:
        raise ValueError("train_window_months is only valid in rolling mode")

    months = _as_months(timestamp)
    unique_months = np.unique(months)
    splits: list[MonthlySplit] = []
    for position in range(min_train_months, len(unique_months)):
        test_month = unique_months[position]
        available = unique_months[:position]
        if mode == "rolling":
            available = available[-int(train_window_months) :]
        if len(available) < min_train_months:
            continue
        train_indices = np.flatnonzero(np.isin(months, available))
        test_indices = np.flatnonzero(months == test_month)
        if len(train_indices) and not np.all(months[train_indices] < test_month):
            raise AssertionError("internal error: training data reached the test month")
        splits.append(
            MonthlySplit(
                test_month=test_month,
                train_months=tuple(available),
                train_indices=train_indices,
                test_indices=test_indices,
            )
        )
    return tuple(splits)


def _slice_rows(data: Any, indices: np.ndarray, n: int) -> Any:
    if isinstance(data, Mapping):
        sliced: dict[str, Any] = {}
        for name, values in data.items():
            array = np.asarray(values)
            if array.ndim == 0:
                sliced[name] = values
            elif len(array) == n:
                sliced[name] = array[indices]
            else:
                raise ValueError(f"data column {name!r} has length {len(array)}; expected {n}")
        return sliced
    array = np.asarray(data)
    if array.ndim == 0 or len(array) != n:
        raise ValueError("data must have one row per timestamp")
    return array[indices]


def _vector(value: Any, name: str, n: int, *, allow_nan: bool = False) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full(n, float(array))
    if array.ndim != 1 or len(array) != n:
        raise ValueError(f"{name} must be one-dimensional with length {n}")
    if np.any(np.isinf(array)) or (not allow_nan and np.any(np.isnan(array))):
        raise ValueError(f"{name} contains invalid values")
    return array


def _data_column(data: Any, name: str) -> np.ndarray:
    if isinstance(data, Mapping):
        if name not in data:
            raise KeyError(f"missing required column: {name}")
        return np.asarray(data[name])
    if isinstance(data, np.ndarray) and data.dtype.names and name in data.dtype.names:
        return np.asarray(data[name])
    try:
        return np.asarray(data[name])
    except (KeyError, TypeError, IndexError) as exc:
        raise KeyError(f"missing required column: {name}") from exc


def _metric_slice(
    values: np.ndarray | None, indices: np.ndarray, event_mask: np.ndarray | None = None
) -> np.ndarray | None:
    if values is None:
        return None
    selected = values[indices]
    if event_mask is not None:
        selected = selected[event_mask]
    return selected[np.isfinite(selected)]


def walk_forward_monthly(
    data: Any,
    count: Any,
    exposure: Any,
    *,
    timestamp: Any | None = None,
    side: Any | None = None,
    min_train_months: int = 1,
    mode: str = "expanding",
    train_window_months: int | None = None,
    model_factory: Callable[[], Any] | None = None,
    markout: Any | None = None,
    pnl: Any | None = None,
    inventory: Any | None = None,
    n_calibration_bins: int = 10,
    max_residual_lag: int = 10,
) -> WalkForwardResult:
    """Refit once per month and evaluate only the immediately held-out month.

    A fresh model (and therefore a fresh feature scaler) is created for every
    fold.  Neither feature normalization nor coefficient estimation can observe
    the test month.  Markouts are aggregated only on rows with at least one fill.
    """

    timestamps = _data_column(data, "timestamp") if timestamp is None else np.asarray(timestamp)
    months = _as_months(timestamps)
    n = len(months)
    counts = _vector(count, "count", n)
    durations = _vector(exposure, "exposure", n)
    if np.any(counts < 0) or not np.allclose(counts, np.round(counts), atol=1e-10):
        raise ValueError("count must contain non-negative integer values")
    if np.any(durations <= 0):
        raise ValueError("exposure must be strictly positive")
    sides = normalize_side(_data_column(data, "side") if side is None else side, n)

    markout_values = None if markout is None else _vector(markout, "markout", n, allow_nan=True)
    pnl_values = None if pnl is None else _vector(pnl, "pnl", n, allow_nan=True)
    inventory_values = (
        None if inventory is None else _vector(inventory, "inventory", n, allow_nan=True)
    )

    splits = monthly_splits(
        timestamps,
        min_train_months=min_train_months,
        mode=mode,
        train_window_months=train_window_months,
    )
    if not splits:
        raise ValueError("no eligible test month after applying training constraints")

    factory = model_factory or StateDependentFillHazard
    rate = np.full(n, np.nan)
    probability = np.full(n, np.nan)
    folds: list[WalkForwardFold] = []
    for split in splits:
        train_data = _slice_rows(data, split.train_indices, n)
        test_data = _slice_rows(data, split.test_indices, n)
        model = factory()
        if side is None:
            model.fit(
                train_data,
                counts[split.train_indices],
                durations[split.train_indices],
            )
            fold_rate = np.asarray(model.predict_rate(test_data), dtype=float)
        else:
            model.fit(
                train_data,
                counts[split.train_indices],
                durations[split.train_indices],
                side=sides[split.train_indices],
            )
            fold_rate = np.asarray(
                model.predict_rate(test_data, side=sides[split.test_indices]), dtype=float
            )
        if fold_rate.ndim != 1 or len(fold_rate) != len(split.test_indices):
            raise ValueError("model.predict_rate returned an incompatible shape")
        if np.any(~np.isfinite(fold_rate)) or np.any(fold_rate < 0):
            raise ValueError("model.predict_rate returned invalid rates")
        fold_probability = fill_probability(fold_rate, durations[split.test_indices])
        rate[split.test_indices] = fold_rate
        probability[split.test_indices] = fold_probability
        test_events = counts[split.test_indices] > 0
        fold_metrics = aggregate_oos_metrics(
            counts[split.test_indices],
            fold_rate,
            durations[split.test_indices],
            markout=_metric_slice(markout_values, split.test_indices, test_events),
            pnl=_metric_slice(pnl_values, split.test_indices),
            inventory=_metric_slice(inventory_values, split.test_indices),
            n_calibration_bins=n_calibration_bins,
        )
        folds.append(
            WalkForwardFold(
                test_month=split.test_month,
                train_months=split.train_months,
                train_indices=split.train_indices.copy(),
                test_indices=split.test_indices.copy(),
                rate=fold_rate,
                probability=fold_probability,
                metrics=fold_metrics,
                model=model,
            )
        )

    oos_mask = np.isfinite(rate)
    oos_indices = np.flatnonzero(oos_mask)
    chronological_order = np.argsort(
        np.asarray(timestamps)[oos_indices].astype("datetime64[ns]"), kind="stable"
    )
    ordered_indices = oos_indices[chronological_order]
    event_rows = counts[ordered_indices] > 0
    metrics = aggregate_oos_metrics(
        counts[ordered_indices],
        rate[ordered_indices],
        durations[ordered_indices],
        markout=_metric_slice(markout_values, ordered_indices, event_rows),
        pnl=_metric_slice(pnl_values, ordered_indices),
        inventory=_metric_slice(inventory_values, ordered_indices),
        n_calibration_bins=n_calibration_bins,
    )
    diagnostics = residual_diagnostics(
        counts[ordered_indices],
        rate[ordered_indices],
        durations[ordered_indices],
        sides[ordered_indices],
        timestamp=np.asarray(timestamps)[ordered_indices],
        max_lag=max_residual_lag,
        out_of_sample=True,
    )
    gate = hawkes_decision_gate(diagnostics)
    return WalkForwardResult(
        folds=tuple(folds),
        rate=rate,
        probability=probability,
        oos_mask=oos_mask,
        metrics=metrics,
        diagnostics=diagnostics,
        hawkes_gate=gate,
    )


# More discoverable alias for callers who naturally search for this name.
monthly_walk_forward = walk_forward_monthly


__all__ = [
    "MonthlySplit",
    "WalkForwardFold",
    "WalkForwardResult",
    "monthly_splits",
    "monthly_walk_forward",
    "walk_forward_monthly",
]
