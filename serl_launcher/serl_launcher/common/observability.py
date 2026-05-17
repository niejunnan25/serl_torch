from __future__ import annotations

"""Common helpers for experiment observability and metric wiring."""

from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any


def _as_float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "detach") and callable(value.detach):
        try:
            value = value.detach()
        except Exception:  # noqa: BLE001
            return None
    if hasattr(value, "numel") and callable(value.numel):
        try:
            if int(value.numel()) != 1:
                return None
        except Exception:  # noqa: BLE001
            return None
    if hasattr(value, "item") and callable(value.item):
        try:
            item = value.item()
        except Exception:  # noqa: BLE001
            return None
        if isinstance(item, bool):
            return None
        if isinstance(item, (int, float)):
            return float(item)
    return None


def define_metric_group(
    run: Any,
    *,
    axis_metric: str,
    metric_names: Sequence[str],
    hide_axis_metric: bool = False,
) -> None:
    """Register a metric axis and bind a set of metrics to that axis."""

    define_metric = getattr(run, "define_metric", None)
    if define_metric is None:
        return

    if bool(hide_axis_metric):
        try:
            define_metric(axis_metric, hidden=True)
        except TypeError:
            define_metric(axis_metric)
    else:
        define_metric(axis_metric)
    for metric_name in metric_names:
        define_metric(metric_name, step_metric=axis_metric)


def extract_numeric_metrics(
    payload: Mapping[str, Any],
    mapping: Mapping[str, str] | Sequence[tuple[str, str]],
) -> dict[str, float]:
    """Extract numeric values from a payload and rename them for logging."""

    items = mapping.items() if isinstance(mapping, Mapping) else mapping
    metrics: dict[str, float] = {}
    for payload_key, metric_key in items:
        value = payload.get(payload_key, None)
        numeric_value = _as_float_or_none(value)
        if numeric_value is not None:
            metrics[str(metric_key)] = float(numeric_value)
    return metrics
