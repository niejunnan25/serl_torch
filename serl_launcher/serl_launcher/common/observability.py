from __future__ import annotations

"""Common helpers for experiment observability and metric wiring."""

from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any


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
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            metrics[str(metric_key)] = float(value)
    return metrics
