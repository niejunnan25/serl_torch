from __future__ import annotations

from typing import Any

import numpy as np


def nested_stack(items: list[Any]) -> Any:
    if not items:
        raise ValueError("nested_stack requires a non-empty items list")
    first = items[0]
    if isinstance(first, dict):
        return {key: nested_stack([item[key] for item in items]) for key in first}
    return np.stack(items, axis=0)


def is_packed_batch(data: Any) -> bool:
    if not isinstance(data, dict) or not data:
        return False
    return _leaf_ndim(data) is not None


def pack_transition_batch(batch_data: list[Any]) -> Any:
    if not batch_data:
        return []
    return nested_stack(batch_data)


def packed_batch_size(batch_data: Any) -> int:
    if isinstance(batch_data, list):
        return len(batch_data)
    if isinstance(batch_data, np.ndarray):
        return int(batch_data.shape[0])
    if isinstance(batch_data, dict):
        leaf = _first_leaf(batch_data)
        if leaf is None:
            return 0
        return int(leaf.shape[0])
    raise TypeError(f"Unsupported batch data type: {type(batch_data).__name__}")


def packed_batch_take(batch_data: Any, index: int) -> Any:
    if isinstance(batch_data, list):
        return batch_data[int(index)]
    if isinstance(batch_data, dict):
        return {
            key: packed_batch_take(value, int(index))
            for key, value in batch_data.items()
        }
    return np.array(batch_data[int(index)], copy=True)


def packed_batch_slice(batch_data: Any, start: int, stop: int) -> Any:
    if isinstance(batch_data, list):
        return list(batch_data[int(start) : int(stop)])
    if isinstance(batch_data, dict):
        return {
            key: packed_batch_slice(value, int(start), int(stop))
            for key, value in batch_data.items()
        }
    return np.array(batch_data[int(start) : int(stop)], copy=True)


def nested_assign_single(dst: Any, src: Any, index: int) -> None:
    if isinstance(dst, dict):
        for key in dst:
            nested_assign_single(dst[key], src[key], int(index))
        return
    dst[int(index)] = src


def nested_assign_slice(
    dst: Any,
    src: Any,
    *,
    dst_start: int,
    dst_stop: int,
    src_start: int,
    src_stop: int,
) -> None:
    if isinstance(dst, dict):
        for key in dst:
            nested_assign_slice(
                dst[key],
                src[key],
                dst_start=int(dst_start),
                dst_stop=int(dst_stop),
                src_start=int(src_start),
                src_stop=int(src_stop),
            )
        return
    dst[int(dst_start) : int(dst_stop)] = src[int(src_start) : int(src_stop)]


def ring_write_batch(dst: Any, src: Any, *, insert_index: int, capacity: int) -> int:
    batch_count = int(packed_batch_size(src))
    if batch_count <= 0:
        return 0
    if batch_count > int(capacity):
        src = packed_batch_slice(src, batch_count - int(capacity), batch_count)
        batch_count = int(capacity)

    first = min(int(batch_count), int(capacity) - int(insert_index))
    nested_assign_slice(
        dst,
        src,
        dst_start=int(insert_index),
        dst_stop=int(insert_index + first),
        src_start=0,
        src_stop=int(first),
    )
    remaining = int(batch_count - first)
    if remaining > 0:
        nested_assign_slice(
            dst,
            src,
            dst_start=0,
            dst_stop=int(remaining),
            src_start=int(first),
            src_stop=int(batch_count),
        )
    return int(batch_count)


def _first_leaf(data: Any) -> np.ndarray | None:
    if isinstance(data, dict):
        for value in data.values():
            leaf = _first_leaf(value)
            if leaf is not None:
                return leaf
        return None
    if isinstance(data, np.ndarray):
        return data
    return np.asarray(data)


def _leaf_ndim(data: Any) -> int | None:
    leaf = _first_leaf(data)
    if leaf is None:
        return None
    return int(leaf.ndim)
