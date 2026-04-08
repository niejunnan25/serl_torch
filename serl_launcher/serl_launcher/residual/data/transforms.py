"""OpenPI-style transform primitives for residual data materialization."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import dataclasses
from typing import Any, Protocol, TypeAlias, runtime_checkable

DataDict: TypeAlias = dict[str, Any]


@runtime_checkable
class DataTransformFn(Protocol):
    def __call__(self, data: DataDict) -> DataDict:
        """Apply a transform and return the transformed dictionary."""


@dataclasses.dataclass(frozen=True)
class Group:
    """A small transform group similar to openpi.transforms.Group."""

    inputs: Sequence[DataTransformFn] = ()

    def push(self, *, inputs: Sequence[DataTransformFn] = ()) -> "Group":
        return Group(inputs=(*self.inputs, *inputs))


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        out = data
        for transform in self.transforms:
            out = transform(out)
        return out


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    return CompositeTransform(tuple(transforms))


def flatten_dict(data: Mapping[str, Any], *, sep: str = "/") -> dict[str, Any]:
    flat: dict[str, Any] = {}

    def _walk(node: Mapping[str, Any], prefix: str) -> None:
        for key, value in node.items():
            path = f"{prefix}{sep}{key}" if prefix else str(key)
            if isinstance(value, Mapping):
                _walk(value, path)
            else:
                flat[path] = value

    _walk(data, "")
    return flat


def get_by_path(data: Mapping[str, Any], path: str) -> Any:
    current: Any = data
    for part in str(path).split("/"):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(f"Path not found: {path!r}")
        current = current[part]
    return current


def set_by_path(data: DataDict, path: str, value: Any) -> DataDict:
    parts = str(path).split("/")
    current: DataDict = data
    for part in parts[:-1]:
        next_value = current.get(part, None)
        if not isinstance(next_value, dict):
            next_value = {}
            current[part] = next_value
        current = next_value
    current[parts[-1]] = value
    return data


def delete_by_path(data: DataDict, path: str) -> None:
    parts = str(path).split("/")
    current: Any = data
    for part in parts[:-1]:
        if not isinstance(current, Mapping) or part not in current:
            return
        current = current[part]
    if isinstance(current, dict):
        current.pop(parts[-1], None)


def copy_data(data: Mapping[str, Any]) -> DataDict:
    return copy.deepcopy(dict(data))


def _materialize_structure(
    structure: Any,
    flat_data: Mapping[str, Any],
    source_data: Mapping[str, Any],
) -> Any:
    if isinstance(structure, Mapping):
        return {
            str(key): _materialize_structure(value, flat_data, source_data)
            for key, value in structure.items()
        }
    if not isinstance(structure, str):
        raise TypeError(
            "RepackTransform leaves must be strings pointing to source paths, "
            f"got {type(structure)!r}"
        )
    if structure not in flat_data:
        try:
            return copy.deepcopy(get_by_path(source_data, structure))
        except KeyError as exc:
            raise KeyError(
                f"Missing source path {structure!r} for repack transform"
            ) from exc
    return copy.deepcopy(flat_data[structure])


@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """Repacks a raw dictionary into a nested canonical structure."""

    structure: Any

    def __call__(self, data: DataDict) -> DataDict:
        return _materialize_structure(self.structure, flatten_dict(data), data)


@dataclasses.dataclass(frozen=True)
class CallableTransform(DataTransformFn):
    """Wrap a plain function as a transform."""

    fn: Any

    def __call__(self, data: DataDict) -> DataDict:
        return self.fn(data)
