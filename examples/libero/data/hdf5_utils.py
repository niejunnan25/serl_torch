"""Helpers for resolving LIBERO HDF5 demonstration files."""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Iterable, List, Optional

from ..env_wrappers import (
    resolve_libero_config_dir,
    resolve_libero_datasets_root,
    resolve_libero_root,
    setup_libero_pythonpath,
)


@dataclasses.dataclass(frozen=True)
class LiberoTaskSpec:
    suite_name: str
    task_id: int
    task_name: str
    task_description: str
    dataset_path: Path

    @property
    def task_key(self) -> str:
        return f"{self.suite_name}_task_{self.task_id}"


def _candidate_dataset_paths(datasets_root: Path, suite_name: str, task_name: str) -> Iterable[Path]:
    filename = f"{task_name}_demo.hdf5"
    yield (datasets_root / suite_name / filename).resolve()
    yield (datasets_root / filename).resolve()


def resolve_task_specs(
    *,
    suite_name: str,
    task_ids: Optional[Iterable[int]] = None,
    libero_root: Optional[str] = None,
    openpi_root: Optional[str] = None,
    libero_config_dir: Optional[str] = None,
    libero_datasets_root: Optional[str] = None,
) -> List[LiberoTaskSpec]:
    resolved_libero_root = resolve_libero_root(libero_root, openpi_root=openpi_root)
    resolved_config_dir = resolve_libero_config_dir(libero_config_dir)
    resolved_datasets_root = resolve_libero_datasets_root(
        libero_datasets_root,
        libero_root=resolved_libero_root,
    )
    setup_libero_pythonpath(
        resolved_libero_root,
        resolved_config_dir,
        datasets_root=resolved_datasets_root,
    )

    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[str(suite_name)]()

    if task_ids is None:
        selected_task_ids = list(range(task_suite.n_tasks))
    else:
        selected_task_ids = [int(task_id) for task_id in task_ids]

    specs: List[LiberoTaskSpec] = []
    for task_id in selected_task_ids:
        task = task_suite.get_task(task_id)
        dataset_path = None
        for candidate in _candidate_dataset_paths(resolved_datasets_root, str(suite_name), str(task.name)):
            if candidate.exists():
                dataset_path = candidate
                break
        if dataset_path is None:
            dataset_path = next(_candidate_dataset_paths(resolved_datasets_root, str(suite_name), str(task.name)))

        specs.append(
            LiberoTaskSpec(
                suite_name=str(suite_name),
                task_id=int(task_id),
                task_name=str(task.name),
                task_description=str(task.language),
                dataset_path=dataset_path,
            )
        )
    return specs

