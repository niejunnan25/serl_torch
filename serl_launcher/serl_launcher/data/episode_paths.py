"""Helpers for resolving episode dataset files from config paths."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence


def _coerce_path_items(
    dataset_paths: str | Path | Sequence[str | Path] | None,
) -> Iterable[str | Path]:
    if dataset_paths is None:
        return ()
    if isinstance(dataset_paths, (str, Path)):
        return (dataset_paths,)
    return tuple(dataset_paths)


def _resolve_episode_file_from_manifest(
    episode_file: str | Path,
    *,
    manifest_path: Path,
    base_dir: Path,
    manifest_paths_relative_to_manifest: bool,
) -> Path:
    episode_path = Path(str(episode_file)).expanduser()
    if episode_path.is_absolute():
        return episode_path.resolve()

    root_dir = manifest_path.parent if manifest_paths_relative_to_manifest else base_dir
    return (root_dir / episode_path).resolve()


def _append_manifest_episode_files(
    resolved: List[Path],
    *,
    manifest_path: Path,
    base_dir: Path,
    manifest_paths_relative_to_manifest: bool,
) -> None:
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for episode_file in manifest.get("episode_files", []):
        resolved.append(
            _resolve_episode_file_from_manifest(
                episode_file,
                manifest_path=manifest_path,
                base_dir=base_dir,
                manifest_paths_relative_to_manifest=manifest_paths_relative_to_manifest,
            )
        )


def resolve_episode_files(
    dataset_paths: str | Path | Sequence[str | Path] | None,
    *,
    base_dir: Path,
    manifest_name: str = "manifest.json",
    episode_glob: str = "episode_*.pkl",
    manifest_paths_relative_to_manifest: bool = True,
) -> List[Path]:
    resolved: List[Path] = []

    for item in _coerce_path_items(dataset_paths):
        candidate = Path(str(item)).expanduser()
        if not candidate.is_absolute():
            candidate = (base_dir / candidate).resolve()
        else:
            candidate = candidate.resolve()

        if candidate.is_file():
            if candidate.name == manifest_name:
                _append_manifest_episode_files(
                    resolved,
                    manifest_path=candidate,
                    base_dir=base_dir,
                    manifest_paths_relative_to_manifest=(
                        manifest_paths_relative_to_manifest
                    ),
                )
            elif candidate.suffix == ".pkl":
                resolved.append(candidate)
            continue

        if candidate.is_dir():
            manifest_path = candidate / manifest_name
            if manifest_path.exists():
                _append_manifest_episode_files(
                    resolved,
                    manifest_path=manifest_path,
                    base_dir=base_dir,
                    manifest_paths_relative_to_manifest=(
                        manifest_paths_relative_to_manifest
                    ),
                )
            else:
                resolved.extend(sorted(path.resolve() for path in candidate.glob(episode_glob)))

    deduped: List[Path] = []
    seen = set()
    for path in resolved:
        path_key = str(path)
        if path_key in seen:
            continue
        deduped.append(path)
        seen.add(path_key)
    return deduped
