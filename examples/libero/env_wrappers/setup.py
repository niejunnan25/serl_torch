"""LIBERO path setup and environment bootstrap helpers."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from ..utils.paths import _find_serl_repo_root, resolve_repo_candidate


def _package_root(libero_root: Path) -> Path:
    return libero_root / "libero" / "libero"


def _is_complete_libero_root(libero_root: Path) -> bool:
    package_root = _package_root(libero_root)
    required_paths = (
        package_root / "bddl_files",
        package_root / "init_files",
        package_root / "assets" / "scenes" / "libero_tabletop_base_style.xml",
    )
    return all(path.exists() for path in required_paths)


def resolve_openpi_root(openpi_root: Optional[str]) -> Path:
    if openpi_root:
        root = Path(openpi_root).expanduser().resolve()
    else:
        root = resolve_repo_candidate("openpi")
    if not root.exists():
        raise FileNotFoundError(f"openpi root not found: {root}")
    return root


def resolve_libero_root(
    libero_root: Optional[str], openpi_root: Optional[str] = None
) -> Path:
    candidates = []
    if libero_root:
        candidates.append(Path(libero_root).expanduser().resolve())
    else:
        serl_root = _find_serl_repo_root()
        candidates.extend(
            [
                resolve_repo_candidate("LIBERO"),
                resolve_openpi_root(openpi_root) / "third_party" / "LIBERO",
                serl_root / "third_party" / "LIBERO",
            ]
        )

    for candidate in candidates:
        if _is_complete_libero_root(candidate):
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not find a complete LIBERO checkout. "
        f"Checked: {[str(path) for path in candidates]}"
    )


def resolve_libero_config_dir(config_dir: Optional[str]) -> Path:
    if config_dir:
        return Path(config_dir).expanduser().resolve()
    return (
        _find_serl_repo_root() / "examples" / "libero" / ".local" / "libero_config"
    ).resolve()


def resolve_libero_datasets_root(
    dataset_root: Optional[str], libero_root: Optional[Path] = None
) -> Path:
    if dataset_root:
        return Path(dataset_root).expanduser().resolve()

    env_override = os.environ.get("LIBERO_DATASETS_ROOT")
    candidates = []
    if env_override:
        candidates.append(Path(env_override).expanduser().resolve())

    serl_root = _find_serl_repo_root()
    candidates.extend(
        [
            (serl_root.parent.parent / "datasets").resolve(),
            (serl_root.parent / "datasets").resolve(),
        ]
    )
    if libero_root is not None:
        candidates.append((_package_root(libero_root).parent / "datasets").resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def write_libero_config(
    libero_root: Path,
    config_dir: Path,
    datasets_root: Optional[Path] = None,
) -> Path:
    package_root = _package_root(libero_root)
    datasets_root = (
        datasets_root.resolve()
        if datasets_root is not None
        else resolve_libero_datasets_root(None, libero_root=libero_root)
    )
    config_text = (
        f"benchmark_root: {package_root.as_posix()}\n"
        f"bddl_files: {(package_root / 'bddl_files').as_posix()}\n"
        f"init_states: {(package_root / 'init_files').as_posix()}\n"
        f"datasets: {datasets_root.as_posix()}\n"
        f"assets: {(package_root / 'assets').as_posix()}\n"
    )
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.yaml"
    if not config_path.exists() or config_path.read_text() != config_text:
        config_path.write_text(config_text)
    return config_path


def setup_libero_pythonpath(
    libero_root: Path,
    config_dir: Path,
    datasets_root: Optional[Path] = None,
) -> Path:
    config_path = write_libero_config(
        libero_root, config_dir, datasets_root=datasets_root
    )
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    if str(libero_root) not in sys.path:
        sys.path.insert(0, str(libero_root))
    return config_path


def setup_openpi_client_pythonpath(openpi_root: Path) -> Path:
    client_src = openpi_root / "packages" / "openpi-client" / "src"
    if not client_src.exists():
        raise FileNotFoundError(f"openpi client src not found: {client_src}")
    if str(client_src) not in sys.path:
        sys.path.insert(0, str(client_src))
    return client_src


def resolve_max_episode_steps(suite_name: str) -> int:
    suite_name = str(suite_name)
    if suite_name == "libero_spatial":
        return 220
    if suite_name == "libero_object":
        return 280
    if suite_name == "libero_goal":
        return 300
    if suite_name == "libero_10":
        return 520
    if suite_name == "libero_90":
        return 400
    raise ValueError(f"Unknown LIBERO suite name: {suite_name}")
