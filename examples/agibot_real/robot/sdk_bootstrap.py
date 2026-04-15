"""Repo-local bootstrap for vendored AgiBot SDK assets."""
from __future__ import annotations

import logging
import os
import platform
import shutil
import sys
import tarfile
from pathlib import Path
from zipfile import ZipFile


LOGGER = logging.getLogger(__name__)

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = EXAMPLE_ROOT / "vendor" / "a2d_sdk"
WHEELS_ROOT = VENDOR_ROOT / "wheels"
SITE_ROOT = VENDOR_ROOT / "_site"
ROBOT_SERVICE_ROOT = EXAMPLE_ROOT / "robot" / "service"
FORWARDER_TAR = VENDOR_ROOT / "forwarder_x86_v1.7.0.tar.gz"
FORWARDER_ROOT = ROBOT_SERVICE_ROOT / "forwarder"
FORWARDER_DIR_ENV = "AGIBOT_FORWARDER_DIR"
FORWARDER_TAR_ENV = "AGIBOT_FORWARDER_TAR"

SUPPORTED_PYTHON = (3, 10)
BASE_WHEELS = (
    "a2d_sdk-1.5.0-py3-none-any.whl",
    "genie_msgs_pb-0.8.0-py3-none-any.whl",
)
ARCH_WHEELS = {
    "x86_64": "cosine_bus-2.0.0-cp310-cp310-linux_x86_64.whl",
    "aarch64": "cosine_bus-2.0.0-cp310-cp310-linux_aarch64.whl",
}


def _canonical_machine() -> str:
    machine = platform.machine().strip().lower()
    if machine in {"x86_64", "amd64"}:
        return "x86_64"
    if machine in {"aarch64", "arm64"}:
        return "aarch64"
    raise RuntimeError(
        f"Unsupported machine for vendored AgiBot SDK bootstrap: {machine!r}"
    )


def _ensure_supported_python() -> None:
    if sys.version_info[:2] != SUPPORTED_PYTHON:
        major, minor = SUPPORTED_PYTHON
        raise RuntimeError(
            "Repo-local AgiBot SDK bootstrap currently requires Python "
            f"{major}.{minor}, but the active interpreter is "
            f"{sys.version_info.major}.{sys.version_info.minor}. "
            "The vendored cosine_bus wheel is CPython 3.10-specific."
        )


def _site_dir() -> Path:
    machine = _canonical_machine()
    return SITE_ROOT / f"py{SUPPORTED_PYTHON[0]}{SUPPORTED_PYTHON[1]}_{machine}"


def _required_wheels() -> list[Path]:
    machine = _canonical_machine()
    names = list(BASE_WHEELS) + [ARCH_WHEELS[machine]]
    return [WHEELS_ROOT / name for name in names]


def _needs_extract(site_dir: Path) -> bool:
    required = (
        site_dir / "a2d_sdk" / "__init__.py",
        site_dir / "cosine_bus" / "__init__.py",
        site_dir / "genie_msgs_pb" / "__init__.py",
    )
    return not all(path.is_file() for path in required)


def _resolve_member_path(member_name: str) -> str | None:
    if not member_name or member_name.endswith("/"):
        return None
    purelib_marker = ".data/purelib/"
    if purelib_marker in member_name:
        return member_name.split(purelib_marker, 1)[1]
    if ".data/" in member_name:
        return None
    return member_name


def _extract_wheel(wheel_path: Path, site_dir: Path) -> None:
    if not wheel_path.is_file():
        raise RuntimeError(f"Vendored AgiBot SDK wheel not found: {wheel_path}")
    LOGGER.info("Extracting vendored AgiBot SDK wheel: %s", wheel_path.name)
    with ZipFile(wheel_path) as zf:
        for info in zf.infolist():
            rel_path = _resolve_member_path(info.filename)
            if rel_path is None:
                continue
            out_path = site_dir / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            mode = info.external_attr >> 16
            if mode:
                out_path.chmod(mode)


def ensure_repo_local_a2d_sdk() -> Path:
    """Make the vendored AgiBot SDK importable from this repo."""
    _ensure_supported_python()
    site_dir = _site_dir()
    site_dir.mkdir(parents=True, exist_ok=True)
    if _needs_extract(site_dir):
        for wheel_path in _required_wheels():
            _extract_wheel(wheel_path, site_dir)
    if str(site_dir) not in sys.path:
        sys.path.insert(0, str(site_dir))
    return site_dir


def _forwarder_marker(root: Path) -> Path:
    return root / "app" / "bin" / "forwarder"


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _validate_forwarder_root(root: Path, *, source_desc: str) -> Path:
    marker = _forwarder_marker(root)
    if not marker.is_file():
        raise RuntimeError(
            "Forwarder bundle is missing the expected executable "
            f"`{marker}` from {source_desc}."
        )
    return root


def _link_forwarder_root(source_root: Path) -> Path:
    source_root = _validate_forwarder_root(
        source_root.expanduser().resolve(),
        source_desc=f"{FORWARDER_DIR_ENV}={source_root}",
    )
    try:
        if FORWARDER_ROOT.is_symlink() and FORWARDER_ROOT.resolve() == source_root:
            return FORWARDER_ROOT
    except FileNotFoundError:
        pass
    if FORWARDER_ROOT.exists() or FORWARDER_ROOT.is_symlink():
        _remove_path(FORWARDER_ROOT)
    FORWARDER_ROOT.parent.mkdir(parents=True, exist_ok=True)
    try:
        FORWARDER_ROOT.symlink_to(source_root, target_is_directory=True)
    except OSError:
        shutil.copytree(source_root, FORWARDER_ROOT)
    return FORWARDER_ROOT


def _extract_forwarder_tar(tar_path: Path) -> Path:
    tar_path = tar_path.expanduser().resolve()
    if not tar_path.is_file():
        raise RuntimeError(f"Forwarder tarball not found: {tar_path}")
    if FORWARDER_ROOT.exists() or FORWARDER_ROOT.is_symlink():
        _remove_path(FORWARDER_ROOT)
    FORWARDER_ROOT.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Extracting AgiBot forwarder bundle: %s", tar_path)
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(FORWARDER_ROOT)
    return _validate_forwarder_root(FORWARDER_ROOT, source_desc=str(tar_path))


def ensure_repo_local_forwarder() -> Path:
    """Resolve the forwarder bundle expected by ros_env_wrapper.sh.

    Resolution order:
    1. `AGIBOT_FORWARDER_DIR`
    2. `AGIBOT_FORWARDER_TAR`
    3. existing `robot/service/forwarder`
    4. local repo cache `vendor/a2d_sdk/forwarder_x86_v1.7.0.tar.gz`
    """
    machine = _canonical_machine()
    env_forwarder_dir = os.environ.get(FORWARDER_DIR_ENV, "").strip()
    if env_forwarder_dir:
        return _link_forwarder_root(Path(env_forwarder_dir))

    env_forwarder_tar = os.environ.get(FORWARDER_TAR_ENV, "").strip()
    if env_forwarder_tar:
        if machine != "x86_64":
            raise RuntimeError(
                f"{FORWARDER_TAR_ENV} currently expects an x86_64 forwarder bundle. "
                "Use AGIBOT_FORWARDER_DIR for a pre-extracted bundle on this machine, "
                "or start robot-service with --no-ros."
            )
        return _extract_forwarder_tar(Path(env_forwarder_tar))

    marker = _forwarder_marker(FORWARDER_ROOT)
    if marker.is_file():
        return FORWARDER_ROOT

    if FORWARDER_TAR.is_file():
        if machine != "x86_64":
            raise RuntimeError(
                "The repo-local cached forwarder tarball is x86_64-only. "
                "Use AGIBOT_FORWARDER_DIR for a prepared bundle on this machine, "
                "or start robot-service with --no-ros."
            )
        return _extract_forwarder_tar(FORWARDER_TAR)

    raise RuntimeError(
        "AgiBot forwarder bundle not found. Provide one of:\n"
        f"  1. {FORWARDER_DIR_ENV}=/path/to/extracted/forwarder\n"
        f"  2. {FORWARDER_TAR_ENV}=/path/to/forwarder_x86_v1.7.0.tar.gz\n"
        "  3. a prepared robot/service/forwarder directory\n"
        "  4. a local cache at examples/agibot_real/vendor/a2d_sdk/forwarder_x86_v1.7.0.tar.gz\n"
        "Or start robot-service with --no-ros / AGIBOT_NO_ROS=1."
    )
