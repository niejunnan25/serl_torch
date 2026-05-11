from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

REPO_PARENT = Path(__file__).resolve().parents[4]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))
SERL_LAUNCHER_ROOT = Path(__file__).resolve().parents[2]
if str(SERL_LAUNCHER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERL_LAUNCHER_ROOT))

from serl_launcher.policy.typed_factory import describe_policy_backend
from serl_launcher.policy.typed_factory import resolve_policy_backend_id
from serl_launcher.policy.typed_factory import resolve_policy_backend_type


def _cfg(policy_type: str, policy_id: str | None) -> SimpleNamespace:
    return SimpleNamespace(policy=SimpleNamespace(type=policy_type, id=policy_id))


def test_policy_backend_helpers_normalize_type_and_id() -> None:
    cfg = _cfg(" JoyRA ", " office_setting ")

    assert resolve_policy_backend_type(cfg) == "joyra"
    assert resolve_policy_backend_id(cfg) == "office_setting"
    assert describe_policy_backend(cfg) == "joyra:office_setting"


def test_policy_backend_helpers_use_type_when_id_is_empty() -> None:
    cfg = _cfg("openpi", " ")

    assert resolve_policy_backend_type(cfg) == "openpi"
    assert resolve_policy_backend_id(cfg) == "openpi"
    assert describe_policy_backend(cfg) == "openpi"
