"""Minimal smoke tests for V1 package skeleton."""


def test_import_serl_torch() -> None:
    import serl_torch  # noqa: F401


def test_import_namespaces() -> None:
    import serl_torch.adapters  # noqa: F401
    import serl_torch.common  # noqa: F401
    import serl_torch.launcher  # noqa: F401


def test_launcher_bridge_exposes_package_path() -> None:
    import serl_torch.launcher as launcher

    assert hasattr(launcher, "__path__")
    assert len(tuple(launcher.__path__)) > 0
