from __future__ import annotations

import pickle
import sys
import tempfile
from pathlib import Path
import unittest

REPO_PARENT = Path(__file__).resolve().parents[4]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

try:
    from omegaconf import OmegaConf
except ModuleNotFoundError:  # pragma: no cover - environment-dependent
    OmegaConf = None

if OmegaConf is not None:
    from serl_torch.examples.libero.config import parse_train_cfg
    from serl_torch.examples.libero.env.offline_data import (
        resolve_and_validate_prepared_paths,
    )


class _FakeLogger:
    def info(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def warning(self, *args: object, **kwargs: object) -> None:
        del args, kwargs


def _train_cfg_with_prepared_path(prepared_path: str) -> object:
    cfg = OmegaConf.load(
        Path(__file__).resolve().parents[3]
        / "examples"
        / "libero"
        / "configs"
        / "train_residual.yaml"
    )
    cfg.runtime.role = "learner"
    cfg.offline.enabled = True
    cfg.offline.prepared_path = prepared_path
    return parse_train_cfg(cfg)


@unittest.skipIf(OmegaConf is None, "omegaconf is not installed")
class LiberoOfflineDataTest(unittest.TestCase):
    def test_validate_prepared_offline_rejects_manifestless_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            prepared_dir = Path(tmpdir) / "prepared"
            prepared_dir.mkdir(parents=True, exist_ok=True)
            with open(prepared_dir / "episode_000000.pkl", "wb") as fp:
                pickle.dump([], fp, protocol=pickle.HIGHEST_PROTOCOL)

            cfg = _train_cfg_with_prepared_path(str(prepared_dir))
            with self.assertRaisesRegex(
                ValueError,
                "must contain manifest.json",
            ):
                resolve_and_validate_prepared_paths(
                    cfg,
                    logger=_FakeLogger(),
                )

    def test_validate_prepared_offline_rejects_manifestless_episode_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            episode_path = Path(tmpdir) / "episode_000000.pkl"
            with open(episode_path, "wb") as fp:
                pickle.dump([], fp, protocol=pickle.HIGHEST_PROTOCOL)

            cfg = _train_cfg_with_prepared_path(str(episode_path))
            with self.assertRaisesRegex(
                ValueError,
                "without manifest is no longer supported",
            ):
                resolve_and_validate_prepared_paths(
                    cfg,
                    logger=_FakeLogger(),
                )


if __name__ == "__main__":
    unittest.main()
