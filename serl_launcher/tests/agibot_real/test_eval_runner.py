from __future__ import annotations

import sys
import tempfile
from pathlib import Path
import unittest

REPO_PARENT = Path(__file__).resolve().parents[4]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - environment-dependent
    torch = None

if torch is not None:
    from serl_torch.examples.agibot_real.eval_runner import _resolve_checkpoint_input


@unittest.skipIf(torch is None, "torch is not installed")
class AgiBotEvalRunnerTest(unittest.TestCase):
    def test_resolve_checkpoint_input_from_directory_and_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / "checkpoint_20.pt"
            torch.save({"step": 20}, checkpoint_path)

            checkpoint_input_path, resolved_checkpoint_path = _resolve_checkpoint_input(
                str(checkpoint_dir.relative_to(Path(tmpdir))),
                20,
                original_cwd=Path(tmpdir),
            )

            self.assertEqual(checkpoint_input_path, checkpoint_dir.resolve())
            self.assertEqual(resolved_checkpoint_path, checkpoint_path.resolve())

    def test_resolve_checkpoint_input_from_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint_10.pt"
            torch.save({"step": 10}, checkpoint_path)

            checkpoint_input_path, resolved_checkpoint_path = _resolve_checkpoint_input(
                str(checkpoint_path),
                None,
            )

            self.assertEqual(checkpoint_input_path, checkpoint_path.resolve())
            self.assertEqual(resolved_checkpoint_path, checkpoint_path.resolve())


if __name__ == "__main__":
    unittest.main()
