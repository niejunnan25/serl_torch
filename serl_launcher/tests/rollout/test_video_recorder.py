from __future__ import annotations

import sys
from pathlib import Path
import unittest

SERL_LAUNCHER_ROOT = Path(__file__).resolve().parents[2]
if str(SERL_LAUNCHER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERL_LAUNCHER_ROOT))

from serl_launcher.rollout.video_recorder import AsyncImageVideoRecorder
from serl_launcher.rollout.video_recorder import AsyncVideoRecorderConfig
from serl_launcher.rollout import (
    AsyncImageVideoRecorder as ExportedAsyncImageVideoRecorder,
)
from serl_launcher.rollout import (
    AsyncVideoRecorderConfig as ExportedAsyncVideoRecorderConfig,
)


class RolloutVideoRecorderTest(unittest.TestCase):
    def test_package_exports_match_module_classes(self) -> None:
        self.assertIs(ExportedAsyncImageVideoRecorder, AsyncImageVideoRecorder)
        self.assertIs(ExportedAsyncVideoRecorderConfig, AsyncVideoRecorderConfig)

    def test_config_instantiates_without_importing_cv2(self) -> None:
        config = AsyncVideoRecorderConfig(
            camera_key="image/head",
            fps=30.0,
            output_dir=Path("videos"),
            max_pending_frames=128,
            drop_frames_when_busy=True,
        )

        self.assertEqual(config.camera_key, "image/head")
        self.assertEqual(config.fps, 30.0)
        self.assertEqual(config.output_dir, Path("videos"))
        self.assertEqual(config.max_pending_frames, 128)
        self.assertTrue(config.drop_frames_when_busy)


if __name__ == "__main__":
    unittest.main()
