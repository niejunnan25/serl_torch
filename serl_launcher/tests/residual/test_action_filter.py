from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

SERL_LAUNCHER_ROOT = Path(__file__).resolve().parents[2]
if str(SERL_LAUNCHER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERL_LAUNCHER_ROOT))

from serl_launcher.residual.action_filter import ResidualDeltaActionFilter
from serl_launcher.residual import (
    ResidualDeltaActionFilter as ExportedResidualDeltaActionFilter,
)


class ResidualDeltaActionFilterTest(unittest.TestCase):
    def test_package_export_matches_module_class(self) -> None:
        self.assertIs(ExportedResidualDeltaActionFilter, ResidualDeltaActionFilter)

    def test_disabled_returns_final_action_chunk(self) -> None:
        action_filter = ResidualDeltaActionFilter(enabled=False, alpha=0.25)
        base = np.asarray([[10.0, 2.0], [20.0, 3.0]], dtype=np.float32)
        final = np.asarray([[10.5, 2.5], [19.5, 2.0]], dtype=np.float32)

        filtered = action_filter.filter_action_chunk(
            base_action_chunk=base,
            final_action_chunk=final,
        )

        np.testing.assert_allclose(filtered, final)
        self.assertEqual(action_filter.total_steps, 0)

    def test_filters_residual_delta_without_smoothing_base_action(self) -> None:
        action_filter = ResidualDeltaActionFilter(enabled=True, alpha=0.25)
        base = np.asarray([[10.0], [20.0], [30.0]], dtype=np.float32)
        final = np.asarray([[10.0], [21.0], [31.0]], dtype=np.float32)

        filtered = action_filter.filter_action_chunk(
            base_action_chunk=base,
            final_action_chunk=final,
        )

        np.testing.assert_allclose(
            filtered,
            np.asarray([[10.0], [20.25], [30.4375]], dtype=np.float32),
        )
        self.assertEqual(action_filter.total_steps, 3)

    def test_max_delta_clamps_smoothed_residual_delta(self) -> None:
        action_filter = ResidualDeltaActionFilter(
            enabled=True,
            alpha=1.0,
            max_delta=0.2,
        )
        base = np.zeros((3, 1), dtype=np.float32)
        final = np.asarray([[0.0], [1.0], [-1.0]], dtype=np.float32)

        filtered = action_filter.filter_action_chunk(
            base_action_chunk=base,
            final_action_chunk=final,
        )

        np.testing.assert_allclose(
            filtered,
            np.asarray([[0.0], [0.2], [0.0]], dtype=np.float32),
            atol=1e-6,
        )

    def test_episode_reset_clears_previous_delta_but_keeps_total_steps(self) -> None:
        action_filter = ResidualDeltaActionFilter(
            enabled=True,
            alpha=0.5,
            warmup_steps=1,
            reset_each_episode=True,
        )
        first = action_filter.filter_residual_delta_chunk(
            np.asarray([[1.0], [1.0]], dtype=np.float32)
        )

        action_filter.reset_episode()
        second = action_filter.filter_residual_delta_chunk(
            np.asarray([[0.0]], dtype=np.float32)
        )

        np.testing.assert_allclose(first, np.asarray([[1.0], [1.0]], dtype=np.float32))
        np.testing.assert_allclose(second, np.asarray([[0.0]], dtype=np.float32))
        self.assertEqual(action_filter.total_steps, 3)


if __name__ == "__main__":
    unittest.main()
