from __future__ import annotations

import unittest

import numpy as np

from serl_launcher.residual.action_metrics import ResidualActionStatsAccumulator


class ResidualActionStatsAccumulatorTest(unittest.TestCase):
    def test_accumulates_residual_and_action_delta_stats(self) -> None:
        stats = ResidualActionStatsAccumulator(saturation_threshold=0.95)

        stats.add(
            residual_action=np.asarray([0.0, 0.5, -1.0], dtype=np.float32),
            base_action=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            final_action=np.asarray([1.0, 1.05, 0.9], dtype=np.float32),
        )
        stats.add(
            residual_action=np.asarray([0.25, -0.25, 0.95], dtype=np.float32),
            base_action=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            final_action=np.asarray([0.02, -0.02, 0.1], dtype=np.float32),
        )

        summary = stats.summary()

        self.assertEqual(summary["value_count"], 6)
        self.assertAlmostEqual(summary["mean_abs"], 2.95 / 6.0)
        self.assertAlmostEqual(summary["max_abs"], 1.0)
        self.assertAlmostEqual(summary["saturation_rate"], 2.0 / 6.0)
        self.assertAlmostEqual(summary["action_delta_mean_abs"], 0.29 / 6.0)
        self.assertAlmostEqual(summary["action_delta_max_abs"], 0.1)
        self.assertGreater(float(summary["std"]), 0.0)

    def test_empty_summary_when_no_residuals_added(self) -> None:
        stats = ResidualActionStatsAccumulator()

        self.assertEqual(stats.summary(), {})


if __name__ == "__main__":
    unittest.main()
