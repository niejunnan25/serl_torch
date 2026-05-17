from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_PARENT = Path(__file__).resolve().parents[4]
SERL_LAUNCHER_ROOT = Path(__file__).resolve().parents[3] / "serl_launcher"
for candidate in (REPO_PARENT, SERL_LAUNCHER_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from serl_torch.examples.libero.env.policy_input import build_libero_policy_input


def test_build_libero_policy_input_from_preparsed_parts() -> None:
    policy_input = build_libero_policy_input(
        prompt="pick up the block",
        state=np.arange(8, dtype=np.float64),
        images={
            "image_rgb_0": np.full((4, 4, 3), 1, dtype=np.int16),
            "image_rgb_1": np.full((4, 4, 3), 2, dtype=np.int16),
            "image_rgb_2": np.full((4, 4, 3), 3, dtype=np.int16),
        },
    )

    assert policy_input.prompt == "pick up the block"
    assert policy_input.state.dtype == np.float32
    assert np.array_equal(policy_input.state, np.arange(8, dtype=np.float32))
    assert policy_input.images["image_rgb_0"].dtype == np.uint8
    assert policy_input.images["image_rgb_1"].dtype == np.uint8
    assert policy_input.images["image_rgb_2"].dtype == np.uint8
    assert policy_input.image_mask == {
        "image_rgb_0": True,
        "image_rgb_1": True,
        "image_rgb_2": False,
    }
    assert policy_input.metadata == {}
