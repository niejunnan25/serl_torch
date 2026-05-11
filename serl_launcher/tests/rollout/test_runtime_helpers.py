from __future__ import annotations

import sys
from pathlib import Path
import unittest
from unittest import mock

SERL_LAUNCHER_ROOT = Path(__file__).resolve().parents[2]
if str(SERL_LAUNCHER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERL_LAUNCHER_ROOT))

from serl_launcher.rollout.runtime_helpers import commit_finished_episode_chunks
from serl_launcher.rollout import (
    commit_finished_episode_chunks as exported_commit_finished_episode_chunks,
)


class RolloutRuntimeHelpersTest(unittest.TestCase):
    def test_package_export_matches_module_function(self) -> None:
        self.assertIs(
            exported_commit_finished_episode_chunks,
            commit_finished_episode_chunks,
        )

    def test_regular_episode_end_respects_nonblocking_config(self) -> None:
        transition_assembler = mock.Mock()
        transition_assembler.finish_episode.return_value = ["assembled"]
        commit_assembled_chunks = mock.Mock()

        commit_finished_episode_chunks(
            transition_assembler=transition_assembler,
            commit_assembled_chunks=commit_assembled_chunks,
            wait_for_episode_commit=False,
        )

        transition_assembler.finish_episode.assert_called_once_with(block=False)
        commit_assembled_chunks.assert_called_once_with(["assembled"])

    def test_zero_step_terminal_forces_blocking_drain(self) -> None:
        transition_assembler = mock.Mock()
        transition_assembler.finish_episode.return_value = ["assembled"]
        commit_assembled_chunks = mock.Mock()

        commit_finished_episode_chunks(
            transition_assembler=transition_assembler,
            commit_assembled_chunks=commit_assembled_chunks,
            wait_for_episode_commit=False,
            require_last_transition_ready=True,
        )

        transition_assembler.finish_episode.assert_called_once_with(block=True)
        commit_assembled_chunks.assert_called_once_with(["assembled"])

    def test_wait_flag_still_blocks_without_terminal_fixup(self) -> None:
        transition_assembler = mock.Mock()
        transition_assembler.finish_episode.return_value = ["assembled"]
        commit_assembled_chunks = mock.Mock()

        commit_finished_episode_chunks(
            transition_assembler=transition_assembler,
            commit_assembled_chunks=commit_assembled_chunks,
            wait_for_episode_commit=True,
        )

        transition_assembler.finish_episode.assert_called_once_with(block=True)
        commit_assembled_chunks.assert_called_once_with(["assembled"])


if __name__ == "__main__":
    unittest.main()
