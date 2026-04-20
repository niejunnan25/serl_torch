from __future__ import annotations

import sys
from pathlib import Path
import unittest
from unittest import mock


REPO_PARENT = Path(__file__).resolve().parents[4]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from serl_torch.examples.agibot_real.runtime_helpers import (  # noqa: E402
    commit_finished_episode_chunks,
)


class AgiBotRuntimeHelpersTest(unittest.TestCase):
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
