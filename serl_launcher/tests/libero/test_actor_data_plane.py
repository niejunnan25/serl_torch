from __future__ import annotations

import sys
from pathlib import Path
import types
import unittest
from unittest import mock


REPO_PARENT = Path(__file__).resolve().parents[4]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

if "gymnasium" not in sys.modules:
    fake_gymnasium = types.ModuleType("gymnasium")
    fake_gymnasium.Env = object
    sys.modules["gymnasium"] = fake_gymnasium

from serl_torch.examples.libero.rollout_data_processor import (  # noqa: E402
    RolloutDataProcessor,
)


class RolloutDataProcessorTest(unittest.TestCase):
    def test_observe_chunk_commits_ready_and_new_chunks(self) -> None:
        ready_chunk = mock.Mock(env_steps_delta=1, transitions=[{"id": "ready"}])
        new_chunk = mock.Mock(env_steps_delta=1, transitions=[{"id": "new"}])
        fake_processor = mock.Mock()
        fake_processor.async_transition_assembly_enabled = True
        fake_processor.drain_ready.return_value = [ready_chunk]
        fake_processor.handle_chunk.return_value = [new_chunk]

        with mock.patch(
            "serl_torch.examples.libero.rollout_data_processor.LiberoActorTransitionAssembler",
            return_value=fake_processor,
        ):
            data_store = mock.Mock()
            update_transport = mock.Mock(return_value=True)
            plane = RolloutDataProcessor(
                cfg=types.SimpleNamespace(
                    training=types.SimpleNamespace(steps_per_update=1)
                ),
                policy_client=object(),
                data_store=data_store,
                update_trainer_transport=update_transport,
                logger=mock.Mock(),
            )

        plane.observe_chunk(raw_chunk=mock.Mock(), task_prompt="task")

        fake_processor.drain_ready.assert_called_once_with()
        fake_processor.handle_chunk.assert_called_once()
        self.assertEqual(data_store.insert.call_count, 2)
        update_transport.assert_has_calls(
            [
                mock.call(context="commit_step_1"),
                mock.call(context="commit_step_2"),
            ]
        )

    def test_finish_episode_respects_wait_flag(self) -> None:
        finished_chunk = mock.Mock(env_steps_delta=2, transitions=[{"id": "a"}])
        fake_processor = mock.Mock()
        fake_processor.async_transition_assembly_enabled = False
        fake_processor.finish_episode.return_value = [finished_chunk]

        with mock.patch(
            "serl_torch.examples.libero.rollout_data_processor.LiberoActorTransitionAssembler",
            return_value=fake_processor,
        ):
            data_store = mock.Mock()
            update_transport = mock.Mock(return_value=True)
            plane = RolloutDataProcessor(
                cfg=types.SimpleNamespace(
                    training=types.SimpleNamespace(steps_per_update=2)
                ),
                policy_client=object(),
                data_store=data_store,
                update_trainer_transport=update_transport,
                logger=mock.Mock(),
            )

        plane.finish_episode(wait_for_episode_commit=False)

        fake_processor.finish_episode.assert_called_once_with(block=False)
        data_store.insert.assert_called_once_with({"id": "a"})
        update_transport.assert_called_once_with(context="commit_step_2")

    def test_close_flushes_and_closes_processor(self) -> None:
        finished_chunk = mock.Mock(env_steps_delta=1, transitions=[{"id": "done"}])
        fake_processor = mock.Mock()
        fake_processor.async_transition_assembly_enabled = False
        fake_processor.finish_episode.return_value = [finished_chunk]

        with mock.patch(
            "serl_torch.examples.libero.rollout_data_processor.LiberoActorTransitionAssembler",
            return_value=fake_processor,
        ):
            data_store = mock.Mock()
            update_transport = mock.Mock(return_value=True)
            plane = RolloutDataProcessor(
                cfg=types.SimpleNamespace(
                    training=types.SimpleNamespace(steps_per_update=1)
                ),
                policy_client=object(),
                data_store=data_store,
                update_trainer_transport=update_transport,
                logger=mock.Mock(),
            )

        plane.close()

        fake_processor.finish_episode.assert_called_once_with(block=True)
        fake_processor.close.assert_called_once_with()
        data_store.insert.assert_called_once_with({"id": "done"})


if __name__ == "__main__":
    unittest.main()
