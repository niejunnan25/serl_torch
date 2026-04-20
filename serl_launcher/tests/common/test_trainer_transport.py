from __future__ import annotations

import socket
import time

from agentlace.data.data_store import DataStoreBase
from agentlace.data.data_store import QueuedDataStore
import gym
import numpy as np

from serl_launcher.common.trainer_transport import TrainerTransportConfig
from serl_launcher.common.trainer_transport import build_actor_trainer_transport
from serl_launcher.common.trainer_transport import build_learner_trainer_transport
from serl_launcher.data.data_store import ReplayBufferDataStore


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _transition(step: int) -> dict:
    return {
        "observations": {
            "state": np.asarray([step, step + 0.1], dtype=np.float32),
        },
        "next_observations": {
            "state": np.asarray([step + 1, step + 1.1], dtype=np.float32),
        },
        "actions": np.asarray([step, -step], dtype=np.float32),
        "rewards": np.float32(step),
        "masks": np.float32(1.0),
        "dones": False,
    }


def test_async_commit_transport_commits_without_duplicates() -> None:
    trainer_port = _find_free_port()
    broadcast_port = _find_free_port()
    data_port = _find_free_port()
    transport_cfg = TrainerTransportConfig(
        mode="async_commit",
        data_port=int(data_port),
        control_timeout_ms=1000,
        data_queue_capacity=4,
        data_socket_hwm=4,
        commit_poll_ms=10,
        wait_committed_on_episode_end=False,
        wait_committed_on_shutdown=True,
    )
    observation_space = gym.spaces.Dict(
        {
            "state": gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(2,),
                dtype=np.float32,
            )
        }
    )
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    replay = ReplayBufferDataStore(observation_space, action_space, capacity=16)

    learner = build_learner_trainer_transport(
        trainer_port=trainer_port,
        broadcast_port=broadcast_port,
        transport_cfg=transport_cfg,
        request_types=("send-stats",),
    )
    learner.register_data_store("actor_env", replay)
    learner.start(threaded=True)

    actor_store = QueuedDataStore(capacity=16)
    actor = build_actor_trainer_transport(
        store_name="actor_env",
        server_ip="127.0.0.1",
        trainer_port=trainer_port,
        broadcast_port=broadcast_port,
        transport_cfg=transport_cfg,
        data_store=actor_store,
        request_types=("send-stats",),
        wait_for_server=True,
    )
    try:
        actor_store.insert(_transition(0))
        actor_store.insert(_transition(1))
        assert actor.update() is True
        assert actor.wait_until_committed(timeout_ms=2000) is True

        status = actor.get_transport_status("actor_env")
        assert int(status["accepted_update_id"]) == 1
        assert int(status["committed_update_id"]) == 1
        assert len(replay) == 2

        assert actor.update() is True
        time.sleep(0.1)
        assert len(replay) == 2
    finally:
        actor.stop()
        learner.stop()


class _SlowStore(DataStoreBase):
    def __init__(self) -> None:
        super().__init__(capacity=8)
        self.items: list[dict] = []

    def latest_data_id(self):
        return len(self.items) - 1

    def get_latest_data(self, from_id: int):
        del from_id
        return []

    def __len__(self):
        return len(self.items)

    def insert(self, data):
        self.items.append(data)

    def batch_insert(self, batch_data):
        time.sleep(0.3)
        if isinstance(batch_data, dict):
            count = int(np.asarray(batch_data["actions"]).shape[0])
            for index in range(count):
                self.items.append({"idx": index})
            return
        for item in batch_data:
            self.items.append(item)


def test_async_commit_control_plane_stays_responsive_under_backlog() -> None:
    trainer_port = _find_free_port()
    broadcast_port = _find_free_port()
    data_port = _find_free_port()
    transport_cfg = TrainerTransportConfig(
        mode="async_commit",
        data_port=int(data_port),
        control_timeout_ms=1000,
        data_queue_capacity=1,
        data_socket_hwm=1,
        commit_poll_ms=10,
        wait_committed_on_episode_end=False,
        wait_committed_on_shutdown=True,
    )
    learner = build_learner_trainer_transport(
        trainer_port=trainer_port,
        broadcast_port=broadcast_port,
        transport_cfg=transport_cfg,
    )
    learner.register_data_store("actor_env", _SlowStore())
    learner.start(threaded=True)

    actor_store = QueuedDataStore(capacity=8)
    actor = build_actor_trainer_transport(
        store_name="actor_env",
        server_ip="127.0.0.1",
        trainer_port=trainer_port,
        broadcast_port=broadcast_port,
        transport_cfg=transport_cfg,
        data_store=actor_store,
        wait_for_server=True,
    )
    try:
        for step in range(3):
            actor_store.insert(_transition(step))
            assert actor.update() is True
        status = actor.get_transport_status("actor_env")
        assert status["transport_mode"] == "async_commit"
        assert "accepted_update_id" in status
        assert "committed_update_id" in status
        assert int(status["accepted_update_id"]) >= int(status["committed_update_id"])
    finally:
        actor.stop()
        learner.stop()
