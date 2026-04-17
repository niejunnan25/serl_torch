#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import socket
import statistics
import time
from pathlib import Path

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


def _make_transition(step: int) -> dict:
    return {
        "observations": {
            "state": np.asarray([step, step + 0.1, step + 0.2], dtype=np.float32),
        },
        "next_observations": {
            "state": np.asarray([step + 1, step + 1.1, step + 1.2], dtype=np.float32),
        },
        "actions": np.asarray([step, -step], dtype=np.float32),
        "rewards": np.float32(step),
        "masks": np.float32(1.0),
        "dones": False,
    }


def _build_transport_cfg(mode: str, data_port: int) -> TrainerTransportConfig:
    return TrainerTransportConfig(
        mode=str(mode),
        data_port=int(data_port),
        control_timeout_ms=2000,
        data_queue_capacity=8,
        data_socket_hwm=8,
        commit_poll_ms=10,
        wait_committed_on_episode_end=False,
        wait_committed_on_shutdown=True,
    )


def _run_case(*, mode: str, iterations: int, batch_size: int) -> dict[str, float | str]:
    trainer_port = _find_free_port()
    broadcast_port = _find_free_port()
    data_port = _find_free_port()
    cfg = _build_transport_cfg(mode, data_port)
    observation_space = gym.spaces.Dict(
        {
            "state": gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(3,),
                dtype=np.float32,
            )
        }
    )
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    replay = ReplayBufferDataStore(observation_space, action_space, capacity=10000)
    learner = build_learner_trainer_transport(
        trainer_port=trainer_port,
        broadcast_port=broadcast_port,
        transport_cfg=cfg,
    )
    learner.register_data_store("actor_env", replay)
    learner.start(threaded=True)

    actor_store = QueuedDataStore(capacity=10000)
    actor = build_actor_trainer_transport(
        store_name="actor_env",
        server_ip="127.0.0.1",
        trainer_port=trainer_port,
        broadcast_port=broadcast_port,
        transport_cfg=cfg,
        data_store=actor_store,
        wait_for_server=True,
    )
    try:
        update_times: list[float] = []
        commit_times: list[float] = []
        for iteration in range(int(iterations)):
            for offset in range(int(batch_size)):
                actor_store.insert(_make_transition(iteration * batch_size + offset))
            start = time.perf_counter()
            assert actor.update() is True
            update_times.append(time.perf_counter() - start)

            start = time.perf_counter()
            assert actor.wait_until_committed(timeout_ms=5000) is True
            commit_times.append(time.perf_counter() - start)
        status = actor.get_transport_status("actor_env")
        return {
            "mode": str(mode),
            "iterations": int(iterations),
            "batch_size": int(batch_size),
            "update_mean_s": float(statistics.mean(update_times)),
            "update_median_s": float(statistics.median(update_times)),
            "commit_mean_s": float(statistics.mean(commit_times)),
            "commit_median_s": float(statistics.median(commit_times)),
            "accepted_update_id": int(status.get("accepted_update_id", -1)),
            "committed_update_id": int(status.get("committed_update_id", -1)),
        }
    finally:
        actor.stop()
        learner.stop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    results = {
        "legacy_reqrep": _run_case(
            mode="legacy_reqrep",
            iterations=int(args.iterations),
            batch_size=int(args.batch_size),
        ),
        "split_queue": _run_case(
            mode="split_queue",
            iterations=int(args.iterations),
            batch_size=int(args.batch_size),
        ),
    }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
