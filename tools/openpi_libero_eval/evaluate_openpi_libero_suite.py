#!/usr/bin/env python3
"""Evaluate one OpenPI policy server on a single LIBERO suite.

This is a lightweight alternative to openpi/examples/libero/main_10.py:
- no video export by default
- writes machine-readable per-task / per-suite summaries
- keeps task-level logs for later inspection
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_PARENT = REPO_ROOT.parent
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))
SERL_LAUNCHER_ROOT = REPO_ROOT / "serl_launcher"
if str(SERL_LAUNCHER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERL_LAUNCHER_ROOT))

from serl_torch.examples.libero.env.observation import build_libero_state
from serl_torch.examples.libero.env.observation import extract_libero_images
from serl_torch.examples.libero.env.setup import resolve_libero_config_dir
from serl_torch.examples.libero.env.setup import resolve_libero_datasets_root
from serl_torch.examples.libero.env.setup import resolve_libero_root
from serl_torch.examples.libero.env.setup import resolve_max_episode_steps
from serl_torch.examples.libero.env.setup import setup_libero_pythonpath


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]


def _resolve_openpi_root(openpi_root: str | None) -> Path:
    candidates: list[Path] = []
    if openpi_root:
        candidates.append(Path(openpi_root).expanduser().resolve())
    env_override = os.environ.get("OPENPI_ROOT")
    if env_override:
        candidates.append(Path(env_override).expanduser().resolve())
    candidates.append((REPO_PARENT / "openpi").resolve())

    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not find OpenPI root. Checked: {[str(path) for path in candidates]}"
    )


def _configure_import_paths(openpi_root: Path, libero_root: Path) -> None:
    openpi_client_src = openpi_root / "packages" / "openpi-client" / "src"
    if str(openpi_client_src) not in sys.path:
        sys.path.insert(0, str(openpi_client_src))
    if str(libero_root) not in sys.path:
        sys.path.insert(0, str(libero_root))


def _make_task_logger(task_id: int, log_path: Path) -> logging.Logger:
    logger = logging.getLogger(f"openpi_libero_eval.task_{task_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _bool_arg(parser: argparse.ArgumentParser, name: str, default: bool) -> None:
    option = name.replace("_", "-")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{option}", dest=name, action="store_true")
    group.add_argument(f"--no-{option}", dest=name, action="store_false")
    parser.set_defaults(**{name: default})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one LIBERO suite against one OpenPI server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument(
        "--task-suite-name",
        required=True,
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
    )
    parser.add_argument("--num-trials-per-task", type=int, default=50)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--strategy-name", required=True)
    parser.add_argument("--openpi-root", default=None)
    parser.add_argument("--libero-root", default=None)
    parser.add_argument("--libero-config-dir", default=None)
    parser.add_argument("--libero-datasets-root", default=None)
    parser.add_argument("--task-ids", nargs="*", type=int, default=None)
    _bool_arg(parser, "save_videos", default=False)
    _bool_arg(parser, "resume", default=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    libero_root = resolve_libero_root(args.libero_root)
    libero_config_dir = resolve_libero_config_dir(args.libero_config_dir)
    libero_datasets_root = resolve_libero_datasets_root(
        args.libero_datasets_root,
        libero_root=libero_root,
    )
    setup_libero_pythonpath(
        libero_root,
        libero_config_dir,
        datasets_root=libero_datasets_root,
    )

    openpi_root = _resolve_openpi_root(args.openpi_root)
    _configure_import_paths(openpi_root, libero_root)

    from libero.libero import benchmark
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import websocket_client_policy as websocket_client_policy

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
    )
    logger = logging.getLogger("openpi_libero_eval")

    logger.info("Strategy: %s", args.strategy_name)
    logger.info("OpenPI root: %s", openpi_root)
    logger.info("LIBERO root: %s", libero_root)
    logger.info("LIBERO datasets root: %s", libero_datasets_root)
    logger.info("Suite: %s", args.task_suite_name)
    logger.info("Server: ws://%s:%s", args.host, args.port)
    logger.info("Output root: %s", output_root)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    max_steps = resolve_max_episode_steps(args.task_suite_name)
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    existing_suite_summary_path = output_root / "suite_summary.json"
    existing_tasks: dict[int, dict[str, Any]] = {}
    if args.resume and existing_suite_summary_path.is_file():
        existing_suite_summary = _load_json(existing_suite_summary_path)
        for task in existing_suite_summary.get("tasks", []):
            existing_tasks[int(task["task_id"])] = task

    suite_summary: dict[str, Any] = {
        "strategy_name": str(args.strategy_name),
        "suite_name": str(args.task_suite_name),
        "server_host": str(args.host),
        "server_port": int(args.port),
        "openpi_root": str(openpi_root),
        "libero_root": str(libero_root),
        "libero_datasets_root": str(libero_datasets_root),
        "num_trials_per_task": int(args.num_trials_per_task),
        "num_steps_wait": int(args.num_steps_wait),
        "resize_size": int(args.resize_size),
        "replan_steps": int(args.replan_steps),
        "seed": int(args.seed),
        "save_videos": bool(args.save_videos),
        "tasks": [],
        "episodes_total": 0,
        "successes_total": 0,
        "success_rate_total": 0.0,
    }
    _write_json(existing_suite_summary_path, suite_summary)

    total_episodes = 0
    total_successes = 0
    task_ids = (
        [int(task_id) for task_id in args.task_ids]
        if args.task_ids
        else list(range(task_suite.n_tasks))
    )
    tasks_bar = tqdm(
        task_ids,
        desc=f"{args.strategy_name}:{args.task_suite_name}",
        dynamic_ncols=True,
    )

    for task_id in tasks_bar:
        if task_id in existing_tasks and int(existing_tasks[task_id].get("episodes", 0)) >= int(args.num_trials_per_task):
            task_summary = existing_tasks[task_id]
            suite_summary["tasks"].append(task_summary)
            total_episodes += int(task_summary["episodes"])
            total_successes += int(task_summary["successes"])
            suite_summary["episodes_total"] = int(total_episodes)
            suite_summary["successes_total"] = int(total_successes)
            suite_summary["success_rate_total"] = float(total_successes) / float(
                max(1, total_episodes)
            )
            _write_json(existing_suite_summary_path, suite_summary)
            tasks_bar.set_postfix(
                skipped_task=int(task_id),
                total_success_rate=f"{100.0 * total_successes / max(1, total_episodes):.1f}%",
                refresh=False,
            )
            continue

        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        task_description = str(task.language)
        task_dir = output_root / f"task_{task_id}"
        task_dir.mkdir(parents=True, exist_ok=True)
        task_log_path = task_dir / "task.log"
        task_logger = _make_task_logger(task_id, task_log_path)
        task_logger.info("Task suite: %s", args.task_suite_name)
        task_logger.info("Task id: %s", task_id)
        task_logger.info("Task description: %s", task_description)
        task_logger.info("Loaded %s init states", len(initial_states))

        task_bddl_file = (
            Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        )
        env = OffScreenRenderEnv(
            bddl_file_name=task_bddl_file,
            camera_heights=256,
            camera_widths=256,
        )
        env.seed(int(args.seed))

        task_episodes = 0
        task_successes = 0
        task_episode_results: list[dict[str, Any]] = []
        episodes_bar = tqdm(
            range(int(args.num_trials_per_task)),
            desc=f"task {task_id}",
            dynamic_ncols=True,
            leave=False,
        )

        try:
            for episode_idx in episodes_bar:
                init_state_idx = int(episode_idx) % len(initial_states)
                env.reset()
                action_plan = collections.deque()
                obs = env.set_init_state(initial_states[init_state_idx])
                done = False
                replay_images: list[np.ndarray] = []
                t = 0

                task_logger.info(
                    "Episode %s/%s init_state_idx=%s",
                    int(episode_idx) + 1,
                    int(args.num_trials_per_task),
                    init_state_idx,
                )

                while t < max_steps + int(args.num_steps_wait):
                    try:
                        if t < int(args.num_steps_wait):
                            obs, _reward, _done, _info = env.step(LIBERO_DUMMY_ACTION)
                            t += 1
                            continue

                        images = extract_libero_images(obs)
                        image = images["image_rgb_0"]
                        wrist_image = images["image_rgb_1"]
                        if args.save_videos:
                            replay_images.append(np.asarray(image, copy=True))

                        if not action_plan:
                            element = {
                                "observation/image": image,
                                "observation/wrist_image": wrist_image,
                                "observation/state": build_libero_state(obs),
                                "prompt": task_description,
                            }
                            action_chunk = client.infer(element)["actions"]
                            if len(action_chunk) < int(args.replan_steps):
                                raise ValueError(
                                    "Policy returned too short action chunk: "
                                    f"{len(action_chunk)} < {int(args.replan_steps)}"
                                )
                            action_plan.extend(action_chunk[: int(args.replan_steps)])

                        action = np.asarray(action_plan.popleft(), dtype=np.float32)
                        obs, _reward, done, _info = env.step(action.tolist())
                        t += 1
                        if bool(done):
                            task_successes += 1
                            total_successes += 1
                            break
                    except Exception as exc:  # noqa: BLE001
                        task_logger.exception("Episode failed with exception: %s", exc)
                        break

                task_episodes += 1
                total_episodes += 1
                task_episode_results.append(
                    {
                        "episode_idx": int(episode_idx),
                        "init_state_idx": int(init_state_idx),
                        "success": bool(done),
                        "steps": int(max(0, t - int(args.num_steps_wait))),
                    }
                )

                if args.save_videos and replay_images:
                    import imageio

                    suffix = "success" if done else "failure"
                    video_path = task_dir / f"episode_{int(episode_idx):03d}_{suffix}.mp4"
                    imageio.mimwrite(video_path, replay_images, fps=10)

                episodes_bar.set_postfix(
                    success_rate=f"{100.0 * task_successes / max(1, task_episodes):.1f}%",
                    refresh=False,
                )

                task_logger.info("Episode success: %s", bool(done))
                task_logger.info(
                    "Task progress: %s/%s, success rate=%.4f",
                    task_episodes,
                    int(args.num_trials_per_task),
                    float(task_successes) / float(max(1, task_episodes)),
                )
        finally:
            env.close()
            episodes_bar.close()

        task_summary = {
            "task_id": int(task_id),
            "task_description": task_description,
            "episodes": int(task_episodes),
            "successes": int(task_successes),
            "success_rate": float(task_successes) / float(max(1, task_episodes)),
            "log_path": str(task_log_path),
            "episodes_detail": task_episode_results,
        }
        _write_json(task_dir / "summary.json", task_summary)
        suite_summary["tasks"].append(task_summary)
        suite_summary["episodes_total"] = int(total_episodes)
        suite_summary["successes_total"] = int(total_successes)
        suite_summary["success_rate_total"] = float(total_successes) / float(
            max(1, total_episodes)
        )
        _write_json(existing_suite_summary_path, suite_summary)

        task_logger.info(
            "Final task success rate: %.4f (%s/%s)",
            task_summary["success_rate"],
            task_summary["successes"],
            task_summary["episodes"],
        )
        tasks_bar.set_postfix(
            total_success_rate=f"{100.0 * total_successes / max(1, total_episodes):.1f}%",
            refresh=False,
        )

    tasks_bar.close()
    logger.info(
        "Suite complete: %s success_rate=%.4f (%s/%s)",
        args.task_suite_name,
        suite_summary["success_rate_total"],
        suite_summary["successes_total"],
        suite_summary["episodes_total"],
    )


if __name__ == "__main__":
    main()
