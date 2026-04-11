from __future__ import annotations

"""Reference-style AgiBot residual actor with external learner ownership.

This prototype keeps the main flow explicit:

1. build env / base policy / residual agent directly
2. reset env
3. infer base chunk
4. sample residual chunk
5. compose final chunk
6. env.step_chunk(...)
7. send executed step records to learner via TrainerClient.update()

Unlike the earlier prototype, this script no longer owns replay sampling or
parameter updates. The standalone learner is responsible for replay + updates,
matching the reference SERL actor/learner split more closely.
"""

import json
import logging
import sys
import threading
import time
from pathlib import Path

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from serl_launcher.agents.continuous.drq_config import create_drq_agent_from_cfg
from serl_launcher.common.checkpoint_codec import apply_checkpoint_payload_to_agent
from serl_launcher.common.checkpoint_codec import snapshot_agent_checkpoint_payload
from serl_launcher.policy.joyra.client import JoyRAPolicyClient
from serl_launcher.policy.openpi.client import OpenPIPolicyClient
from serl_launcher.residual.action import as_numpy_action_chunk
from serl_launcher.residual.action import compose_residual_action_chunk
from serl_launcher.residual.action import select_action_chunk_window
from serl_launcher.residual.action_spec import build_residual_limits
from serl_launcher.residual.train.config import build_residual_action_transform
from serl_launcher.residual.train.config import resolve_control_indices_from_cfg
from serl_launcher.residual.train.config import (
    resolve_residual_observation_state_mode,
)
from serl_launcher.training.loop_utils import _iter_period_hits
from serl_launcher.utils.agentlace_io import resolve_agentlace_bootstrap_path
from serl_launcher.utils.agentlace_io import save_agentlace_bootstrap
from serl_launcher.residual.utils.alpha_utils import require_residual_alpha

REPO_PARENT = Path(__file__).resolve().parents[5]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from serl_torch.examples.agibot_real.config import resolve_agibot_cfg_image_keys
from serl_torch.examples.agibot_real.env.task_env import AgiBotTaskEnv
from serl_torch.examples.agibot_real.runtime.obs_adapter import (
    build_residual_step_core,
)
from serl_torch.examples.agibot_real.runtime.obs_adapter import (
    build_residual_step_obs,
)
from serl_torch.examples.agibot_real.runtime.policy_adapter import (
    build_agibot_policy_input,
)
from serl_torch.examples.agibot_real.training_config import (
    coerce_agibot_agentlace_async_cfg,
)


def _validate_cfg(cfg: DictConfig) -> None:
    if bool(cfg.get("offline", {}).get("enabled", False)):
        raise ValueError(
            "reference-style actor does not support example-local offline injection; "
            "set offline.enabled=false"
        )
    if bool(cfg.get("training", {}).get("online_prefill", {}).get("enabled", False)):
        raise ValueError(
            "reference-style actor does not support online prefill injection; "
            "set training.online_prefill.enabled=false"
        )

    if bool(cfg.get("training", {}).get("async_eval", {}).get("enabled", False)):
        raise ValueError(
            "reference-style actor does not start async eval; "
            "set training.async_eval.enabled=false"
        )
    if int(cfg.get("training", {}).get("warmup", {}).get("episodes", 0)) != 0:
        raise ValueError(
            "reference-style actor does not support warmup episodes yet; "
            "set training.warmup.episodes=0"
        )
    if bool(cfg.get("training", {}).get("expert_check", False)):
        raise ValueError(
            "reference-style actor does not support expert precheck yet; "
            "set training.expert_check=false"
        )
    if bool(cfg.get("training", {}).get("enable_base_probing", False)):
        raise ValueError(
            "reference-style actor does not support base probing yet; "
            "set training.enable_base_probing=false"
        )
    if bool(cfg.get("residual", {}).get("epsilon_gating", {}).get("enabled", False)):
        raise ValueError(
            "reference-style actor does not support epsilon gating yet; "
            "set residual.epsilon_gating.enabled=false"
        )
    if not bool(cfg.get("chunk_step", {}).get("enabled", False)):
        raise ValueError(
            "reference-style actor only covers the current AgiBot chunk-step path; "
            "set chunk_step.enabled=true"
        )
    if int(cfg.get("sac", {}).get("obs_stack_horizon", 1)) != 1:
        raise ValueError(
            "reference-style actor currently supports only sac.obs_stack_horizon=1"
        )
    algorithm_type = (
        str(cfg.get("residual", {}).get("algorithm", {}).get("type", "sac"))
        .strip()
        .lower()
    )
    if algorithm_type != "sac":
        raise ValueError(
            "reference-style actor currently supports only residual.algorithm.type=sac"
        )
    phases = list(cfg.training.phases)
    if len(phases) != 1:
        raise ValueError(
            "reference-style actor currently supports exactly one training phase"
        )
    if not bool(phases[0].get("train", True)):
        raise ValueError(
            "reference-style actor currently supports only train=true phases"
        )


@hydra.main(
    version_base=None, config_path="../../conf", config_name="train_residual_sac"
)
def main(cfg: DictConfig) -> None:
    run_dir = Path(HydraConfig.get().runtime.output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s"
    )
    logger = logging.getLogger("agibot_real_actor_reference_style")

    coerce_agibot_agentlace_async_cfg(cfg)
    logger.info("Hydra run dir: %s", run_dir)
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    _validate_cfg(cfg)

    async_cfg = cfg.training.get("async", {})
    agentlace_cfg = async_cfg.get("agentlace", {})
    bootstrap_file = str(agentlace_cfg.get("bootstrap_file", "agentlace_bootstrap.pkl"))
    if not Path(bootstrap_file).expanduser().is_absolute():
        raise ValueError(
            "reference-style actor with a standalone learner requires "
            "training.async.agentlace.bootstrap_file to be an absolute path"
        )

    task_cfg = cfg.get("task", {})
    robot_cfg = cfg.get("robot", {})
    controller_cfg = OmegaConf.to_container(cfg.get("controller", {}), resolve=True)

    env = AgiBotTaskEnv(
        task_name=str(task_cfg.get("name", "agibot_real_task")),
        prompt=str(task_cfg.get("prompt", task_cfg.get("name", "agibot_real_task"))),
        action_dim=int(cfg.get("env", {}).get("action_dim", 14)),
        control_mode=str(task_cfg.get("control_mode", "camera_position")),
        hz=float(task_cfg.get("hz", 20.0)),
        use_smooth_trajectory=bool(task_cfg.get("use_smooth_trajectory", False)),
        trajectory_time=task_cfg.get("trajectory_time", None),
        max_episode_steps=task_cfg.get("max_episode_steps", None),
        assets_root=robot_cfg.get("assets_root", None),
        retargeter_urdf_path=robot_cfg.get("retargeter_urdf_path", None),
        retargeter_camera_extrinsic_path=robot_cfg.get(
            "retargeter_camera_extrinsic_path", None
        ),
        controller=controller_cfg,
        reset_hook=task_cfg.get("reset_hook", None),
        success_hook=task_cfg.get("success_hook", None),
        expert_precheck_hook=task_cfg.get("expert_precheck_hook", None),
        logger=logger,
    )
    image_keys = tuple(resolve_agibot_cfg_image_keys(cfg))

    policy_type = (
        str(cfg.get("policy", {}).get("type", "openpi")).strip().lower() or "openpi"
    )
    if policy_type == "openpi":
        openpi_cfg = cfg.get("openpi", {})
        policy_client = OpenPIPolicyClient(
            host=str(openpi_cfg.get("host", "localhost")),
            port=int(openpi_cfg.get("port", 30001)),
            logger=logger,
        )
    elif policy_type == "joyra":
        joyra_cfg = cfg.get("joyra", cfg.get("openpi", {}))
        policy_client = JoyRAPolicyClient(
            host=str(joyra_cfg.get("host", "localhost")),
            port=int(joyra_cfg.get("port", 30001)),
            action_dim=int(cfg.get("env", {}).get("action_dim", 14)),
            logger=logger,
        )
    else:
        raise ValueError(
            f"Unsupported policy.type for reference-style actor: {policy_type!r}"
        )

    env_action_dim = int(cfg.get("env", {}).get("action_dim", 14))
    chunk_horizon = int(cfg.residual.chunk_horizon)
    step_action_dim = int(
        len(
            resolve_control_indices_from_cfg(
                cfg,
                full_action_dim=env_action_dim,
            )
        )
    )
    control_indices = resolve_control_indices_from_cfg(
        cfg,
        full_action_dim=env_action_dim,
    )
    residual_alpha = require_residual_alpha(cfg.get("residual", None))
    residual_limits = build_residual_limits(
        control_indices,
        full_action_dim=env_action_dim,
        action_limits=cfg.residual.get("action_limits", None),
    )
    obs_state_mode = resolve_residual_observation_state_mode(cfg)
    action_transform = build_residual_action_transform(
        control_indices=control_indices,
        residual_limits=residual_limits,
        full_action_dim=env_action_dim,
        chunk_horizon=chunk_horizon,
        chunk_step_enabled=True,
        clip_gripper=bool(cfg.residual.clip_gripper),
    )

    sample_obs_raw = env.reset()
    sample_base_chunk = np.zeros(
        (chunk_horizon, env_action_dim),
        dtype=np.float32,
    )
    sample_obs = build_residual_step_obs(
        sample_obs_raw,
        sample_base_chunk[0],
        image_keys=image_keys,
        stack_horizon=1,
        action_dim=env_action_dim,
        base_action_chunk=sample_base_chunk,
        alpha=float(residual_alpha),
        state_mode=obs_state_mode,
    )
    sample_state_core = build_residual_step_core(
        sample_obs_raw,
        image_keys=image_keys,
    )["state_core"]

    agent = create_drq_agent_from_cfg(
        cfg,
        sample_obs=sample_obs,
        action_dim=int(step_action_dim * chunk_horizon),
        image_keys=image_keys,
        critic_action_dim=int(env_action_dim * chunk_horizon),
        action_transform=action_transform,
    )
    agent_lock = threading.RLock()

    bootstrap_path = resolve_agentlace_bootstrap_path(
        run_dir=run_dir,
        bootstrap_file=bootstrap_file,
    )
    save_agentlace_bootstrap(
        bootstrap_path,
        {
            "sample_obs": sample_obs,
            "state_core_dim": int(sample_state_core.shape[0]),
            "env_action_dim": int(env_action_dim),
            "step_action_dim": int(step_action_dim),
            "agent_action_dim": int(step_action_dim * chunk_horizon),
            "critic_action_dim": int(env_action_dim * chunk_horizon),
            "image_keys": tuple(image_keys),
            "action_transform": action_transform,
            "chunk_step_enabled": True,
            "chunk_horizon": int(chunk_horizon),
            "state_mode": str(obs_state_mode),
            "initial_agent_payload": snapshot_agent_checkpoint_payload(
                agent,
                step=int(agent.state.step),
            ),
            "saved_at_unix": float(time.time()),
        },
    )
    logger.info("Agentlace bootstrap saved to %s", bootstrap_path)

    from agentlace.data.data_store import QueuedDataStore
    from agentlace.trainer import TrainerClient
    from agentlace.trainer import TrainerConfig

    data_store = QueuedDataStore(int(async_cfg.get("data_store_queue_size", 2000)))
    client = TrainerClient(
        "actor_env",
        str(async_cfg.get("trainer_host", "127.0.0.1")),
        TrainerConfig(
            port_number=int(async_cfg.get("trainer_port", 5488)),
            broadcast_port=int(async_cfg.get("broadcast_port", 5489)),
            request_types=["send-stats", "save-checkpoint", "get-status", "sync-now"],
        ),
        data_store,
        wait_for_server=True,
    )
    learner_update_steps = {"value": int(agent.state.step)}

    def _update_actor_agent(payload: dict) -> None:
        with agent_lock:
            apply_checkpoint_payload_to_agent(
                agent,
                dict(payload),
                load_optimizers=False,
            )
        learner_update_steps["value"] = int(
            payload.get("step", learner_update_steps["value"])
        )

    client.recv_network_callback(_update_actor_agent)
    try:
        client.request("sync-now", {})
    except Exception:  # noqa: BLE001
        logger.warning(
            "Initial sync-now request failed; relying on broadcast callback instead",
            exc_info=True,
        )

    phase_cfg = cfg.training.phases[0]
    phase_name = str(phase_cfg.get("name", "train"))
    total_episodes = int(phase_cfg.get("episodes", 0))
    send_every_steps = max(1, int(cfg.training.get("update_every", 1)))

    max_train_env_steps = int(cfg.training.get("max_train_env_steps", 0))
    progress = tqdm(
        total=max_train_env_steps if max_train_env_steps > 0 else None,
        desc="train_env_step",
        dynamic_ncols=True,
        leave=True,
    )

    train_env_step = 0
    decision_step = 0
    train_episode_id = 0
    train_total_success = 0
    stopped_by_env_budget = False

    try:
        while train_episode_id < total_episodes:
            if max_train_env_steps > 0 and train_env_step >= max_train_env_steps:
                stopped_by_env_budget = True
                break

            current_train_episode_id = int(train_episode_id + 1)
            obs_raw = env.reset()

            max_episode_steps = int(env.step_limit)
            if cfg.training.max_env_steps_per_episode is not None:
                max_episode_steps = min(
                    max_episode_steps,
                    int(cfg.training.max_env_steps_per_episode),
                )

            episode_steps = 0
            episode_return = 0.0
            episode_success = False
            episode_done = False

            while episode_steps < max_episode_steps and not episode_done:
                if max_train_env_steps > 0 and train_env_step >= max_train_env_steps:
                    stopped_by_env_budget = True
                    break

                decision_step += 1
                policy_input = build_agibot_policy_input(
                    obs_raw,
                    env.current_instruction,
                )

                base_policy_chunk, _ = policy_client.infer_chunk(policy_input)
                base_chunk = select_action_chunk_window(
                    base_policy_chunk,
                    horizon=chunk_horizon,
                    action_dim=env_action_dim,
                )

                residual_obs = build_residual_step_obs(
                    obs_raw,
                    base_chunk[0],
                    image_keys=image_keys,
                    stack_horizon=1,
                    action_dim=env_action_dim,
                    base_action_chunk=base_chunk,
                    alpha=float(residual_alpha),
                    state_mode=obs_state_mode,
                )

                with agent_lock:
                    residual_chunk = as_numpy_action_chunk(
                        agent.sample_actions(
                            residual_obs,
                            deterministic=False,
                        ),
                        action_dim=step_action_dim,
                        chunk_horizon=chunk_horizon,
                    )

                execute_horizon = min(
                    chunk_horizon,
                    max_episode_steps - episode_steps,
                )
                if max_train_env_steps > 0:
                    execute_horizon = min(
                        execute_horizon,
                        max(0, max_train_env_steps - train_env_step),
                    )
                if execute_horizon <= 0:
                    stopped_by_env_budget = bool(
                        max_train_env_steps > 0
                        and train_env_step >= max_train_env_steps
                    )
                    break

                executed_base_chunk = np.asarray(
                    base_chunk[:execute_horizon],
                    dtype=np.float32,
                )
                executed_residual_chunk = np.asarray(
                    residual_chunk[:execute_horizon],
                    dtype=np.float32,
                )
                _, final_chunk = compose_residual_action_chunk(
                    base_chunk=executed_base_chunk,
                    residual_chunk=executed_residual_chunk,
                    indices=control_indices,
                    limits=residual_limits,
                    alpha=float(residual_alpha),
                    clip_gripper=bool(cfg.residual.clip_gripper),
                )

                train_env_step_before = int(train_env_step)
                chunk_result = env.step_chunk(final_chunk)

                chunk_rewards = [float(v) for v in chunk_result["rewards"]]
                chunk_infos = [dict(v) for v in chunk_result["infos"]]
                chunk_dones = [bool(v) for v in chunk_result["dones"]]
                chunk_observations = list(chunk_result["observations"])
                next_obs_raw = chunk_result["obs"]
                actual_chunk_steps = int(len(chunk_rewards))

                executed_base_chunk = executed_base_chunk[:actual_chunk_steps]
                final_chunk = final_chunk[:actual_chunk_steps]

                current_step_obs_raw = obs_raw
                for chunk_step in range(actual_chunk_steps):
                    reward = float(chunk_rewards[chunk_step])
                    info = dict(chunk_infos[chunk_step])

                    done_flag = bool(
                        chunk_dones[chunk_step]
                        or (episode_steps + 1) >= max_episode_steps
                        or (
                            max_train_env_steps > 0
                            and (train_env_step + 1) >= max_train_env_steps
                        )
                    )

                    data_store.insert(
                        {
                            "obs_core": build_residual_step_core(
                                current_step_obs_raw,
                                image_keys=image_keys,
                            ),
                            "base_action": np.asarray(
                                executed_base_chunk[chunk_step],
                                dtype=np.float32,
                            ).reshape(-1),
                            "base_action_norm": np.asarray(
                                executed_base_chunk[chunk_step],
                                dtype=np.float32,
                            ).reshape(-1),
                            "actions": np.asarray(
                                final_chunk[chunk_step],
                                dtype=np.float32,
                            ).reshape(-1),
                            "rewards": float(reward),
                            "dones": bool(done_flag),
                            "alpha": float(residual_alpha),
                            "episode_id": int(train_episode_id),
                            "episode_step": int(episode_steps),
                        }
                    )

                    episode_steps += 1
                    train_env_step += 1
                    episode_return += float(reward)
                    episode_success = bool(info.get("success", episode_success))
                    progress.update(1)

                    if chunk_step < actual_chunk_steps - 1:
                        current_step_obs_raw = chunk_observations[chunk_step]
                    if done_flag:
                        episode_done = True
                        break

                for _step in _iter_period_hits(
                    step_before=int(train_env_step_before),
                    step_after=int(train_env_step),
                    period=int(send_every_steps),
                ):
                    client.update()

                obs_raw = next_obs_raw

            client.update()
            train_total_success += int(episode_success)
            train_episode_id = int(current_train_episode_id)
            running_success_rate = float(train_total_success) / float(train_episode_id)
            client.request(
                "send-stats",
                {
                    "train_episode": {
                        "phase": str(phase_name),
                        "train_episode_id": int(current_train_episode_id),
                        "success": bool(episode_success),
                        "episode_steps": int(episode_steps),
                        "episode_return": float(episode_return),
                        "train_env_step": int(train_env_step),
                        "decision_step": int(decision_step),
                        "running_success_rate": float(running_success_rate),
                        "recent_success_rate": None,
                    }
                },
            )

            logger.info(
                "phase=%s train_episode=%s success=%s steps=%s return=%.2f "
                "train_env_step=%s learner_update_steps=%s",
                phase_name,
                int(current_train_episode_id),
                bool(episode_success),
                int(episode_steps),
                float(episode_return),
                int(train_env_step),
                int(learner_update_steps["value"]),
            )

        summary = {
            "train_env_step": int(train_env_step),
            "decision_step": int(decision_step),
            "train_episode_id": int(train_episode_id),
            "train_total_success": int(train_total_success),
            "stopped_by_env_budget": bool(stopped_by_env_budget),
            "chunk_step_enabled": True,
            "controller_enabled": bool(getattr(env, "controller_enabled", False)),
            "agentlace_bootstrap_path": str(bootstrap_path),
            "learner_update_steps": int(learner_update_steps["value"]),
        }
        with open(
            run_dir / str(cfg.logging.summary_file),
            "w",
            encoding="utf-8",
        ) as fp:
            json.dump(summary, fp, indent=2)
        logger.info("reference-style actor done: %s", summary)

    finally:
        progress.close()
        try:
            client.update()
        except Exception:  # noqa: BLE001
            pass
        try:
            env.close(clear_cache=False)
        except Exception:  # noqa: BLE001
            pass
        policy_client_close = getattr(policy_client, "close", None)
        if callable(policy_client_close):
            try:
                policy_client_close()
            except Exception:  # noqa: BLE001
                pass


if __name__ == "__main__":
    main()
