from __future__ import annotations

"""Reference-style AgiBot residual actor prototype.

This file is intentionally closer to the SERL reference examples:

1. build env / policy / agent directly
2. reset env
3. infer base chunk
4. sample residual chunk
5. compose final chunk
6. env.step_chunk(...)
7. insert replay
8. run learner updates

It is intentionally narrower than the production AgiBot actor entrypoint:
- sync learner only
- chunk-step path only
- no offline injection / online prefill / async eval
- no warmup phases
"""

import json
import logging
import sys
from pathlib import Path

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from serl_launcher.agents.continuous.drq import DrQAgent
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
from serl_launcher.residual.train.step_chunk_replay import ChunkReplayBuffer
from serl_launcher.training.checkpoint import _snapshot_agent_checkpoint_payload
from serl_launcher.training.checkpoint import _write_checkpoint_payload
from serl_launcher.training.loop_utils import _count_env_step_update_triggers
from serl_launcher.training.loop_utils import _iter_period_hits
from serl_launcher.utils.alpha_utils import require_residual_alpha
from serl_launcher.utils.logger import JsonlLogger

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
    if bool(cfg.get("training", {}).get("async", {}).get("enabled", False)):
        raise ValueError(
            "reference-style actor targets the sync learner path; "
            "set training.async.enabled=false"
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
    logger.info("Hydra run dir: %s", run_dir)
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    _validate_cfg(cfg)

    task_cfg = cfg.get("task", {})
    robot_cfg = cfg.get("robot", {})
    controller_cfg = OmegaConf.to_container(cfg.get("controller", {}), resolve=True)
    assets_root = robot_cfg.get("assets_root", None)

    retargeter_urdf_path = robot_cfg.get("retargeter_urdf_path", None)
    if retargeter_urdf_path is None:
        if assets_root is not None:
            retargeter_urdf_path = str(
                (Path(str(assets_root)).expanduser() / "G1" / "model.urdf").resolve()
            )
        else:
            retargeter_urdf_path = str(
                (
                    Path(__file__).resolve().parents[2] / "assets" / "G1" / "model.urdf"
                ).resolve()
            )
    else:
        retargeter_urdf_path = str(
            Path(str(retargeter_urdf_path)).expanduser().resolve()
        )

    retargeter_camera_extrinsic_path = robot_cfg.get(
        "retargeter_camera_extrinsic_path", None
    )
    if retargeter_camera_extrinsic_path is None:
        if assets_root is not None:
            retargeter_camera_extrinsic_path = str(
                (
                    Path(str(assets_root)).expanduser()
                    / "G1"
                    / "head_extrinsic_ours.json"
                ).resolve()
            )
        else:
            retargeter_camera_extrinsic_path = str(
                (
                    Path(__file__).resolve().parents[2]
                    / "assets"
                    / "G1"
                    / "head_extrinsic_ours.json"
                ).resolve()
            )
    else:
        retargeter_camera_extrinsic_path = str(
            Path(str(retargeter_camera_extrinsic_path)).expanduser().resolve()
        )

    env = AgiBotTaskEnv(
        task_name=str(task_cfg.get("name", "agibot_real_task")),
        prompt=str(task_cfg.get("prompt", task_cfg.get("name", "agibot_real_task"))),
        action_dim=int(cfg.get("env", {}).get("action_dim", 14)),
        control_mode=str(task_cfg.get("control_mode", "camera_position")),
        hz=float(task_cfg.get("hz", 20.0)),
        use_smooth_trajectory=bool(task_cfg.get("use_smooth_trajectory", False)),
        trajectory_time=task_cfg.get("trajectory_time", None),
        max_episode_steps=task_cfg.get("max_episode_steps", None),
        retargeter_urdf_path=retargeter_urdf_path,
        retargeter_camera_extrinsic_path=retargeter_camera_extrinsic_path,
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

    opt_cfg = cfg.sac.get("optimizer", None)
    actor_optimizer_kwargs = {"learning_rate": float(cfg.sac.learning_rate)}
    critic_optimizer_kwargs = {"learning_rate": float(cfg.sac.learning_rate)}
    temperature_optimizer_kwargs = {"learning_rate": float(cfg.sac.learning_rate)}
    if opt_cfg is not None:
        opt_type = str(opt_cfg.get("type", "adam")).lower()
        if opt_type not in {"adam", "adamw"}:
            raise ValueError(f"Unsupported sac.optimizer.type: {opt_type}")
        if opt_type == "adamw":
            weight_decay = float(opt_cfg.get("weight_decay", 0.0))
            actor_optimizer_kwargs["weight_decay"] = weight_decay
            critic_optimizer_kwargs["weight_decay"] = weight_decay
        warmup_steps = opt_cfg.get("warmup_steps", None)
        if warmup_steps is not None:
            actor_optimizer_kwargs["warmup_steps"] = int(warmup_steps)
            critic_optimizer_kwargs["warmup_steps"] = int(warmup_steps)
            temperature_optimizer_kwargs["warmup_steps"] = int(warmup_steps)
        cosine_decay_steps = opt_cfg.get("cosine_decay_steps", None)
        if cosine_decay_steps is not None:
            actor_optimizer_kwargs["cosine_decay_steps"] = int(cosine_decay_steps)
            critic_optimizer_kwargs["cosine_decay_steps"] = int(cosine_decay_steps)
            temperature_optimizer_kwargs["cosine_decay_steps"] = int(cosine_decay_steps)
        grad_clip_norm = opt_cfg.get("grad_clip_norm", None)
        if grad_clip_norm is not None:
            actor_optimizer_kwargs["clip_grad_norm"] = float(grad_clip_norm)
            critic_optimizer_kwargs["clip_grad_norm"] = float(grad_clip_norm)
            temperature_optimizer_kwargs["clip_grad_norm"] = float(grad_clip_norm)
        if opt_type == "adamw":
            temp_weight_decay = opt_cfg.get("temperature_weight_decay", None)
            if temp_weight_decay is not None:
                temperature_optimizer_kwargs["weight_decay"] = float(temp_weight_decay)

    resnet_kwargs = None
    resnet_cfg = cfg.sac.get("resnet", None)
    if resnet_cfg is not None:
        model_name = str(resnet_cfg.get("model_name", "microsoft/resnet-18"))
        if not Path(model_name).is_absolute() and not model_name.startswith(
            ("http://", "https://")
        ):
            candidate = Path(get_original_cwd()) / model_name
            if candidate.is_dir():
                model_name = str(candidate)
        resnet_kwargs = {
            "model_name": model_name,
            "pretrained": bool(resnet_cfg.get("pretrained", True)),
            "freeze_backbone": bool(resnet_cfg.get("freeze_backbone", False)),
            "pooling_method": str(
                resnet_cfg.get("pooling_method", "spatial_learned_embeddings")
            ),
            "num_spatial_blocks": int(resnet_cfg.get("num_spatial_blocks", 8)),
            "bottleneck_dim": int(resnet_cfg.get("bottleneck_dim", 256)),
        }

    mixed_precision_cfg = cfg.get("training", {}).get("mixed_precision", None)
    mixed_precision = {
        "enabled": bool(
            mixed_precision_cfg.get("enabled", False)
            if mixed_precision_cfg is not None
            else False
        ),
        "dtype": str(
            mixed_precision_cfg.get("dtype", "bfloat16")
            if mixed_precision_cfg is not None
            else "bfloat16"
        ),
    }

    sample_action = np.zeros((int(step_action_dim * chunk_horizon),), dtype=np.float32)
    sample_critic_action = np.zeros(
        (int(env_action_dim * chunk_horizon),), dtype=np.float32
    )
    agent = DrQAgent.create_drq(
        0,
        sample_obs,
        sample_action,
        critic_actions=sample_critic_action,
        encoder_type=str(cfg.sac.encoder_type),
        shared_encoder=bool(cfg.sac.shared_encoder),
        use_proprio=bool(cfg.sac.use_proprio),
        critic_network_kwargs={
            "activations": str(cfg.sac.critic_activation),
            "use_layer_norm": bool(cfg.sac.critic_layer_norm),
            "hidden_dims": [int(v) for v in cfg.sac.critic_hidden_dims],
        },
        policy_network_kwargs={
            "activations": str(cfg.sac.policy_activation),
            "use_layer_norm": bool(cfg.sac.policy_layer_norm),
            "hidden_dims": [int(v) for v in cfg.sac.policy_hidden_dims],
        },
        policy_kwargs={
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
            "std_min": float(cfg.sac.std_min),
            "std_max": float(cfg.sac.std_max),
        },
        actor_optimizer_kwargs=actor_optimizer_kwargs,
        critic_optimizer_kwargs=critic_optimizer_kwargs,
        temperature_optimizer_kwargs=temperature_optimizer_kwargs,
        discount=float(cfg.sac.discount),
        soft_target_update_rate=float(cfg.sac.soft_target_update_rate),
        temperature_init=float(cfg.sac.temperature_init),
        backup_entropy=bool(cfg.sac.backup_entropy),
        critic_ensemble_size=int(cfg.sac.critic_ensemble_size),
        critic_subsample_size=(
            int(cfg.sac.critic_subsample_size)
            if cfg.sac.critic_subsample_size is not None
            else None
        ),
        image_keys=image_keys,
        resnet_kwargs=resnet_kwargs,
        action_transform=action_transform,
        mixed_precision=mixed_precision,
        otf_num_samples=int(cfg.sac.get("otf_num_samples", 1)),
        cql_n_actions=int(cfg.sac.get("cql_n_actions", 10)),
        cql_temperature=float(cfg.sac.get("cql_temperature", 1.0)),
    )

    replay_buffer = ChunkReplayBuffer(
        sample_observation_template=sample_obs,
        state_core_dim=int(sample_state_core.shape[0]),
        step_action_dim=env_action_dim,
        chunk_horizon=chunk_horizon,
        discount=float(cfg.sac.discount),
        capacity=int(cfg.replay.capacity),
        sample_stride=int(cfg.chunk_step.sample_stride),
        require_full_horizon=bool(cfg.chunk_step.require_full_horizon),
        pad_action_to_horizon=bool(cfg.chunk_step.pad_action_to_horizon),
        state_mode=obs_state_mode,
    )

    checkpoint_cfg = cfg.training.get("checkpoint", {})
    checkpoint_every_steps = int(checkpoint_cfg.get("every_steps", 0))
    checkpoint_keep = int(checkpoint_cfg.get("keep", 0))
    checkpoint_dir = run_dir / str(checkpoint_cfg.get("dir", "checkpoints"))
    episode_logger = JsonlLogger(run_dir / str(cfg.logging.episode_log_file))
    phase_cfg = cfg.training.phases[0]
    phase_name = str(phase_cfg.get("name", "train"))
    total_episodes = int(phase_cfg.get("episodes", 0))

    max_train_env_steps = int(cfg.training.get("max_train_env_steps", 0))
    progress = tqdm(
        total=max_train_env_steps if max_train_env_steps > 0 else None,
        desc="train_env_step",
        dynamic_ncols=True,
        leave=True,
    )

    train_env_step = 0
    train_episode_id = 0
    train_total_success = 0
    stopped_by_env_budget = False
    last_update_info: dict[str, object] = {}

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

                alpha_step = float(residual_alpha)

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
                    alpha=float(alpha_step),
                    state_mode=obs_state_mode,
                )

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
                    alpha=float(alpha_step),
                    clip_gripper=bool(cfg.residual.clip_gripper),
                )

                replay_size_before = int(replay_buffer.num_steps)
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

                    replay_buffer.insert(
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
                            "alpha": float(alpha_step),
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

                replay_size_after = int(replay_buffer.num_steps)
                if replay_size_after >= int(cfg.training.training_starts):
                    trigger_count = _count_env_step_update_triggers(
                        train_step_before=int(train_env_step_before),
                        train_step_after=int(train_env_step),
                        replay_size_before=int(replay_size_before),
                        replay_size_after=int(replay_size_after),
                        training_starts=int(cfg.training.training_starts),
                        update_every=int(cfg.training.update_every),
                    )
                    num_updates = int(trigger_count) * int(
                        cfg.training.updates_per_step
                    )
                    for _ in range(num_updates):
                        batch = replay_buffer.sample(int(cfg.replay.batch_size))
                        agent, last_update_info = agent.update_high_utd(
                            batch,
                            utd_ratio=int(cfg.sac.utd_ratio),
                        )

                for checkpoint_step in _iter_period_hits(
                    step_before=int(train_env_step_before),
                    step_after=int(train_env_step),
                    period=int(checkpoint_every_steps),
                ):
                    _write_checkpoint_payload(
                        None,
                        str(checkpoint_dir),
                        _snapshot_agent_checkpoint_payload(
                            agent, step=int(checkpoint_step)
                        ),
                        step=int(checkpoint_step),
                        keep=int(checkpoint_keep),
                    )

                obs_raw = next_obs_raw

            train_total_success += int(episode_success)
            train_episode_id = int(current_train_episode_id)
            running_success_rate = float(train_total_success) / float(train_episode_id)

            episode_logger.write(
                {
                    "phase": phase_name,
                    "train_episode_id": int(current_train_episode_id),
                    "success": bool(episode_success),
                    "episode_steps": int(episode_steps),
                    "episode_return": float(episode_return),
                    "train_env_step": int(train_env_step),
                    "running_success_rate": float(running_success_rate),
                    "last_update_info": {
                        key: (
                            float(value)
                            if isinstance(value, (int, float, np.floating))
                            else value
                        )
                        for key, value in dict(last_update_info).items()
                    },
                }
            )

            logger.info(
                "phase=%s train_episode=%s success=%s steps=%s return=%.2f train_env_step=%s",
                phase_name,
                int(current_train_episode_id),
                bool(episode_success),
                int(episode_steps),
                float(episode_return),
                int(train_env_step),
            )

        summary = {
            "train_env_step": int(train_env_step),
            "train_episode_id": int(train_episode_id),
            "train_total_success": int(train_total_success),
            "stopped_by_env_budget": bool(stopped_by_env_budget),
            "chunk_step_enabled": True,
            "controller_enabled": bool(getattr(env, "controller_enabled", False)),
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
        episode_logger.close()
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
