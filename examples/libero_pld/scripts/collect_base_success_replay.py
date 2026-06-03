from __future__ import annotations

"""Collect successful base-policy LIBERO rollouts as PLD offline replay."""

import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

REPO_PARENT = Path(__file__).resolve().parents[4]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from serl_launcher.data.offline_prepared import EPISODE_FILE_GLOB  # noqa: E402
from serl_launcher.data.offline_prepared import MANIFEST_FILENAME  # noqa: E402
from serl_launcher.policy.typed_factory import build_policy_client  # noqa: E402
from serl_launcher.residual.observation import build_chunk_residual_obs  # noqa: E402
from serl_launcher.residual.observation import prepare_base_actions_chunk  # noqa: E402
from serl_launcher.utils.seeding import set_global_seeds  # noqa: E402
from serl_torch.examples.libero.config import cfg_to_log_payload  # noqa: E402
from serl_torch.examples.libero.config import parse_train_cfg  # noqa: E402
from serl_torch.examples.libero.env.factory import create_env  # noqa: E402
from serl_torch.examples.libero.env.observation import build_libero_state  # noqa: E402
from serl_torch.examples.libero.env.observation import extract_libero_images  # noqa: E402
from serl_torch.examples.libero.env.policy_input import build_libero_policy_input  # noqa: E402
from serl_torch.examples.libero.env.offline_data import OFFLINE_FORMAT_VERSION  # noqa: E402
from serl_torch.examples.libero.env.offline_data import prepare_fingerprint  # noqa: E402
from serl_torch.examples.libero.env.offline_data import prepared_dir_for_cfg  # noqa: E402
from serl_torch.examples.libero.env.offline_data import resolve_task_spec  # noqa: E402
from serl_torch.examples.libero.env.offline_data import write_manifest  # noqa: E402
from serl_torch.examples.libero.runtime.transition_assembly import (  # noqa: E402
    ChunkExecutionRecord,
)
from serl_torch.examples.libero.runtime.transition_assembly import (  # noqa: E402
    LiberoActorTransitionAssembler,
)


def _pld_section(raw_cfg: DictConfig) -> dict[str, Any]:
    value = OmegaConf.to_container(raw_cfg.get("pld", {}) or {}, resolve=True)
    return dict(value or {})


def _nested(payload: dict[str, Any], key: str, default: Any) -> Any:
    value: Any = payload
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def _attach_mc_returns(
    transitions: list[dict[str, Any]],
    *,
    discount: float,
) -> None:
    """Attach behavior-policy Monte Carlo return-to-go for Cal-QL calibration."""

    running_return = 0.0
    for transition in reversed(transitions):
        reward = float(transition.get("rewards", 0.0))
        mask = float(transition.get("masks", 0.0))
        running_return = reward + float(discount) * mask * running_return
        transition["mc_returns"] = np.float32(running_return)
        transition["mc_returns_valid"] = True


@hydra.main(version_base=None, config_path="../configs", config_name="pld_libero_spatial_task4")
def main(cfg: DictConfig) -> None:
    typed_cfg = parse_train_cfg(cfg)
    pld_cfg = _pld_section(cfg)
    target_successes = int(_nested(pld_cfg, "base_success.target_successes", 50))
    max_attempts = int(_nested(pld_cfg, "base_success.max_attempts", 500))
    clean_output = bool(_nested(pld_cfg, "base_success.clean_output", True))
    run_dir = Path(HydraConfig.get().runtime.output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    logger = logging.getLogger("collect_base_success")
    logger.info("Hydra run dir: %s", run_dir)
    logger.info("Config:\n%s", json.dumps(cfg_to_log_payload(typed_cfg), indent=2))
    set_global_seeds(typed_cfg.global_seed)

    task_spec = resolve_task_spec(typed_cfg)
    configured_path = typed_cfg.offline.prepared_path
    if configured_path:
        output_dir = Path(configured_path)
    else:
        output_dir = prepared_dir_for_cfg(typed_cfg, task_spec=task_spec)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if clean_output:
        for stale_path in output_dir.glob(EPISODE_FILE_GLOB):
            stale_path.unlink()
        manifest_path = output_dir / MANIFEST_FILENAME
        if manifest_path.exists():
            manifest_path.unlink()

    env = create_env(typed_cfg, logger)
    policy_client = build_policy_client(typed_cfg, logger=logger)
    transition_assembler = LiberoActorTransitionAssembler(
        cfg=typed_cfg,
        policy_client=policy_client,
        logger=logger,
    )
    task_prompt = str(env.task_description)
    episode_files: list[Path] = []
    stats: dict[str, Any] = {
        "episodes_attempted": 0,
        "episodes_written": 0,
        "steps_written": 0,
        "successes": 0,
        "failures": 0,
        "target_successes": int(target_successes),
        "max_attempts": int(max_attempts),
        "elapsed_sec": 0.0,
    }
    start_time = time.time()
    progress = tqdm(total=target_successes, desc="base successes", dynamic_ncols=True, leave=True)
    try:
        while int(stats["successes"]) < target_successes and int(stats["episodes_attempted"]) < max_attempts:
            attempt_id = int(stats["episodes_attempted"])
            obs = env.reset(seed=int(typed_cfg.env.seed), init_episode_idx=attempt_id)
            episode_step = 0
            episode_success = False
            episode_transitions: list[dict[str, Any]] = []
            while True:
                robot_state = build_libero_state(obs)
                image_observations = extract_libero_images(obs)
                policy_input = build_libero_policy_input(
                    prompt=task_prompt,
                    state=robot_state,
                    images=image_observations,
                )
                base_actions, _ = policy_client.infer(policy_input)
                base_actions = prepare_base_actions_chunk(
                    base_actions=base_actions,
                    chunk_horizon=int(typed_cfg.residual.chunk_horizon),
                )
                residual_obs = build_chunk_residual_obs(
                    robot_state=robot_state,
                    images=image_observations,
                    image_keys=typed_cfg.obs.image_keys,
                    base_actions=base_actions,
                    residual_alpha=float(typed_cfg.residual.alpha),
                )
                chunk_result = env.step_chunk(np.asarray(base_actions, dtype=np.float32))
                raw_chunk = ChunkExecutionRecord.from_env_chunk_result(
                    episode_id=int(attempt_id),
                    episode_step_start=int(episode_step),
                    residual_obs_before_chunk=residual_obs,
                    action_chunk=np.asarray(base_actions, dtype=np.float32),
                    chunk_result=chunk_result,
                )
                assembled_chunks = transition_assembler.handle_chunk(
                    raw=raw_chunk,
                    task_prompt=task_prompt,
                )
                for assembled_chunk in assembled_chunks:
                    episode_transitions.extend(assembled_chunk.transitions)
                episode_success = bool(
                    episode_success
                    or any(bool(info.get("env_done", False) or info.get("success", False)) for info in raw_chunk.infos)
                )
                episode_step += int(raw_chunk.executed_steps)
                obs = dict(raw_chunk.final_obs)
                if bool(raw_chunk.chunk_done or raw_chunk.chunk_truncated):
                    break
            if transition_assembler.async_transition_assembly_enabled:
                for assembled_chunk in transition_assembler.finish_episode(block=True):
                    episode_transitions.extend(assembled_chunk.transitions)
            stats["episodes_attempted"] = int(stats["episodes_attempted"]) + 1
            if episode_success and episode_transitions:
                _attach_mc_returns(
                    episode_transitions,
                    discount=float(typed_cfg.sac.discount),
                )
                success_idx = int(stats["successes"])
                episode_path = output_dir / f"episode_{success_idx:06d}.pkl"
                with open(episode_path, "wb") as fp:
                    pickle.dump(episode_transitions, fp, protocol=pickle.HIGHEST_PROTOCOL)
                episode_files.append(episode_path)
                stats["successes"] = int(stats["successes"]) + 1
                stats["episodes_written"] = int(stats["episodes_written"]) + 1
                stats["steps_written"] = int(stats["steps_written"]) + int(len(episode_transitions))
                progress.update(1)
            else:
                stats["failures"] = int(stats["failures"]) + 1
            logger.info(
                "attempt=%s success=%s steps=%s collected_successes=%s/%s",
                int(attempt_id),
                bool(episode_success),
                int(episode_step),
                int(stats["successes"]),
                int(target_successes),
            )
    finally:
        progress.close()
        stats["elapsed_sec"] = float(time.time() - start_time)
        manifest = write_manifest(
            manifest_path=output_dir / MANIFEST_FILENAME,
            task_spec=task_spec,
            cfg=typed_cfg,
            fingerprint=prepare_fingerprint(
                typed_cfg,
                task_spec=task_spec,
                offline_format_version=OFFLINE_FORMAT_VERSION,
            ),
            episode_files=episode_files,
            prepare_stats=stats,
        )
        summary = {
            "role": "collect_base_success_replay",
            "prepared_path": str(output_dir),
            "manifest_path": str(output_dir / MANIFEST_FILENAME),
            "stats": stats,
            "manifest": manifest,
        }
        with open(run_dir / typed_cfg.logging.summary_file, "w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2, ensure_ascii=False)
        policy_client_close = getattr(policy_client, "close", None)
        if callable(policy_client_close):
            try:
                policy_client_close()
            except Exception:
                pass
        transition_assembler.close()
        if str(typed_cfg.env.backend) != "remote":
            try:
                env.close(clear_cache=False)
            except Exception:
                pass
    if int(stats["successes"]) < target_successes:
        raise RuntimeError(
            f"Only collected {stats['successes']} successful episodes; target={target_successes}"
        )


if __name__ == "__main__":
    main()
