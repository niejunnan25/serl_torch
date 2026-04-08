#!/usr/bin/env python3
from __future__ import annotations

"""Process queued AgiBot checkpoint evaluation jobs."""

import argparse
import json
import logging
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from omegaconf import OmegaConf

REPO_PARENT = Path(__file__).resolve().parents[5]
SERL_LAUNCHER_ROOT = Path(__file__).resolve().parents[4] / "serl_launcher"
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))
if str(SERL_LAUNCHER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERL_LAUNCHER_ROOT))

from serl_launcher.utils.alpha_utils import require_residual_alpha
from serl_launcher.utils.alpha_utils import validate_alpha


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return False


def _coalesce(value: Any, default: Any) -> Any:
    return default if value is None else value


def _to_override_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        if any(ch in value for ch in (" ", ",", "[", "]", "{", "}", "=", ":")):
            return json.dumps(value)
        return value
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_to_override_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return (
            "{"
            + ",".join(f"{k}:{_to_override_value(v)}" for k, v in value.items())
            + "}"
        )
    raise TypeError(f"Unsupported override value type: {type(value)}")


def _flatten_overrides(prefix: str, value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_key = f"{prefix}.{key}" if prefix else str(key)
            yield from _flatten_overrides(child_key, child)
        return
    yield f"{prefix}={_to_override_value(value)}"


def _extract_checkpoint_step(path: Path) -> Optional[int]:
    name = path.name
    if not (name.startswith("checkpoint_") and name.endswith(".pt")):
        return None
    stem = name[len("checkpoint_") : -len(".pt")]
    if not stem.isdigit():
        return None
    return int(stem)


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _load_completed_eval_indices(path: Path) -> set[int]:
    if not path.exists():
        return set()
    completed: set[int] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            eval_index = payload.get("eval_index", None)
            try:
                if eval_index is not None:
                    completed.add(int(eval_index))
            except Exception:  # noqa: BLE001
                continue
    return completed


def _is_checkpoint_stable(
    *,
    step: int,
    path: Path,
    now: float,
    stable_sec: float,
    stable_state: Dict[int, Dict[str, float]],
) -> bool:
    try:
        stat = path.stat()
    except FileNotFoundError:
        stable_state.pop(step, None)
        return False

    signature = (float(stat.st_size), float(stat.st_mtime))
    state = stable_state.get(step, None)
    if (
        state is None
        or (state["size"] != signature[0])
        or (state["mtime"] != signature[1])
    ):
        stable_state[step] = {"size": signature[0], "mtime": signature[1], "since": now}
        return False
    return bool((now - state["since"]) >= stable_sec)


def _load_summary(summary_path: Path) -> Optional[Dict[str, Any]]:
    if not summary_path.exists():
        return None
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def _resolve_eval_residual_alpha(
    *,
    train_cfg: Dict[str, Any],
    async_eval_cfg: Dict[str, Any],
    checkpoint_step: int,
    logger: logging.Logger,
) -> tuple[float, str, Optional[int]]:
    residual_cfg = train_cfg.get("residual", {}) if isinstance(train_cfg, dict) else {}
    base_alpha = require_residual_alpha(residual_cfg, path="residual.alpha")
    alpha_mode_raw = (
        str(async_eval_cfg.get("alpha_mode", "checkpoint_schedule"))
        .strip()
        .lower()
    )
    valid_modes = {"checkpoint_schedule", "base", "fixed"}
    if alpha_mode_raw not in valid_modes:
        raise ValueError(
            "training.async_eval.alpha_mode must be one of "
            f"{sorted(valid_modes)}, got {alpha_mode_raw!r}"
        )

    if alpha_mode_raw == "base":
        return float(base_alpha), "base", None

    if alpha_mode_raw == "fixed":
        fixed_alpha = async_eval_cfg.get("fixed_alpha", None)
        if fixed_alpha is None:
            raise ValueError(
                "training.async_eval.alpha_mode=fixed requires "
                "training.async_eval.fixed_alpha to be set"
            )
        return (
            validate_alpha(
                fixed_alpha,
                name="training.async_eval.fixed_alpha",
                allow_zero=True,
            ),
            "fixed",
            None,
        )

    training_cfg = train_cfg.get("training", {}) if isinstance(train_cfg, dict) else {}
    alpha_scheduler_cfg = (
        training_cfg.get("alpha_scheduler", {})
        if isinstance(training_cfg, dict)
        else {}
    )
    if (not isinstance(alpha_scheduler_cfg, dict)) or (
        not _as_bool(alpha_scheduler_cfg.get("enabled", False))
    ):
        return float(base_alpha), "checkpoint_schedule_disabled", int(checkpoint_step)

    min_alpha = validate_alpha(
        alpha_scheduler_cfg.get("min_alpha", base_alpha),
        name="training.alpha_scheduler.min_alpha",
        allow_zero=True,
    )
    warmup_steps = int(alpha_scheduler_cfg.get("warmup_steps", 0))
    anneal_steps = int(alpha_scheduler_cfg.get("anneal_steps", 1))
    # Async eval receives checkpoint snapshots keyed by train env-step only.
    # Therefore alpha reconstruction in checkpoint_schedule mode is always env-step based.
    schedule_step = int(checkpoint_step)

    if schedule_step < warmup_steps:
        return float(min_alpha), "checkpoint_schedule", schedule_step
    if anneal_steps <= 0:
        return float(base_alpha), "checkpoint_schedule", schedule_step

    progress = min(1.0, max(0.0, (schedule_step - warmup_steps) / float(anneal_steps)))
    alpha = float(min_alpha + (float(base_alpha) - min_alpha) * progress)
    return alpha, "checkpoint_schedule", schedule_step


def _build_eval_command(
    *,
    python_executable: str,
    eval_script: Path,
    train_cfg: Dict[str, Any],
    eval_run_dir: Path,
    checkpoint_path: Path,
    eval_seed: int,
    eval_cfg: Dict[str, Any],
    eval_residual_alpha: float,
) -> List[str]:
    overrides: List[str] = []

    for top_key in (
        "task",
        "residual",
        "chunk_step",
        "sac",
        "normalization",
        "openpi",
        "env",
    ):
        top_value = train_cfg.get(top_key, None)
        if isinstance(top_value, dict):
            overrides.extend(_flatten_overrides(top_key, top_value))

    # Explicit eval runtime overrides (last writer wins in Hydra).
    overrides.extend(
        [
            "env.backend=remote",
            f"env.remote.host={_to_override_value(eval_cfg['env_host'])}",
            f"env.remote.port={_to_override_value(eval_cfg['env_port'])}",
            f"env.remote.timeout_sec={_to_override_value(eval_cfg['env_timeout_sec'])}",
            f"openpi.host={_to_override_value(eval_cfg['openpi_host'])}",
            f"openpi.port={_to_override_value(eval_cfg['openpi_port'])}",
            f"residual.alpha={_to_override_value(eval_residual_alpha)}",
            f"task.seed_base={_to_override_value(eval_seed)}",
            f"eval.episodes={_to_override_value(eval_cfg['episodes'])}",
            f"eval.expert_check={_to_override_value(eval_cfg['expert_check'])}",
            f"eval.seed={_to_override_value(eval_cfg['seed'])}",
            f"eval.deterministic={_to_override_value(eval_cfg['deterministic'])}",
            f"eval.enable_base_probing={_to_override_value(eval_cfg['enable_base_probing'])}",
            f"eval.probing_alpha={_to_override_value(eval_cfg['probing_alpha'])}",
            f"eval.probing_min_steps={_to_override_value(eval_cfg['probing_min_steps'])}",
            f"eval.probing_max_steps={_to_override_value(eval_cfg['probing_max_steps'])}",
            f"eval.checkpoint_path={_to_override_value(str(checkpoint_path))}",
            f"hydra.run.dir={_to_override_value(str(eval_run_dir))}",
            "logging.tensorboard=false",
        ]
    )
    return [python_executable, str(eval_script), *overrides]


def _timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process queued checkpoint eval jobs for AgiBot training"
    )
    parser.add_argument("--train-run-dir", type=str, required=True)
    parser.add_argument("--train-config", type=str, default=None)
    parser.add_argument("--checkpoints-dir", type=str, default=None)
    parser.add_argument("--summary-jsonl", type=str, default=None)
    parser.add_argument("--queue-file", type=str, default=None)
    parser.add_argument("--poll-sec", type=float, default=None)
    parser.add_argument("--file-stable-sec", type=float, default=None)
    parser.add_argument(
        "--queue-policy", type=str, default=None, choices=("all", "latest")
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s"
    )
    logger = logging.getLogger("agibot_async_eval_watch")

    train_run_dir = Path(args.train_run_dir).expanduser().resolve()
    train_config_path = (
        Path(args.train_config).expanduser().resolve()
        if args.train_config is not None
        else (train_run_dir / ".hydra" / "config.yaml").resolve()
    )
    checkpoints_dir = (
        Path(args.checkpoints_dir).expanduser().resolve()
        if args.checkpoints_dir is not None
        else (train_run_dir / "checkpoints").resolve()
    )

    if not train_config_path.exists():
        raise FileNotFoundError(f"train config not found: {train_config_path}")
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"checkpoints dir not found: {checkpoints_dir}")

    train_cfg_omega = OmegaConf.load(str(train_config_path))
    train_cfg: Dict[str, Any] = OmegaConf.to_container(train_cfg_omega, resolve=False)  # type: ignore[assignment]

    training_cfg = train_cfg.get("training", {}) if isinstance(train_cfg, dict) else {}
    async_eval_cfg_raw = (
        training_cfg.get("async_eval", {}) if isinstance(training_cfg, dict) else {}
    )
    async_eval_cfg = async_eval_cfg_raw if isinstance(async_eval_cfg_raw, dict) else {}
    if not _as_bool(async_eval_cfg.get("enabled", False)):
        logger.info("training.async_eval.enabled is false; watcher exits.")
        return

    # Trigger cadence is decided by the trainer enqueue logic.
    # Watcher keeps this for config sanity checks and startup diagnostics.
    every_episodes = int(async_eval_cfg.get("every_episodes", 0))
    if every_episodes <= 0:
        raise ValueError(
            f"training.async_eval.every_episodes must be > 0, got {every_episodes}"
        )

    poll_sec = (
        float(args.poll_sec)
        if args.poll_sec is not None
        else float(async_eval_cfg.get("poll_sec", 15.0))
    )
    file_stable_sec = (
        float(args.file_stable_sec)
        if args.file_stable_sec is not None
        else float(async_eval_cfg.get("file_stable_sec", 5.0))
    )
    queue_policy = (
        str(args.queue_policy).lower()
        if args.queue_policy is not None
        else str(async_eval_cfg.get("queue_policy", "all")).lower()
    )
    if queue_policy not in {"all", "latest"}:
        raise ValueError(f"queue_policy must be 'all' or 'latest', got {queue_policy}")

    summary_jsonl = (
        Path(args.summary_jsonl).expanduser().resolve()
        if args.summary_jsonl is not None
        else Path(str(async_eval_cfg.get("summary_jsonl", "async_eval_results.jsonl")))
    )
    if not summary_jsonl.is_absolute():
        summary_jsonl = (train_run_dir / summary_jsonl).resolve()

    queue_file = (
        Path(args.queue_file).expanduser().resolve()
        if args.queue_file is not None
        else Path(str(async_eval_cfg.get("queue_file", "async_eval_queue.jsonl")))
    )
    if not queue_file.is_absolute():
        queue_file = (train_run_dir / queue_file).resolve()

    train_env_cfg = train_cfg.get("env", {}) if isinstance(train_cfg, dict) else {}
    train_remote_cfg = (
        train_env_cfg.get("remote", {}) if isinstance(train_env_cfg, dict) else {}
    )
    train_openpi_cfg = (
        train_cfg.get("openpi", {}) if isinstance(train_cfg, dict) else {}
    )

    reuse_openpi = _as_bool(async_eval_cfg.get("reuse_openpi_port", True))
    if reuse_openpi:
        openpi_host = str(train_openpi_cfg.get("host", "localhost"))
        openpi_port = int(train_openpi_cfg.get("port", 30001))
    else:
        openpi_host = str(
            _coalesce(
                async_eval_cfg.get("openpi_host", None),
                train_openpi_cfg.get("host", "localhost"),
            )
        )
        openpi_port = int(
            _coalesce(
                async_eval_cfg.get("openpi_port", None),
                train_openpi_cfg.get("port", 30001),
            )
        )

    env_host = str(
        async_eval_cfg.get("env_host", train_remote_cfg.get("host", "127.0.0.1"))
    )
    env_port = int(async_eval_cfg.get("env_port", 31014))
    env_timeout_sec = float(
        async_eval_cfg.get(
            "env_timeout_sec", train_remote_cfg.get("timeout_sec", 180.0)
        )
    )
    train_env_port = int(train_remote_cfg.get("port", 30000))
    if env_port == train_env_port:
        logger.warning(
            "async eval env port (%s) equals training env port (%s); this may cause reset/state conflicts",
            env_port,
            train_env_port,
        )

    episodes = int(async_eval_cfg.get("episodes", 50))
    seed = int(async_eval_cfg.get("seed", 7))
    deterministic = _as_bool(async_eval_cfg.get("deterministic", True))
    expert_check = _as_bool(async_eval_cfg.get("expert_check", False))
    enable_base_probing = _as_bool(async_eval_cfg.get("enable_base_probing", False))
    probing_alpha = async_eval_cfg.get("probing_alpha", None)
    probing_min_steps = int(async_eval_cfg.get("probing_min_steps", 0))
    probing_max_steps = int(async_eval_cfg.get("probing_max_steps", 0))
    alpha_mode = (
        str(async_eval_cfg.get("alpha_mode", "checkpoint_schedule"))
        .strip()
        .lower()
    )
    valid_modes = {"checkpoint_schedule", "base", "fixed"}
    if alpha_mode not in valid_modes:
        raise ValueError(
            "training.async_eval.alpha_mode must be one of "
            f"{sorted(valid_modes)}, got {alpha_mode!r}"
        )
    fixed_alpha_cfg = async_eval_cfg.get("fixed_alpha", None)
    if alpha_mode == "fixed" and fixed_alpha_cfg is None:
        raise ValueError(
            "training.async_eval.alpha_mode=fixed requires "
            "training.async_eval.fixed_alpha to be set"
        )
    fixed_alpha = (
        None
        if fixed_alpha_cfg is None
        else validate_alpha(
            fixed_alpha_cfg,
            name="training.async_eval.fixed_alpha",
            allow_zero=True,
        )
    )

    if alpha_mode == "checkpoint_schedule":
        chunk_step_cfg = (
            train_cfg.get("chunk_step", {}) if isinstance(train_cfg, dict) else {}
        )
        scheduler_clock = (
            str(chunk_step_cfg.get("scheduler_clock", "env_step")).strip().lower()
        )
        if scheduler_clock != "env_step":
            # Hard guard: checkpoint-schedule async eval only supports env-step clock.
            raise ValueError(
                "training.async_eval.alpha_mode=checkpoint_schedule requires "
                "training.chunk_step.scheduler_clock=env_step "
                f"(got {scheduler_clock})"
            )

    eval_script = async_eval_cfg.get("eval_script", None)
    if eval_script is None:
        eval_script_path = (
            Path(__file__).resolve().parent / "evaluate_checkpoint.py"
        ).resolve()
    else:
        eval_script_path = Path(str(eval_script)).expanduser()
        if not eval_script_path.is_absolute():
            eval_script_path = (Path(__file__).resolve().parent / eval_script_path).resolve()
    if not eval_script_path.exists():
        raise FileNotFoundError(f"eval script not found: {eval_script_path}")

    python_executable = str(
        _coalesce(async_eval_cfg.get("python_executable", None), sys.executable)
    )

    eval_cfg = {
        "episodes": episodes,
        "deterministic": deterministic,
        "expert_check": expert_check,
        "seed": seed,
        "enable_base_probing": enable_base_probing,
        "probing_alpha": probing_alpha,
        "probing_min_steps": probing_min_steps,
        "probing_max_steps": probing_max_steps,
        "env_host": env_host,
        "env_port": env_port,
        "env_timeout_sec": env_timeout_sec,
        "openpi_host": openpi_host,
        "openpi_port": openpi_port,
        "alpha_mode": alpha_mode,
        "fixed_alpha": fixed_alpha,
    }

    logger.info(
        "watch start: checkpoints=%s queue=%s every_episodes=%s eval_episodes=%s seed=%s alpha_mode=%s fixed_alpha=%s "
        "queue_policy=%s poll_sec=%.2f stable_sec=%.2f",
        checkpoints_dir,
        queue_file,
        every_episodes,
        episodes,
        seed,
        alpha_mode,
        fixed_alpha,
        queue_policy,
        poll_sec,
        file_stable_sec,
    )

    stop_requested = {"value": False}

    def _request_stop(signum: int, _frame: Any) -> None:
        logger.info("received signal=%s, stopping async eval watcher...", signum)
        stop_requested["value"] = True

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    completed_eval_indices = _load_completed_eval_indices(summary_jsonl)
    pending_requests: List[Dict[str, Any]] = []
    stable_state: Dict[int, Dict[str, float]] = {}
    async_eval_root = (train_run_dir / "async_eval").resolve()
    summary_file_name = str(
        (train_cfg.get("logging", {}) or {}).get("summary_file", "summary.json")
    )

    running_proc: Optional[subprocess.Popen] = None
    running_eval_index: Optional[int] = None
    running_train_episode_id: Optional[int] = None
    running_train_env_step: Optional[int] = None
    running_checkpoint_step: Optional[int] = None
    running_checkpoint_path: Optional[Path] = None
    running_seed: Optional[int] = None
    running_start_time = 0.0
    running_eval_dir: Optional[Path] = None
    running_log_fp = None
    running_eval_alpha: Optional[float] = None
    running_eval_alpha_mode: Optional[str] = None
    running_eval_alpha_schedule_step: Optional[int] = None

    while not stop_requested["value"]:
        if running_proc is not None:
            return_code = running_proc.poll()
            if return_code is None:
                time.sleep(min(1.0, max(0.1, poll_sec)))
                continue

            if running_log_fp is not None:
                running_log_fp.close()
                running_log_fp = None

            duration_sec = float(max(0.0, time.time() - running_start_time))
            summary = (
                _load_summary(running_eval_dir / summary_file_name)
                if running_eval_dir is not None
                else None
            )
            status = "ok" if return_code == 0 else "failed"
            record = {
                "timestamp": _timestamp(),
                "eval_index": int(running_eval_index)
                if running_eval_index is not None
                else None,
                "train_episode_id": (
                    int(running_train_episode_id)
                    if running_train_episode_id is not None
                    else None
                ),
                "train_env_step": int(running_train_env_step)
                if running_train_env_step is not None
                else None,
                "checkpoint_step": int(running_checkpoint_step)
                if running_checkpoint_step is not None
                else None,
                "checkpoint_path": str(running_checkpoint_path)
                if running_checkpoint_path is not None
                else None,
                "seed": int(running_seed) if running_seed is not None else None,
                "eval_alpha": float(running_eval_alpha)
                if running_eval_alpha is not None
                else None,
                "eval_alpha_mode": running_eval_alpha_mode,
                "eval_alpha_schedule_step": (
                    int(running_eval_alpha_schedule_step)
                    if running_eval_alpha_schedule_step is not None
                    else None
                ),
                "eval_run_dir": str(running_eval_dir)
                if running_eval_dir is not None
                else None,
                "status": status,
                "return_code": int(return_code),
                "duration_sec": duration_sec,
                "summary": summary,
            }
            _append_jsonl(summary_jsonl, record)
            if running_eval_index is not None:
                completed_eval_indices.add(int(running_eval_index))
            logger.info(
                "async eval finished: eval_index=%s train_episode=%s train_env_step=%s status=%s "
                "return_code=%s duration=%.1fs",
                running_eval_index,
                running_train_episode_id,
                running_train_env_step,
                status,
                return_code,
                duration_sec,
            )
            running_proc = None
            running_eval_index = None
            running_train_episode_id = None
            running_train_env_step = None
            running_checkpoint_step = None
            running_checkpoint_path = None
            running_seed = None
            running_eval_dir = None
            running_eval_alpha = None
            running_eval_alpha_mode = None
            running_eval_alpha_schedule_step = None
            continue

        queue_requests: List[Dict[str, Any]] = []
        if queue_file.exists():
            with open(queue_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(payload, dict):
                        continue
                    eval_index_raw = payload.get("eval_index", None)
                    checkpoint_step_raw = payload.get("checkpoint_step", None)
                    checkpoint_path_raw = payload.get("checkpoint_path", None)
                    train_episode_id_raw = payload.get("train_episode_id", None)
                    train_env_step_raw = payload.get("train_env_step", None)
                    try:
                        eval_index = int(eval_index_raw)
                        checkpoint_step = int(checkpoint_step_raw)
                        checkpoint_path = (
                            Path(str(checkpoint_path_raw)).expanduser().resolve()
                        )
                    except Exception:  # noqa: BLE001
                        continue
                    try:
                        train_episode_id = (
                            int(train_episode_id_raw)
                            if train_episode_id_raw is not None
                            else None
                        )
                    except Exception:  # noqa: BLE001
                        train_episode_id = None
                    try:
                        train_env_step = (
                            int(train_env_step_raw)
                            if train_env_step_raw is not None
                            else None
                        )
                    except Exception:  # noqa: BLE001
                        train_env_step = None
                    if eval_index in completed_eval_indices:
                        continue
                    if any(
                        int(request.get("eval_index", -1)) == eval_index
                        for request in pending_requests
                    ):
                        continue
                    queue_requests.append(
                        {
                            "eval_index": int(eval_index),
                            "train_episode_id": train_episode_id,
                            "train_env_step": train_env_step,
                            "checkpoint_step": int(checkpoint_step),
                            "checkpoint_path": checkpoint_path,
                        }
                    )

        now = time.time()
        discovered_steps: set[int] = set()
        for request in queue_requests:
            checkpoint_step = int(request["checkpoint_step"])
            checkpoint_path = Path(request["checkpoint_path"])
            discovered_steps.add(checkpoint_step)
            if not _is_checkpoint_stable(
                step=checkpoint_step,
                path=checkpoint_path,
                now=now,
                stable_sec=file_stable_sec,
                stable_state=stable_state,
            ):
                continue
            pending_requests.append(request)

        for stale_step in list(stable_state.keys()):
            if stale_step not in discovered_steps:
                stable_state.pop(stale_step, None)

        if queue_policy == "latest" and len(pending_requests) > 1:
            pending_requests = [
                max(pending_requests, key=lambda request: int(request["eval_index"]))
            ]

        pending_requests.sort(key=lambda request: int(request["eval_index"]))
        if pending_requests:
            request = pending_requests.pop(0 if queue_policy == "all" else -1)
            eval_index = int(request["eval_index"])
            train_episode_id = request.get("train_episode_id", None)
            train_env_step = request.get("train_env_step", None)
            checkpoint_step = int(request["checkpoint_step"])
            checkpoint_path = Path(request["checkpoint_path"])
            if not checkpoint_path.exists():
                logger.warning(
                    "checkpoint disappeared before eval launch: %s", checkpoint_path
                )
                time.sleep(poll_sec)
                continue

            eval_seed = int(seed)
            episode_label = (
                int(train_episode_id) if train_episode_id is not None else -1
            )
            step_label = (
                int(train_env_step) if train_env_step is not None else checkpoint_step
            )
            eval_run_dir = (
                async_eval_root / f"episode_{episode_label:06d}_step_{step_label:07d}"
            ).resolve()
            eval_run_dir.mkdir(parents=True, exist_ok=True)
            eval_log_path = eval_run_dir / "eval_runner.log"
            eval_alpha, eval_alpha_mode, eval_alpha_schedule_step = _resolve_eval_residual_alpha(
                train_cfg=train_cfg,
                async_eval_cfg=async_eval_cfg,
                checkpoint_step=int(checkpoint_step),
                logger=logger,
            )

            cmd = _build_eval_command(
                python_executable=python_executable,
                eval_script=eval_script_path,
                train_cfg=train_cfg,
                eval_run_dir=eval_run_dir,
                checkpoint_path=checkpoint_path,
                eval_seed=eval_seed,
                eval_cfg=eval_cfg,
                eval_residual_alpha=float(eval_alpha),
            )

            running_log_fp = open(eval_log_path, "a", encoding="utf-8")
            running_proc = subprocess.Popen(
                cmd,
                stdout=running_log_fp,
                stderr=subprocess.STDOUT,
            )
            running_eval_index = int(eval_index)
            running_train_episode_id = (
                int(train_episode_id) if train_episode_id is not None else None
            )
            running_train_env_step = (
                int(train_env_step) if train_env_step is not None else None
            )
            running_checkpoint_step = int(checkpoint_step)
            running_checkpoint_path = checkpoint_path
            running_seed = int(eval_seed)
            running_eval_dir = eval_run_dir
            running_start_time = time.time()
            running_eval_alpha = float(eval_alpha)
            running_eval_alpha_mode = str(eval_alpha_mode)
            running_eval_alpha_schedule_step = (
                int(eval_alpha_schedule_step)
                if eval_alpha_schedule_step is not None
                else None
            )

            logger.info(
                "async eval started: eval_index=%s train_episode=%s train_env_step=%s "
                "checkpoint_step=%s seed=%s alpha=%.6f alpha_mode=%s schedule_step=%s pid=%s run_dir=%s",
                eval_index,
                train_episode_id,
                train_env_step,
                checkpoint_step,
                eval_seed,
                float(eval_alpha),
                eval_alpha_mode,
                eval_alpha_schedule_step,
                running_proc.pid,
                eval_run_dir,
            )
            continue

        time.sleep(poll_sec)

    if running_proc is not None:
        logger.info(
            "terminating running async eval process (eval_index=%s train_episode=%s train_env_step=%s)...",
            running_eval_index,
            running_train_episode_id,
            running_train_env_step,
        )
        running_proc.terminate()
        try:
            running_proc.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            running_proc.kill()
            running_proc.wait(timeout=5.0)
        if running_log_fp is not None:
            running_log_fp.close()
            running_log_fp = None
        _append_jsonl(
            summary_jsonl,
            {
                "timestamp": _timestamp(),
                "eval_index": int(running_eval_index)
                if running_eval_index is not None
                else None,
                "train_episode_id": (
                    int(running_train_episode_id)
                    if running_train_episode_id is not None
                    else None
                ),
                "train_env_step": int(running_train_env_step)
                if running_train_env_step is not None
                else None,
                "checkpoint_step": int(running_checkpoint_step)
                if running_checkpoint_step is not None
                else None,
                "checkpoint_path": str(running_checkpoint_path)
                if running_checkpoint_path is not None
                else None,
                "seed": int(running_seed) if running_seed is not None else None,
                "eval_alpha": float(running_eval_alpha)
                if running_eval_alpha is not None
                else None,
                "eval_alpha_mode": running_eval_alpha_mode,
                "eval_alpha_schedule_step": (
                    int(running_eval_alpha_schedule_step)
                    if running_eval_alpha_schedule_step is not None
                    else None
                ),
                "eval_run_dir": str(running_eval_dir)
                if running_eval_dir is not None
                else None,
                "status": "aborted",
                "return_code": int(running_proc.returncode)
                if running_proc.returncode is not None
                else None,
                "duration_sec": float(max(0.0, time.time() - running_start_time)),
                "summary": None,
            },
        )

    logger.info("async eval watcher exited cleanly")


if __name__ == "__main__":
    main()
