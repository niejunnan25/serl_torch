from __future__ import annotations

"""Hydra-native async stack launcher for LIBERO residual SAC training.

This launcher starts:
1) remote env server
2) optional async-eval env server
3) OpenPI server
4) external agentlace learner
5) actor (foreground)

Compared with shell-only orchestration, this entrypoint lets us manage launch
resources (GPUs/output roots/session tag) from YAML in a RLinf-like style.
"""

import atexit
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Sequence

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf


@dataclass
class _ManagedProcess:
    name: str
    process: subprocess.Popen[bytes]
    log_file: IO[bytes] | None
    log_path: Path


def _as_int(value: object, default: int) -> int:
    if value is None:
        return int(default)
    return int(value)


def _resolve_path(raw: str | Path, *, base: Path) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def _port_open(host: str, port: int, timeout_sec: float = 0.3) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout_sec):
            return True
    except OSError:
        return False


def _wait_for_port(
    *,
    name: str,
    host: str,
    port: int,
    timeout_sec: float,
    logger: logging.Logger,
) -> None:
    deadline = time.time() + float(timeout_sec)
    while time.time() < deadline:
        if _port_open(host, port):
            logger.info("%s is ready at %s:%s", name, host, int(port))
            return
        time.sleep(1.0)
    raise RuntimeError(f"Timed out waiting for {name} at {host}:{int(port)}")


def _launch_background(
    *,
    name: str,
    cmd: Sequence[str],
    cwd: Path,
    log_path: Path,
) -> _ManagedProcess:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fp = log_path.open("ab")
    proc = subprocess.Popen(
        list(cmd),
        cwd=str(cwd),
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    return _ManagedProcess(name=name, process=proc, log_file=log_fp, log_path=log_path)


def _terminate_process(proc: _ManagedProcess, *, logger: logging.Logger) -> None:
    if proc.process.poll() is not None:
        if proc.log_file is not None:
            proc.log_file.close()
            proc.log_file = None
        return
    try:
        os.killpg(proc.process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        logger.warning("%s did not exit on SIGTERM, sending SIGKILL", proc.name)
        try:
            os.killpg(proc.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.process.wait(timeout=5.0)
    if proc.log_file is not None:
        proc.log_file.close()
        proc.log_file = None


def _stream_foreground_to_console_and_log(
    *,
    name: str,
    cmd: Sequence[str],
    cwd: Path,
    log_path: Path,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log_fp:
        proc = subprocess.Popen(
            list(cmd),
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        assert proc.stdout is not None
        stdout_buffer = getattr(sys.stdout, "buffer", None)
        try:
            while True:
                # Read raw chunks so carriage-return based tqdm refreshes are
                # forwarded in real time instead of being delayed until '\n'.
                chunk = proc.stdout.read1(4096)
                if not chunk:
                    break
                if stdout_buffer is not None:
                    stdout_buffer.write(chunk)
                    stdout_buffer.flush()
                else:
                    sys.stdout.write(chunk.decode(errors="replace"))
                    sys.stdout.flush()
                log_fp.write(chunk)
                log_fp.flush()
        finally:
            proc.stdout.close()
        return proc.wait()


@hydra.main(version_base=None, config_path="../conf", config_name="train_residual_sac")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    logger = logging.getLogger("libero_launch_async_train")

    repo_root = Path(__file__).resolve().parents[3]
    tools_dir = repo_root / "examples" / "libero" / "tools"
    original_cwd = Path(get_original_cwd()).resolve()

    if str(cfg.get("env", {}).get("backend", "remote")) != "remote":
        raise ValueError(
            f"launch_async_train requires env.backend=remote, got {cfg.get('env', {}).get('backend')}"
        )

    launch_cfg = cfg.get("launch", {})
    default_gpu = _as_int(launch_cfg.get("gpu_id", 0), 0)
    actor_gpu = _as_int(launch_cfg.get("actor_gpu", default_gpu), default_gpu)
    learner_gpu = _as_int(launch_cfg.get("learner_gpu", default_gpu), default_gpu)
    openpi_gpu = _as_int(launch_cfg.get("openpi_gpu", actor_gpu), actor_gpu)

    output_root = _resolve_path(
        launch_cfg.get("output_root", "outputs/libero/launch_async_train"),
        base=original_cwd,
    )
    config_name_raw = str(HydraConfig.get().job.config_name or "train_residual_sac")
    config_name = Path(config_name_raw).stem or "train_residual_sac"
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    output_root_leaf = Path(output_root.name).stem
    if output_root_leaf == config_name:
        run_parent = output_root
    else:
        run_parent = output_root / config_name
    run_parent.mkdir(parents=True, exist_ok=True)
    run_root = run_parent / stamp
    if run_root.exists():
        suffix = 2
        while (run_parent / f"{stamp}_{suffix}").exists():
            suffix += 1
        run_root = run_parent / f"{stamp}_{suffix}"
    support_dir = run_root / "support"
    actor_run_dir = run_root / "actor"
    learner_run_dir = run_root / "learner"
    support_dir.mkdir(parents=True, exist_ok=True)
    actor_run_dir.mkdir(parents=True, exist_ok=True)
    learner_run_dir.mkdir(parents=True, exist_ok=True)

    bootstrap_path = run_root / "agentlace_bootstrap.pkl"
    resolved_cfg_path = run_root / "resolved_config.yaml"
    resolved_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.save(config=resolved_cfg, f=str(resolved_cfg_path), resolve=True)

    env_host = str(cfg.env.remote.host)
    env_port = int(cfg.env.remote.port)
    openpi_host = str(cfg.openpi.host)
    openpi_port = int(cfg.openpi.port)
    async_eval_cfg = cfg.get("training", {}).get("async_eval", {})
    async_eval_enabled = bool(async_eval_cfg.get("enabled", False))
    async_eval_env_host = str(async_eval_cfg.get("env_host", env_host))
    async_eval_env_port = int(async_eval_cfg.get("env_port", env_port))
    async_cfg = cfg.get("training", {}).get("async", {})
    trainer_host = str(async_cfg.get("trainer_host", "127.0.0.1"))
    trainer_port = int(async_cfg.get("trainer_port", 5488))
    trainer_port_check_host = "127.0.0.1" if trainer_host == "0.0.0.0" else trainer_host

    env_timeout = float(launch_cfg.get("env_start_timeout_sec", 60.0))
    openpi_timeout = float(launch_cfg.get("openpi_start_timeout_sec", 300.0))

    logger.info("Hydra run dir: %s", Path(HydraConfig.get().runtime.output_dir).resolve())
    logger.info("Config name: %s", config_name)
    logger.info(
        "Launch resources: actor_gpu=%s learner_gpu=%s openpi_gpu=%s output_root=%s",
        actor_gpu,
        learner_gpu,
        openpi_gpu,
        output_root,
    )
    logger.info(
        "Ports: env=%s:%s eval_env=%s:%s openpi=%s:%s trainer=%s:%s",
        env_host,
        env_port,
        async_eval_env_host,
        async_eval_env_port,
        openpi_host,
        openpi_port,
        trainer_host,
        trainer_port,
    )
    logger.info("Resolved run root: %s", run_root)

    if _port_open(env_host, env_port):
        raise RuntimeError(f"Env port already in use: {env_host}:{env_port}")
    if _port_open(openpi_host, openpi_port):
        raise RuntimeError(f"OpenPI port already in use: {openpi_host}:{openpi_port}")
    if _port_open(trainer_port_check_host, trainer_port):
        raise RuntimeError(
            f"Trainer port already in use: {trainer_port_check_host}:{trainer_port}"
        )

    managed: list[_ManagedProcess] = []

    def _cleanup() -> None:
        for proc in reversed(managed):
            _terminate_process(proc, logger=logger)

    atexit.register(_cleanup)

    def _signal_handler(_sig: int, _frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    env_log = support_dir / "env_server.log"
    eval_env_log = support_dir / "async_eval_env_server.log"
    openpi_log = support_dir / "openpi_server.log"
    learner_log = support_dir / "learner.log"
    actor_log = support_dir / "actor.log"

    try:
        logger.info("Starting env server...")
        env_proc = _launch_background(
            name="env server",
            cmd=[
                "bash",
                str(tools_dir / "serve_env.sh"),
                "--host",
                env_host,
                "--port",
                str(env_port),
            ],
            cwd=repo_root,
            log_path=env_log,
        )
        managed.append(env_proc)
        _wait_for_port(
            name="env server",
            host=env_host,
            port=env_port,
            timeout_sec=env_timeout,
            logger=logger,
        )

        if async_eval_enabled:
            same_as_train_env = (
                async_eval_env_host == env_host and async_eval_env_port == env_port
            )
            if same_as_train_env:
                logger.warning(
                    "Async eval env matches training env (%s:%s); isolated eval env will not start.",
                    env_host,
                    env_port,
                )
            elif _port_open(async_eval_env_host, async_eval_env_port):
                logger.info(
                    "Reusing existing async eval env server at %s:%s",
                    async_eval_env_host,
                    async_eval_env_port,
                )
            else:
                logger.info("Starting async eval env server...")
                eval_env_proc = _launch_background(
                    name="async eval env server",
                    cmd=[
                        "bash",
                        str(tools_dir / "serve_env.sh"),
                        "--host",
                        async_eval_env_host,
                        "--port",
                        str(async_eval_env_port),
                    ],
                    cwd=repo_root,
                    log_path=eval_env_log,
                )
                managed.append(eval_env_proc)
                _wait_for_port(
                    name="async eval env server",
                    host=async_eval_env_host,
                    port=async_eval_env_port,
                    timeout_sec=env_timeout,
                    logger=logger,
                )

        logger.info("Starting OpenPI server...")
        openpi_proc = _launch_background(
            name="OpenPI server",
            cmd=[
                "bash",
                str(tools_dir / "serve_openpi.sh"),
                "--port",
                str(openpi_port),
                "--gpu-id",
                str(openpi_gpu),
            ],
            cwd=repo_root,
            log_path=openpi_log,
        )
        managed.append(openpi_proc)
        _wait_for_port(
            name="OpenPI server",
            host=openpi_host,
            port=openpi_port,
            timeout_sec=openpi_timeout,
            logger=logger,
        )

        learner_cmd = [
            "bash",
            str(tools_dir / "run_learner.sh"),
            str(resolved_cfg_path),
            "--bootstrap",
            str(bootstrap_path),
            "--gpu_id",
            str(learner_gpu),
            f"hydra.run.dir={learner_run_dir}",
        ]
        logger.info("Starting learner...")
        logger.info("Learner cmd: %s", " ".join(learner_cmd))
        learner_proc = _launch_background(
            name="learner",
            cmd=learner_cmd,
            cwd=repo_root,
            log_path=learner_log,
        )
        managed.append(learner_proc)
        time.sleep(2.0)
        if learner_proc.process.poll() is not None:
            raise RuntimeError(f"Learner exited early. See log: {learner_log}")

        actor_cmd = [
            "bash",
            str(tools_dir / "run_actor.sh"),
            str(resolved_cfg_path),
            "--bootstrap",
            str(bootstrap_path),
            "--gpu_id",
            str(actor_gpu),
            f"hydra.run.dir={actor_run_dir}",
        ]
        logger.info("Starting actor (foreground)...")
        logger.info("Actor cmd: %s", " ".join(actor_cmd))
        rc = _stream_foreground_to_console_and_log(
            name="actor",
            cmd=actor_cmd,
            cwd=repo_root,
            log_path=actor_log,
        )
        if rc != 0:
            raise RuntimeError(f"Actor exited with code {rc}. See log: {actor_log}")
    except KeyboardInterrupt:
        logger.warning("Interrupted by user, shutting down services...")
        raise
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
