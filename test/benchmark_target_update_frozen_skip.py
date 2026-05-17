#!/usr/bin/env python3
from __future__ import annotations

"""Benchmark target-network soft update variants.

This isolates the cost of ``target = (1 - tau) * target + tau * source`` for a
critic that resembles the LIBERO residual learner: two frozen ResNet-18 camera
encoders, proprio projection, and a two-head Q ensemble.
"""

import argparse
import copy
import gc
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "serl_launcher") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "serl_launcher"))

import torch
import torch.nn as nn

from serl_launcher.common.encoding import EncodingWrapper
from serl_launcher.networks.actor_critic_nets import Critic, CriticEnsemble
from serl_launcher.networks.mlp import MLP
from serl_launcher.vision.resnet_v1 import ResNetEncoder


def _summary(values: Iterable[float]) -> dict[str, float]:
    values = [float(value) for value in values]
    if not values:
        return {
            "mean_us": 0.0,
            "median_us": 0.0,
            "min_us": 0.0,
            "max_us": 0.0,
            "updates_per_sec": 0.0,
        }
    mean_s = statistics.mean(values)
    return {
        "mean_us": float(mean_s * 1_000_000.0),
        "median_us": float(statistics.median(values) * 1_000_000.0),
        "min_us": float(min(values) * 1_000_000.0),
        "max_us": float(max(values) * 1_000_000.0),
        "updates_per_sec": float(1.0 / mean_s) if mean_s > 0.0 else 0.0,
    }


def _print_event(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True), flush=True)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _make_critic(args: argparse.Namespace, device: torch.device) -> nn.Module:
    backbone = ResNetEncoder.create_backbone(
        model_name=str(args.model_name),
        pretrained=not bool(args.random_init),
        freeze=bool(args.freeze_backbone),
    )
    encoders = {
        key: ResNetEncoder(
            backbone=backbone,
            freeze_backbone=bool(args.freeze_backbone),
            pooling_method="spatial_learned_embeddings",
            num_spatial_blocks=8,
            bottleneck_dim=int(args.bottleneck_dim),
        )
        for key in ("image0", "image1")
    }
    encoder = EncodingWrapper(
        encoder=encoders,
        use_proprio=True,
        proprio_latent_dim=int(args.proprio_latent_dim),
        image_keys=("image0", "image1"),
        vector_obs_keys=("state",),
        fuse_views=str(args.fuse_views),
    )
    critic_ctor = lambda: Critic(
        encoder=encoder,
        network=MLP(
            hidden_dims=(256, 256, 256),
            activations="tanh",
            activate_final=True,
            use_layer_norm=True,
        ),
    )
    critic = CriticEnsemble(critic_ctor=critic_ctor, num_qs=int(args.num_qs)).to(device)
    return critic


def _materialize(
    critic: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    obs = {
        "image0": torch.randint(
            0,
            256,
            (int(args.batch_size), int(args.image_size), int(args.image_size), 3),
            device=device,
            dtype=torch.uint8,
        ),
        "image1": torch.randint(
            0,
            256,
            (int(args.batch_size), int(args.image_size), int(args.image_size), 3),
            device=device,
            dtype=torch.uint8,
        ),
        "state": torch.randn(
            int(args.batch_size),
            int(args.proprio_dim),
            device=device,
        ),
    }
    actions = torch.randn(int(args.batch_size), int(args.action_dim), device=device)
    with torch.no_grad():
        out = critic(obs, actions, train=False)
        _sync(device)
    if tuple(out.shape) != (int(args.num_qs), int(args.batch_size)):
        raise RuntimeError(f"unexpected critic output shape: {tuple(out.shape)}")


def _parameter_stats(module: nn.Module) -> dict[str, int]:
    total_tensors = 0
    total_elements = 0
    trainable_tensors = 0
    trainable_elements = 0
    frozen_tensors = 0
    frozen_elements = 0
    for param in module.parameters():
        total_tensors += 1
        numel = int(param.numel())
        total_elements += numel
        if bool(param.requires_grad):
            trainable_tensors += 1
            trainable_elements += numel
        else:
            frozen_tensors += 1
            frozen_elements += numel
    return {
        "total_tensors": total_tensors,
        "total_elements": total_elements,
        "trainable_tensors": trainable_tensors,
        "trainable_elements": trainable_elements,
        "frozen_tensors": frozen_tensors,
        "frozen_elements": frozen_elements,
    }


def _current_loop_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)


def _skip_frozen_loop_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            if (not bool(target_param.requires_grad)) and (
                not bool(source_param.requires_grad)
            ):
                continue
            target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)


def _skip_frozen_foreach_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    target_params = []
    source_params = []
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        if (not bool(target_param.requires_grad)) and (
            not bool(source_param.requires_grad)
        ):
            continue
        target_params.append(target_param)
        source_params.append(source_param)
    if not target_params:
        return
    with torch.no_grad():
        torch._foreach_mul_(target_params, 1.0 - tau)
        torch._foreach_add_(target_params, source_params, alpha=tau)


def _make_update_fn(
    strategy: str,
) -> Callable[[nn.Module, nn.Module, float], None]:
    if strategy == "current_loop_all_params":
        return _current_loop_update
    if strategy == "skip_frozen_loop":
        return _skip_frozen_loop_update
    if strategy == "skip_frozen_foreach":
        return _skip_frozen_foreach_update
    raise ValueError(f"unknown strategy: {strategy}")


def _time_update(
    update_fn: Callable[[nn.Module, nn.Module, float], None],
    *,
    source: nn.Module,
    target: nn.Module,
    tau: float,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> list[float]:
    for _ in range(int(warmup)):
        update_fn(target, source, tau)
    _sync(device)
    times: list[float] = []
    for _ in range(int(iterations)):
        start = time.perf_counter()
        update_fn(target, source, tau)
        _sync(device)
        times.append(time.perf_counter() - start)
    return times


def _run_strategy(
    strategy: str,
    *,
    source: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    target = copy.deepcopy(source).to(device)
    if bool(args.compile_modules):
        source_for_update = torch.compile(
            source,
            backend=str(args.compile_backend),
            mode=str(args.compile_mode),
            fullgraph=bool(args.compile_fullgraph),
            dynamic=False,
        )
        target_for_update = torch.compile(
            target,
            backend=str(args.compile_backend),
            mode=str(args.compile_mode),
            fullgraph=bool(args.compile_fullgraph),
            dynamic=False,
        )
    else:
        source_for_update = source
        target_for_update = target

    update_fn = _make_update_fn(strategy)
    times = _time_update(
        update_fn,
        source=source_for_update,
        target=target_for_update,
        tau=float(args.tau),
        warmup=int(args.warmup),
        iterations=int(args.iterations),
        device=device,
    )
    return {
        "strategy": strategy,
        "ok": True,
        "summary": _summary(times),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=[
            "current_loop_all_params",
            "skip_frozen_loop",
            "skip_frozen_foreach",
        ],
        choices=[
            "current_loop_all_params",
            "skip_frozen_loop",
            "skip_frozen_foreach",
        ],
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--action-dim", type=int, default=35)
    parser.add_argument("--proprio-dim", type=int, default=8)
    parser.add_argument("--proprio-latent-dim", type=int, default=64)
    parser.add_argument("--bottleneck-dim", type=int, default=256)
    parser.add_argument("--num-qs", type=int, default=2)
    parser.add_argument("--model-name", default="pretrained_models/microsoft--resnet-18")
    parser.add_argument("--random-init", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fuse-views", default="auto", choices=("auto", "true", "false"))
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--compile-modules", action="store_true")
    parser.add_argument("--compile-backend", default="inductor")
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--compile-fullgraph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")

    torch.manual_seed(int(args.seed))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    source = _make_critic(args, device)
    _materialize(source, args, device)
    stats = _parameter_stats(source)
    _print_event({"event": "parameter_stats", **stats})

    results = []
    for strategy in args.strategies:
        _print_event({"event": "start_strategy", "strategy": strategy})
        try:
            result = _run_strategy(strategy, source=source, args=args, device=device)
        except Exception as exc:  # noqa: BLE001
            result = {
                "strategy": strategy,
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        results.append(result)
        _print_event({"event": "strategy_result", **result})
        gc.collect()
        torch.cuda.empty_cache()

    payload = {
        "event": "summary",
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "compile_modules": bool(args.compile_modules),
        "freeze_backbone": bool(args.freeze_backbone),
        "fuse_views": str(args.fuse_views),
        "tau": float(args.tau),
        "warmup": int(args.warmup),
        "iterations": int(args.iterations),
        "parameter_stats": stats,
        "results": results,
    }
    _print_event(payload)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
