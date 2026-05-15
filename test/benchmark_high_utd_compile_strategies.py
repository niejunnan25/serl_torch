#!/usr/bin/env python3
from __future__ import annotations

"""Benchmark high-UTD critic update dispatch strategies with fake visual data.

The benchmark is intentionally isolated from the production learner.  It uses a
fake SAC critic workload with two 224x224 RGB camera streams, a frozen
ResNet-18 backbone, trainable pooling/head layers, target critic bootstrap, and
an AdamW critic update.  One timed unit is one full high-UTD critic pattern,
that is ``utd_ratio`` critic updates over minibatches split from the fake batch.
"""

import argparse
import copy
import gc
import json
import statistics
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "serl_launcher") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "serl_launcher"))

import torch
from torch import nn

from serl_launcher.vision.resnet_v1 import ResNetEncoder


TensorBatch = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def _summary(values: Iterable[float]) -> dict[str, float]:
    values = [float(value) for value in values]
    if not values:
        return {
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "updates_per_sec": 0.0,
        }
    mean_s = statistics.mean(values)
    return {
        "mean_ms": float(mean_s * 1000.0),
        "median_ms": float(statistics.median(values) * 1000.0),
        "min_ms": float(min(values) * 1000.0),
        "max_ms": float(max(values) * 1000.0),
        "updates_per_sec": float(1.0 / mean_s) if mean_s > 0.0 else 0.0,
    }


def _print_event(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True), flush=True)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], output_dim: int):
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, int(hidden_dim)))
            layers.append(nn.LayerNorm(int(hidden_dim)))
            layers.append(nn.Tanh())
            last_dim = int(hidden_dim)
        layers.append(nn.Linear(last_dim, int(output_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TwoViewResNet18Encoder(nn.Module):
    def __init__(
        self,
        *,
        num_views: int,
        model_name: str,
        pretrained: bool,
        freeze_backbone: bool,
        bottleneck_dim: int,
    ):
        super().__init__()
        backbone = ResNetEncoder.create_backbone(
            model_name=model_name,
            pretrained=bool(pretrained),
            freeze=bool(freeze_backbone),
        )
        self.encoders = nn.ModuleList(
            [
                ResNetEncoder(
                    backbone=backbone,
                    freeze_backbone=bool(freeze_backbone),
                    pooling_method="spatial_learned_embeddings",
                    num_spatial_blocks=8,
                    bottleneck_dim=int(bottleneck_dim),
                )
                for _ in range(int(num_views))
            ]
        )

    def forward(self, images: torch.Tensor, *, train: bool) -> torch.Tensor:
        return torch.cat(
            [
                encoder(images[:, view_index], train=train)
                for view_index, encoder in enumerate(self.encoders)
            ],
            dim=-1,
        )


class FakeActor(nn.Module):
    def __init__(self, encoder: nn.Module, obs_dim: int, action_dim: int):
        super().__init__()
        self.encoder = encoder
        self.head = MLP(obs_dim, (256, 256, 256), int(action_dim))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        obs = self.encoder(images, train=True)
        return torch.tanh(self.head(obs))


class FakeCriticEnsemble(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        obs_dim: int,
        action_dim: int,
        *,
        num_qs: int,
    ):
        super().__init__()
        self.encoder = encoder
        self.q_heads = nn.ModuleList(
            [MLP(obs_dim + int(action_dim), (256, 256, 256), 1) for _ in range(num_qs)]
        )

    def forward(
        self,
        images: torch.Tensor,
        actions: torch.Tensor,
        *,
        train: bool,
    ) -> torch.Tensor:
        obs = self.encoder(images, train=train)
        if actions.ndim == 3:
            batch_size, num_actions, action_dim = actions.shape
            flat_actions = actions.reshape(batch_size * num_actions, action_dim)
            flat_obs = (
                obs.unsqueeze(1)
                .expand(-1, num_actions, -1)
                .reshape(batch_size * num_actions, -1)
            )
            qs = [
                head(torch.cat([flat_obs, flat_actions], dim=-1))
                .squeeze(-1)
                .reshape(batch_size, num_actions)
                for head in self.q_heads
            ]
        else:
            qs = [
                head(torch.cat([obs, actions], dim=-1)).squeeze(-1)
                for head in self.q_heads
            ]
        return torch.stack(qs, dim=0)


@dataclass
class FakeWorld:
    actor: nn.Module
    critic: nn.Module
    target_critic: nn.Module
    optimizer: torch.optim.Optimizer


def _make_fake_batch(args: argparse.Namespace, device: torch.device) -> TensorBatch:
    batch_size = int(args.batch_size)
    num_views = int(args.num_views)
    image_size = int(args.image_size)
    action_dim = int(args.action_dim)
    obs = torch.randint(
        0,
        256,
        (batch_size, num_views, image_size, image_size, 3),
        device=device,
        dtype=torch.uint8,
    )
    next_obs = torch.randint(
        0,
        256,
        (batch_size, num_views, image_size, image_size, 3),
        device=device,
        dtype=torch.uint8,
    )
    actions = torch.randn(batch_size, action_dim, device=device)
    rewards = torch.randn(batch_size, device=device)
    masks = torch.randint(0, 2, (batch_size,), device=device, dtype=torch.int32).float()
    return obs, next_obs, actions, rewards, masks


def _split_batch(batch: TensorBatch, utd_ratio: int) -> list[TensorBatch]:
    batch_size = int(batch[0].shape[0])
    if batch_size % int(utd_ratio) != 0:
        raise ValueError(
            f"batch_size={batch_size} must be divisible by utd_ratio={utd_ratio}"
        )
    mini = batch_size // int(utd_ratio)
    return [
        tuple(tensor[i * mini : (i + 1) * mini] for tensor in batch)  # type: ignore[misc]
        for i in range(int(utd_ratio))
    ]


def _make_world(
    args: argparse.Namespace,
    device: torch.device,
    *,
    optimizer_capturable: bool = False,
) -> FakeWorld:
    obs_dim = int(args.num_views) * int(args.bottleneck_dim)
    actor_encoder = TwoViewResNet18Encoder(
        num_views=int(args.num_views),
        model_name=str(args.model_name),
        pretrained=not bool(args.random_init),
        freeze_backbone=True,
        bottleneck_dim=int(args.bottleneck_dim),
    )
    critic_encoder = TwoViewResNet18Encoder(
        num_views=int(args.num_views),
        model_name=str(args.model_name),
        pretrained=not bool(args.random_init),
        freeze_backbone=True,
        bottleneck_dim=int(args.bottleneck_dim),
    )
    actor = FakeActor(actor_encoder, obs_dim, int(args.action_dim)).to(device)
    critic = FakeCriticEnsemble(
        critic_encoder,
        obs_dim,
        int(args.action_dim),
        num_qs=int(args.num_qs),
    ).to(device)
    target_critic = copy.deepcopy(critic).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in critic.parameters() if p.requires_grad],
        lr=float(args.learning_rate),
        capturable=bool(optimizer_capturable),
        foreach=False if optimizer_capturable else None,
    )
    return FakeWorld(
        actor=actor,
        critic=critic,
        target_critic=target_critic,
        optimizer=optimizer,
    )


def _materialize(
    world: FakeWorld,
    batch: TensorBatch,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    minibatch = _split_batch(batch, int(args.utd_ratio))[0]
    obs, next_obs, actions, rewards, masks = minibatch
    del rewards, masks
    with torch.no_grad():
        _ = world.actor(next_obs)
        _ = world.target_critic(next_obs, actions, train=False)
    with _autocast_context(device, bool(args.bf16)):
        q = world.critic(obs, actions, train=True)
        loss = q.float().square().mean()
    loss.backward()
    world.optimizer.zero_grad(set_to_none=True)
    _sync(device)


@torch.no_grad()
def _soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    for src, dst in zip(source.parameters(), target.parameters()):
        dst.lerp_(src, float(tau))


def _make_loss_fn(
    world: FakeWorld,
    args: argparse.Namespace,
    device: torch.device,
) -> Callable[..., torch.Tensor]:
    discount = float(args.discount)

    def loss_fn(
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        with _autocast_context(device, bool(args.bf16)):
            with torch.no_grad():
                next_actions = world.actor(next_obs)
                next_qs = world.target_critic(next_obs, next_actions, train=False)
                target_next_q = next_qs.min(dim=0).values
                target_q = rewards + discount * masks * target_next_q
            predicted_qs = world.critic(obs, actions, train=True)
            target_qs = target_q.unsqueeze(0).expand_as(predicted_qs)
            return ((predicted_qs.float() - target_qs.float()) ** 2).mean()

    return loss_fn


def _make_update_fn(
    strategy: str,
    world: FakeWorld,
    batch: TensorBatch,
    args: argparse.Namespace,
    device: torch.device,
) -> Callable[[], torch.Tensor | None]:
    minibatches = _split_batch(batch, int(args.utd_ratio))

    if strategy == "current_module_compile":
        compile_kwargs = {
            "backend": str(args.compile_backend),
            "mode": str(args.compile_mode),
            "fullgraph": bool(args.compile_fullgraph),
            "dynamic": False,
        }
        world.actor = torch.compile(world.actor, **compile_kwargs)
        world.critic = torch.compile(world.critic, **compile_kwargs)
        world.target_critic = torch.compile(world.target_critic, **compile_kwargs)

    loss_fn = _make_loss_fn(world, args, device)
    if strategy == "stage1_loss_compile":
        loss_fn = torch.compile(
            loss_fn,
            backend=str(args.compile_backend),
            mode=str(args.compile_mode),
            fullgraph=bool(args.compile_fullgraph),
            dynamic=False,
        )

    def train_minibatch(minibatch: TensorBatch, *, set_to_none: bool = True):
        world.optimizer.zero_grad(set_to_none=set_to_none)
        loss = loss_fn(*minibatch)
        loss.backward()
        world.optimizer.step()
        _soft_update(world.critic, world.target_critic, float(args.tau))
        return loss.detach()

    if strategy == "stage2_step_compile":

        def raw_train_minibatch(
            obs: torch.Tensor,
            next_obs: torch.Tensor,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            masks: torch.Tensor,
        ) -> torch.Tensor:
            world.optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(obs, next_obs, actions, rewards, masks)
            loss.backward()
            world.optimizer.step()
            _soft_update(world.critic, world.target_critic, float(args.tau))
            return loss.detach()

        compiled_train_minibatch = torch.compile(
            raw_train_minibatch,
            backend=str(args.compile_backend),
            mode=str(args.compile_mode),
            fullgraph=False,
            dynamic=False,
        )

        def update_once() -> torch.Tensor:
            total = None
            for minibatch in minibatches:
                loss = compiled_train_minibatch(*minibatch)
                total = loss if total is None else total + loss
            return total if total is not None else torch.zeros((), device=device)

        return update_once

    def update_once() -> torch.Tensor:
        total = None
        for minibatch in minibatches:
            loss = train_minibatch(minibatch)
            total = loss if total is None else total + loss
        return total if total is not None else torch.zeros((), device=device)

    if strategy != "stage3_cuda_graph":
        return update_once

    if device.type != "cuda":
        raise RuntimeError("stage3_cuda_graph requires a CUDA device")

    # Warm up with non-None gradients, then capture the static high-UTD loop.
    for _ in range(max(1, int(args.cuda_graph_warmup))):
        for minibatch in minibatches:
            train_minibatch(minibatch, set_to_none=False)
    _sync(device)

    graph = torch.cuda.CUDAGraph()
    graph_loss = torch.zeros((), device=device)
    with torch.cuda.graph(graph):
        graph_loss = update_once()

    def replay_once() -> torch.Tensor:
        graph.replay()
        return graph_loss

    return replay_once


def _time_update(
    update_fn: Callable[[], torch.Tensor | None],
    *,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> list[float]:
    for _ in range(int(warmup)):
        update_fn()
    _sync(device)
    times: list[float] = []
    for _ in range(int(iterations)):
        start = time.perf_counter()
        update_fn()
        _sync(device)
        times.append(time.perf_counter() - start)
    return times


def _run_strategy(
    strategy: str,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    batch = _make_fake_batch(args, device)
    world = _make_world(
        args,
        device,
        optimizer_capturable=(strategy == "stage3_cuda_graph"),
    )
    _materialize(world, batch, args, device)
    update_fn = _make_update_fn(strategy, world, batch, args, device)
    times = _time_update(
        update_fn,
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
            "eager_python",
            "current_module_compile",
            "stage1_loss_compile",
            "stage2_step_compile",
            "stage3_cuda_graph",
        ],
        choices=[
            "eager_python",
            "current_module_compile",
            "stage1_loss_compile",
            "stage2_step_compile",
            "stage3_cuda_graph",
        ],
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--utd-ratio", type=int, default=4)
    parser.add_argument("--num-views", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--action-dim", type=int, default=35)
    parser.add_argument("--num-qs", type=int, default=2)
    parser.add_argument("--bottleneck-dim", type=int, default=256)
    parser.add_argument("--model-name", default="pretrained_models/microsoft--resnet-18")
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--cuda-graph-warmup", type=int, default=3)
    parser.add_argument("--compile-backend", default="inductor")
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--compile-fullgraph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    results: list[dict[str, Any]] = []
    for strategy in args.strategies:
        _print_event({"event": "start_strategy", "strategy": strategy})
        try:
            result = _run_strategy(strategy, args, device)
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
        if device.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception as exc:  # noqa: BLE001
                _print_event(
                    {
                        "event": "cleanup_warning",
                        "strategy": strategy,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )

    payload = {
        "event": "summary",
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "batch_size": int(args.batch_size),
        "utd_ratio": int(args.utd_ratio),
        "num_views": int(args.num_views),
        "image_size": int(args.image_size),
        "action_dim": int(args.action_dim),
        "bf16": bool(args.bf16),
        "random_init": bool(args.random_init),
        "results": results,
    }
    _print_event(payload)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
