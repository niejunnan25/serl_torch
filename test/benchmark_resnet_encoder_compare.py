#!/usr/bin/env python3
from __future__ import annotations

"""Compare SERL ResNet10 and serl_torch ResNet18 encoder cost.

The benchmark intentionally isolates the visual encoder path:

* two 224x224 RGB camera streams;
* spatial learned embeddings plus a 256-dim bottleneck;
* forward-only timing;
* frozen-backbone forward plus backward through trainable pooling/head;
* optional full-backbone backward timing.

Run the two backends in separate processes so the ``serl_launcher`` package name
can resolve to either the JAX SERL repo or this PyTorch repo.
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
SERL_ROOT = Path("/vla/users/niejunnan/codebase/serl")


def _summary(values: Iterable[float]) -> dict[str, float]:
    values = list(float(v) for v in values)
    if not values:
        return {"mean_ms": 0.0, "median_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
    return {
        "mean_ms": float(statistics.mean(values) * 1000.0),
        "median_ms": float(statistics.median(values) * 1000.0),
        "min_ms": float(min(values) * 1000.0),
        "max_ms": float(max(values) * 1000.0),
    }


def _print_event(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True), flush=True)


def _prepend_paths(*paths: Path) -> None:
    for path in reversed(paths):
        path_str = str(path)
        if path_str in sys.path:
            sys.path.remove(path_str)
        sys.path.insert(0, path_str)


def _bench_torch_resnet18(args: argparse.Namespace) -> dict[str, Any]:
    _prepend_paths(REPO_ROOT, REPO_ROOT / "serl_launcher")

    import torch
    import torch.nn as nn

    from serl_launcher.vision.resnet_v1 import ResNetEncoder

    if not torch.cuda.is_available():
        raise RuntimeError("torch_resnet18 benchmark requires CUDA")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda")
    batch_size = int(args.batch_size)
    image_size = int(args.image_size)
    num_views = int(args.num_views)

    class TwoViewTorchEncoder(nn.Module):
        def __init__(self, *, freeze_backbone: bool):
            super().__init__()
            backbone = ResNetEncoder.create_backbone(
                model_name=str(args.torch_model_name),
                pretrained=not bool(args.random_init),
                freeze=bool(freeze_backbone),
            )
            self.encoders = nn.ModuleList(
                [
                    ResNetEncoder(
                        backbone=backbone,
                        freeze_backbone=bool(freeze_backbone),
                        pooling_method="spatial_learned_embeddings",
                        num_spatial_blocks=8,
                        bottleneck_dim=256,
                    )
                    for _ in range(num_views)
                ]
            )

        def forward(self, images: torch.Tensor, train: bool = True) -> torch.Tensor:
            outs = [
                encoder(images[:, view_index], train=train)
                for view_index, encoder in enumerate(self.encoders)
            ]
            return torch.cat(outs, dim=-1)

    images = torch.randint(
        0,
        256,
        (batch_size, num_views, image_size, image_size, 3),
        device=device,
        dtype=torch.uint8,
    )

    def sync() -> None:
        torch.cuda.synchronize()

    def time_call(fn) -> list[float]:
        for _ in range(int(args.warmup)):
            fn()
        sync()
        times: list[float] = []
        for _ in range(int(args.iterations)):
            start = time.perf_counter()
            fn()
            sync()
            times.append(time.perf_counter() - start)
        return times

    def make_model(*, freeze_backbone: bool) -> nn.Module:
        model = TwoViewTorchEncoder(freeze_backbone=freeze_backbone).to(device)
        model.train()
        # Materialize LazyLinear/spatial parameters before optional compile.
        with torch.enable_grad():
            out = model(images, train=True)
            loss = out.float().square().mean()
            if not freeze_backbone:
                loss.backward()
            else:
                loss.backward()
        model.zero_grad(set_to_none=True)
        sync()
        if bool(args.torch_compile):
            model = torch.compile(
                model,
                backend=str(args.torch_compile_backend),
                mode=str(args.torch_compile_mode),
                fullgraph=bool(args.torch_compile_fullgraph),
                dynamic=False,
            )
            # Pay compile cost outside measurement.
            out = model(images, train=True)
            out.float().square().mean().backward()
            sync()
            try:
                model.zero_grad(set_to_none=True)
            except AttributeError:
                pass
        return model

    frozen_model = make_model(freeze_backbone=True)

    def frozen_forward() -> None:
        with torch.no_grad():
            out = frozen_model(images, train=True)
            _ = out.float().sum()

    def frozen_backward() -> None:
        frozen_model.zero_grad(set_to_none=True)
        out = frozen_model(images, train=True)
        loss = out.float().square().mean()
        loss.backward()

    results: dict[str, Any] = {
        "backend": "torch_resnet18",
        "framework": f"torch {torch.__version__}",
        "device": torch.cuda.get_device_name(device),
        "batch_size": batch_size,
        "num_views": num_views,
        "image_size": image_size,
        "model_name": str(args.torch_model_name),
        "random_init": bool(args.random_init),
        "torch_compile": bool(args.torch_compile),
        "cases": {
            "frozen_forward": _summary(time_call(frozen_forward)),
            "frozen_forward_backward_head": _summary(time_call(frozen_backward)),
        },
    }

    if bool(args.include_full_backward):
        full_model = make_model(freeze_backbone=False)

        def full_backward() -> None:
            full_model.zero_grad(set_to_none=True)
            out = full_model(images, train=True)
            out.float().square().mean().backward()

        results["cases"]["full_forward_backward"] = _summary(time_call(full_backward))

    return results


def _bench_jax_resnet10(args: argparse.Namespace) -> dict[str, Any]:
    # Ensure old SERL wins the shared package name.
    _prepend_paths(SERL_ROOT, SERL_ROOT / "serl_launcher")

    import jax
    import jax.numpy as jnp
    from flax import linen as nn

    from serl_launcher.vision.resnet_v1 import (
        PreTrainedResNetEncoder,
        resnetv1_configs,
    )

    batch_size = int(args.batch_size)
    image_size = int(args.image_size)
    num_views = int(args.num_views)

    class TwoViewFrozenResNet10(nn.Module):
        @nn.compact
        def __call__(self, images, train: bool = True):
            outs = []
            for view_index in range(num_views):
                pretrained_encoder = resnetv1_configs["resnetv1-10-frozen"](
                    pre_pooling=True,
                    name=f"pretrained_encoder_{view_index}",
                )
                encoder = PreTrainedResNetEncoder(
                    pooling_method="spatial_learned_embeddings",
                    num_spatial_blocks=8,
                    bottleneck_dim=256,
                    pretrained_encoder=pretrained_encoder,
                    name=f"encoder_{view_index}",
                )
                outs.append(encoder(images[:, view_index], train=train))
            return jnp.concatenate(outs, axis=-1)

    class TwoViewFullResNet10(nn.Module):
        @nn.compact
        def __call__(self, images, train: bool = True):
            outs = []
            for view_index in range(num_views):
                encoder = resnetv1_configs["resnetv1-10"](
                    pooling_method="spatial_learned_embeddings",
                    num_spatial_blocks=8,
                    bottleneck_dim=256,
                    pre_pooling=False,
                    name=f"encoder_{view_index}",
                )
                outs.append(encoder(images[:, view_index], train=train))
            return jnp.concatenate(outs, axis=-1)

    key = jax.random.PRNGKey(int(args.seed))
    images = jax.random.randint(
        key,
        (batch_size, num_views, image_size, image_size, 3),
        minval=0,
        maxval=256,
        dtype=jnp.uint8,
    )

    def block(value) -> None:
        jax.block_until_ready(value)

    def time_call(fn) -> list[float]:
        for _ in range(int(args.warmup)):
            block(fn())
        times: list[float] = []
        for _ in range(int(args.iterations)):
            start = time.perf_counter()
            block(fn())
            times.append(time.perf_counter() - start)
        return times

    def make_fns(model):
        variables = model.init(
            {"params": key, "dropout": jax.random.fold_in(key, 1)},
            images,
            train=True,
        )

        @jax.jit
        def forward(params, dropout_key, x):
            return model.apply(
                {"params": params},
                x,
                train=True,
                rngs={"dropout": dropout_key},
            )

        def loss_fn(params, dropout_key, x):
            out = model.apply(
                {"params": params},
                x,
                train=True,
                rngs={"dropout": dropout_key},
            )
            return jnp.mean(jnp.square(out.astype(jnp.float32)))

        grad_fn = jax.jit(jax.value_and_grad(loss_fn))
        params = variables["params"]
        dropout_key = jax.random.fold_in(key, 2)
        # Compile outside measurement.
        block(forward(params, dropout_key, images))
        block(grad_fn(params, dropout_key, images))
        return params, dropout_key, forward, grad_fn

    frozen_model = TwoViewFrozenResNet10()
    frozen_params, frozen_dropout, frozen_forward_fn, frozen_grad_fn = make_fns(
        frozen_model
    )

    results: dict[str, Any] = {
        "backend": "jax_resnet10",
        "framework": f"jax {jax.__version__}",
        "device": str(jax.devices()[0]),
        "batch_size": batch_size,
        "num_views": num_views,
        "image_size": image_size,
        "model_name": "serl resnetv1-10 pretrained/frozen shape",
        "random_init": True,
        "cases": {
            "frozen_forward": _summary(
                time_call(lambda: frozen_forward_fn(frozen_params, frozen_dropout, images))
            ),
            "frozen_forward_backward_head": _summary(
                time_call(lambda: frozen_grad_fn(frozen_params, frozen_dropout, images))
            ),
        },
    }

    if bool(args.include_full_backward):
        full_model = TwoViewFullResNet10()
        full_params, full_dropout, _full_forward_fn, full_grad_fn = make_fns(full_model)
        results["cases"]["full_forward_backward"] = _summary(
            time_call(lambda: full_grad_fn(full_params, full_dropout, images))
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=("torch_resnet18", "jax_resnet10"),
        required=True,
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-views", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-full-backward", action="store_true")
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument(
        "--torch-model-name",
        default="pretrained_models/microsoft--resnet-18",
    )
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--torch-compile-backend", default="inductor")
    parser.add_argument("--torch-compile-mode", default="default")
    parser.add_argument("--torch-compile-fullgraph", action="store_true")
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    if args.backend == "torch_resnet18":
        result = _bench_torch_resnet18(args)
    else:
        result = _bench_jax_resnet10(args)

    _print_event({"event": "summary", **result})
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
