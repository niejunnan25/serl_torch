#!/usr/bin/env python3
from __future__ import annotations

"""Benchmark image augmentation/layout strategies with fake observations.

The timed unit is one fake visual training step:

1. apply DrQ-style random crop to two RGB camera streams;
2. run a frozen ResNet-18 backbone plus trainable spatial pooling/head;
3. run backward through the trainable pooling/head.

The variants are cumulative so that each row answers "what if we add this
optimization on top of the previous lower-risk changes?"
"""

import argparse
import gc
import json
import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "serl_launcher") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "serl_launcher"))

import torch
import torch.nn.functional as F
from torch import nn

from serl_launcher.vision.data_augmentations import batched_random_crop
from serl_launcher.vision.resnet_v1 import ResNetEncoder
from serl_launcher.vision.spatial import SpatialLearnedEmbeddings


def _summary(values: Iterable[float]) -> dict[str, float]:
    values = [float(value) for value in values]
    if not values:
        return {
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "steps_per_sec": 0.0,
        }
    mean_s = statistics.mean(values)
    return {
        "mean_ms": float(mean_s * 1000.0),
        "median_ms": float(statistics.median(values) * 1000.0),
        "min_ms": float(min(values) * 1000.0),
        "max_ms": float(max(values) * 1000.0),
        "steps_per_sec": float(1.0 / mean_s) if mean_s > 0.0 else 0.0,
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


def _nchw_batched_random_crop(
    images: torch.Tensor,
    *,
    padding: int,
) -> torch.Tensor:
    """Random crop for [B, V, C, H, W] tensors, preserving NCHW layout."""

    if images.ndim != 5:
        raise ValueError(f"expected [B,V,C,H,W], got {tuple(images.shape)}")
    batch_size, num_views, channels, height, width = images.shape
    flat = images.reshape(batch_size * num_views, channels, height, width)
    if flat.shape[0] == 0:
        return images.clone()

    padding = int(padding)
    padded = F.pad(flat, (padding, padding, padding, padding), mode="replicate")
    n = int(flat.shape[0])
    y = torch.randint(0, 2 * padding + 1, (n,), device=images.device)
    x = torch.randint(0, 2 * padding + 1, (n,), device=images.device)
    batch_idx = torch.arange(n, device=images.device)[:, None, None, None]
    channel_idx = torch.arange(channels, device=images.device)[None, :, None, None]
    row_idx = y[:, None, None, None] + torch.arange(height, device=images.device)[
        None, None, :, None
    ]
    col_idx = x[:, None, None, None] + torch.arange(width, device=images.device)[
        None, None, None, :
    ]
    cropped = padded[batch_idx, channel_idx, row_idx, col_idx]
    return cropped.reshape(batch_size, num_views, channels, height, width)


class ViewHead(nn.Module):
    def __init__(self, *, bottleneck_dim: int, num_spatial_blocks: int = 8):
        super().__init__()
        self.num_spatial_blocks = int(num_spatial_blocks)
        self.spatial_pool: SpatialLearnedEmbeddings | None = None
        self.bottleneck = nn.LazyLinear(int(bottleneck_dim))

    def forward(self, features_bchw: torch.Tensor, *, train: bool) -> torch.Tensor:
        x_hwc = features_bchw.permute(0, 2, 3, 1).contiguous()
        if self.spatial_pool is None:
            height, width, channel = x_hwc.shape[-3:]
            self.spatial_pool = SpatialLearnedEmbeddings(
                height=height,
                width=width,
                channel=channel,
                num_features=self.num_spatial_blocks,
            ).to(x_hwc.device)
        x = self.spatial_pool(x_hwc)
        x = F.dropout(x, p=0.1, training=train)
        x = self.bottleneck(x)
        x = torch.layer_norm(x, x.shape[-1:])
        return torch.tanh(x)


class PerViewCurrentPath(nn.Module):
    """Current-style per-view backbone calls, with optional buffered norm."""

    def __init__(
        self,
        *,
        num_views: int,
        model_name: str,
        pretrained: bool,
        bottleneck_dim: int,
        buffered_norm: bool,
        expects_nchw: bool,
    ):
        super().__init__()
        backbone = ResNetEncoder.create_backbone(
            model_name=model_name,
            pretrained=bool(pretrained),
            freeze=True,
        )
        self.backbone = backbone
        self.expects_nchw = bool(expects_nchw)
        self.buffered_norm = bool(buffered_norm)
        self.heads = nn.ModuleList(
            [ViewHead(bottleneck_dim=int(bottleneck_dim)) for _ in range(int(num_views))]
        )
        if self.buffered_norm:
            self.register_buffer(
                "image_mean",
                torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(
                    1, 3, 1, 1
                ),
                persistent=False,
            )
            self.register_buffer(
                "image_std",
                torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(
                    1, 3, 1, 1
                ),
                persistent=False,
            )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.buffered_norm:
            mean = self.image_mean
            std = self.image_std
        else:
            mean = torch.tensor(
                (0.485, 0.456, 0.406),
                device=x.device,
                dtype=torch.float32,
            ).view(1, 3, 1, 1)
            std = torch.tensor(
                (0.229, 0.224, 0.225),
                device=x.device,
                dtype=torch.float32,
            ).view(1, 3, 1, 1)
        return (x.float() / 255.0 - mean) / std

    def forward(self, images: torch.Tensor, *, train: bool) -> torch.Tensor:
        outs = []
        for view_index, head in enumerate(self.heads):
            if self.expects_nchw:
                x = images[:, view_index]
            else:
                x = images[:, view_index].permute(0, 3, 1, 2).contiguous()
            x = self._normalize(x)
            with torch.no_grad():
                features = self.backbone(pixel_values=x, return_dict=True).last_hidden_state
            outs.append(head(features.detach(), train=train))
        return torch.cat(outs, dim=-1)


class FusedBackbonePath(nn.Module):
    """One shared backbone call over [B*V, C, H, W], separate heads per view."""

    def __init__(
        self,
        *,
        num_views: int,
        model_name: str,
        pretrained: bool,
        bottleneck_dim: int,
    ):
        super().__init__()
        self.num_views = int(num_views)
        self.backbone = ResNetEncoder.create_backbone(
            model_name=model_name,
            pretrained=bool(pretrained),
            freeze=True,
        )
        self.heads = nn.ModuleList(
            [ViewHead(bottleneck_dim=int(bottleneck_dim)) for _ in range(int(num_views))]
        )
        self.register_buffer(
            "image_mean",
            torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, images: torch.Tensor, *, train: bool) -> torch.Tensor:
        batch_size, num_views, channels, height, width = images.shape
        if int(num_views) != self.num_views:
            raise ValueError(f"expected {self.num_views} views, got {num_views}")
        flat = images.reshape(batch_size * num_views, channels, height, width)
        flat = (flat.float() / 255.0 - self.image_mean) / self.image_std
        with torch.no_grad():
            features = self.backbone(
                pixel_values=flat,
                return_dict=True,
            ).last_hidden_state.detach()
        _, feature_channels, feature_height, feature_width = features.shape
        features = features.reshape(
            batch_size,
            num_views,
            feature_channels,
            feature_height,
            feature_width,
        )
        outs = [
            head(features[:, view_index], train=train)
            for view_index, head in enumerate(self.heads)
        ]
        return torch.cat(outs, dim=-1)


def _make_images(args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    images_nhwc = torch.randint(
        0,
        256,
        (
            int(args.batch_size),
            int(args.num_views),
            int(args.image_size),
            int(args.image_size),
            3,
        ),
        device=device,
        dtype=torch.uint8,
    )
    images_nchw = images_nhwc.permute(0, 1, 4, 2, 3).contiguous()
    return images_nhwc, images_nchw


def _make_variant(
    variant: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]:
    pretrained = not bool(args.random_init)
    if variant == "baseline_current":
        model = PerViewCurrentPath(
            num_views=int(args.num_views),
            model_name=str(args.model_name),
            pretrained=pretrained,
            bottleneck_dim=int(args.bottleneck_dim),
            buffered_norm=False,
            expects_nchw=False,
        ).to(device)

        def prepare(images_nhwc: torch.Tensor, images_nchw: torch.Tensor) -> torch.Tensor:
            del images_nchw
            return batched_random_crop(
                images_nhwc,
                padding=int(args.padding),
                num_batch_dims=2,
            )

    elif variant == "layer1_buffered_norm":
        model = PerViewCurrentPath(
            num_views=int(args.num_views),
            model_name=str(args.model_name),
            pretrained=pretrained,
            bottleneck_dim=int(args.bottleneck_dim),
            buffered_norm=True,
            expects_nchw=False,
        ).to(device)

        def prepare(images_nhwc: torch.Tensor, images_nchw: torch.Tensor) -> torch.Tensor:
            del images_nchw
            return batched_random_crop(
                images_nhwc,
                padding=int(args.padding),
                num_batch_dims=2,
            )

    elif variant == "layer2_nchw_layout":
        model = PerViewCurrentPath(
            num_views=int(args.num_views),
            model_name=str(args.model_name),
            pretrained=pretrained,
            bottleneck_dim=int(args.bottleneck_dim),
            buffered_norm=True,
            expects_nchw=True,
        ).to(device)

        def prepare(images_nhwc: torch.Tensor, images_nchw: torch.Tensor) -> torch.Tensor:
            del images_nhwc
            return _nchw_batched_random_crop(
                images_nchw,
                padding=int(args.padding),
            )

    elif variant in {"layer3_fused_backbone", "layer4_compiled_fused"}:
        model = FusedBackbonePath(
            num_views=int(args.num_views),
            model_name=str(args.model_name),
            pretrained=pretrained,
            bottleneck_dim=int(args.bottleneck_dim),
        ).to(device)

        def prepare(images_nhwc: torch.Tensor, images_nchw: torch.Tensor) -> torch.Tensor:
            del images_nhwc
            return _nchw_batched_random_crop(
                images_nchw,
                padding=int(args.padding),
            )

    else:
        raise ValueError(f"unknown variant: {variant}")

    return model, prepare


def _materialize(
    model: nn.Module,
    prepare: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    images_nhwc: torch.Tensor,
    images_nchw: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    cropped = prepare(images_nhwc, images_nchw)
    with _autocast_context(device, bool(args.bf16)):
        out = model(cropped, train=True)
        loss = out.float().square().mean()
    loss.backward()
    model.zero_grad(set_to_none=True)
    _sync(device)


def _make_step_fn(
    model: nn.Module,
    prepare: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    images_nhwc: torch.Tensor,
    images_nchw: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> Callable[[], torch.Tensor]:
    def step() -> torch.Tensor:
        model.zero_grad(set_to_none=True)
        cropped = prepare(images_nhwc, images_nchw)
        with _autocast_context(device, bool(args.bf16)):
            out = model(cropped, train=True)
            loss = out.float().square().mean()
        loss.backward()
        return loss.detach()

    return step


def _time_step(
    step_fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> list[float]:
    for _ in range(int(warmup)):
        step_fn()
    _sync(device)
    times: list[float] = []
    for _ in range(int(iterations)):
        start = time.perf_counter()
        step_fn()
        _sync(device)
        times.append(time.perf_counter() - start)
    return times


def _run_variant(
    variant: str,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))
    images_nhwc, images_nchw = _make_images(args, device)
    model, prepare = _make_variant(variant, args, device)
    _materialize(model, prepare, images_nhwc, images_nchw, args, device)
    if variant == "layer4_compiled_fused":
        model = torch.compile(
            model,
            backend=str(args.compile_backend),
            mode=str(args.compile_mode),
            fullgraph=bool(args.compile_fullgraph),
            dynamic=False,
        )
        prepare = torch.compile(
            prepare,
            backend=str(args.compile_backend),
            mode=str(args.compile_mode),
            fullgraph=bool(args.compile_fullgraph),
            dynamic=False,
        )
    step_fn = _make_step_fn(model, prepare, images_nhwc, images_nchw, args, device)
    times = _time_step(
        step_fn,
        warmup=int(args.warmup),
        iterations=int(args.iterations),
        device=device,
    )
    return {
        "variant": variant,
        "ok": True,
        "summary": _summary(times),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[
            "baseline_current",
            "layer1_buffered_norm",
            "layer2_nchw_layout",
            "layer3_fused_backbone",
            "layer4_compiled_fused",
        ],
        choices=[
            "baseline_current",
            "layer1_buffered_norm",
            "layer2_nchw_layout",
            "layer3_fused_backbone",
            "layer4_compiled_fused",
        ],
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-views", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--padding", type=int, default=4)
    parser.add_argument("--bottleneck-dim", type=int, default=256)
    parser.add_argument("--model-name", default="pretrained_models/microsoft--resnet-18")
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
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
    for variant in args.variants:
        _print_event({"event": "start_variant", "variant": variant})
        try:
            result = _run_variant(variant, args, device)
        except Exception as exc:  # noqa: BLE001
            result = {
                "variant": variant,
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        results.append(result)
        _print_event({"event": "variant_result", **result})
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    payload = {
        "event": "summary",
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "batch_size": int(args.batch_size),
        "num_views": int(args.num_views),
        "image_size": int(args.image_size),
        "padding": int(args.padding),
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
