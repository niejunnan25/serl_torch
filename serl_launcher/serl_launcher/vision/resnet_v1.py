"""HuggingFace ResNet encoder for SERL.

All configuration is passed explicitly via constructor arguments
(typically from YAML config).  No module-level mutable state.
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from serl_launcher.vision.spatial import SpatialLearnedEmbeddings

VARIANT_CONFIGS = {
    "microsoft/resnet-18": dict(
        num_channels=3, embedding_size=64,
        hidden_sizes=[64, 128, 256, 512], depths=[2, 2, 2, 2],
        layer_type="basic",
    ),
    "microsoft/resnet-34": dict(
        num_channels=3, embedding_size=64,
        hidden_sizes=[64, 128, 256, 512], depths=[3, 4, 6, 3],
        layer_type="basic",
    ),
    "microsoft/resnet-50": dict(
        num_channels=3, embedding_size=64,
        hidden_sizes=[256, 512, 1024, 2048], depths=[3, 4, 6, 3],
        layer_type="bottleneck",
    ),
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_model_name(model_name: str) -> str:
    if model_name.startswith(("http://", "https://")):
        return model_name

    candidates = []
    model_path = Path(model_name)
    if model_path.is_absolute():
        candidates.append(model_path)
    else:
        candidates.append(model_path)
        candidates.append(_project_root() / model_path)
        candidates.append(
            _project_root() / "pretrained_models" / model_name.replace("/", "--")
        )

    for candidate in candidates:
        if candidate.is_dir():
            resolved = str(candidate.resolve())
            if resolved != model_name:
                print(
                    "[ResNetEncoder] Resolved local mirror: "
                    f"{model_name} -> {resolved}"
                )
            return resolved
    return model_name


def _to_bchw(x: torch.Tensor):
    squeeze = False
    if x.ndim == 3:
        x = x.unsqueeze(0)
        squeeze = True
    if x.shape[-1] <= 4:
        x = x.permute(0, 3, 1, 2).contiguous()
    return x, squeeze


class SpatialSoftmax(nn.Module):
    def __init__(self, height: int, width: int, channel: int, temperature: float = 1.0):
        super().__init__()
        self.height = height
        self.width = width
        self.channel = channel
        self.temperature = temperature

        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height),
            torch.linspace(-1.0, 1.0, width),
            indexing="ij",
        )
        self.register_buffer("pos_x", pos_x.reshape(-1))
        self.register_buffer("pos_y", pos_y.reshape(-1))

    def forward(self, features: torch.Tensor):
        squeeze = False
        if features.ndim == 3:
            features = features.unsqueeze(0)
            squeeze = True

        b, c, h, w = features.shape
        flat = features.reshape(b, c, h * w)
        attn = torch.softmax(flat / self.temperature, dim=-1)
        expected_x = torch.sum(attn * self.pos_x, dim=-1)
        expected_y = torch.sum(attn * self.pos_y, dim=-1)
        out = torch.cat([expected_x, expected_y], dim=-1)

        if squeeze:
            out = out.squeeze(0)
        return out


# ---------------------------------------------------------------------------
# Core encoder
# ---------------------------------------------------------------------------

class ResNetEncoder(nn.Module):
    """HuggingFace ResNet encoder with configurable pooling.

    Parameters
    ----------
    backbone : nn.Module
        A ``transformers.ResNetModel`` instance (created via
        :meth:`create_backbone`).  Multiple ``ResNetEncoder`` instances can
        **share** the same backbone to avoid duplicating weights.
    freeze_backbone : bool
        If ``True``, forward the backbone under ``torch.no_grad()`` and
        ``.detach()`` the output.
    pooling_method : str
        ``"avg"`` | ``"max"`` | ``"spatial_learned_embeddings"`` |
        ``"spatial_softmax"`` | ``"none"``.
    num_spatial_blocks : int
        Only used when *pooling_method* is ``"spatial_learned_embeddings"``.
    bottleneck_dim : int | None
        If given, append ``Linear -> LayerNorm -> Tanh`` bottleneck.
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        backbone: nn.Module,
        freeze_backbone: bool = False,
        pooling_method: str = "avg",
        num_spatial_blocks: int = 8,
        bottleneck_dim: Optional[int] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        self.pooling_method = pooling_method
        self.num_spatial_blocks = num_spatial_blocks

        self.spatial_pool = None
        self.spatial_softmax = None
        self.bottleneck = (
            nn.LazyLinear(bottleneck_dim) if bottleneck_dim is not None else None
        )
        self.register_buffer(
            "_image_mean",
            torch.tensor(self.IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_image_std",
            torch.tensor(self.IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    @staticmethod
    def create_backbone(
        model_name: str = "microsoft/resnet-18",
        pretrained: bool = True,
        freeze: bool = False,
    ) -> nn.Module:
        """Create a HuggingFace ``ResNetModel`` backbone.

        Call once, then pass the result to one or more ``ResNetEncoder``
        instances to share weights across image keys.
        """
        from transformers import ResNetConfig, ResNetModel

        resolved_model_name = _resolve_model_name(model_name)

        if pretrained:
            backbone = ResNetModel.from_pretrained(resolved_model_name)
            print(f"[ResNetEncoder] Loaded pretrained: {resolved_model_name}")
        else:
            if resolved_model_name in VARIANT_CONFIGS:
                config = ResNetConfig(**VARIANT_CONFIGS[resolved_model_name])
            else:
                config = ResNetConfig.from_pretrained(resolved_model_name)
            backbone = ResNetModel(config)
            print(f"[ResNetEncoder] Random init: {resolved_model_name}")

        if freeze:
            backbone.requires_grad_(False)
            backbone.eval()

        return backbone

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def observations_to_bchw(self, observations: torch.Tensor):
        return _to_bchw(observations)

    def normalize_bchw(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() / 255.0 - self._image_mean) / self._image_std

    def encode_backbone_bchw(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalize_bchw(x)
        if self.freeze_backbone:
            with torch.no_grad():
                out = self.backbone(pixel_values=x, return_dict=True)
            return out.last_hidden_state.detach()

        out = self.backbone(pixel_values=x, return_dict=True)
        return out.last_hidden_state

    def pool_features(
        self,
        features: torch.Tensor,
        *,
        train: bool = True,
    ) -> torch.Tensor:
        x = self._pool(features, train=train)

        if self.bottleneck is not None:
            x = self.bottleneck(x)
            x = torch.layer_norm(x, x.shape[-1:])
            x = torch.tanh(x)

        return x

    def _pool(self, x: torch.Tensor, train: bool):
        if self.pooling_method == "spatial_learned_embeddings":
            x_hwc = x.permute(0, 2, 3, 1).contiguous()
            if self.spatial_pool is None:
                h, w, c = x_hwc.shape[-3:]
                self.spatial_pool = SpatialLearnedEmbeddings(
                    height=h,
                    width=w,
                    channel=c,
                    num_features=self.num_spatial_blocks,
                ).to(x_hwc.device)
            x = self.spatial_pool(x_hwc)
            x = F.dropout(x, p=0.1, training=train)
            return x

        if self.pooling_method == "spatial_softmax":
            if self.spatial_softmax is None:
                _, c, h, w = x.shape
                self.spatial_softmax = SpatialSoftmax(
                    height=h,
                    width=w,
                    channel=c,
                ).to(x.device)
            return self.spatial_softmax(x)

        if self.pooling_method == "avg":
            return torch.mean(x, dim=(-2, -1))
        if self.pooling_method == "max":
            return torch.amax(x, dim=(-2, -1))
        if self.pooling_method == "none":
            return x
        raise ValueError(f"Unknown pooling method: {self.pooling_method}")

    def forward(
        self,
        observations: torch.Tensor,
        train: bool = True,
        **kwargs,
    ):
        x, squeeze = _to_bchw(observations)
        x = self.encode_backbone_bchw(x)
        x = self.pool_features(x, train=train)

        if squeeze and x.ndim > 1:
            x = x.squeeze(0)
        return x
