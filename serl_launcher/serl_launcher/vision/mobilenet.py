from typing import Optional

import torch
import torch.nn as nn

from serl_launcher.vision.spatial import SpatialLearnedEmbeddings


class MobileNetEncoder(nn.Module):
    """Wrapper for an ImageNet-style MobileNet feature encoder."""

    def __init__(
        self,
        encoder: nn.Module,
        params=None,
        pool_method: str = "spatial_learned_embeddings",
        bottleneck_dim: Optional[int] = None,
        spatial_block_size: Optional[int] = 8,
    ):
        super().__init__()
        del params
        self.encoder = encoder
        self.pool_method = pool_method
        self.bottleneck_dim = bottleneck_dim
        self.spatial_block_size = spatial_block_size
        self.spatial_pool = None
        self.bottleneck = nn.LazyLinear(bottleneck_dim) if bottleneck_dim is not None else None

    def forward(self, x: torch.Tensor, train: bool = False):
        del train
        mean = torch.tensor((0.485, 0.456, 0.406), device=x.device).view(1, 1, 1, 3)
        std = torch.tensor((0.229, 0.224, 0.225), device=x.device).view(1, 1, 1, 3)
        x = x.float() / 255.0
        x = (x - mean) / std

        reshape = False
        if x.ndim == 3:
            x = x.unsqueeze(0)
            reshape = True

        if x.shape[-1] <= 4:
            x = x.permute(0, 3, 1, 2).contiguous()

        x = self.encoder(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]

        x = x.detach()

        if self.pool_method == "max":
            x = torch.amax(x, dim=(-2, -1))
        elif self.pool_method == "avg":
            x = torch.mean(x, dim=(-2, -1))
        elif self.pool_method == "spatial_learned_embeddings":
            x_hwc = x.permute(0, 2, 3, 1).contiguous()
            if self.spatial_pool is None:
                h, w, c = x_hwc.shape[-3:]
                self.spatial_pool = SpatialLearnedEmbeddings(
                    h,
                    w,
                    c,
                    self.spatial_block_size,
                ).to(x_hwc.device)
            x = self.spatial_pool(x_hwc)
        else:
            raise ValueError(f"Unsupported pool method: {self.pool_method}")

        if self.bottleneck is not None:
            x = self.bottleneck(x)
            x = torch.layer_norm(x, x.shape[-1:])
            x = torch.tanh(x)

        if reshape:
            x = x.reshape(-1)
        return x
