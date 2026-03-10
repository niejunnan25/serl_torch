from typing import Optional, Sequence, Union

import torch
import torch.nn as nn

from serl_launcher.vision.spatial import SpatialLearnedEmbeddings


def _padding_value(padding, kernel_size: int):
    if isinstance(padding, str):
        if padding.upper() == "VALID":
            return 0
        if padding.upper() == "SAME":
            return kernel_size // 2
    return int(padding)


def _to_bchw(x: torch.Tensor):
    squeeze = False
    if x.ndim == 3:
        x = x.unsqueeze(0)
        squeeze = True
    if x.shape[-1] <= 4:
        x = x.permute(0, 3, 1, 2).contiguous()
    return x, squeeze


class SmallEncoder(nn.Module):
    def __init__(
        self,
        features: Sequence[int] = (16, 16, 16),
        kernel_sizes: Sequence[int] = (3, 3, 3),
        strides: Sequence[int] = (1, 1, 1),
        padding: Union[Sequence[int], str] = (1, 1, 1),
        pool_method: str = "spatial_learned_embeddings",
        bottleneck_dim: Optional[int] = None,
        spatial_block_size: Optional[int] = 8,
        name: Optional[str] = None,
    ):
        super().__init__()
        del name
        self.features = tuple(features)
        self.kernel_sizes = tuple(kernel_sizes)
        self.strides = tuple(strides)
        self.padding = padding
        self.pool_method = pool_method
        self.bottleneck_dim = bottleneck_dim
        self.spatial_block_size = spatial_block_size

        convs = []
        for i, out_channels in enumerate(self.features):
            k = self.kernel_sizes[i]
            s = self.strides[i]
            p = _padding_value(self.padding if isinstance(self.padding, str) else self.padding[i], k)
            if i == 0:
                convs.append(nn.LazyConv2d(out_channels, kernel_size=k, stride=s, padding=p))
            else:
                convs.append(nn.Conv2d(self.features[i - 1], out_channels, kernel_size=k, stride=s, padding=p))
        self.convs = nn.ModuleList(convs)

        self.spatial_pool = None
        self.bottleneck = nn.LazyLinear(bottleneck_dim) if bottleneck_dim is not None else None

    def forward(self, observations: torch.Tensor, train: bool = False, encode: bool = True):
        del train, encode
        x = observations.float() / 255.0
        x, squeeze = _to_bchw(x)

        for conv in self.convs:
            x = torch.relu(conv(x))

        if self.pool_method == "max":
            x = torch.amax(x, dim=(-2, -1))
        elif self.pool_method == "avg":
            x = torch.mean(x, dim=(-2, -1))
        elif self.pool_method == "spatial_learned_embeddings":
            if self.spatial_block_size is None:
                raise ValueError("spatial_block_size must be set")
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

        if squeeze:
            x = x.squeeze(0)
        return x


small_configs = {"small": SmallEncoder}
