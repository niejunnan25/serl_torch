import torch
import torch.nn as nn


def _to_nhwc(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
        return x
    if x.shape[-1] <= 8 and x.shape[1] > x.shape[-1]:
        return x.permute(0, 2, 3, 1)
    return x


class SpatialLearnedEmbeddings(nn.Module):
    def __init__(
        self,
        height: int,
        width: int,
        channel: int,
        num_features: int = 5,
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.channel = channel
        self.num_features = num_features
        self.kernel = nn.Parameter(
            torch.empty(height, width, channel, num_features)
        )
        nn.init.kaiming_normal_(self.kernel)

    def forward(self, features: torch.Tensor):
        squeeze = False
        if features.ndim == 3:
            features = features.unsqueeze(0)
            squeeze = True

        features = _to_nhwc(features)
        batch_size = features.shape[0]
        projected = torch.sum(
            features.unsqueeze(-1) * self.kernel.unsqueeze(0),
            dim=(1, 2),
        )
        projected = projected.reshape(batch_size, -1)
        if squeeze:
            projected = projected.squeeze(0)
        return projected
