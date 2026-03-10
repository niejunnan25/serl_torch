import torch
import torch.nn as nn
from einops import rearrange


class BinaryClassifier(nn.Module):
    def __init__(
        self,
        pretrained_encoder: nn.Module,
        encoder: nn.Module,
        network: nn.Module,
        enable_stacking: bool = False,
    ):
        super().__init__()
        self.pretrained_encoder = pretrained_encoder
        self.encoder = encoder
        self.network = network
        self.enable_stacking = enable_stacking
        self.classifier = nn.LazyLinear(1)

    def forward(self, x, train: bool = False, return_encoded: bool = False, classify_encoded: bool = False):
        if return_encoded:
            if self.enable_stacking:
                if x.ndim == 4:
                    x = rearrange(x, "t h w c -> h w (t c)")
                elif x.ndim == 5:
                    x = rearrange(x, "b t h w c -> b h w (t c)")

            try:
                x = self.pretrained_encoder(x, train=train)
            except TypeError:
                x = self.pretrained_encoder(x)
            return x

        try:
            x = self.encoder(x, train=train, is_encoded=classify_encoded)
        except TypeError:
            x = self.encoder(x)

        x = self.network(x, train=train)
        x = self.classifier(x).squeeze(-1)
        return x
