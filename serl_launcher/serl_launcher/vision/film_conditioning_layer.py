import torch
import torch.nn as nn


class FilmConditioning(nn.Module):
    def __init__(self):
        super().__init__()
        self.add_proj = None
        self.mul_proj = None

    def _ensure_proj(self, conv_filters: torch.Tensor, conditioning: torch.Tensor):
        channels = conv_filters.shape[-1]
        if self.add_proj is None:
            self.add_proj = nn.Linear(conditioning.shape[-1], channels, device=conditioning.device)
            self.mul_proj = nn.Linear(conditioning.shape[-1], channels, device=conditioning.device)
            nn.init.zeros_(self.add_proj.weight)
            nn.init.zeros_(self.add_proj.bias)
            nn.init.zeros_(self.mul_proj.weight)
            nn.init.zeros_(self.mul_proj.bias)

    def forward(self, conv_filters: torch.Tensor, conditioning: torch.Tensor):
        self._ensure_proj(conv_filters, conditioning)
        projected_cond_add = self.add_proj(conditioning)[..., None, None, :]
        projected_cond_mult = self.mul_proj(conditioning)[..., None, None, :]
        return conv_filters * (1.0 + projected_cond_add) + projected_cond_mult
