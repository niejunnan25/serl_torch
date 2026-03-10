from typing import Callable, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _resolve_activation(act: Union[Callable, str]):
    if callable(act):
        return act
    mapping = {
        "relu": F.relu,
        "tanh": torch.tanh,
        "swish": F.silu,
        "silu": F.silu,
        "leaky_relu": F.leaky_relu,
        "gelu": F.gelu,
    }
    if act not in mapping:
        raise ValueError(f"Unsupported activation: {act}")
    return mapping[act]


class MLP(nn.Module):
    def __init__(
        self,
        hidden_dims: Sequence[int],
        activations: Union[Callable[[torch.Tensor], torch.Tensor], str] = "swish",
        activate_final: bool = False,
        use_layer_norm: bool = False,
        dropout_rate: Optional[float] = None,
    ):
        super().__init__()
        self.hidden_dims = tuple(hidden_dims)
        self.activate_final = activate_final
        self.use_layer_norm = use_layer_norm
        self.dropout_rate = dropout_rate
        self.activation = _resolve_activation(activations)

        layers = []
        norms = []
        dropouts = []

        for i, dim in enumerate(self.hidden_dims):
            if i == 0:
                layers.append(nn.LazyLinear(dim))
            else:
                layers.append(nn.Linear(self.hidden_dims[i - 1], dim))

            if self.use_layer_norm:
                norms.append(nn.LayerNorm(dim))
            else:
                norms.append(nn.Identity())

            if self.dropout_rate is not None and self.dropout_rate > 0:
                dropouts.append(nn.Dropout(p=self.dropout_rate))
            else:
                dropouts.append(nn.Identity())

        self.layers = nn.ModuleList(layers)
        self.norms = nn.ModuleList(norms)
        self.dropouts = nn.ModuleList(dropouts)

    def forward(self, x: torch.Tensor, train: bool = False) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            should_activate = (i + 1 < len(self.layers)) or self.activate_final
            if should_activate:
                x = self.dropouts[i](x)
                x = self.norms[i](x)
                x = self.activation(x)
        return x


class MLPResNetBlock(nn.Module):
    def __init__(
        self,
        features: int,
        act: Callable,
        dropout_rate: Optional[float] = None,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        self.features = features
        self.act = _resolve_activation(act)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate else nn.Identity()
        self.norm = nn.LayerNorm(features) if use_layer_norm else nn.Identity()
        self.fc1 = nn.LazyLinear(features * 4)
        self.fc2 = nn.Linear(features * 4, features)
        self.proj = nn.LazyLinear(features)

    def forward(self, x: torch.Tensor, train: bool = False) -> torch.Tensor:
        residual = x
        x = self.dropout(x)
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        if residual.shape[-1] != x.shape[-1]:
            residual = self.proj(residual)

        return residual + x


class MLPResNet(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        out_dim: int,
        dropout_rate: Optional[float] = None,
        use_layer_norm: bool = False,
        hidden_dim: int = 256,
        activations: Union[Callable, str] = "swish",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.out_dim = out_dim
        self.activation = _resolve_activation(activations)

        self.in_proj = nn.LazyLinear(hidden_dim)
        self.blocks = nn.ModuleList(
            [
                MLPResNetBlock(
                    hidden_dim,
                    act=self.activation,
                    use_layer_norm=use_layer_norm,
                    dropout_rate=dropout_rate,
                )
                for _ in range(num_blocks)
            ]
        )
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, train: bool = False) -> torch.Tensor:
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x, train=train)
        x = self.activation(x)
        x = self.out_proj(x)
        return x


class Scalar(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(float(init_value), dtype=torch.float32))

    def forward(self):
        return self.value
