
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

# ── Building blocks ──────────────────────────────────────────────────


class MLP(nn.Module):
    """Feedforward network with optional LayerNorm (Linear -> LN -> ReLU).

    LayerNorm in hidden layers is critical for off-policy/offline RL stability:
    it bounds the activation scale even on OOD inputs, preventing the critic
    from extrapolating Q to arbitrarily large values when the actor explores
    actions far from the data distribution.

    The output layer is plain Linear (no LN/activation) so that the network
    can still represent unbounded outputs (e.g. Q values, raw pre-tanh action
    logits). It is small-initialized so the network output is near zero at the
    start of training.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

        last_linear = self.net[-1]
        nn.init.normal_(last_linear.weight, std=0.01)
        nn.init.zeros_(last_linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)



class RLTokenEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        rl_token_dim: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.rl_token_dim = rl_token_dim

        self.e_rl = nn.Parameter(torch.randn(1, 1, input_dim) * 0.02)

        if input_dim != rl_token_dim:
            self.input_proj = nn.Linear(input_dim, rl_token_dim)
        else:
            self.input_proj = nn.Identity()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=rl_token_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, z_vla: Tensor) -> Tensor:
        batch_size = z_vla.shape[0]
        e_rl = self.e_rl.expand(batch_size, -1, -1)
        seq = torch.cat([z_vla, e_rl], dim=1)  # (B, M+1, D)
        seq = self.input_proj(seq)
        out = self.transformer(seq)
        z_rl = out[:, -1, :]  # output at e_rl position
        return z_rl


class RLTokenDecoder(nn.Module):

    def __init__(
        self,
        rl_token_dim: int,
        output_dim: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.output_dim = output_dim

        if rl_token_dim != output_dim:
            self.rl_proj = nn.Linear(rl_token_dim, output_dim)
        else:
            self.rl_proj = nn.Identity()

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=output_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_head = nn.Linear(output_dim, output_dim)

    def forward(self, z_rl: Tensor, z_vla_stopped: Tensor) -> Tensor:
        seq_len = z_vla_stopped.shape[1]
        z_rl_proj = self.rl_proj(z_rl).unsqueeze(1)

        target = torch.cat([z_rl_proj, z_vla_stopped[:, :-1, :]], dim=1)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=z_rl.device)

        decoded = self.transformer(
            tgt=target,
            memory=z_rl_proj,
            tgt_mask=causal_mask,
        )
        return self.output_head(decoded)




class RLTActor(nn.Module):

    def __init__(self, state_dim: int, action_chunk_dim: int, hidden_dims: list[int], std: float = 0.01):
        super().__init__()
        input_dim = state_dim + action_chunk_dim # 2048+70
        self.net = MLP(input_dim, hidden_dims, action_chunk_dim)
        self.log_std = math.log(std)

    def forward(self, state: Tensor, ref_action_chunk: Tensor) -> Tensor:
        x = torch.cat([state, ref_action_chunk], dim=-1)
        return self.net(x)

    def sample(self, state: Tensor, ref_action_chunk: Tensor) -> tuple[Tensor, Tensor]:
        mean = self.forward(state, ref_action_chunk)
        std = math.exp(self.log_std)
        noise = torch.randn_like(mean) * std
        action = mean + noise
        log_prob = -0.5 * (noise / std).pow(2).sum(dim=-1) - mean.shape[-1] * math.log(
            std * math.sqrt(2 * math.pi)
        )
        return action, log_prob
