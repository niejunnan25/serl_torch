from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from serl_launcher.networks.actor_critic_nets import Critic
from serl_launcher.networks.actor_critic_nets import CriticEnsemble


class _CountingEncoder(nn.Module):
    def __init__(self, obs_dim: int = 3, feature_dim: int = 5):
        super().__init__()
        self.calls = 0
        self.proj = nn.Linear(obs_dim, feature_dim)

    def forward(self, observations: torch.Tensor, train: bool = False):
        del train
        self.calls += 1
        return self.proj(observations)


class _TrainAwareNetwork(nn.Module):
    def __init__(self, input_dim: int = 7, hidden_dim: int = 4):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)

    def forward(self, inputs: torch.Tensor, train: bool = False):
        del train
        return torch.tanh(self.proj(inputs))


class CriticEnsembleTest(unittest.TestCase):
    def test_shared_encoder_is_called_once_for_2d_actions(self) -> None:
        encoder = _CountingEncoder()
        ensemble = CriticEnsemble(
            critic_ctor=lambda: Critic(
                encoder=encoder,
                network=_TrainAwareNetwork(),
            ),
            num_qs=2,
        )

        qs = ensemble(torch.randn(4, 3), torch.randn(4, 2), train=True)

        self.assertEqual(tuple(qs.shape), (2, 4))
        self.assertEqual(encoder.calls, 1)

    def test_shared_encoder_is_called_once_for_3d_actions(self) -> None:
        encoder = _CountingEncoder()
        ensemble = CriticEnsemble(
            critic_ctor=lambda: Critic(
                encoder=encoder,
                network=_TrainAwareNetwork(),
            ),
            num_qs=2,
        )

        qs = ensemble(torch.randn(4, 3), torch.randn(4, 6, 2), train=True)

        self.assertEqual(tuple(qs.shape), (2, 4, 6))
        self.assertEqual(encoder.calls, 1)

    def test_distinct_encoders_keep_original_per_critic_forward(self) -> None:
        encoders: list[_CountingEncoder] = []

        def _critic_ctor() -> Critic:
            encoder = _CountingEncoder()
            encoders.append(encoder)
            return Critic(encoder=encoder, network=_TrainAwareNetwork())

        ensemble = CriticEnsemble(critic_ctor=_critic_ctor, num_qs=2)

        qs = ensemble(torch.randn(4, 3), torch.randn(4, 2), train=True)

        self.assertEqual(tuple(qs.shape), (2, 4))
        self.assertEqual([encoder.calls for encoder in encoders], [1, 1])


if __name__ == "__main__":
    unittest.main()
