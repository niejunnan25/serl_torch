from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
from torch import nn

from serl_launcher.agents.continuous.sac import SACAgent


class _FakeDist:
    def __init__(self, *, batch_size: int, action_dim: int):
        self.batch_size = int(batch_size)
        self.action_dim = int(action_dim)

    def sample(self):
        return torch.zeros((self.batch_size, self.action_dim), dtype=torch.float32)

    def sample_and_log_prob(self):
        return self.sample(), torch.zeros((self.batch_size,), dtype=torch.float32)


class _FakeActor(nn.Module):
    def __init__(self, *, action_dim: int):
        super().__init__()
        self.action_dim = int(action_dim)

    def forward(self, observations, train: bool = True):
        del train
        batch_size = int(observations["state"].shape[0])
        return _FakeDist(batch_size=batch_size, action_dim=self.action_dim)


class _FakeCritic(nn.Module):
    def forward(self, observations, actions, train: bool = True):
        del observations, train
        if actions.ndim == 3:
            batch_size, action_count = int(actions.shape[0]), int(actions.shape[1])
            return torch.zeros((2, batch_size, action_count), dtype=torch.float32)
        batch_size = int(actions.shape[0])
        return torch.ones((2, batch_size), dtype=torch.float32)


class SacCalqlTest(unittest.TestCase):
    def _make_agent(self) -> SACAgent:
        state = SimpleNamespace(
            device=torch.device("cpu"),
            modules={"actor": _FakeActor(action_dim=3), "critic": _FakeCritic()},
            target_modules={"critic": _FakeCritic()},
        )
        return SACAgent(
            state=state,
            config={
                "discount": 0.99,
                "backup_entropy": False,
                "critic_subsample_size": None,
                "otf_num_samples": 1,
                "cql_n_actions": 2,
                "cql_temperature": 1.0,
            },
        )

    def _batch(self) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        return {
            "observations": {"state": torch.zeros((4, 2), dtype=torch.float32)},
            "next_observations": {"state": torch.zeros((4, 2), dtype=torch.float32)},
            "actions": torch.zeros((4, 3), dtype=torch.float32),
            "rewards": torch.zeros((4,), dtype=torch.float32),
            "masks": torch.ones((4,), dtype=torch.float32),
        }

    def test_calql_mc_return_bound_is_optional(self) -> None:
        agent = self._make_agent()
        _loss, info = agent.critic_loss_fn(
            self._batch(),
            calql_alpha=1.0,
            calql_n_actions=2,
            calql_temperature=1.0,
        )

        self.assertEqual(float(info["calql_bound_applied"]), 0.0)

    def test_calql_mc_return_bound_tracks_valid_fraction(self) -> None:
        agent = self._make_agent()
        batch = self._batch()
        batch["mc_returns"] = torch.asarray([3.0, 3.0, 3.0, 3.0])
        batch["mc_returns_valid"] = torch.asarray([True, False, True, False])
        _loss, info = agent.critic_loss_fn(
            batch,
            calql_alpha=1.0,
            calql_n_actions=2,
            calql_temperature=1.0,
        )

        self.assertAlmostEqual(float(info["calql_bound_applied"]), 0.5)


if __name__ == "__main__":
    unittest.main()
