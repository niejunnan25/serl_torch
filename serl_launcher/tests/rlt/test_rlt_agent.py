from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[3]
SERL_LAUNCHER_ROOT = REPO_ROOT / "serl_launcher"
for _path in (SERL_LAUNCHER_ROOT, REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from examples.libero_rlt.config import parse_train_cfg
from examples.libero_rlt.vla_inference import extract_rlt_features, load_frozen_rlt_encoder
from serl_launcher.agents.rlt.agent import RLTAgent
from serl_launcher.agents.rlt.modeling import RLTokenDecoder, RLTokenEncoder
from serl_launcher.async_eval.artifacts import (
    append_async_eval_checkpoint_index,
    prune_async_eval_checkpoints,
)
from serl_launcher.async_eval.queue import load_completed_async_eval_indices
from serl_launcher.policy.vla_features.client import VLAFeatureClient


def _fake_batch(
    *,
    batch_size: int = 8,
    z_dim: int = 16,
    proprio_dim: int = 3,
    action_dim: int = 2,
    chunk_size: int = 4,
    done: float = 0.0,
) -> dict:
    action_chunk_dim = int(action_dim * chunk_size)
    return {
        "observations": {
            "z_rl": np.random.randn(batch_size, z_dim).astype(np.float32),
            "proprio": np.random.randn(batch_size, proprio_dim).astype(np.float32),
            "reference_action": np.random.randn(batch_size, action_chunk_dim).astype(np.float32),
        },
        "next_observations": {
            "z_rl": np.random.randn(batch_size, z_dim).astype(np.float32),
            "proprio": np.random.randn(batch_size, proprio_dim).astype(np.float32),
            "reference_action": np.random.randn(batch_size, action_chunk_dim).astype(np.float32),
        },
        "actions": np.random.randn(batch_size, action_chunk_dim).astype(np.float32),
        "rewards": np.random.randn(batch_size).astype(np.float32),
        "dones": np.full(batch_size, done, dtype=np.float32),
    }


def _minimal_cfg(**rlt_overrides):
    cfg = {
        "global_seed": 42,
        "task": {"suite_name": "libero_10", "task_id": 6},
        "rlt": {
            "pi0_checkpoint_path": "/tmp/pi0",
            "rlt_encoder_path": "/tmp/rlt.pt",
            **rlt_overrides,
        },
    }
    return OmegaConf.create(cfg)


def test_rl_token_encoder_decoder_shapes() -> None:
    encoder = RLTokenEncoder(
        input_dim=16,
        rl_token_dim=16,
        num_layers=1,
        num_heads=4,
        ff_dim=32,
    )
    decoder = RLTokenDecoder(
        rl_token_dim=16,
        output_dim=16,
        num_layers=1,
        num_heads=4,
        ff_dim=32,
    )

    z_vla = torch.randn(2, 5, 16)
    z_rl = encoder(z_vla)
    z_recon = decoder(z_rl, z_vla)

    assert z_rl.shape == (2, 16)
    assert z_recon.shape == (2, 5, 16)


def test_rlt_agent_sample_and_update_high_utd_cpu() -> None:
    agent = RLTAgent(
        z_rl_dim=16,
        proprio_dim=3,
        action_dim=2,
        chunk_size=4,
        execute_horizon=4,
        actor_hidden_dims=(32, 32),
        critic_hidden_dims=(32, 32),
        device="cpu",
    )

    obs = {
        "z_rl": np.zeros(16, dtype=np.float32),
        "proprio": np.zeros(3, dtype=np.float32),
        "reference_action": np.zeros(8, dtype=np.float32),
    }
    action = agent.sample_action(obs)
    assert action.shape == (8,)

    batch = _fake_batch()

    agent, info = agent.update_high_utd(batch, utd_ratio=2)

    assert set(info) >= {
        "loss_critic",
        "target_q_mean",
        "predicted_q_mean",
        "loss_actor",
        "bc_loss",
        "q_value_mean",
    }


def test_rlt_config_parses_execute_horizon() -> None:
    parsed = parse_train_cfg(_minimal_cfg(chunk_size=10, execute_horizon=5))

    assert parsed.rlt.chunk_size == 10
    assert parsed.rlt.execute_horizon == 5


def test_rlt_config_rejects_execute_horizon_larger_than_chunk_size() -> None:
    try:
        parse_train_cfg(_minimal_cfg(chunk_size=4, execute_horizon=5))
    except ValueError as exc:
        assert "execute_horizon" in str(exc)
    else:
        raise AssertionError("expected execute_horizon validation failure")


def test_rlt_agent_uses_batch_discounts() -> None:
    agent = RLTAgent(
        z_rl_dim=16,
        proprio_dim=3,
        action_dim=2,
        chunk_size=4,
        execute_horizon=2,
        actor_hidden_dims=(32, 32),
        critic_hidden_dims=(32, 32),
        discount=0.5,
        device="cpu",
    )
    batch = _fake_batch()
    batch["discounts"] = np.full(8, 0.125, dtype=np.float32)

    fb = agent._convert_batch(batch)

    np.testing.assert_allclose(fb["discount"].cpu().numpy(), np.full((8, 1), 0.125))


def test_rlt_agent_discount_fallback_uses_execute_horizon() -> None:
    agent = RLTAgent(
        z_rl_dim=16,
        proprio_dim=3,
        action_dim=2,
        chunk_size=4,
        execute_horizon=2,
        actor_hidden_dims=(32, 32),
        critic_hidden_dims=(32, 32),
        discount=0.5,
        device="cpu",
    )
    batch = _fake_batch()

    fb = agent._convert_batch(batch)

    np.testing.assert_allclose(fb["discount"].cpu().numpy(), np.full((8, 1), 0.25))


def test_rlt_terminal_transition_does_not_bootstrap() -> None:
    agent = RLTAgent(
        z_rl_dim=16,
        proprio_dim=3,
        action_dim=2,
        chunk_size=4,
        execute_horizon=2,
        actor_hidden_dims=(32, 32),
        critic_hidden_dims=(32, 32),
        discount=0.5,
        device="cpu",
    )
    batch = _fake_batch(done=1.0)
    batch["rewards"] = np.ones(8, dtype=np.float32)
    batch["discounts"] = np.ones(8, dtype=np.float32)
    fb = agent._convert_batch(batch)

    _, target_q_mean, _ = agent._critic_step(fb)

    assert abs(target_q_mean - 1.0) < 1e-6


def test_vla_feature_client_normalizes_result_arrays() -> None:
    class FakeClient:
        def infer(self, raw_obs):
            return {
                "z_rl": [1.0, 2.0],
                "reference_action": [0.1, 0.2, 0.3, 0.4],
                "proprio": [3.0],
            }

    class TestClient(VLAFeatureClient):
        def _make_client(self):
            return FakeClient()

    client = TestClient("127.0.0.1", 1)
    result = client.infer({"dummy": True})

    assert result["z_rl"].dtype == np.float32
    assert result["z_rl"].shape == (2,)
    assert result["reference_action"].shape == (4,)
    assert result["proprio"].shape == (1,)


def test_extract_rlt_features_applies_encoder_max_tokens() -> None:
    class FakeBasePolicy:
        def infer_features(self, raw_obs):
            return SimpleNamespace(
                z_vla=torch.randn(1, 5, 4),
                obs_torch={"state": torch.zeros(1, 8)},
                reference_actions=torch.zeros(1, 10, 7),
            )

        def unnormalize_actions(self, obs_torch, reference_actions):
            return np.zeros((10, 7), dtype=np.float32)

    class RecorderEncoder(torch.nn.Module):
        max_tokens = 3

        def __init__(self):
            super().__init__()
            self.seen_shape = None

        def forward(self, z_vla):
            self.seen_shape = tuple(z_vla.shape)
            return torch.zeros(z_vla.shape[0], 6, device=z_vla.device)

    encoder = RecorderEncoder()

    features = extract_rlt_features(
        FakeBasePolicy(),
        encoder,
        {},
        device="cpu",
        chunk_size=2,
        action_dim=2,
        proprio_dim=3,
    )

    assert encoder.seen_shape == (1, 3, 4)
    assert features["z_rl"].shape == (6,)
    assert features["reference_action"].shape == (4,)
    assert features["proprio"].shape == (3,)


def test_load_frozen_rlt_encoder_restores_max_tokens(tmp_path) -> None:
    encoder = RLTokenEncoder(
        input_dim=4,
        rl_token_dim=4,
        num_layers=1,
        num_heads=2,
        ff_dim=8,
    )
    checkpoint_path = tmp_path / "rlt_encoder.pt"
    torch.save(
        {
            "encoder_state_dict": encoder.state_dict(),
            "config": {
                "rlt": {
                    "input_dim": 4,
                    "rl_token_dim": 4,
                    "num_encoder_layers": 1,
                    "num_heads": 2,
                    "ff_dim": 8,
                    "max_tokens": 3,
                }
            },
        },
        checkpoint_path,
    )

    loaded = load_frozen_rlt_encoder(str(checkpoint_path), device="cpu")

    assert loaded.max_tokens == 3


def test_load_frozen_rlt_encoder_max_tokens_override_wins(tmp_path) -> None:
    encoder = RLTokenEncoder(
        input_dim=4,
        rl_token_dim=4,
        num_layers=1,
        num_heads=2,
        ff_dim=8,
    )
    checkpoint_path = tmp_path / "rlt_encoder.pt"
    torch.save(
        {
            "encoder_state_dict": encoder.state_dict(),
            "config": {
                "rlt": {
                    "input_dim": 4,
                    "rl_token_dim": 4,
                    "num_encoder_layers": 1,
                    "num_heads": 2,
                    "ff_dim": 8,
                    "max_tokens": 3,
                }
            },
        },
        checkpoint_path,
    )

    loaded = load_frozen_rlt_encoder(str(checkpoint_path), device="cpu", max_tokens=2)

    assert loaded.max_tokens == 2


def test_load_frozen_rlt_encoder_zero_max_tokens_override_disables_truncation(tmp_path) -> None:
    encoder = RLTokenEncoder(
        input_dim=4,
        rl_token_dim=4,
        num_layers=1,
        num_heads=2,
        ff_dim=8,
    )
    checkpoint_path = tmp_path / "rlt_encoder.pt"
    torch.save(
        {
            "encoder_state_dict": encoder.state_dict(),
            "config": {
                "rlt": {
                    "input_dim": 4,
                    "rl_token_dim": 4,
                    "num_encoder_layers": 1,
                    "num_heads": 2,
                    "ff_dim": 8,
                    "max_tokens": 3,
                }
            },
        },
        checkpoint_path,
    )

    loaded = load_frozen_rlt_encoder(str(checkpoint_path), device="cpu", max_tokens=0)

    assert loaded.max_tokens is None


def test_load_frozen_rlt_encoder_legacy_checkpoint_accepts_max_tokens_override(tmp_path) -> None:
    encoder = RLTokenEncoder(
        input_dim=4,
        rl_token_dim=4,
        num_layers=1,
        num_heads=2,
        ff_dim=8,
    )
    checkpoint_path = tmp_path / "legacy_rlt_encoder.pt"
    torch.save(encoder.state_dict(), checkpoint_path)

    loaded = load_frozen_rlt_encoder(
        str(checkpoint_path),
        device="cpu",
        num_heads=2,
        ff_dim=8,
        max_tokens=3,
    )

    assert loaded.max_tokens == 3


def test_prune_async_eval_checkpoints_preserves_protected_pending(tmp_path) -> None:
    checkpoint_dir = tmp_path / "eval_checkpoints"
    checkpoint_dir.mkdir()
    for episode_id, checkpoint_step in ((1, 10), (2, 20), (3, 30)):
        checkpoint_path = checkpoint_dir / f"episode_{episode_id:06d}.pt"
        checkpoint_path.write_bytes(b"checkpoint")
        append_async_eval_checkpoint_index(
            checkpoint_dir,
            episode_id=episode_id,
            checkpoint_step=checkpoint_step,
            checkpoint_path=checkpoint_path,
        )

    protected_path = checkpoint_dir / "episode_000001.pt"
    latest_path = checkpoint_dir / "episode_000003.pt"
    pruned_path = checkpoint_dir / "episode_000002.pt"

    prune_async_eval_checkpoints(
        checkpoint_dir,
        keep=1,
        protected_paths=[protected_path],
    )

    assert protected_path.exists()
    assert latest_path.exists()
    assert not pruned_path.exists()

    index_path = checkpoint_dir / "async_eval_checkpoint_index.jsonl"
    records = [json.loads(line) for line in index_path.read_text().splitlines()]
    assert {record["train_episode_id"] for record in records} == {1, 3}


def test_load_completed_async_eval_indices_supports_nested_request(tmp_path) -> None:
    summary_path = tmp_path / "eval_summary.jsonl"
    summary_path.write_text(
        json.dumps({"status": "ok", "request": {"eval_index": 7}})
        + "\n"
        + json.dumps({"status": "ok", "eval_index": 9})
        + "\n"
    )

    assert load_completed_async_eval_indices(summary_path) == {7, 9}
