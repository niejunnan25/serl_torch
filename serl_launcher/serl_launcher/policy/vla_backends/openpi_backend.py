"""OpenPI VLA backend adapter used by RLT.

This module keeps OpenPI as an external dependency. Callers provide an
``openpi_root`` checkout and a checkpoint path; serl_torch owns only the RL
algorithm and infra around the frozen base policy.
"""

from __future__ import annotations

import dataclasses
import datetime as _datetime
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

if not hasattr(_datetime, "UTC"):
    _datetime.UTC = _datetime.timezone.utc


def add_openpi_to_path(openpi_root: str | Path | None) -> None:
    """Add an external OpenPI checkout to ``sys.path`` if provided."""
    if openpi_root is None:
        return
    root = Path(openpi_root).expanduser().resolve()
    candidates = (root / "src", root)
    for candidate in reversed(candidates):
        candidate_str = str(candidate)
        if candidate.exists() and candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


def _tree_map(fn, tree):
    if isinstance(tree, dict):
        return {key: _tree_map(fn, value) for key, value in tree.items()}
    if isinstance(tree, tuple):
        return tuple(_tree_map(fn, value) for value in tree)
    if isinstance(tree, list):
        return [_tree_map(fn, value) for value in tree]
    return fn(tree)


def _numpy_to_batched_torch(tree: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return _tree_map(
        lambda x: torch.from_numpy(np.array(x, copy=True)).to(device)[None, ...],
        tree,
    )


def _to_pytorch_image_layout(image: torch.Tensor) -> torch.Tensor:
    # OpenPI Observation.from_dict already converts torch.uint8 NHWC images to
    # float32 NCHW. Non-uint8 tensors from fake/torch datasets need the same
    # layout normalization here.
    if image.dim() == 4 and image.shape[-1] == 3 and image.dtype != torch.uint8:
        return image.permute(0, 3, 1, 2).contiguous()
    return image


def _normalize_image_layouts(obs_torch: dict[str, Any]) -> dict[str, Any]:
    images = obs_torch.get("image")
    if isinstance(images, dict):
        obs_torch = dict(obs_torch)
        obs_torch["image"] = {key: _to_pytorch_image_layout(value) for key, value in images.items()}
    return obs_torch


@dataclass(frozen=True)
class OpenPIFeatureBatch:
    """Frozen VLA outputs needed by RLT Stage 2."""

    z_vla: torch.Tensor
    reference_actions: torch.Tensor
    obs_torch: dict[str, Any]


class OpenPIBasePolicy:
    """Thin wrapper around OpenPI's trained policy and PyTorch model."""

    def __init__(self, *, policy: Any, observation_cls: Any, device: str | torch.device):
        self.policy = policy
        self.model = policy._model
        self.observation_cls = observation_cls
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def raw_obs_to_torch(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        processed_obs = self.policy._input_transform(raw_obs)
        return _normalize_image_layouts(_numpy_to_batched_torch(processed_obs, self.device))

    def to_observation(self, obs_torch: dict[str, Any]) -> Any:
        return self.observation_cls.from_dict(obs_torch)

    @torch.no_grad()
    def sample_reference_actions(self, obs_obj: Any, *, num_steps: int = 10) -> torch.Tensor:
        return self.model.sample_actions(
            device=self.device,
            observation=obs_obj,
            noise=None,
            num_steps=num_steps,
        )

    @torch.no_grad()
    def extract_embeddings(self, obs_obj: Any, *, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not hasattr(self.model, "extract_embeddings"):
            raise RuntimeError(
                "The selected OpenPI model does not expose extract_embeddings(). "
                "Use an OpenPI checkout/fork with the RLT embedding extractor."
            )
        prefix_embeddings, suffix_embeddings = self.model.extract_embeddings(obs_obj, actions=actions)
        return prefix_embeddings, suffix_embeddings

    @torch.no_grad()
    def infer_features(self, raw_obs: dict[str, Any], *, num_steps: int = 10) -> OpenPIFeatureBatch:
        obs_torch = self.raw_obs_to_torch(raw_obs)
        obs_obj = self.to_observation(obs_torch)
        ref_actions = self.sample_reference_actions(obs_obj, num_steps=num_steps)
        prefix_embeddings, suffix_embeddings = self.extract_embeddings(obs_obj, actions=ref_actions)
        if prefix_embeddings.shape[-1] == suffix_embeddings.shape[-1]:
            z_vla = torch.cat([prefix_embeddings, suffix_embeddings], dim=1)
        else:
            z_vla = prefix_embeddings
        return OpenPIFeatureBatch(
            z_vla=z_vla.to(torch.float32),
            reference_actions=ref_actions,
            obs_torch=obs_torch,
        )

    def unnormalize_actions(self, obs_torch: dict[str, Any], actions: torch.Tensor) -> np.ndarray:
        out_dict = {
            "state": obs_torch["state"].detach().cpu()[0],
            "actions": actions.detach().cpu()[0],
        }
        out_dict_unnorm = self.policy._output_transform(out_dict)
        return np.asarray(out_dict_unnorm["actions"], dtype=np.float32)


class OpenPIBackend:
    """Factory for OpenPI policy, model embeddings, and training dataloader."""

    def __init__(self, *, openpi_root: str | Path | None = None):
        add_openpi_to_path(openpi_root)
        from openpi.models import model as model_module
        from openpi.policies import policy_config as policy_config_module
        from openpi.training import config as train_config_module
        from openpi.training import data_loader as data_loader_module

        self._model_module = model_module
        self._policy_config = policy_config_module
        self._train_config = train_config_module
        self._data_loader = data_loader_module

    def get_train_config(self, config_name: str) -> Any:
        return self._train_config.get_config(config_name)

    def create_train_config(
        self,
        *,
        config_name: str,
        batch_size: int | None = None,
        num_workers: int | None = None,
        assets_base_dir: str | None = None,
        checkpoint_base_dir: str | None = None,
        exp_name: str | None = None,
        repo_id_override: str | None = None,
    ) -> Any:
        train_cfg = self.get_train_config(config_name)
        replace_kwargs: dict[str, Any] = {}
        for key, value in {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "assets_base_dir": assets_base_dir,
            "checkpoint_base_dir": checkpoint_base_dir,
            "exp_name": exp_name,
        }.items():
            if value is not None:
                replace_kwargs[key] = value
        if repo_id_override:
            if repo_id_override == "fake":
                replace_kwargs["data"] = self._train_config.FakeDataConfig(repo_id="fake")
            else:
                try:
                    replace_kwargs["data"] = dataclasses.replace(train_cfg.data, repo_id=repo_id_override)
                except TypeError as exc:
                    raise ValueError(
                        "The selected OpenPI data config does not support repo_id_override; "
                        f"got {type(train_cfg.data).__name__}"
                    ) from exc
        if not replace_kwargs:
            return train_cfg
        return dataclasses.replace(train_cfg, **replace_kwargs)

    def load_base_policy(
        self,
        *,
        config_name: str,
        checkpoint_path: str,
        device: str | torch.device = "cuda",
        assets_base_dir: str | None = None,
        checkpoint_base_dir: str | None = None,
        exp_name: str | None = None,
    ) -> OpenPIBasePolicy:
        logger.info("Loading OpenPI VLA config=%s checkpoint=%s", config_name, checkpoint_path)
        train_cfg = self.create_train_config(
            config_name=config_name,
            assets_base_dir=assets_base_dir,
            checkpoint_base_dir=checkpoint_base_dir,
            exp_name=exp_name,
        )
        try:
            policy = self._policy_config.create_trained_policy(
                train_cfg,
                checkpoint_path,
                pytorch_device=str(device),
            )
        except TypeError:
            policy = self._policy_config.create_trained_policy(train_cfg, checkpoint_path)
        base_policy = OpenPIBasePolicy(
            policy=policy,
            observation_cls=self._model_module.Observation,
            device=device,
        )
        logger.info("Frozen OpenPI VLA loaded on %s", base_policy.device)
        return base_policy

    def create_dataloader(
        self,
        *,
        config_name: str,
        batch_size: int,
        num_workers: int,
        assets_base_dir: str | None = None,
        checkpoint_base_dir: str | None = None,
        exp_name: str | None = None,
        repo_id_override: str | None = None,
        shuffle: bool = True,
    ) -> Any:
        train_cfg = self.create_train_config(
            config_name=config_name,
            batch_size=batch_size,
            num_workers=num_workers,
            assets_base_dir=assets_base_dir,
            checkpoint_base_dir=checkpoint_base_dir,
            exp_name=exp_name,
            repo_id_override=repo_id_override,
        )
        dataloader = self._data_loader.create_data_loader(train_cfg, framework="pytorch", shuffle=shuffle)
        data_cfg = dataloader.data_config()
        if getattr(data_cfg, "repo_id", None) != "fake" and getattr(data_cfg, "norm_stats", None) is None:
            raise ValueError("OpenPI dataloader did not provide normalization stats.")
        return dataloader

    def observation_to_device_dict(self, observation: Any, device: str | torch.device) -> dict[str, Any]:
        torch_device = torch.device(device)
        obs_dict: dict[str, Any] = {
            "image": {key: _to_pytorch_image_layout(value.to(torch_device)) for key, value in observation.images.items()},
            "image_mask": {key: value.to(torch_device) for key, value in observation.image_masks.items()},
            "state": observation.state.to(torch_device),
        }
        for field_name in (
            "tokenized_prompt",
            "tokenized_prompt_mask",
            "token_ar_mask",
            "token_loss_mask",
        ):
            value = getattr(observation, field_name, None)
            if value is not None:
                obs_dict[field_name] = value.to(torch_device)
        return obs_dict
