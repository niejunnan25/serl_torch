from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


def _flatten_history_image(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 4:
        return rearrange(image, "t h w c -> h w (t c)")
    if image.ndim == 5:
        return rearrange(image, "b t h w c -> b h w (t c)")
    return image


def _flatten_vector_features(vector: torch.Tensor) -> torch.Tensor:
    if vector.ndim <= 1:
        return vector.reshape(-1)
    return vector.reshape(vector.shape[0], -1)


def _maybe_call_encoder(module: nn.Module, image: torch.Tensor, train: bool, encode: bool):
    try:
        return module(image, train=train, encode=encode)
    except TypeError:
        try:
            return module(image, train=train)
        except TypeError:
            return module(image)


class EncodingWrapper(nn.Module):
    """PyTorch version of SERL observation encoder wrapper."""

    def __init__(
        self,
        encoder,
        use_proprio: bool,
        proprio_latent_dim: int = 64,
        enable_stacking: bool = False,
        image_keys: Iterable[str] = ("image",),
        vector_obs_keys: Optional[Iterable[str]] = None,
    ):
        super().__init__()
        self.use_proprio = use_proprio
        self.proprio_latent_dim = proprio_latent_dim
        self.enable_stacking = enable_stacking
        self.image_keys = tuple(image_keys)
        self.vector_obs_keys = (
            tuple(str(key) for key in vector_obs_keys)
            if vector_obs_keys is not None
            else None
        )

        if isinstance(encoder, dict):
            self.encoder = nn.ModuleDict(encoder)
        elif isinstance(encoder, nn.Module):
            self.encoder = nn.ModuleDict({self.image_keys[0]: encoder})
        else:
            raise TypeError(f"Unsupported encoder type: {type(encoder)}")

        self.proprio_proj = nn.LazyLinear(self.proprio_latent_dim)

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        train: bool = False,
        stop_gradient: bool = False,
        is_encoded: bool = False,
    ) -> torch.Tensor:
        encoded = []

        for image_key in self.image_keys:
            image = observations[image_key]
            if not is_encoded and self.enable_stacking:
                image = _flatten_history_image(image)

            image_feature = _maybe_call_encoder(
                self.encoder[image_key],
                image,
                train=train,
                encode=not is_encoded,
            )

            if stop_gradient:
                image_feature = image_feature.detach()

            encoded.append(image_feature)

        encoding = torch.cat(encoded, dim=-1)

        vector_inputs = None
        if self.vector_obs_keys is not None and len(self.vector_obs_keys) > 0:
            vector_parts = []
            for key in self.vector_obs_keys:
                if key not in observations:
                    raise KeyError(
                        f"Missing vector observation key {key!r}. "
                        f"Available keys: {list(observations.keys())}"
                    )
                vector_parts.append(_flatten_vector_features(observations[key]))
            vector_inputs = torch.cat(vector_parts, dim=-1)
        elif self.use_proprio:
            state = observations["state"]
            if self.enable_stacking:
                if state.ndim == 2:
                    state = rearrange(state, "t c -> (t c)")
                    encoding = encoding.reshape(-1)
                elif state.ndim == 3:
                    state = rearrange(state, "b t c -> b (t c)")
            vector_inputs = state

        if vector_inputs is not None:
            vector_features = self.proprio_proj(vector_inputs)
            vector_features = F.layer_norm(vector_features, vector_features.shape[-1:])
            vector_features = torch.tanh(vector_features)
            if stop_gradient:
                vector_features = vector_features.detach()
            encoding = torch.cat([encoding, vector_features], dim=-1)

        return encoding


class GCEncodingWrapper(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        goal_encoder: Optional[nn.Module],
        use_proprio: bool,
        stop_gradient: bool,
    ):
        super().__init__()
        self.encoder = encoder
        self.goal_encoder = goal_encoder
        self.use_proprio = use_proprio
        self.stop_gradient = stop_gradient

    def forward(
        self,
        observations_and_goals: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        observations, goals = observations_and_goals

        if observations["image"].ndim == 5:
            batch_size, obs_horizon = observations["image"].shape[:2]
            obs_image = rearrange(observations["image"], "b t h w c -> (b t) h w c")
            goal_image = repeat(goals["image"], "b h w c -> (b repeat) h w c", repeat=obs_horizon)
        else:
            obs_image = observations["image"]
            goal_image = goals["image"]

        if self.goal_encoder is None:
            encoder_inputs = torch.cat([obs_image, goal_image], dim=-1)
            encoding = self.encoder(encoder_inputs)
        else:
            encoding = self.encoder(obs_image)
            goal_encoding = self.goal_encoder(goal_image)
            encoding = torch.cat([encoding, goal_encoding], dim=-1)

        if observations["image"].ndim == 5:
            encoding = rearrange(encoding, "(b t) f -> b (t f)", b=batch_size, t=obs_horizon)

        if self.use_proprio:
            proprio_key = "proprio" if "proprio" in observations else "state"
            encoding = torch.cat([encoding, observations[proprio_key]], dim=-1)

        if self.stop_gradient:
            encoding = encoding.detach()

        return encoding


class LCEncodingWrapper(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        use_proprio: bool,
        stop_gradient: bool,
    ):
        super().__init__()
        self.encoder = encoder
        self.use_proprio = use_proprio
        self.stop_gradient = stop_gradient

    def forward(
        self,
        observations_and_goals: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        observations, goals = observations_and_goals

        if observations["image"].ndim == 5:
            batch_size, obs_horizon = observations["image"].shape[:2]
            obs_image = rearrange(observations["image"], "b t h w c -> (b t) h w c")
            language = repeat(goals["language"], "b e -> (b repeat) e", repeat=obs_horizon)
        else:
            obs_image = observations["image"]
            language = goals["language"]

        try:
            encoding = self.encoder(obs_image, cond_var=language)
        except TypeError:
            encoding = self.encoder(obs_image)

        if observations["image"].ndim == 5:
            encoding = rearrange(encoding, "(b t) f -> b (t f)", b=batch_size, t=obs_horizon)

        if self.use_proprio:
            proprio_key = "proprio" if "proprio" in observations else "state"
            encoding = torch.cat([encoding, observations[proprio_key]], dim=-1)

        if self.stop_gradient:
            encoding = encoding.detach()

        return encoding
