import logging
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from serl_launcher.vision.resnet_v1 import ResNetEncoder


_LOGGER = logging.getLogger(__name__)
_FUSE_VIEW_POLICIES = frozenset({"auto", "true", "false"})


def _flatten_history_image(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 4:
        return rearrange(image, "t h w c -> h w (t c)")
    if image.ndim == 5:
        return rearrange(image, "b t h w c -> b h w (t c)")
    return image


def _flatten_vector_features(
    vector: torch.Tensor,
    *,
    preserve_batch_dim: bool,
) -> torch.Tensor:
    if not preserve_batch_dim or vector.ndim <= 1:
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


def _normalize_fuse_views_policy(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    resolved = str(value if value is not None else "auto").strip().lower()
    if resolved not in _FUSE_VIEW_POLICIES:
        raise ValueError(
            "encoder.fuse_views must be one of {'auto', true, false}, "
            f"got {value!r}"
        )
    return resolved


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
        fuse_views: str | bool = "auto",
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
        self.fuse_views = _normalize_fuse_views_policy(fuse_views)
        self._fuse_static_reason = self._validate_fuse_views_static()
        if self.fuse_views == "true" and self._fuse_static_reason is not None:
            raise ValueError(
                "encoder.fuse_views=true but fused view encoding is unavailable: "
                f"{self._fuse_static_reason}"
            )
        if self.fuse_views == "auto":
            if self._fuse_static_reason is None:
                _LOGGER.info(
                    "encoder.fuse_views auto enabled for image_keys=%s",
                    self.image_keys,
                )
            else:
                _LOGGER.info(
                    "encoder.fuse_views auto disabled: %s",
                    self._fuse_static_reason,
                )

    def _validate_fuse_views_static(self) -> str | None:
        if self.fuse_views == "false":
            return "disabled by config"
        if len(self.image_keys) < 2:
            return "fewer than two image keys"

        encoders = []
        for image_key in self.image_keys:
            if image_key not in self.encoder:
                return f"missing encoder for image key {image_key!r}"
            module = self.encoder[image_key]
            if not isinstance(module, ResNetEncoder):
                return (
                    f"encoder for image key {image_key!r} is "
                    f"{type(module).__name__}, not ResNetEncoder"
                )
            if not bool(module.freeze_backbone):
                return f"encoder for image key {image_key!r} has freeze_backbone=false"
            encoders.append(module)

        first_backbone = encoders[0].backbone
        for image_key, module in zip(self.image_keys[1:], encoders[1:]):
            if module.backbone is not first_backbone:
                return f"encoder for image key {image_key!r} does not share backbone"

        return None

    def _prepare_image_for_encoder(
        self,
        observations: Dict[str, torch.Tensor],
        image_key: str,
        *,
        is_encoded: bool,
    ) -> torch.Tensor:
        image = observations[image_key]
        if not is_encoded and self.enable_stacking:
            image = _flatten_history_image(image)
        return image

    def _encode_images_loop(
        self,
        observations: Dict[str, torch.Tensor],
        *,
        train: bool,
        stop_gradient: bool,
        is_encoded: bool,
    ) -> list[torch.Tensor]:
        encoded = []
        for image_key in self.image_keys:
            image = self._prepare_image_for_encoder(
                observations,
                image_key,
                is_encoded=is_encoded,
            )
            image_feature = _maybe_call_encoder(
                self.encoder[image_key],
                image,
                train=train,
                encode=not is_encoded,
            )

            if stop_gradient:
                image_feature = image_feature.detach()

            encoded.append(image_feature)
        return encoded

    def _fused_view_dynamic_reason(
        self,
        images_bchw: list[torch.Tensor],
        squeezed: list[bool],
    ) -> str | None:
        if not images_bchw:
            return "no image tensors to fuse"
        if any(image.ndim != 4 for image in images_bchw):
            return "all fused images must be 4D BCHW tensors"
        first = images_bchw[0]
        first_squeezed = bool(squeezed[0])
        for image_key, image, was_squeezed in zip(
            self.image_keys[1:],
            images_bchw[1:],
            squeezed[1:],
        ):
            if image.device != first.device:
                return f"image key {image_key!r} is on {image.device}, expected {first.device}"
            if image.dtype != first.dtype:
                return f"image key {image_key!r} has dtype {image.dtype}, expected {first.dtype}"
            if bool(was_squeezed) != first_squeezed:
                return "all fused images must consistently include or omit batch dim"
            if tuple(image.shape) != tuple(first.shape):
                return (
                    f"image key {image_key!r} has shape {tuple(image.shape)}, "
                    f"expected {tuple(first.shape)}"
                )
        return None

    def _encode_images_fused(
        self,
        observations: Dict[str, torch.Tensor],
        *,
        train: bool,
        stop_gradient: bool,
        is_encoded: bool,
    ) -> tuple[list[torch.Tensor] | None, str | None]:
        if self._fuse_static_reason is not None:
            return None, self._fuse_static_reason
        if is_encoded:
            return None, "observations are already encoded"

        encoders = [self.encoder[image_key] for image_key in self.image_keys]
        images_bchw = []
        squeezed = []
        for image_key, encoder in zip(self.image_keys, encoders):
            image = self._prepare_image_for_encoder(
                observations,
                image_key,
                is_encoded=False,
            )
            if not isinstance(image, torch.Tensor):
                return None, f"image key {image_key!r} is not a torch.Tensor"
            image_bchw, was_squeezed = encoder.observations_to_bchw(image)
            images_bchw.append(image_bchw)
            squeezed.append(bool(was_squeezed))

        dynamic_reason = self._fused_view_dynamic_reason(images_bchw, squeezed)
        if dynamic_reason is not None:
            return None, dynamic_reason

        batch_size = int(images_bchw[0].shape[0])
        fused_images = torch.cat(images_bchw, dim=0)
        fused_features = encoders[0].encode_backbone_bchw(fused_images)
        features_by_view = fused_features.split(batch_size, dim=0)

        encoded = []
        squeeze_outputs = bool(squeezed[0])
        for encoder, features in zip(encoders, features_by_view):
            image_feature = encoder.pool_features(features, train=train)
            if squeeze_outputs and image_feature.ndim > 1:
                image_feature = image_feature.squeeze(0)
            if stop_gradient:
                image_feature = image_feature.detach()
            encoded.append(image_feature)

        return encoded, None

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        train: bool = False,
        stop_gradient: bool = False,
        is_encoded: bool = False,
    ) -> torch.Tensor:
        encoded = None
        if self.fuse_views != "false":
            encoded, reason = self._encode_images_fused(
                observations,
                train=train,
                stop_gradient=stop_gradient,
                is_encoded=is_encoded,
            )
            if encoded is None and self.fuse_views == "true":
                raise ValueError(
                    "encoder.fuse_views=true but fused view encoding is unavailable: "
                    f"{reason}"
                )

        if encoded is None:
            encoded = self._encode_images_loop(
                observations,
                train=train,
                stop_gradient=stop_gradient,
                is_encoded=is_encoded,
            )

        encoding = torch.cat(encoded, dim=-1)

        vector_inputs = None
        preserve_vector_batch_dim = encoding.ndim > 1
        if self.vector_obs_keys is not None and len(self.vector_obs_keys) > 0:
            vector_parts = []
            for key in self.vector_obs_keys:
                if key not in observations:
                    raise KeyError(
                        f"Missing vector observation key {key!r}. "
                        f"Available keys: {list(observations.keys())}"
                    )
                vector_parts.append(
                    _flatten_vector_features(
                        observations[key],
                        preserve_batch_dim=preserve_vector_batch_dim,
                    )
                )
            vector_inputs = torch.cat(vector_parts, dim=-1)
        elif self.use_proprio:
            state = observations["state"]
            if self.enable_stacking:
                if state.ndim == 2:
                    state = rearrange(state, "t c -> (t c)")
                elif state.ndim == 3:
                    state = rearrange(state, "b t c -> b (t c)")
            vector_inputs = _flatten_vector_features(
                state,
                preserve_batch_dim=preserve_vector_batch_dim,
            )

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
