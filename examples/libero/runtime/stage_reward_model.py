from __future__ import annotations

"""Stage reward-model runtime for LIBERO KeyRL chunk rollouts."""

from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any
from typing import Sequence

import numpy as np
import torch

from serl_torch.examples.libero.env.observation import extract_libero_images
from serl_torch.examples.libero.env.observation import resolve_libero_image_key


_IMAGE_SIZE = 224
_VIEW_ALIASES = {
    "agentview": "image",
    "front": "image",
    "wrist": "wrist_image",
    "eye_in_hand": "wrist_image",
}


@dataclass
class StageRewardEpisodeState:
    bonus_given: bool = False
    trigger_step: int | None = None
    bonus_sum: float = 0.0
    max_score: float = 0.0
    scored_steps: int = 0


@dataclass(frozen=True)
class StageRewardChunkStats:
    enabled: bool
    scores: tuple[float, ...]
    bonuses: tuple[float, ...]
    bonus_sum: float
    triggered: bool
    trigger_step: int | None
    max_score: float


def empty_stage_reward_chunk_stats(*, enabled: bool) -> StageRewardChunkStats:
    return StageRewardChunkStats(
        enabled=bool(enabled),
        scores=(),
        bonuses=(),
        bonus_sum=0.0,
        triggered=False,
        trigger_step=None,
        max_score=0.0,
    )


def stage_reward_episode_payload(
    *,
    enabled: bool,
    episode_state: StageRewardEpisodeState,
    shaped_episode_return: float,
    env_episode_return: float,
) -> dict[str, Any]:
    return {
        "enabled": int(bool(enabled)),
        "bonus_sum": float(episode_state.bonus_sum),
        "triggered": int(bool(episode_state.bonus_given)),
        "trigger_step": (
            None
            if episode_state.trigger_step is None
            else int(episode_state.trigger_step)
        ),
        "max_score": float(episode_state.max_score),
        "scored_steps": int(episode_state.scored_steps),
        "shaped_episode_return": float(shaped_episode_return),
        "env_episode_return": float(env_episode_return),
    }


def resnet_kwargs_from_cfg(cfg: Any) -> dict[str, Any]:
    resnet_cfg = getattr(getattr(cfg, "encoder", None), "resnet", None)
    if resnet_cfg is None:
        return {
            "model_name": "microsoft/resnet-18",
            "pretrained": True,
            "freeze_backbone": True,
            "pooling_method": "spatial_learned_embeddings",
            "num_spatial_blocks": 8,
            "bottleneck_dim": 256,
        }
    return {
        "model_name": str(getattr(resnet_cfg, "model_name", "microsoft/resnet-18")),
        "pretrained": bool(getattr(resnet_cfg, "pretrained", True)),
        "freeze_backbone": bool(getattr(resnet_cfg, "freeze_backbone", True)),
        "pooling_method": str(
            getattr(resnet_cfg, "pooling_method", "spatial_learned_embeddings")
        ),
        "num_spatial_blocks": int(getattr(resnet_cfg, "num_spatial_blocks", 8)),
        "bottleneck_dim": int(getattr(resnet_cfg, "bottleneck_dim", 256)),
    }


def _canonical_view(view: str) -> str:
    aliased = _VIEW_ALIASES.get(str(view), str(view))
    return resolve_libero_image_key(aliased)


def _sample_obs_for_views(
    *,
    views: Sequence[str],
    image_size: int = _IMAGE_SIZE,
) -> dict[str, np.ndarray]:
    return {
        _canonical_view(view): np.zeros(
            (1, int(image_size), int(image_size), 3),
            dtype=np.uint8,
        )
        for view in views
    }


def _resize_image_if_needed(image: np.ndarray, image_size: int) -> np.ndarray:
    arr = np.asarray(image, dtype=np.uint8)
    if arr.shape[-3:-1] == (int(image_size), int(image_size)):
        return arr
    from PIL import Image

    return np.asarray(
        Image.fromarray(arr).resize(
            (int(image_size), int(image_size)),
            resample=Image.BILINEAR,
        ),
        dtype=np.uint8,
    )


def observations_to_reward_model_obs(
    observations: Sequence[dict[str, Any]],
    *,
    views: Sequence[str],
    image_size: int = _IMAGE_SIZE,
) -> dict[str, np.ndarray]:
    view_slots = tuple((str(view), _canonical_view(str(view))) for view in views)
    output: dict[str, list[np.ndarray]] = {
        slot_key: [] for _, slot_key in view_slots
    }
    for obs in observations:
        obs_dict = dict(obs)
        extracted_images: dict[str, np.ndarray] | None = None
        for _view, slot_key in view_slots:
            image = _lookup_preprocessed_image(obs_dict, slot_key=slot_key)
            if image is None:
                if extracted_images is None:
                    extracted_images = extract_libero_images(obs_dict)
                image = extracted_images[slot_key]
            output[slot_key].append(
                _resize_image_if_needed(image, int(image_size))
            )
    return {
        slot_key: np.expand_dims(np.stack(images, axis=0), axis=1).astype(np.uint8)
        for slot_key, images in output.items()
    }


def _lookup_preprocessed_image(
    obs: dict[str, Any],
    *,
    slot_key: str,
) -> np.ndarray | None:
    # Only canonical image_rgb_* keys mean the image has already gone through the
    # same preprocessing path used by the base-policy request. Raw LIBERO keys
    # such as wrist_image / agentview_image must still be routed through
    # extract_libero_images() so rotation and resize behavior stays aligned.
    if slot_key in obs:
        return np.asarray(obs[slot_key], dtype=np.uint8)
    return None


def _obs_to_torch(
    obs: dict[str, np.ndarray],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        key: torch.as_tensor(value, device=device)
        for key, value in obs.items()
    }


def _stage_bounds(key_rl_cfg: Any, stage_name: str) -> tuple[int, int] | None:
    stage_cfg = getattr(key_rl_cfg, stage_name, None)
    if stage_cfg is None or not bool(getattr(stage_cfg, "enabled", False)):
        return None
    return (
        int(getattr(stage_cfg, "start_step", 0)),
        int(getattr(stage_cfg, "end_step", 0)),
    )


def _step_in_stage(step: int, bounds: tuple[int, int] | None) -> bool:
    if bounds is None:
        return False
    start_step, end_step = bounds
    return int(start_step) <= int(step) < int(end_step)


def apply_stage_reward_scores(
    *,
    raw_chunk: Any,
    scores: Sequence[float],
    reward_model_cfg: Any,
    key_rl_cfg: Any,
    episode_state: StageRewardEpisodeState,
    key_active: bool,
) -> tuple[Any, StageRewardChunkStats]:
    if not bool(getattr(reward_model_cfg, "enabled", False)):
        return raw_chunk, empty_stage_reward_chunk_stats(enabled=False)
    if bool(
        getattr(reward_model_cfg, "apply_only_when_key_rl_active", True)
    ) and not bool(key_active):
        return raw_chunk, empty_stage_reward_chunk_stats(enabled=True)

    stage_name = str(getattr(reward_model_cfg, "stage", "stage1"))
    stage = _stage_bounds(key_rl_cfg, stage_name)
    if stage is None:
        return raw_chunk, empty_stage_reward_chunk_stats(enabled=True)

    executed_steps = int(raw_chunk.executed_steps)
    if len(scores) != executed_steps:
        raise ValueError(
            "reward-model score length mismatch: "
            f"scores={len(scores)} executed_steps={executed_steps}"
        )

    threshold = float(getattr(reward_model_cfg, "threshold", 0.8))
    bonus_value = float(getattr(reward_model_cfg, "bonus", 0.5))
    one_shot = bool(getattr(reward_model_cfg, "one_shot", True))
    shaped_rewards = [float(value) for value in raw_chunk.rewards]
    bonuses = [0.0 for _ in range(executed_steps)]
    triggered = False
    trigger_step: int | None = None
    max_score = 0.0

    for idx, raw_score in enumerate(scores):
        episode_step = int(raw_chunk.episode_step_start) + int(idx)
        if not _step_in_stage(episode_step, stage):
            continue
        score = float(raw_score)
        max_score = max(float(max_score), float(score))
        episode_state.max_score = max(float(episode_state.max_score), float(score))
        episode_state.scored_steps += 1
        if score < threshold:
            continue
        if one_shot and bool(episode_state.bonus_given):
            continue
        shaped_rewards[idx] += float(bonus_value)
        bonuses[idx] = float(bonus_value)
        triggered = True
        if trigger_step is None:
            trigger_step = int(episode_step)
        if one_shot:
            episode_state.bonus_given = True
            if episode_state.trigger_step is None:
                episode_state.trigger_step = int(episode_step)

    bonus_sum = float(sum(bonuses))
    episode_state.bonus_sum += float(bonus_sum)
    if triggered and not one_shot and episode_state.trigger_step is None:
        episode_state.trigger_step = (
            int(trigger_step) if trigger_step is not None else None
        )
        episode_state.bonus_given = True

    stats = StageRewardChunkStats(
        enabled=True,
        scores=tuple(float(score) for score in scores),
        bonuses=tuple(float(value) for value in bonuses),
        bonus_sum=float(bonus_sum),
        triggered=bool(triggered),
        trigger_step=trigger_step,
        max_score=float(max_score),
    )
    if bonus_sum <= 0.0:
        return raw_chunk, stats
    shaped_chunk = replace(
        raw_chunk,
        rewards=shaped_rewards,
        reward_sum=float(sum(shaped_rewards)),
    )
    return shaped_chunk, stats


class StageRewardModelRuntime:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        reward_model_cfg: Any,
        key_rl_cfg: Any,
        device: torch.device,
        image_size: int = _IMAGE_SIZE,
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.reward_model_cfg = reward_model_cfg
        self.key_rl_cfg = key_rl_cfg
        self.device = device
        self.image_size = int(image_size)
        self.views = tuple(str(view) for view in getattr(reward_model_cfg, "views", ()))

    def _should_score_chunk(
        self,
        *,
        raw_chunk: Any,
        key_active: bool,
        episode_state: StageRewardEpisodeState,
    ) -> bool:
        if bool(getattr(self.reward_model_cfg, "one_shot", True)) and bool(
            episode_state.bonus_given
        ):
            return False
        if bool(
            getattr(self.reward_model_cfg, "apply_only_when_key_rl_active", True)
        ) and not bool(key_active):
            return False
        stage = _stage_bounds(
            self.key_rl_cfg,
            str(getattr(self.reward_model_cfg, "stage", "stage1")),
        )
        if stage is None:
            return False
        for idx in range(int(raw_chunk.executed_steps)):
            step = int(raw_chunk.episode_step_start) + int(idx)
            if _step_in_stage(step, stage):
                return True
        return False

    def score_observations(
        self,
        observations: Sequence[dict[str, Any]],
    ) -> tuple[float, ...]:
        if not observations:
            return ()
        obs = observations_to_reward_model_obs(
            observations,
            views=self.views,
            image_size=int(self.image_size),
        )
        obs_t = _obs_to_torch(obs, device=self.device)
        with torch.no_grad():
            logits = self.model(obs_t, train=False)
            scores = torch.sigmoid(logits)
        return tuple(float(value) for value in scores.detach().cpu().tolist())

    def shape_chunk(
        self,
        *,
        raw_chunk: Any,
        episode_state: StageRewardEpisodeState,
        key_active: bool,
    ) -> tuple[Any, StageRewardChunkStats]:
        if not self._should_score_chunk(
            raw_chunk=raw_chunk,
            key_active=key_active,
            episode_state=episode_state,
        ):
            return raw_chunk, empty_stage_reward_chunk_stats(enabled=True)
        scores = self.score_observations(raw_chunk.post_step_observations)
        return apply_stage_reward_scores(
            raw_chunk=raw_chunk,
            scores=scores,
            reward_model_cfg=self.reward_model_cfg,
            key_rl_cfg=self.key_rl_cfg,
            episode_state=episode_state,
            key_active=key_active,
        )


def _checkpoint_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            value = payload.get(key, None)
            if isinstance(value, dict):
                return {
                    str(name).removeprefix("module."): tensor
                    for name, tensor in value.items()
                }
        if all(isinstance(key, str) for key in payload.keys()) and all(
            isinstance(value, torch.Tensor) for value in payload.values()
        ):
            return {
                str(name).removeprefix("module."): tensor
                for name, tensor in payload.items()
            }
    raise ValueError("reward-model checkpoint does not contain a model state dict")


def _checkpoint_metadata(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict) and isinstance(payload.get("model_config", None), dict):
        return dict(payload["model_config"])
    return {}


def create_stage_reward_classifier(
    *,
    views: Sequence[str],
    resnet_kwargs: dict[str, Any],
    device: torch.device,
    image_size: int = _IMAGE_SIZE,
) -> torch.nn.Module:
    from serl_launcher.networks.reward_classifier import create_classifier

    sample = _sample_obs_for_views(views=views, image_size=int(image_size))
    return create_classifier(
        key=None,
        sample=sample,
        image_keys=[_canonical_view(view) for view in views],
        resnet_kwargs=dict(resnet_kwargs),
        device=device,
    )


def load_stage_reward_model_checkpoint(
    *,
    checkpoint_path: str,
    views: Sequence[str],
    resnet_kwargs: dict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"reward-model checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu")
    model_config = _checkpoint_metadata(payload)
    checkpoint_views = tuple(str(view) for view in model_config.get("views", views))
    requested_views = tuple(str(view) for view in views)
    if checkpoint_views != requested_views:
        raise ValueError(
            "reward-model checkpoint views do not match config: "
            f"checkpoint={checkpoint_views} config={requested_views}"
        )
    image_size = int(model_config.get("image_size", _IMAGE_SIZE))
    checkpoint_resnet_kwargs = model_config.get("resnet_kwargs", None)
    if isinstance(checkpoint_resnet_kwargs, dict):
        resolved_resnet_kwargs = dict(checkpoint_resnet_kwargs)
    else:
        resolved_resnet_kwargs = dict(resnet_kwargs)
    model = create_stage_reward_classifier(
        views=requested_views,
        resnet_kwargs=resolved_resnet_kwargs,
        device=device,
        image_size=image_size,
    )
    model.load_state_dict(_checkpoint_state_dict(payload), strict=True)
    model.to(device)
    model.eval()
    return model, {
        **model_config,
        "image_size": image_size,
        "views": list(requested_views),
        "resnet_kwargs": resolved_resnet_kwargs,
    }


def build_stage_reward_model_runtime(
    cfg: Any,
    *,
    logger: Any,
) -> StageRewardModelRuntime | None:
    reward_model_cfg = getattr(cfg, "reward_model", None)
    if reward_model_cfg is None or not bool(getattr(reward_model_cfg, "enabled", False)):
        return None
    if str(getattr(reward_model_cfg, "stage", "stage1")) != "stage1":
        raise ValueError("reward_model.stage currently supports only stage1")
    stage = _stage_bounds(getattr(cfg, "key_rl", None), "stage1")
    if stage is None:
        raise ValueError(
            "reward_model.enabled=true requires key_rl.stage1.enabled=true"
        )
    checkpoint_path = getattr(reward_model_cfg, "checkpoint_path", None)
    if checkpoint_path is None:
        raise ValueError(
            "reward_model.checkpoint_path must be set when reward_model.enabled=true"
        )
    device_text = getattr(reward_model_cfg, "device", None)
    device = torch.device(
        str(device_text)
        if device_text is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    resnet_kwargs = resnet_kwargs_from_cfg(cfg)
    model, model_config = load_stage_reward_model_checkpoint(
        checkpoint_path=str(checkpoint_path),
        views=tuple(getattr(reward_model_cfg, "views", ())),
        resnet_kwargs=resnet_kwargs,
        device=device,
    )
    image_size = int(model_config.get("image_size", _IMAGE_SIZE))
    logger.info(
        "stage reward model enabled: checkpoint=%s views=%s stage=%s "
        "threshold=%.3f bonus=%.3f device=%s image_size=%s resnet_model=%s",
        str(checkpoint_path),
        list(getattr(reward_model_cfg, "views", ())),
        str(getattr(reward_model_cfg, "stage", "stage1")),
        float(getattr(reward_model_cfg, "threshold", 0.8)),
        float(getattr(reward_model_cfg, "bonus", 0.5)),
        str(device),
        int(image_size),
        str(model_config.get("resnet_kwargs", {}).get("model_name", "")),
    )
    return StageRewardModelRuntime(
        model=model,
        reward_model_cfg=reward_model_cfg,
        key_rl_cfg=cfg.key_rl,
        device=device,
        image_size=image_size,
    )


__all__ = [
    "StageRewardChunkStats",
    "StageRewardEpisodeState",
    "StageRewardModelRuntime",
    "apply_stage_reward_scores",
    "build_stage_reward_model_runtime",
    "create_stage_reward_classifier",
    "empty_stage_reward_chunk_stats",
    "load_stage_reward_model_checkpoint",
    "observations_to_reward_model_obs",
    "resnet_kwargs_from_cfg",
    "stage_reward_episode_payload",
]
