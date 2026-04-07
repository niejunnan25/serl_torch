"""Declarative schema for residual-training datasets."""
from __future__ import annotations

import dataclasses

RESIDUAL_TRAINING_FORMAT = "libero_residual_training"
RESIDUAL_TRAINING_MANIFEST_FORMAT = "libero_residual_training_manifest_v1"


@dataclasses.dataclass(frozen=True)
class ImageSlotSpec:
    name: str
    required: bool = True


@dataclasses.dataclass(frozen=True)
class ObservationSchema:
    root: str = "observation"
    image_key: str = "image"
    image_mask_key: str = "image_mask"
    state_key: str = "state"
    image_slots: tuple[ImageSlotSpec, ...] = (
        ImageSlotSpec("image_rgb_0", required=True),
        ImageSlotSpec("image_rgb_1", required=True),
        ImageSlotSpec("image_rgb_2", required=False),
    )

    @property
    def image_root_path(self) -> str:
        return f"{self.root}/{self.image_key}"

    @property
    def image_mask_path(self) -> str:
        return f"{self.root}/{self.image_mask_key}"

    @property
    def state_path(self) -> str:
        return f"{self.root}/{self.state_key}"

    @property
    def image_slot_names(self) -> tuple[str, ...]:
        return tuple(slot.name for slot in self.image_slots)


@dataclasses.dataclass(frozen=True)
class ActionSchema:
    root: str = "action"
    base_chunks_key: str = "base_chunks"
    final_key: str = "final"
    expert_key: str = "expert"
    alpha_key: str = "alpha"

    @property
    def base_chunks_path(self) -> str:
        return f"{self.root}/{self.base_chunks_key}"

    @property
    def final_path(self) -> str:
        return f"{self.root}/{self.final_key}"

    @property
    def expert_path(self) -> str:
        return f"{self.root}/{self.expert_key}"

    @property
    def alpha_path(self) -> str:
        return f"{self.root}/{self.alpha_key}"


@dataclasses.dataclass(frozen=True)
class TrajectorySchema:
    root: str = "trajectory"
    rewards_key: str = "rewards"
    dones_key: str = "dones"

    @property
    def rewards_path(self) -> str:
        return f"{self.root}/{self.rewards_key}"

    @property
    def dones_path(self) -> str:
        return f"{self.root}/{self.dones_key}"


@dataclasses.dataclass(frozen=True)
class EpisodeSchema:
    root: str = "episode"
    source_key: str = "source"
    suite_name_key: str = "suite_name"
    task_id_key: str = "task_id"
    task_key_key: str = "task_key"
    task_description_key: str = "task_description"
    episode_index_key: str = "index"
    episode_steps_key: str = "steps"
    episode_return_key: str = "return"
    episode_success_key: str = "success"

    @property
    def source_path(self) -> str:
        return f"{self.root}/{self.source_key}"

    @property
    def suite_name_path(self) -> str:
        return f"{self.root}/{self.suite_name_key}"

    @property
    def task_id_path(self) -> str:
        return f"{self.root}/{self.task_id_key}"

    @property
    def task_key_path(self) -> str:
        return f"{self.root}/{self.task_key_key}"

    @property
    def task_description_path(self) -> str:
        return f"{self.root}/{self.task_description_key}"

    @property
    def episode_index_path(self) -> str:
        return f"{self.root}/{self.episode_index_key}"

    @property
    def episode_steps_path(self) -> str:
        return f"{self.root}/{self.episode_steps_key}"

    @property
    def episode_return_path(self) -> str:
        return f"{self.root}/{self.episode_return_key}"

    @property
    def episode_success_path(self) -> str:
        return f"{self.root}/{self.episode_success_key}"


@dataclasses.dataclass(frozen=True)
class ResidualTrainingSchema:
    episode_format: str = RESIDUAL_TRAINING_FORMAT
    manifest_format: str = RESIDUAL_TRAINING_MANIFEST_FORMAT
    format_key: str = "format"
    prompt_key: str = "prompt"
    metadata_key: str = "metadata"
    observation: ObservationSchema = dataclasses.field(default_factory=ObservationSchema)
    action: ActionSchema = dataclasses.field(default_factory=ActionSchema)
    trajectory: TrajectorySchema = dataclasses.field(default_factory=TrajectorySchema)
    episode: EpisodeSchema = dataclasses.field(default_factory=EpisodeSchema)

    @property
    def prompt_path(self) -> str:
        return self.prompt_key

    @property
    def metadata_path(self) -> str:
        return self.metadata_key


LIBERO_RESIDUAL_TRAINING_SCHEMA = ResidualTrainingSchema()

