"""
DRQ (Data-regularized Q) Agent 模块。

在 SAC 基础上增加基于图像的观测编码与数据增强，适用于从图像观测学习机械臂策略。
"""

import copy
from typing import Iterable, Optional, Tuple

import torch

from serl_launcher.agents.continuous.sac import SACAgent, _to_torch
from serl_launcher.common.encoding import EncodingWrapper
from serl_launcher.networks.actor_critic_nets import Critic, CriticEnsemble, Policy
from serl_launcher.networks.lagrange import GeqLagrangeMultiplier
from serl_launcher.networks.mlp import MLP
from serl_launcher.utils.train_utils import _unpack
from serl_launcher.vision.data_augmentations import batched_random_crop


class DrQAgent(SACAgent):
    """
    DRQ Agent：继承 SACAgent，支持图像观测与随机裁剪数据增强。

    与 SAC 的主要区别：
    - 观测通过视觉编码器（SmallEncoder / ResNet 等）编码；
    - 训练时对图像做随机裁剪增强，提升样本效率与泛化。
    """

    @classmethod
    def create(
        cls,
        rng,
        observations,
        actions,
        actor_def,
        critic_def,
        temperature_def,
        critic_actions=None,
        actor_optimizer_kwargs={"learning_rate": 3e-4},
        critic_optimizer_kwargs={"learning_rate": 3e-4},
        temperature_optimizer_kwargs={"learning_rate": 3e-4},
        discount: float = 0.95,
        soft_target_update_rate: float = 0.005,
        target_entropy: Optional[float] = None,
        entropy_per_dim: bool = False,
        backup_entropy: bool = False,
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
        image_keys: Iterable[str] = ("image",),
        **kwargs,
    ):
        """从已定义好的 actor/critic/temperature 模块创建 DRQ agent，并记录 image_keys。"""
        agent = super().create(
            rng,
            observations,
            actions,
            actor_def=actor_def,
            critic_def=critic_def,
            temperature_def=temperature_def,
            critic_actions=critic_actions,
            actor_optimizer_kwargs=actor_optimizer_kwargs,
            critic_optimizer_kwargs=critic_optimizer_kwargs,
            temperature_optimizer_kwargs=temperature_optimizer_kwargs,
            discount=discount,
            soft_target_update_rate=soft_target_update_rate,
            target_entropy=target_entropy,
            entropy_per_dim=entropy_per_dim,
            backup_entropy=backup_entropy,
            critic_ensemble_size=critic_ensemble_size,
            critic_subsample_size=critic_subsample_size,
            **kwargs,
        )
        # 记录哪些观测键是图像，用于后续数据增强与 unpack
        agent.config["image_keys"] = tuple(image_keys)
        return agent

    @classmethod
    def create_drq(
        cls,
        rng,
        observations,
        actions,
        critic_actions=None,
        encoder_type: str = "small",
        shared_encoder: bool = True,
        use_proprio: bool = False,
        critic_network_kwargs: dict = {"hidden_dims": [256, 256]},
        policy_network_kwargs: dict = {"hidden_dims": [256, 256]},
        policy_kwargs: dict = {
            "tanh_squash_distribution": True,
            "std_parameterization": "uniform",
        },
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
        temperature_init: float = 1.0,
        image_keys: Iterable[str] = ("image",),
        resnet_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        一站式创建 DRQ agent：根据 encoder_type 构建编码器、策略、Q 网络与温度，再调用 create。

        encoder_type: "small" | "resnet"
        resnet_kwargs: ResNetEncoder 构造参数（model_name, pretrained, freeze_backbone, pooling_method 等），
                       仅当 encoder_type == "resnet" 时使用。
        shared_encoder: True 时 actor 与 critic 共用同一编码器，否则各有一份拷贝。
        """
        policy_network_kwargs = dict(policy_network_kwargs)
        critic_network_kwargs = dict(critic_network_kwargs)
        policy_network_kwargs["activate_final"] = True
        critic_network_kwargs["activate_final"] = True

        if encoder_type == "small":
            from serl_launcher.vision.small_encoders import SmallEncoder

            encoders = {
                image_key: SmallEncoder(
                    features=(32, 64, 128, 256),
                    kernel_sizes=(3, 3, 3, 3),
                    strides=(2, 2, 2, 2),
                    padding="VALID",
                    pool_method="avg",
                    bottleneck_dim=256,
                    spatial_block_size=8,
                )
                for image_key in image_keys
            }
        elif encoder_type in {"resnet", "resnet-pretrained"}:
            from serl_launcher.vision.resnet_v1 import ResNetEncoder

            kw = dict(resnet_kwargs or {})
            backbone = ResNetEncoder.create_backbone(
                model_name=kw.get("model_name", "microsoft/resnet-18"),
                pretrained=kw.get("pretrained", True),
                freeze=kw.get("freeze_backbone", False),
            )
            encoders = {
                key: ResNetEncoder(
                    backbone=backbone,
                    freeze_backbone=kw.get("freeze_backbone", False),
                    pooling_method=kw.get("pooling_method", "spatial_learned_embeddings"),
                    num_spatial_blocks=kw.get("num_spatial_blocks", 8),
                    bottleneck_dim=kw.get("bottleneck_dim", 256),
                )
                for key in image_keys
            }
        else:
            raise NotImplementedError(f"Unknown encoder type: {encoder_type}")

        # 将多路图像编码 + 可选本体感拼接成单一观测编码
        encoder_def = EncodingWrapper(
            encoder=encoders,
            use_proprio=use_proprio,
            enable_stacking=True,
            image_keys=image_keys,
        )

        # 是否共享编码器：共享可减少参数量，不共享则 actor/critic 各一份
        if shared_encoder:
            actor_encoder = encoder_def
            critic_encoder = encoder_def
        else:
            actor_encoder = encoder_def
            critic_encoder = copy.deepcopy(encoder_def)

        # Q 网络：编码器 + MLP，多份组成 ensemble
        critic_ctor = lambda: Critic(encoder=critic_encoder, network=MLP(**critic_network_kwargs))
        critic_def = CriticEnsemble(critic_ctor=critic_ctor, num_qs=critic_ensemble_size)

        # 策略网络：编码器 + MLP，输出动作分布
        policy_def = Policy(
            encoder=actor_encoder,
            network=MLP(**policy_network_kwargs),
            action_dim=actions.shape[-1],
            **policy_kwargs,
        )

        # 温度 / 熵的拉格朗日乘子（≥ 约束）
        temperature_def = GeqLagrangeMultiplier(
            init_value=temperature_init,
            constraint_shape=(),
        )

        agent = cls.create(
            rng,
            observations,
            actions,
            actor_def=policy_def,
            critic_def=critic_def,
            temperature_def=temperature_def,
            critic_actions=critic_actions,
            critic_ensemble_size=critic_ensemble_size,
            critic_subsample_size=critic_subsample_size,
            image_keys=image_keys,
            **kwargs,
        )

        return agent

    def data_augmentation_fn(self, observations):
        """对观测中的图像做随机裁剪增强（DRQ 核心：data regularization）。"""
        observations = dict(observations)
        for pixel_key in self.config["image_keys"]:
            observations[pixel_key] = batched_random_crop(
                observations[pixel_key],
                padding=4,
                num_batch_dims=2,
            )
        return observations

    def update_high_utd(
        self,
        batch,
        *,
        utd_ratio: int,
        pmap_axis: Optional[str] = None,
    ) -> Tuple["DrQAgent", dict]:
        """
        高 UTD（update-to-data）比的一次更新：先对图像做数据增强，再调用 SAC 的 update_high_utd。
        utd_ratio 表示每批数据要做的梯度更新次数。
        """
        del pmap_axis
        # 若 batch 是打包格式（如来自 agentlace），先解包成带 image 等键的 dict
        if self.config["image_keys"][0] not in batch["next_observations"]:
            batch = _unpack(batch)

        batch = _to_torch(batch, self.device)
        batch["observations"] = self.data_augmentation_fn(batch["observations"])
        batch["next_observations"] = self.data_augmentation_fn(batch["next_observations"])

        return super().update_high_utd(batch, utd_ratio=utd_ratio)

    def update_critics(
        self,
        batch,
        *,
        pmap_axis: Optional[str] = None,
    ) -> Tuple["DrQAgent", dict]:
        """
        仅更新 Q 网络（critic）：先做图像增强，再调用 update 并限定 networks_to_update={"critic"}。
        常用于异步架构中 learner 只更新 critic 的步骤。
        """
        del pmap_axis
        if self.config["image_keys"][0] not in batch["next_observations"]:
            batch = _unpack(batch)

        batch = _to_torch(batch, self.device)
        batch["observations"] = self.data_augmentation_fn(batch["observations"])
        batch["next_observations"] = self.data_augmentation_fn(batch["next_observations"])

        return self.update(batch, networks_to_update=frozenset({"critic"}))

    def update_critics_calql(
        self,
        batch,
        *,
        calql_alpha: float,
        calql_n_actions: int,
        calql_temperature: float,
        pmap_axis: Optional[str] = None,
    ) -> Tuple["DrQAgent", dict]:
        """
        仅更新 Q 网络，并启用 Cal-QL/CQL-style 保守项。
        与 update_critics 一样，先对图像观测做 DrQ 随机裁剪增强。
        """
        del pmap_axis
        if self.config["image_keys"][0] not in batch["next_observations"]:
            batch = _unpack(batch)

        batch = _to_torch(batch, self.device)
        batch["observations"] = self.data_augmentation_fn(batch["observations"])
        batch["next_observations"] = self.data_augmentation_fn(batch["next_observations"])
        return super().update_critics_calql(
            batch,
            calql_alpha=float(calql_alpha),
            calql_n_actions=int(calql_n_actions),
            calql_temperature=float(calql_temperature),
        )
