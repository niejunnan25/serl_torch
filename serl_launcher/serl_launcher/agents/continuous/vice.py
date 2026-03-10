import copy
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F

from serl_launcher.agents.continuous.drq import DrQAgent
from serl_launcher.agents.continuous.sac import SACAgent, _to_torch
from serl_launcher.common.encoding import EncodingWrapper
from serl_launcher.common.optimizers import make_optimizer
from serl_launcher.networks.actor_critic_nets import Critic, CriticEnsemble, Policy
from serl_launcher.networks.classifier import BinaryClassifier
from serl_launcher.networks.lagrange import GeqLagrangeMultiplier
from serl_launcher.networks.mlp import MLP
from serl_launcher.utils.train_utils import _unpack


class VICEAgent(DrQAgent):
    @classmethod
    def create(
        cls,
        rng,
        observations,
        actions,
        actor_def,
        critic_def,
        temperature_def,
        vice_def,
        actor_optimizer_kwargs={"learning_rate": 3e-4},
        critic_optimizer_kwargs={"learning_rate": 3e-4},
        temperature_optimizer_kwargs={"learning_rate": 3e-4},
        vice_optimizer_kwargs={"learning_rate": 3e-4},
        discount: float = 0.95,
        soft_target_update_rate: float = 0.005,
        target_entropy: Optional[float] = None,
        entropy_per_dim: bool = False,
        backup_entropy: bool = False,
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
        image_keys: Iterable[str] = ("image",),
    ):
        agent = super().create(
            rng,
            observations,
            actions,
            actor_def=actor_def,
            critic_def=critic_def,
            temperature_def=temperature_def,
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
            image_keys=image_keys,
        )

        device = agent.device
        vice_def = vice_def.to(device)
        obs_t = _to_torch(observations, device)
        with torch.no_grad():
            vice_def(obs_t, train=False)

        vice_bundle = make_optimizer(vice_def.parameters(), **vice_optimizer_kwargs)

        agent.state.modules["vice"] = vice_def
        agent.state.optimizers["vice"] = vice_bundle.optimizer
        agent.state.schedulers["vice"] = vice_bundle.scheduler
        agent.state.grad_clip_norms["vice"] = vice_bundle.clip_grad_norm

        return agent

    @classmethod
    def create_vice(
        cls,
        rng,
        observations,
        actions,
        vice_observations=None,
        encoder_type: str = "small",
        shared_encoder: bool = True,
        use_proprio: bool = False,
        critic_network_kwargs: dict = {"hidden_dims": [256, 256]},
        policy_network_kwargs: dict = {"hidden_dims": [256, 256]},
        vice_network_kwargs: dict = {
            "hidden_dims": [256],
            "activations": "leaky_relu",
            "use_layer_norm": True,
            "dropout_rate": 0.1,
        },
        policy_kwargs: dict = {
            "tanh_squash_distribution": True,
            "std_parameterization": "uniform",
        },
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
        temperature_init: float = 1.0,
        image_keys: Iterable[str] = ("image",),
        vice_image_keys: Optional[Iterable[str]] = None,
        resnet_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        del vice_observations
        vice_image_keys = tuple(vice_image_keys or image_keys)

        policy_network_kwargs = dict(policy_network_kwargs)
        critic_network_kwargs = dict(critic_network_kwargs)
        vice_network_kwargs = dict(vice_network_kwargs)

        policy_network_kwargs["activate_final"] = True
        critic_network_kwargs["activate_final"] = True
        vice_network_kwargs["activate_final"] = True

        backbone_encoder = None
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
            vice_encoders = {
                image_key: SmallEncoder(
                    features=(32, 64, 128, 256),
                    kernel_sizes=(3, 3, 3, 3),
                    strides=(2, 2, 2, 2),
                    padding="VALID",
                    pool_method="avg",
                    bottleneck_dim=256,
                    spatial_block_size=8,
                )
                for image_key in vice_image_keys
            }
        elif encoder_type in {"resnet", "resnet-pretrained"}:
            from serl_launcher.vision.resnet_v1 import ResNetEncoder

            kw = dict(resnet_kwargs or {})
            freeze = kw.get("freeze_backbone", False)
            backbone = ResNetEncoder.create_backbone(
                model_name=kw.get("model_name", "microsoft/resnet-18"),
                pretrained=kw.get("pretrained", True),
                freeze=freeze,
            )
            enc_kw = dict(
                backbone=backbone,
                freeze_backbone=freeze,
                pooling_method=kw.get("pooling_method", "spatial_learned_embeddings"),
                num_spatial_blocks=kw.get("num_spatial_blocks", 8),
                bottleneck_dim=kw.get("bottleneck_dim", 256),
            )
            encoders = {key: ResNetEncoder(**enc_kw) for key in image_keys}
            vice_encoders = {key: ResNetEncoder(**enc_kw) for key in vice_image_keys}
            backbone_encoder = ResNetEncoder(
                backbone=backbone,
                freeze_backbone=freeze,
                pooling_method="none",
                bottleneck_dim=None,
            )
        else:
            raise NotImplementedError(f"Unknown encoder type: {encoder_type}")

        encoder_def = EncodingWrapper(
            encoder=encoders,
            use_proprio=use_proprio,
            enable_stacking=True,
            image_keys=image_keys,
        )

        if shared_encoder:
            actor_encoder = encoder_def
            critic_encoder = encoder_def
        else:
            actor_encoder = encoder_def
            critic_encoder = copy.deepcopy(encoder_def)

        vice_encoder_def = EncodingWrapper(
            encoder=vice_encoders,
            use_proprio=False,
            enable_stacking=True,
            image_keys=vice_image_keys,
        )

        critic_ctor = lambda: Critic(encoder=critic_encoder, network=MLP(**critic_network_kwargs))
        critic_def = CriticEnsemble(critic_ctor=critic_ctor, num_qs=critic_ensemble_size)

        policy_def = Policy(
            encoder=actor_encoder,
            network=MLP(**policy_network_kwargs),
            action_dim=actions.shape[-1],
            **policy_kwargs,
        )

        temperature_def = GeqLagrangeMultiplier(
            init_value=temperature_init,
            constraint_shape=(),
        )

        vice_pretrained = backbone_encoder if backbone_encoder is not None else torch.nn.Identity()
        vice_def = BinaryClassifier(
            pretrained_encoder=vice_pretrained,
            encoder=vice_encoder_def,
            network=MLP(**vice_network_kwargs),
            enable_stacking=True,
        )

        agent = cls.create(
            rng,
            observations,
            actions,
            actor_def=policy_def,
            critic_def=critic_def,
            temperature_def=temperature_def,
            vice_def=vice_def,
            critic_ensemble_size=critic_ensemble_size,
            critic_subsample_size=critic_subsample_size,
            image_keys=image_keys,
            **kwargs,
        )

        return agent

    def encode_images(self, images, train: bool = True):
        vice = self.state.modules["vice"]
        return vice(images, train=train, return_encoded=True)

    def update_vice(self, batch, pmap_axis: Optional[str] = None):
        del pmap_axis
        if self.config["image_keys"][0] not in batch["next_observations"]:
            batch = _unpack(batch)

        batch = _to_torch(batch, self.device)
        observations = batch["next_observations"]
        aug_observations = self.data_augmentation_fn(observations)

        logits = self.state.modules["vice"](aug_observations, train=True)
        labels = torch.zeros_like(logits)
        labels[logits.shape[0] // 2 :] = 1.0
        labels = labels * (1.0 - 0.2) + 0.5 * 0.2

        self.state.zero_grad(["vice"])
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels)
        bce_loss.backward()
        self.state.optimizer_step("vice")
        self.state.step += 1

        info = {
            "vice_bce_loss": float(bce_loss.detach().cpu()),
            "vice_logits_mean": float(logits.mean().detach().cpu()),
        }
        return self, info

    @torch.no_grad()
    def vice_reward(self, observation):
        obs_t = _to_torch(observation, self.device)
        logits = self.state.modules["vice"](obs_t, train=False)
        return torch.sigmoid(logits)

    def update_critics(
        self,
        batch,
        *,
        pmap_axis: Optional[str] = None,
    ) -> Tuple["VICEAgent", dict]:
        del pmap_axis
        if self.config["image_keys"][0] not in batch["next_observations"]:
            batch = _unpack(batch)

        batch = _to_torch(batch, self.device)
        batch["observations"] = self.data_augmentation_fn(batch["observations"])
        batch["next_observations"] = self.data_augmentation_fn(batch["next_observations"])
        rewards = (self.vice_reward(batch["next_observations"]) >= 0.5).float()
        batch["rewards"] = rewards

        return self.update(batch, networks_to_update=frozenset({"critic"}))

    def update_high_utd(
        self,
        batch,
        *,
        utd_ratio: int,
        pmap_axis: Optional[str] = None,
    ) -> Tuple["VICEAgent", dict]:
        del pmap_axis
        if self.config["image_keys"][0] not in batch["next_observations"]:
            batch = _unpack(batch)

        batch = _to_torch(batch, self.device)
        batch["observations"] = self.data_augmentation_fn(batch["observations"])
        batch["next_observations"] = self.data_augmentation_fn(batch["next_observations"])
        rewards = (self.vice_reward(batch["next_observations"]) >= 0.5).float()
        batch["rewards"] = rewards

        agent, info = SACAgent.update_high_utd(self, batch, utd_ratio=utd_ratio)
        info["vice_rewards"] = float(rewards.mean().detach().cpu())
        return agent, info
