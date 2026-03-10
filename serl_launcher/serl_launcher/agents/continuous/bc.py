from typing import Iterable, Optional

import torch

from serl_launcher.common.common import TorchRLTrainState, nonpytree_field
from serl_launcher.common.encoding import EncodingWrapper
from serl_launcher.common.optimizers import make_optimizer
from serl_launcher.networks.actor_critic_nets import Policy
from serl_launcher.networks.mlp import MLP
from serl_launcher.utils.train_utils import _unpack


def _to_torch(data, device: torch.device):
    if isinstance(data, dict):
        return {k: _to_torch(v, device) for k, v in data.items()}
    if isinstance(data, torch.Tensor):
        tensor = data.to(device)
    else:
        tensor = torch.as_tensor(data, device=device)
    if tensor.dtype == torch.float64:
        tensor = tensor.float()
    return tensor


class BCAgent:
    state: TorchRLTrainState
    config: dict = nonpytree_field(default_factory=dict)

    def __init__(self, state: TorchRLTrainState, config: dict):
        self.state = state
        self.config = config

    @property
    def device(self):
        return self.state.device

    def replace(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    def data_augmentation_fn(self, observations):
        return observations

    def update(self, batch, pmap_axis: str = None):
        del pmap_axis
        if self.config["image_keys"] and self.config["image_keys"][0] not in batch["next_observations"]:
            batch = _unpack(batch)

        batch = _to_torch(batch, self.device)
        actor = self.state.modules["actor"]

        self.state.zero_grad(["actor"])
        dist = actor(batch["observations"], temperature=1.0, train=True)
        pi_actions = dist.mode()
        log_probs = dist.log_prob(batch["actions"])
        mse = ((pi_actions - batch["actions"]) ** 2).sum(dim=-1)
        actor_loss = -(log_probs).mean()
        actor_loss.backward()
        self.state.optimizer_step("actor")
        self.state.step += 1

        info = {
            "actor_loss": float(actor_loss.detach().cpu()),
            "mse": float(mse.mean().detach().cpu()),
        }
        info.update(self.state.lr_info())
        return self, info

    @torch.no_grad()
    def sample_actions(
        self,
        observations,
        *,
        seed: Optional[int] = None,
        temperature: float = 1.0,
        argmax: bool = False,
    ):
        del seed
        obs_t = _to_torch(observations, self.device)
        dist = self.state.modules["actor"](obs_t, temperature=temperature, train=False)
        actions = dist.mode() if argmax else dist.sample()
        return actions.detach().cpu().numpy()

    @torch.no_grad()
    def get_debug_metrics(self, batch, **kwargs):
        del kwargs
        batch = _to_torch(batch, self.device)
        dist = self.state.modules["actor"](batch["observations"], temperature=1.0, train=False)
        pi_actions = dist.mode()
        log_probs = dist.log_prob(batch["actions"])
        mse = ((pi_actions - batch["actions"]) ** 2).sum(dim=-1)
        return {
            "mse": mse.detach().cpu().numpy(),
            "log_probs": log_probs.detach().cpu().numpy(),
            "pi_actions": pi_actions.detach().cpu().numpy(),
        }

    @classmethod
    def create(
        cls,
        rng,
        observations,
        actions,
        encoder_type: str = "small",
        image_keys: Iterable[str] = ("image",),
        use_proprio: bool = False,
        network_kwargs: dict = {"hidden_dims": [256, 256]},
        policy_kwargs: dict = {"tanh_squash_distribution": False},
        learning_rate: float = 3e-4,
        resnet_kwargs: Optional[dict] = None,
    ):
        del rng
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        encoder_def = EncodingWrapper(
            encoder=encoders,
            use_proprio=use_proprio,
            enable_stacking=True,
            image_keys=image_keys,
        )

        network_kwargs = dict(network_kwargs)
        network_kwargs["activate_final"] = True

        actor = Policy(
            encoder=encoder_def,
            network=MLP(**network_kwargs),
            action_dim=actions.shape[-1],
            **policy_kwargs,
        ).to(device)

        obs_t = _to_torch(observations, device)
        with torch.no_grad():
            actor(obs_t, temperature=1.0, train=False)

        actor_bundle = make_optimizer(actor.parameters(), learning_rate=learning_rate)

        state = TorchRLTrainState(
            modules={"actor": actor},
            target_modules={},
            optimizers={"actor": actor_bundle.optimizer},
            schedulers={"actor": actor_bundle.scheduler},
            grad_clip_norms={"actor": actor_bundle.clip_grad_norm},
            device=device,
        )
        config = dict(image_keys=tuple(image_keys))

        agent = cls(state, config)

        return agent
