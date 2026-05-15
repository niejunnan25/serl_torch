import torch
from torch import nn

from serl_launcher.common.common import TorchRLTrainState
from serl_launcher.common.common import soft_update


class MixedFrozenModule(nn.Module):
    def __init__(
        self,
        *,
        frozen_value: float,
        trainable_value: float,
    ):
        super().__init__()
        self.frozen = nn.Parameter(
            torch.full((4,), float(frozen_value)),
            requires_grad=False,
        )
        self.trainable = nn.Parameter(
            torch.full((4,), float(trainable_value)),
            requires_grad=True,
        )


def test_target_update_skips_frozen_parameter_pairs():
    source = MixedFrozenModule(frozen_value=7.0, trainable_value=10.0)
    target = MixedFrozenModule(frozen_value=3.0, trainable_value=2.0)
    state = TorchRLTrainState(
        modules={"critic": source},
        target_modules={"critic": target},
        optimizers={},
    )

    state.target_update(tau=0.25)

    torch.testing.assert_close(target.frozen, torch.full((4,), 3.0))
    torch.testing.assert_close(target.trainable, torch.full((4,), 4.0))


def test_target_update_still_updates_if_either_side_is_trainable():
    source = MixedFrozenModule(frozen_value=7.0, trainable_value=10.0)
    target = MixedFrozenModule(frozen_value=3.0, trainable_value=2.0)
    target.frozen.requires_grad_(True)
    state = TorchRLTrainState(
        modules={"critic": source},
        target_modules={"critic": target},
        optimizers={},
    )

    state.target_update(tau=0.25)

    torch.testing.assert_close(target.frozen, torch.full((4,), 4.0))
    torch.testing.assert_close(target.trainable, torch.full((4,), 4.0))


def test_soft_update_keeps_full_update_semantics_for_frozen_parameters():
    source = MixedFrozenModule(frozen_value=7.0, trainable_value=10.0)
    target = MixedFrozenModule(frozen_value=3.0, trainable_value=2.0)

    soft_update(target, source, tau=0.25)

    torch.testing.assert_close(target.frozen, torch.full((4,), 4.0))
    torch.testing.assert_close(target.trainable, torch.full((4,), 4.0))
