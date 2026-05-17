import copy
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
import torch.nn as nn

from serl_launcher.common.torch_module_compat import load_module_state_dict
from serl_launcher.common.torch_module_compat import module_state_dict


def nonpytree_field(**kwargs):
    return field(**kwargs)


def default_init(module: nn.Module):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def shard_batch(batch, _sharding=None):
    return batch


def _clone_state_dict(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


def _soft_update_parameters(
    target: nn.Module,
    source: nn.Module,
    tau: float,
    *,
    skip_frozen: bool = False,
):
    grouped: dict[tuple[torch.device, torch.dtype], list[tuple[torch.Tensor, torch.Tensor]]] = {}
    fallback_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []

    for target_param, source_param in zip(target.parameters(), source.parameters()):
        if (
            skip_frozen
            and (not bool(target_param.requires_grad))
            and (not bool(source_param.requires_grad))
        ):
            continue
        if (
            target_param.device == source_param.device
            and target_param.dtype == source_param.dtype
            and target_param.layout == torch.strided
            and source_param.layout == torch.strided
        ):
            grouped.setdefault((target_param.device, target_param.dtype), []).append(
                (target_param, source_param)
            )
        else:
            fallback_pairs.append((target_param, source_param))

    with torch.no_grad():
        for pairs in grouped.values():
            target_params = [target_param for target_param, _source_param in pairs]
            source_params = [source_param for _target_param, source_param in pairs]
            torch._foreach_mul_(target_params, 1.0 - tau)
            torch._foreach_add_(target_params, source_params, alpha=tau)

        for target_param, source_param in fallback_pairs:
            target_param.mul_(1.0 - tau).add_(source_param, alpha=tau)


class ModuleDict(nn.Module):
    """A thin compatibility wrapper mirroring the old Flax helper."""

    def __init__(self, modules: Dict[str, nn.Module]):
        super().__init__()
        self.modules_map = nn.ModuleDict(modules)

    def forward(self, *args, name: Optional[str] = None, **kwargs):
        if name is None:
            out = {}
            for key, value in kwargs.items():
                if isinstance(value, Mapping):
                    out[key] = self.modules_map[key](**value)
                elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                    out[key] = self.modules_map[key](*value)
                else:
                    out[key] = self.modules_map[key](value)
            return out
        return self.modules_map[name](*args, **kwargs)


@dataclass
class TorchRLTrainState:
    modules: Dict[str, nn.Module]
    optimizers: Dict[str, torch.optim.Optimizer]
    schedulers: Dict[str, Optional[torch.optim.lr_scheduler.LRScheduler]] = field(
        default_factory=dict
    )
    target_modules: Dict[str, nn.Module] = field(default_factory=dict)
    grad_clip_norms: Dict[str, Optional[float]] = field(default_factory=dict)
    step: int = 0
    device: torch.device = torch.device("cpu")

    def __post_init__(self):
        for module in self.modules.values():
            module.to(self.device)
        for module in self.target_modules.values():
            module.to(self.device)

    def replace(self, **kwargs):
        for key, value in kwargs.items():
            if key == "params":
                self.params = value
            elif key == "target_params":
                self.target_params = value
            else:
                setattr(self, key, value)
        return self

    @property
    def params(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return {
            name: _clone_state_dict(module_state_dict(module))
            for name, module in self.modules.items()
        }

    @params.setter
    def params(self, params: Mapping[str, Mapping[str, torch.Tensor]]):
        for name, state_dict in params.items():
            if name in self.modules:
                load_module_state_dict(self.modules[name], state_dict, strict=True)

    @property
    def target_params(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return {
            name: _clone_state_dict(module_state_dict(module))
            for name, module in self.target_modules.items()
        }

    @target_params.setter
    def target_params(self, params: Mapping[str, Mapping[str, torch.Tensor]]):
        for name, state_dict in params.items():
            if name in self.target_modules:
                load_module_state_dict(
                    self.target_modules[name],
                    state_dict,
                    strict=True,
                )

    def target_update(self, tau: float):
        for name, target in self.target_modules.items():
            if name not in self.modules:
                continue
            _soft_update_parameters(
                target,
                self.modules[name],
                tau,
                skip_frozen=True,
            )
        return self

    def optimizer_step(self, name: str):
        optimizer = self.optimizers[name]
        clip_norm = self.grad_clip_norms.get(name)
        if clip_norm is not None:
            params = [p for group in optimizer.param_groups for p in group["params"]]
            torch.nn.utils.clip_grad_norm_(params, clip_norm)
        optimizer.step()
        scheduler = self.schedulers.get(name)
        if scheduler is not None:
            scheduler.step()

    def zero_grad(self, names: Optional[Sequence[str]] = None):
        names = names or tuple(self.optimizers.keys())
        for name in names:
            self.optimizers[name].zero_grad(set_to_none=True)

    def lr_info(self) -> Dict[str, float]:
        info: Dict[str, float] = {}
        for name, optimizer in self.optimizers.items():
            if optimizer.param_groups:
                info[f"{name}_lr"] = float(optimizer.param_groups[0]["lr"])
        return info


def copy_module(module: nn.Module) -> nn.Module:
    return copy.deepcopy(module)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    _soft_update_parameters(target, source, tau, skip_frozen=False)


def hard_update(target: nn.Module, source: nn.Module):
    load_module_state_dict(target, module_state_dict(source))


def replace_dataclass(instance, **kwargs):
    if dataclasses.is_dataclass(instance):
        return dataclasses.replace(instance, **kwargs)
    for key, value in kwargs.items():
        setattr(instance, key, value)
    return instance
