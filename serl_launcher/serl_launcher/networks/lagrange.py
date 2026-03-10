from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class LagrangeMultiplier(nn.Module):
    def __init__(
        self,
        init_value: float = 1.0,
        constraint_shape: Sequence[int] = (),
        constraint_type: str = "eq",
        parameterization: Optional[str] = None,
    ):
        super().__init__()
        self.constraint_type = constraint_type
        self.parameterization = parameterization

        init_tensor = torch.full(tuple(constraint_shape) or (), float(init_value))

        if constraint_type != "eq":
            if init_value <= 0:
                raise ValueError("Inequality constraints require positive init_value")
            if parameterization == "softplus":
                init_tensor = torch.log(torch.exp(init_tensor) - 1.0)
            elif parameterization == "exp":
                init_tensor = torch.log(init_tensor)
            else:
                raise ValueError(f"Invalid parameterization: {parameterization}")
        elif parameterization is not None:
            raise ValueError("Equality constraints do not use parameterization")

        self.lagrange = nn.Parameter(init_tensor.float())

    def _value(self):
        if self.constraint_type == "eq":
            return self.lagrange
        if self.parameterization == "softplus":
            return F.softplus(self.lagrange)
        if self.parameterization == "exp":
            return torch.exp(self.lagrange)
        raise ValueError(f"Invalid parameterization: {self.parameterization}")

    def forward(self, lhs: Optional[torch.Tensor] = None, rhs: Optional[torch.Tensor] = None):
        multiplier = self._value()
        if lhs is None:
            return multiplier

        if rhs is None:
            rhs = torch.zeros_like(lhs)
        diff = lhs - rhs

        if self.constraint_type in {"eq", "geq"}:
            return multiplier * diff
        if self.constraint_type == "leq":
            return -multiplier * diff
        raise ValueError(f"Unknown constraint type: {self.constraint_type}")


def GeqLagrangeMultiplier(**kwargs):
    return LagrangeMultiplier(constraint_type="geq", parameterization="softplus", **kwargs)


def LeqLagrangeMultiplier(**kwargs):
    return LagrangeMultiplier(constraint_type="leq", parameterization="softplus", **kwargs)
