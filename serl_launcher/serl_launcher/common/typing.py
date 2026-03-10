from typing import Any, Callable, Dict, Optional, Sequence, Union

import numpy as np
import torch

PRNGKey = Optional[int]
Params = Dict[str, Dict[str, torch.Tensor]]
Shape = Sequence[int]
Dtype = Any
InfoDict = Dict[str, float]
Array = Union[np.ndarray, torch.Tensor]
Data = Union[Array, Dict[str, "Data"]]
Batch = Dict[str, Data]
ModuleMethod = Union[str, Callable, None]
