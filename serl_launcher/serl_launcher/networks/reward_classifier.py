import glob
import os
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from serl_launcher.common.encoding import EncodingWrapper
from serl_launcher.networks.mlp import MLP
from serl_launcher.vision.resnet_v1 import ResNetEncoder


def _to_torch(data, device):
    if isinstance(data, dict):
        return {k: _to_torch(v, device) for k, v in data.items()}
    if isinstance(data, torch.Tensor):
        return data.to(device)
    return torch.as_tensor(data, device=device)


class BinaryClassifier(nn.Module):
    def __init__(self, encoder_def: nn.Module, hidden_dim: int = 256):
        super().__init__()
        self.encoder_def = encoder_def
        self.hidden = nn.LazyLinear(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x, train: bool = False):
        x = self.encoder_def(x, train=train)
        x = self.hidden(x)
        x = self.dropout(x)
        x = torch.relu(torch.layer_norm(x, x.shape[-1:]))
        return self.out(x).squeeze(-1)


def _resolve_checkpoint_path(checkpoint_path: str, step: Optional[int] = None) -> str:
    if os.path.isfile(checkpoint_path):
        return checkpoint_path

    if step is not None:
        candidate = os.path.join(checkpoint_path, f"classifier_{step}.pt")
        if os.path.exists(candidate):
            return candidate

    candidates = sorted(glob.glob(os.path.join(checkpoint_path, "*.pt")))
    if not candidates:
        raise FileNotFoundError(f"No .pt checkpoints found in {checkpoint_path}")
    return candidates[-1]


def create_classifier(
    key: Optional[int],
    sample: Dict,
    image_keys: List[str],
    resnet_kwargs: Optional[Dict] = None,
    device: Optional[torch.device] = None,
):
    del key
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kw = dict(resnet_kwargs or {
        "model_name": "microsoft/resnet-18",
        "pretrained": True,
        "freeze_backbone": True,
        "pooling_method": "spatial_learned_embeddings",
        "num_spatial_blocks": 8,
        "bottleneck_dim": 256,
    })
    backbone = ResNetEncoder.create_backbone(
        model_name=kw.get("model_name", "microsoft/resnet-18"),
        pretrained=kw.get("pretrained", True),
        freeze=kw.get("freeze_backbone", True),
    )
    encoders = {
        key_name: ResNetEncoder(
            backbone=backbone,
            freeze_backbone=kw.get("freeze_backbone", True),
            pooling_method=kw.get("pooling_method", "spatial_learned_embeddings"),
            num_spatial_blocks=kw.get("num_spatial_blocks", 8),
            bottleneck_dim=kw.get("bottleneck_dim", 256),
        )
        for key_name in image_keys
    }

    encoder_def = EncodingWrapper(
        encoder=encoders,
        use_proprio=False,
        enable_stacking=True,
        image_keys=image_keys,
    )

    classifier = BinaryClassifier(encoder_def=encoder_def).to(device)

    sample_t = _to_torch(sample, device)
    with torch.no_grad():
        classifier(sample_t, train=False)

    return classifier


def load_classifier_func(
    key: Optional[int],
    sample: Dict,
    image_keys: List[str],
    checkpoint_path: str,
    step: Optional[int] = None,
    resnet_kwargs: Optional[Dict] = None,
    device: Optional[torch.device] = None,
) -> Callable[[Dict], np.ndarray]:
    classifier = create_classifier(key, sample, image_keys, resnet_kwargs=resnet_kwargs, device=device)
    ckpt_path = _resolve_checkpoint_path(checkpoint_path, step=step)
    payload = torch.load(ckpt_path, map_location="cpu")

    if isinstance(payload, dict) and "model" in payload:
        classifier.load_state_dict(payload["model"], strict=True)
    elif isinstance(payload, dict):
        classifier.load_state_dict(payload, strict=True)

    classifier.eval()
    device = next(classifier.parameters()).device

    def func(obs: Dict):
        obs_t = _to_torch(obs, device)
        with torch.no_grad():
            logits = classifier(obs_t, train=False)
        return logits.detach().cpu().numpy()

    return func
