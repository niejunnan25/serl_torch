from types import SimpleNamespace

import pytest
import torch
from torch import nn

from serl_launcher.common.encoding import EncodingWrapper
from serl_launcher.vision.resnet_v1 import ResNetEncoder


class DummyBackbone(nn.Module):
    def forward(self, pixel_values, return_dict=True):
        del return_dict
        return SimpleNamespace(last_hidden_state=pixel_values * 0.5 + 1.0)


def _make_resnet_encoder(backbone: nn.Module) -> ResNetEncoder:
    return ResNetEncoder(
        backbone=backbone,
        freeze_backbone=True,
        pooling_method="avg",
        bottleneck_dim=None,
    )


def _make_wrapper(*, fuse_views, shared_backbone: bool = True) -> EncodingWrapper:
    first_backbone = DummyBackbone()
    second_backbone = first_backbone if shared_backbone else DummyBackbone()
    return EncodingWrapper(
        encoder={
            "image0": _make_resnet_encoder(first_backbone),
            "image1": _make_resnet_encoder(second_backbone),
        },
        use_proprio=False,
        image_keys=("image0", "image1"),
        fuse_views=fuse_views,
    )


def test_fused_views_match_loop_for_batched_images():
    torch.manual_seed(0)
    observations = {
        "image0": torch.randint(0, 256, (4, 8, 8, 3), dtype=torch.uint8),
        "image1": torch.randint(0, 256, (4, 8, 8, 3), dtype=torch.uint8),
    }

    loop_wrapper = _make_wrapper(fuse_views=False)
    fused_wrapper = _make_wrapper(fuse_views=True)

    expected = loop_wrapper(observations, train=False)
    actual = fused_wrapper(observations, train=False)

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected)


def test_fused_views_match_loop_for_unbatched_images():
    torch.manual_seed(0)
    observations = {
        "image0": torch.randint(0, 256, (8, 8, 3), dtype=torch.uint8),
        "image1": torch.randint(0, 256, (8, 8, 3), dtype=torch.uint8),
    }

    loop_wrapper = _make_wrapper(fuse_views=False)
    fused_wrapper = _make_wrapper(fuse_views=True)

    expected = loop_wrapper(observations, train=False)
    actual = fused_wrapper(observations, train=False)

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected)


def test_fuse_views_true_requires_shared_backbone():
    with pytest.raises(ValueError, match="does not share backbone"):
        _make_wrapper(fuse_views=True, shared_backbone=False)


def test_fuse_views_true_requires_frozen_backbone():
    backbone = DummyBackbone()
    with pytest.raises(ValueError, match="freeze_backbone=false"):
        EncodingWrapper(
            encoder={
                "image0": ResNetEncoder(
                    backbone=backbone,
                    freeze_backbone=False,
                    pooling_method="avg",
                    bottleneck_dim=None,
                ),
                "image1": ResNetEncoder(
                    backbone=backbone,
                    freeze_backbone=True,
                    pooling_method="avg",
                    bottleneck_dim=None,
                ),
            },
            use_proprio=False,
            image_keys=("image0", "image1"),
            fuse_views=True,
        )


def test_fuse_views_auto_falls_back_when_backbone_is_not_shared():
    torch.manual_seed(0)
    observations = {
        "image0": torch.randint(0, 256, (4, 8, 8, 3), dtype=torch.uint8),
        "image1": torch.randint(0, 256, (4, 8, 8, 3), dtype=torch.uint8),
    }

    loop_wrapper = _make_wrapper(fuse_views=False, shared_backbone=False)
    auto_wrapper = _make_wrapper(fuse_views="auto", shared_backbone=False)

    expected = loop_wrapper(observations, train=False)
    actual = auto_wrapper(observations, train=False)

    torch.testing.assert_close(actual, expected)


def test_fuse_views_true_rejects_dynamic_shape_mismatch():
    torch.manual_seed(0)
    observations = {
        "image0": torch.randint(0, 256, (4, 8, 8, 3), dtype=torch.uint8),
        "image1": torch.randint(0, 256, (4, 10, 8, 3), dtype=torch.uint8),
    }
    wrapper = _make_wrapper(fuse_views=True)

    with pytest.raises(ValueError, match="shape"):
        wrapper(observations, train=False)


def test_fuse_views_auto_falls_back_on_dynamic_shape_mismatch():
    torch.manual_seed(0)
    observations = {
        "image0": torch.randint(0, 256, (4, 8, 8, 3), dtype=torch.uint8),
        "image1": torch.randint(0, 256, (4, 10, 8, 3), dtype=torch.uint8),
    }

    loop_wrapper = _make_wrapper(fuse_views=False)
    auto_wrapper = _make_wrapper(fuse_views="auto")

    expected = loop_wrapper(observations, train=False)
    actual = auto_wrapper(observations, train=False)

    torch.testing.assert_close(actual, expected)
