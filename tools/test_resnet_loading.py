#!/usr/bin/env python3
"""Verify that ResNetEncoder loads correctly from local pretrained_models/."""

import sys
import os

os.chdir(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, "serl_launcher")

import torch
from serl_launcher.vision.resnet_v1 import ResNetEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
dummy_img = torch.randint(0, 255, (2, 128, 128, 3), dtype=torch.uint8, device=device)

# ------------------------------------------------------------------
# Test 1: pretrained + frozen (typical training config)
# ------------------------------------------------------------------
print("\n=== Test 1: pretrained=True, freeze=True (local path, resnet-18) ===")
enc1 = ResNetEncoder(
    model_name="pretrained_models/microsoft--resnet-18",
    pretrained=True,
    freeze_backbone=True,
    pooling_method="spatial_learned_embeddings",
    num_spatial_blocks=8,
    bottleneck_dim=256,
).to(device)
out1 = enc1(dummy_img, train=False)
print(f"  Output shape: {out1.shape}")
print(f"  Backbone frozen: {not any(p.requires_grad for p in enc1.backbone.parameters())}")

out1_train = enc1(dummy_img, train=True)
print(f"  Bottleneck trainable: {enc1.bottleneck.weight.requires_grad}")
print(f"  Spatial pool trainable: {any(p.requires_grad for p in enc1.spatial_pool.parameters())}")

# ------------------------------------------------------------------
# Test 2: pretrained + NOT frozen
# ------------------------------------------------------------------
print("\n=== Test 2: pretrained=True, freeze=False (resnet-18) ===")
enc2 = ResNetEncoder(
    model_name="pretrained_models/microsoft--resnet-18",
    pretrained=True,
    freeze_backbone=False,
    pooling_method="spatial_learned_embeddings",
    num_spatial_blocks=8,
    bottleneck_dim=256,
).to(device)
out2 = enc2(dummy_img, train=True)
print(f"  Output shape: {out2.shape}")
print(f"  Backbone trainable: {any(p.requires_grad for p in enc2.backbone.parameters())}")

# ------------------------------------------------------------------
# Test 3: NOT pretrained (random init)
# ------------------------------------------------------------------
print("\n=== Test 3: pretrained=False (random init, resnet-18) ===")
enc3 = ResNetEncoder(
    model_name="pretrained_models/microsoft--resnet-18",
    pretrained=False,
    freeze_backbone=False,
    pooling_method="spatial_learned_embeddings",
    num_spatial_blocks=8,
    bottleneck_dim=256,
).to(device)
out3 = enc3(dummy_img, train=True)
print(f"  Output shape: {out3.shape}")

# ------------------------------------------------------------------
# Test 4: ResNet-50
# ------------------------------------------------------------------
print("\n=== Test 4: ResNet-50 pretrained + frozen ===")
enc4 = ResNetEncoder(
    model_name="pretrained_models/microsoft--resnet-50",
    pretrained=True,
    freeze_backbone=True,
    pooling_method="spatial_learned_embeddings",
    num_spatial_blocks=8,
    bottleneck_dim=256,
).to(device)
out4 = enc4(dummy_img, train=False)
print(f"  Output shape: {out4.shape}")

# ------------------------------------------------------------------
# Test 5: verify pretrained weights != random init
# ------------------------------------------------------------------
print("\n=== Test 5: verify pretrained weights differ from random ===")
enc_pt = ResNetEncoder(
    model_name="pretrained_models/microsoft--resnet-18",
    pretrained=True, freeze_backbone=True, pooling_method="none",
)
enc_rng = ResNetEncoder(
    model_name="pretrained_models/microsoft--resnet-18",
    pretrained=False, freeze_backbone=True, pooling_method="none",
)
w_pt = list(enc_pt.backbone.parameters())[0].data
w_rng = list(enc_rng.backbone.parameters())[0].data
diff = (w_pt - w_rng).abs().mean().item()
print(f"  Mean abs diff (pretrained vs random): {diff:.6f}")
assert diff > 0.001, "Pretrained weights should differ from random!"
print(f"  Weights are different: True")

# ------------------------------------------------------------------
# Test 6: gradient flows correctly
# ------------------------------------------------------------------
print("\n=== Test 6: gradient flow check ===")
enc6 = ResNetEncoder(
    model_name="pretrained_models/microsoft--resnet-18",
    pretrained=True,
    freeze_backbone=True,
    pooling_method="spatial_learned_embeddings",
    num_spatial_blocks=8,
    bottleneck_dim=256,
).to(device)
out6 = enc6(dummy_img.float(), train=True)
loss = out6.sum()
loss.backward()

backbone_grads = [p.grad for p in enc6.backbone.parameters() if p.grad is not None]
pool_grads = [p.grad for p in enc6.spatial_pool.parameters() if p.grad is not None]
bn_grad = enc6.bottleneck.weight.grad

print(f"  Backbone grads: {len(backbone_grads)} (should be 0)")
print(f"  Spatial pool grads: {len(pool_grads)} (should be > 0)")
print(f"  Bottleneck has grad: {bn_grad is not None} (should be True)")
assert len(backbone_grads) == 0, "Frozen backbone should have no gradients!"
assert len(pool_grads) > 0, "Spatial pool should have gradients!"
assert bn_grad is not None, "Bottleneck should have gradients!"

print("\n" + "=" * 50)
print("ALL TESTS PASSED")
print("=" * 50)
