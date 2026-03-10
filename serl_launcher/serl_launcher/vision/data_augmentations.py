from typing import Optional

import torch
import torch.nn.functional as F


def _as_generator(rng=None, device="cpu"):
    if isinstance(rng, torch.Generator):
        return rng
    g = torch.Generator(device=device)
    if isinstance(rng, int):
        g.manual_seed(int(rng))
    else:
        g.seed()
    return g


def _to_tensor(x):
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x)


def random_crop(img, rng=None, *, padding: int):
    img = _to_tensor(img)
    generator = _as_generator(rng, device=img.device.type)

    if img.ndim != 3:
        raise ValueError(f"random_crop expects HWC tensor, got shape {tuple(img.shape)}")

    h, w, c = img.shape
    padded = F.pad(
        img.permute(2, 0, 1).unsqueeze(0),
        pad=(padding, padding, padding, padding),
        mode="replicate",
    ).squeeze(0).permute(1, 2, 0)

    y = int(torch.randint(0, 2 * padding + 1, (1,), generator=generator, device=img.device))
    x = int(torch.randint(0, 2 * padding + 1, (1,), generator=generator, device=img.device))
    return padded[y : y + h, x : x + w, :]


def batched_random_crop(img, rng=None, *, padding: int, num_batch_dims: int = 1):
    img = _to_tensor(img)
    original_shape = img.shape
    flat = img.reshape(-1, *img.shape[num_batch_dims:])
    generator = _as_generator(rng, device=img.device.type)

    crops = []
    for i in range(flat.shape[0]):
        sample_seed = int(torch.randint(0, 2**31 - 1, (1,), generator=generator, device=img.device))
        sample_generator = _as_generator(sample_seed, device=img.device.type)
        crops.append(random_crop(flat[i], sample_generator, padding=padding))

    out = torch.stack(crops, dim=0).reshape(original_shape)
    return out


def random_flip(image, rng=None):
    image = _to_tensor(image)
    generator = _as_generator(rng, device=image.device.type)
    should_flip = bool(torch.rand(1, generator=generator, device=image.device) <= 0.5)
    if should_flip:
        return torch.flip(image, dims=(-2,))
    return image


def color_transform(
    image,
    rng=None,
    *,
    brightness=0.0,
    contrast=0.0,
    saturation=0.0,
    hue=0.0,
    to_grayscale_prob=0.0,
    color_jitter_prob=1.0,
    apply_prob=1.0,
    shuffle=True,
):
    del hue, shuffle
    image = _to_tensor(image).float()
    generator = _as_generator(rng, device=image.device.type)

    if float(torch.rand(1, generator=generator, device=image.device)) > apply_prob:
        return image
    if float(torch.rand(1, generator=generator, device=image.device)) > color_jitter_prob:
        return image

    out = image
    if brightness > 0:
        delta = torch.empty(1, device=image.device).uniform_(-brightness, brightness, generator=generator)
        out = out + delta

    if contrast > 0:
        factor = torch.empty(1, device=image.device).uniform_(1 - contrast, 1 + contrast, generator=generator)
        mean = out.mean(dim=(-3, -2), keepdim=True)
        out = (out - mean) * factor + mean

    if saturation > 0 and out.shape[-1] == 3:
        factor = torch.empty(1, device=image.device).uniform_(1 - saturation, 1 + saturation, generator=generator)
        gray = out.mean(dim=-1, keepdim=True)
        out = (out - gray) * factor + gray

    if float(torch.rand(1, generator=generator, device=image.device)) <= to_grayscale_prob and out.shape[-1] == 3:
        gray = out.mean(dim=-1, keepdim=True)
        out = gray.repeat_interleave(3, dim=-1)

    return torch.clamp(out, 0.0, 1.0)


def gaussian_blur(
    image,
    rng=None,
    *,
    blur_divider=10.0,
    sigma_min=0.1,
    sigma_max=2.0,
    apply_prob=1.0,
):
    del rng, blur_divider, sigma_min, sigma_max
    image = _to_tensor(image)
    if apply_prob < 1.0 and torch.rand(1, device=image.device) > apply_prob:
        return image
    return image


def solarize(image, rng=None, *, threshold=0.5, apply_prob=1.0):
    del rng
    image = _to_tensor(image)
    if apply_prob < 1.0 and torch.rand(1, device=image.device) > apply_prob:
        return image
    return torch.where(image < threshold, image, 1.0 - image)
