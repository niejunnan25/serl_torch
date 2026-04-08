"""Shared LIBERO schema helpers for runtime and training recipes."""
from __future__ import annotations

from typing import Iterable, Tuple

LIBERO_IMAGE_SLOT_KEYS = ("image_rgb_0", "image_rgb_1", "image_rgb_2")
LIBERO_DEFAULT_IMAGE_KEYS = ("image_rgb_0", "image_rgb_1")
LIBERO_IMAGE_VIEW_TO_SLOT = {
    "image": "image_rgb_0",
    "wrist_image": "image_rgb_1",
    "image_rgb_0": "image_rgb_0",
    "image_rgb_1": "image_rgb_1",
    "image_rgb_2": "image_rgb_2",
}


def resolve_libero_image_key(image_key: str) -> str:
    key = str(image_key)
    resolved = LIBERO_IMAGE_VIEW_TO_SLOT.get(key, None)
    if resolved is None:
        raise KeyError(
            f"Unsupported LIBERO image key {key!r}. "
            f"Expected one of {sorted(LIBERO_IMAGE_VIEW_TO_SLOT)}"
        )
    return resolved


def resolve_libero_image_keys(image_keys: Iterable[str]) -> Tuple[str, ...]:
    resolved_keys = []
    seen = set()
    for image_key in image_keys:
        resolved = resolve_libero_image_key(str(image_key))
        if resolved in seen:
            continue
        seen.add(resolved)
        resolved_keys.append(resolved)
    if not resolved_keys:
        raise ValueError("At least one LIBERO image key is required")
    return tuple(resolved_keys)
