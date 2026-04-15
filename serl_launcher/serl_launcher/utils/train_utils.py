import numpy as np
import torch


def concat_batches(offline_batch, online_batch, axis=1):
    if isinstance(offline_batch, dict):
        return {
            k: concat_batches(offline_batch[k], online_batch[k], axis=axis)
            for k in offline_batch
        }

    if isinstance(offline_batch, torch.Tensor) or isinstance(online_batch, torch.Tensor):
        a = offline_batch if isinstance(offline_batch, torch.Tensor) else torch.as_tensor(offline_batch)
        b = online_batch if isinstance(online_batch, torch.Tensor) else torch.as_tensor(online_batch)
        return torch.cat((a, b), dim=axis)

    return np.concatenate((offline_batch, online_batch), axis=axis)


def _unpack(batch):
    """Unpacks packed observation tensors into obs/next_obs views."""

    observations = dict(batch["observations"])
    next_observations = dict(batch["next_observations"])

    for pixel_key, obs_value in observations.items():
        if pixel_key in next_observations:
            continue
        if obs_value.shape[1] < 2:
            continue

        obs_pixels = obs_value[:, :-1, ...]
        next_obs_pixels = obs_value[:, 1:, ...]
        observations[pixel_key] = obs_pixels
        next_observations[pixel_key] = next_obs_pixels

    new_batch = dict(batch)
    new_batch["observations"] = observations
    new_batch["next_observations"] = next_observations
    return new_batch
