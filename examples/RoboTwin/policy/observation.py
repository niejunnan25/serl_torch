"""残差策略的观测构建：图像提取与 state/base_action 拼接。"""
from __future__ import annotations

import io
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image

from utils.constants import ALOHA_ACTION_DIM

# 与 OpenPI 一致的图像尺寸：resize_with_pad 目标，保证离线数据与环境观测统一
RESIDUAL_IMAGE_HEIGHT = 224
RESIDUAL_IMAGE_WIDTH = 224


def _resize_with_pad(img: np.ndarray, height: int, width: int) -> np.ndarray:
    """与 OpenPI 相同的 resize_with_pad：等比缩放 + 黑边居中补齐，不拉伸变形。"""
    if img.shape[-3:-1] == (height, width):
        return img
    pil_img = Image.fromarray(np.asarray(img, dtype=np.uint8))
    cur_w, cur_h = pil_img.size
    ratio = max(cur_w / width, cur_h / height)
    new_w = int(cur_w / ratio)
    new_h = int(cur_h / ratio)
    resized = pil_img.resize((new_w, new_h), resample=Image.BILINEAR)
    canvas = Image.new(resized.mode, (width, height), 0)
    pad_w = max(0, (width - new_w) // 2)
    pad_h = max(0, (height - new_h) // 2)
    canvas.paste(resized, (pad_w, pad_h))
    return np.asarray(canvas, dtype=np.uint8)


def _decode_camera_rgb(cam_dict: Dict[str, Any]) -> np.ndarray:
    """从相机字典中解码 RGB 图像，支持 numpy 数组和 JPEG bytes 两种存储格式。"""
    if "rgb" in cam_dict:
        return np.asarray(cam_dict["rgb"], dtype=np.uint8)
    if "rgb_jpeg" in cam_dict:
        from PIL import Image

        img = Image.open(io.BytesIO(cam_dict["rgb_jpeg"]))
        return np.asarray(img, dtype=np.uint8)
    raise KeyError(f"Camera dict has no 'rgb' or 'rgb_jpeg' key. Keys: {list(cam_dict.keys())}")


def _extract_residual_images(obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """从原始观测中提取三路 RGB 图像。"""
    return {
        "cam_high": _decode_camera_rgb(obs["observation"]["head_camera"]),
        "cam_left_wrist": _decode_camera_rgb(obs["observation"]["left_camera"]),
        "cam_right_wrist": _decode_camera_rgb(obs["observation"]["right_camera"]),
    }


def build_residual_step_obs(
    obs: Dict[str, Any],
    base_action: np.ndarray,
    image_keys: Tuple[str, ...],
    stack_horizon: int = 1,
) -> Dict[str, np.ndarray]:
    """
    构建"逐步残差策略"的输入观测：当前图像 + 当前状态 + 当前将执行的 base action。

    返回字典包含：

    - 各 image_keys 对应的图像（增加一维时间/堆叠维）
    - ``"state"``：当前关节/动作向量与当前步 base_action 的拼接

    """
    state = np.asarray(obs["joint_action"]["vector"], dtype=np.float32)
    base_action = np.asarray(base_action, dtype=np.float32).reshape(-1)
    if base_action.shape[0] != ALOHA_ACTION_DIM:
        raise ValueError(
            f"Unexpected base action shape: {base_action.shape}, expected ({ALOHA_ACTION_DIM},)"
        )

    fused_state = np.concatenate([state, base_action], axis=-1).astype(np.float32)

    images_all = _extract_residual_images(obs)
    missing_keys = [key for key in image_keys if key not in images_all]
    if missing_keys:
        raise KeyError(
            f"Unsupported image key(s): {missing_keys}. "
            f"Available keys: {list(images_all.keys())}"
        )
    images = {key: images_all[key] for key in image_keys}
    if stack_horizon != 1:
        raise ValueError(f"Only stack_horizon=1 is currently supported, got {stack_horizon}")

    # 统一 resize 到 224x224（与 OpenPI 一致），保证离线数据与环境观测格式统一
    images = {
        key: _resize_with_pad(img, RESIDUAL_IMAGE_HEIGHT, RESIDUAL_IMAGE_WIDTH)
        for key, img in images.items()
    }
    stacked = {key: np.expand_dims(img, axis=0) for key, img in images.items()}
    stacked["state"] = np.expand_dims(fused_state, axis=0)
    return stacked


def build_residual_chunk_obs(
    obs: Dict[str, Any],
    base_chunk: np.ndarray,
    image_keys: Tuple[str, ...],
    stack_horizon: int = 1,
) -> Dict[str, np.ndarray]:
    """兼容旧接口：输入整段 base chunk 时，默认使用第 0 步动作构建逐步残差观测。"""
    chunk = np.asarray(base_chunk, dtype=np.float32)
    if chunk.ndim != 2 or chunk.shape[1] != ALOHA_ACTION_DIM:
        raise ValueError(f"Unexpected base chunk shape: {chunk.shape}")
    return build_residual_step_obs(
        obs=obs,
        base_action=chunk[0],
        image_keys=image_keys,
        stack_horizon=stack_horizon,
    )


def build_residual_obs(obs: Dict[str, Any], base_action: np.ndarray, progress: float) -> np.ndarray:
    """构建单步残差策略的输入状态：state + base_action + progress 标量。"""
    state = np.asarray(obs["joint_action"]["vector"], dtype=np.float32)
    base_action = np.asarray(base_action, dtype=np.float32)
    progress_arr = np.asarray([progress], dtype=np.float32)
    return np.concatenate([state, base_action, progress_arr], axis=-1)
