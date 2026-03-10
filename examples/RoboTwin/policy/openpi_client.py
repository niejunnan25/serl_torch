"""OpenPI 基策略客户端：观测编码、WebSocket 推理、耗时提取。"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from policy.observation import _decode_camera_rgb


# ---------------------------------------------------------------------------
# 观测编码
# ---------------------------------------------------------------------------


def encode_obs_for_openpi(obs: Dict[str, Any], prompt: str) -> Dict[str, Any]:
    """将原始观测编码为 OpenPI 服务端要求的格式。支持 rgb 或 rgb_jpeg 两种存储格式。"""
    input_rgb_arr = [
        _decode_camera_rgb(obs["observation"]["head_camera"]),
        _decode_camera_rgb(obs["observation"]["right_camera"]),
        _decode_camera_rgb(obs["observation"]["left_camera"]),
    ]
    puppet_arm = np.asarray(obs["joint_action"]["vector"], dtype=np.float32)

    img_front, img_right, img_left = input_rgb_arr
    img_front = np.transpose(img_front, (2, 0, 1))
    img_right = np.transpose(img_right, (2, 0, 1))
    img_left = np.transpose(img_left, (2, 0, 1))

    return {
        "state": puppet_arm,
        "images": {
            "cam_high": img_front,
            "cam_left_wrist": img_left,
            "cam_right_wrist": img_right,
        },
        "prompt": prompt,
    }


# ---------------------------------------------------------------------------
# 推理耗时提取
# ---------------------------------------------------------------------------


def maybe_get_policy_infer_ms(pred: Dict[str, Any]) -> Optional[float]:
    """从 OpenPI 推理返回的 pred 中安全取出策略推理耗时（毫秒）。"""
    if "policy_timing" in pred and isinstance(pred["policy_timing"], dict):
        ms = pred["policy_timing"].get("infer_ms")
        if ms is not None:
            return float(ms)
    return None


def maybe_get_server_infer_ms(pred: Dict[str, Any]) -> Optional[float]:
    """从 OpenPI 推理返回的 pred 中安全取出服务端推理耗时（毫秒）。"""
    if "server_timing" in pred and isinstance(pred["server_timing"], dict):
        ms = pred["server_timing"].get("infer_ms")
        if ms is not None:
            return float(ms)
    return None


# ---------------------------------------------------------------------------
# OpenPI 客户端
# ---------------------------------------------------------------------------


class OpenPIChunkClient:
    """
    通过 WebSocket 调用 OpenPI 策略服务，按观测和 prompt 获取基策略动作块及推理耗时信息。
    """

    def __init__(self, host: str, port: int, logger: Optional[logging.Logger] = None):
        from openpi_client.websocket_client_policy import WebsocketClientPolicy

        self._client = WebsocketClientPolicy(host=host, port=port)
        self._logger = logger or logging.getLogger(__name__)

    def infer_chunk(self, obs: Dict[str, Any], prompt: str) -> Tuple[np.ndarray, Dict[str, Optional[float]]]:
        """编码观测与 prompt 后请求推理，返回动作块数组和耗时字典。"""
        send_data = encode_obs_for_openpi(obs, prompt)

        start = time.time()
        pred = self._client.infer(send_data)
        e2e_ms = (time.time() - start) * 1000.0

        chunk = np.asarray(pred["actions"], dtype=np.float32)
        info = {
            "e2e_ms": float(e2e_ms),
            "policy_ms": maybe_get_policy_infer_ms(pred),
            "server_ms": maybe_get_server_infer_ms(pred),
        }
        return chunk, info
