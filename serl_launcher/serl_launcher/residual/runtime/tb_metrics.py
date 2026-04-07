"""TensorBoard metric helpers for LIBERO residual training."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def _log_info_scalars(
    tb_writer,
    info: Dict[str, Any],
    global_step: int,
    pairs: Tuple[Tuple[str, str], ...],
) -> None:
    for tb_key, info_key in pairs:
        if info_key in info and info[info_key] is not None:
            tb_writer.add_scalar(tb_key, float(info[info_key]), global_step)


def _new_tb_step_window() -> Dict[str, List[Any]]:
    return {
        "reward": [],
        "alpha": [],
        "gate_prob": [],
        "gate_on": [],
        "gate_prob_decision": [],
        "gate_on_decision": [],
        "delta_norm": [],
        "policy_raw_norm": [],
        "policy_applied_norm": [],
        "base_norm": [],
        "final_norm": [],
        "residual_actions_raw": [],
        "residual_actions_applied": [],
        "delta_actions": [],
        "infer_e2e_ms": [],
        "infer_policy_ms": [],
        "infer_server_ms": [],
    }


def _append_tb_step_window(
    step_window: Dict[str, List[Any]],
    *,
    reward: float,
    alpha: float,
    gate_prob: float,
    gate_on: bool,
    residual_action_raw: np.ndarray,
    residual_action_applied: np.ndarray,
    delta_action: np.ndarray,
    base_action: np.ndarray,
    final_action: np.ndarray,
    infer_info: Dict[str, Any],
    replan_point: bool,
) -> None:
    residual_action_raw = np.asarray(residual_action_raw, dtype=np.float32).reshape(-1)
    residual_action_applied = np.asarray(
        residual_action_applied, dtype=np.float32
    ).reshape(-1)
    delta_action = np.asarray(delta_action, dtype=np.float32).reshape(-1)
    base_action = np.asarray(base_action, dtype=np.float32).reshape(-1)
    final_action = np.asarray(final_action, dtype=np.float32).reshape(-1)
    step_window["reward"].append(float(reward))
    step_window["alpha"].append(float(alpha))
    step_window["gate_prob"].append(float(gate_prob))
    step_window["gate_on"].append(float(bool(gate_on)))
    if replan_point:
        step_window["gate_prob_decision"].append(float(gate_prob))
        step_window["gate_on_decision"].append(float(bool(gate_on)))
    step_window["delta_norm"].append(float(np.linalg.norm(delta_action)))
    step_window["policy_raw_norm"].append(float(np.linalg.norm(residual_action_raw)))
    step_window["policy_applied_norm"].append(
        float(np.linalg.norm(residual_action_applied))
    )
    step_window["base_norm"].append(float(np.linalg.norm(base_action)))
    step_window["final_norm"].append(float(np.linalg.norm(final_action)))
    step_window["residual_actions_raw"].append(residual_action_raw.copy())
    step_window["residual_actions_applied"].append(residual_action_applied.copy())
    step_window["delta_actions"].append(delta_action.copy())

    if replan_point:
        for key, store_key in (
            ("e2e_ms", "infer_e2e_ms"),
            ("policy_ms", "infer_policy_ms"),
            ("server_ms", "infer_server_ms"),
        ):
            value = infer_info.get(key, None)
            if value is not None:
                step_window[store_key].append(float(value))


def _flush_tb_step_window(
    tb_writer,
    *,
    step_window: Dict[str, List[Any]],
    global_env_step: int,
    control_indices: np.ndarray,
    histogram: bool = False,
) -> None:
    if not step_window["reward"]:
        return

    scalar_lists = (
        ("step/reward", "reward"),
        ("step/reward_nonzero_rate", "reward"),
        ("step/alpha", "alpha"),
        ("step/epsilon_gate_prob", "gate_prob"),
        ("step/epsilon_gate_on_rate", "gate_on"),
        ("step/epsilon_gate_prob_decision", "gate_prob_decision"),
        ("step/epsilon_gate_on_decision_rate", "gate_on_decision"),
        ("step/residual_action_magnitude", "delta_norm"),
        ("step/residual_policy_action_magnitude", "policy_raw_norm"),
        ("step/residual_policy_action_applied_magnitude", "policy_applied_norm"),
        ("step/base_action_magnitude", "base_norm"),
        ("step/final_action_magnitude", "final_norm"),
        ("step/infer_e2e_ms", "infer_e2e_ms"),
        ("step/infer_policy_ms", "infer_policy_ms"),
        ("step/infer_server_ms", "infer_server_ms"),
    )
    for tb_key, value_key in scalar_lists:
        values = step_window[value_key]
        if not values:
            continue
        if tb_key == "step/reward_nonzero_rate":
            metric_value = float(np.mean(np.asarray(values, dtype=np.float32) != 0.0))
        else:
            metric_value = float(np.mean(np.asarray(values, dtype=np.float32)))
        tb_writer.add_scalar(tb_key, metric_value, global_env_step)

    residual_actions_raw = np.asarray(
        step_window["residual_actions_raw"], dtype=np.float32
    )
    residual_actions_applied = np.asarray(
        step_window["residual_actions_applied"], dtype=np.float32
    )
    delta_actions = np.asarray(step_window["delta_actions"], dtype=np.float32)
    if residual_actions_raw.size > 0:
        residual_abs = np.abs(residual_actions_raw)
        tb_writer.add_scalar(
            "step/residual_policy_action_abs_mean",
            float(np.mean(residual_abs)),
            global_env_step,
        )
        tb_writer.add_scalar(
            "step/residual_policy_action_abs_p95",
            float(np.percentile(residual_abs, 95)),
            global_env_step,
        )
        tb_writer.add_scalar(
            "step/residual_policy_action_saturation_frac",
            float(np.mean(residual_abs >= 0.999)),
            global_env_step,
        )
        for dim_idx, control_idx in enumerate(
            np.asarray(control_indices, dtype=np.int64).tolist()
        ):
            dim_abs = residual_abs[:, dim_idx]
            tb_writer.add_scalar(
                f"step/residual_policy_action_abs_dim_{int(control_idx)}",
                float(np.mean(dim_abs)),
                global_env_step,
            )
            tb_writer.add_scalar(
                f"step/residual_policy_action_sat_dim_{int(control_idx)}",
                float(np.mean(dim_abs >= 0.999)),
                global_env_step,
            )

    if residual_actions_applied.size > 0:
        residual_applied_abs = np.abs(residual_actions_applied)
        tb_writer.add_scalar(
            "step/residual_policy_action_applied_abs_mean",
            float(np.mean(residual_applied_abs)),
            global_env_step,
        )
        tb_writer.add_scalar(
            "step/residual_policy_action_applied_abs_p95",
            float(np.percentile(residual_applied_abs, 95)),
            global_env_step,
        )
        tb_writer.add_scalar(
            "step/residual_policy_action_applied_saturation_frac",
            float(np.mean(residual_applied_abs >= 0.999)),
            global_env_step,
        )

    if delta_actions.size > 0:
        controlled_delta = delta_actions[:, np.asarray(control_indices, dtype=np.int64)]
        controlled_delta_abs = np.abs(controlled_delta)
        tb_writer.add_scalar(
            "step/residual_delta_abs_mean",
            float(np.mean(controlled_delta_abs)),
            global_env_step,
        )
        tb_writer.add_scalar(
            "step/residual_delta_abs_p95",
            float(np.percentile(controlled_delta_abs, 95)),
            global_env_step,
        )
        for dim_idx, control_idx in enumerate(
            np.asarray(control_indices, dtype=np.int64).tolist()
        ):
            dim_abs = controlled_delta_abs[:, dim_idx]
            tb_writer.add_scalar(
                f"step/residual_delta_abs_dim_{int(control_idx)}",
                float(np.mean(dim_abs)),
                global_env_step,
            )
        base_norm = np.mean(np.asarray(step_window["base_norm"], dtype=np.float32))
        delta_norm = np.mean(np.asarray(step_window["delta_norm"], dtype=np.float32))
        tb_writer.add_scalar(
            "step/residual_to_base_ratio",
            float(delta_norm / max(base_norm, 1e-6)),
            global_env_step,
        )

    if histogram and residual_actions_raw.size > 0:
        tb_writer.add_histogram(
            "hist/residual_policy_action",
            residual_actions_raw.reshape(-1),
            global_env_step,
        )
    if histogram and residual_actions_applied.size > 0:
        tb_writer.add_histogram(
            "hist/residual_policy_action_applied",
            residual_actions_applied.reshape(-1),
            global_env_step,
        )
    if histogram and delta_actions.size > 0:
        controlled_delta = delta_actions[:, np.asarray(control_indices, dtype=np.int64)]
        tb_writer.add_histogram(
            "hist/residual_delta_action", controlled_delta.reshape(-1), global_env_step
        )

    for values in step_window.values():
        values.clear()


def _log_update_metrics(
    tb_writer, update_info: Dict[str, Any], global_env_step: int
) -> None:
    _log_info_scalars(
        tb_writer,
        update_info,
        global_env_step,
        (
            ("critic/loss", "critic_loss"),
            ("critic/td_loss", "critic_td_loss"),
            ("critic/cql_penalty", "critic_cql_penalty"),
            ("critic/predicted_qs", "predicted_qs"),
            ("critic/target_qs", "target_qs"),
            ("critic/predicted_q_min", "predicted_q_min"),
            ("critic/predicted_q_max", "predicted_q_max"),
            ("critic/predicted_q_std", "predicted_q_std"),
            ("critic/predicted_q_gap", "predicted_q_gap"),
            ("actor/loss", "actor_loss"),
            ("actor/entropy", "entropy"),
            ("actor/log_prob", "log_prob"),
            ("actor/temperature", "temperature"),
            ("actor/temperature_loss", "temperature_loss"),
            ("actor/temperature_entropy", "temperature_entropy"),
            ("actor/target_entropy", "target_entropy"),
            ("actor/target_entropy_abs", "target_entropy_abs"),
            ("actor/target_entropy_gap", "target_entropy_gap"),
            ("actor/temperature_constraint_gap", "temperature_constraint_gap"),
            ("actor/predicted_q", "actor_predicted_q"),
            ("actor/predicted_q_min", "actor_predicted_q_min"),
            ("actor/predicted_q_std", "actor_predicted_q_std"),
            ("data/online_batch_size", "online_batch_size"),
            ("data/offline_batch_size", "offline_batch_size"),
            ("data/offline_fraction", "offline_fraction"),
            ("optim/actor_lr", "actor_lr"),
            ("optim/critic_lr", "critic_lr"),
            ("optim/temperature_lr", "temperature_lr"),
        ),
    )
