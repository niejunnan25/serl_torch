#!/usr/bin/env python3
"""Benchmark msgpack vs pickle serialization on realistic serl_torch payloads.

Payloads are derived from the actual data structures found in:
  - trainer_transport.py  (_serialize_message / _deserialize_message)
  - batch_ops.py          (pack_transition_batch → nested_stack)
  - training_payloads.py  (RolloutStatsPayload)
  - checkpoint_codec.py   (snapshot_actor_network_payload)
  - transition_assembly.py (per-step transition dict)
  - observation.py         (build_chunk_residual_obs)

Real shapes:
  - image:          (B, 224, 224, 3) uint8  — from LIBERO rendering
  - wrist_image:    (B, 224, 224, 3) uint8
  - robot_state:    (B, 8)          float32  — LIBERO_STATE_DIM=8
  - base_action_chunk: (B, 5, 7)    float32  — chunk_horizon=5 × action_dim=7
  - base_action:    (B, 1, 7)       float32
  - alpha:          (B, 1)          float32
  - residual_action:(B, 7)          float32
  - reward/mask:    (B,)            float32
  - done:           (B,)            bool

Weight payload (from checkpoint_codec.snapshot_actor_network_payload):
  - ResNet-18 state_dict: ~44 MB of fp32 tensors in nested dict
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


# ── Payload builders: match real serl_torch data structures ──────────────

def _make_image(batch: int, h: int, w: int) -> np.ndarray:
    return np.random.randint(0, 256, (batch, h, w, 3), dtype=np.uint8)


def build_transition_payload(batch: int) -> dict[str, Any]:
    """A packed batch of transitions, matching pack_transition_batch output."""
    chunk_horizon = 5
    action_dim = 7
    return {
        "observations": {
            "image": _make_image(batch, 224, 224),
            "wrist_image": _make_image(batch, 224, 224),
            "robot_state": np.random.randn(batch, 8).astype(np.float32),
            "base_action_chunk": np.random.randn(batch, chunk_horizon, action_dim).astype(np.float32),
            "base_action": np.random.randn(batch, 1, action_dim).astype(np.float32),
            "alpha": np.full((batch, 1), 0.1, dtype=np.float32),
        },
        "actions": np.random.randn(batch, action_dim).astype(np.float32),
        "next_observations": {
            "image": _make_image(batch, 224, 224),
            "wrist_image": _make_image(batch, 224, 224),
            "robot_state": np.random.randn(batch, 8).astype(np.float32),
            "base_action_chunk": np.random.randn(batch, chunk_horizon, action_dim).astype(np.float32),
            "base_action": np.random.randn(batch, 1, action_dim).astype(np.float32),
            "alpha": np.full((batch, 1), 0.1, dtype=np.float32),
        },
        "rewards": np.random.randn(batch).astype(np.float32),
        "masks": np.ones(batch, dtype=np.float32),
        "dones": np.zeros(batch, dtype=bool),
    }


def build_weight_payload() -> dict[str, Any]:
    """Simulate an actor network state_dict (ResNet-18 scale, ~44 MB)."""
    params = {}
    # ResNet-18 has ~11M parameters in ~100 tensors (weights + biases + BN params)
    total_params = 0
    for layer_idx in range(4):
        blocks = [2, 2, 2, 2][layer_idx]
        in_ch = [64, 64, 128, 256][layer_idx]
        for block in range(blocks):
            for conv in range(2):
                out_ch = [64, 128, 256, 512][layer_idx]
                w = np.random.randn(out_ch, in_ch, 3, 3).astype(np.float32)
                b = np.random.randn(out_ch).astype(np.float32)
                params[f"layer{layer_idx}.block{block}.conv{conv}.weight"] = w
                params[f"layer{layer_idx}.block{block}.conv{conv}.bias"] = b
                total_params += w.size + b.size
                in_ch = out_ch
        # downsample
        out_ch = [64, 128, 256, 512][layer_idx]
        params[f"layer{layer_idx}.downsample.weight"] = np.random.randn(
            out_ch, [64, 64, 128, 256][layer_idx], 1, 1
        ).astype(np.float32)
    # FC layer
    fc_w = np.random.randn(256, 512).astype(np.float32)
    fc_b = np.random.randn(256).astype(np.float32)
    params["fc.weight"] = fc_w
    params["fc.bias"] = fc_b
    total_params += fc_w.size + fc_b.size
    return {
        "step": 1000,
        "params": {
            "actor": params,
        },
        "_total_param_count": total_params,
    }


def build_stats_payload() -> dict[str, Any]:
    """A RolloutStatsPayload as sent via trainer transport."""
    return {
        "env_steps": 50000,
        "rollout": {
            "episode_id": 1024,
            "episode_steps": 238,
            "episode_return": 15.7,
            "success": True,
            "cumulative_success_rate": 0.845,
            "recent_success_rate_20": 0.90,
            "init_episode_idx": 1000,
        },
        "env_info": {
            "task_suite": "libero_spatial",
            "task_id": 4,
            "mean_episode_steps": 238.3,
            "total_episodes": 23,
        },
        "residual": {
            "residual_action_norm_mean": 0.023,
            "residual_action_norm_max": 0.187,
            "entropy_per_dim": 0.59,
            "alpha": 0.1,
        },
    }


# ── Serialization adapters ──────────────────────────────────────────────

import lz4.frame as _lz4_frame  # noqa: E402


def serialize_pickle(obj: Any) -> bytes:
    raw = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    return _lz4_frame.compress(raw)


def deserialize_pickle(data: bytes) -> Any:
    raw = _lz4_frame.decompress(data)
    return pickle.loads(raw)


def serialize_pickle_raw(obj: Any) -> bytes:
    """pickle WITHOUT compression — for ablation."""
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


def deserialize_pickle_raw(data: bytes) -> Any:
    return pickle.loads(data)


# msgpack + msgpack-numpy
try:
    import msgpack
    import msgpack_numpy

    msgpack_numpy.patch()
    HAS_MSGPACK_NUMPY = True
except ImportError:
    HAS_MSGPACK_NUMPY = False


def serialize_msgpack(obj: Any) -> bytes:
    return msgpack.packb(obj, use_bin_type=True)


def deserialize_msgpack(data: bytes) -> Any:
    return msgpack.unpackb(data, use_list=False, raw=False)


# msgpack with manual numpy adapter (no msgpack_numpy)
def _ndarray_to_bytes(arr: np.ndarray) -> dict:
    return {
        b"__ndarray__": True,
        b"data": arr.tobytes(),
        b"shape": arr.shape,
        b"dtype": str(arr.dtype),
    }


def _bytes_to_ndarray(obj: dict) -> np.ndarray:
    return np.frombuffer(obj[b"data"], dtype=np.dtype(obj[b"dtype"])).reshape(obj[b"shape"])


def _encode_numpy(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return _ndarray_to_bytes(obj)
    if isinstance(obj, dict):
        return {k: _encode_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_encode_numpy(v) for v in obj)
    return obj


def _decode_numpy(obj: Any) -> Any:
    if isinstance(obj, dict):
        if obj.get(b"__ndarray__"):
            return _bytes_to_ndarray(obj)
        return {k: _decode_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_decode_numpy(v) for v in obj)
    return obj


def serialize_msgpack_manual(obj: Any) -> bytes:
    encoded = _encode_numpy(obj)
    return msgpack.packb(encoded, use_bin_type=True)


def deserialize_msgpack_manual(data: bytes) -> Any:
    decoded = msgpack.unpackb(data, use_list=False, raw=False)
    return _decode_numpy(decoded)


# ── Correctness verification ─────────────────────────────────────────────

def _verify_equal(original: Any, restored: Any, path: str = "root") -> list[str]:
    errors: list[str] = []
    if isinstance(original, dict):
        if not isinstance(restored, dict):
            errors.append(f"{path}: expected dict, got {type(restored).__name__}")
            return errors
        if set(original.keys()) != set(restored.keys()):
            errors.append(f"{path}: key mismatch: {set(original.keys())} vs {set(restored.keys())}")
        for k in original:
            errors.extend(_verify_equal(original[k], restored[k], f"{path}.{k}"))
    elif isinstance(original, np.ndarray):
        if not isinstance(restored, np.ndarray):
            errors.append(f"{path}: expected ndarray, got {type(restored).__name__}")
            return errors
        if original.shape != restored.shape:
            errors.append(f"{path}: shape mismatch {original.shape} vs {restored.shape}")
        if original.dtype != restored.dtype:
            errors.append(f"{path}: dtype mismatch {original.dtype} vs {restored.dtype}")
        if not np.allclose(original.astype(np.float64), restored.astype(np.float64)):
            errors.append(f"{path}: value mismatch (max diff {np.max(np.abs(original.astype(np.float64) - restored.astype(np.float64)))})")
    elif isinstance(original, (list, tuple)):
        if not isinstance(restored, (list, tuple)):
            errors.append(f"{path}: expected list/tuple, got {type(restored).__name__}")
            return errors
        if len(original) != len(restored):
            errors.append(f"{path}: length mismatch {len(original)} vs {len(restored)}")
        for i in range(min(len(original), len(restored))):
            errors.extend(_verify_equal(original[i], restored[i], f"{path}[{i}]"))
    elif isinstance(original, (np.floating, float)):
        if not np.isclose(float(original), float(restored)):
            errors.append(f"{path}: float mismatch {original} vs {restored}")
    else:
        if original != restored:
            errors.append(f"{path}: value mismatch {original!r} vs {restored!r}")
    return errors


# ── Benchmark runner ─────────────────────────────────────────────────────

def run_benchmark(
    payload_builders: dict[str, Any],
    serializers: dict[str, Any],
    *,
    warmup: int = 10,
    iters: int = 100,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for payload_name, build_fn in payload_builders.items():
        print(f"\n{'='*60}")
        print(f"Payload: {payload_name}")
        payload = build_fn()
        # Estimate payload size
        total_bytes = _estimate_payload_bytes(payload)
        print(f"  Estimated numpy bytes: {total_bytes / 1024 / 1024:.1f} MB")
        print(f"  Top-level keys: {list(payload.keys()) if isinstance(payload, dict) else 'N/A'}")

        for ser_name, (ser_fn, deser_fn) in serializers.items():
            print(f"  Serializer: {ser_name} ...", end=" ", flush=True)

            # Warmup
            for _ in range(warmup):
                data = ser_fn(payload)
                _ = deser_fn(data)

            # Timed runs
            ser_times: list[float] = []
            deser_times: list[float] = []
            sizes: list[int] = []
            gc.collect()

            for _ in range(iters):
                t0 = time.perf_counter()
                data = ser_fn(payload)
                t1 = time.perf_counter()
                restored = deser_fn(data)
                t2 = time.perf_counter()
                ser_times.append(t1 - t0)
                deser_times.append(t2 - t1)
                sizes.append(len(data))

            mean_ser = statistics.mean(ser_times)
            mean_deser = statistics.mean(deser_times)
            mean_size = statistics.mean(sizes)
            throughput_ser = total_bytes / mean_ser / 1e6 if mean_ser > 0 else float("inf")
            throughput_deser = total_bytes / mean_deser / 1e6 if mean_deser > 0 else float("inf")

            # Verify correctness (first iteration)
            final_data = ser_fn(payload)
            final_restored = deser_fn(final_data)
            errors = _verify_equal(payload, final_restored)
            ok = len(errors) == 0

            stats = {
                "payload": payload_name,
                "serializer": ser_name,
                "ok": ok,
                "errors": errors[:3],  # first 3
                "ser_mean_ms": mean_ser * 1000,
                "ser_p95_ms": _p95(ser_times) * 1000,
                "ser_min_ms": min(ser_times) * 1000,
                "ser_max_ms": max(ser_times) * 1000,
                "ser_std_ms": statistics.stdev(ser_times) * 1000 if len(ser_times) > 1 else 0,
                "deser_mean_ms": mean_deser * 1000,
                "deser_p95_ms": _p95(deser_times) * 1000,
                "deser_min_ms": min(deser_times) * 1000,
                "deser_max_ms": max(deser_times) * 1000,
                "deser_std_ms": statistics.stdev(deser_times) * 1000 if len(deser_times) > 1 else 0,
                "size_bytes": int(mean_size),
                "size_mb": mean_size / 1024 / 1024,
                "throughput_ser_mbps": throughput_ser,
                "throughput_deser_mbps": throughput_deser,
                "roundtrip_mean_ms": (mean_ser + mean_deser) * 1000,
                "payloads_per_sec": 1.0 / (mean_ser + mean_deser) if (mean_ser + mean_deser) > 0 else float("inf"),
                "total_param_count": payload.get("_total_param_count", 0),
            }
            results.append(stats)
            tag = "OK" if ok else f"FAIL ({len(errors)} errors)"
            print(f"{tag} | ser={mean_ser*1000:.2f}ms deser={mean_deser*1000:.2f}ms size={mean_size/1024/1024:.1f}MB")
            if not ok:
                for e in errors[:3]:
                    print(f"        {e}")
    return results


def _p95(values: list[float]) -> float:
    return sorted(values)[int(len(values) * 0.95)]


def _estimate_payload_bytes(obj: Any) -> int:
    if isinstance(obj, np.ndarray):
        return obj.nbytes
    if isinstance(obj, dict):
        return sum(_estimate_payload_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_estimate_payload_bytes(v) for v in obj)
    return 64


# ── Report generation ────────────────────────────────────────────────────

def write_report(results: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults JSON: {output_dir / 'results.json'}")

    # CSV
    csv_keys = [
        "payload", "serializer", "ok",
        "ser_mean_ms", "deser_mean_ms", "roundtrip_mean_ms",
        "size_mb", "payloads_per_sec",
    ]
    with open(output_dir / "results.csv", "w") as f:
        f.write(",".join(csv_keys) + "\n")
        for r in results:
            f.write(",".join(_csv_val(r[k]) for k in csv_keys) + "\n")

    # Summary markdown
    lines = ["# msgpack vs pickle Benchmark Results", ""]
    lines.append(f"Warmup: 10, Iters: 100, Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Group by payload
    for payload_name in sorted(set(r["payload"] for r in results)):
        lines.append(f"## {payload_name}")
        lines.append("")
        lines.append("| Serializer | OK | Ser (ms) | Deser (ms) | Roundtrip (ms) | Size (MB) | Payloads/s |")
        lines.append("|---|---|---|---|---|---|---|")
        payload_results = [r for r in results if r["payload"] == payload_name]
        for r in sorted(payload_results, key=lambda x: x["roundtrip_mean_ms"]):
            ok_str = "✓" if r["ok"] else "✗"
            lines.append(
                f"| {r['serializer']} | {ok_str} | {r['ser_mean_ms']:.2f} | {r['deser_mean_ms']:.2f} | "
                f"{r['roundtrip_mean_ms']:.2f} | {r['size_mb']:.2f} | {r['payloads_per_sec']:.1f} |"
            )

        # Pickle vs msgpack comparison
        pickle_row = next((r for r in payload_results if r["serializer"] == "pickle (HIGHEST_PROTOCOL + lz4)"), None)
        msgpack_row = next((r for r in payload_results if r["serializer"] == "msgpack (msgpack_numpy)"), None)
        if pickle_row and msgpack_row:
            speedup = pickle_row["roundtrip_mean_ms"] / msgpack_row["roundtrip_mean_ms"] if msgpack_row["roundtrip_mean_ms"] > 0 else float("inf")
            size_ratio = msgpack_row["size_mb"] / pickle_row["size_mb"] if pickle_row["size_mb"] > 0 else float("inf")
            lines.append("")
            lines.append(f"**msgpack vs pickle**: {speedup:.1f}x faster roundtrip, {size_ratio:.1f}x size ratio")
        lines.append("")

    # Recommendations
    lines.append("## Recommendations")
    lines.append("")
    lines.append("- Based on real serl_torch data structures (batch transitions, weight snapshots, stats).")
    lines.append("- `pickle (HIGHEST_PROTOCOL + lz4)` is the current implementation in `trainer_transport.py`.")
    if HAS_MSGPACK_NUMPY:
        lines.append("- `msgpack (msgpack_numpy)` is the recommended replacement.")
    else:
        lines.append("- `msgpack (msgpack_numpy)` is NOT available; install with `pip install msgpack msgpack-numpy`.")

    with open(output_dir / "summary.md", "w") as f:
        f.write("\n".join(lines))
    print(f"Summary: {output_dir / 'summary.md'}")


def _csv_val(val: Any) -> str:
    if isinstance(val, bool):
        return "1" if val else "0"
    return str(val)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark msgpack vs pickle on serl_torch payloads")
    parser.add_argument("--payload-scale", default="all", choices=["all", "small", "medium", "large", "weight", "stats"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--output-dir", default="/tmp/msgpack_pickle_benchmark")
    args = parser.parse_args()

    print(f"Python: {sys.version}")
    print(f"NumPy:  {np.__version__}")
    print(f"msgpack-numpy available: {HAS_MSGPACK_NUMPY}")
    print(f"lz4 available: {_lz4_frame is not None}")

    # Payload builders
    all_payloads: dict[str, Any] = {
        "transition_batch_1":  lambda: build_transition_payload(1),
        "transition_batch_30": lambda: build_transition_payload(30),
        "transition_batch_128": lambda: build_transition_payload(128),
        "weight_snapshot":      build_weight_payload,
        "rollout_stats":        build_stats_payload,
    }

    if args.payload_scale == "small":
        payloads = {"transition_batch_1": all_payloads["transition_batch_1"]}
    elif args.payload_scale == "medium":
        payloads = {"transition_batch_30": all_payloads["transition_batch_30"]}
    elif args.payload_scale == "large":
        payloads = {"transition_batch_128": all_payloads["transition_batch_128"]}
    elif args.payload_scale == "weight":
        payloads = {"weight_snapshot": all_payloads["weight_snapshot"]}
    elif args.payload_scale == "stats":
        payloads = {"rollout_stats": all_payloads["rollout_stats"]}
    else:
        payloads = all_payloads

    # Serializers to compare
    serializers: dict[str, Any] = {
        "pickle (HIGHEST_PROTOCOL + lz4)": (serialize_pickle, deserialize_pickle),
        "pickle (HIGHEST_PROTOCOL, no compression)": (serialize_pickle_raw, deserialize_pickle_raw),
    }
    if HAS_MSGPACK_NUMPY:
        serializers["msgpack (msgpack_numpy)"] = (serialize_msgpack, deserialize_msgpack)
    serializers["msgpack (manual numpy)"] = (serialize_msgpack_manual, deserialize_msgpack_manual)

    results = run_benchmark(
        payloads, serializers,
        warmup=args.warmup, iters=args.iters,
    )

    output_dir = Path(args.output_dir)
    write_report(results, output_dir)


if __name__ == "__main__":
    main()
