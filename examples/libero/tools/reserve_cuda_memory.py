#!/usr/bin/env python3
"""Reserve CUDA memory for a launcher-managed GPU guard process."""

from __future__ import annotations

import argparse
import signal
import sys
import time


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fraction",
        type=float,
        required=True,
        help="Fraction of total visible GPU memory for this guard process to reserve.",
    )
    parser.add_argument(
        "--chunk-mib",
        type=int,
        default=512,
        help="Maximum allocation chunk size in MiB.",
    )
    parser.add_argument(
        "--safety-margin-mib",
        type=int,
        default=512,
        help="Minimum free memory to leave if the requested reservation exceeds availability.",
    )
    parser.add_argument(
        "--poll-sec",
        type=float,
        default=30.0,
        help="Sleep interval while holding memory.",
    )
    return parser.parse_args()


def _format_gib(num_bytes: int | float) -> str:
    return f"{num_bytes / (1024**3):.2f} GiB"


def main() -> int:
    args = _parse_args()
    if not 0.0 < args.fraction < 1.0:
        raise SystemExit("--fraction must be greater than 0 and less than 1")
    if args.chunk_mib <= 0:
        raise SystemExit("--chunk-mib must be positive")
    if args.safety_margin_mib < 0:
        raise SystemExit("--safety-margin-mib must be non-negative")

    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on runtime env
        print(f"[memory_guard] failed to import torch: {exc}", flush=True)
        return 2

    if not torch.cuda.is_available():
        print("[memory_guard] CUDA is not available", flush=True)
        return 2

    stop = False

    def _handle_signal(signum: int, _frame: object) -> None:
        nonlocal stop
        print(f"[memory_guard] received signal {signum}; releasing memory", flush=True)
        stop = True

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    target_reserved_bytes = int(total_bytes * args.fraction)
    safety_margin_bytes = args.safety_margin_mib * 1024**2
    chunk_bytes = args.chunk_mib * 1024**2
    allocations: list[torch.Tensor] = []

    print(
        "[memory_guard] starting: "
        f"fraction={args.fraction:.4f}, total={_format_gib(total_bytes)}, "
        f"free_before={_format_gib(free_bytes)}, "
        f"target_reserved={_format_gib(target_reserved_bytes)}, "
        f"safety_margin={_format_gib(safety_margin_bytes)}",
        flush=True,
    )

    while not stop:
        reserved_bytes = sum(
            tensor.numel() * tensor.element_size() for tensor in allocations
        )
        remaining_target_bytes = target_reserved_bytes - reserved_bytes
        if remaining_target_bytes <= 0:
            break
        free_bytes, _ = torch.cuda.mem_get_info(device)
        allocatable_bytes = free_bytes - safety_margin_bytes
        if allocatable_bytes <= 0:
            break
        request_bytes = int(
            min(chunk_bytes, remaining_target_bytes, allocatable_bytes)
        )
        if request_bytes <= 0:
            break
        try:
            allocations.append(torch.empty(request_bytes, dtype=torch.uint8, device=device))
        except RuntimeError as exc:
            print(
                "[memory_guard] allocation stopped after RuntimeError: "
                f"{exc.__class__.__name__}: {exc}",
                flush=True,
            )
            torch.cuda.empty_cache()
            break

    reserved_bytes = sum(tensor.numel() * tensor.element_size() for tensor in allocations)
    free_after, _ = torch.cuda.mem_get_info(device)
    print(
        "[memory_guard] holding: "
        f"reserved={_format_gib(reserved_bytes)}, "
        f"free_after={_format_gib(free_after)}, chunks={len(allocations)}",
        flush=True,
    )

    while not stop:
        time.sleep(args.poll_sec)

    allocations.clear()
    torch.cuda.empty_cache()
    print("[memory_guard] released", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
