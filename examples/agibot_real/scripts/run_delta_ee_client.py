"""Print pure-software Quest delta-EE targets without commanding robot."""

from __future__ import annotations

import argparse
import time

from serl_torch.examples.agibot_real.intervention import DeltaEETeleopController
from serl_torch.examples.agibot_real.intervention import QuestVRClient


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read Quest input and print AgiBot EE target state.",
    )
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--duration-sec", type=float, default=0.0)
    parser.add_argument("--scaling-factor", type=float, default=0.5)
    parser.add_argument("--control-freq", type=int, default=30)
    parser.add_argument("--controller-mode", default="both")
    parser.add_argument("--coordinate-mapping", default="sim")
    parser.add_argument("--visualize", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    period = 1.0 / max(float(args.hz), 1e-6)
    start_time = time.time()
    ee_controller = DeltaEETeleopController()

    with QuestVRClient(
        scaling_factor=float(args.scaling_factor),
        control_freq=int(args.control_freq),
        enable_visualization=bool(args.visualize),
        controller_mode=str(args.controller_mode),
        coordinate_mapping=str(args.coordinate_mapping),
    ) as client:
        while True:
            snapshot = client.snapshot()
            ee_controller.update(snapshot.signals)
            print(ee_controller.format_status())
            if float(args.duration_sec) > 0.0:
                if time.time() - start_time >= float(args.duration_sec):
                    break
            time.sleep(period)


if __name__ == "__main__":
    main()
