"""Generic training telemetry helpers."""
from __future__ import annotations

from typing import Any
from typing import Dict
from typing import Tuple


def _log_info_scalars(
    tb_writer,
    info: Dict[str, Any],
    global_step: int,
    pairs: Tuple[Tuple[str, str], ...],
) -> None:
    for tb_key, info_key in pairs:
        if info_key in info and info[info_key] is not None:
            tb_writer.add_scalar(tb_key, float(info[info_key]), global_step)
