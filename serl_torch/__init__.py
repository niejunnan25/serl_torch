"""Repository-local compatibility namespace.

This worktree may not be named ``serl_torch`` on disk, but legacy entrypoints
import modules as ``serl_torch.examples...``.  Point this namespace at the
repository root so those imports resolve to this worktree, not a sibling clone.
"""

from pathlib import Path

__path__ = [str(Path(__file__).resolve().parents[1])]
