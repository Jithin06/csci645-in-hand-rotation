"""Local copy of ``mjlab.utils.os.update_assets``.

mjlab removed this helper in v1.3 (upstream PR #873). The LEAP hand MJCF keeps
its meshes in a sibling ``assets/`` directory, so we keep embedding them into
``MjSpec.assets`` exactly as before to preserve mesh resolution behaviour.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def update_assets(
  assets: dict[str, Any],
  path: str | Path,
  meshdir: str | None = None,
  glob: str = "*",
  recursive: bool = False,
) -> None:
  """Add files under ``path`` to ``assets``, keyed with the ``meshdir`` prefix."""
  for f in Path(path).glob(glob):
    if f.is_file():
      asset_key = f"{meshdir}/{f.name}" if meshdir else f.name
      assets[asset_key] = f.read_bytes()
    elif f.is_dir() and recursive:
      update_assets(assets, f, meshdir, glob, recursive)
