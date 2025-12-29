from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_workspace() -> None:
    """Make sub-repos importable without modifying them.

    This repo vendors multiple projects as subfolders:
      - s4c/ expects to be imported as top-level packages: `datasets`, `models`, `utils`, ...
      - gsplat/ provides python package `gsplat` under gsplat/gsplat
      - map-anything/ provides python package `mapanything` under map-anything/mapanything

    Rather than editing those repos or requiring editable installs, we add their roots to sys.path.
    """

    repo_root = Path(__file__).resolve().parents[1]

    # Ensure repo root is present (so `ssc_gs` itself resolves in -m execution).
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    s4c_root = repo_root / "s4c"
    gsplat_root = repo_root / "gsplat"
    mapanything_root = repo_root / "map-anything"

    for p in (s4c_root, gsplat_root, mapanything_root):
        ps = str(p)
        if p.exists() and ps not in sys.path:
            sys.path.insert(0, ps)
