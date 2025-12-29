"""Compatibility shim for vendored S4C benchmark scripts.

Some environments include a third-party package named `scripts` in site-packages,
which breaks imports inside S4C's SSCBench utilities (they use `scripts.*`).

This shim ensures `import scripts.benchmarks...` resolves to the vendored folder
at `s4c/scripts` within this workspace.
"""

from __future__ import annotations

from pathlib import Path

# Make this package also search in the vendored S4C `scripts/` folder.
_repo_root = Path(__file__).resolve().parents[1]
_s4c_scripts = _repo_root / "s4c" / "scripts"
if _s4c_scripts.exists():
    __path__.append(str(_s4c_scripts))
