"""Thin launcher so the project runs from a fresh checkout without installation."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from comp345.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
