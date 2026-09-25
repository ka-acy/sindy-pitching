#!/usr/bin/env python3
"""Refresh the derivative-focused Stage B baseball pipeline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    python_bin = PROJECT_ROOT / ".venv" / "bin" / "python"
    script_path = PROJECT_ROOT / "src" / "preprocessing" / "refresh_baseball_pipeline.py"
    config_path = PROJECT_ROOT / "config" / "preprocessing.baseball.unaligned.yaml"
    command = [str(python_bin), str(script_path), "--config", str(config_path), *sys.argv[1:]]
    raise SystemExit(subprocess.call(command, cwd=PROJECT_ROOT))


if __name__ == "__main__":
    main()
