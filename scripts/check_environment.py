"""Validate the local Python and SUMO prerequisites."""

from __future__ import annotations

import shutil
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMO_EXECUTABLES = ("sumo", "sumo-gui", "netconvert")


def find_random_trips() -> Path | None:
    """Find SUMO's Python tool on Windows or Unix-like systems."""

    on_path = shutil.which("randomTrips.py")
    if on_path:
        return Path(on_path)

    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        candidate = Path(sumo_home) / "tools" / "randomTrips.py"
        if candidate.is_file():
            return candidate
    return None


def main() -> int:
    """Print prerequisite status and return a process status code."""

    print(f"Python: {sys.version.split()[0]}")
    print(f"Project root: {PROJECT_ROOT}")

    missing = [
        command for command in SUMO_EXECUTABLES if shutil.which(command) is None
    ]
    random_trips = find_random_trips()
    if random_trips is None:
        missing.append("randomTrips.py")
    if missing:
        print("SUMO prerequisites: MISSING")
        print("Missing commands: " + ", ".join(missing))
        print(
            "Install Eclipse SUMO and add its bin/tools directories to PATH, "
            "or set SUMO_HOME before running the simulation."
        )
        return 1

    print("SUMO prerequisites: AVAILABLE")
    for command in SUMO_EXECUTABLES:
        print(f"  {command}: {shutil.which(command)}")
    print(f"  randomTrips.py: {random_trips}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
