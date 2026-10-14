"""Start and close a minimal SUMO process to verify the local installation."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


def main() -> int:
    """Generate a tiny network and verify that SUMO can simulate it."""

    sumo = shutil.which("sumo")
    netconvert = shutil.which("netconvert")
    if sumo is None or netconvert is None:
        print(
            "SUMO smoke test skipped: 'sumo' and 'netconvert' must be on PATH. "
            "Install SUMO and configure PATH/SUMO_HOME first."
        )
        return 2

    with tempfile.TemporaryDirectory(prefix="traffic-light-sumo-") as directory:
        directory_path = Path(directory)
        nodes_path = directory_path / "nodes.nod.xml"
        edges_path = directory_path / "edges.edg.xml"
        network_path = directory_path / "smoke.net.xml"
        nodes_path.write_text(
            """<nodes>
    <node id="left" x="-100" y="0"/>
    <node id="right" x="100" y="0"/>
</nodes>
""",
            encoding="utf-8",
        )
        edges_path.write_text(
            """<edges>
    <edge id="left_to_right" from="left" to="right" numLanes="1" speed="13.9"/>
</edges>
""",
            encoding="utf-8",
        )
        build_network = subprocess.run(
            [
                netconvert,
                "--node-files",
                str(nodes_path),
                "--edge-files",
                str(edges_path),
                "--output-file",
                str(network_path),
                "--no-warnings",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if build_network.returncode != 0:
            print("SUMO smoke test failed while generating the test network.")
            if build_network.stderr:
                print(build_network.stderr.strip())
            return build_network.returncode

        result = subprocess.run(
            [
                sumo,
                "--net-file",
                str(network_path),
                "--end",
                "1",
                "--no-warnings",
                "--duration-log.disable",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

    if result.returncode != 0:
        print("SUMO smoke test failed.")
        if result.stderr:
            print(result.stderr.strip())
        return result.returncode

    print("SUMO smoke test passed: SUMO started and exited successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
