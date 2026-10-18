import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMO_HOME = os.environ["SUMO_HOME"]
RANDOM_TRIPS = os.path.join(SUMO_HOME, "tools", "randomTrips.py")

# volume level -> seconds between vehicles (chota number = zyada traffic)
VOLUME = {"low": 4.0, "medium": 2.0, "high": 1.2, "rush": 0.8}


def make_routes(net_file, out_file, volume="medium", seed=42, duration=600):
    period = VOLUME[volume]
    trips = out_file.replace(".rou.xml", ".trips.xml")
    subprocess.run([
        sys.executable, RANDOM_TRIPS,
        "-n", net_file, "-o", trips, "-r", out_file,
        "-e", str(duration), "-p", str(period),
        "--seed", str(seed), "--validate",
    ], check=True, stdout=subprocess.DEVNULL)
    print(f"Created {os.path.basename(out_file)} | volume={volume} seed={seed}")


if __name__ == "__main__":
    sandbox = os.path.join(ROOT, "data", "sandbox")
    net = os.path.join(sandbox, "single.net.xml")
    make_routes(net, os.path.join(sandbox, "test_a.rou.xml"), "medium", seed=1)
    make_routes(net, os.path.join(sandbox, "test_b.rou.xml"), "medium", seed=1)
    make_routes(net, os.path.join(sandbox, "test_c.rou.xml"), "medium", seed=2)
    make_routes(net, os.path.join(sandbox, "test_d.rou.xml"), "rush", seed=1)