"""
Junction A1 ki har 'link' (movement) ki list + collision check.
Project root se chalao:  python scripts/inspect_links.py
"""
import os
import sumolib
import libsumo as traci

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SANDBOX = os.path.join(ROOT, "data", "sandbox")
cfg = os.path.join(SANDBOX, "run.sumocfg")
net = sumolib.net.readNet(os.path.join(SANDBOX, "single.net.xml"))

DIR_NAME = {"s": "straight", "l": "left", "L": "left(partial)",
            "r": "right", "R": "right(partial)", "t": "U-turn"}

links = {}
for edge in net.getNode("A1").getIncoming():
    for lane in edge.getLanes():
        for conn in lane.getOutgoing():
            idx = conn.getTLLinkIndex()
            if idx >= 0:
                links[idx] = (conn.getFromLane().getID(), conn.getToLane().getID(),
                              DIR_NAME.get(conn.getDirection(), conn.getDirection()))

# junction ke andar bhi collision check on karo (default mein band hota hai)
traci.start(["sumo", "-c", cfg, "--no-warnings", "true",
             "--collision.check-junctions", "true", "--collision.action", "warn"])

logic = traci.trafficlight.getAllProgramLogics("A1")[0]
green_phases = [i for i, p in enumerate(logic.phases) if "y" not in p.state and "G" in p.state]
print("Green phases:", green_phases, "\n")

header = "link | from lane -> to lane      | movement       | " + \
         " | ".join(f"phase{p}" for p in green_phases)
print(header)
print("-" * len(header))
for idx in sorted(links):
    f, t, d = links[idx]
    states = " | ".join(f"  {logic.phases[p].state[idx]}   " for p in green_phases)
    print(f"{idx:4d} | {f:8s} -> {t:10s} | {d:14s} | {states}")
print("\nG = priority green (conflict nahi), g = green lekin yield karna hai, r = red\n")

collisions = 0
for _ in range(1200):                       # 600 s
    traci.simulationStep()
    collisions += traci.simulation.getCollidingVehiclesNumber()
print("Collisions in 600 s (junction check ON):", collisions)
traci.close()