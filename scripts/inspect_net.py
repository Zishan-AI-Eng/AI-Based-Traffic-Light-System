"""
Kisi bhi .net.xml ke traffic-light junction ki movements aur phases dikhata hai.
(Simulation nahi chalata, sirf net file padhta hai.)

Usage (project root se):
  python scripts/inspect_net.py data/sandbox/single3.net.xml A1
"""
import sys
import sumolib

net_file = sys.argv[1]
tl_id = sys.argv[2] if len(sys.argv) > 2 else "A1"

DIR_NAME = {"s": "straight", "l": "left", "L": "left(partial)",
            "r": "right", "R": "right(partial)", "t": "U-turn"}

net = sumolib.net.readNet(net_file, withPrograms=True)

links = {}
for edge in net.getNode(tl_id).getIncoming():
    for lane in edge.getLanes():
        for conn in lane.getOutgoing():
            idx = conn.getTLLinkIndex()
            if idx >= 0:
                links[idx] = (conn.getFromLane().getID(), conn.getToLane().getID(),
                              DIR_NAME.get(conn.getDirection(), conn.getDirection()))


def state_dur(p):                       # sumolib ke alag versions dono handle karo
    return (p.state, p.duration) if hasattr(p, "state") else (p[0], p[1])


program = list(net.getTLS(tl_id).getPrograms().values())[0]
phases = [state_dur(p) for p in program.getPhases()]

print(f"\nJunction {tl_id}: {len(links)} links, {len(phases)} phases (green+yellow+red sab)\n")
for i, (s, d) in enumerate(phases):
    print(f"  phase {i}: {d:>4}s  {s}")

green = [i for i, (s, d) in enumerate(phases) if "y" not in s.lower() and ("G" in s or "g" in s)]
print("\nGreen phases:", green, "\n")

header = "link | from lane -> to lane      | movement       | " + \
         " | ".join(f"ph{p}" for p in green)
print(header)
print("-" * len(header))
for idx in sorted(links):
    f, t, d = links[idx]
    cols = " | ".join(f" {phases[p][0][idx]} " for p in green)
    print(f"{idx:4d} | {f:8s} -> {t:10s} | {d:14s} | {cols}")
print("\nG = priority green, g = green lekin yield, r = red")