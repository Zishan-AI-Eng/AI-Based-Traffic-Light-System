import os
import sumolib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SANDBOX = os.path.join(ROOT, "data", "sandbox")


def _unit(a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = (dx * dx + dy * dy) ** 0.5
    return (dx / n, dy / n)


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def make_asym_routes(net_file, out_file, tl_node="A1",
                     heavy_vph=700, light_vph=50, duration=600, swap=False):
    net = sumolib.net.readNet(net_file)
    node = net.getNode(tl_node)
    incoming = list(node.getIncoming())
    outgoing = list(node.getOutgoing())

    in_dir = {e.getID(): _unit(e.getShape()[-2], e.getShape()[-1]) for e in incoming}
    out_dir = {e.getID(): _unit(e.getShape()[0], e.getShape()[1]) for e in outgoing}

    ids = list(in_dir)
    first = ids[0]
    opposite = min((i for i in ids if i != first),
                   key=lambda i: _dot(in_dir[first], in_dir[i]))
    heavy_axis = {first, opposite}
    if swap:                                   # heavy aur light axis ulat do
        heavy_axis = set(ids) - heavy_axis

    routes, flows = [], []
    for i in ids:
        vph = heavy_vph if i in heavy_axis else light_vph
        outs = sorted(out_dir, key=lambda o: -_dot(in_dir[i], out_dir[o]))
        straight, others = outs[0], outs[1:-1]
        options = [(straight, 0.6)] + [(o, 0.4 / len(others)) for o in others]
        for o, w in options:
            rid = f"r_{i}_{o}"
            routes.append(f'  <route id="{rid}" edges="{i} {o}"/>')
            flows.append(
                f'  <flow id="f_{i}_{o}" route="{rid}" begin="0" end="{duration}" '
                f'probability="{vph * w / 3600:.5f}" departLane="best" departSpeed="max"/>'
            )

    with open(out_file, "w") as f:
        f.write("<routes>\n" + "\n".join(routes + flows) + "\n</routes>\n")

    print("Heavy axis edges:", sorted(heavy_axis))
    print("Light axis edges:", sorted(set(ids) - heavy_axis))
    print("Written:", os.path.basename(out_file))


if __name__ == "__main__":
    net = os.path.join(SANDBOX, "single.net.xml")
    make_asym_routes(net, os.path.join(SANDBOX, "asym_a.rou.xml"))
    make_asym_routes(net, os.path.join(SANDBOX, "asym_swap.rou.xml"), swap=True)