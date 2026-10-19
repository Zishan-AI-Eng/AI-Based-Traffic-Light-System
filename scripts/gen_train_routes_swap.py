import os
from make_asym_routes import make_asym_routes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NET = os.path.join(ROOT, "data", "sandbox", "single.net.xml")
OUT = os.path.join(ROOT, "data", "generated", "train")

for i, (heavy, light) in enumerate([(400, 40), (500, 30), (600, 150),
                                    (700, 100), (900, 50), (1000, 80)]):
    make_asym_routes(NET, os.path.join(OUT, f"asym_swap_{i}.rou.xml"),
                     heavy_vph=heavy, light_vph=light, swap=True)