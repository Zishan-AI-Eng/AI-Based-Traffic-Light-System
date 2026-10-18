import os
from make_routes import make_routes
from make_asym_routes import make_asym_routes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NET = os.path.join(ROOT, "data", "sandbox", "single.net.xml")
OUT = os.path.join(ROOT, "data", "generated", "train")
os.makedirs(OUT, exist_ok=True)

for vol in ["low", "medium", "high", "rush"]:
    for seed in range(100, 105):
        make_routes(NET, os.path.join(OUT, f"{vol}_{seed}.rou.xml"), vol, seed=seed)

for i, (heavy, light) in enumerate([(400, 40), (500, 30), (600, 150),
                                    (700, 100), (900, 50), (1000, 80)]):
    make_asym_routes(NET, os.path.join(OUT, f"asym_{i}.rou.xml"),
                     heavy_vph=heavy, light_vph=light)