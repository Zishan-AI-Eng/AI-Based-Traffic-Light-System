import os, sys

print("Python:", sys.version.split()[0])

import torch
print("Torch:", torch.__version__, "| GPU:", torch.cuda.is_available())

import gymnasium, pettingzoo, numpy
print("Gymnasium:", gymnasium.__version__)
print("PettingZoo:", pettingzoo.__version__)

print("SUMO_HOME:", os.environ.get("SUMO_HOME"))

import libsumo, traci
print("libsumo:", libsumo.__file__ is not None, "| traci:", traci.__version__ if hasattr(traci, "__version__") else "ok")

import traffic_rl
print("traffic_rl package: OK")