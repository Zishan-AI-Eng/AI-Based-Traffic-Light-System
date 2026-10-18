import os
import libsumo as traci

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = os.path.join(ROOT, "data", "sandbox", "run.sumocfg")
print("Config path:", cfg, "| exists:", os.path.exists(cfg))

traci.start(["sumo", "-c", cfg])

print("Traffic lights:", traci.trafficlight.getIDList())
print("A1 incoming lanes:", traci.trafficlight.getControlledLanes("A1"))

for step in range(100):
    traci.simulationStep()

print("Time:", traci.simulation.getTime())
print("Vehicles now:", traci.vehicle.getIDList())
print("A1 phase:", traci.trafficlight.getPhase("A1"))
print("A1 state:", traci.trafficlight.getRedYellowGreenState("A1"))

traci.close()