import os
import libsumo as traci

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = os.path.join(ROOT, "data", "sandbox", "run.sumocfg")

traci.start(["sumo", "-c", cfg, "--no-warnings", "true"])

TL = "A1"
lanes = sorted(set(traci.trafficlight.getControlledLanes(TL)))
print("Unique lanes:", lanes)

total_wait, total_queue, steps = 0.0, 0.0, 0
arrived = 0

while traci.simulation.getTime() < 600:
    traci.simulationStep()
    steps += 1
    arrived += traci.simulation.getArrivedNumber()
    for ln in lanes:
        total_wait += traci.lane.getWaitingTime(ln)
        total_queue += traci.lane.getLastStepHaltingNumber(ln)

print("Steps:", steps)
print("Avg waiting time per step (all lanes):", round(total_wait / steps, 2))
print("Avg queue per step (all lanes):", round(total_queue / steps, 2))
print("Vehicles arrived (throughput):", arrived)

traci.close()