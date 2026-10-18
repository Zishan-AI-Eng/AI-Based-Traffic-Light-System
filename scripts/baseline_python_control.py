import os
import libsumo as traci

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = os.path.join(ROOT, "data", "sandbox", "run.sumocfg")

traci.start(["sumo", "-c", cfg, "--no-warnings", "true"])

TL = "A1"
lanes = sorted(set(traci.trafficlight.getControlledLanes(TL)))
DURATION = {0: 42, 1: 3, 2: 42, 3: 3}   # phase -> seconds

phase = 0
traci.trafficlight.setPhase(TL, phase)
traci.trafficlight.setPhaseDuration(TL, 9999)   # SUMO ka timer band, ab hum badlenge
phase_start = 0.0

total_wait, total_queue, steps, arrived = 0.0, 0.0, 0, 0

while traci.simulation.getTime() < 600:
    traci.simulationStep()
    t = traci.simulation.getTime()
    steps += 1
    arrived += traci.simulation.getArrivedNumber()

    if t - phase_start >= DURATION[phase]:
        phase = (phase + 1) % 4
        traci.trafficlight.setPhase(TL, phase)
        traci.trafficlight.setPhaseDuration(TL, 9999)
        phase_start = t

    for ln in lanes:
        total_wait += traci.lane.getWaitingTime(ln)
        total_queue += traci.lane.getLastStepHaltingNumber(ln)

print("Avg waiting time per step:", round(total_wait / steps, 2))
print("Avg queue per step:", round(total_queue / steps, 2))
print("Vehicles arrived:", arrived)

traci.close()