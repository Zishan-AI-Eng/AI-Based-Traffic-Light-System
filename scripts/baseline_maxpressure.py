import os
import libsumo as traci
from traffic_rl.envs.sumo_env import SumoTrafficEnv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SANDBOX = os.path.join(ROOT, "data", "sandbox")
cfg = os.path.join(SANDBOX, "run.sumocfg")

scenarios = {
    "medium": os.path.join(SANDBOX, "test_a.rou.xml"),
    "rush": os.path.join(SANDBOX, "test_d.rou.xml"),
    "asym": os.path.join(SANDBOX, "asym_a.rou.xml"),
}
seeds = [11, 22, 33, 44, 55]    # AI aur fixed baseline wali wahi seeds

env = SumoTrafficEnv(cfg)
env.reset(seed=1, options={"route_file": scenarios["medium"]})

# har green phase kin lanes ko green deta hai
controlled = traci.trafficlight.getControlledLanes(env.tl_id)
phase_lanes = [sorted({controlled[i] for i, c in enumerate(st) if c in "Gg"})
               for st in env.green_states]
print("Lanes per phase:", [len(p) for p in phase_lanes])

for name, f in scenarios.items():
    waits, queues, sw = [], [], []
    for s in seeds:
        env.reset(seed=s, options={"route_file": f})
        done = False
        while not done:
            pressure = [sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)
                        for lanes in phase_lanes]
            action = max(range(len(pressure)), key=lambda i: pressure[i])
            _, _, term, trunc, info = env.step(action)
            done = term or trunc
        waits.append(info["avg_wait"]); queues.append(info["avg_queue"]); sw.append(info["switches"])
    print(f"{name:7s} | wait={sum(waits)/len(waits):6.2f} | "
          f"queue={sum(queues)/len(queues):5.2f} | switches={sum(sw)/len(sw):4.1f}")
env.close()