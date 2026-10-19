import os
import libsumo as traci
from traffic_rl.envs.sumo_env import SumoTrafficEnv
from traffic_rl.agents.dqn import DQNAgent

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SANDBOX = os.path.join(ROOT, "data", "sandbox")
cfg = os.path.join(SANDBOX, "run.sumocfg")

scenarios = {
    "asym_a (training jaisa axis)": os.path.join(SANDBOX, "asym_a.rou.xml"),
    "asym_swap (ULTA axis)": os.path.join(SANDBOX, "asym_swap.rou.xml"),
}
SEEDS = list(range(201, 211))

env = SumoTrafficEnv(cfg)
env.reset(seed=1, options={"route_file": list(scenarios.values())[0]})
controlled = traci.trafficlight.getControlledLanes(env.tl_id)
phase_lanes = [sorted({controlled[i] for i, c in enumerate(st) if c in "Gg"})
               for st in env.green_states]

agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
agent.load(os.path.join(ROOT, "models", "dqn_v2_best.pt"))


def fixed10(obs):
    return (1 - env.green_idx) if env.phase_time >= 10 else env.green_idx


def maxpressure(obs):
    p = [sum(traci.lane.getLastStepHaltingNumber(l) for l in ls) for ls in phase_lanes]
    return max(range(len(p)), key=lambda i: p[i])


def ai(obs):
    return agent.act(obs, 0.0)


def run(policy, route):
    w, q, s = [], [], []
    for sd in SEEDS:
        obs, _ = env.reset(seed=sd, options={"route_file": route})
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(policy(obs))
            done = term or trunc
        w.append(info["avg_wait"]); q.append(info["avg_queue"]); s.append(info["switches"])
    n = len(w)
    return sum(w) / n, sum(q) / n, sum(s) / n


controllers = {"fixed10": fixed10, "maxpress": maxpressure, "AI_v2": ai}
results = {(c, s): run(p, r) for c, p in controllers.items() for s, r in scenarios.items()}

print(f"\nSeeds: {SEEDS[0]}-{SEEDS[-1]}\n")
for sname in scenarios:
    base = results[("fixed10", sname)][0]
    print(f"[{sname}]")
    for cname in controllers:
        w, q, s = results[(cname, sname)]
        print(f"  {cname:9s} | wait={w:6.2f} | ratio={w/base:4.2f} | queue={q:5.2f} | switches={s:4.1f}")
env.close()