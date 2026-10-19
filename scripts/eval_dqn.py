import os
import random
from traffic_rl.envs.sumo_env import SumoTrafficEnv
from traffic_rl.agents.dqn import DQNAgent

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SANDBOX = os.path.join(ROOT, "data", "sandbox")
cfg = os.path.join(SANDBOX, "run.sumocfg")

scenarios = {
    "medium": os.path.join(SANDBOX, "test_a.rou.xml"),
    "rush": os.path.join(SANDBOX, "test_d.rou.xml"),
    "asym": os.path.join(SANDBOX, "asym_a.rou.xml"),
}
BASE = {"medium": 4.22, "rush": 26.00, "asym": 10.63}

env = SumoTrafficEnv(cfg)
agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
agent.load(os.path.join(ROOT, "models", "dqn_v1_best.pt"))

seeds = [11, 22, 33, 44, 55]    # nayi seeds, training mein nahi use hui
for name, f in scenarios.items():
    waits, queues, sw = [], [], []
    for s in seeds:
        obs, _ = env.reset(seed=s, options={"route_file": f})
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(agent.act(obs, 0.0))
            done = term or trunc
        waits.append(info["avg_wait"]); queues.append(info["avg_queue"]); sw.append(info["switches"])
    w = sum(waits) / len(waits)
    verdict = "BETTER" if w < BASE[name] else "WORSE"
    print(f"{name:7s} | AI wait={w:6.2f} vs base {BASE[name]:5.2f} -> {verdict} | "
          f"queue={sum(queues)/len(queues):5.2f} | switches={sum(sw)/len(sw):4.1f} (base 46)")

env.close()
