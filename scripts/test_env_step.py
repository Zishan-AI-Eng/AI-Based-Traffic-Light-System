import os
import random
from traffic_rl.envs.sumo_env import SumoTrafficEnv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = os.path.join(ROOT, "data", "sandbox", "run.sumocfg")

env = SumoTrafficEnv(cfg)


def run(name, policy):
    obs, _ = env.reset(seed=1)
    total_reward, decisions, done = 0.0, 0, False
    while not done:
        obs, r, term, trunc, info = env.step(policy(obs, decisions))
        total_reward += r
        decisions += 1
        done = term or trunc
    print(f"{name:14s} | decisions={decisions:3d} | switches={info['switches']:3d} | "
          f"wait={info['avg_wait']:6.2f} | queue={info['avg_queue']:5.2f} | "
          f"arrived={info['arrived']:3d} | reward={total_reward:7.2f}")


run("random",     lambda obs, i: random.randint(0, 1))
run("spam-switch", lambda obs, i: i % 2)             # har step switch maangti hai
run("fixed ~45s", lambda obs, i: (i // 9) % 2)       # har 9 decisions (~45s) baad switch
run("never switch", lambda obs, i: 0)                # hamesha ek hi phase

env.close()