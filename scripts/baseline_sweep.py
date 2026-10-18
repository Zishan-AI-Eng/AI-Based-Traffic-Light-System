import os
from traffic_rl.envs.sumo_env import SumoTrafficEnv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sandbox = os.path.join(ROOT, "data", "sandbox")
cfg = os.path.join(sandbox, "run.sumocfg")

traffic = {"medium": "test_a.rou.xml", "rush": "test_d.rou.xml", "asym": "asym_a.rou.xml"}

for name, rou in traffic.items():
    env = SumoTrafficEnv(cfg, route_file=os.path.join(sandbox, rou))
    print(f"\n=== traffic: {name} ===")
    for G in [10, 15, 20, 30, 45]:
        env.reset(seed=1)
        done = False
        while not done:
            action = (1 - env.green_idx) if env.phase_time >= G else env.green_idx
            _, r, term, trunc, info = env.step(action)
            done = term or trunc
        print(f"green={G:2d}s | switches={info['switches']:3d} | wait={info['avg_wait']:7.2f} | "
              f"queue={info['avg_queue']:6.2f} | arrived={info['arrived']:3d}")
    env.close()