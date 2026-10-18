import os
import libsumo as traci
from traffic_rl.envs.sumo_env import SumoTrafficEnv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = os.path.join(ROOT, "data", "sandbox", "run.sumocfg")

env = SumoTrafficEnv(cfg)
print("Green phases:", env.green_phases)
print("Lanes:", len(env.lanes))
print("Observation shape:", env.observation_space.shape)
print("Action space:", env.action_space)

obs, _ = env.reset(seed=1)
print("\nAt t=0  :", obs.round(2))

for _ in range(200):          # 100 sec chalao (200 x 0.5s)
    traci.simulationStep()
obs = env._get_obs()
print("At t=100:", obs.round(2))
print("Min/Max:", obs.min(), obs.max())

obs, _ = env.reset(seed=2)    # dobara reset crash na kare
print("Second reset OK, shape:", obs.shape)

env.close()