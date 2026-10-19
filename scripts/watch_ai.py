"""
AI (ya baseline) ko sumo-gui mein live dekhne ke liye.

Examples (project root se):
  python scripts/watch_ai.py
  python scripts/watch_ai.py --scenario asym_swap --delay 50
  python scripts/watch_ai.py --controller fixed --scenario asym      # same traffic, fixed-time
  python scripts/watch_ai.py --controller maxpressure --scenario asym
"""
import os
import argparse
import traci                                  # GUI ke liye libsumo nahi, asli traci
import traffic_rl.envs.sumo_env as env_module
from traffic_rl.envs.sumo_env import SumoTrafficEnv
from traffic_rl.agents.dqn import DQNAgent

# Env ke andar jo `traci` use hota hai usay libsumo se asli traci par badal do
env_module.traci = traci

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SANDBOX = os.path.join(ROOT, "data", "sandbox")
CFG = os.path.join(SANDBOX, "run.sumocfg")

SCENARIOS = {
    "medium": "test_a.rou.xml",
    "rush": "test_d.rou.xml",
    "asym": "asym_a.rou.xml",
    "asym_swap": "asym_swap.rou.xml",
}


class GuiEnv(SumoTrafficEnv):
    """Wahi env, bas episode sumo-gui mein chalta hai."""

    def __init__(self, *args, delay=80, **kwargs):
        self._use_gui = False          # structure discovery ke waqt GUI nahi kholna
        self._delay = delay
        super().__init__(*args, **kwargs)
        self._use_gui = True

    def _start(self, seed):
        binary = "sumo-gui" if self._use_gui else "sumo"
        cmd = [binary, "-c", self.cfg_file, "--step-length", str(self.step_length),
               "--no-warnings", "true", "--seed", str(seed)]
        if self.route_file:
            cmd += ["--route-files", self.route_file]
        if self._use_gui:
            cmd += ["--delay", str(self._delay), "--start", "--window-size", "1280,800"]
        traci.start(cmd)
        self._running = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=SCENARIOS, default="asym")
    ap.add_argument("--controller", choices=["ai", "fixed", "maxpressure"], default="ai")
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "dqn_v3_best.pt"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--delay", type=int, default=80, help="ms per sim step (chota = tez)")
    args = ap.parse_args()

    route = os.path.join(SANDBOX, SCENARIOS[args.scenario])
    env = GuiEnv(CFG, delay=args.delay)

    agent = None
    if args.controller == "ai":
        agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
        agent.load(args.model)

    obs, _ = env.reset(seed=args.seed, options={"route_file": route})

    phase_lanes = None
    if args.controller == "maxpressure":
        controlled = traci.trafficlight.getControlledLanes(env.tl_id)
        phase_lanes = [sorted({controlled[i] for i, c in enumerate(st) if c in "Gg"})
                       for st in env.green_states]

    def choose(obs):
        if args.controller == "ai":
            return agent.act(obs, 0.0)
        if args.controller == "fixed":
            return (1 - env.green_idx) if env.phase_time >= 10 else env.green_idx
        p = [sum(traci.lane.getLastStepHaltingNumber(l) for l in ls) for ls in phase_lanes]
        return max(range(len(p)), key=lambda i: p[i])

    print(f"\nController: {args.controller} | scenario: {args.scenario} | seed: {args.seed}")
    print("Green phase index: 0 = pehla axis, 1 = doosra axis\n")

    done = False
    info = {}
    while not done:
        want = choose(obs)
        obs, _, term, trunc, info = env.step(want)
        done = term or trunc
        t = traci.simulation.getTime()
        waiting = sum(traci.lane.getLastStepHaltingNumber(l) for l in env.lanes)
        note = "" if info["applied"] == want else "  (safety rule ne override kiya)"
        print(f"t={t:5.0f}s | AI chahta tha={want} | laga={info['applied']} | "
              f"ruki gaadiyan={waiting:3d} | switches={info['switches']:2d}{note}")

    print("\n=== Episode khatam ===")
    print(f"avg wait  : {info['avg_wait']:.2f}")
    print(f"avg queue : {info['avg_queue']:.2f}")
    print(f"arrived   : {info['arrived']}")
    print(f"switches  : {info['switches']}")
    input("\nEnter dabao (GUI band ho jayega)...")
    env.close()


if __name__ == "__main__":
    main()