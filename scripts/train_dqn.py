import os
import glob
import random
import shutil
from torch.utils.tensorboard import SummaryWriter
from traffic_rl.envs.sumo_env import SumoTrafficEnv
from traffic_rl.agents.dqn import DQNAgent

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SANDBOX = os.path.join(ROOT, "data", "sandbox")
cfg = os.path.join(SANDBOX, "run.sumocfg")

train_files = sorted(glob.glob(os.path.join(ROOT, "data", "generated", "train", "*.rou.xml")))
test_files = {
    "medium": os.path.join(SANDBOX, "test_a.rou.xml"),
    "rush": os.path.join(SANDBOX, "test_d.rou.xml"),
    "asym": os.path.join(SANDBOX, "asym_a.rou.xml"),
    "asym_swap": os.path.join(SANDBOX, "asym_swap.rou.xml"),
}
EVAL_SEEDS = [301, 302, 303, 304, 305]     # final test seeds (201-210) se alag

EPISODES = 800
EVAL_EVERY = 50
EPS_START, EPS_END = 1.0, 0.05
EPS_DECAY_EPISODES = int(EPISODES * 0.6)

env = SumoTrafficEnv(cfg)
agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
run_dir = os.path.join(ROOT, "runs", "dqn_v3")
os.makedirs(run_dir, exist_ok=True)
writer = SummaryWriter(run_dir)


def run_eval(policy):
    """Har test scenario par EVAL_SEEDS ka average. Baseline aur AI dono yahi use karte hain."""
    out = {}
    for name, f in test_files.items():
        w, q, a, s = [], [], [], []
        for sd in EVAL_SEEDS:
            obs, _ = env.reset(seed=sd, options={"route_file": f})
            done = False
            while not done:
                obs, _, term, trunc, info = env.step(policy(obs))
                done = term or trunc
            w.append(info["avg_wait"]); q.append(info["avg_queue"])
            a.append(info["arrived"]); s.append(info["switches"])
        n = len(w)
        out[name] = {"avg_wait": sum(w) / n, "avg_queue": sum(q) / n,
                     "arrived": int(sum(a) / n), "switches": round(sum(s) / n, 1)}
    return out


def fixed10(obs):
    return (1 - env.green_idx) if env.phase_time >= 10 else env.green_idx


def ai_policy(obs):
    return agent.act(obs, 0.0)


# Baseline wahi seeds par, script ke andar hi (seed mismatch nahi ho sakta)
print("Computing fixed-10s baseline on eval seeds...")
base = run_eval(fixed10)
BASELINE_WAIT = {k: v["avg_wait"] for k, v in base.items()}
print("Baseline wait:", {k: round(v, 2) for k, v in BASELINE_WAIT.items()})

best_score = float("inf")
print(f"Training files: {len(train_files)}")

for ep in range(1, EPISODES + 1):
    eps = max(EPS_END, EPS_START - (EPS_START - EPS_END) * ep / EPS_DECAY_EPISODES)
    obs, _ = env.reset(seed=random.randint(0, 10**6),
                       options={"route_file": random.choice(train_files)})
    ep_reward, done = 0.0, False

    while not done:
        action = agent.act(obs, eps)
        nobs, r, term, trunc, info = env.step(action)
        done = term or trunc
        agent.buffer.add(obs, info["applied"], r, nobs, float(term))
        agent.learn()
        ep_reward += r
        obs = nobs

    writer.add_scalar("train/reward", ep_reward, ep)
    writer.add_scalar("train/avg_wait", info["avg_wait"], ep)
    writer.add_scalar("train/epsilon", eps, ep)
    if ep % 10 == 0:
        print(f"ep {ep:3d} | eps={eps:.2f} | reward={ep_reward:7.2f} | "
              f"wait={info['avg_wait']:6.2f} | switches={info['switches']}")

    if ep % EVAL_EVERY == 0:
        res = run_eval(ai_policy)
        ratios = {k: res[k]["avg_wait"] / BASELINE_WAIT[k] for k in res}
        score = sum(ratios.values()) / len(ratios)
        print(f"\n--- EVAL after ep {ep} (score {score:.2f}x fixed10, lower is better) ---")
        for k, v in res.items():
            print(f"{k:10s} | wait={v['avg_wait']:6.2f} (base {BASELINE_WAIT[k]:5.2f}) | "
                  f"ratio={ratios[k]:4.2f} | queue={v['avg_queue']:5.2f} | "
                  f"switches={v['switches']}")
            writer.add_scalar(f"eval/{k}_ratio", ratios[k], ep)
        writer.add_scalar("eval/score", score, ep)
        print()
        if score < best_score:
            best_score = score
            agent.save(os.path.join(run_dir, "best.pt"))
        agent.save(os.path.join(run_dir, "last.pt"))

agent.save(os.path.join(run_dir, "last.pt"))
env.close()
print("Done. Best score:", round(best_score, 2))

os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
shutil.copy(os.path.join(run_dir, "best.pt"),
            os.path.join(ROOT, "models", "dqn_v3_best.pt"))
print("Model copied to models/dqn_v3_best.pt")