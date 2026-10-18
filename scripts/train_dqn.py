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
}
BASELINE_WAIT = {"medium": 4.22, "rush": 26.00, "asym": 10.63}   # Baseline #2 (best fixed)

EPISODES = 300
EVAL_EVERY = 25
EPS_START, EPS_END = 1.0, 0.05
EPS_DECAY_EPISODES = int(EPISODES * 0.6)

env = SumoTrafficEnv(cfg)
agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
run_dir = os.path.join(ROOT, "runs", "dqn_v1")
os.makedirs(run_dir, exist_ok=True)
writer = SummaryWriter(run_dir)


def evaluate():
    out = {}
    for name, f in test_files.items():
        obs, _ = env.reset(seed=1, options={"route_file": f})
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(agent.act(obs, 0.0))
            done = term or trunc
        out[name] = info
    return out


best_score = float("inf")
print(f"Training files: {len(train_files)}")

for ep in range(1, EPISODES + 1):
    eps = max(EPS_END, EPS_START - (EPS_START - EPS_END) * ep / EPS_DECAY_EPISODES)
    obs, _ = env.reset(seed=random.randint(0, 10**6),
                       options={"route_file": random.choice(train_files)})
    ep_reward, losses, done = 0.0, [], False

    while not done:
        action = agent.act(obs, eps)
        nobs, r, term, trunc, info = env.step(action)
        done = term or trunc
        agent.buffer.add(obs, info["applied"], r, nobs, float(term))
        loss = agent.learn()
        if loss is not None:
            losses.append(loss)
        ep_reward += r
        obs = nobs

    writer.add_scalar("train/reward", ep_reward, ep)
    writer.add_scalar("train/avg_wait", info["avg_wait"], ep)
    writer.add_scalar("train/epsilon", eps, ep)
    if ep % 5 == 0:
        print(f"ep {ep:3d} | eps={eps:.2f} | reward={ep_reward:7.2f} | "
              f"wait={info['avg_wait']:6.2f} | switches={info['switches']}")

    if ep % EVAL_EVERY == 0:
        res = evaluate()
        score = sum(res[k]["avg_wait"] / BASELINE_WAIT[k] for k in res) / len(res)
        print(f"\n--- EVAL after ep {ep} (score {score:.2f}x baseline, lower is better) ---")
        for k, v in res.items():
            print(f"{k:7s} | wait={v['avg_wait']:6.2f} (base {BASELINE_WAIT[k]:5.2f}) | "
                  f"queue={v['avg_queue']:5.2f} | arrived={v['arrived']:3d} | "
                  f"switches={v['switches']}")
            writer.add_scalar(f"eval/{k}_wait", v["avg_wait"], ep)
        print()
        if score < best_score:
            best_score = score
            agent.save(os.path.join(run_dir, "best.pt"))
        agent.save(os.path.join(run_dir, "last.pt"))

agent.save(os.path.join(run_dir, "last.pt"))
env.close()
print("Done. Best score:", round(best_score, 2))

# best model ko git-tracked models/ folder mein copy karo
os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
shutil.copy(os.path.join(run_dir, "best.pt"),
            os.path.join(ROOT, "models", "dqn_v1_best.pt"))
print("Model copied to models/dqn_v1_best.pt")