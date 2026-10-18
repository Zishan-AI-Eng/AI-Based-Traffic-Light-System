# AI-Based Traffic Light System

A reinforcement-learning traffic light controller (Deep Q-Network) built from scratch on top of the SUMO traffic simulator.

**Goal:** replace fixed-timer traffic lights with an AI that watches the traffic in real time and switches signals dynamically. One trained model should work on any intersection (2 signals up to 10+), handle edge cases, avoid flickering and starvation, and beat classical baselines.

---

## Current Status

- [x] Step 0: Fresh start, git initialized
- [x] Step 1: Environment setup verified
- [x] Step 2: Project structure created
- [x] Step 3: First map generated with `netgenerate`, viewed in `sumo-gui`
- [x] Step 4a: Python controls SUMO via libsumo
- [x] Step 4b: Fixed-time baseline measured (Baseline #1)
- [x] Step 4c: Python-controlled fixed-time reproduces SUMO default exactly (control verified)
- [ ] Step 5: Flexible traffic generation (volume levels + seeds)
- [ ] Step 6: Gymnasium environment (state, action, reward, episode, safety masks)
- [ ] Step 7: DQN agent on a single intersection (must beat baseline)
- [ ] Step 8: Multi-intersection + variable lanes (shared-parameter agents, masking)
- [ ] Step 9: Map generator (2 to 10 signals), curriculum, domain randomization
- [ ] Step 10: Edge-case scenarios (empty lane, traffic spike, blockage)
- [ ] Step 11: Max-Pressure baseline, evaluation on unseen maps
- [ ] Step 12: Smooth GUI demo, graphs, documentation

---

## Environment

| Item | Value |
|---|---|
| OS | Windows |
| Python | 3.11 |
| PyTorch | CPU only (no GPU) |
| SUMO | 1.28.0 |
| traci / libsumo | 1.28.0 (must match the installed SUMO version) |
| SUMO_HOME | `C:\Program Files (x86)\SUMO\` |
| Other libs | gymnasium, pettingzoo, numpy, pandas, matplotlib, tensorboard, pytest, pyyaml |

Setup:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .
python scripts/check_environment.py
```

Because there is no GPU: keep networks small (64-128 neurons), use `libsumo` (faster than `traci`), and use parallel environments for speed.

---

## Project Structure

```
AI-Based-Traffic-Light-System/
├── src/traffic_rl/
│   ├── envs/         SUMO wrapper: state, action, reward
│   ├── agents/       DQN, replay buffer, networks
│   ├── maps/         map generators
│   ├── scenarios/    edge-case traffic generators
│   ├── training/     trainer, curriculum, evaluation
│   ├── baselines/    fixed-time, max-pressure
│   └── utils/        logging, config
├── configs/
├── tests/
├── scripts/          standalone experiment scripts
├── data/
│   └── sandbox/      first test map (single.net.xml, routes.rou.xml, run.sumocfg)
├── runs/             logs and checkpoints (git-ignored)
├── pyproject.toml
└── requirements.txt
```

---

## SUMO Basics (what we learned)

| File | Purpose |
|---|---|
| `.net.xml` | Road network: roads, lanes, junctions, traffic light phases |
| `.rou.xml` | Vehicles and their routes |
| `.sumocfg` | Ties net + routes together, sets time and step length |

Commands used to build the sandbox (run inside `data/sandbox`):

```powershell
# 2x2 grid map
netgenerate --grid --grid.number 2 --grid.length 200 --default.lanenumber 2 --tls.guess true --output-file test.net.xml

# Spider map: 1 central 4-way junction + outer ring
netgenerate --spider --spider.arm-number 4 --spider.circle-number 1 --spider.space-radius 200 --default.lanenumber 2 --tls.guess true --output-file single.net.xml

# Random traffic: 600 s, one vehicle every 1.5 s
python "$env:SUMO_HOME\tools\randomTrips.py" -n single.net.xml -o trips.trips.xml -r routes.rou.xml -e 600 -p 1.5 --validate

# View in GUI
sumo-gui -c run.sumocfg
```

### `single.net.xml` facts

- 5 traffic lights: `A1` (central 4-way junction) and `B1` to `B4` (outer ring corners, 3-way).
- `A1` has 4 incoming roads (`B1A1`, `B2A1`, `B3A1`, `B4A1`), 2 lanes each = **8 unique lanes**.
- `A1` state string is 20 characters (one per lane-connection). `getControlledLanes` returns duplicates because one lane can have several connections (straight/left/right). Always deduplicate with `sorted(set(...))` when building the state.
- `A1` is a simple 2-phase junction (90 s cycle):

| Phase | Duration | State | Meaning |
|---|---|---|---|
| 0 | 42 s | `GGGggrrrrrGGGggrrrrr` | Axis 1 green |
| 1 | 3 s | `yyyyyrrrrryyyyyrrrrr` | Axis 1 yellow |
| 2 | 42 s | `rrrrrGGGggrrrrrGGGgg` | Axis 2 green |
| 3 | 3 s | `rrrrryyyyyrrrrryyyyy` | Axis 2 yellow |

`G` = protected green, `g` = green but yield, `y` = yellow, `r` = red.

**Design decision:** the AI only chooses between green phases (0 and 2). Yellow phases are inserted automatically by rule-based safety logic, never by the AI.

### Simulation settings

- Step length: **0.5 s** (1200 steps = 600 s = 10 minutes)
- Episode length used for baselines: 600 s

---

## Baseline #1: Fixed-Time (SUMO default)

Map: `single.net.xml`, traffic: `routes.rou.xml`, 600 s, junction `A1`.

| Metric | Value |
|---|---|
| Avg waiting time per step (A1, 8 lanes) | 64.72 |
| Avg queue per step (A1, 8 lanes) | 4.86 vehicles |
| Throughput (whole map) | 345 vehicles / 600 s |

**Verification:** a Python-controlled fixed-time controller (42/3/42/3, `scripts/baseline_python_control.py`) gives the exact same numbers (64.72 / 4.86 / 345). This proves Python can fully control the signal. The AI will replace the phase decision in that same loop.

**Target:** the trained AI must beat these numbers. If it does not, training has failed.

Caveats:
- Single seed only. Final comparisons will average many seeds.
- Throughput counts the whole map (including the ring road), not only `A1`.
- Always compare on the same map, traffic and seed.

---

## Scripts

| Script | Purpose |
|---|---|
| `scripts/check_environment.py` | Verify Python, Torch, SUMO, libsumo, package import |
| `scripts/first_control.py` | First libsumo connection test |
| `scripts/baseline_fixed.py` | Baseline with SUMO's built-in timer |
| `scripts/baseline_python_control.py` | Same baseline, timer controlled from Python |

Run from anywhere (paths are resolved from the script location):

```powershell
python scripts/baseline_fixed.py
```

---

## Architecture Plan

- **Algorithm:** DQN with upgrades (Double DQN, Dueling, Prioritized Replay, N-step, soft target update). PPO/MAPPO is the fallback if DQN hits limits.
- **Scalability:** shared-parameter multi-agent. Each intersection is an agent using the same network. Input is a local view of that intersection, so the same model runs on 2, 4 or 10 signals.
- **Variable lanes:** padding + masking, or an attention / set-based encoder.
- **State per lane:** queue length, waiting time, density/occupancy, average speed, plus current phase and time in phase.
- **Action:** choose among green phases (invalid ones masked). Yellow is automatic.
- **Reward:** negative waiting time and queue, fairness penalty (longest waiting vehicle), small switch penalty (anti-flicker).
- **Hard safety rules (action masking, not left to RL):** minimum green time, mandatory yellow, maximum red cap (anti-starvation).
- **Training strategy:** curriculum (simple to complex), domain randomization (random maps, volumes, lane counts), edge-case scenario bank, evaluation on unseen maps and seeds.
- **Baselines to beat:** fixed-time and Max-Pressure.
- **Simulation:** headless `libsumo` and parallel environments for training; `sumo-gui` with small step length and delay for demos.

---

## Troubleshooting Notes

- **`Could not access configuration ...run.sumocfg`:** the script was run from a different working directory. Build paths from `os.path.abspath(__file__)` instead of relative paths.
- **`Activate.ps1` blocked:** `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`
- **`SUMO_HOME` empty:** `setx SUMO_HOME "C:\Program Files (x86)\SUMO"` then reopen the terminal.
- **libsumo/traci version mismatch:** keep pip `libsumo`/`traci` equal to the installed SUMO version (`sumo --version`).
- Run scripts inside the activated `.venv` so the right Python and libraries are used.