# AI-Driven Dynamic Traffic Light Control System via Deep Reinforcement Learning

## 📌 Executive Summary
This project implements an advanced, dynamically adaptive Traffic Light Control System using a Deep Q-Network (DQN). The objective is to eliminate traditional, inefficient cyclic phasing and replace it with an AI agent capable of real-time, observation-based phase selection. The system is designed to operate within a 4-way intersection (2 lanes per approach) simulated via Eclipse SUMO. 

The architecture relies heavily on custom Reward Shaping (incorporating squared penalties and action-switching penalties) to prevent edge cases such as lane starvation, flickering, and gridlocks.

## 🏗️ System Architecture & Markov Decision Process (MDP)

The core intelligence of this system is modeled as an MDP. The Deep Q-Network acts as the policy approximator, mapping complex intersection states to optimal traffic phases.

### 1. State Space Representation (Input Tensor)
The environment must provide a 16-dimensional observation vector to the DQN at every decision step. This state comprehensively captures the real-time traffic snapshot and historical context to prevent starvation.
The 16 dimensions are categorized as follows:
- **[0-3] Queue Lengths:** The number of halted vehicles at each of the 4 incoming approaches.
- **[4-7] Cumulative Wait Times:** The total accumulated waiting time (in seconds) of all halted vehicles at each respective approach.
- **[8-11] Current Phase Encoding:** A One-Hot encoded vector representing the currently active green phase (e.g., `[0, 0, 1, 0]` implies Phase 3 is active). This gives the agent temporal awareness of its own immediate past action.
- **[12-15] Time Since Last Green:** A highly critical metric tracking the elapsed time since each approach last received a green light. This specific metric provides the agent with "memory," allowing it to recognize and mitigate starvation in low-density lanes.

### 2. Action Space (Output Tensor)
The DQN outputs Q-values for 4 discrete actions. The system employs **Dynamic Phase Selection**, meaning the agent is NOT bound by any sequence or cycle. It can transition from any phase to any phase.
- **Action 0:** Activate Green for Approach 1 (North)
- **Action 1:** Activate Green for Approach 2 (East)
- **Action 2:** Activate Green for Approach 3 (South)
- **Action 3:** Activate Green for Approach 4 (West)

**Environmental Safety Constraint:** If the agent selects an action $A_t$ that is different from $A_{t-1}$ (a phase switch), the environment will automatically enforce a strict 3-second Yellow (transition) phase before turning the new approach green.

### 3. Custom Reward Shaping & Optimization Objectives
The most critical component of the architecture is the reward function, mathematically shaped to prevent AI exploitation (loopholes):

- **The Squared Penalty (Preventing Starvation):** 
  Instead of a linear penalty, the agent receives a negative reward equal to the sum of the squared waiting times of all approaches:
  $Reward = - \sum (WaitTime_i)^2$
  *Architectural intent:* By squaring the wait time, exponential pressure is applied to the network. An approach waiting for a long time will generate a massive penalty, forcing the agent to interrupt the current flow and prioritize the starving lane, regardless of vehicle count.

- **The Action-Switching Penalty (Anti-Flickering):**
  A fixed negative scalar penalty (e.g., $-10$) is deducted from the reward every time $A_t \neq A_{t-1}$. 
  *Architectural intent:* This artificially establishes a "Minimum Green Time." The agent learns mathematically that it should only switch a light when the squared penalty of the waiting vehicles outweighs the fixed switching penalty. This completely eliminates high-frequency light flickering.

## 🧠 Neural Network & Deep Learning Specifications
To ensure the DQN converges safely in a highly stochastic traffic environment, the following design principles must be adhered to during implementation:
- **Huber Loss Implementation:** Vehicle densities can spike dramatically during simulated rush hours. Standard Mean Squared Error (MSE) will cause exploding gradients. The DQN must use Huber Loss to maintain stability against massive penalty spikes.
- **Experience Replay Buffer:** The agent must store state-action-reward transitions in a deque buffer and sample randomized mini-batches to break the temporal correlation of traffic flow.
- **Target Network:** A cloned Target Network must be utilized for Q-value target calculation to prevent moving-target oscillations, synchronized via hard or soft updates at regular intervals.
- **Exploration Strategy:** An Epsilon-Greedy strategy with an explicitly defined decay rate to ensure the agent explores various traffic loads before exploiting the learned policy.

## ⚙️ Simulation Environment Integration
- **Platform:** Eclipse SUMO (Simulation of Urban MObility)
- **Wrapper:** `sumo-rl` library to expose the SUMO simulation via the standard Gymnasium API (`reset`, `step`, `observation_space`, `action_space`).
- **Intersection Topology:** 4-way cross intersection with dual lanes per approach.

## 🛠️ Phase 1: Project and Environment Foundation

The initial project scaffold uses Python 3.11 or 3.12 and keeps the simulation,
agent, and tests under separate directories:

- `src/traffic_light_system/`: reusable project modules and shared configuration.
- `scripts/`: environment checks and executable SUMO smoke tests.
- `tests/`: automated validation.
- `outputs/`, `models/`, and `logs/`: local generated artifacts, excluded from Git.

Install the Python dependencies in a virtual environment:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Install Eclipse SUMO separately, then make sure `sumo`, `sumo-gui`,
`netconvert`, and `randomTrips.py` are available on `PATH`. On Windows,
`SUMO_HOME` should point to the SUMO installation directory when required by
TraCI tooling.

Run the prerequisite check and SUMO startup smoke test from the project root:

```powershell
python scripts\check_environment.py
python scripts\sumo_smoke_test.py
python -m pytest
```

The current machine must have SUMO installed before the smoke test can pass.
The checker intentionally reports missing SUMO commands explicitly rather than
silently treating the simulation as available.