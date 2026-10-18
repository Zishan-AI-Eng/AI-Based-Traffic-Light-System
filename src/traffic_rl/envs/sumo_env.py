import numpy as np
import gymnasium as gym
from gymnasium import spaces
import libsumo as traci

MAX_QUEUE = 20.0
MAX_WAIT = 200.0
MAX_PHASE_TIME = 60.0
VEH_LENGTH = 7.5
SWITCH_PENALTY = 0.05
FAIRNESS_WEIGHT = 0.5


class SumoTrafficEnv(gym.Env):
    def __init__(self, cfg_file, tl_id="A1", route_file=None, delta_time=5,
                 yellow_time=3, min_green=10, max_red=60,
                 episode_length=600, step_length=0.5):
        super().__init__()
        self.cfg_file = cfg_file
        self.tl_id = tl_id
        self.route_file = route_file
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_red = max_red
        self.episode_length = episode_length
        self.step_length = step_length
        self._running = False

        # SUMO ek dafa start karke structure nikalo
        self._start(seed=0)
        logic = traci.trafficlight.getAllProgramLogics(tl_id)[0]
        self.green_phases = [
            i for i, p in enumerate(logic.phases)
            if "y" not in p.state and ("G" in p.state or "g" in p.state)
        ]
        self.green_states = [logic.phases[i].state for i in self.green_phases]
        self.lanes = sorted(set(traci.trafficlight.getControlledLanes(tl_id)))
        self.lane_len = [traci.lane.getLength(l) for l in self.lanes]
        self.lane_vmax = [traci.lane.getMaxSpeed(l) for l in self.lanes]
        self._close()

        n = len(self.green_phases)
        obs_size = len(self.lanes) * 4 + n + 1
        self.observation_space = spaces.Box(0.0, 1.0, shape=(obs_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(n)

        self._reset_internal()

    def _reset_internal(self):
        self.green_idx = 0
        self.phase_time = 0.0
        self.red_time = [0.0] * len(self.green_phases)
        self.switches = 0
        self.m_wait = 0.0
        self.m_queue = 0.0
        self.m_steps = 0
        self.m_arrived = 0

    # ---------- SUMO start/stop ----------
    def _start(self, seed):
        cmd = ["sumo", "-c", self.cfg_file, "--step-length", str(self.step_length),
               "--no-warnings", "true", "--seed", str(seed)]
        if self.route_file:
            cmd += ["--route-files", self.route_file]
        traci.start(cmd)
        self._running = True

    def _close(self):
        if self._running:
            traci.close()
            self._running = False

    def close(self):
        self._close()

    # ---------- State ----------
    def _lane_stats(self):
        stats = []
        for ln, length, vmax in zip(self.lanes, self.lane_len, self.lane_vmax):
            queue = traci.lane.getLastStepHaltingNumber(ln) / MAX_QUEUE
            wait = traci.lane.getWaitingTime(ln) / MAX_WAIT
            density = traci.lane.getLastStepVehicleNumber(ln) / max(length / VEH_LENGTH, 1.0)
            speed = traci.lane.getLastStepMeanSpeed(ln) / vmax
            stats.append((queue, wait, density, speed))
        return stats

    def _get_obs(self):
        feats = []
        for s in self._lane_stats():
            feats += list(s)
        phase_onehot = [0.0] * len(self.green_phases)
        phase_onehot[self.green_idx] = 1.0
        feats += phase_onehot
        feats.append(self.phase_time / MAX_PHASE_TIME)
        return np.clip(np.array(feats, dtype=np.float32), 0.0, 1.0)

    def _reward(self):
        stats = self._lane_stats()
        queue = np.mean([s[0] for s in stats])
        wait = np.mean([s[1] for s in stats])
        worst = max(s[1] for s in stats)       # fairness: sabse zyada wait
        return -(queue + wait + FAIRNESS_WEIGHT * worst)

    # ---------- Simulation helpers ----------
    def _advance(self, seconds):
        for _ in range(int(round(seconds / self.step_length))):
            traci.simulationStep()
            self.m_steps += 1
            self.m_arrived += traci.simulation.getArrivedNumber()
            for ln in self.lanes:
                self.m_wait += traci.lane.getWaitingTime(ln)
                self.m_queue += traci.lane.getLastStepHaltingNumber(ln)

    def _set_green(self, idx):
        traci.trafficlight.setRedYellowGreenState(self.tl_id, self.green_states[idx])

    def _set_yellow(self, idx):
        state = self.green_states[idx].replace("G", "y").replace("g", "y")
        traci.trafficlight.setRedYellowGreenState(self.tl_id, state)

    def _safe_action(self, action):
        """Hard safety rules: AI ka action yahan filter hota hai."""
        if self.phase_time < self.min_green:
            return self.green_idx                  # min green
        for idx in range(len(self.green_phases)):
            if idx != self.green_idx and self.red_time[idx] >= self.max_red:
                return idx                         # starvation rokna
        return action

    # ---------- Gymnasium API ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options and "route_file" in options:
            self.route_file = options["route_file"]
        self._close()
        self._start(seed if seed is not None else 0)
        self._reset_internal()
        self._set_green(0)
        return self._get_obs(), {}

    def step(self, action):
        action = self._safe_action(int(action))
        switched = action != self.green_idx
        elapsed = 0.0

        if switched:
            self._set_yellow(self.green_idx)
            self._advance(self.yellow_time)
            elapsed += self.yellow_time
            self._set_green(action)
            self.green_idx = action
            self.phase_time = 0.0
            self.switches += 1

        self._advance(self.delta_time)
        elapsed += self.delta_time
        self.phase_time += self.delta_time

        for idx in range(len(self.red_time)):
            self.red_time[idx] = 0.0 if idx == self.green_idx else self.red_time[idx] + elapsed

        reward = self._reward() - (SWITCH_PENALTY if switched else 0.0)
        truncated = traci.simulation.getTime() >= self.episode_length
        steps = max(self.m_steps, 1)
        info = {
            "applied": action,
            "switches": self.switches,
            "avg_wait": self.m_wait / steps,
            "avg_queue": self.m_queue / steps,
            "arrived": self.m_arrived,
        }
        return self._get_obs(), reward, False, truncated, info