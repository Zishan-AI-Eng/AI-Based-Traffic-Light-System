import numpy as np

from traffic_light_system.config import SimulationConfig, seed_everything


def test_simulation_config_defaults() -> None:
    config = SimulationConfig()

    assert config.seed == 42
    assert config.yellow_duration_seconds == 3


def test_seed_helper_reproduces_numpy_values() -> None:
    seed_everything(123)
    first = np.random.random()

    seed_everything(123)
    second = np.random.random()

    assert first == second
