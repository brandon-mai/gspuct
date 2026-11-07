import pytest
from mcts_forest.envs.gym_adapter import GymAdapter

@pytest.fixture(scope="function")
def deterministic_frozen_lake():
    """Returns a deterministic 4x4 FrozenLake environment."""
    custom_map = [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ]
    return GymAdapter("FrozenLake-v1", desc=custom_map, is_slippery=False)

@pytest.fixture(scope="function")
def stochastic_frozen_lake():
    """Returns a stochastic (slippery) 4x4 FrozenLake environment."""
    return GymAdapter("FrozenLake-v1", map_name="4x4", is_slippery=True)
