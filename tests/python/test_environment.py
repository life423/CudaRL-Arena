import pytest
import sys, os
import numpy as np

# Pytest fixture for importing the module
@pytest.fixture(scope="module")
def cudarl_env():
    sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../../../build/Release")))
    try:
        import cudarl_core_python
        return cudarl_core_python
    except ImportError as e:
        pytest.fail(f"Failed to import cudarl_core_python module: {e}")

# Pytest fixture for creating environment
@pytest.fixture
def env(cudarl_env):
    env = cudarl_env.Environment(5, 5)
    yield env  # this provides the environment instance to your tests

def test_environment_state_access(env):
    """Test getters and observation shape."""
    obs = env.reset()

    assert env.get_width() == 5, "Width should be 5"
    assert env.get_height() == 5, "Height should be 5"
    
    assert isinstance(env.get_agent_x(), int), "Agent X should be integer"
    assert isinstance(env.get_agent_y(), int), "Agent Y should be integer"
    assert isinstance(env.get_reward(), (float, int)), "Reward should be numeric"
    assert isinstance(env.is_done(), bool), "Done should be boolean"
    
    obs_from_get = env.get_observation()
    assert isinstance(obs_from_get, np.ndarray), "Observation should be numpy array"
    assert obs_from_get.shape == obs.shape, "Observation shapes must match"

def test_environment_reset(env):
    """Test reset returns valid observation."""
    obs = env.reset()
    assert isinstance(obs, np.ndarray), "Reset observation must be numpy array"

def test_minimal_step(env):
    """Test single step."""
    env.reset()

    initial_pos = (env.get_agent_x(), env.get_agent_y())
    result = env.step(1)  # perform an action (e.g., action=1)

    new_pos = (env.get_agent_x(), env.get_agent_y())
    
    assert initial_pos != new_pos, "Position should change after step"
    assert isinstance(result, tuple), "Step should return a tuple (obs, reward, done, info)"
    assert len(result) == 4, "Step result must have exactly 4 elements"
