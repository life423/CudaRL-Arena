import sys
import os
import pytest

# Add build directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../build/Release'))

@pytest.fixture(scope="module")
def cudarl_core_python():
    try:
        import cudarl_core_python
        return cudarl_core_python
    except ImportError as e:
        pytest.fail(f"Module import failed: {e}")

def test_basic_environment(cudarl_core_python):
    env = cudarl_core_python.Environment(5, 5)
    assert env.get_width() == 5
    assert env.get_height() == 5

    obs = env.reset()
    assert obs.shape == (5, 5)

    initial_x, initial_y = env.get_agent_x(), env.get_agent_y()
    env.step(0)  # Move up

    new_x, new_y = env.get_agent_x(), env.get_agent_y()
    assert (initial_x, initial_y) != (new_x, new_y)
    assert isinstance(env.get_reward(), (float, int))
