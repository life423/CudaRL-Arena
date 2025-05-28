"""
Pytest configuration and shared fixtures for CudaRL-Arena testing.
"""

import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Generator

import numpy as np
import pytest

# Add parent directory to path to import cudarl package
sys.path.insert(0, str(Path(__file__).parent.parent))

from python.cudarl import Environment  # noqa: E402

# Configure logging for tests
logging.basicConfig(level=logging.WARNING)


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU/CUDA functionality"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance benchmark"
    )
    config.addinivalue_line(
        "markers", "rtx5070: mark test as RTX 5070 specific"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark GPU tests
        if "cuda" in item.nodeid.lower() or "gpu" in item.nodeid.lower():
            item.add_marker(pytest.mark.gpu)

        # Mark slow tests
        if ("performance" in item.nodeid.lower() or
                "benchmark" in item.nodeid.lower()):
            item.add_marker(pytest.mark.slow)

        # Mark integration tests
        if ("integration" in item.nodeid.lower() or
                "end_to_end" in item.nodeid.lower()):
            item.add_marker(pytest.mark.integration)


@pytest.fixture(scope="session")
def cuda_available() -> bool:
    """Check if CUDA is available on the system."""
    try:
        import cupy  # noqa: F401
        cupy.cuda.runtime.getDeviceCount()
        return True
    except ImportError:
        try:
            # Fallback: check if our CUDA test executable exists and runs
            test_exe = (Path(__file__).parent / "bin" / "Release" /
                        "cuda_test.exe")
            if test_exe.exists():
                result = subprocess.run(
                    [str(test_exe)],
                    capture_output=True,
                    timeout=10
                )
                return result.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            pass
        return False


@pytest.fixture(scope="session")
def gpu_info() -> Dict[str, Any]:
    """Get GPU information for tests."""
    info = {
        "cuda_available": False,
        "device_count": 0,
        "devices": []
    }

    try:
        import cupy  # noqa: F401
        info["cuda_available"] = True
        info["device_count"] = cupy.cuda.runtime.getDeviceCount()

        for i in range(info["device_count"]):
            with cupy.cuda.Device(i):
                props = cupy.cuda.runtime.getDeviceProperties(i)
                device_info = {
                    "id": i,
                    "name": props["name"].decode(),
                    "compute_capability": (f"{props['major']}."
                                           f"{props['minor']}"),
                    "memory": props["totalGlobalMem"] // (1024**2),  # MB
                    "multiprocessors": props["multiProcessorCount"]
                }
                info["devices"].append(device_info)
    except ImportError:
        pass

    return info


@pytest.fixture
def small_env() -> Generator[Environment, None, None]:
    """Create a small environment for testing."""
    env = Environment(width=5, height=5, env_id=0)
    yield env
    env.close()


@pytest.fixture
def medium_env() -> Generator[Environment, None, None]:
    """Create a medium environment for testing."""
    env = Environment(width=10, height=10, env_id=1)
    yield env
    env.close()


@pytest.fixture
def large_env() -> Generator[Environment, None, None]:
    """Create a large environment for testing."""
    env = Environment(width=20, height=20, env_id=2)
    yield env
    env.close()


@pytest.fixture
def test_grid() -> np.ndarray:
    """Create a test grid with known properties."""
    grid = np.zeros((5, 5), dtype=np.float32)
    grid[0, 4] = 1.0  # Goal at top-right
    grid[2, 2] = -0.5  # Trap in center
    return grid


@pytest.fixture(scope="function")
def gpu_memory_tracker():
    """Track GPU memory usage during tests."""
    memory_info = {"start": None, "end": None, "peak": None}

    try:
        import cupy  # noqa: F401
        if cupy.cuda.runtime.getDeviceCount() > 0:
            cupy.cuda.runtime.deviceSynchronize()
            memory_info["start"] = cupy.get_default_memory_pool().used_bytes()
            yield memory_info
            cupy.cuda.runtime.deviceSynchronize()
            memory_info["end"] = cupy.get_default_memory_pool().used_bytes()
            memory_info["peak"] = cupy.get_default_memory_pool().total_bytes()
        else:
            yield memory_info
    except ImportError:
        yield memory_info


@pytest.fixture
def performance_timer():
    """Timer fixture for performance testing."""
    times = {}

    class Timer:
        def start(self, name: str):
            times[name] = time.perf_counter()

        def end(self, name: str) -> float:
            if name in times:
                duration = time.perf_counter() - times[name]
                return duration
            return 0.0

    return Timer()


class MockCppEnvironment:
    """Mock C++ environment for testing when real C++ binding not available."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.agent_x = width // 2
        self.agent_y = height // 2
        self.grid = np.zeros((height, width), dtype=np.float32)
        self.grid[0, width-1] = 1.0  # Goal at top-right
        self.reward = 0.0
        self.done = False

    def reset(self):
        self.agent_x = self.width // 2
        self.agent_y = self.height // 2
        self.reward = 0.0
        self.done = False

    def step(self, action: int):
        # Action: 0=up, 1=right, 2=down, 3=left
        dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][action]

        new_x = max(0, min(self.width - 1, self.agent_x + dx))
        new_y = max(0, min(self.height - 1, self.agent_y + dy))

        self.agent_x, self.agent_y = new_x, new_y

        # Calculate reward
        if self.agent_x == self.width - 1 and self.agent_y == 0:
            self.reward = 1.0
            self.done = True
        else:
            self.reward = -0.01  # Step penalty

    def get_reward(self) -> float:
        return self.reward

    def is_done(self) -> bool:
        return self.done

    def get_agent_x(self) -> int:
        return self.agent_x

    def get_agent_y(self) -> int:
        return self.agent_y

    def get_grid_data(self) -> np.ndarray:
        return self.grid.flatten()


@pytest.fixture
def mock_cpp_env():
    """Create a mock C++ environment for testing."""
    return MockCppEnvironment(5, 5)


@pytest.fixture
def env_with_cpp_mock(small_env, mock_cpp_env):
    """Environment with mocked C++ backend."""
    small_env._cpp_env = mock_cpp_env
    return small_env
