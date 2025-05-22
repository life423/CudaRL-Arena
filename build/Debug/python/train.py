"""Simple reinforcement learning training loop using the CUDA environment."""

import ctypes
import numpy as np
import os

# Load the shared library. In a real project this would be built from the CUDA
# sources in the `game` directory. Here we assume a library named `libarena.so`
# exists in a `build` directory one level above this file.

LIB_PATH = os.path.join(os.path.dirname(__file__), '..', 'build', 'libarena.so')
if os.path.exists(LIB_PATH):
    _lib = ctypes.CDLL(LIB_PATH)
    _step = _lib.step
    _step.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
else:
    _lib = None
    def _step(*args):
        raise RuntimeError('CUDA library not found. Compile the sources in game/.')


def train(num_steps: int = 10, width: int = 8, height: int = 8) -> None:
    """Run a tiny training loop calling the CUDA step function."""
    env = np.zeros((height, width), dtype=np.float32)
    ptr = env.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    for _ in range(num_steps):
        _step(ptr, width, height)
    print("Environment after", num_steps, "steps:\n", env)


if __name__ == "__main__":
    train()
