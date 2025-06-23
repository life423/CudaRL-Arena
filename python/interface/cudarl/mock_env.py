"""
CPU‐only fallback environment for testing.
"""
import numpy as np

class MockEnvironment:
    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height
        self.agent_x = 0
        self.agent_y = 0

    def reset(self):
        self.agent_x = self.agent_y = 0
        return np.zeros((self.height, self.width), dtype=np.float32)

    def step(self, action):
        # very basic random walk stub
        self.agent_x = (self.agent_x + 1) % self.width
        obs = np.zeros((self.height, self.width), dtype=np.float32)
        obs[self.agent_y, self.agent_x] = 1.0
        return obs, 0.0, False, {}
