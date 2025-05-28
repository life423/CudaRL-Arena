"""
Mock Environment for CPU-only fallback when CUDA is not available.

This provides the same interface as the CUDA environment but runs entirely on CPU.
"""

import numpy as np
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class MockEnvironment:
    """
    Pure Python implementation of the gridworld environment.
    
    This serves as a fallback when the CUDA environment is not available.
    """
    
    def __init__(self, width: int = 10, height: int = 10):
        """
        Initialize the mock environment.
        
        Args:
            width: Width of the grid environment
            height: Height of the grid environment
        """
        self.width = width
        self.height = height
        
        # Initialize grid with random values
        self.grid = np.random.uniform(0.0, 0.5, (height, width)).astype(np.float32)
        # Set goal at top-right corner
        self.grid[0, width-1] = 1.0
        
        # Agent state
        self.agent_x = width // 2
        self.agent_y = height // 2
        self.reward = 0.0
        self.done = False
        
        logger.info(f"Mock environment initialized ({width}x{height}) - CPU fallback mode")
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment to its initial state.
        
        Returns:
            Initial observation as a numpy array
        """
        self.agent_x = self.width // 2
        self.agent_y = self.height // 2
        self.reward = 0.0
        self.done = False
        
        logger.debug("Mock environment reset")
        return self.get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Integer action to take (0=up, 1=right, 2=down, 3=left)
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Action mapping: 0=up, 1=right, 2=down, 3=left
        dx, dy = 0, 0
        if action == 0:  # up
            dy = -1
        elif action == 1:  # right
            dx = 1
        elif action == 2:  # down
            dy = 1
        elif action == 3:  # left
            dx = -1
        
        # Update agent position with bounds checking
        new_x = self.agent_x + dx
        new_y = self.agent_y + dy
        
        if 0 <= new_x < self.width and 0 <= new_y < self.height:
            self.agent_x = new_x
            self.agent_y = new_y
        
        # Simple reward: -0.01 per step, +1 for reaching goal
        self.reward = -0.01
        
        # Check if agent reached goal (top-right corner)
        if self.agent_x == self.width - 1 and self.agent_y == 0:
            self.reward = 1.0
            self.done = True
        
        info = {
            'agent_x': self.agent_x,
            'agent_y': self.agent_y
        }
        
        return self.get_observation(), self.reward, self.done, info
    
    def get_observation(self) -> np.ndarray:
        """
        Get the current observation.
        
        Returns:
            Current grid state as a numpy array
        """
        return self.grid.copy()
    
    def get_width(self) -> int:
        return self.width
    
    def get_height(self) -> int:
        return self.height
    
    def get_agent_x(self) -> int:
        return self.agent_x
    
    def get_agent_y(self) -> int:
        return self.agent_y
    
    def get_reward(self) -> float:
        return self.reward
    
    def is_done(self) -> bool:
        return self.done
    
    def get_agent_position(self) -> Tuple[int, int]:
        return (self.agent_x, self.agent_y)
