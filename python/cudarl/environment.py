"""
Environment module for CudaRL-Arena.

This module provides a Python interface to the CUDA-accelerated environment.
"""

import logging
import numpy as np
from typing import Tuple, List, Optional, Dict, Any

# Configure logging
logger = logging.getLogger(__name__)

class Environment:
    """
    Python wrapper for the CUDA-accelerated environment.
    
    This class provides a gym-like interface to the underlying C++/CUDA environment.
    """
    
    def __init__(self, width: int = 10, height: int = 10, env_id: int = 0):
        """
        Initialize the environment.
        
        Args:
            width: Width of the grid environment
            height: Height of the grid environment
            env_id: Unique identifier for this environment instance
        """
        self.width = width
        self.height = height
        self.env_id = env_id
        self._cpp_env = None  # Will be set when C++ binding is available
        
        logger.info(f"Created environment {env_id} with size {width}x{height}")
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment to its initial state.
        
        Returns:
            Initial observation as a numpy array
        """
        if self._cpp_env:
            self._cpp_env.reset()
            return self._get_observation()
        else:
            logger.warning("C++ environment not initialized, returning dummy observation")
            return np.zeros((self.height, self.width), dtype=np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Integer action to take (0=up, 1=right, 2=down, 3=left)
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        if self._cpp_env:
            self._cpp_env.step(action)
            obs = self._get_observation()
            reward = self._cpp_env.get_reward()
            done = self._cpp_env.is_done()
            info = {
                'agent_x': self._cpp_env.get_agent_x(),
                'agent_y': self._cpp_env.get_agent_y()
            }
            return obs, reward, done, info
        else:
            logger.warning("C++ environment not initialized, returning dummy step result")
            return (
                np.zeros((self.height, self.width), dtype=np.float32),
                0.0,
                False,
                {'agent_x': 0, 'agent_y': 0}
            )
    
    def _get_observation(self) -> np.ndarray:
        """
        Get the current observation from the environment.
        
        Returns:
            Current grid state as a numpy array
        """
        if self._cpp_env:
            # Convert flat grid to 2D numpy array
            grid_data = self._cpp_env.get_grid_data()
            return np.array(grid_data).reshape(self.height, self.width)
        else:
            return np.zeros((self.height, self.width), dtype=np.float32)
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """
        Render the environment.
        
        Args:
            mode: Rendering mode ('human' or 'rgb_array')
            
        Returns:
            If mode is 'rgb_array', returns a numpy array of the rendered image
        """
        if mode == 'human':
            # Print ASCII representation of the grid
            grid = self._get_observation()
            agent_x = self._cpp_env.get_agent_x() if self._cpp_env else 0
            agent_y = self._cpp_env.get_agent_y() if self._cpp_env else 0
            
            print('-' * (self.width * 2 + 1))
            for y in range(self.height):
                row = '|'
                for x in range(self.width):
                    if x == agent_x and y == agent_y:
                        row += 'A|'
                    else:
                        val = grid[y, x]
                        if val > 0.8:  # Goal
                            row += 'G|'
                        else:
                            row += ' |'
                print(row)
                print('-' * (self.width * 2 + 1))
            return None
        
        elif mode == 'rgb_array':
            # Return a simple RGB representation
            grid = self._get_observation()
            rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            # Set grid values
            rgb[:, :, 1] = (grid * 255).astype(np.uint8)  # Green channel for grid values
            
            # Set agent position
            if self._cpp_env:
                agent_x = self._cpp_env.get_agent_x()
                agent_y = self._cpp_env.get_agent_y()
                rgb[agent_y, agent_x, 0] = 255  # Red for agent
            
            # Set goal (top-right corner)
            rgb[0, self.width-1, 2] = 255  # Blue for goal
            
            return rgb
        
        else:
            raise ValueError(f"Unsupported render mode: {mode}")
    
    def close(self) -> None:
        """
        Clean up resources.
        """
        # The C++ environment will be cleaned up by the Python garbage collector
        self._cpp_env = None
        logger.info(f"Closed environment {self.env_id}")