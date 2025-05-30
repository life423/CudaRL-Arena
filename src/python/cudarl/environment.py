"""
Environment module for CudaRL-Arena.

This module provides a Python interface to the CUDA-accelerated environment.
"""

import logging
import numpy as np
from typing import Tuple, Optional, Dict, Any

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
        # Import the environment at runtime to avoid circular import
        from . import CudaEnvironment, CUDA_AVAILABLE
        
        self.width = width
        self.height = height
        self.env_id = env_id
        self.cuda_available = CUDA_AVAILABLE
        
        # Create the actual backend environment
        self._backend_env = CudaEnvironment(width, height)
        
        logger.info(f"Created environment {env_id} with size {width}x{height} "
                   f"(CUDA: {self.cuda_available})")
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment to its initial state.
        
        Returns:
            Initial observation as a numpy array
        """
        obs = self._backend_env.reset()
        return self._process_observation(obs)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Integer action to take (0=up, 1=right, 2=down, 3=left)
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        result = self._backend_env.step(action)
        
        if isinstance(result, tuple) and len(result) == 4:
            # Already in the correct format (obs, reward, done, info)
            obs, reward, done, info = result
            return self._process_observation(obs), reward, done, info
        else:
            # Handle single observation return (from reset)
            obs = self._backend_env.get_observation()
            reward = self._backend_env.get_reward()
            done = self._backend_env.is_done()
            info = {
                'agent_x': self._backend_env.get_agent_x(),
                'agent_y': self._backend_env.get_agent_y()
            }
            return self._process_observation(obs), reward, done, info
    
    def _process_observation(self, obs) -> np.ndarray:
        """
        Process observation to ensure consistent format.
        
        Args:
            obs: Observation from backend (list or numpy array)
            
        Returns:
            Processed observation as 2D numpy array
        """
        if isinstance(obs, list):
            # Convert list to numpy array and reshape
            obs_array = np.array(obs, dtype=np.float32)
            return obs_array.reshape(self.height, self.width)
        elif isinstance(obs, np.ndarray):
            # Ensure correct shape
            if obs.ndim == 1:
                return obs.reshape(self.height, self.width)
            else:
                return obs
        else:
            logger.warning(f"Unexpected observation type: {type(obs)}")
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
            obs = self._process_observation(self._backend_env.get_observation())
            agent_x = self._backend_env.get_agent_x()
            agent_y = self._backend_env.get_agent_y()
            
            print('-' * (self.width * 2 + 1))
            for y in range(self.height):
                row = '|'
                for x in range(self.width):
                    if x == agent_x and y == agent_y:
                        row += 'A|'
                    else:
                        val = obs[y, x]
                        if val > 0.8:  # Goal
                            row += 'G|'
                        else:
                            row += ' |'
                print(row)
                print('-' * (self.width * 2 + 1))
            return None
        
        elif mode == 'rgb_array':
            # Return a simple RGB representation
            obs = self._process_observation(self._backend_env.get_observation())
            rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            # Set grid values
            rgb[:, :, 1] = (obs * 255).astype(np.uint8)  # Green channel
            
            # Set agent position
            agent_x = self._backend_env.get_agent_x()
            agent_y = self._backend_env.get_agent_y()
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
        self._backend_env = None
        logger.info(f"Closed environment {self.env_id}")
    
    @property
    def observation_space(self):
        """Gym-like observation space property."""
        return (self.height, self.width)
    
    @property
    def action_space(self):
        """Gym-like action space property."""
        return 4  # 4 discrete actions: up, right, down, left
