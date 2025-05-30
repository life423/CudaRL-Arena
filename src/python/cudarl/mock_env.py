"""
Mock Environment for testing and development without CUDA dependencies.

This module provides a pure Python implementation of the CudaRL environment
that mimics the behavior of the CUDA-accelerated version for testing purposes.
"""

import numpy as np
import random
from typing import Tuple, List, Dict, Any, Optional


class MockEnvironment:
    """Mock environment that simulates the CudaRL environment behavior."""
    
    def __init__(
        self,
        width: int = 10,
        height: int = 10,
        obstacle_density: float = 0.2,
        goal_reward: float = 1.0,
        step_penalty: float = -0.01,
        obstacle_penalty: float = -0.5
    ):
        self.width = width
        self.height = height
        self.obstacle_density = obstacle_density
        self.goal_reward = goal_reward
        self.step_penalty = step_penalty
        self.obstacle_penalty = obstacle_penalty
        
        # Environment state
        self.grid = np.zeros((height, width), dtype=np.float32)
        self.agent_x = 0
        self.agent_y = 0
        self.goal_x = width - 1
        self.goal_y = height - 1
        
        # Episode tracking
        self.done = False
        self.episode_steps = 0
        self.max_episode_steps = 200
        self.total_reward = 0.0
        self.current_reward = 0.0
        
        # Action space: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        self.action_space = 4
        self.observation_space = width * height
        
        # Initialize environment
        self._generate_environment()
        
    def _generate_environment(self):
        """Generate the grid environment."""
        # Clear grid
        self.grid.fill(0.0)
        
        # Place obstacles randomly
        for y in range(self.height):
            for x in range(self.width):
                if random.random() < self.obstacle_density:
                    # Skip goal and start positions
                    if (x == 0 and y == 0) or (x == self.goal_x and y == self.goal_y):
                        continue
                    self.grid[y, x] = 0.9  # Obstacle marker
                    
        # Place goal
        self.grid[self.goal_y, self.goal_x] = 1.0  # Goal marker
        
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        # Reset agent position
        self.agent_x = 0
        self.agent_y = 0
        
        # Reset episode state
        self.done = False
        self.episode_steps = 0
        self.total_reward = 0.0
        self.current_reward = 0.0
        
        # Regenerate environment occasionally
        if random.random() < 0.1:  # 10% chance to regenerate
            self._generate_environment()
        
        return self._get_observation()
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        if self.done:
            return self._get_observation(), 0.0, True, {}
            
        self.episode_steps += 1
        
        # Calculate new position
        new_x, new_y = self.agent_x, self.agent_y
        
        if action == 0:  # UP
            new_y = max(0, self.agent_y - 1)
        elif action == 1:  # DOWN
            new_y = min(self.height - 1, self.agent_y + 1)
        elif action == 2:  # LEFT
            new_x = max(0, self.agent_x - 1)
        elif action == 3:  # RIGHT
            new_x = min(self.width - 1, self.agent_x + 1)
            
        # Check if move is valid (not into obstacle)
        reward = self.step_penalty
        
        if self.grid[new_y, new_x] == 0.9:  # Hit obstacle
            reward = self.obstacle_penalty
            # Don't move
        else:
            # Move agent
            self.agent_x = new_x
            self.agent_y = new_y
            
            # Check if reached goal
            if self.agent_x == self.goal_x and self.agent_y == self.goal_y:
                reward = self.goal_reward
                self.done = True
                
        # Check if episode should end
        if self.episode_steps >= self.max_episode_steps:
            self.done = True
            
        self.current_reward = reward
        self.total_reward += reward
        
        info = {
            'episode_steps': self.episode_steps,
            'agent_position': (self.agent_x, self.agent_y),
            'goal_position': (self.goal_x, self.goal_y),
            'success': self.agent_x == self.goal_x and self.agent_y == self.goal_y
        }
        
        return self._get_observation(), reward, self.done, info
        
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        # Create observation with agent position marked
        obs = self.grid.copy()
        if not self.done:
            obs[self.agent_y, self.agent_x] = 0.5  # Agent marker
        return obs.flatten()
        
    # Compatibility methods for C++ interface
    def get_observation(self) -> np.ndarray:
        """Get current observation (C++ compatibility)."""
        return self._get_observation()
        
    def get_reward(self) -> float:
        """Get current reward (C++ compatibility)."""
        return self.current_reward
        
    def is_done(self) -> bool:
        """Check if episode is done (C++ compatibility)."""
        return self.done
        
    def get_agent_x(self) -> int:
        """Get agent X position (C++ compatibility)."""
        return self.agent_x
        
    def get_agent_y(self) -> int:
        """Get agent Y position (C++ compatibility)."""
        return self.agent_y
        
    def render(self, mode: str = 'human'):
        """Render the environment."""
        if mode == 'human':
            print("\n" + "="*40)
            for y in range(self.height):
                row = ""
                for x in range(self.width):
                    if x == self.agent_x and y == self.agent_y:
                        row += "A "  # Agent
                    elif x == self.goal_x and y == self.goal_y:
                        row += "G "  # Goal
                    elif self.grid[y, x] == 0.9:
                        row += "# "  # Obstacle
                    else:
                        row += ". "  # Empty
                print(row)
            print(f"Steps: {self.episode_steps}, Reward: {self.total_reward:.2f}")
            print("="*40)
            
    def get_state_info(self) -> Dict[str, Any]:
        """Get detailed state information."""
        return {
            'agent_position': (self.agent_x, self.agent_y),
            'goal_position': (self.goal_x, self.goal_y),
            'episode_steps': self.episode_steps,
            'total_reward': self.total_reward,
            'current_reward': self.current_reward,
            'done': self.done,
            'grid_shape': (self.width, self.height),
            'obstacle_count': int(np.sum(self.grid == 0.9)),
            'action_space': self.action_space,
            'observation_space': self.observation_space
        }
        
    def is_success(self) -> bool:
        """Check if the agent successfully reached the goal."""
        return self.agent_x == self.goal_x and self.agent_y == self.goal_y


class MockVectorizedEnvironment:
    """Mock vectorized environment for testing parallel training."""
    
    def __init__(
        self,
        num_envs: int = 4,
        width: int = 10,
        height: int = 10,
        obstacle_density: float = 0.2
    ):
        self.num_envs = num_envs
        self.envs = [
            MockEnvironment(width, height, obstacle_density)
            for _ in range(num_envs)
        ]
        
    def reset(self) -> List[np.ndarray]:
        """Reset all environments."""
        return [env.reset() for env in self.envs]
        
    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], List[bool], List[Dict[str, Any]]]:
        """Step all environments."""
        observations = []
        rewards = []
        dones = []
        infos = []
        
        for env, action in zip(self.envs, actions):
            obs, reward, done, info = env.step(action)
            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
            
        return observations, rewards, dones, infos
        
    def stepSingle(self, env_idx: int, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Step a single environment."""
        return self.envs[env_idx].step(action)
        
    def resetSingle(self, env_idx: int) -> np.ndarray:
        """Reset a single environment."""
        return self.envs[env_idx].reset()
        
    def getNumEnvironments(self) -> int:
        """Get number of environments."""
        return self.num_envs
        
    def render(self, env_idx: int = 0):
        """Render a specific environment."""
        self.envs[env_idx].render()
        
    def get_states(self) -> List[Dict[str, Any]]:
        """Get states of all environments."""
        return [env.get_state_info() for env in self.envs]


def create_mock_environment(
    env_type: str = "single",
    **kwargs
):
    """Factory function to create mock environments."""
    if env_type.lower() == "single":
        return MockEnvironment(**kwargs)
    elif env_type.lower() == "vectorized":
        return MockVectorizedEnvironment(**kwargs)
    else:
        raise ValueError(f"Unknown environment type: {env_type}. Available: ['single', 'vectorized']")


# Compatibility functions to match the C++ interface
class Agent:
    """Mock Agent class for compatibility."""
    pass


class QTableAgent:
    """Mock QTableAgent class for compatibility."""
    pass


# Test function
def test_mock_environment():
    """Test the mock environment."""
    print("Testing Mock Environment...")
    
    env = MockEnvironment(width=5, height=5, obstacle_density=0.3)
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    env.render()
    
    # Run a few random steps
    for step in range(10):
        action = random.randint(0, 3)
        obs, reward, done, info = env.step(action)
        print(f"Step {step}: Action={action}, Reward={reward:.3f}, Done={done}")
        
        if done:
            print("Episode finished!")
            break
            
    print("\nTesting Vectorized Environment...")
    vec_env = MockVectorizedEnvironment(num_envs=2, width=5, height=5)
    observations = vec_env.reset()
    print(f"Vectorized observations: {len(observations)} environments")
    
    actions = [random.randint(0, 3) for _ in range(2)]
    obs, rewards, dones, infos = vec_env.step(actions)
    print(f"Step results: rewards={rewards}, dones={dones}")


if __name__ == "__main__":
    test_mock_environment()
