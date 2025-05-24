"""
Agent module for CudaRL-Arena.

This module provides reinforcement learning agents that interact with the environment.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from abc import ABC, abstractmethod

# Configure logging
logger = logging.getLogger(__name__)

class Agent(ABC):
    """
    Abstract base class for all agents.
    """
    
    def __init__(self, action_space_size: int, observation_shape: Tuple[int, ...]):
        """
        Initialize the agent.
        
        Args:
            action_space_size: Number of possible actions
            observation_shape: Shape of the observation space
        """
        self.action_space_size = action_space_size
        self.observation_shape = observation_shape
        logger.info(f"Initialized agent with {action_space_size} actions and observation shape {observation_shape}")
    
    @abstractmethod
    def select_action(self, observation: np.ndarray) -> int:
        """
        Select an action based on the current observation.
        
        Args:
            observation: Current observation from the environment
            
        Returns:
            Selected action
        """
        pass
    
    @abstractmethod
    def update(self, observation: np.ndarray, action: int, reward: float, 
               next_observation: np.ndarray, done: bool) -> Dict[str, float]:
        """
        Update the agent's policy based on experience.
        
        Args:
            observation: Current observation
            action: Action taken
            reward: Reward received
            next_observation: Next observation
            done: Whether the episode is done
            
        Returns:
            Dictionary of metrics from the update
        """
        pass
    
    def save(self, path: str) -> None:
        """
        Save the agent's state.
        
        Args:
            path: Path to save the agent
        """
        logger.info(f"Saving agent to {path}")
    
    def load(self, path: str) -> None:
        """
        Load the agent's state.
        
        Args:
            path: Path to load the agent from
        """
        logger.info(f"Loading agent from {path}")


class RandomAgent(Agent):
    """
    Simple agent that selects actions randomly.
    """
    
    def select_action(self, observation: np.ndarray) -> int:
        """
        Select a random action.
        
        Args:
            observation: Current observation (ignored)
            
        Returns:
            Random action
        """
        return np.random.randint(0, self.action_space_size)
    
    def update(self, observation: np.ndarray, action: int, reward: float, 
               next_observation: np.ndarray, done: bool) -> Dict[str, float]:
        """
        No-op update for random agent.
        
        Returns:
            Empty metrics dictionary
        """
        return {}


class QTableAgent(Agent):
    """
    Tabular Q-learning agent.
    """
    
    def __init__(self, action_space_size: int, observation_shape: Tuple[int, ...], 
                 learning_rate: float = 0.1, discount_factor: float = 0.99, 
                 exploration_rate: float = 1.0, exploration_decay: float = 0.995):
        """
        Initialize the Q-table agent.
        
        Args:
            action_space_size: Number of possible actions
            observation_shape: Shape of the observation space
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            exploration_rate: Initial exploration rate (epsilon)
            exploration_decay: Decay rate for exploration
        """
        super().__init__(action_space_size, observation_shape)
        
        # For Q-table, we need to discretize the observation space
        self.width = observation_shape[1]
        self.height = observation_shape[0]
        
        # Initialize Q-table: state is (x, y), action is direction
        self.q_table = np.zeros((self.width, self.height, action_space_size))
        
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        
        logger.info(f"Initialized Q-table agent with {self.width}x{self.height} states and {action_space_size} actions")
    
    def _get_state_from_observation(self, observation: np.ndarray) -> Tuple[int, int]:
        """
        Extract agent position from observation.
        
        In this simple case, we assume the agent position is encoded in the observation.
        For more complex environments, this would need to be adapted.
        
        Args:
            observation: Current observation
            
        Returns:
            Agent position as (x, y)
        """
        # Find the agent position (highest value in the observation)
        agent_pos = np.unravel_index(np.argmax(observation), observation.shape)
        return agent_pos[1], agent_pos[0]  # Convert to (x, y)
    
    def select_action(self, observation: np.ndarray) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            observation: Current observation
            
        Returns:
            Selected action
        """
        # Epsilon-greedy action selection
        if np.random.random() < self.exploration_rate:
            return np.random.randint(0, self.action_space_size)
        
        # Get state from observation
        x, y = self._get_state_from_observation(observation)
        
        # Select best action
        return np.argmax(self.q_table[x, y])
    
    def update(self, observation: np.ndarray, action: int, reward: float, 
               next_observation: np.ndarray, done: bool) -> Dict[str, float]:
        """
        Update Q-table using Q-learning update rule.
        
        Args:
            observation: Current observation
            action: Action taken
            reward: Reward received
            next_observation: Next observation
            done: Whether the episode is done
            
        Returns:
            Dictionary with update metrics
        """
        # Get current and next state
        x, y = self._get_state_from_observation(observation)
        next_x, next_y = self._get_state_from_observation(next_observation)
        
        # Current Q-value
        current_q = self.q_table[x, y, action]
        
        # Next Q-value (max over actions)
        next_q = np.max(self.q_table[next_x, next_y]) if not done else 0
        
        # Q-learning update
        target = reward + self.discount_factor * next_q
        self.q_table[x, y, action] += self.learning_rate * (target - current_q)
        
        # Decay exploration rate
        self.exploration_rate *= self.exploration_decay
        
        return {
            'q_value': current_q,
            'target': target,
            'td_error': target - current_q,
            'exploration_rate': self.exploration_rate
        }