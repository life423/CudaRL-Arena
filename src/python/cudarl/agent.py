"""
CudaRL Agent implementation for reinforcement learning.

This module provides agent implementations for the CudaRL environment,
including Q-learning and DQN agents optimized for GPU execution.
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import json


class BaseAgent:
    """Base class for all RL agents."""
    
    def __init__(self, action_space: int, observation_space: int, name: str = "BaseAgent"):
        self.action_space = action_space
        self.observation_space = observation_space
        self.name = name
        self.training = True
        
    def act(self, observation: np.ndarray) -> int:
        """Select an action given an observation."""
        raise NotImplementedError
        
    def learn(self, experience: Tuple) -> Dict[str, float]:
        """Learn from experience."""
        raise NotImplementedError
        
    def save(self, filepath: str):
        """Save agent parameters."""
        raise NotImplementedError
        
    def load(self, filepath: str):
        """Load agent parameters."""
        raise NotImplementedError
        
    def set_training(self, training: bool):
        """Set training mode."""
        self.training = training


class RandomAgent(BaseAgent):
    """Random agent for baseline comparison."""
    
    def __init__(self, action_space: int, observation_space: int = None):
        super().__init__(action_space, observation_space, "RandomAgent")
        
    def act(self, observation: np.ndarray) -> int:
        """Return random action."""
        return random.randint(0, self.action_space - 1)
        
    def learn(self, experience: Tuple) -> Dict[str, float]:
        """Random agent doesn't learn."""
        return {}
        
    def save(self, filepath: str):
        """Random agent has no parameters to save."""
        pass
        
    def load(self, filepath: str):
        """Random agent has no parameters to load."""
        pass


class QLearningAgent(BaseAgent):
    """Tabular Q-Learning agent."""
    
    def __init__(
        self,
        action_space: int,
        state_space: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995
    ):
        super().__init__(action_space, state_space, "QLearningAgent")
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Q-table
        self.q_table = np.zeros((state_space, action_space))
        
        # Statistics
        self.total_steps = 0
        self.episodes = 0
        self.total_reward = 0.0
        
    def _state_to_index(self, observation: np.ndarray) -> int:
        """Convert observation to state index."""
        if isinstance(observation, (list, np.ndarray)):
            # Simple hash-based state encoding for grid world
            grid = np.array(observation).flatten()
            # Find agent position (assuming agent is marked with specific value)
            agent_positions = np.where(grid == 0.5)  # Assuming 0.5 is agent marker
            if len(agent_positions[0]) > 0:
                agent_pos = agent_positions[0][0]
            else:
                agent_pos = 0
            
            # Combine with obstacle pattern hash
            obstacle_hash = hash(tuple(grid > 0.8)) % 1000  # Simple hash
            state_idx = agent_pos + obstacle_hash
            return min(state_idx, self.observation_space - 1)
        else:
            return int(observation) % self.observation_space
    
    def act(self, observation: np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        state_idx = self._state_to_index(observation)
        
        if self.training and random.random() < self.epsilon:
            # Explore
            action = random.randint(0, self.action_space - 1)
        else:
            # Exploit
            action = np.argmax(self.q_table[state_idx])
            
        return action
    
    def learn(self, experience: Tuple) -> Dict[str, float]:
        """Update Q-table using Q-learning update rule."""
        state, action, reward, next_state, done = experience
        
        state_idx = self._state_to_index(state)
        next_state_idx = self._state_to_index(next_state)
        
        # Q-learning update
        if done:
            target = reward
        else:
            target = reward + self.discount_factor * np.max(self.q_table[next_state_idx])
            
        old_value = self.q_table[state_idx, action]
        self.q_table[state_idx, action] += self.learning_rate * (target - old_value)
        
        # Decay epsilon
        if self.training and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        # Update statistics
        self.total_steps += 1
        self.total_reward += reward
        
        return {
            'loss': abs(target - old_value),
            'epsilon': self.epsilon,
            'q_value': self.q_table[state_idx, action],
            'learning_rate': self.learning_rate
        }
    
    def save(self, filepath: str):
        """Save Q-table and hyperparameters."""
        data = {
            'q_table': self.q_table.tolist(),
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'total_steps': self.total_steps,
            'episodes': self.episodes,
            'total_reward': self.total_reward
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str):
        """Load Q-table and hyperparameters."""
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        self.q_table = np.array(data['q_table'])
        self.learning_rate = data['learning_rate']
        self.discount_factor = data['discount_factor']
        self.epsilon = data['epsilon']
        self.epsilon_min = data['epsilon_min']
        self.epsilon_decay = data['epsilon_decay']
        self.total_steps = data.get('total_steps', 0)
        self.episodes = data.get('episodes', 0)
        self.total_reward = data.get('total_reward', 0.0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            'total_steps': self.total_steps,
            'episodes': self.episodes,
            'total_reward': self.total_reward,
            'avg_reward': self.total_reward / max(1, self.episodes),
            'epsilon': self.epsilon,
            'q_table_size': self.q_table.shape,
            'q_table_mean': float(np.mean(self.q_table)),
            'q_table_std': float(np.std(self.q_table))
        }


class DQNAgent(BaseAgent):
    """Deep Q-Network agent (stub implementation)."""
    
    def __init__(
        self,
        action_space: int,
        observation_space: int,
        learning_rate: float = 0.001,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        memory_size: int = 10000,
        batch_size: int = 32
    ):
        super().__init__(action_space, observation_space, "DQNAgent")
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Experience replay buffer
        self.memory = deque(maxlen=memory_size)
        
        # Statistics
        self.total_steps = 0
        self.episodes = 0
        self.total_reward = 0.0
        
        # Stub neural network (would be implemented with PyTorch/TensorFlow)
        self.network_weights = np.random.randn(observation_space * action_space) * 0.1
        
        print("DQNAgent initialized (stub implementation)")
    
    def act(self, observation: np.ndarray) -> int:
        """Epsilon-greedy action selection using neural network."""
        if self.training and random.random() < self.epsilon:
            # Explore
            action = random.randint(0, self.action_space - 1)
        else:
            # Exploit using stub network
            obs_flat = np.array(observation).flatten()
            # Simple linear model simulation
            q_values = []
            for a in range(self.action_space):
                # Stub Q-value calculation
                idx = (hash(tuple(obs_flat)) + a) % len(self.network_weights)
                q_val = self.network_weights[idx] + np.sum(obs_flat) * 0.01
                q_values.append(q_val)
            action = np.argmax(q_values)
            
        return action
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))
    
    def learn(self, experience: Tuple) -> Dict[str, float]:
        """Learn from experience replay."""
        # Store experience
        self.remember(*experience)
        
        # Update statistics
        self.total_steps += 1
        self.total_reward += experience[2]  # reward
        
        # Batch learning (stub)
        loss = 0.0
        if len(self.memory) >= self.batch_size:
            # Sample batch (stub implementation)
            batch = random.sample(self.memory, self.batch_size)
            
            # Simulate network training
            total_error = 0.0
            for state, action, reward, next_state, done in batch:
                # Stub loss calculation
                target = reward
                if not done:
                    target += self.discount_factor * 0.5  # Stub next Q-max
                
                error = abs(target - 0.0)  # Stub current Q-value
                total_error += error
                
                # Stub weight update
                idx = hash((tuple(np.array(state).flatten()), action)) % len(self.network_weights)
                self.network_weights[idx] += self.learning_rate * error * 0.01
            
            loss = total_error / self.batch_size
        
        # Decay epsilon
        if self.training and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return {
            'loss': loss,
            'epsilon': self.epsilon,
            'memory_size': len(self.memory),
            'learning_rate': self.learning_rate
        }
    
    def save(self, filepath: str):
        """Save DQN parameters."""
        data = {
            'network_weights': self.network_weights.tolist(),
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'total_steps': self.total_steps,
            'episodes': self.episodes,
            'total_reward': self.total_reward
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str):
        """Load DQN parameters."""
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        self.network_weights = np.array(data['network_weights'])
        self.learning_rate = data['learning_rate']
        self.discount_factor = data['discount_factor']
        self.epsilon = data['epsilon']
        self.epsilon_min = data['epsilon_min']
        self.epsilon_decay = data['epsilon_decay']
        self.total_steps = data.get('total_steps', 0)
        self.episodes = data.get('episodes', 0)
        self.total_reward = data.get('total_reward', 0.0)


def create_agent(agent_type: str, action_space: int, observation_space: int, **kwargs) -> BaseAgent:
    """Factory function to create agents."""
    agents = {
        'random': RandomAgent,
        'qlearning': QLearningAgent,
        'dqn': DQNAgent
    }
    
    if agent_type.lower() not in agents:
        raise ValueError(f"Unknown agent type: {agent_type}. Available: {list(agents.keys())}")
    
    agent_class = agents[agent_type.lower()]
    return agent_class(action_space, observation_space, **kwargs)
