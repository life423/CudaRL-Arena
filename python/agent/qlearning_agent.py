import numpy as np
import sys
import os
import time

# Add the interface directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'interface'))

try:
    import cuda_gridworld_bindings as cgw
except ImportError:
    print("Error: Could not import CUDA GridWorld bindings.")
    print("Make sure to build the C++/CUDA components first.")
    sys.exit(1)

class QLearningAgent:
    def __init__(self, state_space_size, action_space_size, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0, min_exploration_rate=0.01, exploration_decay=0.995):
        """
        Initialize a Q-learning agent with CUDA acceleration.
        
        Args:
            state_space_size: Number of possible states in the environment
            action_space_size: Number of possible actions in the environment
            learning_rate: Alpha parameter for Q-learning update
            discount_factor: Gamma parameter for Q-learning update
            exploration_rate: Initial epsilon for epsilon-greedy policy
            min_exploration_rate: Minimum epsilon value
            exploration_decay: Decay rate for epsilon after each episode
        """
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay = exploration_decay
        
        # Initialize Q-table with zeros
        self.q_table = np.zeros((state_space_size, action_space_size), dtype=np.float32)
        
        # Experience buffer for batch updates
        self.experience_buffer = []
        self.batch_size = 64
    
    def select_action(self, state):
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state ID
            
        Returns:
            Selected action ID
        """
        # Exploration: random action
        if np.random.random() < self.exploration_rate:
            return np.random.randint(0, self.action_space_size)
        
        # Exploitation: best action from Q-table
        return cgw.get_best_action(self.q_table, state)
    
    def store_experience(self, state, action, reward, next_state):
        """
        Store experience in buffer for batch updates.
        
        Args:
            state: Current state ID
            action: Action taken
            reward: Reward received
            next_state: Next state ID
        """
        self.experience_buffer.append((state, action, reward, next_state))
        
        # If buffer is full, update Q-table
        if len(self.experience_buffer) >= self.batch_size:
            self.update_q_table()
    
    def update_q_table(self):
        """
        Update Q-table using CUDA acceleration with experiences in buffer.
        """
        if not self.experience_buffer:
            return
        
        # Prepare batch data
        states = np.array([exp[0] for exp in self.experience_buffer], dtype=np.int32)
        actions = np.array([exp[1] for exp in self.experience_buffer], dtype=np.int32)
        rewards = np.array([exp[2] for exp in self.experience_buffer], dtype=np.float32)
        next_states = np.array([exp[3] for exp in self.experience_buffer], dtype=np.int32)
        
        # Update Q-table using CUDA
        cgw.update_q_table(self.q_table, states, actions, rewards, next_states, 
                          self.learning_rate, self.discount_factor)
        
        # Clear buffer
        self.experience_buffer = []
    
    def decay_exploration(self):
        """
        Decay exploration rate after an episode.
        """
        self.exploration_rate = max(self.min_exploration_rate, 
                                   self.exploration_rate * self.exploration_decay)
    
    def save_model(self, filepath):
        """
        Save Q-table to file.
        
        Args:
            filepath: Path to save the Q-table
        """
        np.save(filepath, self.q_table)
    
    def load_model(self, filepath):
        """
        Load Q-table from file.
        
        Args:
            filepath: Path to load the Q-table from
        """
        self.q_table = np.load(filepath)