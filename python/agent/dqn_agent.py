import numpy as np
import sys
import os
import time
import random
from collections import deque

# Add the interface directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'interface'))

try:
    import cuda_gridworld_bindings as cgw
except ImportError:
    print("Error: Could not import CUDA GridWorld bindings.")
    print("Make sure to build the C++/CUDA components first.")
    sys.exit(1)

class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size=64, learning_rate=0.001, 
                 discount_factor=0.99, exploration_rate=1.0, min_exploration_rate=0.01, 
                 exploration_decay=0.995, memory_size=10000, batch_size=64):
        """
        Initialize a DQN agent with CUDA acceleration.
        
        Args:
            state_size: Dimension of state representation
            action_size: Number of possible actions
            hidden_size: Size of hidden layer in neural network
            learning_rate: Alpha parameter for network updates
            discount_factor: Gamma parameter for future rewards
            exploration_rate: Initial epsilon for epsilon-greedy policy
            min_exploration_rate: Minimum epsilon value
            exploration_decay: Decay rate for epsilon after each episode
            memory_size: Size of replay memory
            batch_size: Number of samples to use for each training step
        """
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay = exploration_decay
        self.memory_size = memory_size
        self.batch_size = batch_size
        
        # Initialize neural network weights
        self.initialize_network()
        
        # Initialize replay memory
        self.memory = deque(maxlen=memory_size)
    
    def initialize_network(self):
        """Initialize neural network weights with random values."""
        # Xavier/Glorot initialization for better convergence
        hidden_limit = np.sqrt(6.0 / (self.state_size + self.hidden_size))
        output_limit = np.sqrt(6.0 / (self.hidden_size + self.action_size))
        
        # Hidden layer weights and biases
        self.hidden_weights = np.random.uniform(-hidden_limit, hidden_limit, 
                                              (self.state_size, self.hidden_size)).astype(np.float32)
        self.hidden_biases = np.zeros(self.hidden_size, dtype=np.float32)
        
        # Output layer weights and biases
        self.output_weights = np.random.uniform(-output_limit, output_limit, 
                                              (self.hidden_size, self.action_size)).astype(np.float32)
        self.output_biases = np.zeros(self.action_size, dtype=np.float32)
        
        # Temporary storage for hidden layer outputs
        self.hidden_output = np.zeros((self.batch_size, self.hidden_size), dtype=np.float32)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))
    
    def select_action(self, state):
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state representation
            
        Returns:
            Selected action ID
        """
        # Exploration: random action
        if np.random.random() < self.exploration_rate:
            return np.random.randint(0, self.action_size)
        
        # Exploitation: best action from Q-values
        state_tensor = np.array([state], dtype=np.float32)
        q_values = np.zeros((1, self.action_size), dtype=np.float32)
        hidden_output = np.zeros((1, self.hidden_size), dtype=np.float32)
        
        # Forward pass using CUDA
        cgw.dqn_forward_cuda(
            state_tensor.flatten(),
            self.hidden_weights.flatten(),
            self.hidden_biases,
            hidden_output.flatten(),
            self.output_weights.flatten(),
            self.output_biases,
            q_values.flatten(),
            1,
            self.state_size,
            self.hidden_size,
            self.action_size
        )
        
        return np.argmax(q_values[0])
    
    def train(self):
        """Train the network on a batch of experiences from replay memory."""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample random batch from memory
        batch = random.sample(self.memory, self.batch_size)
        
        # Extract batch data
        states = np.array([experience[0] for experience in batch], dtype=np.float32)
        actions = np.array([experience[1] for experience in batch], dtype=np.int32)
        rewards = np.array([experience[2] for experience in batch], dtype=np.float32)
        next_states = np.array([experience[3] for experience in batch], dtype=np.float32)
        dones = np.array([experience[4] for experience in batch], dtype=np.bool_)
        
        # Compute current Q values
        current_q_values = np.zeros((self.batch_size, self.action_size), dtype=np.float32)
        cgw.dqn_forward_cuda(
            states.flatten(),
            self.hidden_weights.flatten(),
            self.hidden_biases,
            self.hidden_output.flatten(),
            self.output_weights.flatten(),
            self.output_biases,
            current_q_values.flatten(),
            self.batch_size,
            self.state_size,
            self.hidden_size,
            self.action_size
        )
        
        # Compute next Q values
        next_q_values = np.zeros((self.batch_size, self.action_size), dtype=np.float32)
        next_hidden_output = np.zeros((self.batch_size, self.hidden_size), dtype=np.float32)
        cgw.dqn_forward_cuda(
            next_states.flatten(),
            self.hidden_weights.flatten(),
            self.hidden_biases,
            next_hidden_output.flatten(),
            self.output_weights.flatten(),
            self.output_biases,
            next_q_values.flatten(),
            self.batch_size,
            self.state_size,
            self.hidden_size,
            self.action_size
        )
        
        # Compute target Q values
        targets = current_q_values.copy()
        for i in range(self.batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.discount_factor * np.max(next_q_values[i])
        
        # Update weights using simple gradient descent
        # In a real implementation, you would use a proper optimizer like Adam
        # This is simplified for demonstration purposes
        self._update_weights(states, targets)
    
    def _update_weights(self, states, targets):
        """
        Update network weights using gradient descent.
        
        Args:
            states: Batch of states
            targets: Target Q-values
        """
        # Forward pass
        current_q_values = np.zeros((self.batch_size, self.action_size), dtype=np.float32)
        cgw.dqn_forward_cuda(
            states.flatten(),
            self.hidden_weights.flatten(),
            self.hidden_biases,
            self.hidden_output.flatten(),
            self.output_weights.flatten(),
            self.output_biases,
            current_q_values.flatten(),
            self.batch_size,
            self.state_size,
            self.hidden_size,
            self.action_size
        )
        
        # Compute gradients and update weights
        # This is a simplified implementation of backpropagation
        # In a real implementation, you would use automatic differentiation
        
        # Output layer gradients
        output_deltas = (current_q_values - targets) / self.batch_size
        
        # Update output weights and biases
        for i in range(self.batch_size):
            for j in range(self.action_size):
                delta = output_deltas[i, j]
                for k in range(self.hidden_size):
                    self.output_weights[k, j] -= self.learning_rate * delta * self.hidden_output[i, k]
                self.output_biases[j] -= self.learning_rate * delta
        
        # Hidden layer gradients (simplified, not accounting for ReLU derivative)
        hidden_deltas = np.zeros((self.batch_size, self.hidden_size), dtype=np.float32)
        for i in range(self.batch_size):
            for j in range(self.hidden_size):
                for k in range(self.action_size):
                    hidden_deltas[i, j] += output_deltas[i, k] * self.output_weights[j, k]
                # Apply ReLU derivative
                if self.hidden_output[i, j] <= 0:
                    hidden_deltas[i, j] = 0
        
        # Update hidden weights and biases
        for i in range(self.batch_size):
            for j in range(self.hidden_size):
                delta = hidden_deltas[i, j]
                for k in range(self.state_size):
                    self.hidden_weights[k, j] -= self.learning_rate * delta * states[i, k]
                self.hidden_biases[j] -= self.learning_rate * delta
    
    def decay_exploration(self):
        """Decay exploration rate after an episode."""
        self.exploration_rate = max(self.min_exploration_rate, 
                                   self.exploration_rate * self.exploration_decay)
    
    def save_model(self, filepath):
        """Save model weights to file."""
        np.savez(filepath, 
                hidden_weights=self.hidden_weights,
                hidden_biases=self.hidden_biases,
                output_weights=self.output_weights,
                output_biases=self.output_biases)
    
    def load_model(self, filepath):
        """Load model weights from file."""
        data = np.load(filepath)
        self.hidden_weights = data['hidden_weights']
        self.hidden_biases = data['hidden_biases']
        self.output_weights = data['output_weights']
        self.output_biases = data['output_biases']