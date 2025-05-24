#!/usr/bin/env python3
"""
Simple GUI for CudaRL-Arena using Matplotlib.

This provides a basic visualization of the environment and agent.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button, Slider
import time
import argparse
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create a simple QTableAgent class without importing from train.py
class QTableAgent:
    """
    Tabular Q-learning agent.
    """
    
    def __init__(self, width, height, action_space_size=4, 
                 learning_rate=0.1, discount_factor=0.99, 
                 exploration_rate=1.0, exploration_decay=0.995):
        """Initialize the Q-table agent."""
        self.width = width
        self.height = height
        self.action_space_size = action_space_size
        
        # Initialize Q-table: state is (x, y), action is direction
        self.q_table = np.zeros((width, height, action_space_size))
        
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        
        print(f"Initialized Q-table agent with {width}x{height} states and {action_space_size} actions")
    
    def select_action(self, observation, agent_position, training=True):
        """Select an action using epsilon-greedy policy."""
        # Epsilon-greedy action selection during training
        if training and np.random.random() < self.exploration_rate:
            return np.random.randint(0, self.action_space_size)
        
        # Get state from observation
        x, y = agent_position
        
        # Select best action
        return np.argmax(self.q_table[x, y])
    
    def update(self, observation, action, reward, next_observation, done, agent_position, next_agent_position):
        """Update Q-table using Q-learning update rule."""
        # Get current and next state
        x, y = agent_position
        next_x, next_y = next_agent_position
        
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

# Create a simple mock environment if needed
class MockEnvironment:
    """Mock environment for testing."""
    
    def __init__(self, width=10, height=10):
        """Initialize the environment."""
        self.width = width
        self.height = height
        self.agent_x = width // 2
        self.agent_y = height // 2
        self.grid = np.zeros((height, width))
        self.grid[0, width-1] = 1.0  # Goal in top-right corner
        print(f"Created mock environment with size {width}x{height}")
    
    def reset(self):
        """Reset the environment."""
        self.agent_x = self.width // 2
        self.agent_y = self.height // 2
        return self.grid.flatten().tolist()
    
    def step(self, action):
        """Take a step in the environment."""
        # Action: 0=up, 1=right, 2=down, 3=left
        dx, dy = 0, 0
        if action == 0:
            dy = -1  # up
        elif action == 1:
            dx = 1   # right
        elif action == 2:
            dy = 1   # down
        elif action == 3:
            dx = -1  # left
        
        # Update agent position with bounds checking
        new_x = self.agent_x + dx
        new_y = self.agent_y + dy
        
        if 0 <= new_x < self.width and 0 <= new_y < self.height:
            self.agent_x = new_x
            self.agent_y = new_y
        
        # Simple reward: -0.01 per step, +1 for reaching goal (top-right corner)
        reward = -0.01
        done = False
        
        # Check if agent reached goal (top-right corner)
        if self.agent_x == self.width - 1 and self.agent_y == 0:
            reward = 1.0
            done = True
        
        return self.grid.flatten().tolist(), reward, done, ""
    
    def get_agent_position(self):
        """Get the agent's position."""
        return (self.agent_x, self.agent_y)

# Try to import the environment
try:
    import cudarl_core
    print("Using CUDA environment")
    Environment = cudarl_core.Environment
except ImportError:
    # Fall back to mock environment
    print("Using mock environment")
    Environment = MockEnvironment

class GridWorldGUI:
    """
    Simple GUI for visualizing the GridWorld environment.
    """
    
    def __init__(self, width=10, height=10):
        """Initialize the GUI."""
        self.width = width
        self.height = height
        
        # Create environment
        self.env = Environment(width, height)
        
        # Create agent
        self.agent = QTableAgent(width, height)
        
        # Initialize state
        self.observation = self.env.reset()
        self.agent_pos = self.env.get_agent_position()
        self.episode_reward = 0
        self.step_count = 0
        self.episode_count = 0
        self.running = False
        self.speed = 0.5  # seconds between steps
        self.last_step_time = time.time()
        
        # Create figure and axes
        self.fig = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[3, 1])
        
        # Grid display
        self.ax_grid = self.fig.add_subplot(gs[0, 0])
        self.grid_data = np.zeros((height, width))
        self.grid_img = self.ax_grid.imshow(self.grid_data, cmap='viridis', interpolation='nearest')
        self.ax_grid.set_title('Environment')
        self.ax_grid.set_xticks(np.arange(width))
        self.ax_grid.set_yticks(np.arange(height))
        self.ax_grid.set_xticklabels([])
        self.ax_grid.set_yticklabels([])
        self.ax_grid.grid(True, color='w', linestyle='-', linewidth=1)
        
        # Agent marker
        self.agent_marker = self.ax_grid.plot(self.agent_pos[0], self.agent_pos[1], 'ro', markersize=10)[0]
        
        # Q-values display
        self.ax_q = self.fig.add_subplot(gs[0, 1])
        self.q_data = np.zeros((4, 1))
        self.q_img = self.ax_q.imshow(self.q_data, cmap='coolwarm', interpolation='nearest')
        self.ax_q.set_title('Q-values at Agent Position')
        self.ax_q.set_yticks(np.arange(4))
        self.ax_q.set_yticklabels(['Up', 'Right', 'Down', 'Left'])
        self.ax_q.set_xticks([])
        
        # Reward plot
        self.ax_reward = self.fig.add_subplot(gs[1, 0])
        self.rewards = []
        self.reward_line, = self.ax_reward.plot([], [], 'b-')
        self.ax_reward.set_title('Episode Reward')
        self.ax_reward.set_xlabel('Step')
        self.ax_reward.set_ylabel('Reward')
        self.ax_reward.set_xlim(0, 100)
        self.ax_reward.set_ylim(-5, 1)
        self.ax_reward.grid(True)
        
        # Control panel
        self.ax_controls = self.fig.add_subplot(gs[1, 1])
        self.ax_controls.axis('off')
        
        # Add buttons
        self.btn_step_ax = plt.axes([0.7, 0.3, 0.1, 0.05])
        self.btn_step = Button(self.btn_step_ax, 'Step')
        self.btn_step.on_clicked(self.on_step)
        
        self.btn_reset_ax = plt.axes([0.7, 0.2, 0.1, 0.05])
        self.btn_reset = Button(self.btn_reset_ax, 'Reset')
        self.btn_reset.on_clicked(self.on_reset)
        
        self.btn_run_ax = plt.axes([0.7, 0.1, 0.1, 0.05])
        self.btn_run = Button(self.btn_run_ax, 'Run/Pause')
        self.btn_run.on_clicked(self.on_run)
        
        # Add speed slider
        self.slider_ax = plt.axes([0.7, 0.05, 0.1, 0.03])
        self.slider = Slider(self.slider_ax, 'Speed', 0.1, 2.0, valinit=0.5)
        self.slider.on_changed(self.on_speed_change)
        
        # Status text
        self.status_text = self.ax_controls.text(0.1, 0.8, '', fontsize=10)
        self.update_status()
        
        # Set up animation
        self.ani = FuncAnimation(self.fig, self.update, interval=50, blit=False)
        
        plt.tight_layout()
    
    def update_grid(self):
        """Update the grid visualization."""
        # Reshape observation to grid
        self.grid_data = np.array(self.observation).reshape(self.height, self.width)
        self.grid_img.set_array(self.grid_data)
        
        # Update agent position
        self.agent_pos = self.env.get_agent_position()
        self.agent_marker.set_data(self.agent_pos[0], self.agent_pos[1])
        
        # Update Q-values
        x, y = self.agent_pos
        q_values = self.agent.q_table[x, y].reshape(-1, 1)
        self.q_data = q_values
        self.q_img.set_array(self.q_data)
        self.q_img.set_clim(vmin=np.min(self.agent.q_table), vmax=np.max(self.agent.q_table))
    
    def update_reward_plot(self):
        """Update the reward plot."""
        self.rewards.append(self.episode_reward)
        self.reward_line.set_data(range(len(self.rewards)), self.rewards)
        
        # Adjust x-axis limit if needed
        if len(self.rewards) > self.ax_reward.get_xlim()[1]:
            self.ax_reward.set_xlim(0, len(self.rewards) * 1.5)
        
        # Adjust y-axis limit if needed
        min_reward = min(self.rewards) if self.rewards else -5
        max_reward = max(self.rewards) if self.rewards else 1
        if min_reward < self.ax_reward.get_ylim()[0] or max_reward > self.ax_reward.get_ylim()[1]:
            self.ax_reward.set_ylim(min_reward - 1, max_reward + 1)
    
    def update_status(self):
        """Update the status text."""
        status = f"Episode: {self.episode_count}\n"
        status += f"Step: {self.step_count}\n"
        status += f"Reward: {self.episode_reward:.2f}\n"
        status += f"Agent Position: {self.agent_pos}\n"
        status += f"Exploration Rate: {self.agent.exploration_rate:.3f}"
        
        self.status_text.set_text(status)
    
    def step(self):
        """Take a step in the environment."""
        # Select action
        action = self.agent.select_action(self.observation, self.agent_pos)
        
        # Take step
        next_observation, reward, done, info = self.env.step(action)
        next_agent_pos = self.env.get_agent_position()
        
        # Update agent
        self.agent.update(
            self.observation, action, reward, next_observation, done,
            self.agent_pos, next_agent_pos
        )
        
        # Update state
        self.observation = next_observation
        self.agent_pos = next_agent_pos
        self.episode_reward += reward
        self.step_count += 1
        
        # Check if episode is done
        if done:
            print(f"Episode {self.episode_count} finished with reward {self.episode_reward:.2f}")
            self.observation = self.env.reset()
            self.agent_pos = self.env.get_agent_position()
            self.episode_count += 1
            self.episode_reward = 0
            self.step_count = 0
        
        # Update visualizations
        self.update_grid()
        self.update_reward_plot()
        self.update_status()
    
    def update(self, frame):
        """Update function for animation."""
        if self.running:
            current_time = time.time()
            if current_time - self.last_step_time >= self.speed:
                self.step()
                self.last_step_time = current_time
        
        return self.grid_img, self.agent_marker, self.reward_line, self.status_text
    
    def on_step(self, event):
        """Handle step button click."""
        self.step()
    
    def on_reset(self, event):
        """Handle reset button click."""
        self.observation = self.env.reset()
        self.agent_pos = self.env.get_agent_position()
        self.episode_reward = 0
        self.step_count = 0
        self.episode_count += 1
        self.update_grid()
        self.update_status()
    
    def on_run(self, event):
        """Handle run/pause button click."""
        self.running = not self.running
        if self.running:
            self.last_step_time = time.time()
    
    def on_speed_change(self, val):
        """Handle speed slider change."""
        self.speed = val
    
    def show(self):
        """Show the GUI."""
        plt.show()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='GridWorld GUI')
    parser.add_argument('--width', type=int, default=10, help='Environment width')
    parser.add_argument('--height', type=int, default=10, help='Environment height')
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    gui = GridWorldGUI(args.width, args.height)
    gui.show()


if __name__ == '__main__':
    main()