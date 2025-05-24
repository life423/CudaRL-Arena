#!/usr/bin/env python3
"""
Visualization script for CudaRL-Arena.

This script provides visualization tools for the environment and training progress.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
import time
import os

try:
    import cudarl_core
except ImportError:
    print("Failed to import cudarl_core module. Make sure it's built and in your Python path.")
    print("You may need to run: cmake --build build --target cudarl_core_python")
    exit(1)

class GridWorldVisualizer:
    """
    Visualizer for the GridWorld environment.
    """
    
    def __init__(self, env, agent=None):
        """
        Initialize the visualizer.
        
        Args:
            env: Environment to visualize
            agent: Optional agent to visualize (for showing policy)
        """
        self.env = env
        self.agent = agent
        self.width = env.get_width()
        self.height = env.get_height()
        
        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.fig.canvas.set_window_title('CudaRL-Arena GridWorld')
        
        # Initialize grid
        self.grid_data = np.zeros((self.height, self.width))
        self.grid_image = self.ax.imshow(self.grid_data, cmap='viridis', interpolation='nearest')
        
        # Add colorbar
        self.cbar = self.fig.colorbar(self.grid_image, ax=self.ax)
        self.cbar.set_label('Cell Value')
        
        # Initialize agent marker
        self.agent_marker = None
        
        # Initialize policy arrows
        self.policy_arrows = []
        
        # Set up grid
        self.ax.set_xticks(np.arange(-.5, self.width, 1), minor=True)
        self.ax.set_yticks(np.arange(-.5, self.height, 1), minor=True)
        self.ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
        self.ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        
        # Set title
        self.ax.set_title('GridWorld Environment')
    
    def update_grid(self, observation):
        """
        Update the grid visualization.
        
        Args:
            observation: Current observation (flattened grid)
        """
        # Reshape observation to grid
        self.grid_data = np.array(observation).reshape(self.height, self.width)
        
        # Update grid image
        self.grid_image.set_array(self.grid_data)
        self.grid_image.set_clim(vmin=min(0, np.min(self.grid_data)), vmax=max(1, np.max(self.grid_data)))
        
        # Update agent position
        agent_pos = self.env.get_agent_position()
        if self.agent_marker:
            self.agent_marker.remove()
        self.agent_marker = self.ax.plot(agent_pos[0], agent_pos[1], 'ro', markersize=15, markeredgecolor='black')[0]
        
        # Update policy arrows if agent is provided
        if self.agent and hasattr(self.agent, 'q_table'):
            self.update_policy_arrows()
    
    def update_policy_arrows(self):
        """
        Update policy arrows based on agent's Q-values.
        """
        # Remove old arrows
        for arrow in self.policy_arrows:
            arrow.remove()
        self.policy_arrows = []
        
        # Add new arrows
        for y in range(self.height):
            for x in range(self.width):
                if hasattr(self.agent, 'q_table'):
                    # Get best action
                    q_values = self.agent.q_table[x, y]
                    best_action = np.argmax(q_values)
                    
                    # Skip if all Q-values are zero
                    if np.all(q_values == 0):
                        continue
                    
                    # Calculate arrow direction
                    dx, dy = 0, 0
                    if best_action == 0:  # up
                        dx, dy = 0, -0.4
                    elif best_action == 1:  # right
                        dx, dy = 0.4, 0
                    elif best_action == 2:  # down
                        dx, dy = 0, 0.4
                    elif best_action == 3:  # left
                        dx, dy = -0.4, 0
                    
                    # Add arrow
                    arrow = self.ax.arrow(x, y, dx, dy, head_width=0.2, head_length=0.2, 
                                         fc='white', ec='black', alpha=0.7)
                    self.policy_arrows.append(arrow)
    
    def show(self):
        """
        Show the visualization.
        """
        plt.tight_layout()
        plt.show()
    
    def save(self, filename):
        """
        Save the visualization to a file.
        
        Args:
            filename: Filename to save to
        """
        plt.tight_layout()
        plt.savefig(filename)


def visualize_training(env, agent, num_episodes=10, delay=0.5, save_gif=False):
    """
    Visualize the agent training in the environment.
    
    Args:
        env: Environment to visualize
        agent: Agent to train and visualize
        num_episodes: Number of episodes to visualize
        delay: Delay between frames (seconds)
        save_gif: Whether to save the visualization as a GIF
    """
    # Create visualizer
    vis = GridWorldVisualizer(env, agent)
    
    # Create figure for animation
    fig = vis.fig
    
    # Initialize frames for animation
    frames = []
    
    def update_frame(frame_idx):
        # Clear axis
        vis.ax.clear()
        
        # Set up grid
        vis.ax.set_xticks(np.arange(-.5, vis.width, 1), minor=True)
        vis.ax.set_yticks(np.arange(-.5, vis.height, 1), minor=True)
        vis.ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
        vis.ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        
        # Get episode and step from frame index
        episode = frame_idx // 1000
        step = frame_idx % 1000
        
        # If new episode
        if step == 0:
            observation = env.reset()
            agent_pos = env.get_agent_position()
            vis.update_grid(observation)
            vis.ax.set_title(f'Episode {episode+1}, Step {step}')
            return [vis.grid_image, vis.agent_marker]
        
        # Select action
        agent_pos = env.get_agent_position()
        action = agent.select_action(observation, agent_pos)
        
        # Take step
        next_observation, reward, done, info = env.step(action)
        next_agent_pos = env.get_agent_position()
        
        # Update agent
        agent.update(observation, action, reward, next_observation, done, agent_pos, next_agent_pos)
        
        # Update visualization
        vis.update_grid(next_observation)
        vis.ax.set_title(f'Episode {episode+1}, Step {step}, Action: {action}, Reward: {reward:.2f}')
        
        # If done, start new episode
        if done:
            return [vis.grid_image, vis.agent_marker]
        
        return [vis.grid_image, vis.agent_marker]
    
    # Create animation
    ani = animation.FuncAnimation(fig, update_frame, frames=num_episodes*1000, 
                                 interval=delay*1000, blit=True)
    
    # Save animation if requested
    if save_gif:
        ani.save('training.gif', writer='pillow', fps=1/delay)
    
    # Show animation
    plt.show()


def visualize_q_values(agent, width, height):
    """
    Visualize the agent's Q-values.
    
    Args:
        agent: Agent with Q-values to visualize
        width: Environment width
        height: Environment height
    """
    if not hasattr(agent, 'q_table'):
        print("Agent does not have a Q-table to visualize")
        return
    
    # Create figure with subplots for each action
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Action names
    action_names = ['Up', 'Right', 'Down', 'Left']
    
    # Plot Q-values for each action
    for a in range(4):
        # Extract Q-values for this action
        q_values = np.zeros((height, width))
        for y in range(height):
            for x in range(width):
                q_values[y, x] = agent.q_table[x, y, a]
        
        # Plot Q-values
        im = axes[a].imshow(q_values, cmap='viridis', interpolation='nearest')
        axes[a].set_title(f'Q-values for {action_names[a]}')
        
        # Add colorbar
        fig.colorbar(im, ax=axes[a])
        
        # Set up grid
        axes[a].set_xticks(np.arange(-.5, width, 1), minor=True)
        axes[a].set_yticks(np.arange(-.5, height, 1), minor=True)
        axes[a].grid(which='minor', color='w', linestyle='-', linewidth=1)
        axes[a].tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    
    plt.tight_layout()
    plt.savefig('q_values.png')
    plt.show()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize the GridWorld environment')
    
    # Environment parameters
    parser.add_argument('--width', type=int, default=10, help='Environment width')
    parser.add_argument('--height', type=int, default=10, help='Environment height')
    
    # Visualization parameters
    parser.add_argument('--mode', type=str, default='static', choices=['static', 'training', 'q_values'],
                        help='Visualization mode')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes to visualize')
    parser.add_argument('--delay', type=float, default=0.2, help='Delay between frames (seconds)')
    parser.add_argument('--save-gif', action='store_true', help='Save the visualization as a GIF')
    
    # Agent parameters
    parser.add_argument('--load-agent', type=str, default='', help='Path to load agent from')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Create environment
    env = cudarl_core.Environment(args.width, args.height)
    
    # Create agent
    from train import QTableAgent
    agent = QTableAgent(args.width, args.height)
    
    # Load agent if requested
    if args.load_agent and os.path.exists(args.load_agent):
        agent.load(args.load_agent)
        print(f"Loaded agent from {args.load_agent}")
    
    # Visualize based on mode
    if args.mode == 'static':
        # Reset environment
        observation = env.reset()
        
        # Create visualizer
        vis = GridWorldVisualizer(env, agent)
        vis.update_grid(observation)
        
        # Show visualization
        vis.show()
    
    elif args.mode == 'training':
        # Visualize training
        visualize_training(env, agent, args.episodes, args.delay, args.save_gif)
    
    elif args.mode == 'q_values':
        # Visualize Q-values
        visualize_q_values(agent, args.width, args.height)


if __name__ == "__main__":
    main()