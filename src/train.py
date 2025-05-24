#!/usr/bin/env python3
"""
Training script for CudaRL-Arena.

This script provides a complete training loop for reinforcement learning agents
using the CUDA-accelerated environment.
"""

import argparse
import logging
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

# Try to import the C++ module
try:
    import cudarl_core
    logger.info("Successfully imported cudarl_core module")
except ImportError:
    logger.error("Failed to import cudarl_core module. Make sure it's built and in your Python path.")
    logger.error("You may need to run: cmake --build build --target cudarl_core")
    sys.exit(1)

class QTableAgent:
    """
    Tabular Q-learning agent.
    """
    
    def __init__(self, width, height, action_space_size=4, 
                 learning_rate=0.1, discount_factor=0.99, 
                 exploration_rate=1.0, exploration_decay=0.995):
        """
        Initialize the Q-table agent.
        
        Args:
            width: Width of the environment grid
            height: Height of the environment grid
            action_space_size: Number of possible actions
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            exploration_rate: Initial exploration rate (epsilon)
            exploration_decay: Decay rate for exploration
        """
        self.width = width
        self.height = height
        self.action_space_size = action_space_size
        
        # Initialize Q-table: state is (x, y), action is direction
        self.q_table = np.zeros((width, height, action_space_size))
        
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        
        logger.info(f"Initialized Q-table agent with {width}x{height} states and {action_space_size} actions")
    
    def _get_state_from_observation(self, observation):
        """
        Extract agent position from observation.
        
        Args:
            observation: Current observation (flattened grid)
            
        Returns:
            Agent position as (x, y)
        """
        # For now, we'll use the agent position directly from the environment
        # In a more complex scenario, we would need to extract it from the observation
        return self.agent_position
    
    def select_action(self, observation, agent_position, training=True):
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            observation: Current observation
            agent_position: Current agent position (x, y)
            training: Whether we're in training mode (use exploration) or not
            
        Returns:
            Selected action
        """
        self.agent_position = agent_position
        
        # Epsilon-greedy action selection during training
        if training and np.random.random() < self.exploration_rate:
            return np.random.randint(0, self.action_space_size)
        
        # Get state from observation
        x, y = agent_position
        
        # Select best action
        return np.argmax(self.q_table[x, y])
    
    def update(self, observation, action, reward, next_observation, done, agent_position, next_agent_position):
        """
        Update Q-table using Q-learning update rule.
        
        Args:
            observation: Current observation
            action: Action taken
            reward: Reward received
            next_observation: Next observation
            done: Whether the episode is done
            agent_position: Current agent position (x, y)
            next_agent_position: Next agent position (x, y)
            
        Returns:
            Dictionary with update metrics
        """
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
    
    def save(self, path):
        """
        Save the agent's Q-table to a file.
        
        Args:
            path: Path to save the Q-table
        """
        np.save(path, self.q_table)
        logger.info(f"Saved Q-table to {path}")
    
    def load(self, path):
        """
        Load the agent's Q-table from a file.
        
        Args:
            path: Path to load the Q-table from
        """
        self.q_table = np.load(path)
        logger.info(f"Loaded Q-table from {path}")


def train(env, agent, num_episodes=1000, max_steps=500, render_every=0):
    """
    Train the agent in the environment.
    
    Args:
        env: Environment to train in
        agent: Agent to train
        num_episodes: Number of episodes to train for
        max_steps: Maximum steps per episode
        render_every: Render environment every N episodes (0 to disable)
        
    Returns:
        Dictionary of training metrics
    """
    episode_rewards = []
    episode_lengths = []
    metrics_history = {}
    
    start_time = time.time()
    
    for episode in range(num_episodes):
        # Reset environment
        observation = env.reset()
        agent_position = env.get_agent_position()
        episode_reward = 0
        episode_metrics = {}
        
        for step in range(max_steps):
            # Select and take action
            action = agent.select_action(observation, agent_position)
            next_observation, reward, done, info = env.step(action)
            next_agent_position = env.get_agent_position()
            
            # Update agent
            update_metrics = agent.update(
                observation, action, reward, next_observation, done,
                agent_position, next_agent_position
            )
            
            # Track metrics
            for key, value in update_metrics.items():
                if key not in episode_metrics:
                    episode_metrics[key] = []
                episode_metrics[key].append(value)
            
            # Update state
            observation = next_observation
            agent_position = next_agent_position
            episode_reward += reward
            
            if done:
                break
        
        # Record episode results
        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)
        
        # Average episode metrics
        for key, values in episode_metrics.items():
            avg_value = np.mean(values)
            if key not in metrics_history:
                metrics_history[key] = []
            metrics_history[key].append(avg_value)
        
        # Print progress
        if (episode + 1) % max(1, num_episodes // 20) == 0:
            avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
            avg_length = np.mean(episode_lengths[-100:]) if episode_lengths else 0
            elapsed = time.time() - start_time
            logger.info(
                f"Episode {episode+1}/{num_episodes} | "
                f"Reward: {episode_reward:.2f} | "
                f"Avg Reward: {avg_reward:.2f} | "
                f"Avg Length: {avg_length:.1f} | "
                f"Epsilon: {agent.exploration_rate:.3f} | "
                f"Time: {elapsed:.1f}s"
            )
    
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.1f} seconds")
    
    # Combine all metrics
    all_metrics = {
        'rewards': episode_rewards,
        'lengths': episode_lengths,
        **{k: v for k, v in metrics_history.items()}
    }
    
    return all_metrics


def evaluate(env, agent, num_episodes=10):
    """
    Evaluate the agent without training.
    
    Args:
        env: Environment to evaluate in
        agent: Agent to evaluate
        num_episodes: Number of episodes to evaluate
        
    Returns:
        Dictionary of evaluation metrics
    """
    rewards = []
    lengths = []
    
    for episode in range(num_episodes):
        observation = env.reset()
        agent_position = env.get_agent_position()
        episode_reward = 0
        
        for step in range(1000):  # Reasonable maximum
            # Select action (no exploration)
            action = agent.select_action(observation, agent_position, training=False)
            next_observation, reward, done, info = env.step(action)
            next_agent_position = env.get_agent_position()
            
            observation = next_observation
            agent_position = next_agent_position
            episode_reward += reward
            
            if done:
                break
        
        rewards.append(episode_reward)
        lengths.append(step + 1)
        
        logger.info(f"Evaluation episode {episode+1}: Reward = {episode_reward:.2f}, Length = {step+1}")
    
    # Calculate statistics
    avg_reward = np.mean(rewards)
    avg_length = np.mean(lengths)
    
    logger.info(f"Evaluation results: Avg Reward = {avg_reward:.2f}, Avg Length = {avg_length:.1f}")
    
    return {
        'avg_reward': avg_reward,
        'avg_length': avg_length,
        'rewards': rewards,
        'lengths': lengths
    }


def plot_results(metrics, smoothing=10):
    """
    Plot training results.
    
    Args:
        metrics: Dictionary of training metrics
        smoothing: Window size for smoothing the curves
    """
    if not metrics.get('rewards'):
        logger.warning("No training data to plot")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot rewards
    axes[0].plot(metrics['rewards'], alpha=0.3, label='Raw')
    if len(metrics['rewards']) >= smoothing:
        smooth_rewards = np.convolve(metrics['rewards'], 
                                     np.ones(smoothing)/smoothing, 
                                     mode='valid')
        axes[0].plot(np.arange(len(smooth_rewards)) + smoothing-1, 
                     smooth_rewards, label=f'Smoothed (window={smoothing})')
    axes[0].set_ylabel('Episode Reward')
    axes[0].set_title('Training Rewards')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot episode lengths
    axes[1].plot(metrics['lengths'], alpha=0.3, label='Raw')
    if len(metrics['lengths']) >= smoothing:
        smooth_lengths = np.convolve(metrics['lengths'], 
                                     np.ones(smoothing)/smoothing, 
                                     mode='valid')
        axes[1].plot(np.arange(len(smooth_lengths)) + smoothing-1, 
                     smooth_lengths, label=f'Smoothed (window={smoothing})')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Episode Length')
    axes[1].set_title('Training Episode Lengths')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a reinforcement learning agent')
    
    # Environment parameters
    parser.add_argument('--width', type=int, default=10, help='Environment width')
    parser.add_argument('--height', type=int, default=10, help='Environment height')
    
    # Agent parameters
    parser.add_argument('--learning-rate', type=float, default=0.1, help='Learning rate (alpha)')
    parser.add_argument('--discount-factor', type=float, default=0.99, help='Discount factor (gamma)')
    parser.add_argument('--exploration-rate', type=float, default=1.0, help='Initial exploration rate (epsilon)')
    parser.add_argument('--exploration-decay', type=float, default=0.995, help='Exploration decay rate')
    
    # Training parameters
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--max-steps', type=int, default=500, help='Maximum steps per episode')
    parser.add_argument('--eval-episodes', type=int, default=10, help='Number of evaluation episodes')
    
    # Output parameters
    parser.add_argument('--output-dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--save-agent', action='store_true', help='Save the trained agent')
    parser.add_argument('--plot', action='store_true', help='Plot training results')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    if args.save_agent or args.plot:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Create environment
    env = cudarl_core.Environment(args.width, args.height)
    
    # Create agent
    agent = QTableAgent(
        width=args.width,
        height=args.height,
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        exploration_rate=args.exploration_rate,
        exploration_decay=args.exploration_decay
    )
    
    # Train agent
    logger.info(f"Starting training for {args.episodes} episodes")
    metrics = train(
        env=env,
        agent=agent,
        num_episodes=args.episodes,
        max_steps=args.max_steps
    )
    
    # Evaluate agent
    if args.eval_episodes > 0:
        logger.info(f"Evaluating agent for {args.eval_episodes} episodes")
        eval_metrics = evaluate(env, agent, num_episodes=args.eval_episodes)
    
    # Save agent if requested
    if args.save_agent:
        agent_path = os.path.join(args.output_dir, "qtable_agent.npy")
        agent.save(agent_path)
    
    # Plot results if requested
    if args.plot:
        logger.info("Plotting training results")
        plot_results(metrics)
    
    logger.info("Done!")


if __name__ == "__main__":
    main()