"""
Trainer module for CudaRL-Arena.

This module provides training functionality for reinforcement learning agents.
"""

import logging
import time
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
from .environment import Environment
from .agent import Agent

# Configure logging
logger = logging.getLogger(__name__)

class Trainer:
    """
    Trainer class for reinforcement learning agents.
    """
    
    def __init__(self, env: Environment, agent: Agent):
        """
        Initialize the trainer.
        
        Args:
            env: Environment to train in
            agent: Agent to train
        """
        self.env = env
        self.agent = agent
        self.episode_rewards = []
        self.episode_lengths = []
        self.metrics_history = {}
        
        logger.info("Initialized trainer")
    
    def train(self, num_episodes: int, max_steps_per_episode: int = 1000, 
              render_every: int = 0, verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the agent for a specified number of episodes.
        
        Args:
            num_episodes: Number of episodes to train for
            max_steps_per_episode: Maximum steps per episode
            render_every: Render environment every N episodes (0 to disable)
            verbose: Whether to print progress
            
        Returns:
            Dictionary of training metrics
        """
        start_time = time.time()
        
        for episode in range(num_episodes):
            # Reset environment
            observation = self.env.reset()
            episode_reward = 0
            episode_metrics = {}
            
            for step in range(max_steps_per_episode):
                # Render if needed
                if render_every > 0 and episode % render_every == 0:
                    self.env.render()
                
                # Select and take action
                action = self.agent.select_action(observation)
                next_observation, reward, done, info = self.env.step(action)
                
                # Update agent
                update_metrics = self.agent.update(
                    observation, action, reward, next_observation, done
                )
                
                # Track metrics
                for key, value in update_metrics.items():
                    if key not in episode_metrics:
                        episode_metrics[key] = []
                    episode_metrics[key].append(value)
                
                # Update state
                observation = next_observation
                episode_reward += reward
                
                if done:
                    break
            
            # Record episode results
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(step + 1)
            
            # Average episode metrics
            for key, values in episode_metrics.items():
                avg_value = np.mean(values)
                if key not in self.metrics_history:
                    self.metrics_history[key] = []
                self.metrics_history[key].append(avg_value)
            
            # Print progress
            if verbose and (episode + 1) % max(1, num_episodes // 20) == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_length = np.mean(self.episode_lengths[-100:])
                elapsed = time.time() - start_time
                logger.info(
                    f"Episode {episode+1}/{num_episodes} | "
                    f"Avg Reward: {avg_reward:.2f} | "
                    f"Avg Length: {avg_length:.1f} | "
                    f"Time: {elapsed:.1f}s"
                )
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.1f} seconds")
        
        # Combine all metrics
        all_metrics = {
            'rewards': self.episode_rewards,
            'lengths': self.episode_lengths,
            **self.metrics_history
        }
        
        return all_metrics
    
    def evaluate(self, num_episodes: int = 10, render: bool = True) -> Dict[str, float]:
        """
        Evaluate the agent without training.
        
        Args:
            num_episodes: Number of episodes to evaluate
            render: Whether to render the environment
            
        Returns:
            Dictionary of evaluation metrics
        """
        rewards = []
        lengths = []
        
        for episode in range(num_episodes):
            observation = self.env.reset()
            episode_reward = 0
            
            for step in range(1000):  # Reasonable maximum
                if render:
                    self.env.render()
                
                # Select action (no exploration)
                action = self.agent.select_action(observation)
                next_observation, reward, done, info = self.env.step(action)
                
                observation = next_observation
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
    
    def plot_results(self, smoothing: int = 10) -> None:
        """
        Plot training results.
        
        Args:
            smoothing: Window size for smoothing the curves
        """
        if not self.episode_rewards:
            logger.warning("No training data to plot")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Plot rewards
        axes[0].plot(self.episode_rewards, alpha=0.3, label='Raw')
        if len(self.episode_rewards) >= smoothing:
            smooth_rewards = np.convolve(self.episode_rewards, 
                                         np.ones(smoothing)/smoothing, 
                                         mode='valid')
            axes[0].plot(np.arange(len(smooth_rewards)) + smoothing-1, 
                         smooth_rewards, label=f'Smoothed (window={smoothing})')
        axes[0].set_ylabel('Episode Reward')
        axes[0].set_title('Training Rewards')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot episode lengths
        axes[1].plot(self.episode_lengths, alpha=0.3, label='Raw')
        if len(self.episode_lengths) >= smoothing:
            smooth_lengths = np.convolve(self.episode_lengths, 
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
        plt.show()