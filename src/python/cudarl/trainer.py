"""
CudaRL Trainer module for reinforcement learning training loops.

This module provides training utilities and classes for RL agents,
including training loops, data collection, and performance monitoring.
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from collections import defaultdict, deque
import json
import os


class TrainingStats:
    """Training statistics tracker."""
    
    def __init__(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.losses = []
        self.epsilons = []
        self.success_rate_window = deque(maxlen=100)
        self.start_time = time.time()
        
    def add_episode(self, reward: float, length: int, success: bool, episode_time: float):
        """Add episode statistics."""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.episode_times.append(episode_time)
        self.success_rate_window.append(1.0 if success else 0.0)
        
    def add_training_step(self, loss: float, epsilon: float):
        """Add training step statistics."""
        self.losses.append(loss)
        self.epsilons.append(epsilon)
        
    def get_summary(self, window_size: int = 100) -> Dict[str, Any]:
        """Get training summary statistics."""
        if not self.episode_rewards:
            return {}
            
        recent_rewards = self.episode_rewards[-window_size:]
        recent_lengths = self.episode_lengths[-window_size:]
        
        summary = {
            'total_episodes': len(self.episode_rewards),
            'total_training_time': time.time() - self.start_time,
            'avg_reward': np.mean(recent_rewards),
            'std_reward': np.std(recent_rewards),
            'min_reward': np.min(recent_rewards),
            'max_reward': np.max(recent_rewards),
            'avg_episode_length': np.mean(recent_lengths),
            'success_rate': np.mean(self.success_rate_window) if self.success_rate_window else 0.0,
            'total_reward': np.sum(self.episode_rewards),
        }
        
        if self.losses:
            summary.update({
                'avg_loss': np.mean(self.losses[-window_size:]),
                'current_epsilon': self.epsilons[-1] if self.epsilons else 0.0,
            })
            
        return summary
        
    def save(self, filepath: str):
        """Save training statistics."""
        data = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_times': self.episode_times,
            'losses': self.losses,
            'epsilons': self.epsilons,
            'start_time': self.start_time
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
    def load(self, filepath: str):
        """Load training statistics."""
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        self.episode_rewards = data.get('episode_rewards', [])
        self.episode_lengths = data.get('episode_lengths', [])
        self.episode_times = data.get('episode_times', [])
        self.losses = data.get('losses', [])
        self.epsilons = data.get('epsilons', [])
        self.start_time = data.get('start_time', time.time())


class Trainer:
    """Main trainer class for RL agents."""
    
    def __init__(
        self,
        env,
        agent,
        max_episodes: int = 1000,
        max_steps_per_episode: int = 200,
        save_frequency: int = 100,
        log_frequency: int = 10,
        eval_frequency: int = 50,
        eval_episodes: int = 10,
        target_reward: Optional[float] = None,
        save_dir: str = "checkpoints"
    ):
        self.env = env
        self.agent = agent
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.save_frequency = save_frequency
        self.log_frequency = log_frequency
        self.eval_frequency = eval_frequency
        self.eval_episodes = eval_episodes
        self.target_reward = target_reward
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Training statistics
        self.stats = TrainingStats()
        self.best_avg_reward = float('-inf')
        self.episode = 0
        
        # Callbacks
        self.callbacks = []
        
    def add_callback(self, callback: Callable):
        """Add training callback."""
        self.callbacks.append(callback)
        
    def train(self) -> TrainingStats:
        """Main training loop."""
        print(f"Starting training for {self.max_episodes} episodes...")
        print(f"Agent: {self.agent.name}")
        print(f"Environment: {type(self.env).__name__}")
        print(f"Save directory: {self.save_dir}")
        print("-" * 50)
        
        for episode in range(self.max_episodes):
            self.episode = episode
            episode_start_time = time.time()
            
            # Run episode
            episode_reward, episode_length, success = self._run_episode()
            episode_time = time.time() - episode_start_time
            
            # Update statistics
            self.stats.add_episode(episode_reward, episode_length, success, episode_time)
            
            # Logging
            if episode % self.log_frequency == 0:
                self._log_progress(episode)
                
            # Evaluation
            if episode % self.eval_frequency == 0 and episode > 0:
                eval_reward = self._evaluate()
                if eval_reward > self.best_avg_reward:
                    self.best_avg_reward = eval_reward
                    self._save_best_model()
                    
            # Saving
            if episode % self.save_frequency == 0 and episode > 0:
                self._save_checkpoint(episode)
                
            # Early stopping
            if self.target_reward and episode_reward >= self.target_reward:
                print(f"Target reward {self.target_reward} achieved in episode {episode}!")
                break
                
            # Run callbacks
            for callback in self.callbacks:
                callback(self, episode)
                
        print("\nTraining completed!")
        self._save_final_results()
        return self.stats
        
    def _run_episode(self) -> Tuple[float, int, bool]:
        """Run a single training episode."""
        observation = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        
        for step in range(self.max_steps_per_episode):
            # Select action
            action = self.agent.act(observation)
            
            # Take step
            next_observation, reward, done, info = self.env.step(action)
            
            # Learn from experience
            experience = (observation, action, reward, next_observation, done)
            learn_info = self.agent.learn(experience)
            
            # Update statistics
            if learn_info:
                loss = learn_info.get('loss', 0.0)
                epsilon = learn_info.get('epsilon', 0.0)
                self.stats.add_training_step(loss, epsilon)
                
            # Update for next step
            observation = next_observation
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
                
        # Determine success (environment specific)
        success = self._check_success(episode_reward, info if 'info' in locals() else {})
        
        return episode_reward, episode_length, success
        
    def _check_success(self, reward: float, info: Dict) -> bool:
        """Check if episode was successful."""
        # Default success criteria
        if hasattr(self.env, 'is_success'):
            return self.env.is_success()
        elif 'success' in info:
            return info['success']
        else:
            # Use reward threshold as success criterion
            return reward > 0.5
            
    def _evaluate(self) -> float:
        """Evaluate agent performance."""
        print(f"Evaluating agent for {self.eval_episodes} episodes...")
        
        # Set agent to evaluation mode
        self.agent.set_training(False)
        
        eval_rewards = []
        for _ in range(self.eval_episodes):
            observation = self.env.reset()
            episode_reward = 0.0
            
            for _ in range(self.max_steps_per_episode):
                action = self.agent.act(observation)
                observation, reward, done, _ = self.env.step(action)
                episode_reward += reward
                
                if done:
                    break
                    
            eval_rewards.append(episode_reward)
            
        # Restore training mode
        self.agent.set_training(True)
        
        avg_eval_reward = np.mean(eval_rewards)
        print(f"Evaluation: Average reward = {avg_eval_reward:.3f}")
        
        return avg_eval_reward
        
    def _log_progress(self, episode: int):
        """Log training progress."""
        summary = self.stats.get_summary(window_size=self.log_frequency)
        
        if summary:
            print(f"Episode {episode:4d} | "
                  f"Avg Reward: {summary['avg_reward']:7.3f} | "
                  f"Success Rate: {summary['success_rate']:5.3f} | "
                  f"Eps: {summary.get('current_epsilon', 0):5.3f} | "
                  f"Avg Length: {summary['avg_episode_length']:5.1f}")
                  
    def _save_checkpoint(self, episode: int):
        """Save training checkpoint."""
        checkpoint_path = os.path.join(self.save_dir, f"checkpoint_episode_{episode}")
        
        # Save agent
        agent_path = f"{checkpoint_path}_agent.json"
        self.agent.save(agent_path)
        
        # Save statistics
        stats_path = f"{checkpoint_path}_stats.json"
        self.stats.save(stats_path)
        
        print(f"Checkpoint saved: episode {episode}")
        
    def _save_best_model(self):
        """Save best performing model."""
        best_path = os.path.join(self.save_dir, "best_model.json")
        self.agent.save(best_path)
        print(f"New best model saved! Avg reward: {self.best_avg_reward:.3f}")
        
    def _save_final_results(self):
        """Save final training results."""
        # Save final agent
        final_agent_path = os.path.join(self.save_dir, "final_agent.json")
        self.agent.save(final_agent_path)
        
        # Save final statistics
        final_stats_path = os.path.join(self.save_dir, "final_stats.json")
        self.stats.save(final_stats_path)
        
        # Save summary
        summary = self.stats.get_summary()
        summary_path = os.path.join(self.save_dir, "training_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"Final results saved to {self.save_dir}")
        
    def load_checkpoint(self, episode: int):
        """Load training checkpoint."""
        checkpoint_path = os.path.join(self.save_dir, f"checkpoint_episode_{episode}")
        
        # Load agent
        agent_path = f"{checkpoint_path}_agent.json"
        if os.path.exists(agent_path):
            self.agent.load(agent_path)
            
        # Load statistics
        stats_path = f"{checkpoint_path}_stats.json"
        if os.path.exists(stats_path):
            self.stats.load(stats_path)
            
        print(f"Checkpoint loaded: episode {episode}")


class VectorizedTrainer:
    """Trainer for vectorized environments (multiple environments in parallel)."""
    
    def __init__(
        self,
        env,  # VectorizedEnvironment
        agents,  # List of agents or single agent for all envs
        max_episodes: int = 1000,
        max_steps_per_episode: int = 200,
        save_frequency: int = 100,
        log_frequency: int = 10,
        save_dir: str = "vectorized_checkpoints"
    ):
        self.env = env
        self.num_envs = env.getNumEnvironments()
        
        # Handle agents
        if isinstance(agents, list):
            assert len(agents) == self.num_envs, "Number of agents must match number of environments"
            self.agents = agents
        else:
            # Single agent for all environments
            self.agents = [agents] * self.num_envs
            
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.save_frequency = save_frequency
        self.log_frequency = log_frequency
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Training statistics per environment
        self.env_stats = [TrainingStats() for _ in range(self.num_envs)]
        self.global_stats = TrainingStats()
        self.episode = 0
        
    def train(self) -> List[TrainingStats]:
        """Main vectorized training loop."""
        print(f"Starting vectorized training for {self.max_episodes} episodes...")
        print(f"Number of environments: {self.num_envs}")
        print(f"Agents: {[agent.name for agent in self.agents]}")
        print("-" * 50)
        
        # Reset all environments
        observations = self.env.reset()
        episode_rewards = [0.0] * self.num_envs
        episode_lengths = [0] * self.num_envs
        episode_start_times = [time.time()] * self.num_envs
        
        for episode in range(self.max_episodes):
            self.episode = episode
            
            for step in range(self.max_steps_per_episode):
                # Get actions from all agents
                actions = []
                for i, agent in enumerate(self.agents):
                    action = agent.act(observations[i])
                    actions.append(action)
                
                # Step all environments
                next_observations, rewards, dones, infos = self.env.step(actions)
                
                # Process each environment
                for i in range(self.num_envs):
                    # Learn from experience
                    experience = (
                        observations[i], 
                        actions[i], 
                        rewards[i], 
                        next_observations[i], 
                        dones[i]
                    )
                    learn_info = self.agents[i].learn(experience)
                    
                    # Update statistics
                    episode_rewards[i] += rewards[i]
                    episode_lengths[i] += 1
                    
                    if learn_info:
                        loss = learn_info.get('loss', 0.0)
                        epsilon = learn_info.get('epsilon', 0.0)
                        self.env_stats[i].add_training_step(loss, epsilon)
                    
                    # Handle episode completion
                    if dones[i]:
                        episode_time = time.time() - episode_start_times[i]
                        success = rewards[i] > 0.5  # Simple success criterion
                        
                        # Update environment stats
                        self.env_stats[i].add_episode(
                            episode_rewards[i], 
                            episode_lengths[i], 
                            success, 
                            episode_time
                        )
                        
                        # Update global stats
                        self.global_stats.add_episode(
                            episode_rewards[i], 
                            episode_lengths[i], 
                            success, 
                            episode_time
                        )
                        
                        # Reset for next episode
                        episode_rewards[i] = 0.0
                        episode_lengths[i] = 0
                        episode_start_times[i] = time.time()
                        
                        # Reset single environment
                        observations[i] = self.env.resetSingle(i)
                    else:
                        observations[i] = next_observations[i]
                
                # Check if all environments are done (optional early termination)
                if all(dones):
                    observations = self.env.reset()
                    episode_rewards = [0.0] * self.num_envs
                    episode_lengths = [0] * self.num_envs
                    episode_start_times = [time.time()] * self.num_envs
            
            # Logging
            if episode % self.log_frequency == 0:
                self._log_progress(episode)
                
            # Saving
            if episode % self.save_frequency == 0 and episode > 0:
                self._save_checkpoint(episode)
                
        print("\nVectorized training completed!")
        self._save_final_results()
        return self.env_stats
        
    def _log_progress(self, episode: int):
        """Log training progress."""
        global_summary = self.global_stats.get_summary(window_size=self.log_frequency * self.num_envs)
        
        if global_summary:
            print(f"Episode {episode:4d} | "
                  f"Global Avg Reward: {global_summary['avg_reward']:7.3f} | "
                  f"Success Rate: {global_summary['success_rate']:5.3f} | "
                  f"Total Episodes: {global_summary['total_episodes']:6d}")
                  
    def _save_checkpoint(self, episode: int):
        """Save vectorized training checkpoint."""
        checkpoint_dir = os.path.join(self.save_dir, f"checkpoint_episode_{episode}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save each agent
        for i, agent in enumerate(self.agents):
            agent_path = os.path.join(checkpoint_dir, f"agent_{i}.json")
            agent.save(agent_path)
            
        # Save statistics
        for i, stats in enumerate(self.env_stats):
            stats_path = os.path.join(checkpoint_dir, f"env_{i}_stats.json")
            stats.save(stats_path)
            
        # Save global statistics
        global_stats_path = os.path.join(checkpoint_dir, "global_stats.json")
        self.global_stats.save(global_stats_path)
        
        print(f"Vectorized checkpoint saved: episode {episode}")
        
    def _save_final_results(self):
        """Save final vectorized training results."""
        final_dir = os.path.join(self.save_dir, "final_results")
        os.makedirs(final_dir, exist_ok=True)
        
        # Save final agents
        for i, agent in enumerate(self.agents):
            final_agent_path = os.path.join(final_dir, f"final_agent_{i}.json")
            agent.save(final_agent_path)
            
        # Save final statistics
        for i, stats in enumerate(self.env_stats):
            final_stats_path = os.path.join(final_dir, f"final_env_{i}_stats.json")
            stats.save(final_stats_path)
            
        # Save global summary
        global_summary = self.global_stats.get_summary()
        summary_path = os.path.join(final_dir, "global_training_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(global_summary, f, indent=2)
            
        print(f"Final vectorized results saved to {final_dir}")


def create_trainer(
    env,
    agent,
    trainer_type: str = "standard",
    **kwargs
) -> Trainer:
    """Factory function to create trainers."""
    if trainer_type.lower() == "vectorized":
        return VectorizedTrainer(env, agent, **kwargs)
    elif trainer_type.lower() == "standard":
        return Trainer(env, agent, **kwargs)
    else:
        raise ValueError(f"Unknown trainer type: {trainer_type}. Available: ['standard', 'vectorized']")
