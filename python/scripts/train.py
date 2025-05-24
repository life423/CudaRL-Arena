#!/usr/bin/env python3
"""
Training script for CudaRL-Arena.

This script provides a command-line interface for training agents.
"""

import argparse
import logging
import sys
import os
import time
from pathlib import Path

# Add parent directory to path to import cudarl package
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from python.cudarl import Environment, Trainer
from python.cudarl.agent import RandomAgent, QTableAgent

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

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a reinforcement learning agent')
    
    # Environment parameters
    parser.add_argument('--width', type=int, default=10, help='Environment width')
    parser.add_argument('--height', type=int, default=10, help='Environment height')
    
    # Agent parameters
    parser.add_argument('--agent', type=str, default='qtable', choices=['random', 'qtable'],
                        help='Agent type to use')
    parser.add_argument('--learning-rate', type=float, default=0.1, help='Learning rate (alpha)')
    parser.add_argument('--discount-factor', type=float, default=0.99, help='Discount factor (gamma)')
    parser.add_argument('--exploration-rate', type=float, default=1.0, help='Initial exploration rate (epsilon)')
    parser.add_argument('--exploration-decay', type=float, default=0.995, help='Exploration decay rate')
    
    # Training parameters
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--max-steps', type=int, default=500, help='Maximum steps per episode')
    parser.add_argument('--render-every', type=int, default=100, help='Render every N episodes (0 to disable)')
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
    env = Environment(width=args.width, height=args.height)
    
    # Create agent
    observation_shape = (args.height, args.width)
    action_space_size = 4  # up, right, down, left
    
    if args.agent == 'random':
        agent = RandomAgent(action_space_size, observation_shape)
    elif args.agent == 'qtable':
        agent = QTableAgent(
            action_space_size, 
            observation_shape,
            learning_rate=args.learning_rate,
            discount_factor=args.discount_factor,
            exploration_rate=args.exploration_rate,
            exploration_decay=args.exploration_decay
        )
    else:
        raise ValueError(f"Unknown agent type: {args.agent}")
    
    # Create trainer
    trainer = Trainer(env, agent)
    
    # Train agent
    logger.info(f"Starting training for {args.episodes} episodes")
    start_time = time.time()
    
    metrics = trainer.train(
        num_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        render_every=args.render_every,
        verbose=True
    )
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.1f} seconds")
    
    # Evaluate agent
    if args.eval_episodes > 0:
        logger.info(f"Evaluating agent for {args.eval_episodes} episodes")
        eval_metrics = trainer.evaluate(num_episodes=args.eval_episodes, render=args.render_every > 0)
        logger.info(f"Evaluation results: {eval_metrics}")
    
    # Save agent if requested
    if args.save_agent:
        agent_path = os.path.join(args.output_dir, f"{args.agent}_agent.pkl")
        agent.save(agent_path)
        logger.info(f"Agent saved to {agent_path}")
    
    # Plot results if requested
    if args.plot:
        logger.info("Plotting training results")
        trainer.plot_results()
    
    logger.info("Done!")

if __name__ == "__main__":
    main()