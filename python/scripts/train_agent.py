import sys
import os
import time
import argparse
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import our modules
from agent.qlearning_agent import QLearningAgent
from visualization.gridworld_visualizer import GridWorldVisualizer

# Import CUDA bindings
try:
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'interface'))
    import cuda_gridworld_bindings as cgw
except ImportError:
    print("Error: Could not import CUDA GridWorld bindings.")
    print("Make sure to build the C++/CUDA components first.")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a Q-learning agent in GridWorld')
    parser.add_argument('--grid-width', type=int, default=10, help='Width of the grid')
    parser.add_argument('--grid-height', type=int, default=10, help='Height of the grid')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--max-steps', type=int, default=100, help='Maximum steps per episode')
    parser.add_argument('--learning-rate', type=float, default=0.1, help='Learning rate (alpha)')
    parser.add_argument('--discount-factor', type=float, default=0.99, help='Discount factor (gamma)')
    parser.add_argument('--exploration-rate', type=float, default=1.0, help='Initial exploration rate (epsilon)')
    parser.add_argument('--min-exploration-rate', type=float, default=0.01, help='Minimum exploration rate')
    parser.add_argument('--exploration-decay', type=float, default=0.995, help='Exploration rate decay')
    parser.add_argument('--render', action='store_true', help='Render the environment during training')
    parser.add_argument('--render-fps', type=int, default=10, help='FPS for rendering')
    parser.add_argument('--save-model', type=str, default='q_table.npy', help='Path to save the trained model')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create environment
    env = cgw.GridWorld(args.grid_width, args.grid_height)
    
    # Create agent
    agent = QLearningAgent(
        state_space_size=env.get_state_space_size(),
        action_space_size=env.get_action_space_size(),
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        exploration_rate=args.exploration_rate,
        min_exploration_rate=args.min_exploration_rate,
        exploration_decay=args.exploration_decay
    )
    
    # Create visualizer if rendering is enabled
    visualizer = None
    if args.render:
        visualizer = GridWorldVisualizer(args.grid_width, args.grid_height)
    
    # Training loop
    total_rewards = []
    start_time = time.time()
    
    for episode in range(args.episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(args.max_steps):
            # Select action
            action = agent.select_action(state)
            
            # Take action
            next_state, reward, done = env.step(action)
            
            # Store experience
            agent.store_experience(state, action, reward, next_state)
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            
            # Render if enabled
            if visualizer and episode % 10 == 0:  # Render every 10th episode
                if not visualizer.render(env, agent.q_table, args.render_fps):
                    print("Visualization window closed. Exiting...")
                    return
            
            if done:
                break
        
        # Update Q-table with remaining experiences
        agent.update_q_table()
        
        # Decay exploration rate
        agent.decay_exploration()
        
        # Track progress
        total_rewards.append(episode_reward)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(total_rewards[-10:])
            elapsed_time = time.time() - start_time
            print(f"Episode {episode+1}/{args.episodes} | Avg Reward: {avg_reward:.2f} | "
                  f"Exploration: {agent.exploration_rate:.2f} | Time: {elapsed_time:.2f}s")
    
    # Save the trained model
    if args.save_model:
        agent.save_model(args.save_model)
        print(f"Model saved to {args.save_model}")
    
    # Final statistics
    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.2f} seconds")
    print(f"Final average reward (last 100 episodes): {np.mean(total_rewards[-100:]):.2f}")
    
    # Close visualizer
    if visualizer:
        visualizer.close()

if __name__ == "__main__":
    main()