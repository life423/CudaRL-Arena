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
    parser = argparse.ArgumentParser(description='Run a trained agent in GridWorld')
    parser.add_argument('--grid-width', type=int, default=10, help='Width of the grid')
    parser.add_argument('--grid-height', type=int, default=10, help='Height of the grid')
    parser.add_argument('--model', type=str, default='q_table.npy', help='Path to the trained model')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes to run')
    parser.add_argument('--max-steps', type=int, default=100, help='Maximum steps per episode')
    parser.add_argument('--fps', type=int, default=5, help='FPS for rendering')
    parser.add_argument('--human', action='store_true', help='Enable human player mode (WASD keys)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create environment
    env = cgw.GridWorld(args.grid_width, args.grid_height)
    
    # Create agent
    agent = QLearningAgent(
        state_space_size=env.get_state_space_size(),
        action_space_size=env.get_action_space_size()
    )
    
    # Load trained model if it exists
    if os.path.exists(args.model):
        agent.load_model(args.model)
        print(f"Loaded model from {args.model}")
    else:
        print(f"Warning: Model file {args.model} not found. Using untrained agent.")
    
    # Create visualizer
    visualizer = GridWorldVisualizer(args.grid_width, args.grid_height)
    
    # Run episodes
    for episode in range(args.episodes):
        state = env.reset()
        episode_reward = 0
        
        print(f"\nEpisode {episode+1}/{args.episodes}")
        
        for step in range(args.max_steps):
            # Get action (either from agent or human)
            if args.human:
                # Wait for key press (WASD)
                action = get_human_action()
            else:
                # Use agent with no exploration
                action = cgw.get_best_action(agent.q_table, state)
            
            # Take action
            next_state, reward, done = env.step(action)
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            
            # Render
            if not visualizer.render(env, agent.q_table, args.fps):
                print("Visualization window closed. Exiting...")
                return
            
            print(f"Step {step+1}: Reward = {reward:.2f}, Total = {episode_reward:.2f}")
            
            if done:
                print(f"Episode finished after {step+1} steps with total reward {episode_reward:.2f}")
                time.sleep(1)  # Pause to see the final state
                break
    
    # Close visualizer
    visualizer.close()

def get_human_action():
    """
    Get action from keyboard input.
    W = Up (0), D = Right (1), S = Down (2), A = Left (3)
    """
    import pygame
    
    action = None
    while action is None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    action = 0  # Up
                elif event.key == pygame.K_d:
                    action = 1  # Right
                elif event.key == pygame.K_s:
                    action = 2  # Down
                elif event.key == pygame.K_a:
                    action = 3  # Left
        
        pygame.time.wait(10)  # Small delay to prevent CPU hogging
    
    return action

if __name__ == "__main__":
    main()