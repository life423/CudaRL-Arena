import socket
import json
import time
import argparse
import sys
import os

# Add the build directory to the Python path for importing CUDA bindings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build', 'lib', 'Release'))

try:
    import cudarl_core_python
    CUDA_AVAILABLE = True
    print("CUDA bindings loaded successfully")
except ImportError as e:
    print(f"CUDA bindings not available: {e}")
    CUDA_AVAILABLE = False

class GodotVisualizer:
    def __init__(self, host='localhost', port=9999):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        
    def connect(self):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.connected = True
            print(f"Connected to Godot visualizer at {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"Failed to connect to Godot visualizer: {e}")
            self.connected = False
            return False
    
    def send_message(self, message_type, data):
        if not self.connected:
            return
            
        try:
            message = {"type": message_type, **data}
            json_data = json.dumps(message)
            self.socket.send((json_data + '\n').encode())
        except Exception as e:
            print(f"Failed to send message to Godot: {e}")
            self.connected = False
    
    def send_state_update(self, episode, step, reward, total_reward, state, action, done):
        self.send_message("state_update", {
            "episode": episode,
            "step": step,
            "reward": reward,
            "total_reward": total_reward,
            "state": state.tolist() if hasattr(state, 'tolist') else state,
            "action": action,
            "done": done
        })
    
    def send_episode_start(self, episode):
        self.send_message("episode_start", {"episode": episode})
    
    def send_episode_end(self, episode, total_reward):
        self.send_message("episode_end", {"episode": episode, "total_reward": total_reward})
    
    def disconnect(self):
        if self.socket:
            self.socket.close()
            self.connected = False

class MockEnvironment:
    """Mock environment when CUDA is not available"""
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.agent_pos = [0, 0]
        self.goal_pos = [grid_size-1, grid_size-1]
        self.step_count = 0
        self.max_steps = 100
        
    def reset(self):
        self.agent_pos = [0, 0]
        self.step_count = 0
        return self.get_state()
    
    def step(self, action):
        # Simple movement: 0=up, 1=right, 2=down, 3=left
        old_pos = self.agent_pos[:]
        
        if action == 0 and self.agent_pos[1] > 0:  # up
            self.agent_pos[1] -= 1
        elif action == 1 and self.agent_pos[0] < self.grid_size-1:  # right
            self.agent_pos[0] += 1
        elif action == 2 and self.agent_pos[1] < self.grid_size-1:  # down
            self.agent_pos[1] += 1
        elif action == 3 and self.agent_pos[0] > 0:  # left
            self.agent_pos[0] -= 1
            
        self.step_count += 1
        
        # Calculate reward
        if self.agent_pos == self.goal_pos:
            reward = 100.0
            done = True
        elif self.step_count >= self.max_steps:
            reward = -1.0
            done = True
        else:
            # Small negative reward for each step + bonus for getting closer to goal
            distance_to_goal = abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])
            reward = -0.1 - distance_to_goal * 0.01
            done = False
        
        return self.get_state(), reward, done
    
    def get_state(self):
        # Simple state: agent position + goal position + step count
        return [
            self.agent_pos[0] / self.grid_size,
            self.agent_pos[1] / self.grid_size,
            self.goal_pos[0] / self.grid_size,
            self.goal_pos[1] / self.grid_size,
            self.step_count / self.max_steps
        ]

def train_with_visualization(args):
    """Training loop with optional Godot visualization"""
    
    # Set up visualizer
    visualizer = None
    if args.use_godot_viz:
        visualizer = GodotVisualizer()
        if not visualizer.connect():
            print("Continuing without visualization...")
            visualizer = None
    
    # Set up environment
    if CUDA_AVAILABLE and args.use_cuda:
        print("Using CUDA environment")
        # TODO: Initialize CUDA environment here
        env = MockEnvironment()  # Fallback for now
    else:
        print("Using mock environment")
        env = MockEnvironment()
    
    # Training loop
    total_rewards = []
    
    for episode in range(args.episodes):
        if visualizer:
            visualizer.send_episode_start(episode)
            
        state = env.reset()
        total_reward = 0.0
        step = 0
        done = False
        
        while not done:
            # Simple random policy for demo
            import random
            action = random.randint(0, 3)
            
            # Take step
            next_state, reward, done = env.step(action)
            total_reward += reward
            step += 1
            
            # Send to visualizer
            if visualizer:
                visualizer.send_state_update(episode, step, reward, total_reward, state, action, done)
            
            state = next_state
            
            # Add small delay if visualizing
            if args.use_godot_viz and args.step_delay > 0:
                time.sleep(args.step_delay)
        
        total_rewards.append(total_reward)
        
        if visualizer:
            visualizer.send_episode_end(episode, total_reward)
        
        # Print progress
        if episode % args.print_every == 0:
            avg_reward = sum(total_rewards[-args.print_every:]) / min(len(total_rewards), args.print_every)
            print(f"Episode {episode:4d}: Total Reward = {total_reward:7.2f}, Avg = {avg_reward:7.2f}")
    
    # Cleanup
    if visualizer:
        visualizer.disconnect()
    
    # Final statistics
    print("\nTraining completed!")
    print(f"Episodes: {len(total_rewards)}")
    print(f"Average reward: {sum(total_rewards) / len(total_rewards):.2f}")
    print(f"Best episode: {max(total_rewards):.2f}")
    print(f"Final 10 episodes average: {sum(total_rewards[-10:]) / min(10, len(total_rewards)):.2f}")

def main():
    parser = argparse.ArgumentParser(description='Train CudaRL agent with optional Godot visualization')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes to train')
    parser.add_argument('--use-godot-viz', action='store_true', help='Connect to Godot for visualization')
    parser.add_argument('--use-cuda', action='store_true', default=True, help='Use CUDA environment if available')
    parser.add_argument('--step-delay', type=float, default=0.0, help='Delay between steps when visualizing (seconds)')
    parser.add_argument('--print-every', type=int, default=10, help='Print progress every N episodes')
    
    args = parser.parse_args()
    
    print("CudaRL Training with Optional Godot Visualization")
    print("=" * 50)
    print(f"Episodes: {args.episodes}")
    print(f"CUDA Available: {CUDA_AVAILABLE}")
    print(f"Use CUDA: {args.use_cuda}")
    print(f"Godot Visualization: {args.use_godot_viz}")
    print("=" * 50)
    
    train_with_visualization(args)

if __name__ == "__main__":
    main()
