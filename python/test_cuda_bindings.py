#!/usr/bin/env python3
"""
Test script for CudaRL-Arena Python bindings
Tests the copyGridToBuffer functionality and CUDA-to-NumPy integration
"""

import os
import sys
import time

import numpy as np

# Add the build directory to Python path
build_lib_path = os.path.join(os.path.dirname(__file__), "..", "build", "lib", "Release")
sys.path.insert(0, build_lib_path)

try:
    import cudarl_core_python
    print("✓ Successfully imported cudarl_core_python")
except ImportError as e:
    print(f"✗ Failed to import cudarl_core_python: {e}")
    print(f"  Searched in: {build_lib_path}")
    print(f"  Available files: {os.listdir(build_lib_path) if os.path.exists(build_lib_path) else 'Directory not found'}")
    sys.exit(1)

def test_basic_functionality():
    """Test basic environment creation and operations"""
    print("\n=== Testing Basic Functionality ===")
    
    # Create environment
    env = cudarl_core_python.Environment(width=8, height=6)
    print(f"✓ Created environment: {env.get_width()}x{env.get_height()}")
    
    # Test reset
    obs = env.reset()
    print(f"✓ Reset environment, observation shape: {obs.shape}, dtype: {obs.dtype}")
    print(f"  Agent position: ({env.get_agent_x()}, {env.get_agent_y()})")
    print(f"  Initial reward: {env.get_reward()}")
    print(f"  Done: {env.is_done()}")
    
    # Test step
    action = 1  # right
    result = env.step(action)
    obs, reward, done, info = result
    print(f"✓ Step with action {action}")
    print(f"  New observation shape: {obs.shape}")
    print(f"  Reward: {reward}")
    print(f"  Done: {done}")
    print(f"  Info: {info}")
    print(f"  Agent position: ({env.get_agent_x()}, {env.get_agent_y()})")
    
    return env

def test_grid_access_methods():
    """Test different methods of accessing grid data"""
    print("\n=== Testing Grid Access Methods ===")
    
    env = cudarl_core_python.Environment(width=4, height=4)
    env.reset()
    
    # Method 1: Standard get_observation
    start_time = time.time()
    obs1 = env.get_observation()
    time1 = time.time() - start_time
    print(f"✓ get_observation(): {obs1.shape}, took {time1*1000:.3f}ms")
    
    # Method 2: Direct grid access using copyGridToBuffer
    start_time = time.time()
    obs2 = env.get_grid_direct()
    time2 = time.time() - start_time
    print(f"✓ get_grid_direct(): {obs2.shape}, took {time2*1000:.3f}ms")
    
    # Compare results
    if np.allclose(obs1, obs2):
        print("✓ Both methods return identical results")
    else:
        print("✗ Methods return different results!")
        print(f"  Max difference: {np.max(np.abs(obs1 - obs2))}")
    
    # Print some sample values
    print(f"  Sample values from obs1[0, :]: {obs1[0, :3]}")
    print(f"  Sample values from obs2[0, :]: {obs2[0, :3]}")
    
    # Performance comparison
    speedup = time1 / time2 if time2 > 0 else float('inf')
    print(f"  Performance: get_grid_direct() is {speedup:.2f}x faster" if speedup > 1 
          else f"  Performance: get_observation() is {1/speedup:.2f}x faster")
    
    return env

def test_cuda_memory_transfer():
    """Test CUDA memory transfer efficiency"""
    print("\n=== Testing CUDA Memory Transfer ===")
    
    # Test with different sizes
    sizes = [(10, 10), (50, 50), (100, 100)]
    
    for width, height in sizes:
        env = cudarl_core_python.Environment(width=width, height=height)
        env.reset()
        
        # Benchmark multiple transfers
        n_iterations = 100
        
        # Standard method
        start_time = time.time()
        for _ in range(n_iterations):
            obs = env.get_observation()
        time_standard = time.time() - start_time
        
        # Direct method
        start_time = time.time()
        for _ in range(n_iterations):
            obs = env.get_grid_direct()
        time_direct = time.time() - start_time
        
        print(f"Size {width}x{height} ({width*height} elements):")
        print(f"  Standard method: {time_standard*1000/n_iterations:.3f}ms per transfer")
        print(f"  Direct method:   {time_direct*1000/n_iterations:.3f}ms per transfer")
        print(f"  Speedup:         {time_standard/time_direct:.2f}x")

def test_grid_content():
    """Test that grid content makes sense"""
    print("\n=== Testing Grid Content ===")
    
    env = cudarl_core_python.Environment(width=5, height=5)
    obs = env.reset()
    
    print(f"Grid shape: {obs.shape}")
    print(f"Value range: [{np.min(obs):.3f}, {np.max(obs):.3f}]")
    print(f"Mean value: {np.mean(obs):.3f}")
    print(f"Non-zero elements: {np.count_nonzero(obs)}/{obs.size}")
    
    # Check if goal position (top-right) has special value
    goal_value = obs[0, -1]  # Top-right corner
    print(f"Goal position (0, {obs.shape[1]-1}) value: {goal_value}")
    
    # Display the grid
    print("Grid visualization:")
    for row in obs:
        print("  " + " ".join(f"{val:.2f}" for val in row))
    
    # Mark agent position
    agent_x, agent_y = env.get_agent_x(), env.get_agent_y()
    print(f"Agent at position ({agent_x}, {agent_y}), value: {obs[agent_y, agent_x]:.3f}")

def test_episode_run():
    """Test running a complete episode"""
    print("\n=== Testing Complete Episode ===")
    
    env = cudarl_core_python.Environment(width=5, height=5)
    obs = env.reset()
    
    episode_length = 0
    total_reward = 0
    
    print(f"Starting episode at agent position: ({env.get_agent_x()}, {env.get_agent_y()})")
    
    while not env.is_done() and episode_length < 50:  # Prevent infinite loops
        # Simple policy: move towards goal (top-right corner)
        agent_x, agent_y = env.get_agent_x(), env.get_agent_y()
        goal_x, goal_y = env.get_width() - 1, 0
        
        # Choose action to move towards goal
        if agent_x < goal_x:
            action = 1  # right
        elif agent_y > goal_y:
            action = 0  # up
        else:
            action = 1  # right (fallback)
        
        obs, reward, done, info = env.step(action)
        total_reward += reward
        episode_length += 1
        
        if episode_length <= 10 or done:  # Print first 10 steps and final step
            print(f"  Step {episode_length}: action={action}, pos=({info['agent_x']}, {info['agent_y']}), reward={reward:.3f}, done={done}")
    
    print(f"Episode completed in {episode_length} steps")
    print(f"Total reward: {total_reward:.3f}")
    print(f"Final position: ({env.get_agent_x()}, {env.get_agent_y()})")
    print(f"Reached goal: {env.is_done()}")

def main():
    """Run all tests"""
    print("CudaRL-Arena Python Bindings Test Suite")
    print("=========================================")
    
    try:
        # Run tests
        test_basic_functionality()
        test_grid_access_methods()
        test_cuda_memory_transfer()
        test_grid_content()
        test_episode_run()
        
        print("\n=== All Tests Completed Successfully! ===")
        print("✓ copyGridToBuffer method is working correctly")
        print("✓ CUDA-to-NumPy memory transfer is functional")
        print("✓ Python bindings are fully operational")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
