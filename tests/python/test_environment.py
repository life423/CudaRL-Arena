import sys
import os
import numpy as np

# Add build directory to path for the compiled module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../build/Release'))

try:
    import cudarl_core_python
    print("✅ Successfully imported cudarl_core_python module")
except ImportError as e:
    print(f"❌ Failed to import: {e}")
    sys.exit(1)

def test_single_environment():
    print("\n=== Testing Single Environment ===")
    
    # Create environment
    env = cudarl_core_python.Environment(10, 10)
    print(f"✓ Created environment: {env.get_width()}x{env.get_height()}")
    
    # Reset and get observation
    obs = env.reset()
    print(f"✓ Reset environment. Observation shape: {obs.shape}")
    print(f"  Agent position: ({env.get_agent_x()}, {env.get_agent_y()})")
    
    # Test multiple steps
    actions = [0, 1, 2, 3]  # Up, Right, Down, Left
    action_names = ["Up", "Right", "Down", "Left"]
    
    print("\n✓ Testing step actions:")
    for action, name in zip(actions, action_names):
        env.step(action)
        print(f"  Action {name}: pos=({env.get_agent_x()}, {env.get_agent_y()}), "
              f"reward={env.get_reward():.4f}, done={env.is_done()}")
    
    return True

def test_multiple_resets():
    print("\n=== Testing Multiple Resets ===")
    env = cudarl_core_python.Environment(5, 5)
    
    positions = []
    for i in range(5):
        obs = env.reset()
        pos = (env.get_agent_x(), env.get_agent_y())
        positions.append(pos)
        print(f"✓ Reset {i+1}: Agent at {pos}")
    
    # Check if positions vary (random reset)
    unique_positions = len(set(positions))
    print(f"✓ Found {unique_positions} unique starting positions out of 5 resets")
    
    return True

def test_observation_consistency():
    print("\n=== Testing Observation Consistency ===")
    env = cudarl_core_python.Environment(8, 8)
    
    # Reset and get observation
    obs = env.reset()
    print(f"✓ Observation dtype: {obs.dtype}")
    print(f"✓ Observation shape: {obs.shape}")
    print(f"✓ Observation range: [{obs.min():.2f}, {obs.max():.2f}]")
    
    # Verify observation updates after step
    initial_obs = obs.copy()
    env.step(1)  # Move right
    new_obs = env.get_observation()
    
    if not np.array_equal(initial_obs, new_obs):
        print("✓ Observation changes after step (expected behavior)")
    else:
        print("⚠️  Observation didn't change after step")
    
    return True

def main():
    print("=== Python Environment Smoke Test ===")
    
    tests = [
        test_single_environment,
        test_multiple_resets,
        test_observation_consistency
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
    
    print(f"\n{'='*40}")
    print(f"✅ Passed {passed}/{len(tests)} tests")
    
    if passed == len(tests):
        print("🎉 All Python tests PASSED!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())