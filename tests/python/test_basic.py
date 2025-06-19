import sys
import os

# Add build directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../build/Release'))

try:
    import cudarl_core_python
    print("✅ Module imported successfully")
    
    # Test 1: Create environment
    print("\n1. Creating environment...")
    env = cudarl_core_python.Environment(5, 5)
    print(f"   ✓ Environment created: {env.get_width()}x{env.get_height()}")
    
    # Test 2: Reset
    print("\n2. Testing reset...")
    obs = env.reset()
    print(f"   ✓ Reset successful. Shape: {obs.shape}")
    print(f"   ✓ Agent at: ({env.get_agent_x()}, {env.get_agent_y()})")
    
    # Test 3: Single step (with error handling)
    print("\n3. Testing single step...")
    try:
        print("   Before step: Agent at", env.get_agent_x(), env.get_agent_y())
        env.step(0)  # Try moving up
        print("   ✓ Step successful")
        print("   After step: Agent at", env.get_agent_x(), env.get_agent_y())
        print("   Reward:", env.get_reward())
    except Exception as e:
        print(f"   ❌ Step failed: {e}")
    
    print("\n✅ Basic test completed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()