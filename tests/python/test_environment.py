import sys
import os
import numpy as np

# Add build directory to path for the compiled module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 
                                '../../build/Release'))

try:
    import cudarl_core_python
    print("✅ Successfully imported cudarl_core_python module")
except ImportError as e:
    print(f"❌ Failed to import: {e}")
    sys.exit(1)


def test_environment_state():
    """Test if we can access environment state before step"""
    print("\n=== Testing Environment State Access ===")
    
    try:
        env = cudarl_core_python.Environment(5, 5)
        env.reset()
        
        # Test all getter methods
        print("Testing getter methods:")
        print(f"  get_width(): {env.get_width()}")
        print(f"  get_height(): {env.get_height()}")
        print(f"  get_agent_x(): {env.get_agent_x()}")
        print(f"  get_agent_y(): {env.get_agent_y()}")
        print(f"  get_reward(): {env.get_reward()}")
        print(f"  is_done(): {env.is_done()}")
        
        # Test observation
        obs = env.get_observation()
        print(f"  get_observation(): shape={obs.shape}, dtype={obs.dtype}")
        print("✓ All getters work correctly")
        
        # Now test if we can get obs AFTER reset
        print("\nTesting observation from reset:")
        obs2 = env.reset()
        print(f"  Reset observation: shape={obs2.shape}")
        print("✓ Reset returns observation correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Exception: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_step_minimal():
    """Absolute minimal step test"""
    print("\n=== Minimal Step Test ===")
    
    try:
        env = cudarl_core_python.Environment(3, 3)
        _ = env.reset()
        
        print("Environment state before step:")
        print(f"  Position: ({env.get_agent_x()}, {env.get_agent_y()})")
        print(f"  Reward: {env.get_reward()}")
        print(f"  Done: {env.is_done()}")
        
        print("\nAttempting step with action=1...")
        print("*** If it crashes here, the issue is in C++ Environment::step() ***")
        
        # The crash point
        result = env.step(1)
        
        print("✓ Step completed without crash!")
        print(f"  Result type: {type(result)}")
        print(f"  Result length: {len(result)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Python exception: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=== Python Environment Diagnostic Test ===\n")
    
    # First test if all getters work
    if not test_environment_state():
        print("\n❌ Environment state test failed")
        return 1
    
    # Then test the minimal step
    if not test_step_minimal():
        print("\n❌ Step test failed")
        print("\nDIAGNOSIS: The crash occurs in the C++ Environment::step() function")
        print("Check src/core/environment.cpp for issues like:")
        print("  - Null pointer dereference")
        print("  - Array out of bounds access")
        print("  - Uninitialized member variables")
        return 1
    
    print("\n🎉 All tests PASSED!")
    return 0


if __name__ == "__main__":
    sys.exit(main())