#!/usr/bin/env python3
"""
Test script to verify environment integration works with fallback.
"""

import sys
import numpy as np

def test_environment_import():
    """Test that the environment can be imported and used."""
    try:
        import python.cudarl as cudarl
        print(f"✓ Successfully imported cudarl")
        print(f"  CUDA Available: {cudarl.CUDA_AVAILABLE}")
        print(f"  Version: {cudarl.__version__}")
        
        # Test environment creation
        env = cudarl.Environment(width=5, height=5)
        print(f"✓ Successfully created environment")
        
        # Test reset
        obs = env.reset()
        print(f"✓ Reset successful, observation shape: {obs.shape}")
        print(f"  Observation type: {type(obs)}")
        
        # Test step
        action = 1  # Move right
        obs, reward, done, info = env.step(action)
        print(f"✓ Step successful")
        print(f"  Observation shape: {obs.shape}")
        print(f"  Reward: {reward}")
        print(f"  Done: {done}")
        print(f"  Info: {info}")
        
        # Test rendering
        print("\n--- ASCII Rendering ---")
        env.render(mode='human')
        
        print("\n✓ All tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_environment_import()
    sys.exit(0 if success else 1)
