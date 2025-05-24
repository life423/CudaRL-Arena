#!/usr/bin/env python3
import sys
import os

# Print Python path information
print("Python path:", sys.path)
print("Current directory:", os.getcwd())

# Try to import the module
try:
    from python.cudarl import Environment
    print("Module imported successfully")
    
    # Create environment
    env = Environment(width=5, height=5)
    print(f"Environment created: width={env.width}, height={env.height}")
    
    # Check if C++ environment is initialized
    print("C++ environment initialized:", env._cpp_env is not None)
    
    # If not initialized, print possible reasons
    if env._cpp_env is None:
        print("\nPossible reasons for binding failure:")
        print("1. C++ extension not built correctly")
        print("2. Extension not in Python path")
        print("3. Extension built with incompatible Python version")
        print("4. Missing dependencies")
except ImportError as e:
    print(f"Import error: {e}")
except Exception as e:
    print(f"Other error: {e}")