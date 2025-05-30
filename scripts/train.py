#!/usr/bin/env python3
"""
Simple wrapper to run the training script with the mock environment.
"""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Create a mock cudarl_core module
import python.cudarl.mock_env as mock_env
sys.modules['cudarl_core'] = mock_env

# Import and run the training script
from src.train import main

if __name__ == "__main__":
    main()