#!/usr/bin/env python3
"""
Wrapper script to run the GUI.
"""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the GUI
from src.gui import main

if __name__ == "__main__":
    main()