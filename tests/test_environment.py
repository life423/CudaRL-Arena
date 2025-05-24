#!/usr/bin/env python3
"""
Unit tests for the Python environment wrapper.
"""

import unittest
import sys
import os
import numpy as np
from pathlib import Path

# Add parent directory to path to import cudarl package
sys.path.insert(0, str(Path(__file__).parent.parent))

from python.cudarl import Environment

class TestEnvironment(unittest.TestCase):
    """Test cases for the Environment class."""
    
    def setUp(self):
        """Set up test environment."""
        self.env = Environment(width=5, height=5)
    
    def test_initialization(self):
        """Test environment initialization."""
        self.assertEqual(self.env.width, 5)
        self.assertEqual(self.env.height, 5)
        self.assertEqual(self.env.env_id, 0)
    
    def test_reset(self):
        """Test environment reset."""
        obs = self.env.reset()
        
        # Check observation shape
        self.assertEqual(obs.shape, (5, 5))
        
        # Check observation type
        self.assertTrue(isinstance(obs, np.ndarray))
    
    def test_step(self):
        """Test environment step."""
        self.env.reset()
        
        # Take a step
        obs, reward, done, info = self.env.step(1)  # Move right
        
        # Check return types
        self.assertTrue(isinstance(obs, np.ndarray))
        self.assertTrue(isinstance(reward, float))
        self.assertTrue(isinstance(done, bool))
        self.assertTrue(isinstance(info, dict))
        
        # Check observation shape
        self.assertEqual(obs.shape, (5, 5))
        
        # Check reward is negative for normal step
        self.assertLess(reward, 0)
        
        # Check done is False for normal step
        self.assertFalse(done)
        
        # Check info contains agent position
        self.assertIn('agent_x', info)
        self.assertIn('agent_y', info)
    
    def test_render_human(self):
        """Test human rendering mode."""
        self.env.reset()
        result = self.env.render(mode='human')
        self.assertIsNone(result)
    
    def test_render_rgb_array(self):
        """Test rgb_array rendering mode."""
        self.env.reset()
        result = self.env.render(mode='rgb_array')
        
        # Should return a numpy array with shape (height, width, 3)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertEqual(result.shape, (5, 5, 3))
        self.assertEqual(result.dtype, np.uint8)
    
    def test_invalid_render_mode(self):
        """Test invalid rendering mode."""
        self.env.reset()
        with self.assertRaises(ValueError):
            self.env.render(mode='invalid')
    
    def test_close(self):
        """Test environment close."""
        self.env.reset()
        self.env.close()
        # No assertion needed, just checking it doesn't raise an exception

if __name__ == '__main__':
    unittest.main()