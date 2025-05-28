#!/usr/bin/env python3
"""
Unit tests for the Python environment wrapper.
"""

import pytest
import numpy as np


class TestEnvironment:
    """Test cases for the Environment class."""
    
    def test_initialization(self, small_env):
        """Test environment initialization."""
        assert small_env.width == 5
        assert small_env.height == 5
        assert small_env.env_id == 0
    
    def test_reset(self, small_env):
        """Test environment reset."""
        obs = small_env.reset()
        
        # Check observation shape
        assert obs.shape == (5, 5)
        
        # Check observation type
        assert isinstance(obs, np.ndarray)
    
    def test_step_without_cpp(self, small_env):
        """Test environment step without C++ backend (dummy mode)."""
        small_env.reset()
        
        # Take a step - should return dummy values
        obs, reward, done, info = small_env.step(1)  # Move right
        
        # Check return types
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        
        # Check observation shape
        assert obs.shape == (5, 5)
        
        # In dummy mode, reward should be 0.0
        assert reward == 0.0
        
        # Check done is False for dummy mode
        assert done is False
        
        # Check info contains agent position
        assert 'agent_x' in info
        assert 'agent_y' in info
    
    def test_step_with_cpp_mock(self, env_with_cpp_mock):
        """Test environment step with mocked C++ backend."""
        env_with_cpp_mock.reset()
        
        # Take a step
        obs, reward, done, info = env_with_cpp_mock.step(1)  # Move right
        
        # Check return types
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        
        # Check observation shape
        assert obs.shape == (5, 5)
        
        # With mock, reward should be negative step penalty
        assert reward < 0
        
        # Check done is False for normal step
        assert done is False
        
        # Check info contains agent position
        assert 'agent_x' in info
        assert 'agent_y' in info
        
        # Agent should have moved
        assert info['agent_x'] > 2  # Started at center (2,2), moved right
    
    def test_render_human(self, small_env):
        """Test human rendering mode."""
        small_env.reset()
        result = small_env.render(mode='human')
        assert result is None
    
    def test_render_rgb_array(self, small_env):
        """Test rgb_array rendering mode."""
        small_env.reset()
        result = small_env.render(mode='rgb_array')
        
        # Should return a numpy array with shape (height, width, 3)
        assert isinstance(result, np.ndarray)
        assert result.shape == (5, 5, 3)
        assert result.dtype == np.uint8
    
    def test_invalid_render_mode(self, small_env):
        """Test invalid rendering mode."""
        small_env.reset()
        with pytest.raises(ValueError):
            small_env.render(mode='invalid')
    
    def test_close(self, small_env):
        """Test environment close."""
        small_env.reset()
        small_env.close()
        # No assertion needed, just checking it doesn't raise an exception

    @pytest.mark.parametrize("width,height", [(3, 3), (10, 10), (15, 8)])
    def test_different_sizes(self, width, height):
        """Test environments with different sizes."""
        from python.cudarl import Environment
        env = Environment(width=width, height=height)
        
        obs = env.reset()
        assert obs.shape == (height, width)
        
        obs, reward, done, info = env.step(0)
        assert obs.shape == (height, width)
        
        env.close()
    
    def test_goal_reaching_with_mock(self, env_with_cpp_mock):
        """Test reaching the goal with mocked environment."""
        env_with_cpp_mock.reset()
        
        # Navigate to goal (top-right corner)
        # First move up
        for _ in range(2):  # Start at (2,2), need to go to (4,0)
            obs, reward, done, info = env_with_cpp_mock.step(0)  # Move up
            if done:
                break
        
        # Then move right
        for _ in range(2):
            obs, reward, done, info = env_with_cpp_mock.step(1)  # Move right
            if done:
                break
        
        # Should have reached goal
        assert info['agent_x'] == 4
        assert info['agent_y'] == 0
        assert reward == 1.0
        assert done is True


if __name__ == '__main__':
    pytest.main([__file__])
