"""
CudaRL-Arena Python Package

A high-performance reinforcement learning environment using CUDA acceleration.
"""

import logging
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Smart import with CUDA fallback
try:
    # Attempt to import the compiled CUDA environment
    from cudarl_core_python import Environment as _CudaEnvironment
    CUDA_AVAILABLE = True
    logger.info("CUDA environment loaded successfully")
except ImportError as e:
    # Fall back to mock environment
    from .mock_env import MockEnvironment as _CudaEnvironment
    CUDA_AVAILABLE = False
    logger.warning(f"CUDA environment not available: {e}")
    logger.info("Using CPU-only mock environment as fallback")

# Import other components
from .environment import Environment
from .agent import BaseAgent, RandomAgent, QLearningAgent, DQNAgent, create_agent
from .mock_env import Agent, QTableAgent
from .trainer import Trainer

# Export the CUDA/mock environment directly
CudaEnvironment = _CudaEnvironment

__version__ = '0.1.0'
__all__ = ['Environment', 'CudaEnvironment', 'Agent', 'QTableAgent', 'Trainer', 'CUDA_AVAILABLE']

# Issue deprecation warning if using fallback
if not CUDA_AVAILABLE:
    warnings.warn(
        "CudaRL-Arena is running in CPU fallback mode. "
        "For optimal performance, please build the CUDA extension.",
        UserWarning,
        stacklevel=2
    )
