"""
CudaRL-Arena Python Package

A high-performance reinforcement learning environment using CUDA acceleration.
"""

import logging
import warnings

# Module level logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def configure_logging(level: int = logging.INFO) -> None:
    """Configure basic logging for CudaRL-Arena users."""
    logging.basicConfig(level=level)

# Smart import with CUDA fallback
try:
    # Attempt to import the compiled CUDA extension
    from cudarl_core_python import Environment as _CudaEnvironment
    CUDA_AVAILABLE = True
    logger.info("CUDA environment loaded successfully")
except ImportError as e:
    # Fall back to CPU‐only mock
    from .mock_env import MockEnvironment as _CudaEnvironment
    CUDA_AVAILABLE = False
    logger.warning(f"CUDA environment not available: {e}")
    logger.info("Using CPU-only mock environment as fallback")

# Primary Python API
from .environment import Environment
from .agent import Agent, QTableAgent
from .trainer import Trainer

# Alias for the underlying backend (CUDA or mock)
CudaEnvironment = _CudaEnvironment

__version__ = '0.1.0'
__all__ = [
    'Environment',
    'CudaEnvironment',
    'Agent',
    'QTableAgent',
    'Trainer',
    'configure_logging',
    'CUDA_AVAILABLE',
]

# Warn users if they're stuck on the fallback
if not CUDA_AVAILABLE:
    warnings.warn(
        "CudaRL-Arena is running in CPU fallback mode. "
        "For optimal performance, please build the CUDA extension.",
        UserWarning,
        stacklevel=2,
    )
