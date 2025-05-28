"""
CUDA testing helper functions.
"""

import subprocess
from pathlib import Path
from typing import Dict, Any, Optional


def get_cuda_device_info() -> Dict[str, Any]:
    """Get CUDA device information for testing."""
    try:
        import cupy  # noqa: F401
        devices = []
        device_count = cupy.cuda.runtime.getDeviceCount()
        
        for i in range(device_count):
            with cupy.cuda.Device(i):
                props = cupy.cuda.runtime.getDeviceProperties(i)
                device_info = {
                    "id": i,
                    "name": props["name"].decode(),
                    "compute_capability": (f"{props['major']}."
                                           f"{props['minor']}"),
                    "memory_mb": props["totalGlobalMem"] // (1024**2),
                    "multiprocessors": props["multiProcessorCount"],
                    "max_threads_per_block": props["maxThreadsPerBlock"],
                    "max_grid_size": props["maxGridSize"],
                }
                devices.append(device_info)
        
        return {
            "available": True,
            "device_count": device_count,
            "devices": devices
        }
    except ImportError:
        return {"available": False, "device_count": 0, "devices": []}


def check_cuda_executable(exe_name: str = "cuda_test.exe") -> bool:
    """Check if CUDA test executable exists and runs."""
    test_exe = Path(__file__).parent.parent / "bin" / "Release" / exe_name
    
    if not test_exe.exists():
        return False
    
    try:
        result = subprocess.run(
            [str(test_exe)],
            capture_output=True,
            timeout=30,
            text=True
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def get_gpu_memory_info() -> Optional[Dict[str, int]]:
    """Get GPU memory information."""
    try:
        import cupy  # noqa: F401
        if cupy.cuda.runtime.getDeviceCount() > 0:
            pool = cupy.get_default_memory_pool()
            return {
                "used_bytes": pool.used_bytes(),
                "total_bytes": pool.total_bytes(),
                "free_bytes": pool.free_bytes(),
            }
    except ImportError:
        pass
    return None


def cleanup_gpu_memory():
    """Clean up GPU memory after tests."""
    try:
        import cupy  # noqa: F401
        cupy.get_default_memory_pool().free_all_blocks()
        cupy.cuda.runtime.deviceSynchronize()
    except ImportError:
        pass


def require_cuda_capability(min_major: int, min_minor: int = 0) -> bool:
    """Check if CUDA device has minimum compute capability."""
    info = get_cuda_device_info()
    
    if not info["available"]:
        return False
    
    for device in info["devices"]:
        major, minor = map(int, device["compute_capability"].split('.'))
        if major > min_major or (major == min_major and minor >= min_minor):
            return True
    
    return False


def require_gpu_memory(min_memory_mb: int) -> bool:
    """Check if GPU has minimum memory available."""
    info = get_cuda_device_info()
    
    if not info["available"]:
        return False
    
    return any(
        device["memory_mb"] >= min_memory_mb
        for device in info["devices"]
    )
