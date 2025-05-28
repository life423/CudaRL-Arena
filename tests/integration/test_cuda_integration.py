"""
Comprehensive CUDA integration tests for CudaRL-Arena.
"""

import pytest
import numpy as np
import subprocess
from pathlib import Path


@pytest.mark.gpu
class TestCudaIntegration:
    """Test CUDA functionality and integration."""
    
    def test_cuda_device_detection(self, cuda_available, gpu_info):
        """Test CUDA device detection and properties."""
        if not cuda_available:
            pytest.skip("CUDA not available")
        
        assert gpu_info["cuda_available"] is True
        assert gpu_info["device_count"] > 0
        assert len(gpu_info["devices"]) > 0
        
        # Check first device properties
        device = gpu_info["devices"][0]
        assert "name" in device
        assert "compute_capability" in device
        assert "memory" in device
        assert device["memory"] > 0
    
    @pytest.mark.rtx5070
    def test_rtx5070_detection(self, gpu_info):
        """Test RTX 5070 specific detection."""
        if not gpu_info["cuda_available"]:
            pytest.skip("CUDA not available")
        
        rtx5070_found = any(
            "RTX 5070" in device["name"] 
            for device in gpu_info["devices"]
        )
        
        if rtx5070_found:
            rtx5070 = next(
                device for device in gpu_info["devices"] 
                if "RTX 5070" in device["name"]
            )
            
            # RTX 5070 should have compute capability 8.9 or higher
            major, minor = map(int, rtx5070["compute_capability"].split('.'))
            assert major >= 8
            if major == 8:
                assert minor >= 9
            
            # RTX 5070 should have at least 8GB memory
            assert rtx5070["memory"] >= 8000
    
    def test_cuda_executable(self):
        """Test CUDA test executable runs successfully."""
        test_exe = Path(__file__).parent / "bin" / "Release" / "cuda_test.exe"
        
        if not test_exe.exists():
            pytest.skip("CUDA test executable not found")
        
        result = subprocess.run(
            [str(test_exe)], 
            capture_output=True, 
            text=True,
            timeout=30
        )
        
        assert result.returncode == 0
        assert "CUDA" in result.stdout
        assert "PASSED" in result.stdout
    
    @pytest.mark.slow
    def test_memory_allocation_stress(self, cuda_available, gpu_memory_tracker):
        """Test GPU memory allocation under stress."""
        if not cuda_available:
            pytest.skip("CUDA not available")
        
        try:
            import cupy
        except ImportError:
            pytest.skip("CuPy not available")
        
        # Allocate and free memory multiple times
        allocations = []
        
        for size_mb in [10, 50, 100, 200]:
            size_bytes = size_mb * 1024 * 1024
            size_elements = size_bytes // 4  # float32
            
            # Allocate
            arr = cupy.zeros(size_elements, dtype=cupy.float32)
            allocations.append(arr)
            
            # Verify allocation
            assert arr.size == size_elements
            assert arr.dtype == cupy.float32
        
        # Clean up
        for arr in allocations:
            del arr
        
        cupy.get_default_memory_pool().free_all_blocks()
        
        # Memory should be cleaned up
        used_after = cupy.get_default_memory_pool().used_bytes()
        assert used_after < 1024 * 1024  # Less than 1MB residual
    
    def test_cuda_kernel_execution(self, cuda_available):
        """Test basic CUDA kernel execution."""
        if not cuda_available:
            pytest.skip("CUDA not available")
        
        try:
            import cupy
        except ImportError:
            pytest.skip("CuPy not available")
        
        # Simple vector addition kernel
        size = 1000
        a = cupy.random.random(size, dtype=cupy.float32)
        b = cupy.random.random(size, dtype=cupy.float32)
        
        # GPU computation
        c_gpu = a + b
        
        # CPU verification
        a_cpu = cupy.asnumpy(a)
        b_cpu = cupy.asnumpy(b)
        c_cpu = a_cpu + b_cpu
        c_gpu_cpu = cupy.asnumpy(c_gpu)
        
        # Verify results match
        np.testing.assert_allclose(c_gpu_cpu, c_cpu, rtol=1e-6)
    
    @pytest.mark.parametrize("size", [100, 1000, 10000])
    def test_different_problem_sizes(self, cuda_available, size):
        """Test CUDA operations with different problem sizes."""
        if not cuda_available:
            pytest.skip("CUDA not available")
        
        try:
            import cupy
        except ImportError:
            pytest.skip("CuPy not available")
        
        # Create test data
        data = cupy.random.random(size, dtype=cupy.float32)
        
        # Perform operations
        result = cupy.sqrt(data * 2.0 + 1.0)
        
        # Verify reasonable results
        assert result.size == size
        assert cupy.all(result > 0)
        assert cupy.all(result < 10)  # Reasonable upper bound
    
    def test_multiple_gpu_streams(self, cuda_available):
        """Test multiple CUDA streams if supported."""
        if not cuda_available:
            pytest.skip("CUDA not available")
        
        try:
            import cupy
        except ImportError:
            pytest.skip("CuPy not available")
        
        # Create multiple streams
        streams = [cupy.cuda.Stream() for _ in range(4)]
        
        results = []
        size = 1000
        
        for i, stream in enumerate(streams):
            with stream:
                # Different computation in each stream
                data = cupy.random.random(size, dtype=cupy.float32)
                result = data * (i + 1)
                results.append(result)
        
        # Synchronize all streams
        for stream in streams:
            stream.synchronize()
        
        # Verify results
        for i, result in enumerate(results):
            assert result.size == size
            # Results should be scaled differently
            mean_val = float(cupy.mean(result))
            expected_range = (i + 1) * 0.3, (i + 1) * 0.7  # Rough expected range
            assert expected_range[0] <= mean_val <= expected_range[1]


@pytest.mark.integration
class TestPythonCudaIntegration:
    """Test Python-CUDA integration specific functionality."""
    
    def test_numpy_cupy_interop(self, cuda_available):
        """Test NumPy-CuPy interoperability."""
        if not cuda_available:
            pytest.skip("CUDA not available")
        
        try:
            import cupy
        except ImportError:
            pytest.skip("CuPy not available")
        
        # NumPy -> CuPy
        np_array = np.random.random((100, 100)).astype(np.float32)
        cp_array = cupy.asarray(np_array)
        
        # CuPy -> NumPy
        result_np = cupy.asnumpy(cp_array)
        
        # Verify data integrity
        np.testing.assert_array_equal(np_array, result_np)
    
    def test_memory_transfer_performance(self, cuda_available, performance_timer):
        """Test CPU-GPU memory transfer performance."""
        if not cuda_available:
            pytest.skip("CUDA not available")
        
        try:
            import cupy
        except ImportError:
            pytest.skip("CuPy not available")
        
        sizes = [1024, 10240, 102400]  # Different data sizes
        
        for size in sizes:
            data = np.random.random(size).astype(np.float32)
            
            # Measure H2D transfer
            performance_timer.start(f"h2d_{size}")
            gpu_data = cupy.asarray(data)
            cupy.cuda.runtime.deviceSynchronize()
            h2d_time = performance_timer.end(f"h2d_{size}")
            
            # Measure D2H transfer
            performance_timer.start(f"d2h_{size}")
            result = cupy.asnumpy(gpu_data)
            d2h_time = performance_timer.end(f"d2h_{size}")
            
            # Performance should be reasonable (not too slow)
            data_mb = size * 4 / (1024 * 1024)  # MB
            h2d_bandwidth = data_mb / h2d_time if h2d_time > 0 else float('inf')
            d2h_bandwidth = data_mb / d2h_time if d2h_time > 0 else float('inf')
            
            # Expect at least 1 GB/s bandwidth (very conservative)
            assert h2d_bandwidth > 1000 or h2d_time < 0.001
            assert d2h_bandwidth > 1000 or d2h_time < 0.001
            
            # Verify data integrity
            np.testing.assert_array_equal(data, result)
