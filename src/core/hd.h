#ifndef CUDARL_CORE_HD_H
#define CUDARL_CORE_HD_H

// Marks functions shared by CUDA kernels and plain C++ code. Under nvcc they
// compile for both host and device; under the host compiler the macro is empty.
#ifdef __CUDACC__
#define CUDARL_HD __host__ __device__
#else
#define CUDARL_HD
#endif

#endif // CUDARL_CORE_HD_H
