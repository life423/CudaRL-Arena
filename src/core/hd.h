#ifndef CUDARL_CORE_HD_H
#define CUDARL_CORE_HD_H

// Marks functions shared by CUDA kernels and plain C++ code. Under nvcc they
// compile for both host and device; under the host compiler the macro is empty.
#ifdef __CUDACC__
#define CUDARL_HD __host__ __device__
#else
#define CUDARL_HD
#endif

// Keep the certified terrain search and contact solver out of each other's
// device instruction bodies. On the CPU these remain inline header functions.
#if defined(__CUDA_ARCH__)
#define CUDARL_DEVICE_NOINLINE __noinline__
#else
#define CUDARL_DEVICE_NOINLINE inline
#endif

#endif // CUDARL_CORE_HD_H
