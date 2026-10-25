#ifndef CUDARL_NN_MMA_BF16_H
#define CUDARL_NN_MMA_BF16_H

#include <cstdint>

#include <cuda_runtime.h>

// PTX wrappers shared by the bf16 tensor-core kernels: ldmatrix, mma.sync
// m16n8k16 with fp32 accumulation, and cp.async. Fragment layouts are noted
// with each wrapper. Device-only; include from .cu files.
namespace cudarl::nn::ptx {

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// Four 8x8 b16 matrices from shared memory. Lanes 8i..8i+7 supply the row
// addresses of matrix i; afterwards each lane holds, in r[i], two consecutive
// elements of row (lane / 4) of matrix i, at columns 2 * (lane % 4) and +1.
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

// As ldmatrix_x4, but each matrix is transposed on the way into the registers:
// lane (g = lane / 4, t = lane % 4) receives the elements at rows 2t and 2t + 1
// of column g. This reads a [k][n] operand, such as V stored by key, as the
// column-major B fragment of mma.
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

// d[16x8] += a[16x16] * b[16x8], bf16 inputs, fp32 accumulate. With g = lane / 4
// and t = lane % 4 (PTX ISA, m16n8k16 fragment layouts):
//   a[0]: A[g][2t..2t+1]     a[1]: A[g+8][2t..2t+1]    a[2]: A[g][2t+8..]   a[3]: A[g+8][2t+8..]
//   b[0]: B[2t..2t+1][g]     b[1]: B[2t+8..2t+9][g]
//   d[0..1]: D[g][2t..2t+1]  d[2..3]: D[g+8][2t..2t+1]
__device__ __forceinline__ void mma_16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// 16-byte global -> shared copy that bypasses the registers; a source size of 0
// zero-fills the destination (used for out-of-bounds chunks).
__device__ __forceinline__ void cp_async_16(uint32_t dst, const void* src, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

} // namespace cudarl::nn::ptx

#endif // CUDARL_NN_MMA_BF16_H
