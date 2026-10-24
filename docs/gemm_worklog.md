# Tensor-core GEMM worklog

From a WMMA baseline to **91–96% of cuBLAS** on an RTX 5070. Each version is hand-written CUDA and PTX, and each still exists in the code (`src/nn/gemm_bf16.cu`). Every number here can be reproduced with `nn_bench tc`, `nn_bench tune`, and `nn_bench profile` under Nsight Compute.

**Operation:** `C[m×n] = A[m×k] · B[n×k]ᵀ`, with bf16 inputs, fp32 accumulation, and fp32 output. Both inputs are K-contiguous, which is the layout of a linear layer's forward pass, `Y = X·Wᵀ`.

**Hardware:** RTX 5070 (consumer Blackwell, `sm_120`, 48 SMs), driver 620.02, CUDA 13.2. `sm_120` has no Hopper `wgmma` and no datacenter-Blackwell `tcgen05`, so the tensor cores are driven with warp-level `mma.sync.m16n8k16`.

## Results

4096³, best of 5 timing trials (`nn_bench tc`):

| Version | What changed | TFLOP/s | % of cuBLAS |
|---|---|---|---|
| cuBLAS (`cublasGemmEx`, bf16 → fp32) | reference | 69.6 | 100% |
| v0 | WMMA API, 64×64 tiles through shared memory | 23.6 | 34% |
| v1 | `mma.sync` + `ldmatrix`, 128×128×32 tiles, 2×2 warps of 64×64, synchronous loads | 42.4 | 61% |
| v2 | + 3-stage `cp.async` pipeline | 52.5 | 75% |
| v3 | + XOR-swizzled shared memory | 58.2 | 84% |
| **v4** | **+ 2 stages instead of 3, grouped block order (from the tuning sweep)** | **63.4** | **91%** |

At 8192³, v4 reaches 56.8 TFLOP/s, **96% of cuBLAS**. The tensor-core ceiling, measured with a register-only `mma.sync` loop, is 70.0 TFLOP/s, and cuBLAS gets to 99% of it at 4096³.

Nsight Compute at 4096³ (`ncu`, one launch per version; ncu locks clocks, so absolute times run slower than the timings above):

| | v0 | v1 | v2 | v3 | v4 |
|---|---|---|---|---|---|
| Tensor pipe active | 38% | 63% | 77% | 85% | **94%** |
| Shared-memory load bank conflicts | 0.14 M | 100.9 M | 101.0 M | 0.17 M | 0.23 M |
| Shared-memory load wavefronts | 68 M | 134 M | 135 M | 33.7 M | 33.8 M |
| Blocks per SM (limited by shared memory) | 4 | 5 | 2 | 2 | 3 |
| Registers per thread | 107 | 165 | 164 | 165 | 166 |

## The steps

### v0: WMMA
The WMMA API is the easy route to the tensor cores: load 16×16 fragments, call `mma_sync`, store. With 64×64 tiles it hits **23.6 TFLOP/s** and stays there as sizes grow. It even drops to 12 TFLOP/s at 8192³, most likely because each block's small tile re-reads A and B from DRAM far more often once the matrices no longer fit in L2. The tensor pipe is busy only 38% of the time, and the fragment layout is opaque, so there's not much left to tune.

### v1: `mma.sync` + `ldmatrix`
Going down to PTX gives control over every register:
- Each warp computes a 64×64 tile as 4×8 `m16n8k16` mma tiles, which is 128 fp32 accumulators per thread.
- `ldmatrix.x4` loads four 8×8 bf16 matrices from shared memory in one instruction, already arranged in the mma fragment layout. Lanes 0–15 address A's 16 rows; for B, which is stored `[n][k]`, one x4 covers two n8 tiles.
- A 128×128 block tile keeps 4 warps reusing each A and B element 128 times out of shared memory.

That's **42.4 TFLOP/s (1.8× v0)**. But the profile shows 101 M bank conflicts and a tensor pipe idle a third of the time: every warp waits on synchronous global loads, then stalls on conflicted shared-memory reads.

### v2: `cp.async` pipeline
`cp.async.cg` copies 16-byte chunks from global to shared memory without going through registers. With 3 stages, the copy of slice k+2 overlaps the math on slice k. Each iteration waits for its own slice (`cp.async.wait_group 1`), refills the stage the previous iteration just finished with, then computes. Out-of-bounds chunks are zero-filled through `cp.async`'s source-size operand.

That's **52.5 TFLOP/s**, with the tensor pipe active 77% of the time. The bank conflicts are unchanged (101 M), which isolates the next problem.

### v3: XOR swizzle
Each tile row is 32 bf16, or 64 bytes, so `ldmatrix` reads 8 rows × 16 bytes from the same column chunk. With 64-byte rows, those 8 addresses fall into only 2 of the 8 sixteen-byte bank groups: a **4-way conflict**. Storing chunk `c` of row `r` at chunk `c ^ ((r >> 1) & 3)` spreads them over all 8. The same function is used when writing (`cp.async`) and when reading (`ldmatrix`), so no data moves; only addresses change.

The profiler confirms it: bank conflicts fall from 101 M to 0.17 M, and shared-memory load wavefronts fall **exactly 4×**, from 135 M to 33.7 M, which is the predicted conflict degree. That's **58.2 TFLOP/s**, with the tensor pipe active 85% of the time.

### v4: what the tuning sweep found
My first guess for v4, a deeper pipeline (4 stages), was **slower than v3: 59% of cuBLAS vs. 84%**. More stages need more shared memory per block, and on this GPU the shared-memory budget per SM decides how many blocks can run at once. Instead of guessing again, I swept 14 configurations (`nn_bench tune`, all kept in `kTuningCandidates`). Some of the results, at 2048³ / 4096³ / 8192³ as % of cuBLAS:

| Configuration | Shared memory / block | 2048³ | 4096³ | 8192³ |
|---|---|---|---|---|
| 128×128×32, 4 warps, 3 stages (= v3) | 48 KB | 70% | 79% | 74% |
| … + grouped block order | 48 KB | 69% | 77% | 84% |
| 128×128×32, 4 warps, **2 stages** | 32 KB | 80% | 90% | 81% |
| … + grouped block order (**= v4**) | 32 KB | 76% | 86% | **92%** |
| 128×128×32, 4 warps, 4 stages, grouped | 64 KB | 52% | 60% | 62% |
| 128×128×32, 8 warps of 64×32, 2 stages, grouped | 32 KB | 74% | 88% | 89% |
| 128×256×32, 8 warps, 2–3 stages, grouped | 48–72 KB | 63–64% | 67–72% | 74% |
| 128×128×64, 4 warps, 2 stages, grouped | 64 KB | 65% | 70% | 79% |

Two effects stand out:
1. **Occupancy beats pipeline depth.** Double buffering (32 KB) lets 3 blocks share an SM, versus 2 at 3 stages or 1 at 4 stages. The other resident blocks hide memory latency better than a deeper pipeline inside one block does. The profile shows the switch from 2 to 3 blocks per SM, and the tensor pipe reaching **94% active**.
2. **Block order matters once the problem outgrows L2.** Launching blocks in groups of 8 block rows makes consecutive blocks reuse the same B tiles in L2. It's neutral at 2048³, where everything fits, and worth +11 points at 8192³.

Bigger tiles (128×256, 256×128) and BK = 64 all lost, for the same occupancy reason.

## Correctness
A bf16 × bf16 product is **exact in fp32**: two 8-bit significands multiply into at most 16 bits, which fits fp32's 24. So against an fp64 reference computed on the same bf16-rounded inputs, the only error left is the order of the fp32 additions. That allows a tight test tolerance of **1e-5** (normalized max error), which catches any fragment-layout or indexing mistake instead of blaming it on bf16 precision. Every version passes on shapes chosen to break tiling assumptions (1×1×8, 7×13×8, 129×130×40, 300×257×264, and others), with padded leading dimensions, and bit-identical across runs. The output buffer is filled with NaN first, so a missed element can't pass by accident. Inputs are rounded with our own round-to-nearest-even conversion, which matches on host and device bit for bit.

The bf16 trade-off itself, measured: rounding fp32 inputs to bf16 costs **2.6e-3** relative error in the result. That's about 250× more than fp32 accumulation adds, which is why training in bf16 keeps fp32 accumulators and fp32 master weights.

## What's left on the table
v4 is at 91–96% of cuBLAS on large squares, but only 65% at 1024³, where 64 blocks can't fill 48 SMs × 3 slots. The next steps, in the order I'd try them:
- **Split-K or stream-K for small and skinny shapes.** The 1024³ case, and the MLP-shaped 65536×64×64 at 91%, underfill the GPU.
- **Double-buffer the fragments in registers.** Load the `ldmatrix` fragments for k-step `ks + 1` while the mma for `ks` runs.
- **Stage the epilogue through shared memory** for fully coalesced 128-bit stores, and fuse bias and activation there, as the fp32 kernel already does.
- **Lower-precision formats** that `sm_120`'s `mma.sync` supports (fp8, and fp4/fp6 block-scaled).

## Reproduce
```bash
cmake --preset windows && cmake --build --preset release
build/Release/cudarl_tests.exe gemm_bf16      # correctness, all versions
build/Release/nn_bench.exe tc                 # the main table
build/Release/nn_bench.exe tune               # the tuning sweep (slow)
ncu --kernel-name regex:_gemm_kernel --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum build/Release/nn_bench.exe profile
```
Nsight Compute needs GPU performance-counter access: run it as administrator, or allow access for all users in the NVIDIA Control Panel's developer settings. Timings vary a few percent between runs with GPU boost clocks and temperature; `nn_bench` reports the best of 5 trials for that reason.
