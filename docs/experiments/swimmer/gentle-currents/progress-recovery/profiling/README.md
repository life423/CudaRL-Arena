# Pond PPO bottleneck profile — October 27, 2026

The dominant measured bottleneck is the articulated physics kernel's thread-private working set and memory latency. CPU submission and host/device transfers are not the leading limit in this capture. No performance optimization, physics change, PPO change or model deployment was made in this profiling step.

## What was measured

- RTX 5070, 48 SMs, compute capability 12.0; Release build, CUDA 13.2.
- Nsight Systems 2026.5.1: three updates of a disposable copy of the real seed-71/update-128 progress-selector checkpoint, 8,192 swimmers, after two warmup updates. Startup, evaluation and checkpoint/report writing are outside the capture.
- Nsight Compute 2026.1.1: one representative physics launch at each of 8,192 and 32,768 swimmers, after skipping 20 measured launches, with the full counter set (40 replay passes). The 8,192 run restores the same checkpoint; the 32,768 probe initializes a separate transferred learner because the saved batch is fixed. These are kernel resource probes, not matched policy-quality experiments.
- Native CUDA events and host timestamps separate rollout, bootstrap/GAE, PPO optimization and stream waiting. Optional NVTX labels identify phases. Normal training does not enable the instrumentation callbacks.

Unprofiled 8,192-world repeats measured 5.74–6.34M simulation steps/s, or about 0.83–0.91 seconds/update. Rollout took approximately 92–93% of update time. The variation between runs is not an optimization result: clocks, desktop activity and capture conditions were not controlled as a formal before/after benchmark. Prior batch measurements were 6.27M at 8,192 versus 6.38M at 32,768; a larger batch alone showed little gain in that sample.

## Timeline findings

Source: [timeline summary](timeline-summary.json), [kernel summary](stats_cuda_gpu_kern_sum.csv), [CUDA API summary](stats_cuda_api_sum.csv), [memory operations](stats_cuda_gpu_mem_time_sum.csv), [phase projection](stats_nvtx_gpu_proj_sum.csv).

| Measurement | Result | Meaning |
| --- | ---: | --- |
| Captured interval | 2,527.89 ms | Three measured updates |
| GPU active interval union | 2,492.43 ms / 98.60% | Very little time with no captured device work |
| Physics kernel time | 2,230.38 ms / 89.50% of kernel time | First optimization target |
| Physics share of captured wall span | 88.23% | Different denominator from kernel-time share |
| All memory-copy/memset time | 0.315 ms / 0.0125% of capture | Only device-to-device copies and memset here; no host/device copies in the captured loop |
| Kernel launches | 15,189 | Physics accounts for 1,920 launches |
| Median / 95th-percentile launch API duration | 3.84 / 10.43 microseconds | API duration is not additive CPU overhead |
| Median / 95th-percentile queued kernel delay | 206.91 / 222.06 ms | Substantial device work is already queued |

Kernel time attributed by CPU submission phase: rollout 2,310.97 ms, bootstrap/GAE 1.83 ms, PPO optimization 179.31 ms. GPU spans projected onto NVTX ranges are slightly larger because they include gaps.

The GPU is not mainly waiting for the CPU to submit this workload. Large accumulated CUDA API durations can include launch-queue backpressure while the GPU is executing; they must not be added to GPU time as independent overhead. The 1.4% idle fraction does not establish a universal bound on all possible launch optimization benefits, but CPU starvation is not the dominant observed mechanism. This capture did not measure full application startup, evaluation, checkpoint I/O or Godot rendering overhead.

## Why a busy GPU can still be inefficient

Source: [8,192-world counter details](ncu-details.txt), [32,768-world counter details](ncu-32768-details.txt).

| Physics-kernel counter | 8,192 swimmers | 32,768 swimmers |
| --- | ---: | ---: |
| Compute throughput (Nsight metric) | 9.71% | 9.60% |
| L2 cache throughput | 77.11% | 77.06% |
| L1/TEX cache throughput | 67.60% | 67.32% |
| DRAM throughput | 23.90% | 34.50% |
| Registers per thread | 255 | 255 |
| Stack size per thread | 3,536 bytes | 3,536 bytes |
| Theoretical occupancy | 16.67% | 16.67% |
| Achieved occupancy | 11.03% | 15.47% |
| L1TEX-requested sectors from local memory | 98.09% | 98.09% |

GPU active time means work is in flight; compute throughput measures a different aspect of hardware use. Occupancy measures resident warps relative to hardware capacity, not elapsed GPU activity or a direct percentage of useful work.

The physics uses one swimmer per thread and large, dynamically indexed private arrays: mass/resistance matrices, geometry and solver scratch in `src/envs/swimmer/deep_physics.h`. The kernel consumes 255 registers/thread, limiting theoretical residency to eight warps/SM. Its remaining private data uses CUDA local memory, which is backed by device memory and caches. At 8,192 worlds, about 54.5% of warp issue-cycle stalls are attributed to L1TEX dependencies, consistent with waiting for data. This supports investigating the working set and access pattern before adding CPU workers or increasing batch size again.

Local memory is not synonymous with register spills. Nsight attributes about 19.07% of local load instructions and 2.31% of local stores at 8,192 worlds to spilling; large or dynamically indexed private arrays account for other local accesses. Raw VRAM bandwidth is not saturated. The evidence points more specifically to cache/local-memory traffic, latency and limited residency than to a blanket claim that all GPU memory bandwidth is exhausted.

Counter replay is intrusive, changes cache/clock conditions and reruns a launch many times. Its kernel durations and application timing CSVs are not throughput benchmarks. Nsight's estimated-speedup advice is heuristic, not a measured result or a promise.

## Recommended next experiment

Reduce the physics kernel's per-thread working set while retaining a reference implementation: test packed symmetric matrix storage, reuse solver scratch and/or make hot array accesses easier for the compiler to keep in registers. Change one thing, check numerical/physical agreement and long-rollout behavior, then rerun throughput and counters. Floating-point reordering may change trajectories, so a short timing improvement alone is insufficient validation.

Do not start by forcing a register cap: that can increase spilling. A cooperative/shared-memory solver is another option if the smaller change fails. CPU checkpoint screening and asynchronous logging may help the wider workflow, but do not address this measured hot-loop bottleneck. CUDA graphs can be revisited if launch gaps become significant after physics optimization.

For scale only: if physics were made twice as fast and everything else stayed fixed, an 88.2% wall-time share implies roughly 1.79x total throughput by Amdahl's law. That is a conditional calculation, not an achieved or predicted speedup. Throughput also does not establish better learning per sample or more robust behavior.

## Reproduction and preservation

The new `pond_progress_profile` target accepts read-only model/full-state inputs and writes only timing CSVs. It verifies input bytes and the frozen motor after its disposable updates. NVTX headers are optional and affect only this target. `tools/analyze_pond_profile.cjs` reads an exported Nsight SQLite trace using Node's built-in SQLite API; Node is not a training/runtime dependency.

Example (PowerShell from the repository):

```powershell
cmake --preset windows "-DCUDARL_NVTX_INCLUDE_DIR=C:/Program Files/NVIDIA Corporation/Nsight Compute 2026.1.1/host/target-windows-x64/nvtx/include"
cmake --build --preset release --target pond_progress_profile --parallel 6
build/Release/pond_progress_profile.exe --baseline docs/experiments/swimmer/gentle-currents/frozen-policy.bin --checkpoint build/pond-progress-study/seed-71/training.bin --output build/pond-profile/timings.csv --warmup 2 --updates 3
```

Use Nsight Systems with `--trace=cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop --sample=none --cpuctxsw=none` to bracket measured updates. Export the report to SQLite, then run:

```powershell
node tools/analyze_pond_profile.cjs build/pond-profile/training-8192.sqlite build/pond-profile/timeline-summary.json
```

Full binary captures remain local and ignored under `build/pond-profile/`: `training-8192.nsys-rep`, `physics-full.ncu-rep`, `physics-32768.ncu-rep`. Nsight Systems was downloaded from NVIDIA, signature-verified and extracted as tools; it was not installed as a system application. The initial non-administrator counter request was denied. After the user's authorization, a temporary elevated Nsight Compute process succeeded. No persistent Windows/NVIDIA security or registry setting was changed.

All seven protected policy/full-save files compare unchanged: [hash verification](preserved-inputs.json). This includes both original Godot training saves and both pilot full checkpoints. All **159/159 native tests passed, zero skipped**, including callback-phase ordering and bit-identical instrumented/uninstrumented resume behavior: [regression log](regression.log), [focused checks](tests.log). No profiler or training helper remains running. No UI was changed and no rendered UI verification was needed. Work remains uncommitted; no merge or push was requested for this step.
