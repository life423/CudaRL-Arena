# Quick benchmark of the frozen v2 physics kernel

This host-only C++ tool launches the **actual extracted CUDA step kernel** from
the completed v2 trainer. It does not compile substitute physics, run inference,
train, reset worlds or change a production launch setting. Each run advances the
same 512 saved worlds once using their actual first latent commands, then repeats
those identical inputs with 32, 8 and 1 threads per block. Two recorded stages mean
3,072 attempted world controls total. Evidence includes every post-world, task,
filter/servo state, output, counter and meaningful transition-trace byte.

The public 1.76 MB fixture contains only the initial states, first commands and
recorded first-control outputs from batch 1 and batch 32. It has no neural weights,
optimizer state or complete rollout. Batches were parsed through their native
checksummed format; source and input hashes are in the accompanying bindings.

## Measured result

On the 48-SM RTX 5070 the complete cold native command took **0.867 seconds**,
including process startup, CUDA context/module load, transfers, equality checks
and six binary evidence writes. CUDA context/module/allocation took 129.5 ms;
fixture read/module hash 16.4 ms and evidence writes 28.9 ms.

| Saved stage |32 threads, 16 blocks |8 threads, 64 blocks |1 thread, 512 blocks |
| --- | ---: | ---: | ---: |
| Batch 1 |11.589 ms |2.206 ms |12.466 ms |
| Batch 32 |205.384 ms |182.157 ms |227.542 ms |

All layouts passed exact state/output comparison against 32 threads and exact
comparison to recorded native reward, terminal status, counters and trace values.
Only the 8 uninitialized padding bytes in each 104-byte transition trace are
excluded. Body, filter, servo and world/task state bytes have no exception.

Eight threads reduced the late-stage kernel time 11.3% (1.13×). The early 5.25×
ratio is especially sensitive to first-launch ordering: 32 threads always ran first,
then 8, then 1, with no extra integration warmup or statistical repetitions. These
are single controls; this is not a measured whole-training speedup. The original
update 32 recorded 12,744 ms physics and 15.450 ms optimization; update 1 recorded
144.367 ms physics and 17.447 ms optimization. Optimizer math was already a very
small part of the late update. Late 8-thread physics still costs 82.6× early 8-thread
physics on these saved inputs.

## Dominant late-stage component

A separate bounded, read-only probe evaluated these same 512 initial bodies per
stage. It called the existing whole-body clearance query and the actual first
5 ms water-force/contact-solver algebra, including the original target-filter map
on local copies. It did **not** integrate a body or change saved filter state.

| Late batch 32 component |CUDA event time |Work |
| --- | ---: | --- |
| Certified whole-body terrain query |16.276 ms |63,056 search visits |
| Water-force assembly and contact solve |0.536 ms |513 system evaluations |

The query costs about 30 times as much as the initial force/solve algebra on these
late inputs. All 512 early worlds used four visits each (2,048 total). Late inputs
had 395 such worlds and 117 requiring more search, with at most 642 visits in one
world. CPU and GPU query visit counts matched exactly; all 1,024 certificates
overlapped and the maximum upper-bound difference was 6.8 nm. All 1,024 independent
force/contact evaluations solved successfully. The initial solver count rose
only from 512 to 513 across the two stages.

The runtime calls the body query 11 times per control: a fresh PRE plus one POST
for each of the 10 substeps. Eleven times the late initial query span is about
179 ms, near the original 182 ms late control. This supports prioritizing the
certified terrain search. It is **not an additive timing decomposition**: the
component kernels were compiled separately, scheduling differs, and the body
changes during a real control. Early component event spans (516.5 ms query and
6.0 ms algebra) include first-launch/lazy-loading effects and are not steady-state
cost estimates. The complete component command took 0.738 seconds, used 1,024 CPU
queries plus 1,024 CUDA queries and 1,024 CUDA force/contact evaluations, and ran
zero physical controls, substeps or optimizer updates.

Hardware-counter attribution was unavailable because Nsight returned
`ERR_NVGPUCTRPERM`; machine permissions were left unchanged. Component receipts
and all per-world visit counts are published with the fixture. No extra profiling
or integration repetitions were performed.

## Build and run

Windows requires MSVC, CMake and the CUDA Driver API development headers/library.
The measured module came from CUDA 13.2.78/MSVC 19.44 with the existing isolated
`--fmad=false` physics profile. No NVCC build is needed for this host tool.

```powershell
cmake -S tools/pond-quick-bench -B build/quick32/native -A x64 -DFROZEN_ROOT=C:/path/to/exact-v2-source
cmake --build build/quick32/native --config Release
# One ordinary command after the host tool and exact module are available:
./tools/pond-quick-bench/run.ps1 -Module C:/path/to/pond_ppo.1.sm_120.cubin
```

`FROZEN_ROOT` defaults to the local preserved
`build/turn32/source-snapshot-v2`. A source clone can obtain the v2 headers and
trainer from commit `e013b2a`; use `git archive` or a separate checkout. The large
rollouts are unnecessary. The exact module is **not committed**. Obtain it from
the preserved/rebuilt v2 trainer using `cuobjdump --extract-elf all pond_ppo.exe`;
its SHA256 must match `assets/research-evidence/pond-quick-bench-v1/bindings.json`.
The wrapper fails clearly on another module. Rebuilding with different compiler
versions or flags may produce a different module and is not automatically this
frozen-kernel experiment. This is a hardware diagnostic, not a hosted-CI test.

Each explicit invocation creates fresh evidence under `build/`, with the same
fixed 3,072-control limit. Failed comparisons stop remaining launches. Raw native
results are preserved locally; only compact source/metric bindings ship in Git.

The separate pure component probe can be built and explicitly run without the
extracted module (NVCC is required):

```powershell
cmake -S tools/pond-quick-bench/component -B build/quick32/component-native -A x64 -DFROZEN_ROOT=C:/path/to/exact-v2-source
cmake --build build/quick32/component-native --config Release
./build/quick32/component-native/Release/pond_component_bench.exe assets/research-evidence/pond-quick-bench-v1/fixtures.bin build/fresh-component-result
```
