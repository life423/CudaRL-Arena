# Racing optimization phase: CUDA Graphs

This follows the [traffic execution study](racing_performance.md), using its
distributed traffic path in both arms. All training mathematics, kernels,
memory management and benchmark code remain C++/CUDA, without Python or
PyTorch. The work we control here is GPU work submission through the CUDA
runtime; the actor, critic, gradients and Adam state already live on the GPU.

## Diagnosis

At 2,048 agents, separate diagnostic event spans put optimization near 24 ms:
roughly 2.2 ms gathering/normalizing minibatches, 4.9 ms in forward/loss work,
9.2 ms in actor/exploration backpropagation and Adam, and 7.5 ms in critic
backpropagation and Adam. These stream elapsed spans include submission gaps.
They are not hardware-counter measurements.

CPU wall time inside the optimization submission region was about 35-37 ms.
That interval can include driver waiting/backpressure while previously queued
GPU work runs. It is not 35-37 ms of CPU computation, and it must not be added
to GPU timing spans. The potential improvement was submitting a repeated GPU
workflow more cheaply, rather than moving matrix multiplication off the CPU:
the matrix multiplication was already on the GPU.

## Implementation and correctness

The default optimizer uses a CUDA graph with 1,888 operation nodes for the
normal configuration. It captures the same ordered minibatch workflow, reuses
the instantiated executable with `cudaGraphExecUpdate`, and launches it once
per training update. The rollout remains outside the graph. `graphs=0` retains
ordinary stream submission, independently of the traffic execution flag.

The workflow is re-captured before each update's execution. This deliberately
reruns the host control calculations so each graph gets fresh permutations,
learning rates and Adam bias corrections. The original Adam host counters
advance exactly once for the prospective optimizer steps; capture itself does
not execute the GPU operations. The original double-precision host calculation
of the bias corrections is preserved.

This trades a small capture/update cost for a smaller implementation and exact
numerical agreement. It does not eliminate the CPU loop. Updating a static graph
from a GPU-resident control block could remove more CPU work, but is a separate
experiment that must preserve shuffle, optimizer and annealing semantics.

The design follows NVIDIA's [graph update documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)
and [discussion of dynamic parameters](https://developer.nvidia.com/blog/constructing-cuda-graphs-with-dynamic-parameters/).
A capture error ends the capture and stops training with an error; it does not
silently retry Adam steps after their host counters have advanced. Graphs and
events are owned by the trainer and released when it stops.

Diagnostic minibatch events use external event-record nodes during capture so
the timestamps actually execute. Profiling adds 160 nodes, for 2,048 total;
it is disabled in throughput comparisons.

## Measurements — October 26, 2026

Windows, RTX 5070 (48 SMs), driver 621.01, CUDA 13.2, Release build. Same seed,
same 64-step rollout, same networks and four-epoch/eight-minibatch PPO workload.
Both paths use distributed traffic. Each trial trains ten warmup updates and
then measures forty; five pairs alternate order. The table reports the median
of the five trial means. All per-update graph setup and launch costs are inside
the wall-clock measurements. The annealing horizon remains 1,200 updates.

| Agents | Stream ms/update | Graph ms/update | Stream M agent-steps/s | Graph M agent-steps/s | Throughput gain |
|---:|---:|---:|---:|---:|---:|
| 256 | 32.17 | 23.27 | 0.509 | 0.704 | 38.3% |
| 2,048 | 46.49 | 40.71 | 2.820 | 3.220 | 14.2% |
| 8,192 | 107.68 | 103.23 | 4.869 | 5.079 | 4.3% |

At the default batch, latency fell 12.4% and throughput rose 14.2%. These are
equivalent descriptions of one improvement. Graphs were faster in every pair
at all three batch sizes. The smaller improvement at 8,192 agents is consistent
with larger kernels making submission overhead less significant, but no hardware
counters were available to establish a detailed utilization explanation.

This is an incremental comparison against the first performance pass. Do not
add its percentages to the earlier 12.7% traffic result or call their product
a contemporaneously measured combined speedup.

In diagnostic runs, optimization fell from about 23.9 to 18.0 ms and rollout
stayed around 23.1 ms. Submission-region CPU wall time fell from roughly 35-37
to 12-13 ms, including any waiting. Warm graph capture/update took roughly
1.4-1.5 ms per update; that cost overlaps queued GPU work and cannot simply be
subtracted from total update latency. CPU and GPU intervals are not additive.

Three separate new-trainer/zero-warmup samples measured first graph setup at
8.16, 3.18 and 3.35 ms, with total first updates at 49.33, 40.36 and 39.00 ms.
These include new graph instantiation and cache/lazy-load effects. CUDA context
and trainer construction occur outside that timer, so this is not full program
startup latency. Warm throughput does not hide these separately reported costs.

## Validation

All **112 local CPU/GPU tests passed, zero skipped**, with
`CUDARL_REQUIRE_GPU=1`. A graph-specific regression compares exact actor,
critic and exploration parameters plus learning statistics after each update
for three seeds and two batch sizes. It exercises changing permutations and
Adam steps, profiling on the graph path, clipping enabled/disabled, and annealing
through zero learning rate. Both the original isolated-track learning tests
and the 1,200-update eight-car learning regression pass.

The three paired 200-update/60-second showcase evaluations also matched:

| Training seed | Clean laps, both paths | Contacts, both paths | Wall hits, both paths | Best lap, both paths |
|---:|---:|---:|---:|---:|
| 1 | 16 | 0 | 0 | 28.20 s |
| 2 | 14 | 0 | 2 | 28.10 s |
| 3 | 15 | 0 | 1 | 28.05 s |

This establishes agreement for the measured cases, not indefinitely clean
driving or a universal speedup. No hardware-counter occupancy/utilization claims
or comparisons against PyTorch are made. Hosted CI was not run for this local
milestone.

A headless Godot integration smoke test also completed twelve background
training updates through `RacingLab`, then read eight showcase agents and
stopped the worker without errors. This verifies graph capture on the actual
training thread through the GDExtension; it is not a rendered UI inspection.

## Reproduce and evidence

```powershell
cmake --build --preset release
build/Release/racing_bench.exe envs=2048 warmup=10 updates=40 repeats=5 distributed=1 compare_graphs=1
build/Release/racing_bench.exe envs=2048 warmup=10 updates=30 repeats=2 distributed=1 compare_graphs=1 profile=1
build/Release/racing_bench.exe envs=2048 warmup=0 updates=1 repeats=3 graphs=1
build/Release/racing_bench.exe envs=2048 seed=1 warmup=10 updates=190 repeats=1 compare_graphs=1 evaluate=1
$env:CUDARL_REQUIRE_GPU='1'
build/Release/cudarl_tests.exe
& 'C:/Users/aiand/Godot/Godot.exe' --headless --path godot-project --script "$PWD/tests/godot/racing_graph_smoke.gd"
```

Repeat the first command with `envs=256` and `envs=8192`, and evaluation with
`seed=2` and `seed=3`. Close other trainers/the rendered race before timing.
`compare` compares traffic execution; `compare_graphs` compares optimization
submission. The two comparison switches cannot be combined. Each output row's
`graphs` column identifies its actual mode; the comment header shows starting
configuration. `graph_setup_ms` includes capture and instantiate/update, while
total timing includes launch/execution too. Zero profiling fields mean profiling
was disabled; evaluation counts of -1 mean evaluation was not requested.

Raw evidence: [256 agents](benchmarks/2026-10-26-optimizer/training-256.csv),
[2,048 agents](benchmarks/2026-10-26-optimizer/training-2048.csv),
[8,192 agents](benchmarks/2026-10-26-optimizer/training-8192.csv),
[phase profile](benchmarks/2026-10-26-optimizer/phases-2048.csv),
[cold graph setup](benchmarks/2026-10-26-optimizer/cold-graph.csv),
[seed 1](benchmarks/2026-10-26-optimizer/learning-seed-1.csv),
[seed 2](benchmarks/2026-10-26-optimizer/learning-seed-2.csv),
[seed 3](benchmarks/2026-10-26-optimizer/learning-seed-3.csv), and
[test log](benchmarks/2026-10-26-optimizer/tests.log), and
[Godot worker smoke](benchmarks/2026-10-26-optimizer/godot-worker.txt).
