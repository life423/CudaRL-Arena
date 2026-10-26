# Racing training performance

This portfolio milestone measures and improves the system beneath an ML
framework: memory ownership, CUDA kernels, work distribution, streams, rollout
collection and learning updates are implemented in C++/CUDA. Neither training
nor this benchmark uses Python, PyTorch, cuDNN or cuBLAS. The separate
`nn_bench` tool retains cuBLAS solely as a kernel benchmark reference.

The cars are **agents**, grouped into eight-agent **races** (environments).
The agents share a **policy model**. The benchmark counts individual agent
transitions, not whole races, as agent-steps. Godot's eight demonstration agents
use policy checkpoints; they are separate from the GPU training population.

## Workload and measurement

Default training: 2,048 agents in 256 races, a 64-step rollout, separate
30-64-64-2 actor and 30-64-64-1 critic networks, four PPO epochs and eight
minibatches per epoch. Each update collects 131,072 agent transitions and then
optimizes on that data. Training and numerical rules are unchanged.

`racing_bench` constructs the actual `RacingPpoTrainer`. There is no renderer,
policy checkpoint publication or deliberate display throttle in the measured
loop. Thus this is the trainer's wall-clock throughput, not Godot frame rate,
an isolated kernel throughput, or time to reach a driving-quality threshold.

Each trial starts from the same seed, initializes outside the timing region,
warms up with ten learning updates, and measures the next forty. The default
learning-rate annealing horizon stays at 1,200 updates, matching Godot, even for
a shorter benchmark. Five paired trials alternate reference/optimized order
to reduce clock and thermal bias. Every row reports mean, p10, p50 and p90
update time; initialization and optional policy evaluation are excluded.

All results are from one Windows RTX 5070 with 48 SMs, the Release build,
CUDA 13.2 runtime and driver 621.01. GPU clocks, temperature, Windows scheduling
and other desktop GPU work affect timings. Close the running Godot race and
other trainers before reproducing these measurements. Do not run a profiler
and an uninstrumented throughput trial concurrently.

## Diagnosis and retained optimization

Opt-in CUDA events split the stream timeline into rollout, value bootstrapping
and GAE, metrics, optimization and summary readback. The simulation portion is
further divided into advance, traffic resolution and observations. These are
stream elapsed spans, including submission gaps and profiling overhead; they
are not hardware-counter measurements or pure kernel execution times. Ordinary
throughput runs disable the diagnostic events entirely.

The initial profile put roughly 29 ms in rollout collection and 23 ms in
optimization at the default batch. Traffic resolution alone accounted for
about 12.5 ms per update. Its original one-thread-per-race kernel placed all
256 races in a single 256-thread block. It combined collision resolution,
expensive final sensor observations and ordered respawns inside that thread.

The retained path separates independent work into stream-ordered kernels:

1. Integrate each agent, preserving its previous state.
2. Determine each agent's contacts and outcomes from immutable advanced states.
3. Collect required final observations before any respawn changes those states.
4. Respawn ended agents in the original per-race order, using the same shared
   CPU/CUDA helper and separation rules.
5. Compute the next observations with the original sensor implementation.

Car tasks use 64-thread blocks; ordered per-race respawns use 32-thread blocks.
The original 256-thread execution remains selectable as `distributed=0`.
There are more kernel launches, but less independent work is serialized inside
the race-resolution kernel. This is an example of measuring total cost rather
than optimizing kernel count in isolation.

During optimized training, final observations for true terminals are omitted:
GAE bootstraps a crash/contact terminal from zero, so those sensor readings
were discarded. Time-limit final observations remain mandatory and are taken
before respawning. The public simulation output defaults to recording all
ended observations; callers explicitly opt into `bootstrap_only` when only
truncation values are consumed. The final-observation buffer is initialized,
and unused rows cannot feed the advantage calculation.

This pass does not introduce CUDA Graphs, change the model, alter physics or
rewards, or change the number of training steps. Graphs and optimization-phase
launch overhead remain candidates for a subsequent measured experiment.

## Numerical and learning verification

Direct GPU-reference tests compare 8, 40 and 2,048 agents over 140 fixed-action
steps: sensors, final observations, rewards, flags, lengths, lap metrics and
sampled physical states must match exactly. A targeted timeout test checks
final observations before respawn, and a targeted crash checks that an optional
terminal sensor buffer remains untouched. Training tests compare actor,
critic and exploration parameters plus learning metrics after every update
for three seeds and two batch sizes, including profiling enabled on one path.

An earlier prototype parallelized individual sensor rays. It was faster but
introduced up to about 6e-6 normalized sensor differences and changed learning
trajectories. On one 200-update sample the reference finished 16 clean showcase
laps without wall hits while the prototype finished seven with 14 wall hits.
It passed the existing broad tests, which was insufficient evidence to accept
it. That prototype was removed; its throughput is not a result of the retained
implementation.

Nsight Compute was attempted but returned `ERR_NVGPUCTRPERM`. No hardware-counter
utilization, occupancy, bandwidth ceiling or claim of maximizing the GPU is
inferred from these event timings. Counter-based analysis remains unverified.

## Measured results — October 26, 2026

Median of the five trial means; each mean covers forty uninstrumented training
updates after ten warmup updates. Both paths use identical seeds, models,
training budgets and rules. Agent-steps/s includes rollout and optimization.

| Training agents | Races | Reference ms/update | Optimized ms/update | Reference M agent-steps/s | Optimized M agent-steps/s | Throughput gain |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 32 | 36.77 | 34.70 | 0.446 | 0.472 | 6.0% |
| 2,048 | 256 | 51.66 | 45.83 | 2.537 | 2.860 | 12.7% |
| 8,192 | 1,024 | 114.98 | 106.77 | 4.560 | 4.911 | 7.7% |

The default case's update latency fell 11.3%; throughput rose 12.7%. Those are
two expressions of the same measured change, not additional independent gains.
The optimized path was faster in all five pairs at each measured batch size.
The smaller gain at larger batches is consistent with the original launch
layout spreading more races over the GPU; counter-based confirmation is absent.

The separate identical-action simulation workload at 2,048 agents took 47.24 ms
versus 18.14 ms per 64-step block (2.60x). It contains no network optimization
and exercises more crashes than the learned-policy runs. This is **not** a
2.60x training speedup, and it is not evidence of faster learning per sample.

In a separate paired diagnostic run, default-batch rollout fell from roughly
29.1 to 22.9 ms and traffic resolution from about 12.5 to 7.0 ms. Optimization
remained about 23.3 ms. After this pass, optimization is a useful next profiling
target; this result does not establish full hardware efficiency.

After 200 updates with the original 1,200-update annealing schedule, the paired
60-second showcase evaluations also matched:

| Training seed | Clean laps, both paths | Contact cars, both paths | Wall hits, both paths | Best lap, both paths |
|---:|---:|---:|---:|---:|
| 1 | 16 | 0 | 0 | 28.20 s |
| 2 | 14 | 0 | 2 | 28.10 s |
| 3 | 15 | 0 | 1 | 28.05 s |

These finite tests establish agreement for the sampled workloads; they do not
establish that all seeds drive cleanly or that all hardware will show the same
gain. The seed-2/3 wall hits are retained in the result, not hidden by selecting
only the clean seed.

Raw CSV evidence: [256-agent training](benchmarks/2026-10-26-racing/training-256.csv),
[2,048-agent training](benchmarks/2026-10-26-racing/training-2048.csv),
[8,192-agent training](benchmarks/2026-10-26-racing/training-8192.csv),
[fixed-action simulation](benchmarks/2026-10-26-racing/simulation-2048.csv),
[diagnostic phases](benchmarks/2026-10-26-racing/phases-2048.csv), and
[seed 1](benchmarks/2026-10-26-racing/learning-seed-1.csv),
[seed 2](benchmarks/2026-10-26-racing/learning-seed-2.csv),
[seed 3](benchmarks/2026-10-26-racing/learning-seed-3.csv) evaluations.

## Reproduce

From the repository root in PowerShell:

```powershell
cmake --build --preset release
build/Release/racing_bench.exe --help
build/Release/racing_bench.exe envs=2048 warmup=10 updates=40 repeats=5 compare=1
build/Release/racing_bench.exe envs=2048 warmup=10 updates=30 repeats=2 compare=1 profile=1
build/Release/racing_bench.exe envs=2048 warmup=10 updates=40 repeats=5 compare=1 simulation=1
build/Release/racing_bench.exe seed=1 envs=2048 warmup=10 updates=190 repeats=1 compare=1 evaluate=1
$env:CUDARL_REQUIRE_GPU='1'
build/Release/cudarl_tests.exe
```

Repeat the training benchmark with `envs=256` and `envs=8192`, and the quality
evaluation with `seed=2` and `seed=3`. The simulation-only mode replays the same
64-step action tape, excluding its setup/copy, so policy behavior cannot explain
its speed difference. It requests the same training-relevant observations as
the corresponding training execution path. Evaluation runs the final mean-action
policy for 60 simulated seconds on the same CPU showcase grid (seed 55).

The tool prints comment-prefixed hardware metadata followed by CSV. To read
the table in PowerShell, filter lines beginning with `#` before `ConvertFrom-Csv`.
Unknown options, malformed numeric values and invalid sizes fail with a nonzero
exit code. `profile=1` and `evaluate=1` add diagnostic work and should be kept
separate from ordinary performance comparisons.

Validation of the retained implementation: Release build with warnings as
errors; **111/111 local CPU/GPU tests passed, zero skipped**, with
`CUDARL_REQUIRE_GPU=1`. The 1,200-update traffic regression completed 16 clean
showcase laps with zero contacts/wall hits and a 27.95 s best lap. This is
finite local validation; hosted CI and hardware-counter profiling were not
rerun as part of this local milestone.

Zero phase fields mean event profiling was disabled. Evaluation counts of -1
mean no showcase evaluation was requested. Raw timings describe this measured
hardware/workload, not a comparison against PyTorch or another ML framework.
