# Physics solver workspace reuse — October 27, 2026

The first optimization after [bottleneck profiling](../profiling/README.md) reuses resistance-matrix storage for the Cholesky factor. On this RTX 5070, alternating old/new executable runs measured **4.29% more end-to-end PPO training throughput** at 8,192 swimmers. The compiled physics kernel's private stack shrank by **400 bytes/thread (11.31%)**. The change is retained and the native Godot extension is rebuilt. PPO, physical equations, timestep, policy architecture, and saved controllers are unchanged.

## The change and why it is safe

`src/envs/swimmer/deep_physics.h::substep` previously allocated a separate `float L[10][10]` for factorization. The solver consumes each lower-triangular resistance entry exactly once when forming `M[i][j] + h*R[i][j]`. After consumption, that entry can hold `L[i][j]`. Subsequent rows need the factor, not the overwritten resistance entry. The right-hand side uses the mass matrix, which remains intact. Damping and joint-limit terms are applied before factorization.

The implementation replaces that allocation with `auto& L = a.R`. Loop order, arithmetic expressions and integration are retained. No matrix packing, register cap, block-size change, fast-math flag or cooperative solver was added. The pre-change dense factor workspace is retained as a test-only reference in `tests/fixtures/deep_solver_reference.h`.

## Paired throughput experiment

- Same hardware/build settings as profiling: RTX 5070, CUDA 13.2, Release, sm_120.
- Preserved original profiling executable versus rebuilt executable; no counter profiler during timing.
- Each invocation restores the same seed-71/update-128 full checkpoint into a disposable trainer, runs three warmup updates, then measures six updates (132–137).
- Six sequential invocations, order: old, new, new, old, old, new. Three invocations / 18 measured updates per variant. GPU workloads were not run concurrently during these timings.
- No policy or checkpoint output from the timing harness. Identical 8,192-world batch, motor, PPO, current distribution and training seed.

| Measurement | Original | Reused workspace |
| --- | ---: | ---: |
| Mean update time | 835.44 ms | 801.06 ms |
| Run-mean update-time range | 830.77–838.47 ms | 800.69–801.61 ms |
| Effective simulation throughput | 6.276M steps/s | 6.545M steps/s |
| Kernel private stack | 3,536 bytes/thread | 3,136 bytes/thread |
| Registers/thread | 255 | 255 |

Update time fell 4.12%; inverse-time throughput rose 4.29%. These percentages use different denominators. See [aggregate calculations](summary.json), [per-run summaries](paired-summary.json), and the six timing CSVs beside this report. Three invocations per variant are a bounded local result, not a hardware-independent speedup guarantee. Timings exclude startup, validation, checkpoint writing and Godot rendering; the displayed scene's frame rate was not benchmarked.

## Numerical and behavior validation

- **CPU:** 32 cases × 100 control ticks, comparing every physical substep to the retained dense reference. Zero bit differences in positions, quaternion, angles, velocities, work and travel; all finite.
- **CUDA:** 56 water cases × 6,000 ticks (five minutes each), plus eight short vacuum cases × 100 ticks. All 3,368,000 compared substeps match bit-for-bit and remain finite. Cases cover calm/current water, folded joints, pond floor/surface contacts and box walls. See [reference check](reference-check.log).
- A first test incorrectly drove an undamped vacuum body for five minutes; both solvers became non-finite identically. Vacuum is therefore a short internal-torque check, not evidence of stable indefinite vacuum actuation. The long water cases remain five minutes each; this existing integrator limitation was not hidden by changing physics.
- **Real checkpoint continuation:** copied the seed-71/update-128 full state into two disposable directories; each executable restored it, evaluated, completed one PPO update and saved update 129. Entire full-state saves and policy bundles match by SHA-256, covering actor, critic, Adam and background worlds: [comparison](resume-comparison.json). Source checkpoint is unchanged.
- **Full native suite:** 161/161 pass, zero skipped, including currents, shared CPU/CUDA dynamics, energy/internal momentum checks, food rules and exact training resume: [regression log](regression.log).
- **Existing CPU playback regression:** passes, including its preserved trajectory fixture: [CPU log](cpu-regression.log). This verifies the shared-header change; portability work remains paused.
- **Application:** CUDA playback module and Godot extension rebuilt successfully. No UI/shader changes or rendered-camera checks were needed. No viewer was opened for this experiment.
- All seven protected policies/full saves remain unchanged: [preservation record](preserved-inputs.json). No background training/profiling helper remains running.

Agreement is evidence for these builds, inputs and hardware, not a mathematical guarantee for all future compiler versions. The reference remains available to catch regressions.

## Resource counters and remaining limit

`cuobjdump --dump-resource-usage` confirms the exact stack reduction in both original/new binaries: [original resources](resources-baseline.txt), [new resources](resources-reuse.txt).

A full Nsight Compute capture of one representative new physics launch also reports 3,136-byte stack, 255 registers/thread, 16.67% theoretical occupancy and 10.89% achieved occupancy. Compute throughput is 15.14%, L2 throughput 84.34%, and local-memory sectors still dominate L1TEX traffic (98.07%): [counter details](ncu-reuse-details.txt). This capture used replay and a higher sampled clock than the earlier baseline counter run; do not interpret counter percentages or captured durations as a controlled before/after throughput measurement. The unprofiled alternating experiment above is the speed evidence. Temporary elevated profiling succeeded; no persistent security settings were changed.

Register-limited residency and large geometry/matrix scratch remain. This is a small verified memory improvement, not full hardware saturation or a major solver redesign. A separate next experiment could investigate compact matrix storage or geometry lifetime, using the same reference and benchmark gates. It has not been implemented. More simulation steps per second does not itself demonstrate better learning per sample or current robustness.

## Reproduction

The preserved original executables and header are local under `build/pond-solver-study/baseline/`. Full counter capture is `build/pond-solver-study/reuse-8192.ncu-rep`. They are ignored build artifacts, not release assets. The test-only original solver and small evidence files are repository sources.

```powershell
cmake --build --preset release --target cudarl_tests pond_progress_profile pond_progress_train cudarl --parallel 6
$env:CUDARL_REQUIRE_GPU='1'
build/Release/cudarl_tests.exe 'deep solver'
ctest --preset release
build/Release/pond_progress_profile.exe --baseline docs/experiments/swimmer/gentle-currents/frozen-policy.bin --checkpoint build/pond-progress-study/seed-71/training.bin --warmup 3 --updates 6 --output build/pond-solver-study/repeat.csv
```

Use the preserved original executable with the same profiling arguments for a repeat comparison, without running other GPU tests simultaneously. To compare full-state continuation, copy the original full checkpoint into separate disposable output directories before calling each `pond_progress_train --baseline ... --output COPY_DIR --updates 1`; never point this comparison at the source save directory.

Work remains on `codex/portable-pond`, uncommitted alongside earlier work. No merge, push, new training curriculum or model deployment was performed.
