# GPU optimization checkpoint and future work

Saved October 27, 2026 at the user's explicit request. Performance work is parked while we return to current-swimming AI. This is a backlog, not authorization to implement every item or start background training.

## Measured starting point

The [profiling report](experiments/swimmer/gentle-currents/progress-recovery/profiling/README.md) measures the native progress-selector PPO loop on an RTX 5070. GPU work is active for 98.6% of the captured interval, physics takes 89.5% of GPU kernel time, and memory copies/memset take about 0.01% of captured time. Work is queued ahead of execution. CPU submission is not the dominant observed limiter.

The physics kernel's large private working set creates heavy local-memory/cache traffic. Its 255 registers/thread limit theoretical occupancy to 16.67%. A high GPU activity percentage is not proof of efficient arithmetic use. Larger batches alone did not solve the limit: prior short tests gave 6.27M steps/s at 8,192 worlds and 6.38M at 32,768. Those batch sizes also change PPO data per update; steps/s is not learning quality.

## Completed improvement

[Reuse the resistance matrix for the Cholesky factor](experiments/swimmer/gentle-currents/progress-recovery/solver-workspace/README.md). This removes a separate 10×10 float scratch matrix without changing the equations, arithmetic order or timestep.

- Compiled stack: 3,536 → 3,136 bytes/thread (−11.31%). Registers remain 255.
- Three alternating invocations / 18 measured updates per variant at 8,192 worlds: 835.44 → 801.06 ms/update; 6.276 → 6.545M steps/s (+4.29% throughput).
- Tested CPU and five-minute water CUDA trajectories match the retained reference bit-for-bit. A real full-state checkpoint continuation also matches exactly.
- All 161 native tests and the existing CPU playback reference pass; native Godot extension rebuilt. Original policies and training saves are preserved.

The speed result excludes startup, rendering, evaluation and snapshot I/O. It is a local measurement on this GPU/build, not a universal performance guarantee. Full binary captures/original executables remain under ignored `build/pond-profile/` and `build/pond-solver-study/`; small evidence and the reference solver are retained in the repository.

## Candidate experiments when performance work resumes

1. Reduce remaining physics-private storage: examine geometry/Jacobian lifetimes, solver scratch reuse and compact matrix representations. Preserve the dense reference; changing floating-point order requires additional numerical and long-rollout validation.
2. If small changes are insufficient, evaluate a cooperative/shared-memory solver. Do not assume a register cap helps: it may increase spilling.
3. Reprofile after each accepted change. Revisit CUDA graphs, launch overhead or synchronization if the new timeline shows they have become important.
4. Consider CPU support for checkpoint screening, logging and safe snapshot writing as separate workflow improvements. Validate CPU/GPU evaluation agreement before using a new CPU path for policy ranking; retain authoritative GPU comparisons where execution-path sensitivity matters.

Do not enlarge the network, fill VRAM, add worlds or chase a utilization percentage merely to make hardware look busy. Keep native C++/CUDA training without Python/PyTorch. Preserve PPO, physical rules, learned skills and source saves unless a later experiment explicitly calls for a controlled change.

## Acceptance gate

Change one thing; benchmark alternating original/new runs from identical checkpoints after warmup; keep other GPU workloads separate. Report timing variation and use unprofiled timing for speed claims. Check physical invariants, reference trajectories, long swims, checkpoint continuation and retained calm/current behavior as appropriate. An optimization must not gain speed by silently reducing physics accuracy, training work or task difficulty. Save the measurements and remaining limit here, then return to the AI goal.
