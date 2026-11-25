# Flexible-tail motor PPO workflow

One native C++/CUDA trainer, shared motor runtime and parallel CPU evaluator.
The runner builds, verifies the policy interface and exact resume, collects
GPU batches, saves complete checkpoints, and evaluates immutable exports every
eight updates while training continues. Protocols describe closed finite runs;
having a runnable command does not authorize repeating or extending a study.

The latest completed program is the closed seven-option study. Its shared
`program` mode covers rhythm, joint stroke/steering, hidden16 capacity, requested
speed, continuous requested course rate and an explicit material variant. The
existing chooser trainer supplies bank-navigation PPO. See
[results](../../docs/SEVEN_OPTIONS_RESULTS_2026-11-24.md),
[finite protocol](SEVEN_STAGE_PROTOCOL.md) and the
[counted chooser continuation](CHOOSER_CONTINUATION.md). Model FTYO3 and full
session FTSSv6 bind the program features, frozen skip and material. Complete
checkpoints and protected live models remain separate. The user requested a
stop after all seven: runnable commands do not authorize another session.

The historical `phase-aware` command remains:

```powershell
& tools/pond-flexible-tail-steering/run-session.ps1 `
  -SessionDir build/flexible-tail/phase-aware-v4 `
  -BuildDirectory build/flexible-tail/phase-aware-v4-build `
  -FeedbackMode phase-aware
```

That directory is already complete at 128 updates. Do not run it again to
collect additional data. A separately authorized session needs a new frozen
protocol and directory. `-Resume` restores full simulation/history, sampling,
actor, critic and optimizer state within the original session allowance.

`src/rl/flexible_tail_steering_policy.h` owns the feature layout, model kind,
parameter validation, trainable mask and model serialization. CPU evaluation
and CUDA collection use the same observer through `flexible_tail_fast_steering.h`.
The phase model retains four frozen base coefficients and trains nine added
phase/speed coefficients in the existing 31-input linear readout. There is no
additional behavioral network or alternate physics implementation.

Phase models use FTYO version 2 and feature schema 1; full sessions use FTSS
version 4. The legacy FTYO1 loader rejects phase models. Do not copy a phase
actor into the Godot live bundle or load it using the historical observer.
The current Godot project still loads its protected original motor.

See [phase protocol](PHASE_AWARE_PROTOCOL.md) and
[measured results](../../assets/research-evidence/pond-flexible-tail-phase-aware-v4/README.md).
