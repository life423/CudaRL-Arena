# Native PPO chooser workflow

The opt-in `CUDARL_FRONTIER_REWARD` continuation is CLOSED without adoption.
It imports a full35-input parent using explicit `ParentTraining`, `ParentUpdate`
and `ParentInputs=35`, retaining all optimizer ages and world/visit history.
Discount-consistent unvisited-water progress does not retain bank safety in the
128-update experiment. Godot remains appended16/motor88. Read
[results](../../assets/research-evidence/pond-flexible-tail-frontier-progress-v4/README.md)
and [protocol](FRONTIER_PROGRESS_PROTOCOL.md). The user requested a stopping
point; do not automatically launch another session or desktop inspection.

The latest `CUDARL_APPENDED_COVERAGE` session is closed with selected16 adopted
in Godot. It retains all32 local inputs and appends three zero-weight distant
inputs to actor and critic. Explicit migration preserves inherited weights,
Adam moments/ages, world and visit state; new columns have their own saved age.
The original32-input checkpoint compacts back exactly, and initial GPU actor
and critic outputs are exactly equal across1,024 worlds. The session completes
128 new updates; selected16 raises worst/mean individual30-minute coverage to
42.61/43.91%, with4/4 safe long runs. Final128 is unsafe and rejected. Read
[results](../../assets/research-evidence/pond-flexible-tail-appended-coverage-v3/README.md)
and [protocol](APPENDED_COVERAGE_PROTOCOL.md). `-Appended` implies `-Distant` and
requires the appended compiled target. Closed output directories remain
non-restartable. Further training requires a new authorized session.

The opt-in `CUDARL_DISTANT_COVERAGE` experiment is closed and unqualified.
It replaced only the three far inputs and lost bank safety during128 updates;
Godot stays on coverage80. Version4 observation decoding is explicit and keeps
version3 behavior intact. See [results](../../assets/research-evidence/pond-flexible-tail-distant-coverage-v2/README.md)
and [protocol](DISTANT_COVERAGE_PROTOCOL.md). `-Distant` selects the explicit
coverage80 full-state parent; its compiled target must enable both coverage
options. `inspect-fork` performs zero-physics/zero-PPO input checks only.

The earlier coverage continuation is closed at 128 updates. Its preserved
selected chooser80 uses frozen motor88; individual 30-minute coverage is
32–45%, not full pond coverage. Read [results](../../assets/research-evidence/pond-flexible-tail-coverage-v1/README.md)
and [the finite protocol](COVERAGE_PROTOCOL.md). `run-coverage-session.ps1`
uses the compiled opt-in coverage/reliable targets, the recorded baseline and
an explicit full-state parent import. It refuses an existing training output;
it does not restart this closed study. Historical workflows below remain
available with their original model/task contracts.

Use `run-session.ps1` for sustained training with 1,024 CUDA worlds.
The runner builds the native Release executables when their bound source
changes, launches one continuing session, and evaluates saved candidates
while training continues.

From the repository root, after the steering interface has been qualified
for the registered session, run

```powershell
./tools/pond-flexible-tail-chooser/run-session.ps1 `
    -SessionDir build/flexible-tail/chooser-session-v1 `
    -SteeringMode held-heading
```

Resume the same actor, critic, optimizer, world and random state with

```powershell
./tools/pond-flexible-tail-chooser/run-session.ps1 `
    -SessionDir build/flexible-tail/chooser-session-v1 `
    -SteeringMode held-heading -Resume
```

Both invocations use the same registered maximum of 64 updates and a
cooperative target of 1,800 seconds of total training-process wall time.
An update already started finishes and checkpoints within the external
600-second grace window. The mode and bound inputs
must match on resume. Immutable update exports feed the original held-out
suite every eight updates and at the final completed update. A qualified
candidate still requires the tested Godot publication step.

Before a new session is admitted, the runner requires a passing full-resume
comparison for the exact native executable and frozen inputs, plus the
matched steering diagnostic. Their default locations are
`build/flexible-tail/chooser-session-resume-check` and
`build/flexible-tail/steering-interface-v3`. Both evidence sets are copied
and hash-bound into the session.

The session folder contains the source snapshot and hash manifest,
append-only invocation receipts and logs, native full checkpoints,
learning curves, immutable actor exports and per-update evaluations.
`SESSION_PROTOCOL.md` defines the budget, resume verification and admission
criteria, and describes the current coverage and physical limitations.

`run-study.ps1`, `PROTOCOL.md` and `BANK_CURRICULUM_PROTOCOL.md` preserve the
closed initial chooser investigations and their exact commands. Their
short process limits belong to those historical experiments.
