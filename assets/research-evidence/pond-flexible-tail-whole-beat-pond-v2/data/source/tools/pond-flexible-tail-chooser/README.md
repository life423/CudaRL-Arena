# Native PPO chooser workflow

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
