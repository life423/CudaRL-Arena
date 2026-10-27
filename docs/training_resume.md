# Full pond training persistence — October 26, 2026

The user requested closing/reopening the app without starting training over. They explicitly approved the one-time upgrade: preserve the old build's strongest learned controller while restarting critic, optimizer, exploration and background worlds once. That old process could not export its complete state. Subsequent saves restore the full learner at completed PPO update boundaries.

## Using it

Opening a pond automatically resumes its matching training file. Automatic saves occur every **30 wall-clock seconds** and on **normal scene/app closure**, including Back to demos. The UI shows **Training saved · update N**; the comparison also shows **Resumed from N · saved at M**. A crash/forced termination can lose work since the latest automatic save. Save errors are displayed separately and do not falsely claim success.

Pause learning retains the live learner. Watching a best actor changes visible inference while training continues. **Start new run** intentionally resets current learning and the graph, archives the previous full checkpoint with a `.before-reset-<timestamp>` suffix, and retains the separate best playback actor. Archives are not silently removed.

The running Godot game owns the native C++ training thread and 1,024 CUDA worlds. Keep that process open, unpaused and the PC awake to continue; minimizing does not stop the native worker. The editor/project manager alone does not train. No independent headless trainer backs this session.

The displayed pond starts a new visible trial on scene entry. Its position and local eaten count are separate from training progress. Background worlds, policy, optimizer, counters and learning graph resume.

## Files

Full training files are separate from the strongest evaluated JSON actors:

```text
user://pond-policies/regional-v1/best.json
user://pond-policies/regional-v1/best.json.training
user://pond-policies/shuffle-v1/best.json
user://pond-policies/shuffle-v1/best.json.training
```

These live in Godot's application data, **outside Git**. A Git commit/push does not back up the live training save. The running process holds an exclusive `.lock` lease; a second session cannot write the same training file concurrently. Test files use isolated application-data paths.

## Saved state and integrity

The full file contains actor/critic parameters; Gaussian log standard deviations; Adam moments, step counts and configuration for all three optimizers; every background-world state and observation; episode/food-placement counters; seeds; update and migration-origin counters; input-ordering mode; evaluation state; exact selected best actor/score; and complete per-update learning history.

Counter-based random streams are reproduced from saved seeds/update/world/placement counters. Scratch activations, gradients and rollout tapes are overwritten before reuse. Saving happens after an entire PPO update, on the worker after CUDA stream synchronization, rather than midway through an epoch or physics step.

The bounded binary format includes a marker, length and FNV-1a checksum. The writer flushes a temporary sibling and atomically replaces the destination; Windows uses `FlushFileBuffers` and replacement/write-through `MoveFileExW`. Failed writes retain the prior destination. This checksum detects accidental corruption; it is not an authenticity mechanism.

Restore checks task/body/layout compatibility, population, seed, input ordering, PPO/food protocol, finite parameters, Adam moments and world/history/optimizer counters. A damaged/incompatible full file stops resume and is retained. It never silently becomes fresh learning or gets overwritten on failed startup. An explicit new run archives even an unreadable previous file before resetting. Initial write failures without an existing full save allow in-memory learning with a reported save failure.

Exact continuation was tested on this build/GPU. Cross-version, compiler-ABI and GPU-family portability is not established. Format/physics/PPO semantic changes must update the checkpoint protocol. History grows with the run and is rewritten at saves; long-duration save latency and storage growth remain profiling work.

## Upgrade and verification

The old process was closed normally only after the user approved migration. Its strongest actor was copied before and after closure into `user://pond-policies/resume-upgrade-backup-20261026/`. The final retained controller was update **23,975**, scored **15.625 food/min**. The new build imported its weights and update origin, then immediately created a full training file. The frozen original comparison actor was unchanged.

The user's later screenshot shows **Resumed from 29,266**, advancing to **29,299** with best **15.63**. That is full-state restore, not repeated actor-only migration.

- **4 native persistence tests pass**: canonical/shuffled exact continuation across an episode reset and evaluation; actor, critic, exploration, Adam and observation equality; corruption/write-failure handling; exclusive ownership; shutdown saving; paused restore; incompatible-variant protection; and actor import followed by full resume.
- **6 PPO-session and 10 foraging tests pass**, including publication/pause behavior, CPU/CUDA physical parity and evaluation isolation. PPO mathematics and physics are unchanged.
- A **two-process Godot test** quits with unsaved updates, restores update **37** in another engine process, stays paused without advancing, applies the matching actor and resumes learning.
- The new Godot persistence smoke test passes **34 checks** for migration, save/resume, graph retention, reset/archive, best retention, corruption protection and four rendered window sizes.
- Comparison **92 checks**, normal pond **187 checks** and checkpoint-error handling **14 checks** pass. The normal pond test caught compact reset-button clipping and a resumed-status prefix mismatch; the final layout fixes both. Restored trained policies are not labelled untrained.
- Rendered 1280×720, 768×1024, 390×844 and 844×390 states are inspected. Save text is visible, controls retain 44 px targets, and the chart remains separate from the pond. A chart can begin at a nonzero imported update.

Internal UI review: hierarchy, consistency and feedback 9; usability 8 (short-landscape sidebar scrolling), accessibility 8 (keyboard/text/targets tested; assistive technology untested), responsiveness 8 (four windows, physical devices untested), performance 8 (worker-boundary saves functional; long-run latency unprofiled). Priorities are save/frame-time measurements, history/storage growth and assistive/device checks. Further gameplay and morphology remain separate decisions.
