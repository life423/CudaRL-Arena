# Workspace instructions

Read README.md for current usage, architecture and limitations. Check asset JSON
provenance before changing a checkpoint or interpreting acceptance.

- Work only in this minimal-foundation worktree unless explicitly instructed.
  Preserve the original checkout, its editor, personal saves and ignored evidence.
- Baseline tag: baseline/pre-minimal-foundation-2026-10-30, commit
  e5e9bd1db7e442715ac8ba3a5f5030c0ede26430. Original save backups/hashes are in
  the original checkout's build/foundation-baseline/2026-10-30/; Git does not save
  ignored files. e84db5f and626e313 preserve later source milestones.
- All prior physical/training budgets are closed. Freeze a new finite protocol
  before any explicitly approved run; count backend/internal attempts and failures.
  No retries, tuning, tolerance relaxation or unused-budget reuse implicitly.
- Preserve model6 physics, two universal joints/four axes and +/-10microN*m torque
  limits unless a model change is authorized. No global propulsion or scripted
  rescue. Bounded calm holding from rest qualified in the November 22 learned-
  amplitude study: six original and six withheld 64-second cases, plus two
  uninterrupted 300-second cases. Read docs/HOLD_LEARNED_AMPLITUDE_RESULTS_2026-11-22.md
  for its restricted learned-parameter scope. Contact, moving-start braking,
  arbitrary slowdown/mode switches and disturbance/current holding remain
  unvalidated; no new budget or production adoption follows automatically.
- Keep native physics/learning independent of Godot. One native world owner,
  one bridge and one scene; one chooser and one reusable motor, not per-skill nets.
  Store/replay actual sampled actions consistently; critics are training auxiliaries.
- Keep checkpoint formats explicit/checksummed. Reject incompatible data without
  random fallback, silently resetting state or overwriting old personal saves.
- Delete obsolete active code/docs rather than commenting out or adding archives.
  Preserve unique checkpoint/data/evidence and dependencies; inspect hidden contents
  before removing directories. Do not restore retired code merely for history.
- Use focused nonphysical/CPU/CUDA checks; distinguish actual GPU execution from
  skips. No training/expanded physical studies after cleanup without approval.
- Preserve usable44px UI controls and narrow layouts; inspect rendered UI changes.
  Document exactly what was verified. Keep main merges/deployment separate from
  feature-branch publication; new changes stay local until publication is requested.

## User workflow preference — November 22, 2026

- Keep actual reinforcement learning through native C++/CUDA PPO. A frozen viewer,
  scripted motion or manually chosen quiet coefficient is not a learning result.
- The user explicitly requested rebuilding the whole Godot GUI from scratch
  because it had become a mess. Maintain one coherent primary interface around
  the actual creature. Keep ordinary controls simple; disclose research detail
  and history on demand. Remove obsolete routes/controls after dependency checks
  instead of continually adding parallel viewers or dashboard panels.
- Whenever changing learning, simulation or other meaningful app behavior, check
  the GUI in the same pass: update visible information/actions and the selected
  saved-policy metadata, and remove anything no longer needed. Preserve distinct
  labels for actual learning, live frozen inference and recordings. Never show a
  stale or rejected candidate as a successful current capability.
- Keep the visible native body faithful to the shared physical state and leave
  previous source/evidence available for rollback. Verify the rendered result
  and relevant interactions before saying the GUI is finished.
- Keep automated GUI validation windows hidden. Launch the final interactive
  app normally, without test/capture auto-exit arguments, and verify that Godot
  Project Manager points at the current project rather than an old build copy.
