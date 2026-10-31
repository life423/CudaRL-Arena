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
  rescue. Contact, slowdown/switches, current and holding are not validated.
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
