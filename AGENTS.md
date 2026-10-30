# Minimal creature foundation

This worktree is the physical foundation of CudaRL-Arena, not a trained replacement creature. Read `docs/CREATURE_PLAN.md` and `docs/SESSION_CHECKPOINT.md` before work.

## Preservation boundary

- Comparison baseline: `e5e9bd1db7e442715ac8ba3a5f5030c0ede26430`, annotated tag `baseline/pre-minimal-foundation-2026-10-30`.
- The original checkout remains on `codex/autonomous-pond-practice`; this worktree is `codex/minimal-foundation`. Do not clean or modify the original tree, its running editor, ignored results or personal saves.
- Personal save backups and artifact hashes are in the original checkout's `build/foundation-baseline/2026-10-30/`. Git tags do not preserve ignored files.
- Historical source/results are recoverable through the baseline. Do not restore unused implementations into the active tree merely for history.

## Product and architecture

Build a physically simulated creature with our own C++/CUDA learning system and Godot presentation. The journey is competent swimming, interaction-driven measured PPO adaptation, growth/limbs, leaving water, walking and eventual manipulation. Later stages inform interfaces; they are not current implementation work.

Keep one learned goal chooser and one reusable movement motor. Critics are training auxiliaries. The motor request is persistent destination, desired terminal velocity and heading. No chooser training or one network per ability. Existing learner design may be replaced when justified; sound numerical utilities need not be rebuilt.

## Current physical gate and scope

- Preserve body dimensions, mass/inertia, two universal joints/four controlled axes, water forces and ±10 µN·m actuator limits. The optional servo supplies limited internal joint torques every5ms, never extra body propulsion or scripted rescue.
- Full-envelope contact geometry was corrected, but the registered dynamic support gate FAILED on rolled-head recontact. The contact response remains unresolved; do not label it validated, silently revert it or retune tolerances during structural cleanup.
- Diagnostic actuator inputs and paused playback are not learned movement. Servo regulation does not prove swimming. Unsupported calm holding, pond-relative holding in a specified validated current and supported bottom rest are separate future behavior tests.
- No replacement training, extra experiment budget, checkpoint overwrite, merging, deployment, new limbs or production replacement is authorized by cleanup.
- Latest follow-up authorizes one opt-in contact-response comparison after the verified cleanup rollback commit. Maximum43,600 physical substeps across both responses/backends; preparation/settings and exact cases must be frozen before the single run. No motor training or live-runtime/contact-save migration is included. See the latest checkpoint.
- That comparison is now STOPPED, not resumable:22,649 attempts, candidate rejected. Both responses pass5/8 primary cases; bent/pitched/rolled exceed0.5mm, bent/rolled also at finer steps. Legacy damping branch caused the final held-out CPU/CUDA rate mismatch. No free-water/passivity stage or new learning ran. Keep runtime physics2 and saved states unchanged; discuss a physically calibrated contact-response change before another run.
- Diagnostic state has its own versioned/checksummed schema and `user://foundation/` namespace. It must reject legacy learner saves without altering them.
- New user approval authorizes ONE separately registered compliant-versus-hard-contact screening pilot, maximum2,024 total attempted advances including all CPU/CUDA checks/failures/retries. Freeze numeric protocol/source before the sole run; static geometry/mass and cheap energy/parity are prerequisites. Stop affected work rather than tune/retry. Live physics2, body/water/actuators,0.5mm safety and saves remain unchanged. No PPO, engine migration, follow-up, merge or deployment.
- That pilot is now FINISHED and closed: 1,626 attempted advances (1,624 committed), no retries. Independent three-state J/M checks and 24 same-input parity steps passed. Calibrated A passed 20/20 short CPU/CUDA cases, including 0.25 s rolling and two fresh conditions. B was quarantined after its first passive rolling step: actual gap −3.725 µm versus its 2 µm hard-model gate (NOT a 0.5 mm breach). The unused 398 advances remain unspent; no reuse/follow-up is authorized. A is promising, not adopted or training-ready. Longer loaded support and its timestep-dependent damping remain unproved. See the latest checkpoint/evidence.

## Implementation and checks

Keep physical simulation independent of Godot and learning. One native world owns physical state; one extension exposes it; one scene displays it. Update build/resource references with deletions. Do not add inheritance ladders, duplicate session owners or speculative abstractions.

Use the small default physics, save/load and numerical checks. CPU/CUDA checks must distinguish compilation/skips from real GPU execution. Expensive studies require relevance and authorization; do not run training after structural edits. Measure actual runtime before deleting protections.

For UI changes preserve clear hierarchy, keyboard focus, usable44px action targets and narrow-screen layouts. Inspect real rendered states; a successful build is not visual verification. Report exactly what was verified.
