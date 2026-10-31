# Creature direction and foundation boundary

The master goal is a physically simulated creature that learns increasingly capable behavior through our native C++/CUDA reinforcement-learning system. Godot presents the world and accepts player interaction; it does not secretly move the creature or replace the learner. No Python/PyTorch replacement.

The pond tadpole is the first stage. A capable young swimmer responds to player changes and adapts through measured background PPO practice. Later come physical growth, useful underwater limbs, shallow crawling, leaving water, walking and eventual manipulation. Those future stages are unimplemented.

Two learned behavioral policies remain the intended design:

1. A goal chooser owns persistent intentions: feeding, safety, exploration, holding/resting.
2. One reusable motor executes destination, desired terminal velocity and heading requests.

PPO critics support training, not behavior selection. The chooser remains excluded from current motor work. Joint-target/fast-servo control is a preference under evaluation, not a proven whole-body motor. Preserve four controlled axes and physical torque limits.

## Current foundation and next behavior

One model6 native world and frozen learned stroke-strength/frequency actor drive
one live Godot scene. Straight calm-water8mm/s passed the bounded evaluation.
The supervised constant-speed lesson passes10/10four-second cases from rest at
4..8mm/s, including held-out5/7. The checkpoint is preserved separately; live
Godot still uses the previous8mm/s actor. Slowdown/switches remain unpassed.
See CURRENT_MOVEMENT.md.
The physical contact response is still unresolved. Free-water swimming is not
contact validation, current holding, braking, arrival or stable head orientation.

Keep the physical body/forces/actuator limits fixed while improving request
conditioning. Turning, braking/arrival and unsupported calm/current holding remain
separate future behavior gates; supported bottom rest is distinct. No chooser,
limbs or growth implementation is part of this stage. One shared motor path and
one native owner; no per-speed networks or parallel legacy scene/session hierarchy.

## Rollback and evidence

Baseline commit `e5e9bd1db7e442715ac8ba3a5f5030c0ede26430`, tag `baseline/pre-minimal-foundation-2026-10-30`, original branch `codex/autonomous-pond-practice`. The original checkout, ignored experiment directories and personal save namespaces remain untouched by this worktree. The baseline retains the detailed protocols, exact executed source and historical viewer roles. See the current checkpoint for preservation paths and actual validation.
