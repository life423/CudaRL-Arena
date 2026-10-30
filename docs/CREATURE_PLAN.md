# Creature direction and foundation boundary

The master goal is a physically simulated creature that learns increasingly capable behavior through our native C++/CUDA reinforcement-learning system. Godot presents the world and accepts player interaction; it does not secretly move the creature or replace the learner. No Python/PyTorch replacement.

The pond tadpole is the first stage. A capable young swimmer responds to player changes and adapts through measured background PPO practice. Later come physical growth, useful underwater limbs, shallow crawling, leaving water, walking and eventual manipulation. Those future stages are unimplemented.

Two learned behavioral policies remain the intended design:

1. A goal chooser owns persistent intentions: feeding, safety, exploration, holding/resting.
2. One reusable motor executes destination, desired terminal velocity and heading requests.

PPO critics support training, not behavior selection. The chooser remains excluded from current motor work. Joint-target/fast-servo control is a preference under evaluation, not a proven whole-body motor. Preserve four controlled axes and physical torque limits.

## What the minimal foundation does

A standalone physical world can accept explicitly bounded diagnostic actuator inputs, step the shared equations, report full physical state and save/load its own state. One Godot scene inspects it. It starts paused and labels all commands as diagnostic, not learned.

Body dimensions, mass/inertia, water forces and physical integration are retained from the tagged baseline. Full-body/fin envelope contact is retained rather than quietly reverting to the previous proxy mismatch. **The current contact response has an unresolved dynamic gate.** A rolled support fixture exceeded the registered0.5mm allowance at all three tested timesteps around0.185s. This occurred at zero command before new learning. Geometry133/133 and2,400 executed CPU/CUDA comparisons passed; complete support, timestep acceptance and free-water gates did not finish. No new motor trained.

## What comes next, after this structural foundation

Resolve the concrete contact-response gate through a separately bounded physical comparison. Do not tune it under the cleanup instruction. Then register a new motor lesson/evaluation design around purposeful movement, turning, braking, slow controlled arrival and sustained position/heading holding.

Declare speed, position, heading, stopping time/distance, duration, effort and safety tolerances before training. Evaluate unsupported calm holding, holding relative to the pond against a specified validated current, and supported bottom rest separately. Active joints resisting current are permitted; drifting with water is not position holding.

Learning policy/interface, observations, rewards, curricula and session organization may be revised when evidence supports it. Preserve reliable numerical utilities; prior implementation effort is not an obligation to retain a bad task design. No replacement training is authorized by this cleanup.

Only after dependable motor behavior should learned feeding/safety choices be integrated, followed by exploration/rest and much later growth. Food rate, example fitting, counters and servo regulation are not substitutes for movement acceptance.

## Rollback and evidence

Baseline commit `e5e9bd1db7e442715ac8ba3a5f5030c0ede26430`, tag `baseline/pre-minimal-foundation-2026-10-30`, original branch `codex/autonomous-pond-practice`. The original checkout, ignored experiment directories and personal save namespaces remain untouched by this worktree. The baseline retains the detailed protocols, exact executed source and historical viewer roles. See the current checkpoint for preservation paths and actual validation.
