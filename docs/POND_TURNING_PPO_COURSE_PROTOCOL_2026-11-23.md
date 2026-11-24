# Turning exercise with a course-sensitive reward

Study ID `pond-motor-ppo-v3-course-2026-11-23`. This is a separate,
food-free research candidate. Use the same original motor 336 and saved
holding gain, deterministic seed, starting fixtures, 25% Rest / 25% Straight /
50% Turn request mix and turn-angle distribution as exercise 2. Straight keeps
its restored lateral feedback. Preserve all physics, servos, torque limits,
filter states, water forces and termination criteria. The optional paired-angle
curriculum is disabled in this comparison.

## One learning change

After a Turn request changes, multiply the existing positive progress term by
`1 / (1 + (1 − cos(error)) / (1 − cos(15°)))`. The cosine comes from the
horizontal whole-body COM velocity and the requested heading. A stationary
body earns no forward-progress credit. Keep the speed component (35%), progress
component (65%), motion gate, effort, depth, tilt and task failure terms.
Leave the reward for Rest, Straight and pre-turn warmup unchanged.

At the target speed of 8 mm/s, equal depth, tilt and effort, a 15° course
error now earns about 66% of the aligned movement reward. In exercise 2 it
earned about 98%. The change acts on measured travel direction; it supplies
no desired torque, imposed yaw, fixed path or forced motion.

## Admission, budget and reporting

Use eight threads per physical world-step block, covering all 512 worlds.
The matched early/late GPU replay found equivalent states and outputs for
this layout. Reuse the qualified runtime 4 CPU/CUDA proofs and original parent.
Bind the new trainer, program, evaluator, protocol and parent to fresh outputs
under `build/turn33`. A full, frozen update-zero assessment must pass all 28
required long-swim windows, four short swims and 14 rests before optimizer
updates. Measure all 16 turns with the original thresholds. Then use one local
stage runner to dispatch each eligible eight-update chunk and its skill check.

Allow at most 32 updates × 512 worlds × 64 controls, ten substeps per
control, minibatches of 4,096 and four epochs. Training is capped at 1,048,576
controls, 10,485,760 substeps, 1,024 actor and 1,024 critic Adam calls, 1,200
native seconds and 4 GiB of evidence. Five assessments allow at most 184,000
controls, 1,840,000 substeps, 900 seconds and 1 GiB. Save each completed
update. Stop on invalid physics, nonfinite data, KL above 0.05, required
retention failure or an exhausted budget. A failed turn remains a measured
training outcome. Preserve the original and exercise 2 checkpoints.

At 0/8/16/24/32 report all skill counts, comparable full-four-second COM
course error, angle-specific completion time, turn failure classifications,
and phase/experiment timing. Compare with the matching exercise 2 checkpoints.
Report actual throughput; completing the maximum allocation in 5 s would
require 209,715 transitions/s. The short layout benchmark times a single
matched control, not a full training run. Keep the visible motor separate.
