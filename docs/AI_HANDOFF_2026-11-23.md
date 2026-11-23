# Copy-ready engineering review request

This is a handoff for review, not authorization to execute another experiment.
Work is on `codex/current-swimming-v1`; do not merge it into main.

```text
Please critically review this physical reinforcement-learning project and
recommend the smallest credible route to the actual behavior, rather than
another experiment that only improves a proxy score.

PROJECT AND GOAL
Tadpole Pond uses Godot for presentation/interaction and our own native
C++/CUDA PPO and shared body/water simulation. Do not replace its runtime with
Python/PyTorch or script the creature's movement. The eventual architecture is
one learned goal chooser and one reusable request-conditioned body motor;
value critics are training auxiliaries, not extra behavioral controllers.

The immediate goal is food-free, autonomous swimming around a finite pond for
an open-ended session: keep moving, turn before the bank and preserve learned
swimming/resting. Later it must swim in current, adapt during player interaction,
grow new capabilities, and eventually climb up the bank out of the water.
Therefore bank avoidance must be task/context dependent, not a permanent law
that forbids shore contact or land transitions. Limbs/climbing are future work,
not capabilities of the present body.

BODY AND ACTUATION
Model 6 has two universal joints with two axes each: four neural actuator
outputs. They are joint-angle targets, passed through tanh, a 1 Hz reference
filter and physical servos with +/-10 microN*m torque limits. Each 50 ms control
uses ten 5 ms physical substeps. Movement must come from joint/water forces.
No teleports, rendered rotations, bank bounces or hidden propulsion/rescue.
The current pond's active horizontal ellipse has radii 0.20/0.12 m, with a
slightly inset behavioral boundary. Preserve original weights, saves and data.

WHAT IS ACTUALLY WORKING
The saved update 336 motor has a learned swimming stroke and qualified bounded
calm holding from rest. Four short swims,28 previously passing 60 s windows,
3 depth recoveries,148 previously passing 300 s windows and 14 hold cases were
retained in the last full retention audit. This does not prove turning,
moving-start braking, arbitrary mode switches, current competence or unlimited
pond swimming. The Godot app still uses this saved motor; it is not training.

FAILED CONTINUOUS-SWIMMING APPROACH
A 64-update nativeGPU study froze the original motor and learned 16 steering
coefficients, sampled once per episode. Commands depended on live state, but
the objective ended after 16 s. Its observer replaced desired heading with current
head azimuth every tick and removed horizontal lane feedback. It passed 0/16
60 s qualifying starts: the short objective improved while it learned to slow
and stall. Canonical old-mode bypasses retained old skills, which did not mean
the autonomous branch retained them.

A proposed 80-coefficient/all-four-axis 60 s follow-up was NOT run. The audit
found: negative living rewards plus a fixed failure cost could favor quitting
early; the +/-0.75 pre-tanh residual had limited target authority; longer
episodes still supplied one score per sampled controller, not per-action
turning credit. That proposal is explicitly on design hold.

MOST RECENT PHYSICAL EVIDENCE
One approved frozen-motor probe tested 22 total cases: poses 0/3 at 0,+/-5/15/30/
60/90 degrees. It swam 4 s, then received one persistent requested course relative
to the ORIGINAL course, not the oscillating head. The original 41-input motor
retained its lane/depth feedback. This was a diagnostic: zero learning updates.

The independently audited result:
-0 degree baselines: late speed 7.92-7.94 mm/s, course error<=0.61 degree.
-+/-5 degrees: late speed 7.75-7.83 mm/s, course error 2.93-4.21 degrees,
 depth<=0.67 mm and tilt<=9.36 degrees. Some lane offsets exceed 2 mm.
-+/-15 degrees: late speed 5.97-6.54 mm/s, late course error 9.64-10.69 degrees,
 although the overall 12 s displacement direction looks almost correct.
-+/-30 degrees: late speed only 2.20-2.34 mm/s, depth 1.19-2.11 mm,
 tilt roughly 12-17.5 degrees. Apparent heading alignment hides slow travel.
-All 8 cases at+/-60/90 degrees stopped on the 20 degree tilt guard.
-No bank/depth termination or hard physics fault.14/22 reached 16 s.

Actual spend:5782 controls/57800 substeps,13.706 s process time. The full
saved-data audit passed. No actor was replaced. Repeated incremental requests
have NOT been tested; do not assume small single-turn results prove navigation.

UNEXECUTED WORK IN PROGRESS
A separate full-motor PPO lesson is SOURCE PREPARATION ONLY. It extends the
41-input motor to 42 inputs with a zero-initialized heading-only intent column,
and uses a 43-input critic with remaining-time context. All motor weights would
learn, with per-control GaussianAR PPO credit and both current/previous-mean
gradients. Old Swim/Rest use the actual current trained motor, not a canonical
bypass. Rest's saved gain scales sampled raw latent actions before tanh.

The proposed 12 s lessons have 4 s warmup followed by a persistent turn. In the
new heading-only mode, remove horizontal lane error in WORLD coordinates
before inverse rotation; retain global depth and the desired course. This is
not the same observer as the diagnostic above, so it needs a matched new
baseline. Bounded motion quality and a remaining-time failure cost address
the early-exit loophole. Old practice/physical retention remain necessary.

Crucially, this proposed 42-input motor sees NO bank/object distances and
trains no goal chooser. Even success would only establish a turning primitive,
not solve autonomous pond swimming. Its new physical gate and independent
full-training data auditor are unfinished. No new training is approved or run.

THE USER'S DIRECT IDEA
Give the neural policy distance/direction information about nearby banks or
objects and reward safely moving away while continuing to swim. Measure actual
whole-body COM travel, not tail activity. Avoid rewards for stalling, spinning,
repeated approach/retreat reward farming, or early episode termination. Preserve
existing skills and the eventual ability to approach/climb a bank intentionally.

REVIEW QUESTIONS
1.Is staged requested-turn training necessary, or can bank-aware PPO learn
 turning and safe continuous movement together from the inherited motor?
2.Where should local avoidance belong in the chooser/motor architecture?
 What observations and persistent intent representation are actually needed?
3.What concrete reward/termination/curriculum prevents the observed shortcuts
 while permitting useful turns and eventual task-dependent shore exit?
4.Could small, smooth learned course changes reuse the existing motor, or
 does the physical evidence require motor retraining? Identify the uncertainty.
5.What is the smallest bounded experiment that discriminates these designs?
 Separate actual capability tests from numerical checks and proxy scores.

Challenge our assumptions. Do not claim the creature swims indefinitely or
recommend more training solely because the GPU/math checks pass. Keep native
CUDA/PPO and the physical body unless a justified model change is proposed.
Do not run another physical/training study, extend closed budgets, overwrite
originals, merge or deploy without the relevant explicit authorization.
```

The most useful project sources are `docs/AI_STATUS.md`,
`docs/CONTINUOUS_POND_DESIGN_AUDIT_2026-11-23.md`,
`docs/CONTINUOUS_POND_CONTROL_AUDIT_2026-11-23.md`,
`docs/FROZEN_REQUEST_TURN_RESULTS_2026-11-23.md`, and the unapproved
`docs/REQUESTED_TURN_PPO_V1_PROTOCOL_2026-11-23.md`. Read `AGENTS.md` before work.
